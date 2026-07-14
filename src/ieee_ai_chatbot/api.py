"""
IEEE AI Chatbot — REST API Layer
=================================
Two usage modes:

1. **Embedded in Gradio** (HF Spaces / local):
       from ieee_ai_chatbot.api import build_router
       router = build_router()
       demo.app.include_router(router, prefix="/api/v1")

2. **Standalone FastAPI server** (see app_api.py):
       uvicorn app_api:api_app --host 0.0.0.0 --port 8000

Endpoints (all under the prefix used at mount time)
------------------------------------------------------
GET  /health                          liveness probe
GET  /status                          agent + KB status
POST /chat                            stateless Q&A (no server-side memory)
POST /chat/session                    stateful Q&A  (SQLite memory per session)
GET  /chat/history/{session_key}      fetch conversation history
DELETE /chat/history/{session_key}    clear a session's history
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow direct import when module is used without package install
_ROOT = Path(__file__).resolve().parents[2]  # project root
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .chat import RAGAgent
from .chat_history import ChatHistoryManager
from .config import Settings
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The user's question")
    history: list[dict] = Field(
        default_factory=list,
        description="Optional conversation history [{role, content}] — manage it yourself",
    )
    generate_suggestions: bool = Field(
        default=False, description="Include follow-up question suggestions in response"
    )


class SessionChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The user's question")
    session_key: str = Field(
        default="default",
        description=(
            "Unique session identifier (e.g. user ID or UUID). "
            "Conversation history is persisted server-side per key."
        ),
    )
    generate_suggestions: bool = Field(
        default=False, description="Include follow-up question suggestions in response"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    confidence: str = ""
    suggestions: list[str] = []
    run_id: str = ""


class HistoryResponse(BaseModel):
    session_key: str
    messages: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# build_router() — returns a reusable APIRouter
# ─────────────────────────────────────────────────────────────────────────────

def build_router() -> APIRouter:
    """
    Build and return a FastAPI APIRouter with all chatbot REST endpoints.

    Mount it wherever you need:
        demo.app.include_router(build_router(), prefix="/api/v1")   # Gradio
        fastapi_app.include_router(build_router(), prefix="/api/v1") # standalone
    """
    settings = Settings.from_env()
    chat_history_mgr = ChatHistoryManager(settings.chat_history_db_path)
    agent = RAGAgent(settings, get_chunk_boosts=chat_history_mgr.get_chunk_boosts)
    rate_limiter = RateLimiter(
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )

    router = APIRouter(tags=["IEEE AI Chatbot"])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rate_check(request: Request) -> None:
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        client_ip = forwarded or (request.client.host if request.client else "unknown")
        allowed, _ = rate_limiter.check(client_ip)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait before sending another request.",
            )

    def _history_to_text(messages: list[dict]) -> str:
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        )

    # ── Routes ────────────────────────────────────────────────────────────────

    @router.get("/health", summary="Liveness probe")
    async def health() -> dict:
        """Returns 200 OK if the server is running."""
        return {"status": "ok"}

    @router.get("/status", summary="Agent + KB status")
    async def status() -> dict:
        """Return model info, vector store config, and readiness flags."""
        return {"status": agent.status()}

    @router.post(
        "/chat",
        response_model=ChatResponse,
        summary="Stateless Q&A — you manage the history",
    )
    async def chat_stateless(body: ChatRequest, request: Request) -> ChatResponse:
        """
        Ask a question without server-side memory.

        Pass `history` to maintain context across calls:
        ```json
        {
          "message": "What is IEEE?",
          "history": [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]
        }
        ```
        """
        _rate_check(request)
        history_text = _history_to_text(body.history)
        try:
            answer, sources, run_id, confidence, suggestions, _ = agent.answer(
                body.message,
                history_text=history_text,
                generate_suggestions=body.generate_suggestions,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error in /chat: %s", exc)
            raise HTTPException(status_code=500, detail="Internal error") from exc

        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            suggestions=suggestions if isinstance(suggestions, list) else [],
            run_id=run_id,
        )

    @router.post(
        "/chat/session",
        response_model=ChatResponse,
        summary="Stateful Q&A — server stores memory per session_key",
    )
    async def chat_with_session(body: SessionChatRequest, request: Request) -> ChatResponse:
        """
        Ask a question with server-side memory tied to a `session_key`.

        Pass any unique string (user ID, UUID, etc.) as `session_key`.
        The server automatically loads and saves the conversation.

        ```json
        {"message": "Who chairs the CS chapter?", "session_key": "user-abc-123"}
        ```
        """
        _rate_check(request)
        conv_id = chat_history_mgr.get_or_create_conversation(body.session_key)
        history_items = chat_history_mgr.get_history(conv_id)
        history_text = _history_to_text(history_items)

        chat_history_mgr.auto_title(conv_id, body.message)
        chat_history_mgr.add_message(conv_id, "user", body.message)

        try:
            answer, sources, run_id, confidence, suggestions, _ = agent.answer(
                body.message,
                history_text=history_text,
                generate_suggestions=body.generate_suggestions,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error in /chat/session: %s", exc)
            raise HTTPException(status_code=500, detail="Internal error") from exc

        chat_history_mgr.add_message(conv_id, "assistant", answer)

        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            suggestions=suggestions if isinstance(suggestions, list) else [],
            run_id=run_id,
        )

    @router.get(
        "/chat/history/{session_key}",
        response_model=HistoryResponse,
        summary="Fetch conversation history for a session",
    )
    async def get_history(session_key: str, request: Request) -> HistoryResponse:
        """Returns all messages for the given `session_key` in chronological order."""
        _rate_check(request)
        conv_id = chat_history_mgr.get_or_create_conversation(session_key)
        messages = chat_history_mgr.get_history(conv_id)
        return HistoryResponse(session_key=session_key, messages=messages)

    @router.delete(
        "/chat/history/{session_key}",
        summary="Clear all history for a session",
    )
    async def delete_history(session_key: str, request: Request) -> dict:
        """
        Delete all conversations for a given `session_key`.
        Useful for implementing a 'Clear Chat' button on your website.
        """
        _rate_check(request)
        convs = chat_history_mgr.list_conversations(session_key)
        for conv in convs:
            chat_history_mgr.delete_conversation(conv["id"])
        return {"deleted": len(convs), "session_key": session_key}

    return router


# ─────────────────────────────────────────────────────────────────────────────
# build_api() — returns a self-contained FastAPI app (for app_api.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_api() -> FastAPI:
    """
    Wrap build_router() in a standalone FastAPI app with CORS.
    Used by app_api.py for running the REST server independently of Gradio.
    """
    cors_raw = os.getenv("API_CORS_ORIGINS", "*")
    allow_all = cors_raw.strip() == "*"
    cors_origins: list[str] = ["*"] if allow_all else [o.strip() for o in cors_raw.split(",")]

    app = FastAPI(
        title="IEEE AI Chatbot — REST API",
        description=(
            "Standalone REST API for the IEEE Beni Suef RAG Chatbot.\n\n"
            "When running embedded in Gradio (HF Spaces), endpoints are available "
            "under `/api/v1/` on the same port as the UI."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = build_router()
    app.include_router(router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app
