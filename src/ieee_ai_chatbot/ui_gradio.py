from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from .chat import RAGAgent
from .config import Settings
from .ingest import ingest_files, ingest_website, sync_local_docs


def _user_requested_sources(message: str) -> bool:
    prompt = message.lower()
    source_triggers = [
        "source",
        "sources",
        "citation",
        "citations",
        "reference",
        "references",
        "where did you get",
    ]
    return any(trigger in prompt for trigger in source_triggers)


def _history_to_text(history: list[dict[str, str]] | list[list[str]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    if isinstance(history[0], dict):
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    for pair in history:
        if len(pair) != 2:
            continue
        lines.append(f"user: {pair[0]}")
        lines.append(f"assistant: {pair[1]}")
    return "\n".join(lines)


def _normalize_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                normalized.append({"role": role, "content": content})
            continue

        if isinstance(item, list | tuple) and len(item) == 2:
            user_text = str(item[0]).strip()
            assistant_text = str(item[1]).strip()
            if user_text:
                normalized.append({"role": "user", "content": user_text})
            if assistant_text:
                normalized.append({"role": "assistant", "content": assistant_text})

    return normalized


def create_demo() -> gr.Blocks:
    settings = Settings.from_env()
    agent = RAGAgent(settings)
    session_histories: dict[str, list[dict[str, str]]] = {}

    def _session_key(request: gr.Request | None) -> str:
        if request is None:
            return "default"

        session_hash = getattr(request, "session_hash", None)
        if isinstance(session_hash, str) and session_hash.strip():
            return f"session:{session_hash.strip()}"

        header_session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("x-client-id")
            or ""
        ).strip()
        if header_session_id:
            return f"header:{header_session_id}"

        forwarded_for = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded_for:
            return f"ip:{forwarded_for}"

        if request.client and request.client.host:
            return f"ip:{request.client.host}"

        return "default"

    def chat_fn(
        message: str,
        history: list[dict[str, str]] | list[list[str]] | None = None,
    ) -> str:
        history_text = _history_to_text(history)
        answer, sources = agent.answer(message, history_text=history_text)
        if not sources or not _user_requested_sources(message):
            return answer
        source_text = "\n".join(f"- {source}" for source in sources[:8])
        return f"{answer}\n\nSources:\n{source_text}"

    def chat_api_fn(message: str, request: gr.Request | None = None) -> str:
        if not (message or "").strip():
            return ""

        key = _session_key(request)
        history_items = session_histories.get(key, [])
        answer = chat_fn(message, history=history_items)
        updated_history = [
            *history_items,
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        session_histories[key] = updated_history[-30:]
        return answer

    def chat_turn_api_fn(message: str, history_json: str) -> tuple[str, str]:
        parsed_history: Any
        try:
            parsed_history = json.loads(history_json or "[]")
        except Exception:
            parsed_history = []

        history_items = _normalize_history(parsed_history)
        answer = chat_fn(message, history=history_items)
        updated_history = [
            *history_items,
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return answer, json.dumps(updated_history, ensure_ascii=False)

    def upload_fn(files: list[Any] | None) -> str:
        if not files:
            return "No files selected."
        try:
            paths = [str(Path(file.name).resolve()) for file in files]
            result = ingest_files(settings, paths, origin="upload")
            return (
                f"Indexed: {result['indexed']} | Skipped: {result['skipped']} | "
                f"Deleted old chunks: {result['deleted']}"
            )
        except Exception as error:
            return f"Upload indexing failed: {error}"

    def sync_fn() -> str:
        try:
            result = sync_local_docs(settings)
            return (
                f"Synced files: {result.get('total_files', 0)} | Indexed: {result['indexed']} | "
                f"Skipped: {result['skipped']} | Deleted chunks: {result['deleted']}"
            )
        except Exception as error:
            return f"Local sync failed: {error}"

    def status_fn() -> str:
        status = agent.status()
        lines = [f"- {key}: {value}" for key, value in status.items()]
        return "\n".join(lines)

    def website_fn(url: str, max_pages: int) -> str:
        target_url = (url or settings.website_default_url).strip()
        if not target_url:
            return "Website URL is required."
        try:
            pages_limit = max(1, int(max_pages or settings.website_max_pages))
            result = ingest_website(settings, start_url=target_url, max_pages=pages_limit)
            return (
                f"Crawled pages: {result['total_pages']} | Indexed: {result['indexed']} | "
                f"Skipped: {result['skipped']} | Deleted old chunks: {result['deleted']}"
            )
        except Exception as error:
            return f"Website crawl failed: {error}"

    with gr.Blocks(title="IEEE AI RAG Chatbot") as demo:
        gr.Markdown("# IEEE AI RAG Chatbot")

        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=chat_fn,
                title="Ask IEEE knowledge questions",
            )

        with gr.Tab("Ingestion"):
            uploader = gr.Files(
                label="Upload PDF/PPT/DOC files",
                file_count="multiple",
                file_types=[".pdf", ".ppt", ".pptx", ".docx", ".doc"],
            )
            upload_output = gr.Textbox(label="Upload Status")
            sync_output = gr.Textbox(label="Local Sync Status")
            website_url = gr.Textbox(
                label="Website URL",
                value=settings.website_default_url,
            )
            website_max_pages = gr.Number(
                label="Website max pages",
                value=settings.website_max_pages,
                precision=0,
            )
            website_output = gr.Textbox(label="Website Crawl Status")
            upload_button = gr.Button("Upload + Index")
            sync_button = gr.Button("Sync docs/pdf, docs/ppt, and docs/doc")
            website_button = gr.Button("Crawl Website + Index")
            upload_button.click(fn=upload_fn, inputs=[uploader], outputs=[upload_output])
            sync_button.click(fn=sync_fn, inputs=None, outputs=[sync_output])
            website_button.click(
                fn=website_fn,
                inputs=[website_url, website_max_pages],
                outputs=[website_output],
            )

        with gr.Tab("Status"):
            status_output = gr.Markdown()
            status_button = gr.Button("Refresh status")
            status_button.click(fn=status_fn, inputs=None, outputs=[status_output])

        api_message = gr.Textbox(visible=False)
        api_output = gr.Textbox(visible=False)
        api_trigger = gr.Button(visible=False)
        api_trigger.click(
            fn=chat_api_fn,
            inputs=[api_message],
            outputs=[api_output],
            api_name="chat_once",
            show_api=True,
            queue=False,
        )

        api_turn_message = gr.Textbox(visible=False)
        api_turn_history = gr.Textbox(visible=False)
        api_turn_reply = gr.Textbox(visible=False)
        api_turn_history_out = gr.Textbox(visible=False)
        api_turn_trigger = gr.Button(visible=False)
        api_turn_trigger.click(
            fn=chat_turn_api_fn,
            inputs=[api_turn_message, api_turn_history],
            outputs=[api_turn_reply, api_turn_history_out],
            api_name="chat_turn",
            show_api=True,
            queue=False,
        )

    return demo
