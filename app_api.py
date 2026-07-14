"""
IEEE AI Chatbot — Standalone REST API Entry Point
==================================================
Runs the FastAPI REST server independently of Gradio on a separate port.
Use this when you want the API without the UI, or on a separate host.

Usage:
    uvicorn app_api:api_app --host 0.0.0.0 --port 8000 --reload
    python app_api.py

For HF Spaces, the API is instead mounted ON Gradio's port 7860 (via app.py).
This file is for local development or separate API deployments.

Environment variables:
    API_CORS_ORIGINS  Comma-separated allowed origins. Default: * (all)
    API_PORT          Port when run directly. Default: 8000
    API_HOST          Bind host. Default: 0.0.0.0
"""
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ieee_ai_chatbot.api import build_api

# Build the standalone FastAPI app — importable by uvicorn as `app_api:api_app`
api_app = build_api()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("app_api:api_app", host=host, port=port, reload=False)
