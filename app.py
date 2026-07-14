import asyncio
import logging
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

from ieee_ai_chatbot.ui_gradio import create_demo
from ieee_ai_chatbot.api import build_router


# Silence Python 3.11 asyncio shutdown noise (harmless ValueError from Gradio)
if sys.version_info < (3, 12):
    _orig_close = asyncio.selector_events.BaseSelectorEventLoop.close

    def _patched_close(self):
        try:
            _orig_close(self)
        except ValueError:
            pass

    asyncio.selector_events.BaseSelectorEventLoop.close = _patched_close


def main() -> None:
    demo = create_demo()

    # CSS and THEME are now baked into the gr.Blocks() constructor inside
    # create_demo(), so they don't need to be passed to launch() again.
    _, _local, _share = demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True,
    )

    # ── Mount the REST API router onto Gradio's underlying FastAPI app ──
    # This makes /api/v1/* endpoints available on the SAME port as the UI,
    # which is required for Hugging Face Spaces (only one port exposed).
    router = build_router()
    demo.app.include_router(router, prefix="/api/v1")

    # Keep the process alive (launch() returned because prevent_thread_lock=True)
    import threading
    threading.Event().wait()


if __name__ == "__main__":
    main()
