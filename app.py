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

from ieee_ai_chatbot.ui_gradio import create_demo, CSS, THEME


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
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=THEME)


if __name__ == "__main__":
    main()
