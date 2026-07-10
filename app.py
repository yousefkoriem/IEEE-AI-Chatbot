import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ieee_ai_chatbot.ui_gradio import create_demo, CSS, THEME


def main() -> None:
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=THEME)


if __name__ == "__main__":
    main()
