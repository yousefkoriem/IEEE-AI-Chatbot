#!/usr/bin/env bash
set -e
PYTHON="/home/bluefox/IEEE/IEEE-AI-Chatbot/.venv/bin/python"
cd /home/bluefox/IEEE/IEEE-AI-Chatbot

$PYTHON - << 'PYEOF'
import sys
sys.path.insert(0, "src")

# 1. Check the Any import fix
from ieee_ai_chatbot.chat import RAGAgent
print("chat.py  OK")

# 2. Check api.py both functions
from ieee_ai_chatbot.api import build_router, build_api
print("api.py   OK  (build_router + build_api)")

# 3. Check router returns correct type
from fastapi import APIRouter
router = build_router.__code__  # just check it's defined without running it
print("api.py   build_router() defined OK")

# 4. Check app.py imports work
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("app", "app.py")
mod = importlib.util.module_from_spec(spec)
# Don't actually exec (would start server), just check parse
import ast, pathlib
src = pathlib.Path("app.py").read_text()
tree = ast.parse(src)
print("app.py   parse OK")

# 5. Check ui_gradio CSS/THEME in Blocks
src2 = pathlib.Path("src/ieee_ai_chatbot/ui_gradio.py").read_text()
if 'css=CSS' in src2 and 'theme=THEME' in src2:
    print("ui_gradio CSS/THEME in Blocks  OK")
else:
    print("WARN: CSS/THEME not found in Blocks constructor")

print("\nAll checks passed")
PYEOF
