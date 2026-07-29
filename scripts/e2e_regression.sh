#!/usr/bin/env bash
# End-to-end LLM answer test for the 4 regression queries
PYTHON="/home/bluefox/IEEE/IEEE-AI-Chatbot/.venv/bin/python"
cd /home/bluefox/IEEE/IEEE-AI-Chatbot

$PYTHON - <<'PYEOF'
import sys, os, logging
sys.path.insert(0, "src")
logging.basicConfig(level=logging.WARNING)

from pathlib import Path
env_file = Path("/home/bluefox/IEEE/IEEE-AI-Chatbot/.env")
for line in env_file.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from ieee_ai_chatbot.config import Settings
from ieee_ai_chatbot.chat import RAGAgent

settings = Settings.from_env()
agent = RAGAgent(settings)

QUERIES = [
    ("Who is the chairperson of IEEE BSU?", "Mohamed Sharaf"),
    ("Who made this chatbot?", "Yousef Koriem"),
    ("Who is the AI head?", "Yousef Koriem"),
    ("Who is the high board?", "Mohamed Sharaf"),
]

for q, expected_name in QUERIES:
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print(f"Expected to contain: {expected_name}")
    print(f"{'='*70}")
    try:
        result = agent.answer(q, history_text="")
        # answer() returns a tuple: (answer, sources, confidence, run_id, suggestions, chunk_ids)
        answer_text = result[0] if isinstance(result, tuple) else str(result)
        confidence = result[2] if isinstance(result, tuple) and len(result) > 2 else "?"
        # Truncate for display
        answer_short = answer_text[:600]
        print(f"Confidence: {confidence}")
        print(f"ANSWER:\n{answer_short}")
        if expected_name.lower() in answer_text.lower():
            print(f"\n✅ PASS — found '{expected_name}' in answer")
        else:
            print(f"\n❌ FAIL — '{expected_name}' NOT found in answer")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
PYEOF
