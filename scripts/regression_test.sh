#!/usr/bin/env bash
# Full regression test: run all 4 critical queries + 3 edge cases
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
from ieee_ai_chatbot.vectorstore import get_vector_store
import re

settings = Settings.from_env()
vs = get_vector_store(settings)

QUERIES = [
    "Who is the chairperson of IEEE BSU?",
    "Who are the board of the branch?",
    "Who is the high board?",
    "Who made this chatbot?",
    "What is machine learning?",
    "Who is the AI head?",
]

for q in QUERIES:
    print(f"\n{'='*70}")
    print(f"QUERY: {q}")
    print(f"{'='*70}")
    results = vs.similarity_search_with_score(q, k=10)

    # Dedup
    seen = set()
    unique = []
    for doc, score in results:
        key = re.sub(r"\s+", " ", doc.page_content).strip().casefold()
        if key not in seen and key:
            seen.add(key)
            unique.append((doc, score))
    
    print(f"  Raw: {len(results)} docs | After dedup: {len(unique)} unique")
    for i, (doc, score) in enumerate(unique[:5], 1):
        snippet = doc.page_content[:100].replace("\n", " ")
        src = doc.metadata.get("filename", doc.metadata.get("source", "?"))
        above = "✅" if score >= 0.15 else "❌"
        print(f"  [{i}] {above} score={score:.4f} src={src}")
        print(f"       {snippet!r}")
PYEOF
