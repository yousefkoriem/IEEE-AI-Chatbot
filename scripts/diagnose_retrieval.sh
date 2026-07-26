#!/usr/bin/env bash
# Deep diagnostic: trace exactly what happens for a chairman question
PYTHON="/home/bluefox/IEEE/IEEE-AI-Chatbot/.venv/bin/python"
cd /home/bluefox/IEEE/IEEE-AI-Chatbot

$PYTHON - << 'PYEOF'
import sys, os, logging
sys.path.insert(0, "src")
logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

# Load .env manually
from pathlib import Path
env_file = Path("/home/bluefox/IEEE/IEEE-AI-Chatbot/.env")
for line in env_file.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from ieee_ai_chatbot.config import Settings
from ieee_ai_chatbot.prompts import build_system_prompt, build_user_prompt, build_prompt_config, CONTEXT_AVAILABLE_INSTRUCTION

settings = Settings.from_env()
print(f"\n=== CONFIG ===")
print(f"  PINECONE_INDEX_NAME : {settings.pinecone_index_name}")
print(f"  PINECONE_NAMESPACE  : {settings.pinecone_namespace}")
print(f"  RETRIEVER_K         : {settings.retriever_k}")
print(f"  CHAT_MODEL          : {settings.chat_model}")
print(f"  SYSTEM_PROMPT       : {build_system_prompt(settings)[:120]}...")

from ieee_ai_chatbot.vectorstore import get_vector_store
vs = get_vector_store(settings)

question = "who is the chairman of the branch"
print(f"\n=== RAW RETRIEVAL for: {question!r} ===")
results = vs.similarity_search_with_score(question, k=settings.retriever_k)
print(f"  Got {len(results)} docs from Pinecone")
for i, (doc, score) in enumerate(results, 1):
    snippet = doc.page_content[:120].replace("\n", " ")
    print(f"  [{i}] score={score:.4f} | {snippet!r}")

print("\n=== USER PROMPT that would be sent to LLM ===")
if results:
    context = "\n\n".join(doc.page_content for doc, _ in results)
    prompt = build_user_prompt(
        question=question,
        history_text="",
        context=context,
        prompt_config=build_prompt_config(settings),
    )
    print(prompt[:1500])
else:
    print("  NO CONTEXT — would use NO_CONTEXT_INSTRUCTION")
PYEOF
