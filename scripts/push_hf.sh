#!/usr/bin/env bash
set -e
cd /home/bluefox/IEEE/IEEE-AI-Chatbot

git add \
  src/ieee_ai_chatbot/prompts.py \
  src/ieee_ai_chatbot/chat.py \
  src/ieee_ai_chatbot/config.py \
  .env.example

git status --short

git commit -m "fix: improve retrieval quality for people/leadership questions

Root causes of 'chairman' not being answered:
1. System prompt applied date-caution rules to ALL facts, causing
   the LLM to hedge even when the answer was clearly in context.
   Fix: separate date-specific caution from general fact answering;
   add explicit instruction to trust retrieved context for people/roles.

2. CONTEXT_AVAILABLE_INSTRUCTION was too weak ('use context first'),
   allowing the LLM to ignore it. Fix: explicitly say 'do NOT say you
   don't know if the answer is present in context below'.

3. Confidence thresholds (0.85/0.70) were too high for Pinecone cosine
   similarity. Chairman chunks scored 0.63-0.74 which was always 'Low',
   subtly priming the LLM to be less confident. Fix: 0.75/0.60.

4. Default RETRIEVER_K was 3. Raised to 5 so the relevant chunk is
   less likely to be displaced by noisy results."

echo "Committed. Pushing to HF..."
git push hf HEAD:main
echo "Done."
