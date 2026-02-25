#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <space_id> [branch]"
  echo "Example: $0 your-username/ieee-ai-chatbot-space main"
  exit 1
fi

SPACE_ID="$1"
BRANCH="${2:-main}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: Hugging Face CLI is not installed or not in PATH (expected command: hf)."
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Error: Hugging Face CLI is not authenticated. Run: hf auth login"
  exit 1
fi

if ! hf repo create "$SPACE_ID" --type space --space_sdk gradio >/dev/null 2>&1; then
  echo "Space may already exist or could not be created. Continuing to push..."
fi

HF_REMOTE_URL="https://huggingface.co/spaces/${SPACE_ID}"

if git remote get-url hf >/dev/null 2>&1; then
  git remote set-url hf "$HF_REMOTE_URL"
else
  git remote add hf "$HF_REMOTE_URL"
fi

git push hf "${BRANCH}:main"

echo "Deployment push finished: ${HF_REMOTE_URL}"
