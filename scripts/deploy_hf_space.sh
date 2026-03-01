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

if GIT_TERMINAL_PROMPT=0 git push hf "${BRANCH}:main"; then
  echo "Deployment push finished via git: ${HF_REMOTE_URL}"
  exit 0
fi

echo "Git push failed; falling back to 'hf upload' (single-commit deploy)."

HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

TOKEN_ARG=()
if [[ -n "$HF_TOKEN" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

hf upload "$SPACE_ID" . . \
  --repo-type space \
  --commit-message "Deploy from $(basename "$(pwd)")" \
  --exclude ".git/*" \
  --exclude ".venv/*" \
  --exclude "__pycache__/*" \
  --exclude "**/__pycache__/*" \
  --exclude ".env" \
  "${TOKEN_ARG[@]}"

echo "Deployment finished via hf upload: ${HF_REMOTE_URL}"
