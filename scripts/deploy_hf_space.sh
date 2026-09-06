#!/usr/bin/env bash
# deploy_hf_space.sh — Deploy IEEE AI Chatbot to Hugging Face Spaces
#
# Usage:
#   ./scripts/deploy_hf_space.sh <space-name> [branch]
#
# Example:
#   ./scripts/deploy_hf_space.sh yousefkoriem/IEEE-AI-Chatbot main
#
# Prerequisites:
#   - huggingface-cli installed and authenticated (huggingface-cli login)
#   - Git LFS installed

set -euo pipefail

SPACE_NAME="${1:?Usage: $0 <username/space-name> [branch]}"
BRANCH="${2:-main}"
REPO_URL="https://huggingface.co/spaces/${SPACE_NAME}"

echo "=========================================="
echo " IEEE AI Chatbot — HF Spaces Deployment"
echo "=========================================="
echo " Space:  ${SPACE_NAME}"
echo " Branch: ${BRANCH}"
echo " URL:    ${REPO_URL}"
echo "=========================================="

# Check auth
if ! huggingface-cli whoami &>/dev/null; then
    echo "❌ Not logged in. Run: huggingface-cli login"
    exit 1
fi

# Add HF remote if not present
if ! git remote get-url hf &>/dev/null; then
    echo "➕ Adding HF remote..."
    git remote add hf "${REPO_URL}"
fi

# Push to HF Spaces
echo "🚀 Pushing ${BRANCH} to HF Spaces..."
git push hf "${BRANCH}:main" --force

echo ""
echo "✅ Deployed successfully!"
echo "🌐 View at: ${REPO_URL}"
echo "📄 API docs: ${REPO_URL}/?view=api"
