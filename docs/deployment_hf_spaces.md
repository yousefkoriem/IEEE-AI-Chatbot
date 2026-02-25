# Deploy to Hugging Face Spaces

## 1) Create Space

- Create a new **Gradio** Space.
- Set SDK to Gradio.

## 2) Push repository

Push this repository contents so Space includes:

- `app.py` (entrypoint)
- `requirements.txt`
- `src/ieee_ai_chatbot/*`

## 3) Add Space secrets

In Space Settings → Variables and secrets:

- `GOOGLE_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_NAMESPACE`
- `PINECONE_CLOUD`
- `PINECONE_REGION`
- `CHAT_MODEL` (default: `gemini-2.5-flash-lite`)
- `EMBEDDING_MODEL` (default: `models/gemini-embedding-001`)
- `WEBSITE_DEFAULT_URL` (default: `https://ieee-mangment.vercel.app/`)
- `WEBSITE_MAX_PAGES` (default: `25`)
- `WEBSITE_TIMEOUT_SECONDS` (default: `20`)

Optional LangSmith:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`

## 4) Runtime notes

- This project targets Python `>=3.11` for better Spaces compatibility.
- If your Space uses a different default Python, set a compatible version in Space settings when available.

## 5) Verify

- Open the Space UI.
- In **Status** tab, verify `ready: yes` and LangSmith status.
- Upload a sample PDF/PPT and ask a question in **Chat** tab.
