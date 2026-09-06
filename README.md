---
title: IEEE AI Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# IEEE-AI-Chatbot

RAG chatbot for answering questions about IEEE Beni Suef Student Branch using:

- LangChain for retrieval + response orchestration
- Google GenAI `gemini-2.5-flash` for chat generation
- Pinecone as vector database (dynamic upsert/update/delete)
- LangSmith for tracing/status
- Gradio UI + FastAPI REST API (same port, no extra infra needed)

## Project structure

```text
app.py                    # Application entry point
config/                   # Environment variables and configuration
models/                   # Gemini Flash and Flash-Lite models
prompts/                  # System prompts
agent/                    # LangGraph agent and state
rag/                      # Embeddings, Pinecone, retrieval, ingestion
ui/                       # Gradio interface
utils/                    # Logging and shared helpers
evaluation/               # RAG and LangSmith evaluation
```

## REST API

The REST API is served on the **same port as the Gradio UI** (7860 on HF Spaces).  
Interactive docs: `https://your-space.hf.space/api/v1/docs`

### Endpoints

| Method   | Path                              | Description                               |
|----------|-----------------------------------|-------------------------------------------|
| `GET`    | `/api/v1/health`                  | Liveness probe                            |
| `GET`    | `/api/v1/status`                  | Model + KB status                         |
| `POST`   | `/api/v1/chat`                    | Stateless Q&A — you manage history        |
| `POST`   | `/api/v1/chat/session`            | Stateful Q&A — server stores memory       |
| `GET`    | `/api/v1/chat/history/{key}`      | Fetch conversation history                |
| `DELETE` | `/api/v1/chat/history/{key}`      | Clear a session                           |

### Quick example (JavaScript)

```js
// Stateful — server remembers the conversation
const res = await fetch("https://your-space.hf.space/api/v1/chat/session", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "What is the IEEE CS chapter about?",
    session_key: "user-abc-123",        // any unique string
    generate_suggestions: true
  })
});
const { answer, sources, confidence, suggestions } = await res.json();
```

```js
// Stateless — pass your own history array
const res = await fetch("https://your-space.hf.space/api/v1/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "Who is the chair?",
    history: [
      { role: "user",      content: "Tell me about IEEE." },
      { role: "assistant", content: "IEEE is a global organization..." }
    ]
  })
});
const { answer } = await res.json();
```

## Local setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
# Fill in GOOGLE_API_KEY, PINECONE_API_KEY at minimum
```

3. Run (Gradio UI + REST API on port 7860):

```bash
python app.py
```

Or run only the standalone REST API on port 8000:

```bash
uvicorn app_api:api_app --host 0.0.0.0 --port 8000 --reload
```

## Hugging Face Spaces

Set the following secrets in your Space settings:

| Secret | Required |
|--------|----------|
| `GOOGLE_API_KEY` | ✅ |
| `PINECONE_API_KEY` | ✅ |
| `PINECONE_INDEX_NAME` | ✅ |
| `LANGSMITH_API_KEY` | optional |

Deploy command (after `hf auth login`):

```bash
./scripts/deploy_hf_space.sh <your-username/your-space-name> main
```

## LangSmith tracing

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` to enable run tracking.
Check the **Status** panel in the Gradio sidebar to verify.
