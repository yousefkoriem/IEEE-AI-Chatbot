---
title: IEEE AI Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
python_version: 3.11
---

# IEEE-AI-Chatbot

RAG chatbot for answering questions about IEEE Beni Suef Student Branch using:

- LangChain for retrieval + response orchestration
- Google GenAI `gemini-2.5-flash-lite` for chat generation
- Pinecone as vector database (dynamic upsert/update/delete)
- LangSmith for tracing/status
- Gradio UI for hosting locally and on Hugging Face Spaces

## Project structure

```text
app.py
main.py
src/ieee_ai_chatbot/
	config.py
	vectorstore.py
	ingest.py
	retrieval.py
	prompts.py
	chat.py
	ui_gradio.py
docs/
	architecture.md
	deployment_hf_spaces.md
docs/pdf/
docs/ppt/
```

## Local setup

1. Install dependencies:

```bash
uv sync
```

or

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
```

Fill in at least:

- `GOOGLE_API_KEY`
- `PINECONE_API_KEY`

3. Run app:

```bash
python app.py
```

## How ingestion works

- **Upload + Index** tab: upload PDF/PPT/PPTX/DOCX/DOC files directly in Gradio and index immediately.
- **Local Sync** button: scans `docs/pdf`, `docs/ppt`, and `docs/doc`, then:
	- indexes new/updated files,
	- skips unchanged files,
	- deletes vectors for removed local files.
- **Crawl Website + Index** button: crawls same-domain pages from a start URL (default: `https://ieee-mangment.vercel.app/`) and indexes page content incrementally.

The incremental state is tracked in `.rag_manifest.json`.

By default, chat answers do not include sources unless you explicitly ask for sources/citations in your prompt.

If Pinecone has no relevant context for a question, the chatbot can fall back to quick web search snippets (configurable via env vars).

For faster responses, defaults are tuned to smaller retrieval depth (`RETRIEVER_K=3`, `RETRIEVER_FETCH_K=10`) and capped generation length (`MAX_OUTPUT_TOKENS=400`).

## LangSmith

Set:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=IEEE-AI-Chatbot`

Then use the **Status** tab in Gradio to verify tracing configuration.

## Hugging Face Spaces

See `docs/deployment_hf_spaces.md` for full steps.

Quick deploy command after `hf auth login`:

```bash
./scripts/deploy_hf_space.sh <your-username/your-space-name> main
```
