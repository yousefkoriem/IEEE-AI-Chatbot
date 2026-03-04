# Architecture

## Components

- `config.py`: environment-driven settings, required key validation, LangSmith runtime toggles.
- `vectorstore.py`: Pinecone index creation/check and LangChain Pinecone vector store binding.
- `ingest.py`: document extraction (PDF/PPT/PPTX/DOCX/DOC + website crawl), chunking, deterministic IDs, upsert/delete, manifest tracking.
- `retrieval.py`: MMR retrieval strategy.
- `prompts.py`: centralized system prompt, user prompt assembly, and prompt configuration/validation helpers.
- `chat.py`: RAG orchestration with Google `gemini-2.5-flash-lite`.
- `ui_gradio.py`: user interface for chat, upload indexing, local sync, and status.

## Data flow

1. Ingestion extracts raw text from PDF/PPT files or crawled website pages.
2. Text is chunked with `RecursiveCharacterTextSplitter`.
3. Chunks are embedded with `models/gemini-embedding-001`.
4. Embeddings are stored in Pinecone with metadata.
5. Query triggers MMR retrieval from Pinecone.
6. Retrieved chunks are injected into the generation prompt.
7. LLM returns answer + listed sources.

## Optimization choices

- Incremental indexing by file hash to skip unchanged files.
- Deterministic chunk IDs for predictable updates.
- Local-file deletion reconciliation removes stale vectors.
- MMR retrieval reduces redundant context.
- Configurable chunk sizes and retrieval depth via environment variables.
