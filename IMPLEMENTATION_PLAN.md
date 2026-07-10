# IEEE AI Chatbot — Comprehensive Implementation Plan

> **Project**: RAG chatbot for IEEE Beni Suef Student Branch  
> **Stack**: Gradio 6.20, LangChain, Google Gemini, Pinecone, LangSmith  
> **Date**: 2026-07-10

---

## Table of Contents

1. [Features](#features)
   - [1. Persistent Chat History (SQLite)](#1-persistent-chat-history-sqlite)
   - [2. Fix .env Key Exposure](#2-fix-env-key-exposure)
   - [3. Rate Limiting / Abuse Protection](#3-rate-limiting--abuse-protection)
   - [4. Document Preview in UI](#4-document-preview-in-ui)
   - [5. Chunk/Vector Management UI](#5-chunkvector-management-ui)
   - [6. Multi-Language Support (i18n)](#6-multi-language-support-i18n)
   - [7. Async Migration](#7-async-migration)
   - [8. Follow-Up Suggestions](#8-follow-up-suggestions)
   - [9. Admin Dashboard](#9-admin-dashboard)
   - [10. Auto Re-Ingestion (File Watcher)](#10-auto-re-ingestion-file-watcher)
   - [11. Multiple Vector DB Support](#11-multiple-vector-db-support)
   - [12. Search Engine Fallback Diversity](#12-search-engine-fallback-diversity)
   - [13. Shareable Answer Links](#13-shareable-answer-links)
   - [14. RAG Feedback Loop](#14-rag-feedback-loop)
   - [15. Batch Document Upload from URL](#15-batch-document-upload-from-url)
   - [16. Citation Highlighting](#16-citation-highlighting)
2. [UI Updates](#ui-updates)
   - [Chat Tab](#chat-tab)
   - [Ingest Tab](#ingest-tab)
   - [Analytics Tab](#analytics-tab)
   - [General UI/UX](#general-uiux)

---

## Features

### 1. Persistent Chat History (SQLite)

**Goal**: Replace in-memory `session_histories` dict with SQLite-backed storage.

**Files**: `src/ieee_ai_chatbot/chat_history.py` (new), `src/ieee_ai_chatbot/config.py`, `src/ieee_ai_chatbot/ui_gradio.py`

**Steps**:
1. Add `chat_history_db_path: str = "chat_history.db"` to `Settings` dataclass and `from_env` in `config.py`
2. Create `ChatHistoryManager` class in `chat_history.py`:
   - `__init__(self, db_path)` — connect SQLite, `CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY AUTOINCREMENT, session_key TEXT, title TEXT, created_at TIMESTAMP, updated_at TIMESTAMP)` and `messages(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER REFERENCES conversations(id), role TEXT, content TEXT, created_at TIMESTAMP)`
   - `get_or_create_conversation(session_key: str) -> int`
   - `add_message(conv_id: int, role: str, content: str)`
   - `get_history(conv_id: int, limit: int = 30) -> list[dict]`
   - `list_conversations(session_key: str) -> list[dict]`
   - `delete_conversation(conv_id: int)`
   - `rename_conversation(conv_id: int, title: str)`
3. In `ui_gradio.py`:
   - Initialize `ChatHistoryManager` alongside `agent`
   - Replace `session_histories` usage with manager calls
   - Track `active_conv_id: gr.State`
   - On first message: get_or_create conversation, add messages as they come
4. Add periodic cleanup of conversations older than 30 days

---

### 2. Fix .env Key Exposure

**Goal**: Remove live API keys from git history and prevent future commits.

**Files**: `.gitignore`, `.env`, `.env.example`

**Steps**:
1. Ensure `.env` is listed in `.gitignore`
2. Remove from git tracking: `git rm --cached .env`
3. Scrub `.env.example` — ensure it contains only placeholder values
4. **Rotate all API keys** on provider dashboards (Google, Pinecone, LangSmith)
5. Update local `.env` with new rotated keys
6. Commit the fix

---

### 3. Rate Limiting / Abuse Protection

**Goal**: Prevent excessive requests per session/IP.

**Files**: `src/ieee_ai_chatbot/rate_limiter.py` (new), `src/ieee_ai_chatbot/config.py`, `src/ieee_ai_chatbot/ui_gradio.py`

**Steps**:
1. Add config: `rate_limit_max_requests: int = 30`, `rate_limit_window_seconds: int = 60`
2. Create `RateLimiter` class:
   - `dict[str, list[float]]` mapping key -> timestamps
   - `check(key: str) -> tuple[bool, int]` — (allowed, remaining)
   - `_cleanup()` — remove expired entries
3. In `ui_gradio.py`:
   - Apply to `chat_fn`, `chat_api_fn`, `chat_turn_api_fn`
   - Apply to ingest endpoints (upload, sync, crawl, text)
   - Return friendly rate limit message with retry-after info
4. Periodic cleanup (every N requests)

---

### 4. Document Preview in UI

**Goal**: View uploaded files (PDF, DOCX, MD, etc.) directly in the browser.

**Files**: `src/ieee_ai_chatbot/ui_gradio.py`, `src/ieee_ai_chatbot/ingest.py`

**Steps**:
1. Add `uploads/` directory config — store uploaded files to known location
2. For each ingested file in File Manager, add "Preview" button
3. On click:
   - **PDF**: Embed via `<iframe>` with Google Docs viewer or serve via Gradio file endpoint
   - **MD/HTML/TXT**: Read and display in `gr.Markdown` or `gr.Textbox`
   - **DOCX/PPT**: Extract text (reuse `_extract_*_text` from `ingest.py`), display as formatted text
4. Use `gr.Modal` to show preview
5. Add `gr.DownloadButton` for direct file download

---

### 5. Chunk/Vector Management UI

**Goal**: Browse, search, and delete sources/chunks from the UI.

**Files**: `src/ieee_ai_chatbot/ui_gradio.py`, `src/ieee_ai_chatbot/ingest.py`, `src/ieee_ai_chatbot/stats.py`

**Steps**:
1. Add new top-level tab "🔍 Vectors" or section in Ingest
2. New functions:
   - `list_all_sources(settings)` — from manifest, return source_id, origin, chunk count, hash, timestamp
   - `search_chunks(settings, query, top_k=10)` — similarity search, return (chunk_id, score, snippet, source)
   - `delete_source(settings, source_id)` — delete chunk_ids from Pinecone + remove from manifest
   - `delete_multiple_sources(settings, source_ids: list)`
3. UI:
   - Search bar + "Search Vectors" button → results table
   - Source Manager: list of all sources with origin badge, chunk count, delete button
   - Confirmation dialog before delete

---

### 6. Multi-Language Support (i18n)

**Goal**: UI and prompts in English and Arabic.

**Files**: `src/ieee_ai_chatbot/i18n.py` (new), `src/ieee_ai_chatbot/prompts.py`, `src/ieee_ai_chatbot/config.py`, `src/ieee_ai_chatbot/ui_gradio.py`

**Steps**:
1. Create `i18n.py`:
   - `TRANSLATIONS = {"en": {...}, "ar": {...}}` — all UI strings
   - `get_text(key: str, lang: str = "en") -> str`
   - `LANG_COOKIE = "ieee_lang"`
2. Add `language: str = "en"` to `Settings`
3. In `prompts.py` — add Arabic system prompt variant
4. In `ui_gradio.py`:
   - Wrap all visible strings with `get_text(key, lang)`
   - Language switcher dropdown in header
   - Store in `gr.State`, pass through handlers
   - For Arabic: `dir="rtl"`, switch font
5. LLM instructed to respond in user's language via system prompt

---

### 7. Async Migration

**Goal**: Convert synchronous I/O to `asyncio` to avoid blocking under concurrent users.

**Files**: `src/ieee_ai_chatbot/chat.py`, `ingest.py`, `retrieval.py`, `vectorstore.py`, `ui_gradio.py`, `analytics.py`

**Steps**:
1. Install `httpx`: `pip install httpx`
2. `retrieval.py`: `requests.get` → `httpx.AsyncClient.get`, rename to `async_search_web_snippets`
3. `ingest.py`:
   - `_crawl_same_domain` → use `httpx.AsyncClient` with `asyncio.Semaphore`
   - `ingest_files`, `ingest_website`, `ingest_text` → async
   - CPU-bound parsing → `asyncio.to_thread`
   - Pinecone ops → `run_in_executor` or verify async support
4. `chat.py`:
   - `answer()`, `answer_stream()` → async
   - `_invoke_with_retry_and_fallback` → `asyncio.sleep`
5. `ui_gradio.py`: handlers → `async def`, use `await`
6. `analytics.py` → async

---

### 8. Follow-Up Suggestions

**Goal**: After each answer, suggest 2-3 follow-up questions.

**Files**: `src/ieee_ai_chatbot/chat.py`, `src/ieee_ai_chatbot/prompts.py`, `src/ieee_ai_chatbot/ui_gradio.py`

**Steps**:
1. After generating answer, make lightweight LLM call: "Given context: {context} and Q&A: Q:{q} A:{a}, suggest 2-3 short follow-up questions."
2. Return suggestions alongside answer + sources + confidence
3. Display as clickable chips below answer
4. Click chip → auto-fill message box → submit

---

### 9. Admin Dashboard

**Goal**: Centralized view of system health, sources, config.

**Files**: `src/ieee_ai_chatbot/ui_gradio.py`, `analytics.py`, `stats.py`

**Steps**:
1. Add "🛠️ Admin" top-level tab
2. Sections:
   - System Health — agent status, Pinecone index info
   - Source Management — full source list with delete (reuse from #5)
   - LangSmith Traces — recent runs table with drill-down
   - Configuration — read-only display (keys masked)
3. Add `get_pinecone_index_stats(settings)` to `stats.py`

---

### 10. Auto Re-Ingestion (File Watcher)

**Goal**: Watch `docs/` directories and re-index on file changes.

**Files**: `src/ieee_ai_chatbot/watcher.py` (new), `app.py`, `ui_gradio.py`

**Steps**:
1. Create `watcher.py` using `watchdog`:
   - `DocWatcher(settings, callback)` — observe `docs/pdf`, `docs/ppt`, `docs/doc`
   - Debounce 300ms, call `ingest_files` or `sync_local_docs`
2. In `app.py`: start watcher before demo, stop on shutdown
3. In `ui_gradio.py`: toggle switch for auto-sync

---

### 11. Multiple Vector DB Support

**Goal**: Abstract Pinecone behind interface supporting Chroma, FAISS, Qdrant.

**Files**: `src/ieee_ai_chatbot/vectorstore.py`, `config.py`, new adapters

**Steps**:
1. Define `VectorStoreProtocol` (or ABC)
2. Implement adapters: `PineconeVectorStoreAdapter`, `ChromaVectorStoreAdapter`, `FAISSVectorStoreAdapter`, `QdrantVectorStoreAdapter`
3. Add `vector_store_type: str = "pinecone"` to config
4. Factory function `get_vector_store(settings)` dispatches to correct adapter
5. Optional extras in `pyproject.toml`

---

### 12. Search Engine Fallback Diversity

**Goal**: Support Tavily, SerpAPI, Bing as DuckDuckGo alternatives.

**Files**: `src/ieee_ai_chatbot/retrieval.py`, `config.py`

**Steps**:
1. Provider registry: `WEB_SEARCH_PROVIDERS = {"duckduckgo": ..., "tavily": ..., "serpapi": ..., "bing": ...}`
2. Config: `web_search_provider`, + API key vars for each provider
3. Normalize all results to common `Document` format
4. Dropdown in UI to select active provider

---

### 13. Shareable Answer Links

**Goal**: Permanent links to specific Q&A pairs.

**Files**: `src/ieee_ai_chatbot/sharing.py` (new), `ui_gradio.py`

**Steps**:
1. Hash (question + answer + timestamp) → short share ID
2. Store in SQLite `shared_answers` table (or reuse chat history DB)
3. "Share" button per answer → copy link to clipboard via JS
4. Gradio page/endpoint renders shared Q&A

---

### 14. RAG Feedback Loop

**Goal**: Use thumbs up/down to boost/penalize chunk retrieval weights.

**Files**: `src/ieee_ai_chatbot/chat.py`, `retrieval.py`, `analytics.py`

**Steps**:
1. Track chunk_ids per question (store in SQLite alongside history)
2. Record feedback per chunk: `chunk_feedback` table
3. `ChunkBoostScorer`: `adjusted_score = original * (1 + boost_factor * chunk_boost)`
4. Apply in `_retrieve_docs()` after Pinecone results
5. Config: `feedback_boost_enabled`, `feedback_boost_factor`

---

### 15. Batch Document Upload from URL

**Goal**: Ingest from a list of URLs (not just single crawl).

**Files**: `src/ieee_ai_chatbot/ingest.py`, `ui_gradio.py`

**Steps**:
1. `ingest_url_list(settings, urls: list[str])` — fetch, extract, chunk, index each URL
2. UI: multiline textbox for URLs + "Crawl & Index" button
3. Process sequentially with per-URL timeout

---

### 16. Citation Highlighting

**Goal**: Highlight passages in answer that map to source chunks.

**Files**: `src/ieee_ai_chatbot/chat.py`, `ui_gradio.py`

**Steps**:
1. After answer, use string overlap or embedding similarity to map sentences → source chunks
2. Wrap matches in `<span class="citation" title="Source: filename">text</span>`
3. Render answer as `gr.HTML` with citation CSS styling
4. Tooltip on hover shows source name

---

## UI Updates

### Chat Tab

#### A. Chat History Sidebar
- Add left sidebar column with conversation list
- Load/save conversations via `ChatHistoryManager`
- "New Chat" button, click to load, delete button per conversation
- Store `active_conv_id` in `gr.State`

#### B. Markdown Rendering Improvements
- Add KaTeX for LaTeX rendering (CSS + JS)
- Table CSS styling (borders, alternating rows, scroll)
- Code block "Copy" button via JS
- Consider custom `gr.HTML` rendering if Gradio built-in is insufficient

#### C. Typing Indicator Animation
- CSS animated dots (`@keyframes` shimmer)
- Show during streaming, replace with actual content on first chunk
- Graceful removal on error/end

#### D. Source Preview Popover
- Render source citations as clickable chips/chips
- Click → show `gr.Modal` with source excerpt, filename, confidence
- Or use inline `gr.Accordion` expansion

#### E. Confidence Badge
- Color-coded badge below answer: 🟢 High / 🟡 Medium / 🔴 Low / 🌐 Web Search / ⚪ None
- CSS: colored dot + label, right-aligned in bubble footer

#### F. Regenerate Button
- "🔄 Regenerate" button next to feedback buttons
- On click: remove last assistant message, re-run with same user message
- Disabled during streaming; only available if there's ≥1 pair

#### G. Edit Sent Messages
- "✏️ Edit" button on last user message
- On click: copy text to input, remove last user+assistant pair
- Only for most recent message; disabled during streaming

---

### Ingest Tab

#### A. Ingestion Progress Bars
- Replace text status with `gr.Progress` + `gr.HTML` updates
- File upload: "Processing file 3/5: name.pptx"
- Website crawl: "Crawling 12/25: url"
- Use generator functions that yield `(status_msg, gr.Progress(fraction))`

#### B. File Manager View
- Table of all ingested sources: Name, Origin (badge), Chunks, Last Indexed, Delete
- Populate from manifest, refreshable
- Filter by origin: All / Uploaded / Local / Website / Text
- "Delete All" with confirmation

#### C. Website Crawl Preview
- "Preview URLs" button — crawl without indexing, show discovered URLs
- Checkbox selection to exclude URLs
- "Index Selected" button proceeds with chosen URLs

#### D. Ingestion Log Panel
- Collapsible `gr.Accordion` at bottom: "📋 Ingestion Log"
- Custom `logging.Handler` that captures last 100 entries in deque
- Display with timestamps, auto-scroll, "Clear Log" button

---

### Analytics Tab

#### A. Time Range Selector
- `gr.Radio` or `gr.Dropdown`: "Last 24h", "Last 7 days", "Last 30 days", "All time"
- Pass to analytics functions, compute `start_time` accordingly
- Auto-refresh on change

#### B. Visual Charts
- `gr.Plot` with matplotlib (dark theme styling)
- Latency trend: line chart (avg_ms per day)
- Feedback trend: stacked bar (👍/👎 per day)
- Query volume: bar chart (queries per day)

#### C. Top Queries Leaderboard
- Count question frequency from recent runs
- Display as ranked table: Question, Count, Avg Latency

#### D. Retrieval Quality Metrics
- Confidence distribution: pie chart (High/Medium/Low/Web/None)
- Context usage rate: % with context vs. web search vs. none
- Requires storing confidence in run metadata

#### E. Export Analytics
- "📥 Export CSV" button
- Columns: timestamp, question, latency_ms, confidence, feedback, sources
- Limit to last 500 runs

---

### General UI/UX

#### A. Responsive Layout (Tablet)
- CSS media queries for 1024px and 768px breakpoints
- Stack sidebars below main content on smaller screens
- Wrap suggestion cards to grid
- Ensure no horizontal scroll

#### B. Toast Notifications
- Custom `gr.HTML` toast container
- JS: `showToast(message, type, duration)`
- CSS: fixed position, animated slide-in, auto-dismiss
- Types: success (green), error (red), info (blue)

#### C. Keyboard Shortcuts
- JS event listeners scoped to Gradio container
- `Ctrl+Enter` → send message
- `Esc` → cancel streaming (via state signal)
- `Ctrl+K` → focus input
- `Ctrl+Shift+C` → clear conversation

#### D. Custom Loading Skeleton
- Replace welcome cards with skeleton placeholders during streaming
- CSS shimmer animation
- Smooth transition to actual content

#### E. Settings Panel
- Gear icon ⚙️ in header → `gr.Modal`
- Controls: Model, Temperature, Max Tokens, Retriever-K, Internet Fallback, Language
- Store overrides in `gr.State`
- "Reset to Defaults" button
- Apply overrides via mutable agent properties

---

## Implementation Order (Recommended)

### Phase 1 — Foundation & Security
1. Fix `.env` key exposure
2. Persistent chat history (SQLite)
3. Rate limiting

### Phase 2 — Core UI Improvements
4. Chat UI updates (history sidebar, confidence badge, typing indicator, regenerate, edit)
5. Toast notifications
6. Keyboard shortcuts

### Phase 3 — Management & Observability
7. File Manager + Chunk/Vector Management
8. Admin Dashboard
9. Document Preview

### Phase 4 — Advanced Features
10. Multi-language support
11. Follow-up suggestions
12. Settings Panel

### Phase 5 — Performance & Scale
13. Async migration
14. Auto re-ingestion (file watcher)

### Phase 6 — Polish
15. Analytics improvements (charts, time range, export)
16. Responsive layout
17. Ingest progress bars + log panel
18. Loading skeleton
19. Crawl preview
20. Source preview popover

### Phase 7 — Future Enhancements
21. Multiple vector DB support
22. Search engine fallback diversity
23. RAG feedback loop
24. Batch URL upload
25. Citation highlighting
26. Shareable answer links
