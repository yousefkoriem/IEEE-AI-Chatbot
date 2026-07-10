from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
import gradio as gr

from .chat import RAGAgent
from .config import Settings
from .ingest import ingest_files, ingest_website, sync_local_docs, ingest_text
from .stats import get_kb_stats
from .analytics import get_recent_runs, get_feedback_summary, get_latency_stats



CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root, .gradio-container {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    
    /* Gradio Theme overrides to support IEEE Blue and CS Orange */
    --primary-500: #00629B !important;
    --primary-600: #004b77 !important;
    --button-primary-background-fill: linear-gradient(135deg, #00629B 0%, #004b77 100%) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #F58220 0%, #d86d12 100%) !important;
}

.welcome-title {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00629B 0%, #F58220 50%, #8E2DE2 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    margin-top: 2rem !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: -1px !important;
    line-height: 1.25 !important;
}

.welcome-subtitle {
    font-size: 1.4rem !important;
    color: var(--body-text-color-subdued, #555) !important;
    text-align: center !important;
    margin-bottom: 2.5rem !important;
    font-weight: 400 !important;
}

.suggestion-card {
    background: var(--block-background-fill, #f8f9fa) !important;
    border: 1px solid var(--border-color-primary, #e5e7eb) !important;
    border-radius: 16px !important;
    padding: 18px 14px !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--body-text-color, #1f2937) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    min-height: 80px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.suggestion-card:hover {
    border-color: #F58220 !important;
    background: var(--block-background-fill-hover, #ffffff) !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 15px -3px rgba(245, 130, 32, 0.15), 0 4px 6px -2px rgba(245, 130, 32, 0.1) !important;
}

.chatbot-container {
    border-radius: 20px !important;
    border: 1px solid var(--border-color-primary, #e5e7eb) !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05) !important;
    overflow: hidden !important;
}

/* Custom chatbot message bubbles reflecting colors */
.chatbot-container .message.user, .chatbot-container .user {
    background-color: rgba(0, 98, 155, 0.07) !important;
    border: 1px solid rgba(0, 98, 155, 0.12) !important;
}

.chatbot-container .message.bot, .chatbot-container .bot, .chatbot-container .assistant {
    background-color: rgba(245, 130, 32, 0.05) !important;
    border: 1px solid rgba(245, 130, 32, 0.12) !important;
}

.input-row {
    background: var(--background-fill-secondary, #f3f4f6) !important;
    border-radius: 30px !important;
    padding: 6px 16px !important;
    border: 1px solid var(--border-color-primary, #e5e7eb) !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02) !important;
    margin-top: 10px !important;
}

.input-row:focus-within {
    border-color: #00629B !important;
    background: var(--background-fill-primary, #ffffff) !important;
    box-shadow: 0 0 0 3px rgba(0, 98, 155, 0.15) !important;
}

.input-textbox {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    flex-grow: 1 !important;
}

.input-textbox textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.send-btn {
    background: linear-gradient(135deg, #00629B 0%, #004b77 100%) !important;
    color: white !important;
    border-radius: 50% !important;
    border: none !important;
    min-width: 42px !important;
    max-width: 42px !important;
    height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 6px -1px rgba(0, 98, 155, 0.3) !important;
    transition: all 0.2s ease !important;
    padding: 0 !important;
    font-size: 1.15rem !important;
}

.send-btn:hover {
    background: linear-gradient(135deg, #F58220 0%, #d86d12 100%) !important;
    box-shadow: 0 4px 10px rgba(245, 130, 32, 0.4) !important;
    transform: scale(1.05) !important;
}

.feedback-row {
    justify-content: center !important;
    gap: 12px !important;
    margin-top: 8px !important;
}

.feedback-btn {
    border-radius: 20px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    background: var(--block-background-fill, #f3f4f6) !important;
    border: 1px solid var(--border-color-primary, #e5e7eb) !important;
    color: var(--body-text-color, #374151) !important;
    transition: all 0.2s ease !important;
}

.feedback-btn:hover {
    background: var(--background-fill-primary, #ffffff) !important;
    border-color: #00629B !important;
    color: #00629B !important;
}

.clear-btn {
    border-radius: 20px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    background: #fee2e2 !important;
    border: 1px solid #fca5a5 !important;
    color: #991b1b !important;
    transition: all 0.2s ease !important;
}

.clear-btn:hover {
    background: #fecaca !important;
    color: #7f1d1d !important;
}

/* Sidebar Tab buttons styling to show Orange and Blue themes */
.sidebar-tabs .tab-nav button {
    font-size: 0.85rem !important;
    padding: 6px 10px !important;
    border-bottom: 2px solid transparent !important;
    color: var(--body-text-color-subdued, #555) !important;
    font-weight: 600 !important;
}

.sidebar-tabs .tab-nav button.selected {
    color: #00629B !important;
    border-bottom-color: #F58220 !important;
    background-color: rgba(0, 98, 155, 0.04) !important;
}

.sidebar-tabs .tab-nav button:hover {
    color: #00629B !important;
}
"""


def _user_requested_sources(message: str) -> bool:
    prompt = message.lower()
    source_triggers = [
        "source",
        "sources",
        "citation",
        "citations",
        "reference",
        "references",
        "where did you get",
    ]
    return any(trigger in prompt for trigger in source_triggers)


def _history_to_text(history: list[dict[str, str]] | list[list[str]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    if isinstance(history[0], dict):
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    for pair in history:
        if len(pair) != 2:
            continue
        lines.append(f"user: {pair[0]}")
        lines.append(f"assistant: {pair[1]}")
    return "\n".join(lines)


def _normalize_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                normalized.append({"role": role, "content": content})
            continue

        if isinstance(item, list | tuple) and len(item) == 2:
            user_text = str(item[0]).strip()
            assistant_text = str(item[1]).strip()
            if user_text:
                normalized.append({"role": "user", "content": user_text})
            if assistant_text:
                normalized.append({"role": "assistant", "content": assistant_text})

    return normalized


def create_demo() -> gr.Blocks:
    settings = Settings.from_env()
    agent = RAGAgent(settings)
    session_histories: dict[str, list[dict[str, str]]] = {}

    def _session_key(request: gr.Request | None) -> str:
        if request is None:
            return "default"

        session_hash = getattr(request, "session_hash", None)
        if isinstance(session_hash, str) and session_hash.strip():
            return f"session:{session_hash.strip()}"

        header_session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("x-client-id")
            or ""
        ).strip()
        if header_session_id:
            return f"header:{header_session_id}"

        forwarded_for = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded_for:
            return f"ip:{forwarded_for}"

        if request.client and request.client.host:
            return f"ip:{request.client.host}"

        return "default"

    def chat_fn(
        message: str,
        history: list[dict[str, str]] | list[list[str]] | None = None,
    ) -> str:
        history_text = _history_to_text(history)
        answer, sources, _, _ = agent.answer(message, history_text=history_text)
        if not sources or not _user_requested_sources(message):
            return answer
        source_text = "\n".join(f"- {source}" for source in sources[:8])
        return f"{answer}\n\nSources:\n{source_text}"

    def chat_api_fn(message: str, request: gr.Request | None = None) -> str:
        if not (message or "").strip():
            return ""

        key = _session_key(request)
        history_items = session_histories.get(key, [])
        answer = chat_fn(message, history=history_items)
        updated_history = [
            *history_items,
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        session_histories[key] = updated_history[-30:]
        return answer

    def chat_turn_api_fn(message: str, history_json: str) -> tuple[str, str]:
        parsed_history: Any
        try:
            parsed_history = json.loads(history_json or "[]")
        except Exception:
            parsed_history = []

        history_items = _normalize_history(parsed_history)
        answer = chat_fn(message, history=history_items)
        updated_history = [
            *history_items,
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return answer, json.dumps(updated_history, ensure_ascii=False)

    def upload_fn(files: list[Any] | None) -> str:
        if not files:
            return "No files selected."
        try:
            paths = [str(Path(file.name).resolve()) for file in files]
            result = ingest_files(settings, paths, origin="upload")
            return (
                f"Indexed: {result['indexed']} | Skipped: {result['skipped']} | "
                f"Deleted old chunks: {result['deleted']}"
            )
        except Exception as error:
            return f"Upload indexing failed: {error}"

    def sync_fn() -> str:
        try:
            result = sync_local_docs(settings)
            return (
                f"Synced files: {result.get('total_files', 0)} | Indexed: {result['indexed']} | "
                f"Skipped: {result['skipped']} | Deleted chunks: {result['deleted']}"
            )
        except Exception as error:
            return f"Local sync failed: {error}"

    def status_fn() -> str:
        status = agent.status()
        lines = [f"- {key}: {value}" for key, value in status.items()]
        return "\n".join(lines)

    def kb_stats_fn() -> str:
        stats = get_kb_stats(settings)
        lines = [
            f"**Total Sources:** {stats['total_sources']}",
            f"**Total Chunks:** {stats['total_chunks']}",
            f"**Last Sync:** {stats['last_sync']}",
            "",
            "**Origins:**",
            f"- Local: {stats['origins'].get('local', 0)}",
            f"- Upload: {stats['origins'].get('upload', 0)}",
            f"- Website: {stats['origins'].get('website', 0)}",
            "",
            "**Sources:**"
        ]
        for src in stats['source_names']:
            lines.append(f"- {src}")
        return "\n".join(lines)

    def analytics_fn() -> tuple[str, str, str]:
        if not settings.langsmith_tracing:
            msg = "LangSmith tracing is disabled. Enable LANGSMITH_TRACING in your environment to see analytics."
            return msg, msg, msg
        
        # Feedback summary
        fb = get_feedback_summary(settings)
        fb_str = f"**Total Feedback:** {fb['total']} (👍 {fb['up']} | 👎 {fb['down']})"
        
        # Latency stats
        lat = get_latency_stats(settings)
        lat_str = f"**Average Latency (last 7 days):** {lat['avg_ms']} ms"
        
        # Recent runs
        runs = get_recent_runs(settings, limit=10)
        if not runs:
            runs_str = "No recent runs found."
        else:
            runs_str = "| Time | Question | Latency (ms) | Feedback |\n|---|---|---|---|\n"
            for r in runs:
                runs_str += f"| {r['time']} | {r['question']} | {r['latency_ms']} | {r['feedback']} |\n"
                
        return fb_str, lat_str, runs_str

    def website_fn(url: str, max_pages: int) -> str:
        target_url = (url or settings.website_default_url).strip()
        if not target_url:
            return "Website URL is required."
        try:
            pages_limit = max(1, int(max_pages or settings.website_max_pages))
            result = ingest_website(settings, start_url=target_url, max_pages=pages_limit)
            return (
                f"Crawled pages: {result['total_pages']} | Indexed: {result['indexed']} | "
                f"Skipped: {result['skipped']} | Deleted old chunks: {result['deleted']}"
            )
        except Exception as error:
            return f"Website crawl failed: {error}"

    def text_ingest_fn(text: str, source_name: str) -> str:
        if not text.strip():
            return "Text cannot be empty."
        target_source = source_name.strip() or "Raw Text Input"
        try:
            result = ingest_text(settings, text, target_source)
            return (
                f"Indexed chunks: {result['indexed']} | "
                f"Skipped: {result['skipped']} | Deleted old chunks: {result['deleted']}"
            )
        except Exception as error:
            return f"Text ingestion failed: {error}"

    with gr.Blocks(
        title="IEEE AI RAG Chatbot",
    ) as demo:
        with gr.Sidebar(label="Control Center", open=True):
            gr.Markdown("## ⚙️ Control Center")
            
            with gr.Tabs(elem_classes=["sidebar-tabs"]):
                with gr.Tab("📥 Ingestion"):
                    with gr.Group():
                        gr.Markdown("### 📄 Upload Files")
                        uploader = gr.Files(
                            label="Upload PDF/PPT/DOC/MD/HTML files",
                            file_count="multiple",
                            file_types=[".pdf", ".ppt", ".pptx", ".docx", ".doc", ".md", ".html"],
                        )
                        upload_button = gr.Button("Upload + Index", variant="primary")
                        upload_output = gr.Textbox(label="Upload Status", interactive=False)
                        upload_button.click(fn=upload_fn, inputs=[uploader], outputs=[upload_output])

                    with gr.Group():
                        gr.Markdown("### 📝 Raw Text")
                        text_input = gr.Textbox(
                            label="Text Content",
                            lines=5,
                            placeholder="Paste text here to index into the knowledge base...",
                        )
                        text_source = gr.Textbox(
                            label="Source Name (Optional)",
                            placeholder="e.g., meeting_notes_2026.txt",
                        )
                        text_button = gr.Button("Ingest Text", variant="secondary")
                        text_output = gr.Textbox(label="Text Ingestion Status", interactive=False)
                        text_button.click(
                            fn=text_ingest_fn,
                            inputs=[text_input, text_source],
                            outputs=[text_output],
                        )

                    with gr.Group():
                        gr.Markdown("### 🌐 Website Crawl")
                        website_url = gr.Textbox(
                            label="Website URL",
                            value=settings.website_default_url,
                        )
                        website_max_pages = gr.Number(
                            label="Max pages to crawl",
                            value=settings.website_max_pages,
                            precision=0,
                        )
                        website_button = gr.Button("Crawl Website + Index", variant="secondary")
                        website_output = gr.Textbox(label="Website Crawl Status", interactive=False)
                        website_button.click(
                            fn=website_fn,
                            inputs=[website_url, website_max_pages],
                            outputs=[website_output],
                        )

                    with gr.Group():
                        gr.Markdown("### 📂 Local Sync")
                        sync_button = gr.Button("Sync docs/pdf, docs/ppt, and docs/doc", variant="secondary")
                        sync_output = gr.Textbox(label="Local Sync Status", interactive=False)
                        sync_button.click(fn=sync_fn, inputs=None, outputs=[sync_output])

                with gr.Tab("📊 Status"):
                    status_output = gr.Markdown("Click refresh to load status.")
                    status_button = gr.Button("Refresh Status", variant="primary")
                    status_button.click(fn=status_fn, inputs=None, outputs=[status_output])

                with gr.Tab("📚 KB Info"):
                    kb_output = gr.Markdown("Click refresh to load KB Stats.")
                    kb_button = gr.Button("Refresh KB Stats", variant="primary")
                    kb_button.click(fn=kb_stats_fn, inputs=None, outputs=[kb_output])

                with gr.Tab("📈 Analytics"):
                    with gr.Row():
                        analytics_fb = gr.Markdown("Loading feedback...")
                        analytics_lat = gr.Markdown("Loading latency...")
                    analytics_runs = gr.Markdown("Loading runs...")
                    analytics_btn = gr.Button("Refresh Analytics", variant="primary")
                    analytics_btn.click(
                        fn=analytics_fn, 
                        inputs=None, 
                        outputs=[analytics_fb, analytics_lat, analytics_runs]
                    )

        # Main Chat Area
        with gr.Column():
            welcome_container = gr.Column(visible=True)
            with welcome_container:
                gr.Markdown("IEEE AI Chatbot", elem_classes=["welcome-title"])
                gr.Markdown("Ask me anything about the IEEE Beni Suef Student Branch, societies, chapters, or events!", elem_classes=["welcome-subtitle"])
                
                with gr.Row():
                    card1 = gr.Button("IEEE Beni Suef", elem_classes=["suggestion-card"])
                    card2 = gr.Button("Computer Society (CS)", elem_classes=["suggestion-card"])
                    card3 = gr.Button("Computational Intelligence Society (CIS)", elem_classes=["suggestion-card"])
                    card4 = gr.Button("AESH, T.I.M.E, RYM events", elem_classes=["suggestion-card"])

            chatbot = gr.Chatbot(
                label="Conversation History",
                elem_classes=["chatbot-container"],
            )
            
            with gr.Row(elem_classes=["input-row"]):
                msg_box = gr.Textbox(
                    placeholder="Ask a question...",
                    show_label=False,
                    elem_classes=["input-textbox"],
                    container=False,
                )
                submit_btn = gr.Button("➔", elem_classes=["send-btn"])

            with gr.Row(elem_classes=["feedback-row"]):
                upvote_btn = gr.Button("👍 Useful", elem_classes=["feedback-btn"])
                downvote_btn = gr.Button("👎 Unhelpful", elem_classes=["feedback-btn"])
                clear_btn = gr.Button("🗑️ Clear Chat", elem_classes=["clear-btn"])

            feedback_status = gr.Markdown()
            current_run_id = gr.State("")

            def user(user_message, history):
                return "", history + [{"role": "user", "content": user_message}], gr.update(visible=False)

            def bot(history):
                user_message = history[-1]["content"]
                history_text = _history_to_text(history[:-1])
                
                history.append({"role": "assistant", "content": ""})
                
                run_id = ""
                sources = []
                confidence = ""
                for chunk, src, r_id, conf in agent.answer_stream(user_message, history_text=history_text):
                    run_id = r_id
                    sources = src
                    confidence = conf
                    history[-1]["content"] += chunk
                    yield history, run_id

                if sources and _user_requested_sources(user_message):
                    source_text = "\n".join(f"- {source}" for source in sources[:8])
                    history[-1]["content"] += f"\n\n**Sources:**\n{source_text}"
                
                if confidence:
                    badge = "🟢 High" if confidence == "High" else "🟡 Medium" if confidence == "Medium" else "🔴 Low"
                    if confidence == "Web Search":
                        badge = "🌐 Web Search"
                    elif confidence == "None":
                        badge = "⚪ None"
                    history[-1]["content"] += f"\n\n*(Retrieval Confidence: {badge})*"
                    
                yield history, run_id

            # Setup triggers
            msg_box.submit(
                fn=user,
                inputs=[msg_box, chatbot],
                outputs=[msg_box, chatbot, welcome_container],
                queue=False,
            ).then(
                fn=bot,
                inputs=[chatbot],
                outputs=[chatbot, current_run_id],
            )
            
            submit_btn.click(
                fn=user,
                inputs=[msg_box, chatbot],
                outputs=[msg_box, chatbot, welcome_container],
                queue=False,
            ).then(
                fn=bot,
                inputs=[chatbot],
                outputs=[chatbot, current_run_id],
            )

            # Suggestion cards logic
            def click_card(card_val, history):
                return history + [{"role": "user", "content": card_val}], gr.update(visible=False)

            for card in [card1, card2, card3, card4]:
                card.click(
                    fn=click_card,
                    inputs=[card, chatbot],
                    outputs=[chatbot, welcome_container],
                    queue=False,
                ).then(
                    fn=bot,
                    inputs=[chatbot],
                    outputs=[chatbot, current_run_id],
                )

            # Clear button logic
            clear_btn.click(
                fn=lambda: ([], gr.update(visible=True)),
                inputs=None,
                outputs=[chatbot, welcome_container],
                queue=False,
            )

            # Feedback mechanics
            def handle_feedback(run_id, score):
                if not run_id:
                    return "No response to evaluate yet."
                success = agent.submit_feedback(run_id, score=score)
                if success:
                    return "Feedback sent ✓ Thank you!"
                return "Failed to send feedback. Please check LangSmith tracing is enabled."

            upvote_btn.click(lambda r: handle_feedback(r, 1.0), inputs=[current_run_id], outputs=[feedback_status])
            downvote_btn.click(lambda r: handle_feedback(r, 0.0), inputs=[current_run_id], outputs=[feedback_status])

        # API Endpoints
        api_message = gr.Textbox(visible=False)
        api_output = gr.Textbox(visible=False)
        api_trigger = gr.Button(visible=False)
        api_trigger.click(
            fn=chat_api_fn,
            inputs=[api_message],
            outputs=[api_output],
            api_name="chat_once",
            queue=False,
        )

        api_turn_message = gr.Textbox(visible=False)
        api_turn_history = gr.Textbox(visible=False)
        api_turn_reply = gr.Textbox(visible=False)
        api_turn_history_out = gr.Textbox(visible=False)
        api_turn_trigger = gr.Button(visible=False)
        api_turn_trigger.click(
            fn=chat_turn_api_fn,
            inputs=[api_turn_message, api_turn_history],
            outputs=[api_turn_reply, api_turn_history_out],
            api_name="chat_turn",
            queue=False,
        )

    return demo
