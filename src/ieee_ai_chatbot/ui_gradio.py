from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from .chat import RAGAgent
from .config import Settings
from .ingest import ingest_files, ingest_website, sync_local_docs


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


def create_demo() -> gr.Blocks:
    settings = Settings.from_env()
    agent = RAGAgent(settings)

    def chat_fn(message: str, history: list[dict[str, str]] | list[list[str]] | None) -> str:
        history_text = _history_to_text(history)
        answer, sources = agent.answer(message, history_text=history_text)
        if not sources or not _user_requested_sources(message):
            return answer
        source_text = "\n".join(f"- {source}" for source in sources[:8])
        return f"{answer}\n\nSources:\n{source_text}"

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

    with gr.Blocks(title="IEEE AI RAG Chatbot") as demo:
        gr.Markdown("# IEEE AI RAG Chatbot")

        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=chat_fn,
                title="Ask IEEE knowledge questions",
            )

        with gr.Tab("Ingestion"):
            uploader = gr.Files(
                label="Upload PDF/PPT/DOC files",
                file_count="multiple",
                file_types=[".pdf", ".ppt", ".pptx", ".docx", ".doc"],
            )
            upload_output = gr.Textbox(label="Upload Status")
            sync_output = gr.Textbox(label="Local Sync Status")
            website_url = gr.Textbox(
                label="Website URL",
                value=settings.website_default_url,
            )
            website_max_pages = gr.Number(
                label="Website max pages",
                value=settings.website_max_pages,
                precision=0,
            )
            website_output = gr.Textbox(label="Website Crawl Status")
            upload_button = gr.Button("Upload + Index")
            sync_button = gr.Button("Sync docs/pdf, docs/ppt, and docs/doc")
            website_button = gr.Button("Crawl Website + Index")
            upload_button.click(fn=upload_fn, inputs=[uploader], outputs=[upload_output])
            sync_button.click(fn=sync_fn, inputs=None, outputs=[sync_output])
            website_button.click(
                fn=website_fn,
                inputs=[website_url, website_max_pages],
                outputs=[website_output],
            )

        with gr.Tab("Status"):
            status_output = gr.Markdown()
            status_button = gr.Button("Refresh status")
            status_button.click(fn=status_fn, inputs=None, outputs=[status_output])

    return demo
