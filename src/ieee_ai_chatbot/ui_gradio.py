from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from .chat import RAGAgent
from .config import Settings
from .ingest import ingest_files, sync_local_docs


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
        if not sources:
            return answer
        source_text = "\n".join(f"- {source}" for source in sources[:8])
        return f"{answer}\n\nSources:\n{source_text}"

    def upload_fn(files: list[Any] | None) -> str:
        if not files:
            return "No files selected."
        paths = [str(Path(file.name).resolve()) for file in files]
        result = ingest_files(settings, paths, origin="upload")
        return (
            f"Indexed: {result['indexed']} | Skipped: {result['skipped']} | "
            f"Deleted old chunks: {result['deleted']}"
        )

    def sync_fn() -> str:
        result = sync_local_docs(settings)
        return (
            f"Synced files: {result.get('total_files', 0)} | Indexed: {result['indexed']} | "
            f"Skipped: {result['skipped']} | Deleted chunks: {result['deleted']}"
        )

    def status_fn() -> str:
        status = agent.status()
        lines = [f"- {key}: {value}" for key, value in status.items()]
        return "\n".join(lines)

    with gr.Blocks(title="IEEE AI RAG Chatbot") as demo:
        gr.Markdown("# IEEE AI RAG Chatbot")

        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=chat_fn,
                title="Ask IEEE knowledge questions",
            )

        with gr.Tab("Ingestion"):
            uploader = gr.Files(
                label="Upload PDF/PPT files",
                file_count="multiple",
                file_types=[".pdf", ".ppt", ".pptx"],
            )
            upload_output = gr.Textbox(label="Upload Status")
            sync_output = gr.Textbox(label="Local Sync Status")
            upload_button = gr.Button("Upload + Index")
            sync_button = gr.Button("Sync docs/pdf and docs/ppt")
            upload_button.click(fn=upload_fn, inputs=[uploader], outputs=[upload_output])
            sync_button.click(fn=sync_fn, inputs=None, outputs=[sync_output])

        with gr.Tab("Status"):
            status_output = gr.Markdown()
            status_button = gr.Button("Refresh status")
            status_button.click(fn=status_fn, inputs=None, outputs=[status_output])

    return demo
