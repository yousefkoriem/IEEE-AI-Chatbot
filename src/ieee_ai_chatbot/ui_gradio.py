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


# ─────────────────────────────────────────────────────────────────────
# MODERN GLASSMORPHIC CSS — IEEE Blue / CS Orange / Gemini Violet
# ─────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Global (Forcing beautiful DARK mode) ── */
:root, .dark, .gradio-container, .gradio-container.dark {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;

    /* Brand Colors */
    --ieee-blue: #0088D6; /* Brightened for dark mode */
    --ieee-blue-dark: #00629B;
    --cs-orange: #F89B48; /* Brightened for dark mode */
    --cs-orange-dark: #F58220;
    --gemini-violet: #A855F7; /* Brightened */
    --gemini-violet-dark: #8E2DE2;

    /* Base theme colors */
    --body-background-fill: #090E17 !important;
    --body-text-color: #F8FAFC !important;
    --body-text-color-subdued: #94A3B8 !important;
    --block-title-text-color: #F8FAFC !important;
    --block-label-text-color: #E2E8F0 !important;
    --block-info-text-color: #94A3B8 !important;

    /* Panels and Blocks */
    --background-fill-primary: rgba(30, 41, 59, 0.45) !important;
    --background-fill-secondary: rgba(15, 23, 42, 0.6) !important;
    --block-background-fill: rgba(30, 41, 59, 0.45) !important;
    
    /* Borders */
    --border-color-primary: rgba(255, 255, 255, 0.1) !important;
    --border-color-secondary: rgba(255, 255, 255, 0.05) !important;
    --block-border-color: rgba(255, 255, 255, 0.1) !important;
    
    /* Inputs */
    --input-background-fill: rgba(15, 23, 42, 0.7) !important;
    --input-background-fill-focus: rgba(30, 41, 59, 0.9) !important;
    --input-border-color: rgba(255, 255, 255, 0.15) !important;
    --input-border-color-focus: var(--ieee-blue) !important;
    --input-text-color: #F8FAFC !important;
    --input-text-color-focus: #FFFFFF !important;
    --input-placeholder-color: #64748B !important;
    
    /* Buttons */
    --button-primary-background-fill: linear-gradient(135deg, var(--ieee-blue-dark) 0%, var(--ieee-blue) 100%) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, var(--cs-orange-dark) 0%, var(--cs-orange) 100%) !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: rgba(255, 255, 255, 0.05) !important;
    --button-secondary-background-fill-hover: rgba(255, 255, 255, 0.1) !important;
    --button-secondary-text-color: #F8FAFC !important;
}

/* ── Overrides to Ensure Text is Visible in Dark Theme ── */
.gradio-container,
.gradio-container *,
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .tab-nav button,
.gradio-container .tabitem,
.gradio-container .form,
.gradio-container .form *,
.gradio-container .block,
.gradio-container .block *,
.gradio-container label,
.gradio-container span {
    color: var(--body-text-color) !important;
}

.gradio-container .header-logo,
.gradio-container .header-logo *,
.gradio-container .primary,
.gradio-container .primary *,
.gradio-container .send-btn,
.gradio-container .send-btn * {
    color: #ffffff !important;
}

/* Ensure SVG icons inside buttons remain visible */
.gradio-container button svg {
    fill: currentColor !important;
    color: inherit !important;
}

/* ── Page Background ───────────────────────────────────────────────── */
.gradio-container {
    background:
        radial-gradient(ellipse at 15% 15%, rgba(0, 136, 214, 0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 85%, rgba(168, 85, 247, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(248, 155, 72, 0.05) 0%, transparent 60%),
        var(--body-background-fill) !important;
    min-height: 100vh !important;
    color: var(--body-text-color) !important;
}

/* ── Glass Panel Base ──────────────────────────────────────────────── */
.glass-card {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    padding: 12px 20px !important;
}

/* ── HEADER ────────────────────────────────────────────────────────── */
.header-bar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    padding: 14px 24px !important;
    margin-bottom: 20px !important;
}

.header-logo {
    width: 44px !important;
    height: 44px !important;
    background: linear-gradient(135deg, var(--cs-orange), var(--cs-orange-dark)) !important;
    border-radius: 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    flex-shrink: 0 !important;
}

.header-info {
    flex: 1 !important;
}

.header-title, .header-title * {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: var(--cs-orange) !important;
    margin-bottom: 2px !important;
}

.header-subtitle, .header-subtitle * {
    font-size: 0.82rem !important;
    color: var(--ieee-blue) !important;
}

.header-badge {
    font-size: 0.75rem !important;
    color: var(--cs-orange) !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}
/* ── Sidebar Container ─────────────────────────────────────────────── */
.sidebar-container {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 18px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    height: 100% !important;
    max-width: 400px !important;
}

.control-header {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--cs-orange) !important;
    margin-bottom: 14px !important;
    letter-spacing: -0.3px !important;
}

/* ── TAB NAVIGATION ────────────────────────────────────────────────── */
.sidebar-tabs .tab-nav {
    display: flex !important;
    gap: 3px !important;
    background: rgba(0, 0, 0, 0.2) !important;
    border-radius: 12px !important;
    padding: 3px !important;
    margin-bottom: 16px !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
}

.sidebar-tabs .tab-nav button {
    font-size: 0.78rem !important;
    padding: 7px 10px !important;
    border-radius: 8px !important;
    border: none !important;
    background: transparent !important;
    color: var(--body-text-color-subdued) !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
    flex: 1 0 auto !important;
    text-align: center !important;
    white-space: nowrap !important;
    min-width: 70px !important;
}

.sidebar-tabs .tab-nav button.selected {
    color: #ffffff !important;
    background: linear-gradient(135deg, var(--ieee-blue-dark) 0%, var(--gemini-violet-dark) 100%) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

.sidebar-tabs .tab-nav button:hover:not(.selected) {
    color: var(--ieee-blue) !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.sidebar-tabs .tabitem {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Ingestion Groups ──────────────────────────────────────────────── */
.ingest-group {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    margin-bottom: 12px !important;
    transition: all 0.2s ease !important;
}

.ingest-group:hover {
    background: rgba(15, 23, 42, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
}

.ingest-group-title {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: var(--ieee-blue) !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── File Upload Override ──────────────────────────────────────────── */
.upload-area {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 2px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.upload-area:hover {
    background: rgba(30, 41, 59, 0.7) !important;
    border-color: var(--ieee-blue) !important;
}

/* ── Buttons ───────────────────────────────────────────────────────── */
.secondary-btn {
    background: linear-gradient(135deg, var(--cs-orange-dark) 0%, var(--cs-orange) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 7px 14px !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

.secondary-btn:hover {
    box-shadow: 0 4px 16px rgba(245, 130, 32, 0.4) !important;
    transform: translateY(-1px) !important;
}

.secondary-btn:active {
    transform: scale(0.97) !important;
}

/* ── Output Boxes ──────────────────────────────────────────────────── */
.output-box {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.80rem !important;
    color: #E2E8F0 !important;
    padding: 8px 12px !important;
}

/* ── Welcome Section ───────────────────────────────────────────────── */
.welcome-title, .welcome-title * {
    font-size: 1.8rem !important;
    font-weight: 900 !important;
    color: var(--gemini-violet) !important;
    text-align: center !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.8px !important;
    line-height: 1.25 !important;
}

.welcome-subtitle {
    font-size: 1.05rem !important;
    color: var(--body-text-color-subdued) !important;
    text-align: center !important;
    margin-bottom: 1.5rem !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
}

/* ── Suggestion Cards ──────────────────────────────────────────────── */
.suggestion-card {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.suggestion-card, .suggestion-card button {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 14px 10px !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--body-text-color) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    min-height: 64px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    backdrop-filter: blur(8px) !important;
}

.suggestion-card:hover, .suggestion-card button:hover {
    border-color: var(--cs-orange) !important;
    background: rgba(30, 41, 59, 0.8) !important;
    color: var(--cs-orange) !important;
}

/* ── Input Row ─────────────────────────────────────────────────────── */
.input-row {
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 32px !important;
    padding: 6px 10px 6px 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    margin-top: 10px !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(8px) !important;
}

.input-row:focus-within {
    border-color: var(--ieee-blue) !important;
    background: rgba(30, 41, 59, 0.9) !important;
    box-shadow: 0 0 0 1px var(--ieee-blue), 0 4px 16px rgba(0, 0, 0, 0.4) !important;
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
    font-size: 0.95rem !important;
    color: var(--body-text-color) !important;
}

/* ── Send Button ───────────────────────────────────────────────────── */
.send-btn {
    background: linear-gradient(135deg, var(--ieee-blue-dark) 0%, var(--ieee-blue) 100%) !important;
    color: white !important;
    border-radius: 50% !important;
    border: none !important;
    min-width: 42px !important;
    max-width: 42px !important;
    height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.15s ease !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    margin-left: 2px !important;
}

.send-btn:hover {
    background: linear-gradient(135deg, var(--cs-orange-dark) 0%, var(--cs-orange) 100%) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
    transform: scale(1.08) !important;
}

.send-btn:active {
    transform: scale(0.95) !important;
}

/* ── Feedback Row ──────────────────────────────────────────────────── */
.feedback-row {
    justify-content: center !important;
    gap: 10px !important;
    margin-top: 8px !important;
}

.feedback-btn {
    border-radius: 24px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: var(--body-text-color-subdued) !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(8px) !important;
}

.feedback-btn:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: var(--ieee-blue) !important;
    color: var(--ieee-blue) !important;
    transform: translateY(-1px) !important;
}

.clear-btn {
    border-radius: 24px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    color: #FCA5A5 !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(8px) !important;
}

.clear-btn:hover {
    background: rgba(239, 68, 68, 0.2) !important;
    color: #FECACA !important;
}

/* ── Status / KB / Analytics ───────────────────────────────────────── */
.prose-custom {
    color: var(--body-text-color-subdued) !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

.prose-custom strong {
    color: var(--ieee-blue) !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────── */
.gradio-container ::-webkit-scrollbar {
    width: 5px;
}

.gradio-container ::-webkit-scrollbar-track {
    background: transparent;
}

.gradio-container ::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}

.gradio-container ::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* ── Gradio Overrides ──────────────────────────────────────────────── */
.gradio-container .label-text,
.gradio-container .label-text span {
    color: var(--body-text-color) !important;
    font-weight: 500 !important;
}

.gradio-container input,
.gradio-container textarea {
    background: var(--input-background-fill) !important;
    border: 1px solid var(--input-border-color) !important;
    border-radius: 8px !important;
    color: var(--body-text-color) !important;
    font-family: 'Outfit', sans-serif !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: var(--input-placeholder-color) !important;
}

/* Tabs inside sidebar — ensure all 4 tabs are visible */
.sidebar-tabs {
    width: 100% !important;
}
"""

THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)


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

        fb = get_feedback_summary(settings)
        fb_str = f"**Total Feedback:** {fb['total']} (👍 {fb['up']} | 👎 {fb['down']})"

        lat = get_latency_stats(settings)
        lat_str = f"**Average Latency (last 7 days):** {lat['avg_ms']} ms"

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
        title="IEEE AI Chatbot",
    ) as demo:

        # ── HEADER BAR ────────────────────────────────────────────────
        gr.HTML("""
        <div class="glass-card header-bar">
            <div class="header-logo">I</div>
            <div class="header-info">
                <div class="header-title">IEEE AI Chatbot</div>
                <div class="header-subtitle">IEEE Beni Suef Student Branch</div>
            </div>
            <div class="header-badge">⚡ AI Powered</div>
        </div>
        """)

        # ── MAIN LAYOUT: TOP-LEVEL TABS ──────────────────────────────
        with gr.Tabs(elem_classes=["top-tabs"]):

            # ── TOP TAB: CHAT ──────────────────────────
            with gr.Tab("💬 Chat"):
                with gr.Row():

                    # ── LEFT: SIDEBAR (Control Center) ──────────────────────
                    with gr.Column(scale=1, elem_classes=["sidebar-container"], min_width=320):
                        gr.Markdown("## ⚙️ Control Center", elem_classes=["control-header"])

                        gr.Markdown("### 📊 Status", elem_classes=["prose-custom"])
                        status_output = gr.Markdown(
                            "Click refresh to load agent status.",
                            elem_classes=["prose-custom"],
                        )
                        status_button = gr.Button("Refresh Status", variant="primary")
                        status_button.click(fn=status_fn, inputs=None, outputs=[status_output])

                        gr.HTML("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>")

                        gr.Markdown("### 📚 Knowledge Base", elem_classes=["prose-custom"])
                        kb_output = gr.Markdown(
                            "Click refresh to load knowledge base stats.",
                            elem_classes=["prose-custom"],
                        )
                        kb_button = gr.Button("Refresh KB Stats", variant="primary")
                        kb_button.click(fn=kb_stats_fn, inputs=None, outputs=[kb_output])

                    # ── RIGHT: MAIN CHAT AREA ───────────────────────────────
                    with gr.Column(scale=3, min_width=500):

                        # Welcome Container
                        welcome_container = gr.Column(visible=True)
                        with welcome_container:
                            gr.Markdown(
                                "Ask anything about IEEE Beni Suef",
                                elem_classes=["welcome-title"],
                            )
                            gr.Markdown(
                                "Societies, chapters, events — I'll search the knowledge base and give you accurate answers.",
                                elem_classes=["welcome-subtitle"],
                            )

                            with gr.Row():
                                card1 = gr.Button("IEEE Beni Suef", elem_classes=["suggestion-card"])
                                card2 = gr.Button("Computer Society", elem_classes=["suggestion-card"])
                                card3 = gr.Button("CIS Society", elem_classes=["suggestion-card"])
                                card4 = gr.Button("AESH & Events", elem_classes=["suggestion-card"])

                        # Chatbot
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            elem_classes=["chatbot-container"],
                            height=480,
                        )

                        # Input Row
                        with gr.Row(elem_classes=["input-row"]):
                            msg_box = gr.Textbox(
                                placeholder="Ask a question about IEEE Beni Suef...",
                                show_label=False,
                                elem_classes=["input-textbox"],
                                container=False,
                                scale=9,
                            )
                            submit_btn = gr.Button("➤", elem_classes=["send-btn"], scale=1)

                        # Feedback Row
                        with gr.Row(elem_classes=["feedback-row"]):
                            upvote_btn = gr.Button("👍 Useful", elem_classes=["feedback-btn"])
                            downvote_btn = gr.Button("👎 Unhelpful", elem_classes=["feedback-btn"])
                            clear_btn = gr.Button("🗑️ Clear", elem_classes=["clear-btn"])

                        feedback_status = gr.Markdown()
                        current_run_id = gr.State("")

                        # ── CHAT LOGIC ─────────────────────────────────────
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
                                badge = "High" if confidence == "High" else "Medium" if confidence == "Medium" else "Low"
                                if confidence == "Web Search":
                                    badge = "Web Search"
                                elif confidence == "None":
                                    badge = "None"
                                history[-1]["content"] += f"\n\n*(Retrieval Confidence: {badge})*"

                            yield history, run_id

                        # Triggers
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

                        # Suggestion cards
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

                        # Clear
                        clear_btn.click(
                            fn=lambda: ([], gr.update(visible=True)),
                            inputs=None,
                            outputs=[chatbot, welcome_container],
                            queue=False,
                        )

                        # Feedback
                        def handle_feedback(run_id, score):
                            if not run_id:
                                return "No response to evaluate yet."
                            success = agent.submit_feedback(run_id, score=score)
                            if success:
                                return "Feedback sent! Thank you."
                            return "Failed to send feedback. Check LangSmith tracing."

                        upvote_btn.click(
                            lambda r: handle_feedback(r, 1.0),
                            inputs=[current_run_id],
                            outputs=[feedback_status],
                        )
                        downvote_btn.click(
                            lambda r: handle_feedback(r, 0.0),
                            inputs=[current_run_id],
                            outputs=[feedback_status],
                        )

        # ── TOP TAB: INGEST ──────────────────────────
            with gr.Tab("📥 Ingest"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["ingest-group"]):
                            gr.Markdown('<span class="ingest-group-title">📄 Upload Files</span>')
                            uploader = gr.Files(
                                label="PDF, PPT, DOC, MD, HTML",
                                file_count="multiple",
                                file_types=[".pdf", ".ppt", ".pptx", ".docx", ".doc", ".md", ".html"],
                            )
                            upload_button = gr.Button("Upload + Index", variant="primary")
                            upload_output = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_classes=["output-box"],
                            )
                            upload_button.click(fn=upload_fn, inputs=[uploader], outputs=[upload_output])
                            
                        with gr.Group(elem_classes=["ingest-group"]):
                            gr.Markdown('<span class="ingest-group-title">📁 Local Sync</span>')
                            sync_button = gr.Button(
                                "Sync Local Docs",
                                elem_classes=["secondary-btn"],
                            )
                            sync_output = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_classes=["output-box"],
                            )
                            sync_button.click(fn=sync_fn, inputs=None, outputs=[sync_output])

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["ingest-group"]):
                            gr.Markdown('<span class="ingest-group-title">📝 Raw Text</span>')
                            text_input = gr.Textbox(
                                label="Text Content",
                                lines=3,
                                placeholder="Paste text to index...",
                            )
                            text_source = gr.Textbox(
                                label="Source Name",
                                placeholder="e.g., notes_2026.txt",
                            )
                            text_button = gr.Button(
                                "Ingest Text",
                                elem_classes=["secondary-btn"],
                            )
                            text_output = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_classes=["output-box"],
                            )
                            text_button.click(
                                fn=text_ingest_fn,
                                inputs=[text_input, text_source],
                                outputs=[text_output],
                            )
                            
                        with gr.Group(elem_classes=["ingest-group"]):
                            gr.Markdown('<span class="ingest-group-title">🌐 Website Crawl</span>')
                            website_url = gr.Textbox(
                                label="URL",
                                value=settings.website_default_url,
                            )
                            website_max_pages = gr.Number(
                                label="Max pages",
                                value=settings.website_max_pages,
                                precision=0,
                            )
                            website_button = gr.Button(
                                "Crawl + Index",
                                elem_classes=["secondary-btn"],
                            )
                            website_output = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_classes=["output-box"],
                            )
                            website_button.click(
                                fn=website_fn,
                                inputs=[website_url, website_max_pages],
                                outputs=[website_output],
                            )

            # ── TOP TAB: ANALYTICS ──────────────────────────
            with gr.Tab("📈 Analytics"):
                with gr.Column(elem_classes=["glass-card"]):
                    with gr.Row():
                        analytics_fb = gr.Markdown(
                            "Loading feedback...",
                            elem_classes=["prose-custom"],
                        )
                        analytics_lat = gr.Markdown(
                            "Loading latency...",
                            elem_classes=["prose-custom"],
                        )
                    analytics_runs = gr.Markdown(
                        "Loading runs...",
                        elem_classes=["prose-custom"],
                    )
                    analytics_btn = gr.Button("Refresh Analytics", variant="primary")
                    analytics_btn.click(
                        fn=analytics_fn,
                        inputs=None,
                        outputs=[analytics_fb, analytics_lat, analytics_runs],
                    )

        # ── API ENDPOINTS ─────────────────────────────────────────────
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
