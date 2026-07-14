from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
import gradio as gr

from .chat import RAGAgent
from .config import Settings
from .ingest import ingest_files, ingest_website, sync_local_docs, ingest_text, ingest_url_list
from .stats import get_kb_stats, list_all_sources, delete_source, search_chunks, get_source_chunks
from .analytics import (
    get_recent_runs, get_feedback_summary, get_latency_stats,
    get_top_queries, get_latency_timeseries, get_feedback_timeseries,
    export_runs_csv,
)
from .sharing import ShareManager

from .chat_history import ChatHistoryManager
from .rate_limiter import RateLimiter


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

.control-header, .control-header * {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--ieee-blue) !important;
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
    background: linear-gradient(135deg, var(--ieee-blue) 0%, var(--gemini-violet) 50%, var(--cs-orange) 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
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

/* ── Conversation Selector ──────────────────────────────────────────── */
.conv-selector label {
    font-size: 0.82rem !important;
    color: var(--body-text-color-subdued) !important;
}

.conv-selector .radio-item {
    padding: 8px 12px !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    transition: all 0.15s ease !important;
    font-size: 0.85rem !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid transparent !important;
}

.conv-selector .radio-item:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.1) !important;
}

.conv-selector .radio-item.selected {
    background: rgba(0,136,214,0.15) !important;
    border-color: var(--ieee-blue) !important;
}

/* ── Action Buttons ─────────────────────────────────────────────────── */
.action-btn {
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: var(--body-text-color-subdued) !important;
    transition: all 0.15s ease !important;
}

.action-btn:hover {
    background: rgba(255,255,255,0.12) !important;
    border-color: var(--ieee-blue) !important;
    color: var(--ieee-blue) !important;
}

/* ── Confidence Badge in chat ───────────────────────────────────────── */
.conf-high { color: #22C55E; font-weight: 600; }
.conf-medium { color: #EAB308; font-weight: 600; }
.conf-low { color: #EF4444; font-weight: 600; }
.conf-web { color: var(--ieee-blue); font-weight: 600; }

/* ── Smaller buttons in feedback row ────────────────────────────────── */
.feedback-row .gr-button {
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
}

/* ── Toast Notifications ────────────────────────────────────────────── */
#toast-container {
    position: fixed !important;
    top: 16px !important;
    right: 16px !important;
    z-index: 9999 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    pointer-events: none !important;
}

.toast {
    padding: 12px 20px !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #fff !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
    pointer-events: auto !important;
    animation: toastIn 0.3s ease, toastOut 0.3s ease 2.7s forwards;
    max-width: 360px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

.toast-success { background: rgba(34,197,94,0.9) !important; }
.toast-error { background: rgba(239,68,68,0.9) !important; }
.toast-info { background: rgba(0,136,214,0.9) !important; }
.toast-warning { background: rgba(245,158,11,0.9) !important; }

@keyframes toastIn {
    from { opacity: 0; transform: translateX(40px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes toastOut {
    from { opacity: 1; transform: translateX(0); }
    to { opacity: 0; transform: translateX(40px); }
}

/* ── Pill-style Radio Group ──────────────────────────────────────────── */
.pill-group .radio-group {
    display: flex !important;
    flex-direction: row !important;
    gap: 4px !important;
}
.pill-group label {
    border-radius: 20px !important;
    padding: 4px 16px !important;
    font-size: 0.82rem !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    transition: all 0.15s ease !important;
}
.pill-group label.selected {
    background: var(--ieee-blue) !important;
    border-color: var(--ieee-blue) !important;
    color: #fff !important;
}

/* ── Loading Skeleton ────────────────────────────────────────────────── */
.skeleton-line {
    height: 14px;
    border-radius: 6px;
    background: linear-gradient(90deg, rgba(255,255,255,0.04) 25%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.04) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    margin-bottom: 8px;
}
.skeleton-line:nth-child(2) { width: 85%; }
.skeleton-line:nth-child(3) { width: 70%; }
.skeleton-line:last-child { width: 55%; }

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Responsive ──────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .gradio-container .tab-nav { font-size: 0.78rem !important; }
    .gradio-container .prose-custom { font-size: 0.82rem !important; }
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
    chat_history_mgr = ChatHistoryManager(settings.chat_history_db_path)
    share_manager = ShareManager(settings.chat_history_db_path)
    agent = RAGAgent(settings, get_chunk_boosts=chat_history_mgr.get_chunk_boosts)
    rate_limiter = RateLimiter(
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )

    from collections import deque
    ingestion_log: deque = deque(maxlen=100)

    def _log_ingestion(action: str, detail: str, status: str = "info") -> None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        ingestion_log.append(f"[{timestamp}] [{status.upper()}] {action}: {detail}")

    from .watcher import DocWatcher
    _watcher = DocWatcher(settings, callback=lambda msg: _log_ingestion("AutoSync", msg, "info"))
    _watcher.start()
    import atexit
    atexit.register(_watcher.stop)

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

    def _check_rate_limit(key: str) -> tuple[bool, str]:
        allowed, remaining = rate_limiter.check(key)
        if not allowed:
            return False, "rate_limited"
        return True, ""

    def chat_fn(
        message: str,
        history: list[dict[str, str]] | list[list[str]] | None = None,
        generate_suggestions: bool = False,
    ) -> tuple[str, list[str], str, str, list[str], list[str]]:
        history_text = _history_to_text(history)
        answer, sources, run_id, confidence, suggestions, chunk_ids = agent.answer(
            message, history_text=history_text, generate_suggestions=generate_suggestions,
        )
        return answer, sources, run_id, confidence, suggestions, chunk_ids

    def chat_api_fn(message: str, request: gr.Request | None = None) -> str:
        if not (message or "").strip():
            return ""

        key = _session_key(request)
        allowed, _ = _check_rate_limit(key)
        if not allowed:
            return "Rate limit exceeded. Please wait before sending another message."

        conv_id = chat_history_mgr.get_or_create_conversation(key)
        history_items = chat_history_mgr.get_history(conv_id)
        answer, sources, _, _, suggestions, chunk_ids = chat_fn(message, history=history_items)

        chat_history_mgr.auto_title(conv_id, message)
        chat_history_mgr.add_message(conv_id, "user", message)
        chat_history_mgr.add_message(conv_id, "assistant", answer)

        if sources and _user_requested_sources(message):
            source_text = "\n".join(f"- {source}" for source in sources[:8])
            answer = f"{answer}\n\nSources:\n{source_text}"

        if suggestions:
            answer += "\n\nFollow-up:\n" + "\n".join(f"- {s}" for s in suggestions if isinstance(s, str) and s.strip())

        return answer

    def chat_turn_api_fn(message: str, history_json: str) -> tuple[str, str]:
        parsed_history: Any
        try:
            parsed_history = json.loads(history_json or "[]")
        except Exception:
            parsed_history = []

        history_items = _normalize_history(parsed_history)
        answer, sources, _, _, suggestions, _ = chat_fn(message, history=history_items)

        if sources and _user_requested_sources(message):
            source_text = "\n".join(f"- {source}" for source in sources[:8])
            answer = f"{answer}\n\nSources:\n{source_text}"

        if suggestions:
            answer += "\n\nFollow-up:\n" + "\n".join(f"- {s}" for s in suggestions if isinstance(s, str) and s.strip())

        updated_history = [
            *history_items,
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return answer, json.dumps(updated_history, ensure_ascii=False)

    def _toast(message: str, toast_type: str = "info") -> str:
        icons = {"success": "✅", "error": "❌", "info": "ℹ️", "warning": "⚠️"}
        icon = icons.get(toast_type, "ℹ️")
        return f'<div id="toast-container"><div class="toast toast-{toast_type}">{icon} {message}</div></div>'

    def upload_fn(files: list[Any] | None, progress: gr.Progress = gr.Progress()) -> str:
        if not files:
            return _toast("No files selected.", "warning")
        progress(0.0, desc="Preparing files...")
        try:
            paths = [str(Path(file.name).resolve()) for file in files]
            total = len(paths)
            indexed = 0
            skipped = 0
            deleted = 0
            _log_ingestion("Upload", f"Processing {total} file(s)", "info")
            for i, p in enumerate(paths):
                fname = Path(p).name
                progress((i + 1) / total, desc=f"Processing {i + 1}/{total}: {fname}")
                result = ingest_files(settings, [p], origin="upload")
                indexed += result['indexed']
                skipped += result['skipped']
                deleted += result['deleted']
                _log_ingestion("Upload", f"{fname}: {result['indexed']} chunks indexed", "success" if result['indexed'] else "warning")
            msg = f"Indexed: {indexed} | Skipped: {skipped} | Deleted: {deleted}"
            progress(1.0, desc="Done!")
            return _toast(msg, "success" if indexed > 0 else "info")
        except Exception as error:
            _log_ingestion("Upload", f"Failed: {error}", "error")
            return _toast(f"Upload indexing failed: {error}", "error")

    def sync_fn(progress: gr.Progress = gr.Progress()) -> str:
        progress(0.0, desc="Scanning local directories...")
        _log_ingestion("Sync", "Started local doc sync", "info")
        try:
            result = sync_local_docs(settings)
            msg = f"Synced {result.get('total_files', 0)} files | Indexed: {result['indexed']} | Deleted: {result['deleted']}"
            progress(1.0, desc="Done!")
            _log_ingestion("Sync", msg, "success" if result['indexed'] > 0 else "info")
            return _toast(msg, "success" if result['indexed'] > 0 else "info")
        except Exception as error:
            _log_ingestion("Sync", f"Failed: {error}", "error")
            return _toast(f"Local sync failed: {error}", "error")

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

    def analytics_fn(time_range: str) -> tuple[str, str, Any, Any, str, str, Any]:
        if not settings.langsmith_tracing:
            msg = "LangSmith tracing is disabled."
            return msg, msg, None, None, msg, msg, None

        fb = get_feedback_summary(settings, time_range=time_range)
        fb_str = f"**Feedback:** 👍 {fb['up']} | 👎 {fb['down']} | Total: {fb['total']}"

        lat = get_latency_stats(settings, time_range=time_range)
        lat_str = f"**Avg Latency ({time_range}):** {lat['avg_ms']} ms"

        runs = get_recent_runs(settings, limit=10, time_range=time_range)
        if not runs:
            runs_str = "No recent runs found."
        else:
            runs_str = "| Time | Question | Latency (ms) | Feedback |\n|---|---|---|---|\n"
            for r in runs:
                runs_str += f"| {r['time']} | {r['question']} | {r['latency_ms']} | {r['feedback']} |\n"

        queries = get_top_queries(settings, limit=5, time_range=time_range)
        if not queries:
            top_str = "No query data available."
        else:
            top_str = "| # | Question | Count | Avg Latency |\n|---|---|---|---|\n"
            for i, q in enumerate(queries, 1):
                top_str += f"| {i} | {q['question']} | {q['count']} | {q['avg_latency_ms']} ms |\n"

        csv_data = export_runs_csv(settings, time_range=time_range, limit=200)
        csv_str = csv_data if csv_data else ""

        lat_series = get_latency_timeseries(settings, time_range=time_range)
        fb_series = get_feedback_timeseries(settings, time_range=time_range)

        lat_fig = None
        fb_fig = None

        if lat_series and len(lat_series) > 1:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime

            dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in lat_series]
            avgs = [d["avg_ms"] for d in lat_series]
            counts = [d["count"] for d in lat_series]

            fig, ax1 = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor("#1e1e2e")
            ax1.set_facecolor("#1e1e2e")
            ax1.plot(dates, avgs, color="#89b4fa", linewidth=2, marker="o", markersize=4)
            ax1.set_ylabel("Avg Latency (ms)", color="#cdd6f4")
            ax1.tick_params(colors="#a6adc8")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax1.spines["bottom"].set_color("#313244")
            ax1.spines["top"].set_color("#313244")
            ax1.spines["left"].set_color("#313244")
            ax1.spines["right"].set_color("#313244")

            ax2 = ax1.twinx()
            ax2.bar(dates, counts, alpha=0.3, color="#f38ba8", width=0.6, label="Runs")
            ax2.set_ylabel("Run Count", color="#cdd6f4")
            ax2.tick_params(colors="#a6adc8")

            fig.tight_layout()
            lat_fig = fig

        if fb_series and len(fb_series) > 1:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime

            dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in fb_series]
            ups = [d["up"] for d in fb_series]
            downs = [d["down"] for d in fb_series]

            fig, ax = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor("#1e1e2e")
            ax.set_facecolor("#1e1e2e")
            ax.bar(dates, ups, color="#a6e3a1", label="👍", width=0.6, alpha=0.8)
            ax.bar(dates, downs, color="#f38ba8", label="👎", width=0.6, alpha=0.8, bottom=ups)
            ax.set_ylabel("Feedback", color="#cdd6f4")
            ax.tick_params(colors="#a6adc8")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")
            ax.spines["bottom"].set_color("#313244")
            ax.spines["top"].set_color("#313244")
            ax.spines["left"].set_color("#313244")
            ax.spines["right"].set_color("#313244")
            fig.tight_layout()
            fb_fig = fig

        return fb_str, lat_str, lat_fig, fb_fig, runs_str, top_str, csv_str

    def website_fn(url: str, max_pages: int, progress: gr.Progress = gr.Progress()) -> str:
        target_url = (url or settings.website_default_url).strip()
        if not target_url:
            return _toast("Website URL is required.", "warning")
        progress(0.0, desc="Starting crawl...")
        _log_ingestion("Crawl", f"Starting crawl: {target_url}", "info")
        try:
            pages_limit = max(1, int(max_pages or settings.website_max_pages))
            result = ingest_website(settings, start_url=target_url, max_pages=pages_limit)
            msg = f"Crawled {result['total_pages']} pages | Indexed: {result['indexed']} | Deleted: {result['deleted']}"
            progress(1.0, desc="Done!")
            _log_ingestion("Crawl", f"Completed: {result['total_pages']} pages from {target_url}", "success")
            return _toast(msg, "success" if result['indexed'] > 0 else "info")
        except Exception as error:
            _log_ingestion("Crawl", f"Failed: {error}", "error")
            return _toast(f"Website crawl failed: {error}", "error")

    def text_ingest_fn(text: str, source_name: str, progress: gr.Progress = gr.Progress()) -> str:
        if not text.strip():
            return _toast("Text cannot be empty.", "warning")
        progress(0.0, desc="Indexing text...")
        target_source = source_name.strip() or "Raw Text Input"
        _log_ingestion("Text", f"Ingesting: {target_source}", "info")
        try:
            result = ingest_text(settings, text, target_source)
            msg = f"Indexed {result['indexed']} chunks | Deleted old: {result['deleted']}"
            progress(1.0, desc="Done!")
            _log_ingestion("Text", f"{target_source}: {result['indexed']} chunks", "success" if result['indexed'] else "warning")
            return _toast(msg, "success" if result['indexed'] > 0 else "info")
        except Exception as error:
            _log_ingestion("Text", f"Failed: {error}", "error")
            return _toast(f"Text ingestion failed: {error}", "error")

    def batch_url_fn(urls_text: str, progress: gr.Progress = gr.Progress()) -> str:
        urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
        if not urls:
            return _toast("Enter at least one URL.", "warning")
        _log_ingestion("BatchURL", f"Starting: {len(urls)} URLs", "info")
        progress(0.0, desc=f"Processing {len(urls)} URLs...")
        try:
            result = ingest_url_list(settings, urls, progress_fn=lambda p, desc: progress(p, desc=desc))
            msg = f"Indexed: {result['indexed']} | Errors: {result['errors']} | Skipped: {result['skipped']}"
            progress(1.0, desc="Done!")
            _log_ingestion("BatchURL", msg, "success" if result['indexed'] else "warning")
            return _toast(msg, "success" if result['indexed'] > 0 else "warning")
        except Exception as error:
            _log_ingestion("BatchURL", f"Failed: {error}", "error")
            return _toast(f"Batch URL ingestion failed: {error}", "error")

    with gr.Blocks(
        title="IEEE AI Chatbot",
        css=CSS,
        theme=THEME,
    ) as demo:

        # ── STATE ──────────────────────────────────────────────────────
        conv_id_state = gr.State(0)
        session_key_state = gr.State("default")
        current_run_id = gr.State("")
        current_sources = gr.State([])
        current_confidence = gr.State("")
        current_suggestions = gr.State([])
        current_chunk_ids = gr.State([])
        current_share_id = gr.State("")
        share_question = gr.State("")
        # ── TOAST CONTAINER ────────────────────────────────────────────
        toast_container = gr.HTML("<div id='toast-container'></div>")

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
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                var sendBtn = document.querySelector('.send-btn');
                if (sendBtn) sendBtn.click();
            }
            if (e.key === 'Escape') {
                var input = document.querySelector('.input-textbox textarea');
                if (input && document.activeElement === input) input.blur();
            }
        });
        </script>
        """)

        # ── MAIN LAYOUT: TOP-LEVEL TABS ──────────────────────────────
        with gr.Tabs(elem_classes=["top-tabs"]):

            # ── TOP TAB: CHAT ──────────────────────────
            with gr.Tab("💬 Chat"):
                with gr.Row():

                    # ── LEFT: HISTORY SIDEBAR ──────────────────────────────
                    with gr.Column(scale=1, elem_classes=["sidebar-container"], min_width=280):
                        gr.Markdown("## 💬 Conversations", elem_classes=["control-header"])
                        with gr.Row():
                            new_chat_btn = gr.Button("+ New Chat", variant="primary", size="sm")
                            refresh_conv_btn = gr.Button("🔄", size="sm", scale=0)
                        conv_selector = gr.Radio(
                            choices=[], label="", info="Select a conversation to load",
                            interactive=True, visible=True,
                        )

                        gr.HTML("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>")

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
                            welcome_title = gr.Markdown(
                                "Ask anything about IEEE Beni Suef",
                                elem_classes=["welcome-title"],
                            )
                            welcome_subtitle = gr.Markdown(
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
                            height=440,
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

                        # Action Row
                        with gr.Row(elem_classes=["feedback-row"]):
                            upvote_btn = gr.Button("👍 Useful", elem_classes=["feedback-btn"], size="sm")
                            downvote_btn = gr.Button("👎 Unhelpful", elem_classes=["feedback-btn"], size="sm")
                            regenerate_btn = gr.Button("🔄 Regenerate", elem_classes=["feedback-btn"], size="sm")
                            edit_btn = gr.Button("✏️ Edit", elem_classes=["feedback-btn"], size="sm")
                            share_btn = gr.Button("🔗 Share", elem_classes=["feedback-btn"], size="sm")
                            clear_btn = gr.Button("🗑️ Clear", elem_classes=["clear-btn"], size="sm")

                        feedback_status = gr.Markdown(visible=False)

                        # ── CHAT LOGIC ─────────────────────────────────────
                        def _build_conv_choices(session_key: str, active_conv_id: int = 0):
                            convs = chat_history_mgr.list_conversations(session_key)
                            if not convs:
                                return [], 0
                            choices = []
                            found_active = False
                            for c in convs:
                                label = f"{c['title'][:45]}"
                                choices.append((label, c["id"]))
                                if c["id"] == active_conv_id:
                                    found_active = True
                            if not found_active:
                                active_conv_id = convs[0]["id"]
                            return choices, active_conv_id

                        def init_session(request: gr.Request | None):
                            key = _session_key(request)
                            conv_id = chat_history_mgr.get_or_create_conversation(key)
                            history = chat_history_mgr.get_history(conv_id)
                            hist_for_chatbot = [{"role": m["role"], "content": m["content"]} for m in history]
                            choices, active_id = _build_conv_choices(key, conv_id)
                            return conv_id, key, hist_for_chatbot, gr.update(choices=choices, value=active_id)

                        def load_conversation(conv_id: int, session_key: str):
                            if not conv_id:
                                conv_id = chat_history_mgr.get_or_create_conversation(session_key)
                            history = chat_history_mgr.get_history(conv_id)
                            hist_for_chatbot = [{"role": m["role"], "content": m["content"]} for m in history]
                            show_welcome = len(hist_for_chatbot) == 0
                            return conv_id, hist_for_chatbot, gr.update(visible=show_welcome)

                        conv_selector.change(
                            fn=load_conversation,
                            inputs=[conv_selector, session_key_state],
                            outputs=[conv_id_state, chatbot, welcome_container],
                        )

                        def user(user_message, history, conv_id, session_key):
                            if not user_message.strip():
                                return "", history, conv_id
                            allowed, _ = _check_rate_limit(session_key)
                            if not allowed:
                                return "", history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "Rate limit exceeded. Please wait before sending another message."}], conv_id
                            chat_history_mgr.auto_title(conv_id, user_message)
                            chat_history_mgr.add_message(conv_id, "user", user_message)
                            return "", history + [{"role": "user", "content": user_message}], conv_id

                        def bot(history, conv_id, session_key):
                            user_message = history[-1]["content"]
                            history_text = _history_to_text(history[:-1])

                            history.append({"role": "assistant", "content": ""})

                            run_id = ""
                            sources = []
                            final_confidence = ""
                            final_suggestions = []
                            chunk_ids = []
                            html_answer = ""
                            for chunk, src, r_id, conf, sugg, html_a, cids in agent.answer_stream(user_message, history_text=history_text):
                                run_id = r_id
                                sources = src
                                final_confidence = conf or ""
                                if sugg:
                                    final_suggestions = sugg
                                if html_a:
                                    html_answer = html_a
                                if cids:
                                    chunk_ids = cids
                                history[-1]["content"] += chunk
                                yield history, run_id, sources, final_confidence, conv_id, final_suggestions, chunk_ids

                            msg = html_answer or history[-1]["content"]
                            has_refs = "\U0001f4da **References" in msg
                            if not has_refs:
                                if sources and _user_requested_sources(history[-2]["content"] if len(history) >= 2 else ""):
                                    src_text = "\n".join(f"- {s}" for s in sources[:8])
                                    msg += "\n\nSources:\n" + src_text
                            if final_confidence:
                                emoji = "\U0001f7e2" if final_confidence == "High" else "\U0001f7e1" if final_confidence == "Medium" else "\U0001f534" if final_confidence == "Low" else "\U0001f310"
                                msg += "\n" + emoji + " Confidence: " + final_confidence
                            if isinstance(final_suggestions, list) and final_suggestions and not has_refs:
                                safe = [s for s in final_suggestions if isinstance(s, str) and s.strip()]
                                if safe:
                                    sep = "\n- "
                                    msg += "\n\nFollow-up:\n- " + sep.join(safe)
                            history[-1]["content"] = msg
                            chat_history_mgr.add_message(conv_id, "assistant", msg)
                            yield history, run_id, sources, final_confidence, conv_id, final_suggestions, chunk_ids

                        def share_last_answer(history, sources, confidence):
                            if not history or len(history) < 2:
                                return gr.update(value="", visible=True), ""
                            question = history[-2]["content"] if history[-2]["role"] == "user" else ""
                            answer = history[-1]["content"] if history[-1]["role"] == "assistant" else ""
                            if not answer:
                                return gr.update(value="Nothing to share.", visible=True), ""
                            share_id = share_manager.create_share(question, answer, sources, confidence)
                            return gr.update(value=f"✅ Shared! ID: `{share_id}`", visible=True), share_id

                        # Initialize session on load
                        demo.load(
                            fn=init_session,
                            inputs=None,
                            outputs=[conv_id_state, session_key_state, chatbot, conv_selector],
                            queue=False,
                        )

                        bot_outputs = [chatbot, current_run_id, current_sources, current_confidence, conv_id_state, current_suggestions, current_chunk_ids]

                        # Triggers
                        msg_box.submit(
                            fn=user,
                            inputs=[msg_box, chatbot, conv_id_state, session_key_state],
                            outputs=[msg_box, chatbot, conv_id_state],
                            queue=False,
                        ).then(
                            fn=bot,
                            inputs=[chatbot, conv_id_state, session_key_state],
                            outputs=bot_outputs,
                        )

                        submit_btn.click(
                            fn=user,
                            inputs=[msg_box, chatbot, conv_id_state, session_key_state],
                            outputs=[msg_box, chatbot, conv_id_state],
                            queue=False,
                        ).then(
                            fn=bot,
                            inputs=[chatbot, conv_id_state, session_key_state],
                            outputs=bot_outputs,
                        )

                        # Suggestion cards
                        def click_card(card_val, history, conv_id, session_key):
                            chat_history_mgr.auto_title(conv_id, card_val)
                            chat_history_mgr.add_message(conv_id, "user", card_val)
                            return history + [{"role": "user", "content": card_val}], gr.update(visible=False), conv_id

                        for card in [card1, card2, card3, card4]:
                            card.click(
                                fn=click_card,
                                inputs=[card, chatbot, conv_id_state, session_key_state],
                                outputs=[chatbot, welcome_container, conv_id_state],
                                queue=False,
                            ).then(
                                fn=bot,
                                inputs=[chatbot, conv_id_state, session_key_state],
                                outputs=bot_outputs,
                            )

                        # Clear
                        def clear_chat(session_key):
                            conv_id = chat_history_mgr.create_new_conversation(session_key)
                            choices, active_id = _build_conv_choices(session_key, conv_id)
                            return [], gr.update(visible=True), conv_id, gr.update(choices=choices, value=active_id)

                        clear_btn.click(
                            fn=clear_chat,
                            inputs=[session_key_state],
                            outputs=[chatbot, welcome_container, conv_id_state, conv_selector],
                            queue=False,
                        )

                        # New Chat
                        def new_chat(session_key):
                            conv_id = chat_history_mgr.create_new_conversation(session_key)
                            choices, active_id = _build_conv_choices(session_key, conv_id)
                            return [], gr.update(visible=True), conv_id, gr.update(choices=choices, value=active_id)

                        new_chat_btn.click(
                            fn=new_chat,
                            inputs=[session_key_state],
                            outputs=[chatbot, welcome_container, conv_id_state, conv_selector],
                            queue=False,
                        )

                        # Refresh conversation list
                        def refresh_conv_list(session_key, active_conv_id):
                            choices, active_id = _build_conv_choices(session_key, active_conv_id)
                            return gr.update(choices=choices, value=active_id)

                        refresh_conv_btn.click(
                            fn=refresh_conv_list,
                            inputs=[session_key_state, conv_id_state],
                            outputs=[conv_selector],
                        )

                        # Regenerate
                        def regenerate_last(history, conv_id, session_key):
                            if len(history) < 2:
                                return history, conv_id
                            history = history[:-1]
                            last_user_msg = history[-1]["content"] if history else ""
                            if not last_user_msg:
                                return history, conv_id
                            conv_id = chat_history_mgr.get_or_create_conversation(session_key)
                            history.append({"role": "assistant", "content": ""})
                            history_text = _history_to_text(history[:-1])
                            run_id = ""
                            sources = []
                            final_confidence = ""
                            final_suggestions = []
                            chunk_ids = []
                            html_answer = ""
                            for chunk, src, r_id, conf, sugg, html_a, cids in agent.answer_stream(last_user_msg, history_text=history_text):
                                run_id = r_id
                                sources = src
                                final_confidence = conf or ""
                                if sugg:
                                    final_suggestions = sugg
                                if html_a:
                                    html_answer = html_a
                                if cids:
                                    chunk_ids = cids
                                history[-1]["content"] += chunk
                                yield history, run_id, sources, final_confidence, conv_id, final_suggestions, chunk_ids
                            msg = html_answer or history[-1]["content"]
                            has_refs = "\U0001f4da **References" in msg
                            if not has_refs:
                                if sources and _user_requested_sources(history[-2]["content"] if len(history) >= 2 else ""):
                                    src_text = "\n".join(f"- {s}" for s in sources[:8])
                                    msg += "\n\nSources:\n" + src_text
                            if final_confidence:
                                emoji = "\U0001f7e2" if final_confidence == "High" else "\U0001f7e1" if final_confidence == "Medium" else "\U0001f534" if final_confidence == "Low" else "\U0001f310"
                                msg += "\n" + emoji + " Confidence: " + final_confidence
                            if isinstance(final_suggestions, list) and final_suggestions and not has_refs:
                                safe = [s for s in final_suggestions if isinstance(s, str) and s.strip()]
                                if safe:
                                    sep = "\n- "
                                    msg += "\n\nFollow-up:\n- " + sep.join(safe)
                            history[-1]["content"] = msg
                            chat_history_mgr.add_message(conv_id, "assistant", msg)
                            yield history, run_id, sources, final_confidence, conv_id, final_suggestions, chunk_ids

                        regenerate_btn.click(
                            fn=regenerate_last,
                            inputs=[chatbot, conv_id_state, session_key_state],
                            outputs=bot_outputs,
                            queue=False,
                        )

                        # Edit - copy last user message to input, remove last pair
                        def edit_last(history):
                            if len(history) < 2:
                                return history, ""
                            last_user = history[-2]["content"] if len(history) >= 2 else ""
                            history = history[:-2]
                            return history, last_user

                        edit_btn.click(
                            fn=edit_last,
                            inputs=[chatbot],
                            outputs=[chatbot, msg_box],
                            queue=False,
                        )

                        share_output = gr.Markdown(visible=True, elem_classes=["prose-custom"])
                        share_btn.click(
                            fn=share_last_answer,
                            inputs=[chatbot, current_sources, current_confidence],
                            outputs=[share_output, current_share_id],
                            queue=False,
                        )
                        # Feedback
                        def handle_feedback(run_id, chunk_ids, score):
                            if not run_id:
                                return _toast("No response to evaluate yet.", "warning")
                            success = agent.submit_feedback(run_id, score=score)
                            if chunk_ids:
                                for cid in chunk_ids:
                                    chat_history_mgr.record_chunk_feedback(cid, 1 if score >= 0.5 else -1)
                            if success:
                                return _toast("Feedback sent! Thank you.", "success")
                            return _toast("Failed to send feedback. Check LangSmith tracing.", "error")

                        upvote_btn.click(
                            lambda r, c: handle_feedback(r, c, 1.0),
                            inputs=[current_run_id, current_chunk_ids],
                            outputs=[feedback_status],
                        )

                        downvote_btn.click(
                            lambda r, c: handle_feedback(r, c, 0.0),
                            inputs=[current_run_id, current_chunk_ids],
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
                            upload_output = gr.HTML(
                                value="",
                                elem_classes=["output-box"],
                            )
                            upload_button.click(fn=upload_fn, inputs=[uploader], outputs=[upload_output])
                            
                        with gr.Group(elem_classes=["ingest-group"]):
                            gr.Markdown('<span class="ingest-group-title">📁 Local Sync</span>')
                            sync_button = gr.Button(
                                "Sync Local Docs",
                                elem_classes=["secondary-btn"],
                            )
                            sync_output = gr.HTML(
                                value="",
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
                            text_output = gr.HTML(
                                value="",
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
                            website_output = gr.HTML(
                                value="",
                                elem_classes=["output-box"],
                            )
                            website_button.click(
                                fn=website_fn,
                                inputs=[website_url, website_max_pages],
                                outputs=[website_output],
                            )

                        with gr.Group(elem_classes=["ingest-group"]):
                            gr.Markdown('<span class="ingest-group-title">🔗 Batch URLs</span>')
                            batch_urls_input = gr.Textbox(
                                label="One URL per line",
                                lines=3,
                                placeholder="https://example.com/page1\nhttps://example.com/page2",
                            )
                            batch_url_button = gr.Button(
                                "Crawl & Index All",
                                elem_classes=["secondary-btn"],
                            )
                            batch_url_output = gr.HTML(
                                value="",
                                elem_classes=["output-box"],
                            )
                            batch_url_button.click(
                                fn=batch_url_fn,
                                inputs=[batch_urls_input],
                                outputs=[batch_url_output],
                            )

                with gr.Accordion("📂 Source Manager", open=False):
                    ingest_sources_display = gr.Markdown(
                        "Click refresh to list sources.",
                        elem_classes=["prose-custom"],
                    )
                    with gr.Row():
                        refresh_ingest_sources_btn = gr.Button("🔄 List Sources", variant="primary", size="sm")

                    def list_ingest_sources_fn() -> str:
                        sources = list_all_sources(settings)
                        if not sources:
                            return "No sources found in the knowledge base."
                        lines = [
                            "| # | Source | Origin | Chunks |",
                            "|---|---|---|---|",
                        ]
                        for i, s in enumerate(sources, 1):
                            sid = s["id"][:55] + "..." if len(s["id"]) > 55 else s["id"]
                            lines.append(f"| {i} | `{sid}` | {s['origin']} | {s['chunk_count']} |")
                        return "\n".join(lines)

                    refresh_ingest_sources_btn.click(
                        fn=list_ingest_sources_fn,
                        inputs=None,
                        outputs=[ingest_sources_display],
                    )

                with gr.Accordion("📋 Ingestion Log", open=False):
                    log_display = gr.HTML(
                        "<div class='prose-custom'>No ingestion activity yet.</div>",
                    )
                    def refresh_log_fn():
                        if not ingestion_log:
                            return "<div class='prose-custom'>No ingestion activity yet.</div>"
                        entries = "\n".join(
                            f'<div style="font-family: JetBrains Mono, monospace; font-size: 0.78rem; '
                            f'padding: 2px 0; color: var(--body-text-color-subdued);">{e}</div>'
                            for e in list(ingestion_log)[-50:]
                        )
                        return f'<div style="max-height: 300px; overflow-y: auto;">{entries}</div>'
                    with gr.Row():
                        refresh_log_btn = gr.Button("🔄 Refresh Log", size="sm")
                        refresh_log_btn.click(fn=refresh_log_fn, inputs=None, outputs=[log_display])
                    clear_log_btn = gr.Button("🗑️ Clear Log", size="sm")
                    def clear_log_fn():
                        ingestion_log.clear()
                        return "<div class='prose-custom'>Log cleared.</div>"
                    clear_log_btn.click(fn=clear_log_fn, inputs=None, outputs=[log_display])

            # ── TOP TAB: VECTORS ──────────────────────────
            with gr.Tab("🔍 Vectors"):
                with gr.Column(elem_classes=["glass-card"]):
                    gr.Markdown("### 🔎 Search Vectors", elem_classes=["prose-custom"])
                    with gr.Row():
                        search_query = gr.Textbox(
                            label="Query",
                            placeholder="Search across all indexed chunks...",
                            scale=4,
                        )
                        search_top_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Results",
                            scale=1,
                        )
                        search_btn = gr.Button("Search", variant="primary", scale=1)
                    search_results = gr.Markdown(
                        "Enter a query and click Search.",
                        elem_classes=["prose-custom"],
                    )
                    preview_content = gr.Markdown(
                        "", elem_classes=["prose-custom"], visible=False
                    )

                    search_state = gr.State([])

                    def vector_search_fn(query: str, top_k: int) -> tuple[str, list, str]:
                        if not query.strip():
                            return _toast("Enter a search query.", "warning"), [], ""
                        chunks = search_chunks(settings, query, top_k=int(top_k))
                        if not chunks:
                            return "No results found.", [], ""
                        lines = []
                        for i, c in enumerate(chunks, 1):
                            snippet = c["content"][:300]
                            lines.append(f"**{i}. Score: {c['score']}** — *Source: {c['filename']}*")
                            lines.append(f"> {snippet}")
                            lines.append("")
                        lines.append("\n---\n💡 *Click a result number below and press 'View Full' to see the complete chunk.*")
                        return "\n".join(lines), chunks, ""

                    def view_full_chunk(result_idx: str, chunks: list) -> str:
                        try:
                            idx = int(result_idx) - 1
                            if idx < 0 or idx >= len(chunks):
                                return _toast("Invalid result number.", "warning")
                            c = chunks[idx]
                            return (
                                f"**Full Content — Score: {c['score']}**\n"
                                f"*Source: `{c['filename']}`*\n\n"
                                f"{c['content']}"
                            )
                        except ValueError:
                            return _toast("Enter a valid number.", "warning")

                    search_btn.click(
                        fn=vector_search_fn,
                        inputs=[search_query, search_top_k],
                        outputs=[search_results, search_state, preview_content],
                    )

                    with gr.Row():
                        view_idx = gr.Textbox(
                            label="View result #",
                            placeholder="e.g., 1",
                            scale=1,
                        )
                        view_btn = gr.Button("View Full", variant="primary", scale=0)
                    view_btn.click(
                        fn=view_full_chunk,
                        inputs=[view_idx, search_state],
                        outputs=[preview_content],
                    )

                    gr.HTML("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>")

                    gr.Markdown("### 🔎 View Source Chunks", elem_classes=["prose-custom"])
                    with gr.Row():
                        view_source_id = gr.Textbox(
                            label="Source ID",
                            placeholder="Paste source ID to see its chunks",
                            scale=4,
                        )
                        view_source_btn = gr.Button("View Chunks", variant="primary", scale=1)
                    source_chunks_display = gr.Markdown(
                        "", elem_classes=["prose-custom"],
                    )

                    def view_source_chunks_fn(source_id: str) -> str:
                        if not source_id.strip():
                            return _toast("Enter a source ID.", "warning")
                        chunks = get_source_chunks(settings, source_id.strip())
                        if not chunks:
                            return "No chunks found for this source."
                        lines = []
                        for i, c in enumerate(chunks, 1):
                            snippet = c["content"][:300]
                            lines.append(f"**Chunk {i}** — `{c['chunk_id'][:16]}...`")
                            lines.append(f"> {snippet}")
                            lines.append("")
                        return "\n".join(lines)

                    view_source_btn.click(
                        fn=view_source_chunks_fn,
                        inputs=[view_source_id],
                        outputs=[source_chunks_display],
                    )

                    gr.HTML("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>")

                    gr.Markdown("### 📂 Source Manager", elem_classes=["prose-custom"])

                    def list_sources_fn() -> str:
                        sources = list_all_sources(settings)
                        if not sources:
                            return "No sources found in the knowledge base."
                        lines = [
                            "| # | Source | Origin | Chunks |",
                            "|---|---|---|---|",
                        ]
                        for i, s in enumerate(sources, 1):
                            sid = s["id"][:60] + "..." if len(s["id"]) > 60 else s["id"]
                            lines.append(f"| {i} | {sid} | {s['origin']} | {s['chunk_count']} |")
                        return "\n".join(lines)

                    sources_display = gr.Markdown(
                        "Click refresh to list sources.",
                        elem_classes=["prose-custom"],
                    )
                    with gr.Row():
                        refresh_sources_btn = gr.Button("🔄 List Sources", variant="primary")
                        refresh_sources_btn.click(
                            fn=list_sources_fn,
                            inputs=None,
                            outputs=[sources_display],
                        )

                    gr.Markdown("### 🗑️ Delete Source", elem_classes=["prose-custom"])
                    with gr.Row():
                        delete_source_input = gr.Textbox(
                            label="Source ID to delete",
                            placeholder="Paste the full source ID from the list above",
                            scale=4,
                        )
                        delete_source_btn = gr.Button("Delete", variant="stop", scale=1)

                    def delete_source_fn(source_id: str) -> str:
                        if not source_id.strip():
                            return _toast("Enter a source ID to delete.", "warning")
                        success = delete_source(settings, source_id.strip())
                        if success:
                            return _toast(f"Deleted source: {source_id[:60]}", "success")
                        return _toast(f"Source not found: {source_id[:60]}", "error")

                    delete_source_btn.click(
                        fn=delete_source_fn,
                        inputs=[delete_source_input],
                        outputs=[toast_container],
                    )

            # ── TOP TAB: ANALYTICS ──────────────────────────
            with gr.Tab("📈 Analytics"):
                with gr.Column(elem_classes=["glass-card"]):
                    with gr.Row():
                        analytics_range = gr.Radio(
                            ["24h", "7d", "30d"], value="7d", label="Time Range",
                            elem_classes=["pill-group"],
                        )
                    with gr.Row():
                        analytics_fb = gr.Markdown(
                            "Loading feedback...",
                            elem_classes=["prose-custom"],
                        )
                        analytics_lat = gr.Markdown(
                            "Loading latency...",
                            elem_classes=["prose-custom"],
                        )
                    with gr.Row():
                        lat_chart = gr.Plot(label="Latency Trend")
                        fb_chart = gr.Plot(label="Feedback Trend")
                    analytics_runs = gr.Markdown(
                        "Loading runs...",
                        elem_classes=["prose-custom"],
                    )
                    analytics_top = gr.Markdown(
                        "Loading top queries...",
                        elem_classes=["prose-custom"],
                    )
                    csv_download = gr.DownloadButton(
                        "📥 Download CSV", variant="secondary", size="sm",
                    )
                    analytics_btn = gr.Button("Refresh Analytics", variant="primary")
                    analytics_outs = [analytics_fb, analytics_lat, lat_chart, fb_chart, analytics_runs, analytics_top, csv_download]
                    analytics_btn.click(fn=analytics_fn, inputs=[analytics_range], outputs=analytics_outs)
                    analytics_range.change(fn=analytics_fn, inputs=[analytics_range], outputs=analytics_outs)

            with gr.Tab("🛠️ Admin"):
                with gr.Column(elem_classes=["glass-card"]):
                    admin_info = gr.Markdown(
                        _get_admin_info(agent, settings),
                        elem_classes=["prose-custom"],
                    )
                    admin_refresh = gr.Button("🔄 Refresh System Info", variant="primary")
                    admin_refresh.click(
                        fn=lambda: _get_admin_info(agent, settings),
                        inputs=None, outputs=[admin_info],
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

        # ── SHARE VIEWER ENDPOINT ─────────────────────────────────────
        share_id_input = gr.Textbox(visible=False)
        share_viewer_output = gr.Markdown(visible=False, elem_classes=["prose-custom"])
        share_viewer_trigger = gr.Button(visible=False)
        share_viewer_trigger.click(
            fn=lambda sid: _render_share(sid),
            inputs=[share_id_input],
            outputs=[share_viewer_output],
            api_name="view_share",
            queue=False,
        )

    return demo


def _render_share(share_id: str) -> str:
    """Render a shared Q&A as a standalone page."""
    try:
        settings = Settings.from_env()
        mgr = ShareManager(settings.chat_history_db_path)
        data = mgr.get_share(share_id.strip())
        if not data:
            return "⚠️ Share not found. The link may be expired or invalid."
        return (
            f"### ❓ {data['question']}\n\n"
            f"{data['answer']}\n\n"
            f"---\n"
            f"*Shared from IEEE AI Chatbot*"
        )
    except Exception as e:
        return f"Error: {e}"


def _get_admin_info(agent: RAGAgent, settings: Settings) -> str:
    """Build an admin dashboard Markdown string."""
    import platform
    from datetime import datetime

    status = agent.status() if hasattr(agent, "status") else {}
    py_ver = platform.python_version()
    lines = [
        "## 🛠️ System Information",
        "",
        f"**Python:** {py_ver}",
        f"**Chat Model:** {settings.chat_model}",
        f"**Fallback Model:** {settings.chat_model_fallback}",
        f"**Embedding Model:** {settings.embedding_model}",
        f"**Vector Store:** {settings.vector_store_type}",
        f"**Pinecone Index:** {settings.pinecone_index_name}",
        f"**Web Search Provider:** {settings.web_search_provider}",
        f"**LangSmith Tracing:** {'enabled' if settings.langsmith_tracing else 'disabled'}",
        f"**Rate Limit:** {settings.rate_limit_max_requests} req/{settings.rate_limit_window_seconds}s",
        f"**Feedback Boost:** {'enabled' if settings.feedback_boost_enabled else 'disabled'} (factor: {settings.feedback_boost_factor})",
        f"**Retriever K:** {settings.retriever_k}",
        f"**Max Output Tokens:** {settings.max_output_tokens}",
        f"**Temperature:** {settings.temperature}",
        f"**Chunk Size:** {settings.chunk_size} / Overlap: {settings.chunk_overlap}",
        f"**Max Pages (crawl):** {settings.website_max_pages}",
        "",
        "### Agent Status",
    ]
    for key, val in status.items():
        lines.append(f"- **{key}:** {val}")
    return "\n".join(lines)
