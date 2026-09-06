"""Gradio chat interface for the IEEE BSU Student Branch AI Chatbot."""

import logging
import uuid
from typing import Any

import gradio as gr

from ui.theme import IEEETheme, CUSTOM_CSS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

WELCOME_MESSAGE = """
## 🤖 IEEE Beni-Suef University Student Branch AI Assistant

Welcome! I can help you with:
- 📋 **Branch info** — committees, leadership, structure
- 📅 **Events** — upcoming and past activities
- 🎓 **Membership** — how to join, benefits
- 🌐 **General IEEE** — policies, conferences, standards

Type your question below or try one of the examples!
"""

EXAMPLE_QUESTIONS = [
    "What is IEEE BSU Student Branch?",
    "What committees does the branch have?",
    "What upcoming events are planned?",
    "How can I join IEEE?",
    "Tell me about the Computer Society chapter",
]


class GradioUI:
    """Rich Gradio chat interface with IEEE theming and API endpoints."""

    def __init__(self, agent, vectorstore=None):
        self.agent = agent
        self.vectorstore = vectorstore

    # ------------------------------------------------------------------
    # Chat handler
    # ------------------------------------------------------------------

    def _extract_text(self, content) -> str:
        """Safely extract text from message content (str, list, or other)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # LangChain messages can have list content with text/tool_use blocks
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            return "".join(parts)
        return str(content) if content else ""

    # ------------------------------------------------------------------
    # Chat handler
    # ------------------------------------------------------------------

    def _chat(self, message: str, history: list[dict], session_id: str) -> Any:
        """Process a user message and yield streaming response chunks."""
        if not message or not message.strip():
            yield history, ""
            return

        # Build message list from history
        messages = []
        for entry in history:
            messages.append((entry["role"], entry["content"]))
        messages.append(("human", message))

        config = {"configurable": {"thread_id": session_id}}

        # Try streaming, fall back to invoke
        try:
            full_response = ""
            for chunk in self.agent.stream(messages, config=config):
                # Extract token from streaming chunk
                if isinstance(chunk, dict):
                    for node_output in chunk.values():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for msg in node_output["messages"]:
                                content = self._extract_text(getattr(msg, "content", ""))
                                if content:
                                    full_response += content
                                    history_with_response = history + [
                                        {"role": "user", "content": message},
                                        {"role": "assistant", "content": full_response},
                                    ]
                                    yield history_with_response, ""
                else:
                    content = self._extract_text(getattr(chunk, "content", str(chunk)))
                    if content:
                        full_response += content
                        history_with_response = history + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": full_response},
                        ]
                        yield history_with_response, ""

            # If streaming produced no output, fall back to invoke
            if not full_response:
                result = self.agent.invoke(messages, config=config)
                if isinstance(result, dict) and "messages" in result:
                    last_msg = result["messages"][-1]
                    full_response = self._extract_text(getattr(last_msg, "content", str(last_msg)))
                else:
                    full_response = str(result)

                history_with_response = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": full_response},
                ]
                yield history_with_response, ""

        except Exception as e:
            logger.exception("Chat error")
            error_msg = f"⚠️ Sorry, something went wrong: {e}"
            history_with_error = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg},
            ]
            yield history_with_error, ""

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _get_status_html(self) -> str:
        """Generate HTML for the sidebar status panel."""
        from config.tracing import get_tracing_status

        tracing = get_tracing_status()
        tracing_dot = "🟢" if tracing.get("enabled") else "🔴"
        tracing_text = (
            f"Project: {tracing['project']}"
            if tracing.get("enabled")
            else "Disabled"
        )

        kb_status = "🟢 Connected"
        kb_vectors = "N/A"
        if self.vectorstore:
            try:
                stats = self.vectorstore.get_index_stats()
                kb_vectors = f"{stats.get('total_vectors', 0):,} vectors"
            except Exception:
                kb_status = "🔴 Error"

        return f"""
        <div class="status-panel">
            <h4 style="margin:0 0 10px;color:#FFD100;">📊 System Status</h4>
            <div class="status-item">🤖 <b>Model:</b> Gemini 2.5 Flash</div>
            <div class="status-item">🧠 <b>Fallback:</b> Gemini 2.5 Flash-Lite</div>
            <div class="status-item">{kb_status.split()[0]} <b>Knowledge Base:</b> {kb_vectors}</div>
            <div class="status-item">{tracing_dot} <b>Tracing:</b> {tracing_text}</div>
        </div>
        """

    def _health_check(self) -> str:
        """API health check endpoint."""
        return "ok"

    def _status_check(self) -> dict:
        """API status endpoint returning model + KB health."""
        result = {
            "model": "gemini-2.5-flash",
            "fallback": "gemini-2.5-flash-lite",
            "status": "healthy",
        }
        if self.vectorstore:
            try:
                stats = self.vectorstore.get_index_stats()
                result["knowledge_base"] = stats
            except Exception as e:
                result["knowledge_base"] = {"error": str(e)}
        return result

    # ------------------------------------------------------------------
    # Build the Gradio app
    # ------------------------------------------------------------------

    def build(self) -> gr.Blocks:
        """Construct and return the Gradio Blocks application."""
        self._theme = IEEETheme()
        self._css = CUSTOM_CSS

        with gr.Blocks(
            title="IEEE BSU AI Chatbot",
            analytics_enabled=False,
        ) as demo:

            # Session state
            session_id = gr.State(value=lambda: str(uuid.uuid4()))

            # ---- Header ----
            gr.HTML(
                """
                <div style="text-align:center; padding:16px 0 8px;">
                    <h1 style="color:#00629B; margin:0; font-family:Montserrat,sans-serif;">
                        🤖 IEEE BSU AI Assistant
                    </h1>
                    <p style="color:#506D8C; margin:4px 0 0; font-size:0.95rem;">
                        IEEE Beni-Suef University Student Branch • Egypt Section • Region 8
                    </p>
                </div>
                """
            )

            with gr.Row():
                # ---- Main chat column ----
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        value=[],
                        height=520,
                        show_label=False,
                        avatar_images=(None, None),
                        placeholder=WELCOME_MESSAGE,
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me about IEEE BSU...",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=1,
                            min_width=80,
                        )

                    # Example questions
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=msg,
                        label="💡 Try asking:",
                    )

                # ---- Sidebar ----
                with gr.Column(scale=1, min_width=260):
                    status_html = gr.HTML(value=self._get_status_html)

                    gr.Markdown("### ⚙️ Controls")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")

                    gr.Markdown(
                        """
                        ---
                        ### ℹ️ About
                        Built with LangChain, Gemini, and Pinecone.
                        Open source on [GitHub](https://github.com/yousefkoriem/IEEE-AI-Chatbot).
                        """
                    )

            # ---- Event handlers ----

            # Chat submit (Enter key)
            msg.submit(
                fn=self._chat,
                inputs=[msg, chatbot, session_id],
                outputs=[chatbot, msg],
                api_name="chat",
            )

            # Chat submit (Send button)
            send_btn.click(
                fn=self._chat,
                inputs=[msg, chatbot, session_id],
                outputs=[chatbot, msg],
                api_name=False,
            )

            # Clear chat
            clear_btn.click(
                fn=lambda: ([], str(uuid.uuid4())),
                outputs=[chatbot, session_id],
                api_name=False,
            )

            # Refresh status
            refresh_btn.click(
                fn=self._get_status_html,
                outputs=[status_html],
                api_name=False,
            )

            # ---- API-only endpoints (no UI component) ----
            health_btn = gr.Button(visible=False)
            health_output = gr.Textbox(visible=False)
            health_btn.click(
                fn=self._health_check,
                outputs=[health_output],
                api_name="health",
            )

            status_btn = gr.Button(visible=False)
            status_output = gr.JSON(visible=False)
            status_btn.click(
                fn=self._status_check,
                outputs=[status_output],
                api_name="status",
            )

        return demo