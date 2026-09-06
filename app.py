"""Application entry point — Gradio UI + API on port 7860."""

import logging

from config.settings import settings
from config.tracing import configure_tracing
from models.model import GeminiModels
from rag.embeddings import GeminiEmbeddings
from rag.vectorstore import PineconeStore
from rag.retriever import RAGRetriever
from agent.tools import init_tools
from agent.agent import RAGAgent
from ui.gradio_ui import GradioUI
from utils.logging import configure_logging


def main():
    # ------------------------------------------------------------------
    # 1. Logging & tracing
    # ------------------------------------------------------------------
    configure_logging()
    configure_tracing()
    logger = logging.getLogger(__name__)
    logger.info("Starting IEEE AI Chatbot...")

    # ------------------------------------------------------------------
    # 2. Models
    # ------------------------------------------------------------------
    models = GeminiModels()
    logger.info("Models initialised: primary=%s, fallback=%s",
                settings.chat_model, settings.lite_model)

    # ------------------------------------------------------------------
    # 3. RAG pipeline
    # ------------------------------------------------------------------
    embeddings = GeminiEmbeddings()
    vectorstore = PineconeStore(embeddings=embeddings)
    retriever = RAGRetriever(vectorstore=vectorstore)
    logger.info("RAG pipeline ready (index=%s, dim=%d)",
                settings.pinecone_index_name, settings.embedding_dimensions)

    # ------------------------------------------------------------------
    # 4. Tools & Agent
    # ------------------------------------------------------------------
    tools = init_tools(retriever=retriever, fallback_model=models.fallback)
    agent = RAGAgent(models=models, tools=tools)
    logger.info("Agent initialised with %d tools", len(tools))

    # ------------------------------------------------------------------
    # 5. UI
    # ------------------------------------------------------------------
    ui = GradioUI(agent=agent, vectorstore=vectorstore)
    app = ui.build()
    logger.info("Launching Gradio on port 7860...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=ui._theme,
        css=ui._css,
    )


if __name__ == "__main__":
    main()