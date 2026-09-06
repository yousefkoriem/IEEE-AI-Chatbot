"""Gemini embedding integration."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings


class GeminiEmbeddings:
    """A class to manage the Gemini embedding model."""

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            task_type="retrieval_query",
            output_dimensionality=settings.embedding_dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings."""
        return self.embeddings.embed_documents(texts)