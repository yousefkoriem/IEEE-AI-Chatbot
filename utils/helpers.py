"""General helper functions."""


def format_context(documents: list[dict]) -> str:
    """Format retrieved document records for a model prompt."""
    return "\n\n".join(str(document.get("text", "")) for document in documents)
