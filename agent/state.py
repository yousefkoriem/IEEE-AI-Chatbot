"""State shared by the LangGraph agent."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    """Conversation and retrieval state."""

    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    sources: list[dict]
    confidence: float
    suggestions: list[str]
