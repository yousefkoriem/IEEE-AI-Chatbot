"""LangGraph-backed RAG agent."""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolRetryMiddleware
from langgraph.checkpoint.memory import MemorySaver

from models.model import GeminiModels
from prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RAGAgent:
    """ReAct agent with RAG tools and conversation memory."""

    def __init__(self, models: GeminiModels, tools: list):
        self.models = models
        self.tools = tools
        self.memory = MemorySaver()

        self.agent = create_agent(
            model=self.models.primary,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.memory,
            middleware=[
                SummarizationMiddleware(
                    self.models.fallback,
                    trigger=("messages", 20),
                ),
                ToolRetryMiddleware(max_retries=3),
            ],
        )

    def invoke(self, messages, config=None):
        """Run the agent and return the full response."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        return self.agent.invoke({"messages": messages}, config=config)

    def stream(self, messages, config=None):
        """Stream the agent response token by token."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        return self.agent.stream({"messages": messages}, config=config)