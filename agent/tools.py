"""Agent tools for the IEEE BSU Student Branch chatbot."""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Module-level references
_retriever = None
_fallback_model = None

def init_tools(retriever, fallback_model):
    """Initialize tools with dependencies."""
    global _retriever, _fallback_model
    _retriever = retriever
    _fallback_model = fallback_model
    return [
        retrieve_knowledge_base,
        web_search,
        scrape_ieee_page,
        get_ieee_events,
        suggest_followups
    ]

@tool
def retrieve_knowledge_base(query: str) -> str:
    """Search the IEEE Beni-Suef University Student Branch knowledge base for information about the branch, its committees, leadership, events, and activities."""
    if not _retriever:
        return "Retriever not initialized"
    try:
        from utils.helpers import format_context
        results = _retriever.retrieve(query)
        return format_context(results)
    except Exception as e:
        return f"Error retrieving knowledge base: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the web for general IEEE information, current events, conferences, technical standards, or anything not found in the local knowledge base."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception:
        return "Search unavailable"

@tool
def scrape_ieee_page(url: str) -> str:
    """Fetch and read content from a specific IEEE or university webpage. Only works with trusted domains: ieee.org, ieee.org.eg, bsu.edu.eg"""
    allowed_domains = ['ieee.org', 'ieee.org.eg', 'bsu.edu.eg']
    try:
        domain = urlparse(url).netloc
        if not any(domain == d or domain.endswith('.' + d) for d in allowed_domains):
            return f"Error: Domain {domain} is not whitelisted."
            
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        return text[:3000]
    except Exception as e:
        return f"Error scraping page: {str(e)}"

@tool
def get_ieee_events(time_range: str = 'upcoming') -> str:
    """Get IEEE Beni-Suef Student Branch events from the official IEEE vTools Events system. Use time_range="upcoming" for future events or "past" for past events."""
    try:
        from config.settings import settings
        url = "https://events.vtools.ieee.org/api/v8/events"
        params = {"time_range": time_range}
        if settings.vtools_ou_code:
            params["ou_code"] = settings.vtools_ou_code
            
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        events = response.json()
        
        # Simple formatting of the result
        formatted = f"Found {len(events)} events.\n"
        for idx, event in enumerate(events[:5]):
            formatted += f"{idx+1}. {event.get('title', 'Unknown')} - {event.get('start_time', 'Unknown')}\n"
        return formatted
    except Exception as e:
        return f"Error fetching events: {str(e)}"

@tool
def suggest_followups(conversation_summary: str) -> str:
    """Generate 3 follow-up question suggestions based on the current conversation context."""
    if not _fallback_model:
        return "Fallback model not initialized"
    try:
        prompt = f"Based on this conversation summary, generate 3 follow-up questions.\nSummary: {conversation_summary}"
        response = _fallback_model.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Error suggesting followups: {str(e)}"
