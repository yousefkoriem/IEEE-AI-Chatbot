from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings


class GeminiModels:
    """A class to manage the primary and fallback Gemini models."""

    def __init__(self):
        self.primary = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key, 
        )

        self.fallback = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.google_api_key,
            
        )

