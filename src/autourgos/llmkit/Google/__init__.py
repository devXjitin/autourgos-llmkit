"""
Autourgos LLMKit - Google Gemini Provider
========================================

Google Gemini LLM integration for Autourgos Agentic Kit.

Example:
    >>> from autourgos.llmkit.Google import GoogleLLM, google_llm
    >>> 
    >>> # Class-based
    >>> llm = GoogleLLM(model="gemini-1.5-pro", api_key="your-key")
    >>> response = llm.invoke("What is AI?")
    >>> 
    >>> # Function-based
    >>> response = google_llm(
    ...     prompt="What is AI?",
    ...     model="gemini-1.5-pro",
    ...     api_key="your-key"
    ... )

Author: Autourgos Developer
Version: 1.1.0
"""

from .base import (
    google_llm,
    google_llm_stream,
    GoogleLLM,
    GoogleLLMError,
    GoogleLLMAPIError,
    GoogleLLMImportError,
    GoogleLLMResponseError,
)

__all__ = [
    "google_llm",
    "google_llm_stream",
    "GoogleLLM",
    "GoogleLLMError",
    "GoogleLLMAPIError",
    "GoogleLLMImportError",
    "GoogleLLMResponseError",
]
