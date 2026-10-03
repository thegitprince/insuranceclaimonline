"""LLM factory. The ONLY place a ChatOpenAI client is created.

SECURITY: never wrap `build_llm` (or anything that receives an api_key) in
st.cache_resource / st.cache_data / functools.lru_cache. Cached objects are shared
across every visitor of the app, which is exactly how keys leak between users.

Every supported provider (OpenAI, OpenRouter, ...) speaks the OpenAI-compatible chat
completions API, so one ChatOpenAI client handles all of them -- only the base_url and
the key differ. See securecare.config.LLM_PROVIDERS for the provider registry.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from securecare.config import DEFAULT_MODEL, DEFAULT_PROVIDER, LLM_PROVIDERS, LLM_TIMEOUT_SECONDS


def build_llm(api_key: str, model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER) -> ChatOpenAI:
    """Create a fresh, short-lived client. The key is passed explicitly, never via os.environ."""
    base_url = LLM_PROVIDERS.get(provider, LLM_PROVIDERS[DEFAULT_PROVIDER])["base_url"]
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=1,
    )
