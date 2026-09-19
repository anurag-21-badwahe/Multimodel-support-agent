"""Provider-agnostic LangChain chat-model factory.

The assignment deliberately uses the model name as the routing signal:

- ``provider/model`` -> ChatOpenAI against OpenRouter
- ``hf:<repo-id>`` -> ChatHuggingFace wrapping HuggingFaceEndpoint
- ``gpt-*`` -> direct OpenAI
- ``claude-*`` -> direct Anthropic
- ``gemini-*`` -> direct Google Gemini

Secrets are never embedded in source code. Provider integrations read their
credentials from environment variables (or the factory fails early with a
clear configuration error).
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _require_env(name: str) -> str:
    """Return a required environment variable or raise a useful configuration error."""

    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Create a .env file from .env.example before starting the application."
        )
    return value


def get_llm(model_name: str) -> BaseChatModel:
    """Instantiate the appropriate LangChain chat model for ``model_name``.

    The function is intentionally stateless: callers can select different models
    at runtime without changing graph code. Every branch uses the canonical
    LangChain provider integration requested by the assignment.
    """

    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    name = model_name.strip()

    # Hugging Face is the more specific prefix rule. HF repository IDs commonly
    # contain '/', so this must be checked before the generic OpenRouter branch.
    if name.startswith("hf:"):
        repo_id = name.removeprefix("hf:").strip()
        if not repo_id:
            raise ValueError("Hugging Face model name must look like hf:<repo-id>")

        token = _require_env("HUGGINGFACEHUB_API_TOKEN")
        endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip() or None

        endpoint_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "huggingfacehub_api_token": token,
            "task": "text-generation",
            "temperature": 0.0,
            "max_new_tokens": 512,
        }
        if endpoint_url:
            endpoint_kwargs["endpoint_url"] = endpoint_url

        endpoint = HuggingFaceEndpoint(**endpoint_kwargs)
        return ChatHuggingFace(llm=endpoint)

    # Assignment rule: any non-HF model containing '/' is an OpenRouter model.
    # We intentionally use ChatOpenAI + custom base_url because that is an explicit
    # architecture requirement, even though newer LangChain releases also expose a
    # dedicated ChatOpenRouter package.
    if "/" in name:
        api_key = _require_env("OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=name,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            max_retries=2,
            timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45")),
            # OpenRouter is an OpenAI-compatible Chat Completions endpoint.
            use_responses_api=False,
        )

    if name.startswith("gpt-"):
        api_key = _require_env("OPENAI_API_KEY")
        return ChatOpenAI(
            model=name,
            api_key=api_key,
            temperature=0,
            max_retries=2,
            timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45")),
        )

    if name.startswith("claude-"):
        api_key = _require_env("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=name,
            api_key=api_key,
            temperature=0,
            max_retries=2,
            timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45")),
        )

    if name.startswith("gemini-"):
        # ChatGoogleGenerativeAI reads GOOGLE_API_KEY / GEMINI_API_KEY itself, but
        # validating here produces a faster and more consistent startup/runtime error.
        api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY for the selected Gemini model."
            )
        return ChatGoogleGenerativeAI(
            model=name,
            api_key=api_key,
            temperature=0,
        )

    raise ValueError(
        "Unsupported model name. Use provider/model, hf:<repo-id>, or a gpt-/claude-/gemini- model."
    )
