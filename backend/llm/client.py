"""Unified LLM client supporting both SiliconFlow and Anthropic."""

from typing import Optional, Dict, Any, List
import json

from openai import OpenAI

from backend.config import settings

# 使用硅基流动的LLM
SILICONFLOW_MODELS = {
    "stepfun-ai/Step-3.5-Flash": "stepfun-ai/Step-3.5-Flash",
}

# Anthropic models
ANTHROPIC_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
}


class LLMClient:
    """Unified LLM client for SiliconFlow and Anthropic."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider or settings.llm_provider or "siliconflow"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        if self.provider in ("siliconflow", "custom", "openai", "openai-compatible"):
            resolved_api_key = self.api_key or settings.llm_api_key or settings.siliconflow_api_key
            if not resolved_api_key:
                raise ValueError("LLM API key not configured")
            self.client = OpenAI(
                api_key=resolved_api_key,
                base_url=self.base_url or settings.llm_base_url,
            )
            self.default_model = self.model or settings.llm_model or "deepseek-ai/DeepSeek-R1"
        elif self.provider == "anthropic":
            import anthropic
            resolved_api_key = self.api_key or settings.anthropic_api_key
            if not resolved_api_key:
                raise ValueError("Anthropic API key not configured")
            resolved_base_url = (self.base_url or settings.llm_base_url).strip() or None
            kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
            if resolved_base_url and resolved_base_url != "https://api.siliconflow.cn/v1":
                kwargs["base_url"] = resolved_base_url
            self.client = anthropic.Anthropic(**kwargs)
            self.default_model = self.model or settings.llm_model or "claude-haiku-4-5-20251001"
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat request and return the response text."""
        model = model or self.model or self.default_model

        # Map model aliases for OpenAI-compatible providers
        if (
            self.provider in ("siliconflow", "custom", "openai", "openai-compatible")
            and model.lower() in SILICONFLOW_MODELS
        ):
            model = SILICONFLOW_MODELS[model.lower()]

        # Map model aliases for Anthropic
        if self.provider == "anthropic" and model.lower() in ANTHROPIC_MODELS:
            model = ANTHROPIC_MODELS[model.lower()]

        if self.provider in ("siliconflow", "custom", "openai", "openai-compatible"):
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        else:
            # Anthropic
            # Convert messages to single prompt for Anthropic
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                else:
                    prompt += f"User: {content}\n"

            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text" and hasattr(block, "text")
            ]
            if text_parts:
                return "\n".join(text_parts)
            return ""

    def chat_simple(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Simple single-message chat."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
        )


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_llm_client_for(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """Create a non-cached LLM client for a specific use case."""
    return LLMClient(provider=provider, model=model, api_key=api_key, base_url=base_url)


def reset_llm_client() -> None:
    """Reset the global LLM client (useful for testing)."""
    global _llm_client
    _llm_client = None
