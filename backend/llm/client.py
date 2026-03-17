"""Unified LLM client supporting both SiliconFlow and Anthropic."""

from typing import Optional, Dict, Any, List
import json

from openai import OpenAI

from backend.config import settings

# 使用硅基流动的LLM
SILICONFLOW_MODELS = {
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    # "qwq-32b": "Qwen/QwQ-32B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
}

# Anthropic models
ANTHROPIC_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
}


class LLMClient:
    """Unified LLM client for SiliconFlow and Anthropic."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.llm_provider or "siliconflow"
        self.model = model

        if self.provider == "siliconflow":
            if not settings.siliconflow_api_key:
                raise ValueError("SiliconFlow API key not configured")
            self.client = OpenAI(
                api_key=settings.siliconflow_api_key,
                base_url="https://api.siliconflow.cn/v1",
            )
            self.default_model = "deepseek-ai/DeepSeek-R1"
        elif self.provider == "anthropic":
            import anthropic
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.default_model = "claude-haiku-4-5-20251001"
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

        # Map model aliases for SiliconFlow
        if self.provider == "siliconflow" and model.lower() in SILICONFLOW_MODELS:
            model = SILICONFLOW_MODELS[model.lower()]

        # Map model aliases for Anthropic
        if self.provider == "anthropic" and model.lower() in ANTHROPIC_MODELS:
            model = ANTHROPIC_MODELS[model.lower()]

        if self.provider == "siliconflow":
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
            return response.content[0].text if response.content else ""

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


def reset_llm_client() -> None:
    """Reset the global LLM client (useful for testing)."""
    global _llm_client
    _llm_client = None