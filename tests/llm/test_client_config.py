from backend.config import settings
import backend.llm.client as client_module
from backend.llm.client import LLMClient


class FakeOpenAI:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)


def test_custom_openai_compatible_config_uses_env_settings(monkeypatch):
    FakeOpenAI.calls = []
    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "siliconflow_api_key", "")
    monkeypatch.setattr(settings, "llm_base_url", "https://example.com/v1")
    monkeypatch.setattr(settings, "llm_model", "custom-model")

    client = LLMClient()

    assert FakeOpenAI.calls == [{"api_key": "test-key", "base_url": "https://example.com/v1"}]
    assert client.default_model == "custom-model"


def test_siliconflow_key_is_backwards_compatible(monkeypatch):
    FakeOpenAI.calls = []
    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(settings, "llm_provider", "siliconflow")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "siliconflow_api_key", "siliconflow-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")
    monkeypatch.setattr(settings, "llm_model", "deepseek-ai/DeepSeek-R1")

    LLMClient()

    assert FakeOpenAI.calls == [
        {"api_key": "siliconflow-key", "base_url": "https://api.siliconflow.cn/v1"}
    ]
