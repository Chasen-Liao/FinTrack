from pydantic_settings import BaseSettings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    polygon_api_key: str = ""
    anthropic_api_key: str = ""
    siliconflow_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "deepseek-ai/DeepSeek-R1"
    llm_provider: str = "siliconflow"  # options: "siliconflow", "custom", or "anthropic"
    layer1_llm_api_key: str = ""
    layer1_llm_base_url: str = ""
    layer1_llm_model: str = ""
    layer1_llm_provider: str = ""
    database_path: str = str(PROJECT_ROOT / "pokieticker.db")

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8"}


settings = Settings()
