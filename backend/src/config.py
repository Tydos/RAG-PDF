from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = ""
    log_level: str = "INFO"
    query_top_k_max: int = 20

    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_bucket: str = "files"

    hf_token: str = ""
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    pdf_chunk_size: int = 800
    pdf_chunk_overlap: int = 100

    hf_embed_batch_size: int = 32
    hf_embed_timeout: int = 60

    hf_llm_model: str = "meta-llama/Llama-3.2-1B-Instruct:featherless-ai"
    hf_llm_max_tokens: int = 400

    claude_token: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"
    claude_max_tokens: int = 1024

    llm_temperature: float = 0.2

    hf_chat_url: str = "https://router.huggingface.co/v1/chat/completions"
    claude_api_url: str = "https://api.anthropic.com/v1/messages"
    claude_api_version: str = "2023-06-01"


settings = Settings()
