from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = ""
    log_level: str = "INFO"
    query_top_k_max: int = 20

    blob_read_write_token: str = ""
    blob_token_ttl_ms: int = 60 * 60 * 1000
    blob_max_pdf_bytes: int = 100 * 1024 * 1024
    blob_allowed_content_types: list[str] = [
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    ]

    hf_token: str = ""
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    pdf_chunk_size: int = 800
    pdf_chunk_overlap: int = 100

    hf_embed_batch_size: int = 32
    hf_embed_timeout: int = 60

    hf_llm_model: str = "meta-llama/Llama-3.2-1B-Instruct:novita"
    hf_llm_max_tokens: int = 400

    claude_token: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"
    claude_max_tokens: int = 1024

    llm_temperature: float = 0.2


settings = Settings()
_HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
_CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_API_VERSION = "2023-06-01"
