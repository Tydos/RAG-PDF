from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = ""
    log_level: str = "INFO"
    query_top_k_max: int = 20

    storage_backend: str = "supabase"  # "supabase" or "s3" (MinIO)

    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_bucket: str = "files"

    s3_endpoint: str = ""
    s3_public_url: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "files"

    hf_token: str = ""
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    pdf_chunk_size: int = 800
    pdf_chunk_overlap: int = 100

    hf_embed_batch_size: int = 32
    hf_embed_timeout: int = 60

    hf_llm_model: str = "meta-llama/Llama-3.2-1B-Instruct:featherless-ai"
    hf_llm_max_tokens: int = 400

    hf_judge_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    hf_judge_max_tokens: int = 200

    claude_token: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"
    claude_max_tokens: int = 1024

    llm_temperature: float = 0.2

    rerank_enabled: bool = True
    hf_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hf_rerank_timeout: int = 30
    rerank_candidate_multiplier: int = 4
    rerank_max_candidates: int = 25

    eval_gold_set_path: str = "tests/retriever-evaluation/gold_set.json"

    hf_chat_url: str = "https://router.huggingface.co/v1/chat/completions"
    claude_api_url: str = "https://api.anthropic.com/v1/messages"
    claude_api_version: str = "2023-06-01"

    @property
    def use_s3_storage(self) -> bool:
        return self.storage_backend.lower() == "s3" or bool(self.s3_endpoint.strip())


settings = Settings()
