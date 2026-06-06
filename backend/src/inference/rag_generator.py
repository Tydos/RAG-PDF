import logging

from src.config import settings
from src.inference.llm_adapters import ClaudeAdapter, LLMAdapter
from src.inference.prompt_builder import PromptBuilder

_NO_CONTEXT_ANSWER = "I couldn't find anything relevant in the indexed documents."


class LLMResponseGenerator:
    def __init__(self, llm: LLMAdapter, max_tokens: int, temperature: float) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate(self, question: str, chunks: list[dict], history: list[dict]) -> str:
        if not chunks:
            return _NO_CONTEXT_ANSWER

        messages = PromptBuilder.build_messages(question, chunks, history)

        try:
            text = await self._llm.generate(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception:
            logging.exception("LLM inference failed")
            return _NO_CONTEXT_ANSWER

        if not text:
            logging.warning("LLM returned an empty response")
            return _NO_CONTEXT_ANSWER
        return text


def create_rag_generator() -> LLMResponseGenerator:
    if settings.claude_token:
        llm = ClaudeAdapter(model=settings.claude_model, token=settings.claude_token)
        max_tokens = settings.claude_max_tokens
        logging.info("Using Claude adapter (model: %s)", settings.claude_model)
    else:
        from src.inference.llm_adapters import HuggingFaceAdapter

        llm = HuggingFaceAdapter(model=settings.hf_llm_model, token=settings.hf_token)
        max_tokens = settings.hf_llm_max_tokens
        logging.info("Using HuggingFace adapter (model: %s)", settings.hf_llm_model)

    return LLMResponseGenerator(
        llm=llm, max_tokens=max_tokens, temperature=settings.llm_temperature
    )
