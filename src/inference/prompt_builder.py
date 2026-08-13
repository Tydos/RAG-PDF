class PromptBuilder:
    SYSTEM_PROMPT = (
        "You are a retrieval-augmented assistant answering questions about the user's PDFs. "
        "Answer strictly from the provided context. If the answer is not in the context, "
        "say you don't know. Cite sources inline as [filename p.N] where N is the page number. "
        "Keep answers concise and faithful to the source material."
    )

    @staticmethod
    def format_chunks(chunks: list[dict]) -> str:
        blocks: list[str] = []
        for i, c in enumerate(chunks, 1):
            page = c.get("page", "?")
            blocks.append(
                f"[{i}] [{c.get('filename', 'unknown')} p.{page}]\n{c.get('content', '').strip()}"
            )
        return "\n\n".join(blocks)

    @classmethod
    def build_messages(cls, question: str, chunks: list[dict], history: list[dict]) -> list[dict]:
        context = cls.format_chunks(chunks)
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            *[{"role": m["role"], "content": m["content"]} for m in history[-6:]],
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
