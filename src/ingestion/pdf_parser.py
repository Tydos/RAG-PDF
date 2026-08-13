from pypdf import PdfReader
from src.config import settings

class PDFParser:
    def _split_into_chunks(self, text: str) -> list[str]:
        step = max(1, settings.pdf_chunk_size - settings.pdf_chunk_overlap)
        chunks = []
        i = 0
        while i < len(text):
            piece = text[i : i + settings.pdf_chunk_size].strip()
            if piece:
                chunks.append(piece)
            i += step
        return chunks

    def extract_chunks(self, file_path: str) -> tuple[list[int], list[int], list[str], int]:
        reader = PdfReader(file_path)
        pages: list[int] = []
        indexes: list[int] = []
        texts: list[str] = []
        page_count = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            page_count += 1
            for chunk_idx, chunk in enumerate(self._split_into_chunks(page_text)):
                pages.append(page_number)
                indexes.append(chunk_idx)
                texts.append(chunk)

        return pages, indexes, texts, page_count
