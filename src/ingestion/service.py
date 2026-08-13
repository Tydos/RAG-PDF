import asyncio
import logging
import os
import tempfile

import httpx

from src.interfaces import DatabaseProtocol, EmbedderProtocol, ExtractorProtocol


class IngestionService:
    def __init__(self, db: DatabaseProtocol, embedder: EmbedderProtocol, extractor: ExtractorProtocol) -> None:
        self.db = db
        self._embedder = embedder
        self._extractor = extractor

    async def index_document(self, filename: str, blob_url: str) -> None:
        tmp_path = None
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.get(blob_url)
                response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            pages, indexes, texts, page_count = await asyncio.to_thread(
                self._extractor.extract_chunks, tmp_path
            )

            if not texts:
                self.db.set_status(filename, "skipped", 0)
                return

            vectors = await self._embedder.embed(texts)
            self.db.delete_chunks(filename)
            self.db.save_chunks(filename, pages, indexes, texts, vectors)
            self.db.set_status(filename, "indexed", page_count)
            logging.info("Indexed %s: %d pages, %d chunks", filename, page_count, len(texts))

        except Exception:
            logging.exception("Indexing failed for %s", filename)
            self.db.set_status(filename, "failed", 0)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
