import logging
import tempfile
import urllib.request
from src.interfaces import DatabaseProtocol, EmbedderProtocol, ExtractorProtocol

class IngestionService:
    def __init__(self, db: DatabaseProtocol, embedder: EmbedderProtocol, extractor: ExtractorProtocol) -> None:
        self.db = db
        self._embedder = embedder
        self._extractor = extractor

    async def index_document(self, filename: str, blob_url: str) -> None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                urllib.request.urlretrieve(blob_url.replace(" ", "%20"), tmp.name)
                pages, indexes, texts, page_count = self._extractor.extract_chunks(tmp.name)

            if not texts:
                self.db.set_status(filename, "skipped", 0)
                return

            vectors = await self._embedder.embed(texts)
            self.db.delete_chunks(filename)
            self.db.save_chunks(filename, pages, indexes, texts, vectors)
            self.db.set_status(filename, "indexed", page_count)
            logging.info(f"Indexed {filename}: {page_count} pages, {len(texts)} chunks")

        except Exception:
            logging.exception(f"Indexing failed for {filename}")
            self.db.set_status(filename, "failed", 0)
