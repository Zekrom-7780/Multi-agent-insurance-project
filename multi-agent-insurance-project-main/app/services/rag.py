"""ChromaDB-backed RAG service for FAQ retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FAQResult:
    question: str
    answer: str
    distance: float


class RAGService:
    """Retrieves relevant FAQs from a ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._collection_name = collection_name or settings.CHROMA_COLLECTION
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def init(self) -> None:
        """Create the ChromaDB client and get the collection."""
        try:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
            logger.info(
                "RAG service initialised: %s (%d docs)",
                self._collection_name,
                self._collection.count(),
            )
        except Exception:
            logger.warning(
                "RAG service unavailable — ChromaDB failed to initialise",
                exc_info=True,
            )
            self._collection = None

    def retrieve(self, query: str, n_results: int = 3) -> list[FAQResult]:
        """Query the collection and return ranked FAQ results."""
        if self._collection is None:
            logger.warning("RAG collection not initialised")
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
        except Exception:
            logger.warning("RAG query failed (embedding runtime unavailable)", exc_info=True)
            return []

        faqs: list[FAQResult] = []
        if results and results.get("metadatas") and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                faqs.append(
                    FAQResult(
                        question=meta.get("question", ""),
                        answer=meta.get("answer", ""),
                        distance=results["distances"][0][i],
                    )
                )
        return faqs

    def format_for_prompt(self, results: list[FAQResult]) -> str:
        """Render FAQ results as text for prompt injection."""
        if not results:
            return "No relevant FAQs found."
        lines: list[str] = []
        for i, faq in enumerate(results, 1):
            lines.append(
                f"FAQ {i} (score: {faq.distance:.3f})\n"
                f"Q: {faq.question}\n"
                f"A: {faq.answer}"
            )
        return "\n\n".join(lines)
