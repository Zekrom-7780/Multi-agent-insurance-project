"""Tests for the RAG service with a temporary ChromaDB collection.

NOTE: ChromaDB's default embedding model (onnxruntime) may crash on some
Windows/Python combinations. Tests that require embedding are marked with
xfail so the rest of the suite is not affected.
"""

import tempfile
import sys

import pytest

from app.services.rag import RAGService, FAQResult

_CHROMADB_WINDOWS_ISSUE = sys.platform == "win32"


@pytest.fixture
def temp_rag():
    with tempfile.TemporaryDirectory() as tmpdir:
        svc = RAGService(persist_dir=tmpdir, collection_name="test_faq")
        svc.init()

        svc._collection.add(
            documents=[
                "Question: What does life insurance cover?\nAnswer: Life insurance pays a death benefit.",
                "Question: How do I file a claim?\nAnswer: Contact your agent to file a claim.",
                "Question: What is a deductible?\nAnswer: A deductible is the amount you pay before insurance kicks in.",
            ],
            metadatas=[
                {"question": "What does life insurance cover?", "answer": "Life insurance pays a death benefit."},
                {"question": "How do I file a claim?", "answer": "Contact your agent to file a claim."},
                {"question": "What is a deductible?", "answer": "A deductible is the amount you pay before insurance kicks in."},
            ],
            ids=["1", "2", "3"],
        )
        yield svc


@pytest.mark.xfail(
    _CHROMADB_WINDOWS_ISSUE,
    reason="ChromaDB default embedder may crash on Windows",
    strict=False,
)
def test_retrieve_returns_results(temp_rag: RAGService):
    results = temp_rag.retrieve("life insurance coverage", n_results=2)
    assert len(results) == 2
    assert "life insurance" in results[0].question.lower()


@pytest.mark.xfail(
    _CHROMADB_WINDOWS_ISSUE,
    reason="ChromaDB default embedder may crash on Windows",
    strict=False,
)
def test_retrieve_returns_distances(temp_rag: RAGService):
    results = temp_rag.retrieve("deductible", n_results=1)
    assert len(results) == 1
    assert results[0].distance >= 0


@pytest.mark.xfail(
    _CHROMADB_WINDOWS_ISSUE,
    reason="ChromaDB default embedder may crash on Windows",
    strict=False,
)
def test_format_for_prompt(temp_rag: RAGService):
    results = temp_rag.retrieve("claim", n_results=2)
    formatted = temp_rag.format_for_prompt(results)
    assert "FAQ 1" in formatted
    assert "FAQ 2" in formatted


def test_format_empty():
    svc = RAGService()
    formatted = svc.format_for_prompt([])
    assert "No relevant FAQs" in formatted


def test_faq_result_dataclass():
    faq = FAQResult(question="Q?", answer="A.", distance=0.5)
    assert faq.question == "Q?"
    assert faq.distance == 0.5


def test_format_for_prompt_with_results():
    svc = RAGService()
    results = [
        FAQResult(question="What is a deductible?", answer="Amount you pay first.", distance=0.3),
        FAQResult(question="How to file?", answer="Call your agent.", distance=0.5),
    ]
    formatted = svc.format_for_prompt(results)
    assert "FAQ 1" in formatted
    assert "FAQ 2" in formatted
    assert "deductible" in formatted.lower()
