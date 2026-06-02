"""PDF ingestion and text chunking module.
Handles loading PDFs and splitting them into semantically meaningful chunks.
Author: Rohit Thakur
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import config

logger = logging.getLogger(__name__)


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF file and return a list of Document objects (one per page).

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects with page content and metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {file_path.suffix}")

    logger.info(f"Loading PDF: {file_path.name}")
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on
    paragraphs, then sentences, then words — preserving semantic meaning.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def ingest_pdf(file_path: str | Path) -> List[Document]:
    """End-to-end: load a PDF and return chunked documents.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)
    return chunks
