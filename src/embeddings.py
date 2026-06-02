"""Embedding generation and FAISS vector store management.
Author: Rohit Thakur
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import config

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str = None) -> HuggingFaceEmbeddings:
    """Initialize the HuggingFace embedding model.

    Uses sentence-transformers/all-MiniLM-L6-v2 by default — a lightweight
    model that runs locally without any API key.

    Args:
        model_name: HuggingFace model ID for embeddings.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    model_name = model_name or config.embedding_model_name
    logger.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def create_vectorstore(
    chunks: List[Document],
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    persist_path: Optional[Path] = None,
) -> FAISS:
    """Create a FAISS vector store from document chunks.

    Args:
        chunks: List of chunked Document objects.
        embedding_model: Pre-initialized embedding model (creates one if None).
        persist_path: Directory to save the FAISS index. Uses config default if None.

    Returns:
        FAISS vector store instance.
    """
    if not chunks:
        raise ValueError("No chunks provided to create vector store.")

    embedding_model = embedding_model or get_embedding_model()
    persist_path = persist_path or config.vectorstore_dir

    logger.info(f"Creating FAISS vector store from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Persist to disk
    persist_path = Path(persist_path)
    if persist_path.exists():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))
    logger.info(f"Vector store saved to {persist_path}")

    return vectorstore


def load_vectorstore(
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    persist_path: Optional[Path] = None,
) -> FAISS:
    """Load an existing FAISS vector store from disk.

    Args:
        embedding_model: Pre-initialized embedding model.
        persist_path: Directory containing the saved FAISS index.

    Returns:
        FAISS vector store instance.
    """
    embedding_model = embedding_model or get_embedding_model()
    persist_path = persist_path or config.vectorstore_dir

    if not Path(persist_path).exists():
        raise FileNotFoundError(f"No vector store found at {persist_path}")

    logger.info(f"Loading vector store from {persist_path}")
    vectorstore = FAISS.load_local(
        str(persist_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
