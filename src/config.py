"""Configuration settings for the PDF Q&A RAG Chatbot.
Author: Rohit Thakur
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


@dataclass
class AppConfig:
    """Central configuration for the RAG pipeline."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    upload_dir: Path = field(default=None)
    vectorstore_dir: Path = field(default=None)

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding model (HuggingFace - runs locally, no API key needed)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM (HuggingFace Inference API - free tier)
    llm_model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    hf_api_token: str = field(default_factory=lambda: os.getenv("HF_API_TOKEN", ""))
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.95

    # Retriever
    search_k: int = 4

    # MLflow
    mlflow_tracking_uri: str = "mlflow_runs"
    mlflow_experiment_name: str = "pdf-qa-rag-chatbot"

    def __post_init__(self):
        if self.upload_dir is None:
            self.upload_dir = self.project_root / "data" / "uploads"
        if self.vectorstore_dir is None:
            self.vectorstore_dir = self.project_root / "data" / "vectorstore"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()
