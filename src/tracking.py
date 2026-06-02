"""MLflow experiment tracking for the RAG pipeline.
Tracks queries, retrieval metrics, and model performance.
Author: Rohit Thakur
"""

import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

import mlflow
from langchain.schema import Document

from src.config import config

logger = logging.getLogger(__name__)


def init_mlflow(
    tracking_uri: str = None,
    experiment_name: str = None,
) -> str:
    """Initialize MLflow tracking.

    Args:
        tracking_uri: MLflow tracking server URI or local path.
        experiment_name: Name for the MLflow experiment.

    Returns:
        The experiment ID.
    """
    tracking_uri = tracking_uri or config.mlflow_tracking_uri
    experiment_name = experiment_name or config.mlflow_experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        f"MLflow initialized — experiment: {experiment_name}, "
        f"ID: {experiment.experiment_id}"
    )
    return experiment.experiment_id


def log_ingestion(
    pdf_name: str,
    num_pages: int,
    num_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Log PDF ingestion metrics to MLflow."""
    with mlflow.start_run(run_name=f"ingest_{pdf_name}"):
        mlflow.log_params({
            "pdf_name": pdf_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })
        mlflow.log_metrics({
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "avg_chunk_size": chunk_size,
        })
        mlflow.set_tag("stage", "ingestion")
    logger.info(f"Logged ingestion run for {pdf_name}")


def log_query(
    question: str,
    answer: str,
    source_documents: List[Document],
    latency_seconds: float,
    model_name: str = None,
) -> None:
    """Log a single Q&A query to MLflow."""
    model_name = model_name or config.llm_model_name

    with mlflow.start_run(run_name="query"):
        mlflow.log_params({
            "model_name": model_name,
            "search_k": config.search_k,
            "temperature": config.temperature,
        })
        mlflow.log_metrics({
            "response_latency_sec": round(latency_seconds, 3),
            "answer_length": len(answer),
            "num_sources_retrieved": len(source_documents),
        })
        mlflow.set_tag("stage", "query")

        # Log the Q&A as a text artifact
        qa_text = f"Q: {question}\n\nA: {answer}\n\nSources: {len(source_documents)}"
        mlflow.log_text(qa_text, "qa_pair.txt")

    logger.info(f"Logged query run (latency={latency_seconds:.2f}s)")


def log_pipeline_config() -> None:
    """Log the full pipeline configuration to MLflow."""
    with mlflow.start_run(run_name="pipeline_config"):
        mlflow.log_params({
            "embedding_model": config.embedding_model_name,
            "llm_model": config.llm_model_name,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "search_k": config.search_k,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        })
        mlflow.set_tag("stage", "config")
    logger.info("Logged pipeline configuration")
