"""RAG chain setup with HuggingFace LLM.
Orchestrates retrieval and generation for question answering.
Author: Rohit Thakur
"""

import logging
from typing import Dict, Any, Optional

from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from src.config import config

logger = logging.getLogger(__name__)

# RAG prompt template — instructs the LLM to answer ONLY from the context
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.
Use ONLY the following context to answer the question. If the answer is not found in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""


def get_llm(
    model_name: str = None,
    hf_api_token: str = None,
    max_new_tokens: int = None,
    temperature: float = None,
) -> HuggingFaceEndpoint:
    """Initialize the HuggingFace LLM via Inference API.

    Uses the free HuggingFace Inference API — requires an HF_API_TOKEN.

    Args:
        model_name: HuggingFace model repository ID.
        hf_api_token: HuggingFace API token.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        HuggingFaceEndpoint LLM instance.
    """
    model_name = model_name or config.llm_model_name
    hf_api_token = hf_api_token or config.hf_api_token
    max_new_tokens = max_new_tokens or config.max_new_tokens
    temperature = temperature if temperature is not None else config.temperature

    if not hf_api_token:
        raise ValueError(
            "HuggingFace API token not found. "
            "Set the HF_API_TOKEN environment variable."
        )

    logger.info(f"Initializing LLM: {model_name}")
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=hf_api_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=config.top_p,
    )
    return llm


def build_rag_chain(
    vectorstore: FAISS,
    llm: Optional[HuggingFaceEndpoint] = None,
    search_k: int = None,
) -> RetrievalQA:
    """Build the RAG (Retrieval-Augmented Generation) chain.

    Connects the FAISS retriever to the HuggingFace LLM with a
    custom prompt that grounds answers in the document context.

    Args:
        vectorstore: FAISS vector store with document embeddings.
        llm: Pre-initialized LLM instance.
        search_k: Number of relevant chunks to retrieve.

    Returns:
        RetrievalQA chain ready for querying.
    """
    llm = llm or get_llm()
    search_k = search_k or config.search_k

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": search_k},
    )

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    logger.info(f"RAG chain built (retriever k={search_k})")
    return rag_chain


def ask_question(rag_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """Query the RAG chain and return the answer with sources.

    Args:
        rag_chain: Initialized RetrievalQA chain.
        question: User's question string.

    Returns:
        Dict with 'answer' and 'source_documents' keys.
    """
    logger.info(f"Question: {question}")
    result = rag_chain.invoke({"query": question})
    answer = result.get("result", "").strip()
    sources = result.get("source_documents", [])
    logger.info(f"Answer length: {len(answer)} chars, Sources: {len(sources)}")
    return {"answer": answer, "source_documents": sources}
