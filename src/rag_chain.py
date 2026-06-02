"""RAG chain setup — supports Gemini (primary) and HuggingFace (fallback).
Orchestrates retrieval and generation for question answering.
Author: Rohit Thakur
"""

import logging
import os
from typing import Dict, Any, Optional

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
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
    gemini_api_key: str = None,
    max_new_tokens: int = None,
    temperature: float = None,
):
    """Initialize the LLM — tries Gemini first, then HuggingFace.

    Args:
        model_name: Model name (used for HuggingFace; Gemini always uses gemini-1.5-flash).
        hf_api_token: HuggingFace API token.
        gemini_api_key: Google Gemini API key.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        LLM instance (ChatGoogleGenerativeAI or HuggingFaceEndpoint).
    """
    hf_api_token = hf_api_token or config.hf_api_token
    gemini_api_key = gemini_api_key or config.gemini_api_key
    temperature = temperature if temperature is not None else config.temperature
    max_new_tokens = max_new_tokens or config.max_new_tokens

    # ── Try Gemini first ──────────────────────────────────────────────────
    if gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("Initializing Gemini 2.0 Flash LLM")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_api_key,
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            )
            return llm
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                raise ValueError(
                    "Gemini API rate limit reached (free tier). "
                    "Please wait a minute and try again, or check your quota at "
                    "https://ai.dev/rate-limit"
                )
            logger.warning(f"Gemini init failed: {e}. Falling back to HuggingFace.")

    # ── Fallback: HuggingFace Inference API ──────────────────────────────
    if not hf_api_token:
        raise ValueError(
            "No API key found. Please set HF_API_TOKEN or GEMINI_API_KEY in your .env file."
        )

    hf_model = model_name or "HuggingFaceH4/zephyr-7b-beta"
    logger.info(f"Initializing HuggingFace LLM: {hf_model}")

    from langchain_huggingface import HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=hf_model,
        huggingfacehub_api_token=hf_api_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=config.top_p,
        # Let HF auto-select the best available provider
        task="text-generation",
    )
    return llm


def build_rag_chain(
    vectorstore: FAISS,
    llm=None,
    search_k: int = None,
) -> RetrievalQA:
    """Build the RAG (Retrieval-Augmented Generation) chain.

    Connects the FAISS retriever to the LLM with a custom prompt
    that grounds answers in the document context.

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
