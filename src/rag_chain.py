"""RAG chain setup — HuggingFace InferenceClient + optional Gemini fallback.
Orchestrates retrieval and generation for question answering.
Author: Rohit Thakur
"""

import logging
import os
from typing import Dict, Any, Optional, List

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_community.vectorstores import FAISS

from src.config import config

logger = logging.getLogger(__name__)

# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.
Use ONLY the following context to answer the question. If the answer is not in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""


class HuggingFaceInferenceLLM(LLM):
    """Thin LangChain wrapper around HuggingFace InferenceClient (chat API)."""

    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    api_token: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=self.api_token)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model": self.model}


def get_llm(
    model_name: str = None,
    hf_api_token: str = None,
    gemini_api_key: str = None,
    max_new_tokens: int = None,
    temperature: float = None,
):
    """Initialize the LLM — HuggingFace first, Gemini as fallback.

    Args:
        model_name: HuggingFace model repo ID.
        hf_api_token: HuggingFace API token.
        gemini_api_key: Google Gemini API key (optional fallback).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        LLM instance ready for inference.
    """
    hf_api_token = hf_api_token or config.hf_api_token
    gemini_api_key = gemini_api_key or config.gemini_api_key
    temperature = temperature if temperature is not None else config.temperature
    max_new_tokens = max_new_tokens or config.max_new_tokens

    # ── Primary: HuggingFace InferenceClient (chat completions) ──────────
    if hf_api_token:
        hf_model = model_name or "meta-llama/Llama-3.2-3B-Instruct"
        logger.info(f"Initializing HuggingFace LLM: {hf_model}")
        try:
            llm = HuggingFaceInferenceLLM(
                model=hf_model,
                api_token=hf_api_token,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            logger.info("HuggingFace LLM ready")
            return llm
        except Exception as e:
            logger.warning(f"HuggingFace LLM init failed: {e}")

    # ── Fallback: Gemini ──────────────────────────────────────────────────
    if gemini_api_key:
        logger.info("Trying Gemini 2.0 Flash as fallback...")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_api_key,
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            )
            logger.info("Gemini LLM ready")
            return llm
        except Exception as e:
            logger.warning(f"Gemini LLM init failed: {e}")

    raise ValueError(
        "Could not initialize any LLM. "
        "Please check HF_API_TOKEN or GEMINI_API_KEY in your .env file."
    )


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
