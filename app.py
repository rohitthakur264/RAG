"""PDF Q&A RAG Chatbot — Streamlit Application.
A production-grade RAG chatbot that lets users upload PDFs and ask questions.
Built with LangChain, FAISS, HuggingFace, and MLflow.

Author: Rohit Thakur
Project: IBM RAG Certification Capstone
"""

import os
import time
import logging
import tempfile
from pathlib import Path

import streamlit as st

from src.config import config
from src.ingestion import ingest_pdf, load_pdf, chunk_documents
from src.embeddings import get_embedding_model, create_vectorstore, load_vectorstore
from src.rag_chain import build_rag_chain, ask_question, get_llm
from src.tracking import init_mlflow, log_ingestion, log_query

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Chatbot — Rohit Thakur",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #6C63FF;
    --primary-dark: #5A52D5;
    --accent: #00D2FF;
    --bg-dark: #0E1117;
    --bg-card: #1A1D29;
    --bg-card-hover: #232738;
    --text-primary: #FFFFFF;
    --text-secondary: #8B8FA3;
    --border: #2A2D3E;
    --success: #00C48C;
    --warning: #FFB020;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.app-header h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 2.5rem;
    background: linear-gradient(135deg, #6C63FF, #00D2FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.app-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-family: 'Inter', sans-serif;
}

/* Status cards */
.status-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card-hover));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    font-family: 'Inter', sans-serif;
}

.status-card.success {
    border-left: 4px solid var(--success);
}

.status-card.info {
    border-left: 4px solid var(--primary);
}

/* Chat messages */
.chat-message {
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-message.user {
    background: linear-gradient(135deg, #6C63FF22, #6C63FF11);
    border: 1px solid #6C63FF44;
    margin-left: 2rem;
}

.chat-message.assistant {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card-hover));
    border: 1px solid var(--border);
    margin-right: 2rem;
}

/* Source documents expander */
.source-doc {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117, #1A1D29);
}

.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}

/* Metrics row */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
}

.metric-card .label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: var(--text-secondary);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "vectorstore": None,
        "rag_chain": None,
        "pdf_name": None,
        "num_pages": 0,
        "num_chunks": 0,
        "embedding_model": None,
        "processing": False,
        "mlflow_initialized": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ── MLflow Init ──────────────────────────────────────────────────────────────
if not st.session_state.mlflow_initialized:
    try:
        init_mlflow()
        st.session_state.mlflow_initialized = True
    except Exception as e:
        logger.warning(f"MLflow init failed (non-critical): {e}")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>📄 PDF Q&A Chatbot</h1>
    <p>Upload a PDF document and ask questions — powered by RAG</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to start asking questions.",
        key="pdf_uploader",
    )

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        process_btn = st.button(
            "🚀 Process Document",
            use_container_width=True,
            type="primary",
        )

        if process_btn:
            st.session_state.processing = True

            with st.status("Processing PDF...", expanded=True) as status:
                try:
                    # Step 1: Save uploaded file
                    st.write("📥 Saving uploaded file...")
                    save_path = config.upload_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Step 2: Load and chunk
                    st.write("📖 Loading and chunking PDF...")
                    documents = load_pdf(save_path)
                    chunks = chunk_documents(documents)
                    st.session_state.num_pages = len(documents)
                    st.session_state.num_chunks = len(chunks)

                    # Step 3: Create embeddings and vector store
                    st.write("🧠 Generating embeddings...")
                    if st.session_state.embedding_model is None:
                        st.session_state.embedding_model = get_embedding_model()
                    vectorstore = create_vectorstore(
                        chunks, st.session_state.embedding_model
                    )
                    st.session_state.vectorstore = vectorstore

                    # Step 4: Build RAG chain
                    st.write("⛓️ Building RAG chain...")
                    llm = get_llm()
                    rag_chain = build_rag_chain(vectorstore, llm)
                    st.session_state.rag_chain = rag_chain

                    # Step 5: Update state
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []

                    # Step 6: Log to MLflow
                    if st.session_state.mlflow_initialized:
                        log_ingestion(
                            pdf_name=uploaded_file.name,
                            num_pages=len(documents),
                            num_chunks=len(chunks),
                            chunk_size=config.chunk_size,
                            chunk_overlap=config.chunk_overlap,
                        )

                    status.update(
                        label="✅ Document processed successfully!",
                        state="complete",
                    )

                except Exception as e:
                    status.update(label=f"❌ Error: {str(e)}", state="error")
                    logger.error(f"Processing error: {e}", exc_info=True)

                finally:
                    st.session_state.processing = False

    # ── Document Info ────────────────────────────────────────────────────
    if st.session_state.pdf_name:
        st.markdown("---")
        st.markdown("### 📊 Document Info")
        st.markdown(f"""
        <div class="status-card success">
            <strong>📄 {st.session_state.pdf_name}</strong><br>
            <span style="color: var(--text-secondary);">
                {st.session_state.num_pages} pages · {st.session_state.num_chunks} chunks
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.num_pages)
        col2.metric("Chunks", st.session_state.num_chunks)

    # ── Settings ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    with st.expander("Model Configuration"):
        st.text_input(
            "Embedding Model",
            value=config.embedding_model_name,
            disabled=True,
        )
        st.text_input(
            "LLM Model",
            value=config.llm_model_name,
            disabled=True,
        )
        st.slider(
            "Retrieved Chunks (k)",
            min_value=1, max_value=10,
            value=config.search_k,
            disabled=True,
        )

    # ── About ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 👤 About")
    st.markdown("""
    **Built by Rohit Thakur**

    RAG pipeline using LangChain, FAISS,
    HuggingFace, and MLflow.

    *IBM RAG Certification Project*
    """)


# ── Main Chat Interface ──────────────────────────────────────────────────────
if not st.session_state.pdf_name:
    # Welcome state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
        <h2 style="color: #8B8FA3; font-family: 'Inter', sans-serif; font-weight: 400;">
            Upload a PDF to get started
        </h2>
        <p style="color: #5A5E72; font-family: 'Inter', sans-serif;">
            Use the sidebar to upload a document, then ask any question about its contents.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(content)
            if role == "assistant" and "sources" in msg:
                with st.expander(f"📑 Sources ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"]):
                        page = src.metadata.get("page", "?")
                        st.markdown(
                            f"<div class='source-doc'>"
                            f"<strong>Chunk {i+1}</strong> (Page {page})<br>"
                            f"{src.page_content[:300]}..."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your document...",
        key="chat_input",
    ):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                try:
                    result = ask_question(st.session_state.rag_chain, prompt)
                    latency = time.time() - start_time
                    answer = result["answer"]
                    sources = result["source_documents"]

                    st.markdown(answer)

                    # Show sources
                    if sources:
                        with st.expander(
                            f"📑 Sources ({len(sources)} chunks)"
                        ):
                            for i, src in enumerate(sources):
                                page = src.metadata.get("page", "?")
                                st.markdown(
                                    f"<div class='source-doc'>"
                                    f"<strong>Chunk {i+1}</strong> (Page {page})<br>"
                                    f"{src.page_content[:300]}..."
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # Show latency
                    st.caption(f"⏱️ Response time: {latency:.2f}s")

                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency": latency,
                    })

                    # Log to MLflow
                    if st.session_state.mlflow_initialized:
                        log_query(
                            question=prompt,
                            answer=answer,
                            source_documents=sources,
                            latency_seconds=latency,
                        )

                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by <strong>Rohit Thakur</strong> · 
    LangChain · FAISS · HuggingFace · MLflow · Streamlit<br>
    <em>IBM RAG Certification Capstone Project</em>
</div>
""", unsafe_allow_html=True)
