import os

# Must be set before torch / tokenizers / sentence-transformers are imported anywhere.
# Works around a macOS-specific crash/hang where PyTorch operations run from a
# background thread (which is what Streamlit's script runner uses).
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import shutil
import tempfile

import streamlit as st
import torch
from dotenv import load_dotenv

torch.set_num_threads(1)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="RAG Book Chat", page_icon="📚", layout="wide")

PERSIST_DIR = "Chroma-VDB"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_REPO_ID = "deepseek-ai/DeepSeek-R1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

SYSTEM_PROMPT = """You are a helpful AI assistant.
Use only the provided context to answer the question dont add more of extra things just explain it well.
If the answer is not present in the context, say: 'I could not find the answer in the document'"""


# --------------------------------------------------------------------------
# Cached resources (created once, reused across reruns)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_llm() -> ChatHuggingFace:
    llm_endpoint = HuggingFaceEndpoint(repo_id=LLM_REPO_ID)
    return ChatHuggingFace(llm=llm_endpoint)


@st.cache_resource(show_spinner=False)
def get_vectorstore(_embedding_model: HuggingFaceEmbeddings) -> Chroma:
    # Cached so every rerun reuses the same Chroma connection instead of reopening it.
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embedding_model)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context: {context}\nQuestion: {question}"),
    ]
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def db_has_content() -> bool:
    return (
        os.path.isdir(PERSIST_DIR)
        and any(os.scandir(PERSIST_DIR))
    )


def process_book(uploaded_file, chunk_size: int, chunk_overlap: int) -> int:
    """Load an uploaded PDF, split it into chunks, and add it to the vector DB.

    Returns the number of chunks added.
    """
    embedding_model = load_embedding_model()

    # PyPDFLoader needs a real file path, so the upload is written to a temp file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.remove(tmp_path)

    # Keep the real file name in the metadata instead of the temp path.
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    vectorstore = get_vectorstore(embedding_model)
    vectorstore.add_documents(chunks)

    return len(chunks)


def reset_database() -> None:
    get_vectorstore.clear()
    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    st.session_state.uploaded_books = []
    st.session_state.chat_history = []


def get_retriever(k: int, fetch_k: int, lambda_mult: float):
    embedding_model = load_embedding_model()
    vectorstore = get_vectorstore(embedding_model)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def answer_question(query: str, k: int, fetch_k: int, lambda_mult: float):
    retriever = get_retriever(k, fetch_k, lambda_mult)
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.invoke({"context": context, "question": query})
    llm = load_llm()
    response = llm.invoke(final_prompt)
    return response.content, docs


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "uploaded_books" not in st.session_state:
    st.session_state.uploaded_books = []  # names of books added this session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "sources"}

# --------------------------------------------------------------------------
# Sidebar: upload books + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Your library")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF books",
        type=["pdf"],
        accept_multiple_files=True,
    )

    with st.expander("Chunking options"):
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=CHUNK_SIZE, step=100)
        chunk_overlap = st.number_input(
            "Chunk overlap", min_value=0, max_value=1000, value=CHUNK_OVERLAP, step=50
        )

    if st.button("Add to library", type="primary", use_container_width=True, disabled=not uploaded_files):
        progress = st.progress(0.0, text="Starting...")
        total = len(uploaded_files)
        for i, file in enumerate(uploaded_files, start=1):
            progress.progress((i - 1) / total, text=f"Processing {file.name}...")
            try:
                n_chunks = process_book(file, chunk_size, chunk_overlap)
                st.session_state.uploaded_books.append(file.name)
                st.toast(f"Added **{file.name}** ({n_chunks} chunks)", icon="✅")
            except Exception as exc:
                st.error(f"Failed to process **{file.name}**: {exc}")
            progress.progress(i / total, text=f"Processed {file.name}")
        progress.empty()

    st.divider()

    if st.session_state.uploaded_books:
        st.caption("Added this session:")
        for name in st.session_state.uploaded_books:
            st.markdown(f"- {name}")
    elif db_has_content():
        st.caption("A vector database already exists on disk from a previous session.")
    else:
        st.caption("No books added yet. Upload a PDF to get started.")

    st.divider()
    with st.expander("Retriever settings"):
        k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)
        fetch_k = st.slider("Candidates to consider (fetch_k)", min_value=k, max_value=30, value=max(10, k))
        lambda_mult = st.slider("Diversity (lambda_mult)", min_value=0.0, max_value=1.0, value=0.5)

    st.divider()
    col_clear_chat, col_reset_db = st.columns(2)
    if col_clear_chat.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if col_reset_db.button("Reset library", use_container_width=True):
        reset_database()
        st.rerun()

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN") and not os.getenv("HF_TOKEN"):
        st.warning("No Hugging Face token found. Add `HUGGINGFACEHUB_API_TOKEN` to your `.env` file.")

# --------------------------------------------------------------------------
# Main area: chat
# --------------------------------------------------------------------------
st.title("📁 Chat with your 📄Documents")
st.caption(f"Answers come only from the PDFs you've added. Model: `{LLM_REPO_ID}`")

if not db_has_content():
    st.info("Upload a PDF from the sidebar and select **Add to library** to get started.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, doc in enumerate(msg["sources"], start=1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page")
                    label = f"**{i}. {source}**" + (f" — page {page + 1}" if page is not None else "")
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

if query := st.chat_input("Ask a question about your books..."):
    if not db_has_content():
        st.warning("Add at least one book before asking a question.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your books..."):
                    answer, sources = answer_question(query, k, fetch_k, lambda_mult)
            except Exception as exc:
                st.session_state.chat_history.pop()
                st.error(
                    "The request failed. Check your Hugging Face token and internet "
                    "connection, then send your question again."
                )
                with st.expander("Error details"):
                    st.code(str(exc))
                st.stop()

            st.markdown(answer)
            with st.expander("Sources"):
                for i, doc in enumerate(sources, start=1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page")
                    label = f"**{i}. {source}**" + (f" — page {page + 1}" if page is not None else "")
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
