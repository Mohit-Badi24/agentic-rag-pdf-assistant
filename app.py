import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

load_dotenv()

DOCS_DIR = Path("./docs_file")

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .app-subtitle {
        color: #808080;
        margin-top: -12px;
        margin-bottom: 25px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def load_and_store_pdf(pdf_path):
    loader = PyPDFDirectoryLoader(str(pdf_path))
    docs = loader.load()

    if not docs:
        raise ValueError("No readable text found in the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = InMemoryVectorStore.from_documents(
        split_docs, embeddings
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    @tool
    def retrieve_info(query: str) -> str:
        """Retrieve relevant information from the uploaded PDF documents."""
        results = vectorstore.similarity_search(query, k=3)

        return "\n\n".join(
            f"Page {doc.metadata.get('page', 0) + 1}: "
            f"{doc.page_content}"
            for doc in results
        )

    system_prompt = """
    You are an assistant that answers questions about the uploaded PDFs.
    Use retrieve_info to find relevant information before answering
    document-related questions. Base answers on the retrieved content
    and do not invent facts. If the information is unavailable, say so.
    If the question is unrelated to the documents, politely explain
    that you only answer questions about the uploaded PDFs.
    """

    agent = create_agent(
        model=llm,
        tools=[retrieve_info],
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


with st.sidebar:
    st.title("📄 Agentic RAG")
    st.caption("Your personal PDF question-answering assistant")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("Please configure GROQ_API_KEY in your .env file.")
        else:
            try:
                with st.spinner("Processing your documents..."):
                    DOCS_DIR.mkdir(parents=True, exist_ok=True)

                    for old_file in DOCS_DIR.glob("*.pdf"):
                        old_file.unlink()

                    for pdf in uploaded_files:
                        safe_name = Path(pdf.name).name
                        with open(DOCS_DIR / safe_name, "wb") as file:
                            file.write(pdf.getvalue())

                    st.session_state.agent = None
                    st.session_state.document_uploaded = False

                    load_and_store_pdf(DOCS_DIR)

                    st.session_state.uploaded_files = [
                        pdf.name for pdf in uploaded_files
                    ]
                    st.session_state.messages = []
                    st.session_state.thread_id = str(uuid.uuid4())

                st.success("Documents processed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Could not process documents: {e}")

    st.divider()

    if st.session_state.document_uploaded:
        st.success("Documents ready")

        st.caption("Loaded documents")
        for filename in st.session_state.uploaded_files:
            st.write(f"📄 {filename}")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


st.title("Ask Your Documents")
st.markdown(
    '<p class="app-subtitle">'
    'Upload PDFs and ask questions using AI-powered document retrieval.'
    '</p>',
    unsafe_allow_html=True
)

if not st.session_state.document_uploaded:
    st.info("👈 Upload your PDF documents from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📚 Upload")
        st.caption("Add one or more PDF documents.")

    with col2:
        st.subheader("🔍 Retrieve")
        st.caption("Find relevant information from your documents.")

    with col3:
        st.subheader("💬 Ask")
        st.caption("Get answers grounded in your PDFs.")

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                {"role": "user", "content": query}
                            ]
                        },
                        config={
                            "configurable": {
                                "thread_id": st.session_state.thread_id
                            }
                        }
                    )

                    answer = response["messages"][-1].content
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")