# Agentic RAG PDF Assistant

An AI-powered PDF question-answering application built with Streamlit, LangChain, LangGraph, and Groq. Users can upload multiple PDF documents and ask questions about their contents using an AI agent equipped with a document retrieval tool.

## Features

- Upload and process multiple PDF documents.
- Extract and split document text into smaller chunks.
- Generate embeddings using Hugging Face Sentence Transformers.
- Retrieve relevant document chunks using semantic similarity search.
- Use a LangChain agent with a retrieval tool to answer questions.
- Maintain conversational state during a session.
- Chat interface built with Streamlit.
- Display uploaded document names in the sidebar.

## Tech Stack

- **Python** — application logic
- **Streamlit** — user interface
- **LangChain** — LLM integration and retrieval tools
- **LangGraph** — agent execution and checkpointing
- **Groq** — language model inference
- **Hugging Face Sentence Transformers** — text embeddings
- **InMemoryVectorStore** — in-memory vector storage
- **PyPDF** — PDF text extraction

## Project Structure

```text
agentic-rag-pdf/
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
└── docs_file/
```

The `docs_file/` directory is created automatically when documents are processed.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd agentic-rag-pdf
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own Groq API key. Ensure the configured model is available to your account.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can access the application in your browser.

## How It Works

1. **Document upload:** The user uploads one or more PDF files.
2. **Text extraction:** PyPDF extracts the document text.
3. **Text splitting:** Recursive Character Text Splitter divides the text into chunks with overlapping context.
4. **Embedding generation:** Sentence Transformers converts the chunks into vector representations.
5. **Vector storage:** The embeddings and document chunks are stored in an in-memory vector store.
6. **Agent execution:** The user submits a question to the LangChain agent.
7. **Tool-based retrieval:** The agent can call the retrieval tool to search for relevant document chunks.
8. **Answer generation:** The language model generates a response based on the retrieved information.

## Limitations

- The vector store is in memory and must be rebuilt after an application restart.
- PDF extraction quality depends on the document. Scanned PDFs may require OCR.
- The current implementation does not provide durable chat-history storage.
- The agent may occasionally generate inaccurate answers, so important information should be verified.
- This implementation is an agent-based RAG prototype; it does not yet include advanced planning, evaluation, or multi-agent workflows.

## Future Improvements

- Add persistent vector storage using FAISS or Chroma.
- Display source pages and document citations for answers.
- Add OCR support for scanned PDFs.
- Improve retrieval with reranking and metadata filtering.
- Add automated evaluation for retrieval quality and answer accuracy.
- Deploy the application to a cloud platform.

## Security

Keep API keys in environment variables or a secrets manager. Do not commit `.env`, uploaded private documents, or credentials to version control.

## License

This project is available for educational and portfolio purposes. Add a license if you intend to distribute or reuse it under specific terms.
