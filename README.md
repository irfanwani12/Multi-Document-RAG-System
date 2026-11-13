# Generative AI Intelligent Multi-Document RAG System

A Generative AI-powered application that allows users to upload multiple PDF documents, ask questions about their contents, compare information across documents, and receive context-aware answers with source citations.

Built using Python, FastAPI, LangGraph, ChromaDB, Hugging Face Embeddings, and Groq LLM.

## Features

- **Multi-PDF Upload:** Upload and process multiple PDF documents.
- **Document Processing:** Extract text and split documents into manageable chunks.
- **Embeddings:** Generate semantic embeddings using Hugging Face Sentence Transformers.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB.
- **Semantic Search:** Retrieve relevant document chunks based on user questions.
- **RAG Pipeline:** Generate context-aware answers using retrieved document information.
- **LangGraph Workflow:** Organize retrieval and answer generation into a graph-based workflow.
- **Conversation Memory:** Maintain conversation context using LangGraph and SQLite checkpointing.
- **Source Citations:** Provide document names and page numbers as answer references.
- **Relevance Filtering:** Filter irrelevant document chunks to reduce unsupported answers.
- **REST API:** Interact with the application through FastAPI and Swagger UI.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| FastAPI | REST API |
| LangChain | LLM and document integration |
| LangGraph | RAG workflow orchestration |
| ChromaDB | Vector database |
| Hugging Face | Text embeddings |
| Sentence Transformers | Semantic embedding model |
| Groq | LLM inference |
| SQLite | Persistent conversation checkpoints |
| PyPDF | PDF text extraction |
| Uvicorn | ASGI server |

## Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture.

1. **Document Upload:** Users upload PDF documents through the API.
2. **Text Extraction:** Text is extracted from uploaded PDF files.
3. **Chunking:** Extracted text is divided into smaller chunks.
4. **Embedding Generation:** Hugging Face generates vector representations of the chunks.
5. **Vector Storage:** Embeddings and document metadata are stored in ChromaDB.
6. **Retrieval:** Relevant chunks are retrieved using semantic similarity search.
7. **Generation:** LangGraph passes the retrieved context to the Groq language model.
8. **Response:** The application generates an answer with document and page citations.
9. **Memory:** SQLite checkpointing maintains conversation history across interactions.

## RAG Workflow

```text
        PDF Documents
              |
              v
       Text Extraction
              |
              v
          Chunking
              |
              v
     Hugging Face Embeddings
              |
              v
          ChromaDB
              |
              |
       User Question
              |
              v
      Semantic Retrieval
              |
              v
       Relevance Filtering
              |
              v
       LangGraph Workflow
              |
              v
          Groq LLM
              |
              v
      Generated Answer
              |
              v
      Source Citations
```

## Installation and Setup

### Prerequisites

- Python 3.11 or later (a compatible Python version is required by the installed dependencies).
- uv package manager.
- Groq API key.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Assignment
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload your actual API key to GitHub.

### 4. Run the application

```bash
uv run uvicorn main:app --reload
```

### 5. Open Swagger UI

Visit:

http://127.0.0.1:8000/docs

Use the interactive API documentation to upload PDF documents and ask questions.

## Example Use Cases

**Single-document question answering**

Question: What was the company's net profit in 2025?

Expected answer: ₹15 crore, with the relevant document and page citation.

**Multi-document comparison**

Question: Compare the company's net profit in 2024 and 2025.

Expected answer: Net profit increased from ₹12 crore in 2024 to ₹15 crore in 2025, an increase of ₹3 crore.

**Conversation memory**

Ask a follow-up question referring to a previous answer using the same conversation ID.

**Irrelevant question handling**

Ask something unrelated to the uploaded documents. The system should indicate when relevant information cannot be found.

## Project Structure

```text
Assignment/
│
├── api/
│   └── routes.py
│
├── config/
│   └── settings.py
│
├── data/
│   └── sample_documents/
│
├── docs/
│   └── architecture.md
│
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── memory.py
│   ├── pdf_processor.py
│   ├── rag_chain.py
│   ├── retriever.py
│   └── vector_store.py
│
├── chroma_db/
├── main.py
├── checkpoints.sqlite
├── pyproject.toml
├── README.md
└── .env
```

## Future Improvements

- Hybrid search combining semantic and keyword retrieval.
- Reranking for improved retrieval accuracy.
- Advanced RAG evaluation and benchmarking.
- Improved citation verification.
- Support for additional document formats.
- Authentication and user-specific document management.
- Automated testing and monitoring.

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG).
- Multi-document information retrieval.
- Vector databases and semantic search.
- LLM integration.
- LangGraph-based AI workflows.
- Conversational memory and checkpointing.
- REST API development using FastAPI.

---

**Project:** Generative AI Intelligent Multi-Document RAG System

**Status:** Core implementation completed; documentation and final validation in progress.
