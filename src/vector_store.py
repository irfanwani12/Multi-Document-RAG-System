
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import get_embedding_model


PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "assignment_documents"


def get_vector_store() -> Chroma:
    """
    Create or load the persistent ChromaDB vector store.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store


def add_chunks_to_vector_store(chunks: list[dict]) -> list[str]:
    """
    Convert text chunks into LangChain Documents
    and store them in ChromaDB.
    """

    if not chunks:
        raise ValueError("No chunks provided for storage.")

    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        for chunk in chunks
    ]

    ids = [
        f"{chunk['metadata']['source']}_"
        f"{chunk['metadata']['page']}_"
        f"{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]

    vector_store = get_vector_store()

    stored_ids = vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    return stored_ids
