
from langchain_core.documents import Document

from src.vector_store import get_vector_store


# Maximum L2 distance allowed for a retrieved document chunk.
# Lower distance means a more relevant match.
DISTANCE_THRESHOLD = 1.2


def retrieve_relevant_chunks(
    question: str,
    k: int = 4
) -> list[Document]:
    """
    Retrieve relevant document chunks from ChromaDB
    and filter out chunks above the distance threshold.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if k <= 0:
        raise ValueError("The number of results must be positive.")

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=question,
        k=k
    )

    relevant_documents = [
        document
        for document, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    return relevant_documents
