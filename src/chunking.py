
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[dict]:
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk retains its original document name
    and page number.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative "
            "and smaller than chunk size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        chunks = text_splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_index": chunk_index
                }
            })

    return all_chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[dict]:
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk retains its original document name
    and page number.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative "
            "and smaller than chunk size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        chunks = text_splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_index": chunk_index
                }
            })

    return all_chunks
