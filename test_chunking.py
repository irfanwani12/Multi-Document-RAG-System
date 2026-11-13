
from pathlib import Path

from src.pdf_processor import extract_pdf_text
from src.chunking import split_documents

pdf_path = Path(
    "data/sample_documents/Annual_Report_2025.pdf"
)

try:
    # Step 1: Extract PDF pages
    pdf_bytes = pdf_path.read_bytes()

    pages = extract_pdf_text(
        pdf_bytes,
        pdf_path.name
    )

    # Step 2: Split extracted pages
    chunks = split_documents(
        pages,
        chunk_size=800,
        chunk_overlap=150
    )

    # Step 3: Display results
    print("\nPDF successfully chunked!")
    print(f"Total PDF pages: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks[:5], start=1):
        print("\n----------------------------")
        print(f"Chunk number: {index}")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Text length: {len(chunk['text'])}")
        print(f"Content: {chunk['text'][:200]}")

except Exception as error:
    print(f"Error: {error}")
