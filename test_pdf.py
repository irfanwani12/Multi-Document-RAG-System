
from pathlib import Path
from src.pdf_processor import extract_pdf_text

# Locate the sample PDF
pdf_path = Path("data/sample_documents/Annual_Report_2025.pdf")

try:
    # Read PDF as bytes
    pdf_bytes = pdf_path.read_bytes()

    # Extract text and metadata
    pages = extract_pdf_text(pdf_bytes, pdf_path.name)

    print("\nPDF successfully processed!")
    print(f"Total pages with extractable text: {len(pages)}")

    for page in pages:
        print("\n------------------------------")
        print(f"Source: {page['metadata']['source']}")
        print(f"Page: {page['metadata']['page']}")
        print("Extracted text:")
        print(page["text"][:500])

except Exception as error:
    print(f"Error: {error}")
