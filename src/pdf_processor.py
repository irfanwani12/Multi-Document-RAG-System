
from io import BytesIO
from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text from every page of an uploaded PDF.

    Args:
        pdf_bytes: PDF file content in bytes.
        filename: Original name of the uploaded PDF.

    Returns:
        A list containing extracted text and page metadata.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                extracted_pages.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": filename,
                        "page": page_number
                    }
                })

        if not extracted_pages:
            raise ValueError(
                "No extractable text found in this PDF. "
                "It may be scanned or image-based."
            )

        return extracted_pages

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to process PDF '{filename}': {error}"
        ) from error
