
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.pdf_processor import extract_pdf_text
from src.chunking import split_documents
from src.vector_store import add_chunks_to_vector_store
from src.graph import run_rag_graph

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str = Field(default="default", min_length=1)


@router.get("/health")
def health_check():
    return {"status": "healthy", "project": "Assignment"}


@router.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF.")

    uploaded_documents = []

    for file in files:
        filename = Path(file.filename or "").name

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{filename or 'Uploaded file'} is not a PDF."
            )

        pdf_bytes = await file.read()

        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} is empty."
            )

        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the 20 MB file-size limit."
            )

        try:
            pages = extract_pdf_text(pdf_bytes, filename)
            chunks = split_documents(pages)

            stored_ids = add_chunks_to_vector_store(chunks)

            uploaded_documents.append({
                "filename": filename,
                "pages_processed": len(pages),
                "chunks_stored": len(stored_ids),
            })

        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Could not process {filename}: {str(error)}"
            )

    return {
        "message": "Documents processed successfully.",
        "documents": uploaded_documents,
        "total_documents": len(uploaded_documents),
    }


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        result = run_rag_graph(
            question=request.question,
            conversation_id=request.conversation_id,
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "conversation_id": request.conversation_id,
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate an answer: {str(error)}"
        )
