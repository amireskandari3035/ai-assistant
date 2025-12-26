"""
Chat API endpoints for the AI Assistant.
"""

import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.config import settings
from backend.services.chat_service import chat_service
from backend.services.document_service import document_service
from backend.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    sources: list[dict] = []
    has_context: bool = False


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    documents_loaded: int
    model: str
    has_api_key: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the AI assistant and get a response.
    
    The assistant will use RAG to find relevant information from
    uploaded documents and provide a helpful response in Persian.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="پیام نمی‌تواند خالی باشد")
    
    try:
        result = await chat_service.chat(request.message)
        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            has_context=result.get("has_context", False)
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="خطا در پردازش پیام. لطفاً دوباره تلاش کنید."
        )


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be processed and indexed.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="فقط فایل‌های PDF پشتیبانی می‌شوند"
        )
    
    try:
        # Save the uploaded file
        file_path = settings.documents_path / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Process the document
        documents = document_service.load_single_pdf(file_path)
        chunks = document_service.chunk_documents(documents)
        
        # Add to vector store
        count = vector_store_service.add_documents(chunks)
        
        return {
            "success": True,
            "message": f"فایل با موفقیت آپلود شد",
            "filename": file.filename,
            "pages_processed": len(documents),
            "chunks_created": count
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطا در پردازش فایل: {str(e)}"
        )


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get the current status of the AI assistant.
    """
    return StatusResponse(
        status="ready",
        documents_loaded=vector_store_service.get_document_count(),
        model=settings.llm_model,
        has_api_key=bool(settings.groq_api_key)
    )


@router.post("/clear")
async def clear_history():
    """
    Clear the conversation history.
    """
    chat_service.clear_history()
    return {"success": True, "message": "تاریخچه گفتگو پاک شد"}


@router.post("/reindex")
async def reindex_documents():
    """
    Clear and rebuild the vector store from all documents.
    """
    try:
        # Clear existing
        vector_store_service.clear_collection()
        
        # Reprocess all documents
        chunks = document_service.process_documents()
        
        if chunks:
            vector_store_service.add_documents(chunks)
        
        return {
            "success": True,
            "message": "اسناد با موفقیت مجدداً ایندکس شدند",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطا در ایندکس‌گذاری مجدد: {str(e)}"
        )
