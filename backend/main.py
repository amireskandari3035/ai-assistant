"""
Main FastAPI application for the Master's AI Assistant.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings, validate_settings
from backend.routers import chat
from backend.services.document_service import document_service
from backend.services.vector_store import vector_store_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Master's AI Assistant...")
    
    # Validate settings
    errors = validate_settings()
    if errors:
        for error in errors:
            logger.warning(f"⚠️ {error}")
    
    # Load existing documents on startup
    try:
        doc_count = vector_store_service.get_document_count()
        if doc_count == 0:
            logger.info("📚 Checking for booklet.pdf...")
            
            chunks = []
            # Check for specific booklet.pdf first
            if settings.default_booklet_path.exists():
                logger.info(f"📄 Found {settings.default_booklet_path.name}, indexing...")
                docs = document_service.load_single_pdf(settings.default_booklet_path)
                chunks = document_service.chunk_documents(docs)
            else:
                # Fallback to documents directory
                chunks = document_service.process_documents()
                
            if chunks:
                vector_store_service.add_documents(chunks)
                logger.info(f"✅ Indexed {len(chunks)} document chunks")
            else:
                logger.info("📭 No documents found to index.")
        else:
            logger.info(f"✅ Found {doc_count} existing document chunks")
    except Exception as e:
        logger.error(f"❌ Error loading documents: {e}")
    
    logger.info(f"🤖 Using model: {settings.llm_model}")
    logger.info(f"🌐 Server ready at http://{settings.host}:{settings.port}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Master's AI Assistant",
    description="دستیار هوشمند فارسی‌زبان برای دانشجویان کارشناسی ارشد",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)

# Mount static files for frontend
frontend_path = settings.frontend_path
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    """Serve the main chat interface."""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Master's AI Assistant API",
        "docs": "/docs",
        "status": "Frontend not found. Place index.html in the frontend folder."
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
