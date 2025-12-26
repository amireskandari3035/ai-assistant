"""
Document processing service for loading and chunking PDF documents.
Handles Persian text with appropriate chunking strategies.
"""

import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from backend.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for loading and processing PDF documents."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " ", ""]
        )
    
    def load_single_pdf(self, file_path: Path) -> list[Document]:
        """Load a single PDF file and return documents."""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def load_all_documents(self) -> list[Document]:
        """Load all PDF documents from the documents directory."""
        documents_path = settings.documents_path
        
        if not documents_path.exists():
            logger.warning(f"Documents directory does not exist: {documents_path}")
            return []
        
        # Find all PDF files
        pdf_files = list(documents_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {documents_path}")
            return []
        
        all_documents = []
        for pdf_file in pdf_files:
            try:
                docs = self.load_single_pdf(pdf_file)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {pdf_file}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks for embedding."""
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def process_documents(self) -> list[Document]:
        """Load and chunk all documents in one step."""
        documents = self.load_all_documents()
        chunks = self.chunk_documents(documents)
        return chunks


# Global service instance
document_service = DocumentService()
