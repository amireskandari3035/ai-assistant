"""
Vector store service using ChromaDB for document storage and retrieval.
"""

import logging
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from backend.config import settings
from backend.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing the vector store."""
    
    COLLECTION_NAME = "master_booklet"
    
    def __init__(self):
        self._vectorstore: Chroma | None = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Get or create the vector store."""
        if self._vectorstore is None:
            self._vectorstore = self._load_or_create_vectorstore()
        return self._vectorstore
    
    def _load_or_create_vectorstore(self) -> Chroma:
        """Load existing vector store or create a new one."""
        persist_directory = str(settings.vectorstore_path)
        
        vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=embedding_service.embeddings,
            persist_directory=persist_directory
        )
        
        # Check if collection has documents
        collection = vectorstore._collection
        count = collection.count()
        
        if count > 0:
            logger.info(f"Loaded existing vector store with {count} documents")
        else:
            logger.info("Created new empty vector store")
        
        return vectorstore
    
    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        self.vectorstore.add_documents(documents)
        count = self.vectorstore._collection.count()
        logger.info(f"Added {len(documents)} documents. Total: {count}")
        return len(documents)
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self._vectorstore = None
        
        # Delete and recreate
        persist_directory = str(settings.vectorstore_path)
        vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=embedding_service.embeddings,
            persist_directory=persist_directory
        )
        vectorstore.delete_collection()
        self._vectorstore = None
        logger.info("Cleared vector store collection")
    
    def similarity_search(self, query: str, k: int = None) -> list[Document]:
        """Search for similar documents."""
        if k is None:
            k = settings.retriever_k
        
        results = self.vectorstore.similarity_search(query, k=k)
        logger.debug(f"Found {len(results)} similar documents for query")
        return results
    
    def get_retriever(self):
        """Get a retriever for RAG chain."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_k}
        )
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return self.vectorstore._collection.count()


# Global service instance
vector_store_service = VectorStoreService()
