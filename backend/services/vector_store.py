"""
Lightweight vector store using numpy for document storage and retrieval.
Optimized for low RAM usage (<500MB peak) for small document sets.
"""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional

from backend.config import settings
from backend.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """
    A simple, memory-efficient vector store using numpy and pickle.
    Perfect for small document sets (<10,000 documents).
    """
    
    def __init__(self, persist_directory: str):
        self.persist_path = Path(persist_directory)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.data_path = self.persist_path / "vectorstore.pkl"
        
        self.documents = []
        self.embeddings = None
        self._load()
    
    def _load(self):
        """Load documents and embeddings from disk."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.embeddings = data.get("embeddings")
                logger.info(f"Loaded {len(self.documents)} documents from {self.data_path}")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self.documents = []
                self.embeddings = None
    
    def _save(self):
        """Save documents and embeddings to disk."""
        try:
            with open(self.data_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "embeddings": self.embeddings
                }, f)
            logger.info(f"Saved {len(self.documents)} documents to {self.data_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def add_documents(self, documents):
        """Add documents to the store."""
        if not documents:
            return 0
        
        texts = [doc.page_content for doc in documents]
        new_embeddings = embedding_service.embeddings.embed_documents(texts)
        new_embeddings_np = np.array(new_embeddings).astype(np.float32)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings_np
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings_np])
        
        self.documents.extend(documents)
        self._save()
        return len(documents)
    
    def similarity_search(self, query: str, k: int = 4):
        """Search for similar documents using cosine similarity."""
        if not self.documents or self.embeddings is None:
            return []
        
        query_embedding = embedding_service.embeddings.embed_query(query)
        query_np = np.array(query_embedding).astype(np.float32)
        
        # Cosine similarity: (A . B) / (||A|| * ||B||)
        # Since we normalize embeddings in the service, we just need dot product
        similarities = np.dot(self.embeddings, query_np)
        
        # Get top k indices
        top_k_indices = np.argsort(-similarities)[:k]
        
        results = [self.documents[i] for i in top_k_indices]
        return results
    
    def delete_collection(self):
        """Clear all documents."""
        self.documents = []
        self.embeddings = None
        if self.data_path.exists():
            self.data_path.unlink()
        logger.info("Cleared vector store")
    
    def count(self) -> int:
        """Get number of documents."""
        return len(self.documents)


class VectorStoreService:
    """Service for managing the vector store."""
    
    def __init__(self):
        self._vectorstore = None
    
    @property
    def vectorstore(self) -> SimpleVectorStore:
        """Get or create the vector store."""
        if self._vectorstore is None:
            self._vectorstore = SimpleVectorStore(str(settings.vectorstore_path))
        return self._vectorstore
    
    def add_documents(self, documents) -> int:
        """Add documents to the vector store."""
        return self.vectorstore.add_documents(documents)
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.vectorstore.delete_collection()
        logger.info("Cleared vector store collection")
    
    def similarity_search(self, query: str, k: int = None):
        """Search for similar documents."""
        if k is None:
            k = settings.retriever_k
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self):
        """Get a retriever-like object for RAG chain."""
        # We need to return an object that has an 'invoke' or 'get_relevant_documents' method
        class SimpleRetriever:
            def __init__(self, vs, k):
                self.vs = vs
                self.k = k
            def invoke(self, query: str):
                return self.vs.similarity_search(query, k=self.k)
            def get_relevant_documents(self, query: str):
                return self.vs.similarity_search(query, k=self.k)
        
        return SimpleRetriever(self.vectorstore, settings.retriever_k)
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return self.vectorstore.count()


# Global service instance
vector_store_service = VectorStoreService()
