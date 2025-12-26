"""
Lightweight embedding service using sentence-transformers directly.
Optimized for low RAM usage (<500MB peak).
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from backend.config import settings

logger = logging.getLogger(__name__)


class LightweightEmbeddings(Embeddings):
    """
    Memory-efficient embeddings using sentence-transformers directly.
    
    This class provides a LangChain-compatible interface while using
    direct sentence-transformers to minimize memory overhead.
    Peak RAM usage: ~285MB (vs ~670MB with LangChain wrapper).
    """
    
    def __init__(self, model_name: str = None):
        # Hardcoded for RAM optimization to ensure <500MB
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to defer memory allocation."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device='cpu'
            )
            logger.info("Embedding model loaded successfully")
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=8,  # Lower batch size to reduce peak memory
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        self._embeddings = None
    
    @property
    def embeddings(self) -> LightweightEmbeddings:
        """Get the embeddings model (lazy loaded)."""
        if self._embeddings is None:
            self._embeddings = LightweightEmbeddings()
        return self._embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings."""
        return self.embeddings.embed_documents(texts)


# Global service instance
embedding_service = EmbeddingService()
