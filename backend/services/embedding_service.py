"""
Embedding service using HuggingFace models.
Uses a multilingual model that supports Persian text.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        self._embeddings = None
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy load embeddings model."""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully")
        return self._embeddings
    
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        return self.embeddings.embed_documents(texts)


# Global service instance
embedding_service = EmbeddingService()
