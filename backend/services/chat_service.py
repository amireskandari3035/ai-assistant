"""
Chat service using Groq AI for LLM responses with RAG.
"""

import logging
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage

from backend.config import settings
from backend.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat interactions with Groq AI."""
    
    def __init__(self):
        self._llm: ChatGroq | None = None
        self._rag_chain = None
        self._conversation_history: list[dict] = []
    
    @property
    def llm(self) -> ChatGroq:
        """Lazy load the LLM."""
        if self._llm is None:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is not set")
            
            logger.info(f"Initializing Groq LLM with model: {settings.llm_model}")
            self._llm = ChatGroq(
                api_key=settings.groq_api_key,
                model_name=settings.llm_model,
                temperature=0.7,
                max_tokens=2048
            )
            logger.info("Groq LLM initialized successfully")
        return self._llm
    
    def _get_rag_chain(self):
        """Get or create the RAG chain."""
        if self._rag_chain is None:
            prompt_template = PromptTemplate(
                template=settings.system_prompt,
                input_variables=["context", "question"]
            )
            
            self._rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store_service.get_retriever(),
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True
            )
            logger.info("RAG chain created")
        return self._rag_chain
    
    async def chat(self, message: str) -> dict:
        """
        Process a chat message and return the AI response.
        
        Args:
            message: The user's message in Persian
            
        Returns:
            dict with 'response' and optionally 'sources'
        """
        try:
            # Check if we have documents in the vector store
            doc_count = vector_store_service.get_document_count()
            
            if doc_count == 0:
                # No documents, use direct LLM
                logger.warning("No documents in vector store, using direct LLM")
                response = await self._direct_chat(message)
                return {
                    "response": response,
                    "sources": [],
                    "has_context": False
                }
            
            # Use RAG chain
            rag_chain = self._get_rag_chain()
            result = rag_chain.invoke({"query": message})
            
            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    sources.append({
                        "page": doc.metadata.get("page", "Unknown"),
                        "content_preview": doc.page_content[:100] + "..."
                    })
            
            # Store in conversation history
            self._conversation_history.append({
                "role": "user",
                "content": message
            })
            self._conversation_history.append({
                "role": "assistant",
                "content": result["result"]
            })
            
            return {
                "response": result["result"],
                "sources": sources,
                "has_context": True
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    async def _direct_chat(self, message: str) -> str:
        """Direct chat without RAG when no documents are available."""
        system_message = """شما یک دستیار هوشمند فارسی‌زبان هستید. 
        در حال حاضر هیچ سندی بارگذاری نشده است.
        لطفاً به کاربر بگویید که ابتدا باید کتابچه درسی را آپلود کند."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> list[dict]:
        """Get conversation history."""
        return self._conversation_history.copy()


# Global service instance
chat_service = ChatService()
