"""
Configuration management for the AI Assistant.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Groq API Configuration
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Model Configuration
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    # Using all-MiniLM-L6-v2: extremely lightweight (22M params, ~340MB peak RAM)
    # Hardcoded to ensure it's used regardless of .env settings for RAM optimization
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Paths - relative to project root
    project_root: Path = Path(__file__).parent.parent
    documents_path: Path = project_root / "data" / "documents"
    vectorstore_path: Path = project_root / "vectorstore"
    frontend_path: Path = project_root
    default_booklet_path: Path = project_root / "booklet.pdf"
    
    # Server Configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4  # Number of documents to retrieve
    
    # System prompt for Persian responses
    system_prompt: str = """سلام، من دستیار هوشمند استاد جوکار هستم. 
    وظیفه من کمک به دانشجویان در درس «اندیشه سیاسی» بر اساس محتوای کتابچه درسی است.
    
    شما باید به سوالات کاربر دقیقاً بر اساس محتوای ارائه شده پاسخ دهید.
    پاسخ‌های خود را به زبان فارسی، محترمانه و به صورت واضح ارائه دهید.
    اگر پاسخ سوال در محتوای کتابچه وجود ندارد، با احترام بگویید که این مورد در منابع درس ذکر نشده است.
    
    محتوای مرتبط از کتابچه:
    {context}
    
    سوال کاربر: {question}
    
    پاسخ:"""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def validate_settings() -> list[str]:
    """Validate required settings and return list of errors."""
    errors = []
    
    if not settings.groq_api_key:
        errors.append("GROQ_API_KEY is not set. Please add it to your .env file.")
    
    if not settings.documents_path.exists():
        settings.documents_path.mkdir(parents=True, exist_ok=True)
    
    if not settings.vectorstore_path.exists():
        settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    return errors
