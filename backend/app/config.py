"""
Configuration settings for MediTrack
"""
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_TITLE: str = "MediTrack API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "AI-Powered Wound Healing Monitor with Multi-Factor Classification"
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:8501",
        "http://localhost:3000",
        "https://*.streamlit.app",
        "*"  # Allow all for demo (restrict in production)
    ]
    
    # Database - Use environment variable or default to local
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/meditrack.db")
    
    # Directories - Handle both local and production
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    
    # Computer Vision
    DEFAULT_CALIBRATION_FACTOR: float = 0.1
    
    # LLM API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()