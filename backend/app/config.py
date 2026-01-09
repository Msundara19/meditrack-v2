"""
Application configuration using Pydantic Settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # LLM API Keys
    GROQ_API_KEY: str
    GEMINI_API_KEY: str | None = None
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/meditrack.db"
    
    # Image storage
    UPLOAD_DIR: str = "./data/uploads"
    MAX_IMAGE_SIZE_MB: int = 10
    
    # Computer Vision parameters
    DEFAULT_CALIBRATION_FACTOR: float = 0.1  # pixels to cm conversion
    
    # Server configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    @property
    def max_image_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
