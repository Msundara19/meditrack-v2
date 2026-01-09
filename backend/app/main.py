"""
MediTrack FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from app.config import settings
from app.database import engine, Base
from app.api import wounds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting MediTrack v2.0...")
    
    # Create necessary directories
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("✓ Directories initialized")
    
    # Initialize database
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Database initialized")
    
    logger.info(f"✅ MediTrack v2.0 is ready! (Environment: {settings.ENVIRONMENT})")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MediTrack...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(wounds.router)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "MediTrack API v2.0",
        "status": "online",
        "docs": "/docs",
        "environment": settings.ENVIRONMENT
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT
        }
    )