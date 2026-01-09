"""
MediTrack v2.0 - FastAPI Backend
AI-Powered Wound Healing Monitor
"""
import sys
from pathlib import Path

# Add parent directory to path so imports work correctly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from app.config import settings
from app.database import init_db
from app.models import HealthCheckResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MediTrack API",
    description="AI-Powered Wound Healing Monitoring System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting MediTrack v2.0...")
    
    # Ensure directories exist
    settings.ensure_directories()
    logger.info("✓ Directories initialized")
    
    # Initialize database
    init_db()
    logger.info("✓ Database initialized")
    
    logger.info("✅ MediTrack v2.0 is ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down MediTrack v2.0...")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "MediTrack API v2.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="2.0.0"
    )


# Include API routers
from app.api import wounds
app.include_router(wounds.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )