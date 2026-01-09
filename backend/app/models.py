"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class WoundMetricsResponse(BaseModel):
    """Response model for wound metrics"""
    wound_area_pixels: int
    wound_area_cm2: float
    redness_index: float
    edge_sharpness: float
    healing_score: float
    
    # NEW: Classification fields
    wound_type: str = "unknown"
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    aspect_ratio: Optional[float] = None
    circularity: Optional[float] = None
    measurement_type: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response model for LLM analysis"""
    risk_level: str = Field(..., description="Risk level: low, medium, or high")
    summary: str = Field(..., description="Patient-friendly analysis summary")
    recommendations: str = Field(..., description="Care recommendations")


class WoundAnalysisResponse(BaseModel):
    """Complete wound analysis response"""
    scan_id: str
    patient_id: str
    scan_date: datetime
    metrics: WoundMetricsResponse
    analysis: AnalysisResponse
    image_url: str
    annotated_image_url: Optional[str] = None


class PatientHistoryItem(BaseModel):
    """Single item in patient history"""
    scan_id: str
    scan_date: datetime
    healing_score: float
    wound_area_cm2: float
    risk_level: str
    wound_type: Optional[str] = None
    
    class Config:
        from_attributes = True


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str = "2.0.0"