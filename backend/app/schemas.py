"""
SQLAlchemy database models
"""
from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class Patient(Base):
    """Patient information table"""
    __tablename__ = "patients"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    scans = relationship("WoundScan", back_populates="patient")


class WoundScan(Base):
    """Wound scan analysis results table"""
    __tablename__ = "wound_scans"
    
    id = Column(String, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), index=True)
    
    image_path = Column(String, nullable=False)
    scan_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Computer vision metrics
    wound_area_pixels = Column(Float)
    wound_area_cm2 = Column(Float)
    redness_index = Column(Float)
    edge_sharpness = Column(Float)
    healing_score = Column(Float)
    
    # LLM analysis
    risk_level = Column(String)
    llm_summary = Column(Text)
    recommendations = Column(Text)
    
    # Metadata
    calibration_factor = Column(Float)
    
    # NEW: Classification fields
    wound_type = Column(String, default="unknown")
    length_cm = Column(Float, nullable=True)
    width_cm = Column(Float, nullable=True)
    aspect_ratio = Column(Float, nullable=True)
    circularity = Column(Float, nullable=True)
    solidity = Column(Float, nullable=True)
    
    patient = relationship("Patient", back_populates="scans")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "image_path": self.image_path,
            "scan_date": self.scan_date.isoformat() if self.scan_date else None,
            "metrics": {
                "wound_area_pixels": self.wound_area_pixels,
                "wound_area_cm2": self.wound_area_cm2,
                "redness_index": self.redness_index,
                "edge_sharpness": self.edge_sharpness,
                "healing_score": self.healing_score,
                "wound_type": self.wound_type,
                "length_cm": self.length_cm,
                "width_cm": self.width_cm,
            },
            "analysis": {
                "risk_level": self.risk_level,
                "summary": self.llm_summary,
                "recommendations": self.recommendations,
            },
            "calibration_factor": self.calibration_factor,
        }