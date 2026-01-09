"""
Wound Analysis API Endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging

from app.database import get_db
from app.services.cv_service import WoundAnalyzer
from app.services.llm_service import get_llm_analyzer
from app import schemas, models
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wounds", tags=["wounds"])

# Initialize services
wound_analyzer = WoundAnalyzer(calibration_factor=settings.DEFAULT_CALIBRATION_FACTOR)


@router.post("/analyze", response_model=models.WoundAnalysisResponse)
async def analyze_wound(
    file: UploadFile = File(..., description="Wound image (JPG, PNG)"),
    patient_id: str = Form(default="default_patient"),
    db: Session = Depends(get_db)
):
    """
    Analyze uploaded wound image
    
    **Process:**
    1. Validate and save uploaded image
    2. Run computer vision analysis with classification
    3. Generate LLM-powered insights
    4. Store results in database
    5. Return comprehensive analysis
    
    **Returns:**
    - Wound type classification
    - Wound metrics (area, redness, healing score, etc.)
    - Risk assessment and AI summary
    - Links to original and annotated images
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    # Generate unique scan ID
    scan_id = str(uuid.uuid4())
    
    # Ensure upload directory exists
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix or ".jpg"
    image_filename = f"{scan_id}{file_extension}"
    image_path = upload_dir / image_filename
    
    try:
        # Save file
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded image: {image_path}")
        
        # Run CV analysis with classification
        try:
            cv_metrics = wound_analyzer.analyze_wound(str(image_path))
        except Exception as e:
            logger.error(f"CV analysis failed: {e}")
            # Cleanup
            if image_path.exists():
                image_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Computer vision analysis failed: {str(e)}"
            )
        
        # Save annotated image
        annotated_filename = f"{scan_id}_annotated{file_extension}"
        annotated_path = upload_dir / annotated_filename
        wound_analyzer.save_annotated_image(cv_metrics.annotated_image, str(annotated_path))
        
        # Get previous scan for comparison
        previous_scan = db.query(schemas.WoundScan)\
            .filter(schemas.WoundScan.patient_id == patient_id)\
            .order_by(schemas.WoundScan.scan_date.desc())\
            .first()
        
        previous_metrics = None
        if previous_scan:
            previous_metrics = {
                "area_cm2": previous_scan.wound_area_cm2,
                "redness_index": previous_scan.redness_index,
                "healing_score": previous_scan.healing_score
            }
            logger.info(f"Found previous scan for comparison: {previous_scan.id}")
        
        # Generate LLM analysis
        try:
            llm_analyzer = get_llm_analyzer()
            llm_result = llm_analyzer.generate_analysis(
                area_cm2=cv_metrics.wound_area_cm2,
                redness_index=cv_metrics.redness_index,
                edge_sharpness=cv_metrics.edge_sharpness,
                healing_score=cv_metrics.healing_score,
                previous_metrics=previous_metrics
            )
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Use fallback
            llm_result = {
                "risk_level": "unknown",
                "summary": "Analysis completed but AI summary unavailable. Please review metrics.",
                "recommendations": "Continue wound care as directed by your healthcare provider."
            }
        
        # Ensure patient exists in database
        patient = db.query(schemas.Patient).filter(schemas.Patient.id == patient_id).first()
        if not patient:
            patient = schemas.Patient(
                id=patient_id,
                name=None,
                created_at=datetime.utcnow()
            )
            db.add(patient)
            logger.info(f"Created new patient: {patient_id}")
        
        # Save scan to database with classification data
        scan = schemas.WoundScan(
            id=scan_id,
            patient_id=patient_id,
            image_path=str(image_path),
            scan_date=datetime.utcnow(),
            wound_area_pixels=cv_metrics.wound_area_pixels,
            wound_area_cm2=cv_metrics.wound_area_cm2,
            redness_index=cv_metrics.redness_index,
            edge_sharpness=cv_metrics.edge_sharpness,
            healing_score=cv_metrics.healing_score,
            risk_level=llm_result["risk_level"],
            llm_summary=llm_result["summary"],
            recommendations=llm_result["recommendations"],
            calibration_factor=settings.DEFAULT_CALIBRATION_FACTOR,
            # Classification fields
            wound_type=cv_metrics.wound_type,
            length_cm=cv_metrics.wound_features.length_cm if cv_metrics.wound_features else None,
            width_cm=cv_metrics.wound_features.width_cm if cv_metrics.wound_features else None,
            aspect_ratio=cv_metrics.wound_features.aspect_ratio if cv_metrics.wound_features else None,
            circularity=cv_metrics.wound_features.circularity if cv_metrics.wound_features else None,
            solidity=cv_metrics.wound_features.solidity if cv_metrics.wound_features else None,
        )
        
        db.add(scan)
        db.commit()
        db.refresh(scan)
        
        logger.info(f"✓ Analysis complete for scan {scan_id}")
        
        # Return response with classification
        return models.WoundAnalysisResponse(
            scan_id=scan_id,
            patient_id=patient_id,
            scan_date=scan.scan_date,
            metrics=models.WoundMetricsResponse(
                wound_area_pixels=cv_metrics.wound_area_pixels,
                wound_area_cm2=cv_metrics.wound_area_cm2,
                redness_index=cv_metrics.redness_index,
                edge_sharpness=cv_metrics.edge_sharpness,
                healing_score=cv_metrics.healing_score,
                # Classification fields
                wound_type=cv_metrics.wound_type,
                length_cm=cv_metrics.wound_features.length_cm if cv_metrics.wound_features else None,
                width_cm=cv_metrics.wound_features.width_cm if cv_metrics.wound_features else None,
                aspect_ratio=cv_metrics.wound_features.aspect_ratio if cv_metrics.wound_features else None,
                circularity=cv_metrics.wound_features.circularity if cv_metrics.wound_features else None,
                measurement_type=cv_metrics.wound_features.measurement_type if cv_metrics.wound_features else "area",
            ),
            analysis=models.AnalysisResponse(
                risk_level=llm_result["risk_level"],
                summary=llm_result["summary"],
                recommendations=llm_result["recommendations"]
            ),
            image_url=f"/api/wounds/{scan_id}/image",
            annotated_image_url=f"/api/wounds/{scan_id}/annotated"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        # Cleanup uploaded file
        if image_path.exists():
            image_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/{scan_id}")
async def get_scan_details(
    scan_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific wound scan
    
    **Args:**
    - scan_id: UUID of the scan
    
    **Returns:**
    - Complete scan details with metrics and analysis
    """
    scan = db.query(schemas.WoundScan)\
        .filter(schemas.WoundScan.id == scan_id)\
        .first()
    
    if not scan:
        raise HTTPException(
            status_code=404,
            detail=f"Scan {scan_id} not found"
        )
    
    return scan.to_dict()


@router.get("/patient/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get wound scan history for a patient
    
    **Args:**
    - patient_id: Patient identifier
    - limit: Maximum number of scans to return (default: 50)
    
    **Returns:**
    - List of scans ordered by date (newest first)
    """
    scans = db.query(schemas.WoundScan)\
        .filter(schemas.WoundScan.patient_id == patient_id)\
        .order_by(schemas.WoundScan.scan_date.desc())\
        .limit(limit)\
        .all()
    
    if not scans:
        return {
            "patient_id": patient_id,
            "scan_count": 0,
            "scans": []
        }
    
    return {
        "patient_id": patient_id,
        "scan_count": len(scans),
        "scans": [scan.to_dict() for scan in scans]
    }


@router.get("/patient/{patient_id}/latest")
async def get_latest_scan(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the most recent wound scan for a patient
    
    **Args:**
    - patient_id: Patient identifier
    
    **Returns:**
    - Latest scan details
    """
    scan = db.query(schemas.WoundScan)\
        .filter(schemas.WoundScan.patient_id == patient_id)\
        .order_by(schemas.WoundScan.scan_date.desc())\
        .first()
    
    if not scan:
        raise HTTPException(
            status_code=404,
            detail=f"No scans found for patient {patient_id}"
        )
    
    return scan.to_dict()


@router.delete("/{scan_id}")
async def delete_scan(
    scan_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a wound scan and associated images
    
    **Args:**
    - scan_id: UUID of the scan to delete
    
    **Returns:**
    - Success message
    """
    scan = db.query(schemas.WoundScan)\
        .filter(schemas.WoundScan.id == scan_id)\
        .first()
    
    if not scan:
        raise HTTPException(
            status_code=404,
            detail=f"Scan {scan_id} not found"
        )
    
    # Delete image files
    try:
        image_path = Path(scan.image_path)
        if image_path.exists():
            image_path.unlink()
        
        # Try to delete annotated image
        annotated_path = image_path.parent / f"{scan_id}_annotated{image_path.suffix}"
        if annotated_path.exists():
            annotated_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete image files: {e}")
    
    # Delete database record
    db.delete(scan)
    db.commit()
    
    logger.info(f"Deleted scan {scan_id}")
    
    return {
        "message": f"Scan {scan_id} deleted successfully"
    }


@router.get("/stats/overview")
async def get_stats_overview(db: Session = Depends(get_db)):
    """
    Get overall statistics about the system
    
    **Returns:**
    - Total patients
    - Total scans
    - Average healing score
    - Risk distribution
    - Wound type distribution
    """
    from sqlalchemy import func
    
    total_patients = db.query(func.count(schemas.Patient.id)).scalar()
    total_scans = db.query(func.count(schemas.WoundScan.id)).scalar()
    avg_healing_score = db.query(func.avg(schemas.WoundScan.healing_score)).scalar()
    
    # Risk level distribution
    risk_counts = db.query(
        schemas.WoundScan.risk_level,
        func.count(schemas.WoundScan.id)
    ).group_by(schemas.WoundScan.risk_level).all()
    
    risk_distribution = {risk: count for risk, count in risk_counts}
    
    # Wound type distribution
    wound_type_counts = db.query(
        schemas.WoundScan.wound_type,
        func.count(schemas.WoundScan.id)
    ).group_by(schemas.WoundScan.wound_type).all()
    
    wound_type_distribution = {wtype: count for wtype, count in wound_type_counts}
    
    return {
        "total_patients": total_patients or 0,
        "total_scans": total_scans or 0,
        "average_healing_score": round(avg_healing_score, 2) if avg_healing_score else 0,
        "risk_distribution": risk_distribution,
        "wound_type_distribution": wound_type_distribution
    }