"""
Wound Analysis API Endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from pathlib import Path
import hashlib
import io
import uuid
import numpy as np
from datetime import datetime
import logging

from app.database import get_db
from app.services.cv_service import WoundAnalyzer, validate_image
from app.services.llm_service import get_llm_analyzer
from app.services.report_service import generate_scan_report
from app import schemas, models
from app.config import settings

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wounds", tags=["wounds"])

# Initialize services
wound_analyzer = WoundAnalyzer(calibration_factor=settings.DEFAULT_CALIBRATION_FACTOR)


@router.post("/analyze", response_model=models.WoundAnalysisResponse)
async def analyze_wound(
    file: UploadFile = File(..., description="Wound image (JPG, PNG)"),
    patient_id: str = Form(default="default_patient"),
    calibration_factor: float = Form(default=None, ge=0.005, le=0.2),
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
    # ── 1. File type check ───────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    # ── 2. Read content + size limit ─────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(content)//1024//1024}MB). Maximum allowed size is 20MB.",
        )

    # ── 3. Compute hash for duplicate detection ───────────────────────────────
    image_hash = hashlib.md5(content).hexdigest()
    duplicate = (
        db.query(schemas.WoundScan)
        .filter(
            schemas.WoundScan.patient_id == patient_id,
            schemas.WoundScan.image_hash == image_hash,
        )
        .first()
    )
    if duplicate:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This exact image was already analyzed for patient '{patient_id}' "
                f"(scan ID: {duplicate.id[:8]}). Please upload a new photo."
            ),
        )

    # ── 4. Save file ──────────────────────────────────────────────────────────
    scan_id = str(uuid.uuid4())
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_extension = Path(file.filename or "wound.jpg").suffix or ".jpg"
    image_path = upload_dir / f"{scan_id}{file_extension}"

    try:
        image_path.write_bytes(content)
        logger.info(f"Saved uploaded image: {image_path}")

        # ── 5. Basic image validation (size / readability) ────────────────────
        if not validate_image(str(image_path)):
            image_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail=(
                    "Image is too small or unreadable. "
                    "Please upload a clear photo (minimum 100×100 px)."
                ),
            )

        # ── 6. CV analysis ────────────────────────────────────────────────────
        try:
            cv_metrics = wound_analyzer.analyze_wound(str(image_path), calibration_factor=calibration_factor)
        except ValueError as e:
            image_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"CV analysis failed: {e}")
            image_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Computer vision analysis failed: {e}")
        
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
            image_hash=image_hash,
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
                confidence_scores=cv_metrics.wound_features.confidence_scores if cv_metrics.wound_features else None,
                ml_confidence=cv_metrics.wound_features.ml_confidence if cv_metrics.wound_features else None,
                classified_by=cv_metrics.wound_features.classified_by if cv_metrics.wound_features else "heuristic",
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
            "scans": [],
            "healing_trajectory": {"available": False, "reason": "No scans found"},
        }

    scans_data = [scan.to_dict() for scan in scans]
    return {
        "patient_id": patient_id,
        "scan_count": len(scans),
        "scans": scans_data,
        "healing_trajectory": _predict_healing_trajectory(scans_data),
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


@router.get("/{scan_id}/image")
async def get_scan_image(scan_id: str, db: Session = Depends(get_db)):
    """Serve the original wound image for a scan."""
    scan = db.query(schemas.WoundScan).filter(schemas.WoundScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    image_path = Path(scan.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(str(image_path), media_type="image/jpeg")


@router.get("/{scan_id}/annotated")
async def get_annotated_image(scan_id: str, db: Session = Depends(get_db)):
    """Serve the annotated wound image for a scan."""
    scan = db.query(schemas.WoundScan).filter(schemas.WoundScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    image_path = Path(scan.image_path)
    annotated_path = image_path.parent / f"{scan_id}_annotated{image_path.suffix}"
    if not annotated_path.exists():
        # Fall back to original if annotated not found
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        return FileResponse(str(image_path), media_type="image/jpeg")
    return FileResponse(str(annotated_path), media_type="image/jpeg")


@router.get("/{scan_id}/report")
async def download_report(scan_id: str, db: Session = Depends(get_db)):
    """Generate and download a PDF report for a scan."""
    scan = db.query(schemas.WoundScan).filter(schemas.WoundScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

    scan_data = scan.to_dict()
    scan_data["id"] = scan.id

    image_path = Path(scan.image_path)
    annotated_path = image_path.parent / f"{scan_id}_annotated{image_path.suffix}"

    try:
        pdf_bytes = generate_scan_report(
            scan_data,
            annotated_image_path=str(annotated_path) if annotated_path.exists() else None,
        )
    except Exception as e:
        logger.error(f"PDF generation failed for scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=meditrack_report_{scan_id[:8]}.pdf"},
    )


def _predict_healing_trajectory(scans: list) -> dict:
    """
    Predict healing trajectory from chronological scan history.
    Uses linear regression on healing scores to estimate scans until good healing.
    Requires at least 3 scans — a 2-point line is not a reliable trend.
    """
    if len(scans) < 3:
        return {
            "available": False,
            "reason": f"Need at least 3 scans for prediction (have {len(scans)}).",
        }

    confidence = "low" if len(scans) < 5 else "medium" if len(scans) < 10 else "high"

    # scans are newest-first; reverse for chronological order
    scores = [s["metrics"]["healing_score"] for s in reversed(scans)]
    x = np.arange(len(scores), dtype=float)
    slope, intercept = np.polyfit(x, scores, 1)

    current = scores[-1]

    if slope <= 0:
        return {
            "available": True,
            "trend": "declining",
            "confidence": confidence,
            "slope_per_scan": round(float(slope), 2),
            "current_score": round(current, 1),
            "message": "Healing score is not improving. Consider consulting a healthcare provider.",
        }

    target = 85.0
    if current >= target:
        return {
            "available": True,
            "trend": "good",
            "confidence": confidence,
            "slope_per_scan": round(float(slope), 2),
            "current_score": round(current, 1),
            "message": "Wound is healing well (score ≥ 85).",
        }

    scans_needed = int(round((target - current) / slope))
    return {
        "available": True,
        "trend": "improving",
        "confidence": confidence,
        "slope_per_scan": round(float(slope), 2),
        "current_score": round(current, 1),
        "scans_to_target": scans_needed,
        "message": f"At current rate, ~{scans_needed} more scan(s) to reach good healing (score ≥ 85).",
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