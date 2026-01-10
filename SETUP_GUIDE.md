# 🚀 MediTrack v2.0 - Backend Setup Guide

## ✅ What's Been Built

### Backend Components (COMPLETE)
- ✓ **FastAPI Application** (`app/main.py`)
- ✓ **Computer Vision Service** (`app/services/cv_service.py`)
  - Real Otsu thresholding + color-based segmentation
  - Multi-metric extraction (area, redness, edge quality, healing score)
  - Annotated image generation
- ✓ **LLM Service** (`app/services/llm_service.py`)
  - Groq (Llama 3.1) integration
  - Google Gemini fallback
  - Patient-friendly summaries
- ✓ **API Endpoints** (`app/api/wounds.py`)
  - POST /api/wounds/analyze - Analyze wound image
  - GET /api/wounds/{scan_id} - Get scan details
  - GET /api/wounds/patient/{patient_id}/history - Patient history
  - GET /api/wounds/stats/overview - System statistics
- ✓ **Database Models** (SQLAlchemy + SQLite)
- ✓ **Configuration System** (Pydantic Settings)

---

## 📋 Prerequisites

- Python 3.11+
- pip (Python package manager)
- Groq API key (free at https://console.groq.com)
- Optional: Google Gemini API key

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
cd meditrack-v2/backend
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `backend/.env` and add your API keys:

```bash
# Required
GROQ_API_KEY=gsk_your_actual_groq_key_here

# Optional (fallback)
GEMINI_API_KEY=your_gemini_key_here

# These are already configured
DATABASE_URL=sqlite:///./data/meditrack.db
UPLOAD_DIR=./data/uploads
DEFAULT_CALIBRATION_FACTOR=0.1
API_HOST=0.0.0.0
API_PORT=8000
```

**Get your Groq API key:**
1. Go to https://console.groq.com
2. Sign up (it's free!)
3. Go to API Keys section
4. Create new API key
5. Copy and paste into `.env`

### 3. Generate Test Images

```bash
cd meditrack-v2
python scripts/generate_samples.py
```

This creates sample wound images in `data/samples/`:
- wound_mild.jpg
- wound_moderate.jpg
- wound_severe.jpg

---

## 🎯 Running the Backend

### Start the Server

```bash
cd meditrack-v2/backend
python -m app.main
```

You should see:
```
🚀 Starting MediTrack v2.0...
✓ Directories initialized
✓ Database initialized
✅ MediTrack v2.0 is ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test the API

**Option 1: Run automated tests**
```bash
# In a new terminal
cd meditrack-v2/backend
python test_api.py
```

**Option 2: Use the interactive docs**
1. Open browser: http://localhost:8000/docs
2. Try the `/health` endpoint
3. Try the `/api/wounds/analyze` endpoint with a sample image

**Option 3: Use curl**
```bash
curl -X POST http://localhost:8000/api/wounds/analyze \
  -F "file=@data/samples/wound_moderate.jpg" \
  -F "patient_id=test_patient"
```

---

## 📊 API Documentation

### Interactive Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

**1. Health Check**
```bash
GET /health
```
Returns server status

**2. Analyze Wound**
```bash
POST /api/wounds/analyze
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- patient_id: String (default: "default_patient")

Response:
{
  "scan_id": "uuid",
  "patient_id": "test_patient",
  "scan_date": "2025-01-08T...",
  "metrics": {
    "wound_area_pixels": 12500,
    "wound_area_cm2": 12.5,
    "redness_index": 0.65,
    "edge_sharpness": 0.82,
    "healing_score": 72.3
  },
  "analysis": {
    "risk_level": "medium",
    "summary": "Your wound measures 12.5 cm²...",
    "recommendations": "Keep wound clean..."
  },
  "image_url": "/api/wounds/{scan_id}/image",
  "annotated_image_url": "/api/wounds/{scan_id}/annotated"
}
```

**3. Get Patient History**
```bash
GET /api/wounds/patient/{patient_id}/history
```

**4. Get Statistics**
```bash
GET /api/wounds/stats/overview
```

---

## 🧪 Testing the CV Pipeline

### Test with Different Severities

```bash
# Test mild wound
curl -X POST http://localhost:8000/api/wounds/analyze \
  -F "file=@data/samples/wound_mild.jpg" \
  -F "patient_id=test_patient"

# Test moderate wound
curl -X POST http://localhost:8000/api/wounds/analyze \
  -F "file=@data/samples/wound_moderate.jpg" \
  -F "patient_id=test_patient"

# Test severe wound
curl -X POST http://localhost:8000/api/wounds/analyze \
  -F "file=@data/samples/wound_severe.jpg" \
  -F "patient_id=test_patient"
```

### Check Results in Database

```bash
# Using Python
python
>>> from app.database import SessionLocal
>>> from app.schemas import WoundScan
>>> db = SessionLocal()
>>> scans = db.query(WoundScan).all()
>>> for scan in scans:
...     print(f"{scan.id}: Score={scan.healing_score}, Risk={scan.risk_level}")
```

---

## 🔍 Understanding the Metrics

### Wound Area (cm²)
- **What**: Physical size of the wound
- **Range**: 0+ cm²
- **Interpretation**: Smaller is better (wound closing)

### Redness Index (0-1)
- **What**: Inflammation/redness level
- **Range**: 0 (minimal) to 1 (high inflammation)
- **Interpretation**: Lower is better

### Edge Sharpness (0-1)
- **What**: Boundary definition quality
- **Range**: 0 (fuzzy) to 1 (sharp)
- **Interpretation**: Higher is better (well-defined)

### Healing Score (0-100)
- **What**: Composite metric
- **Range**: 0 (poor) to 100 (excellent)
- **Calculation**: 
  - 40% area score (smaller is better)
  - 40% redness score (lower is better)
  - 20% edge score (sharper is better)

### Risk Level
- **low**: Area <10cm², redness <0.5, score >60
- **medium**: Area 10-25cm², redness 0.5-0.7, score 30-60
- **high**: Area >25cm², redness >0.7, score <30

---

## 🐛 Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Verify Python version: `python --version` (need 3.11+)
- Check API keys in `.env`

### CV analysis fails
- Ensure OpenCV is installed: `pip install opencv-python-headless`
- Check image file is valid (JPEG, PNG)
- Check file size (max 10MB)

### LLM analysis fails
- Verify Groq API key is correct
- Check internet connection
- Look at server logs for detailed error

### Database errors
- Delete `data/meditrack.db` and restart server
- Check write permissions in `data/` directory

---

## ⏭️ Next Steps

### 1. Test the Backend Thoroughly
- Upload multiple images
- Check patient history
- Verify metrics make sense

### 2. Frontend Development (Coming Next)
- Streamlit UI for easy testing
- Image upload interface
- History visualization
- Healing progress charts

### 3. Docker Deployment
- Containerize application
- Deploy to Render/Railway
- Set up environment variables

---

## 📞 Support

If you encounter issues:
1. Check server logs for errors
2. Verify all dependencies installed
3. Ensure API keys are configured
4. Try the `/docs` endpoint for interactive testing

---


Next: Build the Streamlit frontend to make it user-friendly.
