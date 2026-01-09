# 🩹 MediTrack v2.0

**AI-Powered Wound Healing Monitor with Multi-Factor Classification**

An advanced wound analysis system using computer vision and LLMs to classify wound types, track healing progress, and provide AI-powered care recommendations.

![MediTrack Demo](docs/demo-screenshot.png)

---

## ✨ Features

### 🔬 Multi-Factor Wound Classification
- **7 Wound Types Detected**: Surgical incisions, lacerations, burns, pressure ulcers, diabetic ulcers, abrasions, punctures
- **6+ Classification Features**: Aspect ratio, circularity, solidity, edge smoothness, straight edge detection, suture detection
- **Smart Measurements**: Automatic detection of measurement type (linear vs area-based)

### 🤖 AI-Powered Analysis
- **Computer Vision Pipeline**: HSV/LAB color space segmentation, Otsu thresholding, morphological operations
- **LLM Integration**: Groq (Llama 3.1) and Google Gemini for patient-friendly summaries
- **Risk Assessment**: Automatic classification (low/medium/high risk)

### 📊 Progress Tracking
- **Healing Score**: Composite metric (0-100) based on area, redness, and edge quality
- **Historical Data**: Track wounds over time with visual charts
- **Patient Management**: Multi-patient support with unique IDs

### 🎨 Professional UI
- **Clean Medical Interface**: Inspired by modern healthcare applications
- **Responsive Design**: Works on desktop and mobile
- **Real-time Analysis**: 10-20 second processing time

---

## 🏗️ Architecture
```
meditrack-v2/
├── backend/              # FastAPI REST API
│   ├── app/
│   │   ├── api/         # API endpoints
│   │   ├── services/    # CV & LLM services
│   │   ├── models.py    # Pydantic models
│   │   └── schemas.py   # Database schemas
│   ├── data/            # Database & uploads
│   └── requirements.txt
├── frontend/            # Streamlit UI
│   ├── app.py
│   └── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Msundara19/meditrack-v2.git
cd meditrack-v2
```

2. **Set up backend**
```bash
cd backend
pip install -r requirements.txt

# Set environment variables (optional)
# GROQ_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here

# Start backend server
python -m app.main
```

3. **Set up frontend** (new terminal)
```bash
cd frontend
pip install -r requirements.txt

# Start frontend
streamlit run app.py
```

4. **Access the application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🔬 Technical Details

### Computer Vision Pipeline

1. **Preprocessing**
   - Resize (max 1024px)
   - Non-local means denoising
   - Color space conversion (HSV, LAB)

2. **Segmentation**
   - Red hue detection (HSV: 0-10°, 170-180°)
   - LAB a* channel thresholding
   - Morphological operations (closing, opening)
   - Largest contour selection

3. **Feature Extraction**
   - Aspect ratio (length/width)
   - Circularity: 4π×Area/Perimeter²
   - Solidity: Area/ConvexHullArea
   - Edge smoothness (polygon approximation)
   - Straight edge detection (segment analysis)
   - Suture detection (blue/dark color ranges)

4. **Classification**
   - Multi-factor decision tree
   - Threshold-based rules optimized for clinical accuracy

### LLM Integration

- **Primary**: Groq (llama-3.1-8b-instant)
- **Fallback**: Google Gemini
- **Prompt Engineering**: Medical context, previous scan comparison
- **Output**: Risk level, patient summary, care recommendations

---

## 📊 Wound Classification Rules

| Wound Type | Key Features | Measurement |
|------------|--------------|-------------|
| **Surgical Incision** | Aspect ≥1.5, straight edges, high smoothness | Length × Width |
| **Laceration** | Aspect ≥3.0, rough edges | Length × Width |
| **Burn** | Irregular shape, aspect <2.0 | Area |
| **Pressure Ulcer** | Large (>15cm²), irregular, low solidity | Area |
| **Diabetic Ulcer** | Medium size, circular (>0.65) | Area |
| **Abrasion** | Shallow, irregular | Area |
| **Puncture** | Small (<2cm²), circular | Area |

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI 0.115.0
- OpenCV 4.10.0
- SQLAlchemy 2.0.36
- Groq SDK 0.11.0
- Google Generative AI 0.8.3

**Frontend:**
- Streamlit 1.40.0
- Plotly 5.24.1
- Pandas 2.2.3

**Database:**
- SQLite (development)
- PostgreSQL-ready (production)

---



---

## ⚠️ Disclaimer

**FOR EDUCATIONAL PURPOSES ONLY**

This is a demonstration project and should **NOT** be used for:
- Medical diagnosis
- Treatment decisions
- Clinical care

Always consult qualified healthcare professionals for medical advice.

---



## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## 📧 Contact

- **Developer**: Meenakshi Sridharan Sundaram
- **GitHub**: [@Msundara19](https://github.com/Msundara19)
- **Email**: msridharansundaram@hawk.illinoistech.edu

---

## 🙏 Acknowledgments

- Anthropic Claude for development assistance
- OpenCV community
- Groq & Google for LLM APIs

---

**⭐ Star this repo if you find it helpful!**
```

---