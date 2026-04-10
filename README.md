# MediTrack v2.0

**Patient-First Wound Monitoring — AI-Powered Analysis Between Clinic Visits**

<img width="1084" height="914" alt="image" src="https://github.com/user-attachments/assets/9dce61ab-c851-4633-bd29-17880b1f2c35" />

> **Live Demo:** [meditrack-v2.streamlit.app](https://meditrack-v2.streamlit.app)

MediTrack is a portfolio/educational project that lets patients photograph a wound, get a plain-English explanation of how it is healing, and track progress over time. It is built for the gap between clinic visits — not as a clinical documentation tool.

---

## The Problem It Solves

Every commercial wound monitoring app (WoundGenius, Minuteful, eKare, Tissue Analytics) is designed for clinicians — they output clinical metrics, integrate with EMRs, and require trained staff. None of them are designed for the patient sitting at home wondering whether their wound looks worse than yesterday.

MediTrack's angle: **photograph your wound, get a plain-English explanation, know when to call your doctor.** No training required, no calibration sticker, no clinical background needed.

---

## What It Does

- Classifies wound type across 7 classes (surgical incision, laceration, burn, pressure ulcer, diabetic ulcer, abrasion, venous ulcer) using EfficientNet-B0 (91.5% accuracy, 0.989 macro AUC) with a shape-based heuristic fallback when ML confidence falls below 60%
- Segments the wound region using an OpenCV pipeline (HSV/LAB color thresholding + center-bias masking + morphological cleanup) and extracts area, redness, and edge quality metrics
- Generates a patient-friendly plain-English summary via Groq (Llama 3.1-8b) — no clinical jargon, no raw numbers exposed to the patient
- Detects healing trajectory using linear regression across scans and flags worsening wounds
- Produces a downloadable PDF report with annotated image, metrics, AI assessment, and wound-type-specific care recommendations
- Prevents duplicate uploads via MD5 hashing; cross-scan comparisons are gated by wound type and recency (same type, within 60 days) to avoid comparing different wounds

---

## Architecture

```
frontend/app.py          Streamlit UI (wide layout, Montserrat theme)
        │
        │  HTTP (REST)
        ▼
backend/app/
  api/wounds.py          FastAPI endpoints — upload, analyze, history, PDF
  services/
    cv_service.py        OpenCV segmentation pipeline
    wound_classifier.py  Shape-based heuristic classifier
    ml_classifier.py     EfficientNet-B0 inference (graceful fallback if model missing)
    llm_service.py       Groq / Gemini prompt builder + patient-friendly summary
    report_service.py    ReportLab PDF generation
    storage_service.py   Local filesystem (dev) / Supabase Storage (production)
  schemas.py             SQLAlchemy models (SQLite dev / PostgreSQL production)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115, Uvicorn |
| Computer Vision | OpenCV 4.10, NumPy, Pillow |
| ML Classification | PyTorch, EfficientNet-B0 (via timm) |
| LLM | Groq llama-3.1-8b-instant (primary), Google Gemini (fallback) |
| Database | SQLite (dev), PostgreSQL via Supabase (production) |
| Storage | Local filesystem (dev), Supabase Storage (production) |
| Frontend | Streamlit 1.40, Plotly, Pandas |
| PDF | ReportLab |
| Deployment | Railway (backend), Streamlit Cloud (frontend), Supabase (DB + storage) |

---

## ML Model Performance

EfficientNet-B0 fine-tuned on 7 wound classes, evaluated on 375 held-out samples.

| Wound Type | Precision | Recall | F1 |
|---|---|---|---|
| Surgical Incision | 0.98 | 0.83 | 0.90 |
| Laceration | 1.00 | 0.94 | 0.97 |
| Burn | 0.90 | 0.95 | 0.93 |
| Pressure Ulcer | 0.90 | 0.91 | 0.91 |
| Diabetic Ulcer | 0.82 | 0.88 | 0.85 |
| Abrasion | 0.96 | 1.00 | 0.98 |
| Venous Ulcer | 0.92 | 0.97 | 0.95 |
| **Overall** | **0.93** | **0.93** | **0.93** |

Overall accuracy: **91.5%** — Macro AUC: **0.989**

The model classifies wound *type*. Wound region *segmentation* is performed by the OpenCV pipeline, not a trained segmentation model.

---

## Quick Start

**Prerequisites:** Python 3.11+, a Groq API key (free at console.groq.com)

```bash
git clone https://github.com/Msundara19/meditrack-v2.git
cd meditrack-v2
```

**Backend**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python -m app.main
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Frontend** (new terminal)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
# UI at http://localhost:8501
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for LLM summaries |
| `GEMINI_API_KEY` | No | Google Gemini fallback (optional) |
| `DATABASE_URL` | No | PostgreSQL URL for production (defaults to SQLite) |
| `SUPABASE_URL` | No | Supabase project URL for image storage |
| `SUPABASE_KEY` | No | Supabase anon key (legacy `eyJ...` format) |
| `SUPABASE_BUCKET` | No | Storage bucket name (default: `wound-images`) |

Without `SUPABASE_URL`/`SUPABASE_KEY`, images are stored locally. Without `DATABASE_URL`, SQLite is used. Both fall back automatically — no config needed for local development.

---

## Production Deployment

Stack: **Railway** (FastAPI backend) + **Supabase** (PostgreSQL + Storage) + **Streamlit Cloud** (frontend).

1. Create a Supabase project → copy PostgreSQL session pooler URI and legacy anon key (`eyJ...`)
2. Create a public storage bucket named `wound-images` → add an INSERT + SELECT policy with `true`
3. Deploy `backend/` on Railway → set all env vars → set start command to `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`
4. Deploy `frontend/app.py` on Streamlit Cloud (Python 3.11) → set secret `BACKEND_URL = "https://your-railway-url"`

---

## Honest Limitations

- **Not a medical device.** Not FDA-cleared. Not validated on clinical data. For educational demonstration only.
- **Segmentation is heuristic.** The OpenCV pipeline works best on well-lit wound photos with a plain background. Complex backgrounds cause over-segmentation. A trained UNet would be the proper fix.
- **Measurements are approximate.** Pixel-to-cm conversion uses a user-selected distance estimate, not a depth sensor or calibration sticker.
- **Single-wound assumption.** The system cannot verify that two scans are of the same physical wound — it gates comparisons by wound type and recency as a proxy.
- **LLM summaries are not medical advice.** Groq/Gemini output is generated text and can be wrong.

---

## Disclaimer

This project is for educational and portfolio purposes only. It must not be used for medical diagnosis, treatment decisions, or clinical care. Always consult a qualified healthcare professional.

---

## Contact

**Meenakshi Sridharan Sundaram**
GitHub: [@Msundara19](https://github.com/Msundara19)
Email: msridharansundaram@hawk.illinoistech.edu
