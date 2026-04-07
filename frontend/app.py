"""
MediTrack v2.0 — AI-Powered Wound Monitoring
UI: Montserrat, navy/blue theme, white cards on light-blue gradient.
"""
import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MediTrack — Wound Monitoring AI",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { font-family: 'Montserrat', sans-serif !important; }

/* ── App shell ── */
.stApp { background: linear-gradient(150deg, #EEF2FF 0%, #F5F7FF 55%, #EBF0FF 100%); }
#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton { display: none !important; }

.block-container {
    padding: 0 3rem 3rem !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}
section[data-testid="stMain"] > div:first-child {
    display: flex;
    justify-content: center;
}
section[data-testid="stMain"] > div:first-child > div {
    max-width: 1100px !important;
    width: 100% !important;
}

/* ── Nav bar ── */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.4rem 0 1rem;
    margin-bottom: 0.5rem;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-icon {
    width: 38px; height: 38px;
    background: #2563EB;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; line-height: 1;
}
.nav-brand { font-size: 1.25rem; font-weight: 800; color: #1E2A78; letter-spacing: -0.5px; }
.nav-pill {
    background: #EEF2FF; border: 1.5px solid #C7D7FE;
    color: #2563EB; font-size: 0.6rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
}
.nav-gh {
    background: white; border: 1.5px solid #E2E8F0;
    color: #1E2A78; font-size: 0.82rem; font-weight: 600;
    padding: 7px 18px; border-radius: 8px;
    text-decoration: none; transition: all 0.2s;
}
.nav-gh:hover { border-color: #2563EB; color: #2563EB; }

/* ── Hero ── */
.hero { text-align: center; padding: 2.8rem 0 2rem; }
.hero-tag {
    display: inline-block; background: white;
    border: 1.5px solid #C7D7FE; color: #2563EB;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; padding: 5px 18px;
    border-radius: 20px; margin-bottom: 1.4rem;
}
.hero-h1 {
    font-size: 3.4rem; font-weight: 900; color: #1E2A78;
    line-height: 1.08; margin: 0; letter-spacing: -2px;
}
.hero-h1 .blue { color: #2563EB; }
.hero-sub {
    color: #64748B; font-size: 1rem; font-weight: 400;
    max-width: 540px; margin: 1.3rem auto 0; line-height: 1.65;
}

/* ── Stats bar ── */
.stats-bar {
    background: white; border-radius: 18px;
    border: 1px solid #E0E8FF;
    box-shadow: 0 4px 28px rgba(37,99,235,0.07);
    display: flex; justify-content: space-around; align-items: center;
    padding: 1.5rem 2rem; margin: 2.2rem auto 2.5rem;
    max-width: 780px;
}
.stat { text-align: center; }
.stat-val { font-size: 1.9rem; font-weight: 900; color: #1E2A78; letter-spacing: -1.5px; line-height: 1; }
.stat-lbl { font-size: 0.62rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #94A3B8; margin-top: 5px; }
.sdiv { width: 1px; height: 42px; background: #E8EEFF; }

/* ── Section label ── */
.sec-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 1.2rem;
}

/* ── Cards (column containers) ── */
[data-testid="column"] {
    background: white !important;
    border-radius: 18px !important;
    border: 1px solid #E0E8FF !important;
    box-shadow: 0 4px 28px rgba(37,99,235,0.07) !important;
    padding: 1.8rem 1.6rem !important;
}

/* Nested columns (delta, comparison) — lighter card */
[data-testid="column"] [data-testid="column"] {
    background: #F8FAFF !important;
    border: 1px solid #E8EEFF !important;
    box-shadow: none !important;
    padding: 1rem 0.9rem !important;
    border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #2563EB !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 0.92rem !important;
    padding: 0.65rem 2rem !important; width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 6px 22px rgba(37,99,235,0.42) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    background: white !important; color: #2563EB !important;
    border: 1.5px solid #2563EB !important; border-radius: 10px !important;
    font-weight: 700 !important; width: 100% !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover { background: #EEF2FF !important; }

/* ── Inputs ── */
.stTextInput label {
    font-size: 0.62rem !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: #94A3B8 !important;
}
.stTextInput input {
    background: #F8FAFF !important; border: 1.5px solid #E0E8FF !important;
    border-radius: 9px !important; font-weight: 500 !important;
    color: #1E2A78 !important; font-size: 0.9rem !important;
}
.stTextInput input:focus { border-color: #2563EB !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #F8FAFF !important;
    border: 2px dashed #C7D7FE !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] label {
    font-size: 0.85rem !important; font-weight: 600 !important;
    color: #1E2A78 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: white; border-radius: 10px; padding: 4px;
    gap: 0; border: 1px solid #E0E8FF;
    width: fit-content; margin: 0 auto 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748B;
    border-radius: 7px; font-weight: 600;
    font-size: 0.82rem; padding: 7px 22px; letter-spacing: 0.2px;
}
.stTabs [aria-selected="true"] { background: #2563EB !important; color: white !important; }

/* ── Metric values ── */
[data-testid="stMetric"] {
    background: #F8FAFF !important; border: 1px solid #E0E8FF !important;
    border-radius: 10px !important; padding: 0.9rem !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.6rem !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: #94A3B8 !important;
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 900 !important; color: #1E2A78 !important; }

/* ── Badges ── */
.badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; }
.b-high  { background: #FEE2E2; color: #991B1B; }
.b-medium{ background: #FEF3C7; color: #92400E; }
.b-low   { background: #D1FAE5; color: #065F46; }
.b-type  { background: #EEF2FF; color: #2563EB; }
.b-ml    { background: #F0FDF4; color: #166534; }

/* ── Empty result state ── */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 340px; text-align: center;
    color: #CBD5E1;
}
.empty-state .es-icon { font-size: 2.8rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state p { font-size: 0.88rem; font-weight: 500; color: #CBD5E1; max-width: 200px; margin: 0; }

/* ── Emergency notice ── */
.emergency {
    background: #FFFBEB; border: 1px solid #FCD34D;
    border-radius: 10px; padding: 9px 16px;
    font-size: 0.78rem; font-weight: 600; color: #92400E;
    text-align: center; margin-bottom: 2rem;
}

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; font-size: 0.85rem !important; }

/* ── Expanders ── */
.streamlit-expanderHeader { font-weight: 700 !important; color: #1E2A78 !important; }

/* ── Plotly ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Horizontal rule ── */
hr { border-color: #E0E8FF !important; }

/* ── Footer ── */
.footer {
    text-align: center; font-size: 0.72rem; font-weight: 500;
    color: #94A3B8; padding: 2.5rem 0 1rem; letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "patient_id" not in st.session_state:
    st.session_state.patient_id = "patient_demo"
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

# ── Nav bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-left">
    <div class="nav-icon">🩹</div>
    <span class="nav-brand">MediTrack</span>
    <span class="nav-pill">ML Powered</span>
  </div>
  <a class="nav-gh" href="https://github.com/Msundara19/meditrack-v2" target="_blank">
    View on GitHub →
  </a>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Wound Monitoring AI</div>
  <h1 class="hero-h1">
    Precise analysis.<br>
    <span class="blue">Measurable healing.</span>
  </h1>
  <p class="hero-sub">
    Upload a wound image and get AI-powered classification,
    healing trajectory prediction, side-by-side progress comparison,
    and a downloadable clinical report — in seconds.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
  <div class="stat"><div class="stat-val">91.5%</div><div class="stat-lbl">Test Accuracy</div></div>
  <div class="sdiv"></div>
  <div class="stat"><div class="stat-val">7</div><div class="stat-lbl">Wound Types</div></div>
  <div class="sdiv"></div>
  <div class="stat"><div class="stat-val">0.989</div><div class="stat-lbl">Macro AUC</div></div>
  <div class="sdiv"></div>
  <div class="stat"><div class="stat-val">&lt;1s</div><div class="stat-lbl">Inference</div></div>
</div>
""", unsafe_allow_html=True)

# ── Emergency notice ──────────────────────────────────────────────────────────
st.markdown("""
<div class="emergency">
  ⚠️ If this is a medical emergency, call 911 or your local emergency number immediately.
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["New Scan", "History", "About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — NEW SCAN
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left column: upload ──
    with col_left:
        st.markdown('<div class="sec-label">New Scan</div>', unsafe_allow_html=True)

        patient_id = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id,
            placeholder="e.g. patient_001",
        )
        st.session_state.patient_id = patient_id

        # Calibration selector
        st.markdown('<div class="sec-label" style="margin-top:0.8rem">Camera Distance</div>', unsafe_allow_html=True)
        cal_option = st.selectbox(
            "Camera distance from wound",
            options=[
                "Very close — ~15 cm",
                "Close — ~25 cm",
                "Normal — ~35 cm",
                "Far — ~50 cm",
            ],
            index=1,
            help="Approximate distance affects measurement accuracy. Place a coin or ruler in frame for precise results.",
            label_visibility="collapsed",
        )
        cal_map = {
            "Very close — ~15 cm": 0.015,
            "Close — ~25 cm":      0.022,
            "Normal — ~35 cm":     0.030,
            "Far — ~50 cm":        0.044,
        }
        calibration_factor = cal_map[cal_option]
        st.caption("⚠️ Measurements are approximate. Place a coin or ruler in frame for accurate sizing.")

        uploaded_file = st.file_uploader(
            "Drop an image or click to upload",
            type=["jpg", "jpeg", "png"],
            help="JPG, PNG · any size",
            label_visibility="visible",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded image")

        analyze_clicked = st.button("Analyze Wound", type="primary")

        if analyze_clicked:
            if not uploaded_file:
                st.warning("Please upload a wound image first.")
            else:
                with st.spinner("Analyzing with EfficientNet-B0 + LLM…"):
                    try:
                        response = requests.post(
                            f"{API_URL}/api/wounds/analyze",
                            files={"file": ("wound.jpg", uploaded_file.getvalue(), "image/jpeg")},
                            data={"patient_id": patient_id, "calibration_factor": calibration_factor},
                            timeout=60,
                        )
                        if response.status_code == 200:
                            st.session_state.scan_result = response.json()
                            st.session_state.analysis_complete = True
                            st.rerun()
                        else:
                            st.error(f"Analysis failed: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                        st.info("Make sure the backend is running: `python -m app.main`")

    # ── Right column: results ──
    with col_right:
        st.markdown('<div class="sec-label">Analysis Result</div>', unsafe_allow_html=True)

        if not st.session_state.analysis_complete or not st.session_state.scan_result:
            st.markdown("""
            <div class="empty-state">
              <div class="es-icon">🔬</div>
              <p>Upload an image and click Analyze to see results</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            result = st.session_state.scan_result
            metrics = result["metrics"]
            analysis = result["analysis"]

            # Risk + wound type badges
            risk = analysis["risk_level"]
            risk_cls = {"high": "b-high", "medium": "b-medium", "low": "b-low"}.get(risk, "b-type")
            wound_display = metrics.get("wound_type", "unknown").replace("_", " ").title()
            classified_by = metrics.get("classified_by", "heuristic")
            ml_conf = metrics.get("ml_confidence")
            conf_str = f" · {ml_conf*100:.0f}% confidence" if ml_conf else ""

            st.markdown(f"""
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1rem;">
              <span class="badge {risk_cls}">{risk.upper()} RISK</span>
              <span class="badge b-type">🔬 {wound_display}</span>
              <span class="badge b-ml">{classified_by.upper()}{conf_str}</span>
            </div>
            """, unsafe_allow_html=True)

            # Key metrics
            is_linear = metrics.get("measurement_type") == "linear"
            if is_linear and metrics.get("length_cm"):
                m1_label, m1_val = "Length", f"{metrics['length_cm']:.1f} cm"
                m2_label, m2_val = "Width", f"{metrics['width_cm']:.1f} cm"
            else:
                m1_label, m1_val = "Area", f"{metrics['wound_area_cm2']:.1f} cm²"
                m2_label, m2_val = "Redness", f"{metrics['redness_index']:.2f}"

            c1, c2, c3 = st.columns(3)
            c1.metric("Healing Score", f"{metrics['healing_score']:.0f}/100")
            c2.metric(m1_label, m1_val)
            c3.metric(m2_label, m2_val)

            # AI summary
            st.markdown("**AI Assessment**")
            st.info(analysis["summary"])

            st.markdown("**Care Recommendations**")
            st.warning(analysis["recommendations"])

            # Classification detail expander
            with st.expander("Classification detail"):
                conf_scores = metrics.get("confidence_scores")
                if conf_scores:
                    sorted_scores = sorted(conf_scores.items(), key=lambda x: x[1], reverse=True)
                    fig_bar = go.Figure(go.Bar(
                        x=[s[1] for s in sorted_scores],
                        y=[s[0].replace("_", " ").title() for s in sorted_scores],
                        orientation="h",
                        marker_color=["#2563EB" if i == 0 else "#C7D7FE" for i in range(len(sorted_scores))],
                    ))
                    fig_bar.update_layout(
                        height=240, margin=dict(l=0, r=0, t=10, b=10),
                        xaxis=dict(range=[0, 1], tickformat=".0%"),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Montserrat", color="#1E2A78"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.json({
                        "wound_type": metrics.get("wound_type"),
                        "classified_by": classified_by,
                        "circularity": f"{metrics.get('circularity', 0):.3f}" if metrics.get("circularity") else "N/A",
                        "aspect_ratio": f"{metrics.get('aspect_ratio', 0):.2f}" if metrics.get("aspect_ratio") else "N/A",
                    })

            # PDF download
            pdf_resp = requests.get(
                f"{API_URL}/api/wounds/{result['scan_id']}/report", timeout=30
            )
            if pdf_resp.status_code == 200:
                st.download_button(
                    "Download PDF Report",
                    data=pdf_resp.content,
                    file_name=f"meditrack_{result['scan_id'][:8]}.pdf",
                    mime="application/pdf",
                )

            if st.button("New Scan"):
                st.session_state.analysis_complete = False
                st.session_state.scan_result = None
                st.rerun()

    # ── Side-by-side comparison (below both columns) ──
    if st.session_state.analysis_complete and st.session_state.scan_result:
        try:
            hist = requests.get(
                f"{API_URL}/api/wounds/patient/{st.session_state.patient_id}/history",
                timeout=10,
            ).json()
            scans = hist.get("scans", [])
            if len(scans) >= 2:
                cur, prev = scans[0], scans[1]
                st.markdown("---")
                st.markdown(
                    '<div class="sec-label" style="margin-top:1.5rem;">Progress Comparison</div>',
                    unsafe_allow_html=True,
                )
                img_col1, img_col2 = st.columns(2, gap="large")
                with img_col1:
                    st.caption(f"Current — {cur['scan_date'][:10]}")
                    r = requests.get(f"{API_URL}/api/wounds/{cur['id']}/annotated", timeout=15)
                    if r.status_code == 200:
                        st.image(r.content, use_container_width=True)
                with img_col2:
                    st.caption(f"Previous — {prev['scan_date'][:10]}")
                    r = requests.get(f"{API_URL}/api/wounds/{prev['id']}/annotated", timeout=15)
                    if r.status_code == 200:
                        st.image(r.content, use_container_width=True)

                # Delta metrics
                cm, pm = cur["metrics"], prev["metrics"]
                d1, d2, d3 = st.columns(3, gap="large")

                def _delta_html(label, curr_val, prev_val, fmt, lower_is_better=False):
                    diff = curr_val - prev_val
                    is_good = (diff < 0) if lower_is_better else (diff >= 0)
                    color = "#059669" if is_good else "#DC2626"
                    sign = "+" if diff >= 0 else ""
                    return (
                        f"<div style='font-size:0.62rem;font-weight:700;letter-spacing:2px;"
                        f"text-transform:uppercase;color:#94A3B8;margin-bottom:4px'>{label}</div>"
                        f"<div style='font-size:1.5rem;font-weight:900;color:#1E2A78;letter-spacing:-0.5px'>"
                        f"{curr_val:{fmt}}</div>"
                        f"<div style='font-size:0.8rem;font-weight:700;color:{color}'>"
                        f"{sign}{diff:{fmt}} vs previous</div>"
                    )

                with d1:
                    st.markdown(_delta_html("Healing Score", cm["healing_score"], pm["healing_score"], ".0f"), unsafe_allow_html=True)
                with d2:
                    st.markdown(_delta_html("Area (cm²)", cm["wound_area_cm2"], pm["wound_area_cm2"], ".1f", lower_is_better=True), unsafe_allow_html=True)
                with d3:
                    st.markdown(_delta_html("Redness", cm["redness_index"], pm["redness_index"], ".2f", lower_is_better=True), unsafe_allow_html=True)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    h_col, _ = st.columns([1, 1], gap="large")
    with h_col:
        st.markdown('<div class="sec-label">Patient History</div>', unsafe_allow_html=True)
        history_pid = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id,
            key="history_patient",
            placeholder="Enter patient ID",
        )
        load = st.button("Load History", key="load_btn")

    if load:
        with st.spinner("Fetching history…"):
            try:
                resp = requests.get(
                    f"{API_URL}/api/wounds/patient/{history_pid}/history", timeout=10
                )
                if resp.status_code != 200:
                    st.error(f"Failed to load: {resp.text}")
                else:
                    data = resp.json()
                    if data["scan_count"] == 0:
                        st.info(f"No scans found for patient: {history_pid}")
                    else:
                        # Trajectory banner
                        traj = data.get("healing_trajectory", {})
                        if traj.get("available"):
                            trend, msg = traj.get("trend", ""), traj.get("message", "")
                            slope = traj.get("slope_per_scan", 0)
                            if trend == "good":
                                st.success(f"🟢  {msg}")
                            elif trend == "improving":
                                st.info(f"📈  {msg}  (+{slope:.1f} pts/scan)")
                            else:
                                st.warning(f"📉  {msg}")

                        scans = data["scans"]
                        df = pd.DataFrame([{
                            "Date": s["scan_date"][:10],
                            "Type": s["metrics"].get("wound_type", "unknown").replace("_", " ").title(),
                            "Healing Score": s["metrics"]["healing_score"],
                            "Area (cm²)": s["metrics"]["wound_area_cm2"],
                            "Risk": s["analysis"]["risk_level"].upper(),
                        } for s in scans])

                        # Chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df["Healing Score"],
                            mode="lines+markers", name="Healing Score",
                            line=dict(color="#2563EB", width=3),
                            marker=dict(size=9, color="#2563EB",
                                        line=dict(color="white", width=2)),
                            fill="tozeroy",
                            fillcolor="rgba(37,99,235,0.08)",
                        ))

                        # Add trajectory prediction line if improving
                        if traj.get("available") and traj.get("trend") == "improving":
                            import numpy as np
                            scores = list(reversed([s["metrics"]["healing_score"] for s in scans]))
                            x_vals = np.arange(len(scores))
                            slope_v, intercept = np.polyfit(x_vals, scores, 1)
                            n_extra = traj.get("scans_to_target", 3) + 2
                            x_pred = np.arange(len(scores) - 1, len(scores) + n_extra)
                            y_pred = slope_v * x_pred + intercept
                            fig.add_trace(go.Scatter(
                                x=x_pred, y=np.clip(y_pred, 0, 100),
                                mode="lines", name="Predicted trajectory",
                                line=dict(color="#2563EB", width=2, dash="dot"),
                            ))
                            fig.add_hline(
                                y=85, line_dash="dash",
                                line_color="#059669", line_width=1.5,
                                annotation_text="Good healing threshold",
                                annotation_font_color="#059669",
                            )

                        fig.update_layout(
                            xaxis_title="Scan #", yaxis_title="Healing Score (0–100)",
                            yaxis=dict(range=[0, 105]),
                            hovermode="x unified",
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=380,
                            font=dict(family="Montserrat", color="#1E2A78"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=0, r=0, t=30, b=0),
                        )
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(gridcolor="#F0F4FF", gridwidth=1)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown('<div class="sec-label" style="margin-top:1rem;">Scan Records</div>', unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    a_col, _ = st.columns([1, 1], gap="large")
    with a_col:
        st.markdown('<div class="sec-label">About MediTrack</div>', unsafe_allow_html=True)
        st.markdown("""
MediTrack is an AI-powered wound monitoring system combining a trained computer
vision model with LLM-generated insights.

**Model**
- EfficientNet-B0 fine-tuned on 7 wound classes
- 91.5% test accuracy · 0.989 macro AUC
- Heuristic fallback when ML confidence < 60%

**Features**
- Wound type classification (7 classes)
- Healing score tracking over time
- Side-by-side progress comparison
- Healing trajectory prediction
- Downloadable PDF reports
- LLM-powered patient-friendly summaries

**Wound types detected**
Surgical incision · Laceration · Burn · Pressure ulcer ·
Diabetic ulcer · Abrasion · Venous ulcer

**Stack**
FastAPI · PyTorch · EfficientNet-B0 · OpenCV · Groq LLM · Streamlit

---

**Developer**: Meenakshi Sridharan Sundaram
**GitHub**: [Msundara19](https://github.com/Msundara19)
        """)

    with _:
        st.markdown('<div class="sec-label">Model Performance</div>', unsafe_allow_html=True)
        perf_data = {
            "Wound Type": ["Laceration", "Abrasion", "Venous Ulcer", "Burn", "Pressure Ulcer", "Surgical Incision", "Diabetic Ulcer"],
            "F1 Score": [0.97, 0.98, 0.95, 0.93, 0.91, 0.90, 0.85],
        }
        fig_perf = go.Figure(go.Bar(
            x=perf_data["F1 Score"],
            y=perf_data["Wound Type"],
            orientation="h",
            marker=dict(
                color=perf_data["F1 Score"],
                colorscale=[[0, "#C7D7FE"], [1, "#2563EB"]],
                showscale=False,
            ),
            text=[f"{v:.2f}" for v in perf_data["F1 Score"]],
            textposition="outside",
        ))
        fig_perf.update_layout(
            xaxis=dict(range=[0.8, 1.02], tickformat=".0%"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=300,
            margin=dict(l=0, r=40, t=10, b=10),
            font=dict(family="Montserrat", color="#1E2A78"),
        )
        fig_perf.update_xaxes(showgrid=False)
        fig_perf.update_yaxes(showgrid=False)
        st.plotly_chart(fig_perf, use_container_width=True)

        st.markdown("""
<div style="background:#F8FAFF;border:1px solid #E0E8FF;border-radius:10px;padding:1rem;margin-top:0.5rem">
  <div style="font-size:0.6rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#94A3B8;margin-bottom:0.5rem">Disclaimer</div>
  <div style="font-size:0.78rem;color:#64748B;line-height:1.6">
    For <strong>educational purposes only</strong>. Not intended for medical diagnosis, treatment decisions, or clinical care. Always consult qualified healthcare professionals.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  MediTrack v2.0 · AI Wound Monitoring · Educational Project · Meenakshi Sridharan Sundaram
</div>
""", unsafe_allow_html=True)
