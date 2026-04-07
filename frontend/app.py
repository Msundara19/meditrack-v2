"""
MediTrack - Professional Medical UI with Wound Classification
Clean, light theme inspired by Doctronic
"""
import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"

# Page config - LIGHT THEME
st.set_page_config(
    page_title="MediTrack - Wound Monitoring",
    page_icon="🩹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional Medical UI CSS - Doctronic Style
st.markdown("""
<style>
    /* Global Styles - Doctronic Colors */
    :root {
        --background: #FFFFFF;
        --text: #4A5568;
        --primary: #007bff;
        --card-shadow: rgba(0, 0, 0, 0.08);
        --risk-high: #FEE2E2;
        --risk-med: #FEF3C7;
        --risk-low: #D1FAE5;
    }
    
    /* Main background */
    .stApp {
        background-color: var(--background);
    }
    
    /* All text */
    .stApp, .stMarkdown, p, span, div, label {
        color: var(--text) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #718096 !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #F7FAFC;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--text);
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: var(--primary) !important;
        box-shadow: 0 1px 3px var(--card-shadow);
    }
    
    /* Emergency banner */
    .emergency-banner {
        background-color: #FEF3C7;
        border: 1px solid #FCD34D;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Buttons - Primary */
    .stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0, 123, 255, 0.3);
        transition: all 0.2s;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #0056b3 !important;
        box-shadow: 0 4px 6px rgba(0, 123, 255, 0.4);
        transform: translateY(-1px);
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        color: var(--text);
        padding: 0.5rem 0.75rem;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #F7FAFC;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    
    div[data-testid="stMetric"] label {
        color: #718096 !important;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 1rem 0;
    }
    
    .risk-high {
        background-color: var(--risk-high);
        color: #991B1B;
    }
    
    .risk-medium {
        background-color: var(--risk-med);
        color: #92400E;
    }
    
    .risk-low {
        background-color: var(--risk-low);
        color: #065F46;
    }
    
    /* Wound type badge */
    .wound-type-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
        background-color: #EBF8FF;
        color: #1565c0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = "patient_demo"
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'scan_result' not in st.session_state:
    st.session_state.scan_result = None

# Header
st.markdown("# MediTrack")
st.markdown('<div class="subtitle">AI-Powered Wound Healing Monitor with Classification</div>', unsafe_allow_html=True)

# Emergency notice
st.markdown("""
<div class="emergency-banner">
    ⚠️ If this is an emergency, call 911 or your local emergency number.
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 New Scan", "📊 History", "ℹ️ About"])

with tab1:
    st.markdown("### Upload Wound Image")
    st.caption("Take a clear photo of the wound in good lighting. Include a reference object (coin/ruler) if possible.")
    
    # Patient ID
    patient_id = st.text_input(
        "Patient ID",
        value=st.session_state.patient_id,
        placeholder="Enter patient identifier",
        help="Unique identifier for tracking this patient's progress"
    )
    st.session_state.patient_id = patient_id
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, PNG (max 10MB)",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Show image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("#### Ready to analyze")
            st.markdown("AI will classify wound type and extract detailed metrics.")
        
        # Analyze button
        if st.button("🔍 Analyze Wound", type="primary"):
            with st.spinner("⏳ Analyzing wound with AI classification... This may take 10-20 seconds..."):
                try:
                    files = {
                        "file": ("wound.jpg", uploaded_file.getvalue(), "image/jpeg")
                    }
                    data = {
                        "patient_id": patient_id
                    }
                    
                    response = requests.post(
                        f"{API_URL}/api/wounds/analyze",
                        files=files,
                        data=data,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.scan_result = result
                        st.session_state.analysis_complete = True
                        st.rerun()
                    else:
                        st.error(f"❌ Analysis failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
                    st.info("💡 Make sure the backend server is running: `py -m app.main`")
    
    # Results display
    if st.session_state.analysis_complete and st.session_state.scan_result:
        result = st.session_state.scan_result
        
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")
        
        # Risk badge
        risk_level = result['analysis']['risk_level']
        risk_class = f"risk-{risk_level}"
        
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="risk-badge {risk_class}">
                {risk_level.upper()} RISK
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Wound type classification
        metrics = result['metrics']
        if 'wound_type' in metrics and metrics['wound_type'] != 'unknown':
            wound_type_display = metrics['wound_type'].replace('_', ' ').title()
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="wound-type-badge">
                    🔬 Wound Type: {wound_type_display}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics - Show different measurements based on wound type
        st.markdown("#### 📏 Key Metrics")
        
        # Check measurement type
        is_linear = metrics.get('measurement_type') == 'linear'
        
        if is_linear and metrics.get('length_cm') and metrics.get('width_cm'):
            # Linear wound (incision/laceration) - show length x width
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Healing Score", f"{metrics['healing_score']:.0f}/100")
            with col2:
                st.metric("Length", f"{metrics['length_cm']:.1f} cm")
            with col3:
                st.metric("Width", f"{metrics['width_cm']:.1f} cm")
            with col4:
                st.metric("Aspect Ratio", f"{metrics.get('aspect_ratio', 0):.1f}")
        else:
            # Area wound (burn/ulcer) - show area
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Healing Score", f"{metrics['healing_score']:.0f}/100")
            with col2:
                st.metric("Wound Area", f"{metrics['wound_area_cm2']:.1f} cm²")
            with col3:
                st.metric("Redness", f"{metrics['redness_index']:.2f}")
        
        # AI Summary
        st.markdown("#### 🤖 AI Assessment")
        st.info(result['analysis']['summary'])
        
        # Recommendations
        st.markdown("#### 💡 Care Recommendations")
        st.warning(result['analysis']['recommendations'])
        
        # Details
        with st.expander("📊 View Detailed Classification Metrics"):
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "Scan ID": result['scan_id'],
                    "Wound Type": metrics.get('wound_type', 'unknown'),
                    "Measurement Type": metrics.get('measurement_type', 'area'),
                })
            with col2:
                st.json({
                    "Circularity": f"{metrics.get('circularity', 0):.3f}" if metrics.get('circularity') else "N/A",
                    "Edge Sharpness": f"{metrics.get('edge_sharpness', 0):.3f}",
                    "Total Pixels": metrics['wound_area_pixels'],
                })
        
        # --- Side-by-side comparison with previous scan ---
        try:
            history_resp = requests.get(
                f"{API_URL}/api/wounds/patient/{patient_id}/history",
                timeout=10
            )
            if history_resp.status_code == 200:
                history_data = history_resp.json()
                scans = history_data.get("scans", [])
                if len(scans) >= 2:
                    current_scan = scans[0]
                    prev_scan = scans[1]
                    st.markdown("---")
                    st.markdown("### 🔄 Comparison with Previous Scan")

                    col_curr, col_prev = st.columns(2)
                    with col_curr:
                        st.caption(f"**Current** — {current_scan['scan_date'][:10]}")
                        img_resp = requests.get(
                            f"{API_URL}/api/wounds/{current_scan['id']}/annotated", timeout=15
                        )
                        if img_resp.status_code == 200:
                            st.image(img_resp.content, use_container_width=True)

                    with col_prev:
                        st.caption(f"**Previous** — {prev_scan['scan_date'][:10]}")
                        img_resp = requests.get(
                            f"{API_URL}/api/wounds/{prev_scan['id']}/annotated", timeout=15
                        )
                        if img_resp.status_code == 200:
                            st.image(img_resp.content, use_container_width=True)

                    # Delta metrics
                    curr_m = current_scan["metrics"]
                    prev_m = prev_scan["metrics"]

                    def delta(curr, prev, label, fmt=".1f", invert=False):
                        diff = curr - prev
                        arrow = "▲" if diff > 0 else "▼"
                        color = "green" if (diff > 0) != invert else "red"
                        return f"{curr:{fmt}} <span style='color:{color}'>{arrow} {abs(diff):{fmt}}</span>"

                    st.markdown("#### Metric Changes")
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        score_diff = curr_m["healing_score"] - prev_m["healing_score"]
                        sign = "+" if score_diff >= 0 else ""
                        color = "green" if score_diff >= 0 else "red"
                        st.markdown(f"**Healing Score**<br>{curr_m['healing_score']:.0f} <span style='color:{color}'>({sign}{score_diff:.0f})</span>", unsafe_allow_html=True)
                    with d2:
                        area_diff = curr_m["wound_area_cm2"] - prev_m["wound_area_cm2"]
                        sign = "+" if area_diff >= 0 else ""
                        color = "green" if area_diff <= 0 else "red"
                        st.markdown(f"**Area (cm²)**<br>{curr_m['wound_area_cm2']:.1f} <span style='color:{color}'>({sign}{area_diff:.1f})</span>", unsafe_allow_html=True)
                    with d3:
                        red_diff = curr_m["redness_index"] - prev_m["redness_index"]
                        sign = "+" if red_diff >= 0 else ""
                        color = "green" if red_diff <= 0 else "red"
                        st.markdown(f"**Redness**<br>{curr_m['redness_index']:.2f} <span style='color:{color}'>({sign}{red_diff:.2f})</span>", unsafe_allow_html=True)
        except Exception:
            pass  # Comparison is non-critical

        # --- PDF download ---
        st.markdown("---")
        pdf_resp = requests.get(
            f"{API_URL}/api/wounds/{result['scan_id']}/report", timeout=30
        )
        if pdf_resp.status_code == 200:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_resp.content,
                file_name=f"meditrack_report_{result['scan_id'][:8]}.pdf",
                mime="application/pdf",
            )

        # New scan button
        if st.button("📸 Analyze Another Image"):
            st.session_state.analysis_complete = False
            st.session_state.scan_result = None
            st.rerun()

with tab2:
    st.markdown("### 📊 Patient Wound History")
    st.caption("View healing progress over time")
    
    history_patient_id = st.text_input(
        "Patient ID",
        value=st.session_state.patient_id,
        key="history_patient",
        placeholder="Enter patient ID to view history"
    )
    
    if st.button("📥 Load History", key="load_btn"):
        with st.spinner("Loading history..."):
            try:
                response = requests.get(
                    f"{API_URL}/api/wounds/patient/{history_patient_id}/history",
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['scan_count'] == 0:
                        st.info(f"No scans found for patient: {history_patient_id}")
                    else:
                        st.success(f"✓ Found {data['scan_count']} scans")

                        # Healing trajectory prediction
                        traj = data.get("healing_trajectory", {})
                        if traj.get("available"):
                            trend = traj.get("trend", "")
                            msg = traj.get("message", "")
                            if trend == "good":
                                st.success(f"🟢 {msg}")
                            elif trend == "improving":
                                st.info(f"📈 {msg}  (improving +{traj.get('slope_per_scan', 0):.1f} pts/scan)")
                            else:
                                st.warning(f"📉 {msg}")
                        
                        scans = data['scans']
                        df_data = []
                        for scan in scans:
                            df_data.append({
                                "Date": scan['scan_date'][:10],
                                "Type": scan['metrics'].get('wound_type', 'unknown').replace('_', ' ').title(),
                                "Healing Score": scan['metrics']['healing_score'],
                                "Area (cm²)": scan['metrics']['wound_area_cm2'],
                                "Risk": scan['analysis']['risk_level'].upper()
                            })
                        
                        df = pd.DataFrame(df_data)
                        
                        # Chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['Healing Score'],
                            mode='lines+markers',
                            name='Healing Score',
                            line=dict(color='#007bff', width=3),
                            marker=dict(size=10, color='#007bff'),
                            fill='tozeroy',
                            fillcolor='rgba(0, 123, 255, 0.1)'
                        ))
                        
                        fig.update_layout(
                            title="Healing Progress",
                            xaxis_title="Scan Number",
                            yaxis_title="Score (0-100)",
                            hovermode='x unified',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400,
                            font=dict(family="Arial, sans-serif", color='#4A5568')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Table
                        st.markdown("#### Scan Records")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.error(f"Failed to load: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.markdown("### About MediTrack")
    
    st.markdown("""
    MediTrack is an AI-powered wound healing monitoring system with advanced classification.
    
    #### Features
    - 🔬 **Multi-Factor Classification**: Identifies wound types automatically
    - 🤖 **AI Analysis**: LLM-powered patient-friendly insights
    - 📈 **Progress Tracking**: Monitor healing over time
    - 📏 **Smart Measurements**: Length×width for incisions, area for burns
    
    #### Wound Types Detected
    - **Surgical Incision**: Linear wounds with sutures
    - **Laceration**: Irregular linear tears
    - **Burn**: Area wounds with irregular boundaries
    - **Pressure Ulcer**: Large irregular wounds
    - **Diabetic Ulcer**: Round medium-sized wounds
    - **Abrasion**: Shallow irregular wounds
    - **Puncture**: Small circular wounds
    
    #### Metrics
    - **Healing Score**: Composite metric (0-100)
    - **Wound Area/Length**: Size measurement
    - **Redness Index**: Inflammation level (0-1)
    - **Classification Features**: Circularity, aspect ratio, edge quality
    
    #### Important Disclaimer
    ⚠️ **For Educational Use Only**
    
    This is a demonstration project and should NOT be used for medical diagnosis, treatment decisions, or clinical care.
    
    Always consult healthcare professionals for medical advice.
    
    ---
    
    **Version**: 2.0.0 with Multi-Factor Classification  
    **Developer**: Meenakshi Sridharan Sundaram  
    **Portfolio**: [GitHub](https://github.com/Msundara19)
    """)

# Footer
st.markdown("""
<div class="footer">
    MediTrack v2.0 | Multi-Factor Wound Classification | Educational Project
</div>
""", unsafe_allow_html=True)