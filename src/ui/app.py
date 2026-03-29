import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Base reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #080c12 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #c8d0dc !important;
}

.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Header bar ── */
.ds-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #1a2235;
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
}
.ds-logo {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    color: #3b82f6;
    text-transform: uppercase;
}
.ds-title {
    font-size: 1.05rem;
    font-weight: 500;
    color: #e2e8f0;
    margin: 0;
}
.ds-user-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.05em;
}

/* ── Section labels ── */
.ds-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.75rem;
}

/* ── Cards ── */
.ds-card {
    background: #0d1420;
    border: 1px solid #1a2235;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.ds-card-accent {
    border-left: 3px solid #3b82f6;
}

/* ── Mode selector ── */
.ds-mode-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.ds-mode-card {
    background: #0d1420;
    border: 1px solid #1a2235;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
}
.ds-mode-card:hover { border-color: #3b82f6; }
.ds-mode-card.active {
    border-color: #3b82f6;
    background: #0f1e35;
}
.ds-mode-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
}
.ds-mode-desc {
    font-size: 0.72rem;
    color: #64748b;
    line-height: 1.5;
}
.ds-mode-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin-top: 0.5rem;
}
.tag-fast { background: #0f3d2e; color: #34d399; }
.tag-balanced { background: #1a2f4a; color: #60a5fa; }
.tag-full { background: #2d1f4a; color: #a78bfa; }

/* ── Result blocks ── */
.ds-result-risk {
    background: #1a0f0f;
    border: 1px solid #7f1d1d;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ds-result-safe {
    background: #0a1f14;
    border: 1px solid #14532d;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ds-result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.ds-result-label.risk { color: #f87171; }
.ds-result-label.safe { color: #4ade80; }
.ds-result-diagnosis {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: -0.01em;
}
.ds-result-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Probability bars ── */
.ds-prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
}
.ds-prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #94a3b8;
    width: 48px;
    flex-shrink: 0;
}
.ds-prob-bar-bg {
    flex: 1;
    height: 5px;
    background: #1a2235;
    border-radius: 3px;
    overflow: hidden;
}
.ds-prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: #3b82f6;
    transition: width 0.4s ease;
}
.ds-prob-bar-fill.top { background: #f87171; }
.ds-prob-value {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #64748b;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Conflict / consensus banners ── */
.ds-conflict-banner {
    background: #1c1208;
    border: 1px solid #78350f;
    border-radius: 7px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #fbbf24;
    margin-bottom: 1rem;
}
.ds-consensus-banner {
    background: #0a1a10;
    border: 1px solid #166534;
    border-radius: 7px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #86efac;
    margin-bottom: 1rem;
}

/* ── Warning / info strips ── */
.ds-warning-strip {
    background: #1a1206;
    border-left: 3px solid #d97706;
    padding: 0.65rem 1rem;
    font-size: 0.78rem;
    color: #fbbf24;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.75rem;
}
.ds-info-strip {
    background: #0c1829;
    border-left: 3px solid #1d4ed8;
    padding: 0.65rem 1rem;
    font-size: 0.78rem;
    color: #93c5fd;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.75rem;
}

/* ── Debug panel ── */
.ds-debug-panel {
    background: #080c12;
    border: 1px solid #1a2235;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
}
.ds-debug-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
}
.ds-debug-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #0f1825;
    font-size: 0.75rem;
}
.ds-debug-row:last-child { border-bottom: none; }
.ds-debug-key { color: #64748b; }
.ds-debug-val {
    font-family: 'DM Mono', monospace;
    color: #94a3b8;
}

/* ── Image containers ── */
.ds-img-wrap {
    border: 1px solid #1a2235;
    border-radius: 8px;
    overflow: hidden;
}
.ds-img-caption {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    padding: 0.5rem 0.75rem;
    border-top: 1px solid #1a2235;
}

/* ── Divider ── */
.ds-divider {
    border: none;
    border-top: 1px solid #1a2235;
    margin: 1.75rem 0;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.5rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1e40af !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0d1420 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 7px !important;
    color: #c8d0dc !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > label,
.stNumberInput > label,
.stSelectbox > label,
.stFileUploader > label,
.stRadio > label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0d1420 !important;
    border: 1px dashed #1a2235 !important;
    border-radius: 8px !important;
}

.stProgress > div > div > div > div {
    background: #3b82f6 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a2235 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

div[data-testid="stAlert"] {
    background: #0d1420 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1.1rem 3rem !important; }
    .ds-mode-grid { grid-template-columns: 1fr; }
    .ds-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
    .ds-result-diagnosis { font-size: 1.15rem; }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .stButton > button { width: 100% !important; }
}
@media (max-width: 480px) {
    .ds-title { font-size: 0.95rem; }
}
</style>
""", unsafe_allow_html=True)


if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None


if st.session_state.token is None:
    st.markdown("""
    <div style="max-width:420px;margin:4rem auto 0;">
        <p class="ds-logo" style="margin-bottom:0.25rem;">DermaScan AI</p>
        <h1 style="font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:2rem;line-height:1.3;">
            Clinical Decision<br>Support System
        </h1>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In", type="primary", use_container_width=True):
                if email and password:
                    try:
                        r = requests.post(f"{API_BASE}/users/login",
                                          json={"email": email, "password": password})
                        if r.status_code == 200:
                            st.session_state.token = r.json()["access_token"]
                            st.session_state.user_email = email
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                else:
                    st.warning("Please fill in all fields.")

        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            full_name_r = st.text_input("Full Name", key="reg_name")
            reg_email = st.text_input("Email", key="reg_email")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True):
                if full_name_r and reg_email and reg_pass:
                    try:
                        r = requests.post(f"{API_BASE}/users/register",
                                          json={"email": reg_email, "password": reg_pass,
                                                "full_name": full_name_r})
                        if r.status_code == 200:
                            st.success("Account created. Please sign in.")
                        else:
                            st.error(r.json().get("detail", "Registration failed."))
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                else:
                    st.warning("Please fill in all fields.")
    st.stop()




st.markdown(f"""
<div class="ds-header">
    <div>
        <p class="ds-logo">DermaScan AI</p>
        <p class="ds-title">Meta-Learning Fusion System</p>
    </div>
    <span class="ds-user-badge">{st.session_state.user_email}</span>
</div>
""", unsafe_allow_html=True)


col_so, _ = st.columns([1, 5])
with col_so:
    if st.button("Sign Out"):
        st.session_state.token = None
        st.session_state.user_email = None
        st.rerun()


st.markdown("""
<div class="ds-warning-strip">
    <strong>Research prototype only.</strong> Not a certified medical device. Always consult a licensed dermatologist for clinical decisions.
</div>
<div class="ds-info-strip">
    Uploaded images are processed in-memory and never stored. Patient records are encrypted in PostgreSQL.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)


st.markdown('<p class="ds-section-label">Analysis Mode</p>', unsafe_allow_html=True)
st.markdown("""
<div class="ds-mode-grid">
    <div class="ds-mode-card">
        <div class="ds-mode-title">Quick Scan</div>
        <div class="ds-mode-desc">Image only. ONNX Runtime inference. Binary risk output. No metadata required.</div>
        <span class="ds-mode-tag tag-fast">~50ms</span>
    </div>
    <div class="ds-mode-card">
        <div class="ds-mode-title">Standard Analysis</div>
        <div class="ds-mode-desc">ONNX + XGBoost fusion. Clinical metadata included. 8-class probability output.</div>
        <span class="ds-mode-tag tag-balanced">~150ms</span>
    </div>
    <div class="ds-mode-card">
        <div class="ds-mode-title">Detailed Analysis</div>
        <div class="ds-mode-desc">Full PyTorch pipeline + XGBoost + Grad-CAM heatmap. Maximum diagnostic depth.</div>
        <span class="ds-mode-tag tag-full">~500ms</span>
    </div>
</div>
""", unsafe_allow_html=True)

analysis_mode_label = st.radio(
    "Select mode",
    ["Quick Scan", "Standard Analysis", "Detailed Analysis"],
    horizontal=True,
    label_visibility="collapsed"
)

MODE_MAP = {
    "Quick Scan": "quick",
    "Standard Analysis": "standard",
    "Detailed Analysis": "detailed"
}
selected_mode = MODE_MAP[analysis_mode_label]

st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)


col_img, col_meta = st.columns([1, 1], gap="large")

image = None
image_ready = False

with col_img:
    st.markdown('<p class="ds-section-label">Dermoscopic Image</p>', unsafe_allow_html=True)

    sample_folder = "test_samples"
    samples = os.listdir(sample_folder) if os.path.exists(sample_folder) else []
    selected_sample = st.selectbox("Test sample", ["None"] + samples)

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_ready = True
        st.image(image, use_container_width=True)
        st.markdown('<p class="ds-img-caption">Uploaded scan</p>', unsafe_allow_html=True)
    elif selected_sample != "None":
        image = Image.open(os.path.join(sample_folder, selected_sample)).convert("RGB")
        image_ready = True
        st.image(image, use_container_width=True)
        st.markdown(f'<p class="ds-img-caption">Sample: {selected_sample}</p>', unsafe_allow_html=True)

with col_meta:
    st.markdown('<p class="ds-section-label">Patient & Clinical Data</p>', unsafe_allow_html=True)

    full_name = st.text_input("Patient full name", placeholder="e.g. John Smith")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    
    if selected_mode in ["standard", "detailed"]:
        sex = st.selectbox("Sex", ["male", "female", "unknown"])
        site_options = [
            'anterior torso', 'upper extremity', 'posterior torso',
            'lower extremity', 'flat', 'head/neck', 'palms/soles', 'unknown'
        ]
        anatom_site = st.selectbox("Anatomical site", site_options)
        needs_heatmap = False
        if selected_mode == "detailed":
            needs_heatmap = st.checkbox("Generate Grad-CAM heatmap", value=True)
        st.markdown("""
        <div class="ds-info-strip" style="margin-top:1rem;">
            Clinical metadata is fused with visual features via XGBoost meta-learner.
        </div>
        """, unsafe_allow_html=True)
    else:
        sex = "unknown"
        anatom_site = "unknown"
        needs_heatmap = False
        st.markdown("""
        <div class="ds-info-strip" style="margin-top:1rem;">
            Quick Scan uses image features only. No metadata required.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)


if analyze_btn and image_ready and full_name:
    mode_label = analysis_mode_label.upper()
    with st.spinner(f"Running {mode_label}..."):
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")

            files = {"file": ("scan.jpg", img_byte_arr.getvalue(), "image/jpeg")}
            data = {
                "age": str(age),
                "sex": sex,
                "anatom_site": anatom_site,
                "needs_heatmap": str(needs_heatmap).lower(),
                "full_name": full_name,
                "analysis_mode": selected_mode
            }

            response = requests.post(
                f"{API_BASE}/analyze",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )

            if response.status_code == 401:
                st.error("Session expired. Please sign in again.")
                st.session_state.token = None
                st.rerun()

            elif response.status_code == 200:
                result = response.json()
                st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
                st.markdown('<p class="ds-section-label">Analysis Results</p>', unsafe_allow_html=True)

                pred = result["prediction"]
                diag = result["diagnosis"]
                conf = result["confidence"]

                if pred == "Risky":
                    st.markdown(f"""
                    <div class="ds-result-risk">
                        <p class="ds-result-label risk">ELEVATED RISK DETECTED</p>
                        <p class="ds-result-diagnosis">{diag}</p>
                        <p class="ds-result-conf">Confidence: {conf*100:.1f}% &nbsp;|&nbsp; Mode: {analysis_mode_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ds-result-safe">
                        <p class="ds-result-label safe">LOW RISK</p>
                        <p class="ds-result-diagnosis">{diag}</p>
                        <p class="ds-result-conf">Confidence: {conf*100:.1f}% &nbsp;|&nbsp; Mode: {analysis_mode_label}</p>
                    </div>
                    """, unsafe_allow_html=True)

                
                debug = result.get("debug", {})
                if debug.get("low_confidence_warning"):
                    st.markdown("""
                    <div class="ds-warning-strip">
                        <strong>Low confidence:</strong> Model certainty below threshold. Consider running Detailed Analysis.
                    </div>
                    """, unsafe_allow_html=True)

               
                all_probs = result.get("all_probabilities", {})
                if all_probs:
                    st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
                    st.markdown('<p class="ds-section-label">Differential Diagnosis</p>', unsafe_allow_html=True)

                    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                    top_class = sorted_probs[0][0] if sorted_probs else ""

                    bars_html = ""
                    for i, (cls, score) in enumerate(sorted_probs):
                        pct = score * 100
                        fill_class = "top" if cls == top_class else ""
                        bars_html += f"""
                        <div class="ds-prob-row">
                            <span class="ds-prob-label">{cls}</span>
                            <div class="ds-prob-bar-bg">
                                <div class="ds-prob-bar-fill {fill_class}" style="width:{pct:.1f}%"></div>
                            </div>
                            <span class="ds-prob-value">{pct:.1f}%</span>
                        </div>
                        """
                    st.markdown(bars_html, unsafe_allow_html=True)

                
                if selected_mode == "detailed":
                    st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)

                    if debug.get("conflict"):
                        st.markdown(f"""
                        <div class="ds-conflict-banner">
                            <strong>Clinical conflict:</strong> Visual model predicted <strong>{debug.get('cnn_diagnosis','—')}</strong>,
                            meta-learner overrode to <strong>{diag}</strong>.
                            Metadata features shifted the final decision.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="ds-consensus-banner">
                            <strong>Consensus:</strong> CNN and meta-learner agreed on the final diagnosis.
                        </div>
                        """, unsafe_allow_html=True)

                  
                    st.markdown(f"""
                    <div class="ds-debug-panel">
                        <p class="ds-debug-title">Model internals</p>
                        <div class="ds-debug-row">
                            <span class="ds-debug-key">CNN prediction</span>
                            <span class="ds-debug-val">{debug.get('cnn_diagnosis','—')}</span>
                        </div>
                        <div class="ds-debug-row">
                            <span class="ds-debug-key">CNN confidence</span>
                            <span class="ds-debug-val">{debug.get('cnn_confidence',0)*100:.1f}%</span>
                        </div>
                        <div class="ds-debug-row">
                            <span class="ds-debug-key">Meta-learner prediction</span>
                            <span class="ds-debug-val">{diag}</span>
                        </div>
                        <div class="ds-debug-row">
                            <span class="ds-debug-key">Conflict</span>
                            <span class="ds-debug-val">{"YES" if debug.get("conflict") else "NO"}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                  
                    hm_b64 = result.get("heatmap_base64", "")
                    if hm_b64:
                        st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
                        st.markdown('<p class="ds-section-label">Visual Explainability — Grad-CAM</p>', unsafe_allow_html=True)

                        c1, c2 = st.columns(2, gap="medium")
                        with c1:
                            st.image(image, use_container_width=True)
                            st.markdown('<p class="ds-img-caption">Original scan</p>', unsafe_allow_html=True)
                        with c2:
                            decoded = base64.b64decode(hm_b64)
                            np_arr = np.frombuffer(decoded, np.uint8)
                            hm_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            hm_img = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
                            orig_cv = np.array(image)
                            hm_img = cv2.resize(hm_img, (orig_cv.shape[1], orig_cv.shape[0]))
                            overlay = cv2.addWeighted(orig_cv, 0.6, hm_img, 0.4, 0)
                            st.image(overlay, use_container_width=True)
                            st.markdown('<p class="ds-img-caption">Attention map — blue: low / red: high suspicion</p>', unsafe_allow_html=True)

               
                if selected_mode == "quick":
                    msg = debug.get("message", "")
                    if msg:
                        st.markdown(f"""
                        <div class="ds-debug-panel" style="margin-top:1rem;">
                            <p class="ds-debug-title">Quick scan summary</p>
                            <div class="ds-debug-row">
                                <span class="ds-debug-key">Assessment</span>
                                <span class="ds-debug-val">{msg}</span>
                            </div>
                            <div class="ds-debug-row">
                                <span class="ds-debug-key">Note</span>
                                <span class="ds-debug-val">Run Detailed Analysis for clinical depth</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            else:
                st.error(f"Backend error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Connection failed: {e}")

elif analyze_btn and not image_ready:
    st.warning("Please upload or select an image first.")
elif analyze_btn and not full_name:
    st.warning("Please enter the patient name.")