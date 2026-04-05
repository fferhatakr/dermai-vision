import streamlit as st
import os
from PIL import Image

def show_analysis_form():
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

        sample_folder = "assets/demo_images"
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
        return image, image_ready, full_name , age , sex ,anatom_site, needs_heatmap, selected_mode , analyze_btn , analysis_mode_label





















































