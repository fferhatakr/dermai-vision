import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os
from styles import inject_styles
from auth import show_auth
from components.analysis_form import show_analysis_form
from components.result_detailed import show_detailed_result
from components.result_standard import show_standard_result
from components.result_quick import show_quick_result



API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_styles()

if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None

show_auth(API_BASE)
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

image, image_ready, full_name, age, sex, anatom_site, needs_heatmap, selected_mode, analyze_btn, analysis_mode_label = show_analysis_form()
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
            if selected_mode == "detailed":
                progress = st.progress(0, text="Preprocessing image...")
                import time
            
                progress.progress(15, text="Running vision model...")
                time.sleep(0.3)
            
                progress.progress(35, text="Running XGBoost meta-learner...")
                time.sleep(0.3)
                
                progress.progress(55, text="Generating Grad-CAM heatmap...")
                time.sleep(0.3)
                
                progress.progress(75, text="Generating AI clinical report...")
            
                response = requests.post(
                    f"{API_BASE}/analyze",
                    files=files,
                    data=data,
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                progress.progress(100, text="Analysis complete.")
                time.sleep(0.3)
                progress.empty() 
            else:
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
                diag = result["diagnosis"]
                debug = result.get("debug", {})
                st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
                st.markdown('<p class="ds-section-label">Analysis Results</p>', unsafe_allow_html=True)
                show_standard_result(result, analysis_mode_label)
                show_detailed_result(result, image, diag, selected_mode, debug)
               
                show_quick_result(debug, selected_mode)
                if selected_mode == "detailed":
                    st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
                    st.markdown('<p class="ds-section-label">AI Clinical Report</p>',
                                unsafe_allow_html=True)

                    with st.spinner("Generating clinical report..."):
                        report_response = requests.post(
                            f"{API_BASE}/report",
                            json={
                                "diagnosis": diag,
                                "confidence": result["confidence"],
                                "all_probabilities": result["all_probabilities"],
                                "is_risky": result["prediction"] == "Risky",
                                "age": age,
                                "sex": sex,
                                "anatom_site": anatom_site
                            },
                            headers={"Authorization": f"Bearer {st.session_state.token}"}
                        )

                    if report_response.status_code == 200:
                        clinical_report = report_response.json().get("clinical_report", "")
                        if clinical_report:
                            st.markdown(
                                f'<div class="ds-card ds-card-accent" style="padding:1.5rem;">',
                                unsafe_allow_html=True
                            )
                            st.markdown(clinical_report) 
                            st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.error(f"Backend error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Connection failed: {e}")

elif analyze_btn and not image_ready:
    st.warning("Please upload or select an image first.")
elif analyze_btn and not full_name:
    st.warning("Please enter the patient name.")