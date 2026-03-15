import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os

st.set_page_config(page_title="DermaScan AI", page_icon="", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    img { border-radius: 8px; border: 1px solid #30363d; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); }
    </style>
    """, unsafe_allow_html=True)

st.title("DermaScan AI: Meta-Learning Fusion System")
st.warning("**Disclaimer:** This application is solely an engineering portfolio project. It is not a medical diagnostic device. Please consult a dermatologist regarding any genuine health concerns.")
st.info("**Zero Data Retention:** User privacy is our top priority. Uploaded images are processed strictly in-memory (RAM) during the inference pipeline and are instantly destroyed once the analysis is complete. No data is stored on any server.")
st.markdown("Combined analysis of Deep Learning (CNN) and Clinical Metadata (XGBoost)")
st.markdown("---")

col_input, col_meta = st.columns([1, 1])

image = None
image_ready = False

with col_input:
    st.subheader("1.Image Upload")
    
    sample_folder = "test_samples"
    samples = os.listdir(sample_folder) if os.path.exists(sample_folder) else []
    selected_sample = st.selectbox("Select a test sample", ["None"] + samples)
    
    uploaded_file = st.file_uploader("Or Select Dermatoscopic Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_ready = True
        st.image(image, caption="Current Scan", use_container_width=True)
    elif selected_sample != "None":
        image = Image.open(os.path.join(sample_folder, selected_sample)).convert("RGB")
        image_ready = True
        st.image(image, caption=f"Sample: {selected_sample}", use_container_width=True)

with col_meta:
    st.subheader("2.Clinical Metadata")
    
    age = st.number_input("Patient Age:", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Gender:", ["male", "female", "unknown"])
    
    site_options = [
        'anterior torso', 'upper extremity', 'posterior torso', 
        'lower extremity', 'flat', 'head/neck', 'palms/soles', 'unknown'
    ]
    anatom_site = st.selectbox("Anatomical Site:", site_options)
    
    st.info("The Meta-Learner combines these clinical factors with visual patterns for a higher accuracy.")
    analyze_btn = st.button("EXECUTE HYBRID ANALYSIS", type="primary", use_container_width=True)

if analyze_btn and image_ready:
    with st.spinner("Deep Learning & XGBoost Fusion in progress"):
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            
            files = {"file": img_byte_arr.getvalue()}
            
            data = {
                "age": str(age),
                "sex": sex,
                "anatom_site": anatom_site,
                "needs_heatmap": "true"
            }
            
            response = requests.post("https://technull1-dermascan-ai.hf.space/analyze", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                st.divider()
                
                st.subheader("Debug Mode: CNN - Meta-Learner ")
                db_col1, db_col2 = st.columns(2)
                
                with db_col1:
                    st.info("VISION ONLY")
                    st.markdown(f"**Diagnosis:** `{result['debug']['cnn_diagnosis']}`")
                    st.markdown(f"**Confidence:** %{result['debug']['cnn_confidence']*100:.1f}")
                    st.progress(float(result['debug']['cnn_confidence']))

                with db_col2:
                    st.warning("META-LEARNER")
                    st.markdown(f"**Diagnosis:** `{result['diagnosis']}`")
                    st.markdown(f"**Confidence:** %{result['confidence']*100:.1f}")
                    st.progress(float(result['confidence']))

                if result['debug']['conflict']:
                    st.error(f"**CLINICAL CONFLICT:** Metadata (Age/Sex/Site) changed the visual diagnosis from **{result['debug']['cnn_diagnosis']}** to **{result['diagnosis']}**.")
                else:
                    st.success("**CONSENSUS:** Both systems agreed on the diagnosis.")
                
                st.divider()
                pred = result["prediction"]
                diag = result["diagnosis"]
                confidence = result["confidence"]
                
                if pred == "Risky":
                    st.error(f"SYSTEM ALERT: {pred.upper()}")
                    st.subheader(f"Suspected Diagnosis: {diag}")
                else:
                    st.success(f"ANALYSIS: {pred.upper()}")
                    st.subheader(f"Likely Diagnosis: {diag}")
                
                st.divider()

                if "all_probabilities" in result:
                    st.subheader("Differential Diagnosis (Probabilities)")
                    probs = result["all_probabilities"]
                    for disease, score in list(probs.items())[:3]:
                        st.write(f"**{disease}**")
                        st.progress(float(score))
                
                st.divider()

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Original Scan")
                    st.image(image, use_container_width=True)
                with c2:
                    st.subheader("AI Focus Map (Grad-CAM)")
                    hm_b64 = result.get("heatmap_base64", "")
                    if hm_b64:
                        decoded = base64.b64decode(hm_b64)
                        np_arr = np.frombuffer(decoded, np.uint8)
                        hm_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        hm_img = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
                        
                        orig_cv = np.array(image)
                        hm_img = cv2.resize(hm_img, (orig_cv.shape[1], orig_cv.shape[0]))
                        overlay = cv2.addWeighted(orig_cv, 0.6, hm_img, 0.4, 0)
                        st.image(overlay, use_container_width=True, caption="Heatmap: Blue (Safe) -> Red (Suspicious)")
                    else:
                        st.warning("Heatmap could not be generated for this scan.")

            else:
                st.error(f"Backend Error: {response.text}")

        except Exception as e:
            st.error(f"Connection Failed: {e}")