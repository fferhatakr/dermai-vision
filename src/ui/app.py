import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="DermaScan AI", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    img { border-radius: 8px; border: 1px solid #30363d; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); }
    </style>
    """, unsafe_allow_html=True)

if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None


if st.session_state.token is None:
    st.title("DermaScan AI")
    st.markdown("---")

    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Sign In", type="primary"):
            if email and password:
                try:
                    r = requests.post(f"{API_BASE}/users/login",
                                      json={"email": email, "password": password})
                    if r.status_code == 200:
                        st.session_state.token = r.json()["access_token"]
                        st.session_state.user_email = email
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                st.warning("Please fill in all fields.")

    with tab_register:
        full_name = st.text_input("Full Name", key="reg_name")
        reg_email = st.text_input("Email", key="reg_email")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")

        if st.button("Create Account", type="primary"):
            if full_name and reg_email and reg_pass:
                try:
                    r = requests.post(f"{API_BASE}/users/register",
                                      json={"email": reg_email, "password": reg_pass,
                                            "full_name": full_name})
                    if r.status_code == 200:
                        st.success("Account created. Please sign in.")
                    else:
                        st.error(r.json().get("detail", "Registration failed."))
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                st.warning("Please fill in all fields.")

    st.stop()
col_title, col_user = st.columns([4, 1])
with col_title:
    st.title("DermaScan AI: Meta-Learning Fusion System")
with col_user:
    st.markdown(f"<br>👤 {st.session_state.user_email}", unsafe_allow_html=True)
    if st.button("Sign Out"):
        st.session_state.token = None
        st.session_state.user_email = None
        st.rerun()

st.warning("**Disclaimer:** This application is solely an engineering portfolio project. It is not a medical diagnostic device. Please consult a dermatologist regarding any genuine health concerns.")
st.info("**Zero Data Retention:** User privacy is our top priority. Uploaded images are processed strictly in-memory (RAM) during the inference pipeline and are instantly destroyed once the analysis is complete. No data is stored on any server.")
st.markdown("Combined analysis of Deep Learning (CNN) and Clinical Metadata (XGBoost)")
st.markdown("---")

col_input, col_meta = st.columns([1, 1])

image = None
image_ready = False

with col_input:
    st.subheader("1. Image Upload")

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
    st.subheader("2. Patient & Clinical Data")
    age = st.number_input("Patient Age:", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Gender:", ["male", "female", "unknown"])

    site_options = [
        'anterior torso', 'upper extremity', 'posterior torso',
        'lower extremity', 'flat', 'head/neck', 'palms/soles', 'unknown'
    ]
    anatom_site = st.selectbox("Anatomical Site:", site_options)

    st.info("The Meta-Learner combines these clinical factors with visual patterns for higher accuracy.")
    analyze_btn = st.button("EXECUTE HYBRID ANALYSIS", type="primary")

if analyze_btn and image_ready:
    with st.spinner("Deep Learning & XGBoost Fusion in progress"):
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")

            files = {"file": ("scan.jpg", img_byte_arr.getvalue(), "image/jpeg")}
            data = {
                "age": str(age),
                "sex": sex,
                "anatom_site": anatom_site,
                "needs_heatmap": "true",
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
                st.divider()

                st.subheader("Debug Mode: CNN - Meta-Learner")
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
                    st.error(f"**CLINICAL CONFLICT:** Metadata changed diagnosis from **{result['debug']['cnn_diagnosis']}** to **{result['diagnosis']}**.")
                else:
                    st.success("**CONSENSUS:** Both systems agreed on the diagnosis.")

                st.divider()
                pred = result["prediction"]
                diag = result["diagnosis"]

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
                        st.image(overlay, use_container_width=True,
                                 caption="Heatmap: Blue (Safe) -> Red (Suspicious)")
                    else:
                        st.warning("Heatmap could not be generated for this scan.")

            else:
                st.error(f"Backend Error: {response.text}")

        except Exception as e:
            st.error(f"Connection Failed: {e}")

elif analyze_btn and not image_ready:
    st.warning("Please select or upload an image first.")