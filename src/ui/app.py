import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2
import numpy as np

st.set_page_config(page_title="DermaScan AI", page_icon="shield", layout="wide")


st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 30px; }
    img { border-radius: 8px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("DermaScan AI: Professional Skin Analysis")
st.markdown("---")

col_input, col_settings = st.columns([1, 1])

with col_input:
    st.subheader("1. Image Upload")
    uploaded_file = st.file_uploader("Select dermoscopy image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Lesion Image", width=300)

with col_settings:
    st.subheader("2. Clinical Details")
    patient_text = st.text_area(
        "Patient Symptoms / History:", 
        height=150,
        placeholder="E.g., Itching, bleeding, color change within 2 months..."
    )

    st.subheader("3. Analysis Mode")

    detail = st.toggle("Deep Analysis (Enable Heatmap & Grad-CAM)", value=True)
    
    if detail:
        st.info("Mode: V2 Ultimate (TTA Enabled + Grad-CAM)")
    else:
        st.warning("Mode: Standard Analysis (TTA Enabled)")

    st.divider()
    analyze_btn = st.button("START DIAGNOSIS", use_container_width=True, type="primary")

if analyze_btn:
    if uploaded_file is None:
        st.error("Please upload an image first!")
    else:
        with st.spinner("AI is examining the lesion with TTA strategy..."):
            try:
                
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG")
                img_bytes = img_byte_arr.getvalue()

                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                data_payload = {"text": patient_text, "needs_heatmap": detail} 

                
                response = requests.post(
                    "http://127.0.0.1:8000/analyze", 
                    files=files, 
                    data=data_payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        
                        if result["prediction"] == "Risky":
                            st.error(f"RESULT: {result['prediction'].upper()}")
                        else:
                            st.success(f"RESULT: {result['prediction'].upper()}")
                    with m2:
                        
                        st.metric("Hybrid Risk Score", f"%{result['confidence']*100:.2f}")
                    with m3:
                        
                        st.caption("Engine: V2 Ultimate (PyTorch)")
                        st.write(f"Note: {result['message']}")

                    st.write("---")
                    
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.subheader("Analysis Focus")
                        st.image(image, width=350)
                       
                        st.metric("Visual Risk Score", f"%{result['scores']['image']*100:.1f}")

                    with img_col2:
                        heatmap_base64 = result.get("heatmap_base64", "")
                        if heatmap_base64:
                            st.subheader("Attention Map (Grad-CAM)")
                            
                            
                            decoded_bytes = base64.b64decode(heatmap_base64)
                            np_arr = np.frombuffer(decoded_bytes, np.uint8)
                            heatmap_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

                            
                            original_image_cv = np.array(image.convert("RGB"))
                            h, w, _ = original_image_cv.shape
                            heatmap_resized = cv2.resize(heatmap_img, (w, h))
                            
                            superimposed_img = cv2.addWeighted(original_image_cv, 0.6, heatmap_resized, 0.4, 0)
                            
                            st.image(superimposed_img, width=350)
                            st.metric("NLP Symptom Risk", f"%{result['scores']['text']*100:.1f}")
                        else:
                            st.info("Heatmap only available in Deep Mode")

                else:
                    st.error(f"Server Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}")