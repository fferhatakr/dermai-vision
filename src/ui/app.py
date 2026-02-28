import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2
import numpy as np



st.set_page_config(page_title="DermaScan AI", page_icon="🛡️", layout="centered")


st.title(" DermaScan AI: Skin Analysis")
st.markdown("""
This application analyzes skin lesions using artificial intelligence.
**Please note:** This is not a diagnostic tool, but only a decision-support system.
For definitive results, always consult a medical specialist.
""")

st.divider()


uploaded_file = st.file_uploader("Upload a photo of your skin...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Photo", width=250)
    
    st.markdown("### Patient Symptoms")
    patient_text = st.text_area(
        "Please describe your symptoms (e.g., itching, bleeding, rapid growth, how long it has been there):", 
        height=100
    )
    
    if st.button(" Start Analysis"):
        with st.spinner("AI is analyzing the image, please wait..."):
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG")
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                data_payload = {"text": patient_text} 
                response = requests.post(
                    "http://127.0.0.1:8000/analyze", 
                    files=files, 
                    data=data_payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    heatmap_base64  = result["heatmap_base64"]

                    decaoding_byte = base64.b64decode(heatmap_base64)
                    np_byte = np.frombuffer(decaoding_byte,np.uint8)
                    heatmap_img = cv2.imdecode(np_byte,cv2.IMREAD_COLOR)
                    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)


                    original_image = Image.open(uploaded_file).convert("RGB")
                    original_image = original_image.resize((224,224))
                    original_np = np.array(original_image)
                    
                    superimposed_img = cv2.addWeighted(original_np,0.6,heatmap_img,0.4,0)

                    
                    st.divider() 
                    col1, col2 = st.columns(2)

                    with col1:
                        st.info("Original Image")
                        st.image(original_image,use_container_width=True)
                    
                    with col2:
                        st.warning("AI Attention Map (Grad-CAM)")
                        st.image(superimposed_img, caption="Where did I focus?", use_container_width=True)

                    st.subheader(" Analysis Result")
                    
                    if prediction == "Risky":
                        st.error(" **Detected Condition: RISKY**")
                        st.warning(f"Confidence Score: %{confidence*100:.2f}")
                    else:
                        st.success(" **Detected Condition: NORMAL**")
                        st.info(f"Confidence Score: %{confidence*100:.2f}")
                        
                    st.info(f" {result['message']}")

                    



                else:
                    st.error("An error occurred while communicating with the API.")
                    
            except Exception as e:
                st.error(f"Connection error: {e}. Please make sure the FastAPI server is running.")

st.divider()
st.caption("Ondokuz Mayis University - Computer Engineering | Ferhat - AI Project 2026")