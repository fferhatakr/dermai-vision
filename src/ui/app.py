import streamlit as st
import requests
from PIL import Image
import io


st.set_page_config(page_title="DermaScan AI", page_icon="🛡️", layout="centered")


st.title("🛡️ DermaScan AI: Skin Analysis")
st.markdown("""
This application analyzes skin lesions using artificial intelligence.
**Please note:** This is not a diagnostic tool, but only a decision-support system.
For definitive results, always consult a medical specialist.
""")

st.divider()


uploaded_file = st.file_uploader("Upload a photo of your skin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Photo", use_container_width=True)
    
    if st.button("🔍 Start Analysis"):
        with st.spinner("AI is analyzing the image, please wait..."):
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG")
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}

                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    st.subheader("📊 Analysis Result")
                    
                    if prediction == "Risky":
                        st.error("⚠️ **Detected Condition: RISKY**")
                        st.warning(f"Confidence Score: %{confidence*100:.2f}")
                    else:
                        st.success("✅ **Detected Condition: NORMAL**")
                        st.info(f"Confidence Score: %{confidence*100:.2f}")
                        
                    st.info(f"💡 {result['message']}")
                else:
                    st.error("An error occurred while communicating with the API.")
                    
            except Exception as e:
                st.error(f"Connection error: {e}. Please make sure the FastAPI server is running.")

st.divider()
st.caption("Ondokuz Mayis University - Computer Engineering | Ferhat - AI Project 2026")