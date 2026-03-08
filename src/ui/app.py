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
    img { border-radius: 8px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("DermaScan AI: Ultimate Fusion System")
st.markdown("---")


col_input, col_text = st.columns(2)

with col_input:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Select Stain Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

with col_text:
    st.subheader("Clinical Symptoms")
    patient_text = st.text_area(
        "Patient Complaint:", 
        height=150,
        placeholder="For example: It has been growing for two months, there is bleeding, it causes itching."
    )
    
    st.info("The system analyses by combining the image and the text you have written.")
    analyze_btn = st.button("START ANALYSIS", type="primary", use_container_width=True)


if analyze_btn and uploaded_file:
    with st.spinner("Image and text analysis is being performed."):
        try:
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            
            files = {"file": img_byte_arr.getvalue()}
            
            data = {"text": patient_text, "needs_heatmap": "true"}
            
            response = requests.post("http://127.0.0.1:8000/analyze", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                st.divider()
                
               
                pred = result["prediction"]
                diag = result["diagnosis"]
                hybrid_score = result["hybrid_score"]
                
                if pred == "Risky":
                    st.error(f"Result: {pred.upper()}")
                    st.markdown(f"Diagnosis: **{diag}**")
                else:
                    st.success(f"Result: {pred.upper()}")
                    st.markdown(f"Diagnosis: **{diag}**")

                
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Image Risk", f"%{result['scores']['image_raw']*100:.1f}")
                with k2:
                    st.metric("Symptom Risk", f"%{result['scores']['text']*100:.1f}")
                with k3:
                    st.metric("Hybrid Score", f"%{hybrid_score*100:.2f}")

               
                severity = result['scores']['severity']
                st.write("Cancer Risk Level:")
                st.progress(int(severity))
                
                st.divider()
                if "all_probabilities" in result:
                    st.subheader("Artificial Intelligence Differential Diagnosis (Top 3)")
                    
                    
                    probs = result["all_probabilities"]
                    top_3 = list(probs.items())[:3]
                    
                    for disease, score in top_3:
                        color = "red" if disease.startswith(("MEL", "BCC", "SCC")) else "green"
                        st.write(f"**{disease}**: %{score*100:.1f}")
                        st.progress(float(score))
                    st.caption("*The model's distribution of uncertainty among other diseases.*")
                    st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Original")
                    st.image(image, width=350)
                with c2:
                    st.subheader("The point of focus in the decision-making process")
                    hm_b64 = result.get("heatmap_base64", "")
                    if hm_b64:
                        decoded = base64.b64decode(hm_b64)
                        np_arr = np.frombuffer(decoded, np.uint8)
                        hm_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        hm_img = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
                        
                        orig_cv = np.array(image)
                        hm_img = cv2.resize(hm_img, (orig_cv.shape[1], orig_cv.shape[0]))
                        overlay = cv2.addWeighted(orig_cv, 0.6, hm_img, 0.4, 0)
                        
                        st.image(overlay, width=350, caption="Red Areas = Risky Areas")
                    else:
                        st.warning("A heat map could not be generated.")

            else:
                st.error("Server Error.")

        except Exception as e:
            st.error(f"Conneciton Error: {e}")