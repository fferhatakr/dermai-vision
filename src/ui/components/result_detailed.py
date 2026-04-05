import streamlit as st
import cv2
import numpy as np
import base64


def show_detailed_result(result: dict, image, diag: str, selected_mode: str, debug:dict,):
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

            clinical_report = result.get("clinical_report", "")
            if clinical_report:
                st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
                st.markdown('<p class="ds-section-label">AI Clinical Report</p>', 
                            unsafe_allow_html=True)
                st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
                st.markdown('<p class="ds-section-label">AI Clinical Report</p>', 
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ds-card ds-card-accent" style="padding: 1.5rem;">', 
                    unsafe_allow_html=True
                )
                st.markdown(clinical_report) 
                st.markdown('</div>', unsafe_allow_html=True)

