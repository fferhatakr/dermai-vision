import streamlit as st

def show_standard_result(result: dict, analysis_mode_label: str):
    
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
            <strong>Low confidence:</strong> Model certainty below threshold. 
            Consider running Detailed Analysis.
        </div>
        """, unsafe_allow_html=True)


    all_probs = result.get("all_probabilities", {})
    if all_probs:
        st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
        st.markdown('<p class="ds-section-label">Differential Diagnosis</p>', 
                    unsafe_allow_html=True)

        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top_class = sorted_probs[0][0] if sorted_probs else ""

        bars_html = ""
        for cls, score in sorted_probs:
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