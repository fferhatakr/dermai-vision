import streamlit as st

def show_quick_result(debug: dict,selected_mode: str):
    if selected_mode == "quick":
        msg = debug.get("message", "")
        if msg:
            st.markdown(f"""
            <div class="ds-debug-panel" style="margin-top:1rem;">
                <p class="ds-debug-title">Quick scan summary</p>
                <div class="ds-debug-row">
                    <span class="ds-debug-key">Assessment</span>
                    <span class="ds-debug-val">{msg}</span>
                </div>
                <div class="ds-debug-row">
                    <span class="ds-debug-key">Note</span>
                    <span class="ds-debug-val">Run Detailed Analysis for clinical depth</span>
                </div>
            </div>
            """, unsafe_allow_html=True)