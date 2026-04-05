import streamlit as st
import requests

def show_auth(API_BASE: str):
    if st.session_state.token is None:
        st.markdown("""
        <div style="max-width:420px;margin:4rem auto 0;">
            <p class="ds-logo" style="margin-bottom:0.25rem;">DermaScan AI</p>
            <h1 style="font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:2rem;line-height:1.3;">
                Clinical Decision<br>Support System
            </h1>
        </div>
        """, unsafe_allow_html=True)

        _, center, _ = st.columns([1, 2, 1])
        with center:
            tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

            with tab_login:
                st.markdown("<br>", unsafe_allow_html=True)
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_pass")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Sign In", type="primary", use_container_width=True):
                    if email and password:
                        try:
                            r = requests.post(f"{API_BASE}/users/login",
                                            json={"email": email, "password": password})
                            if r.status_code == 200:
                                st.session_state.token = r.json()["access_token"]
                                st.session_state.user_email = email
                                st.rerun()
                            else:
                                st.error("Invalid credentials.")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
                    else:
                        st.warning("Please fill in all fields.")

            with tab_register:
                st.markdown("<br>", unsafe_allow_html=True)
                full_name_r = st.text_input("Full Name", key="reg_name")
                reg_email = st.text_input("Email", key="reg_email")
                reg_pass = st.text_input("Password", type="password", key="reg_pass")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Create Account", type="primary", use_container_width=True):
                    if full_name_r and reg_email and reg_pass:
                        try:
                            r = requests.post(f"{API_BASE}/users/register",
                                            json={"email": reg_email, "password": reg_pass,
                                                    "full_name": full_name_r})
                            if r.status_code == 200:
                                st.success("Account created. Please sign in.")
                            else:
                                st.error(r.json().get("detail", "Registration failed."))
                        except Exception as e:
                            st.error(f"Connection error: {e}")
                    else:
                        st.warning("Please fill in all fields.")
        st.stop()