import streamlit as st

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.markdown(
            """
            <h1 style='text-align: center;'>FinRecover AI</h1>
            <p style='text-align: center; color: gray;'>Secure Loan Recovery Intelligence Platform</p>
            """,
            unsafe_allow_html=True
        )

        with st.container():
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        st.stop()

def logout():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()