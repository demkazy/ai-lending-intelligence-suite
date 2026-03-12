import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from auth import login, logout
from utils import get_risk_level, get_recommended_action

st.set_page_config(page_title="FinRecover AI", page_icon="💳", layout="wide")

login()

model = joblib.load("Smart_loan_recovery/recovery_model.joblib")
label_encoder = joblib.load("recovery_label_encoder.joblib")
df = pd.read_csv("Smart Loan Recovery System.csv")

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.metric-box {
    background: linear-gradient(135deg, #111827, #1f2937);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #374151;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
}
.metric-title {
    color: #d1d5db;
    font-size: 14px;
}
.metric-value {
    color: white;
    font-size: 28px;
    font-weight: 700;
}
.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: white;
}
.sub-title {
    color: #9ca3af;
    margin-bottom: 20px;
}
.section-card {
    background: #111827;
    border: 1px solid #374151;
    padding: 18px;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("FinRecover AI")
st.sidebar.write(f"Welcome, **{st.session_state.username}**")
logout()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Single Customer Prediction",
        "Batch Prediction",
        "Portfolio Analytics"
    ]
)

st.markdown('<div class="main-title">FinRecover AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Professional Smart Loan Recovery Intelligence Platform</div>', unsafe_allow_html=True)

if page == "Executive Dashboard":
    total_borrowers = len(df)
    avg_loan = df["Loan_Amount"].mean()
    avg_outstanding = df["Outstanding_Loan_Amount"].mean()
    avg_dpd = df["Days_Past_Due"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Total Borrowers</div><div class='metric-value'>{total_borrowers}</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Average Loan Amount</div><div class='metric-value'>{avg_loan:,.0f}</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Average Outstanding</div><div class='metric-value'>{avg_outstanding:,.0f}</div></div>",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Average Days Past Due</div><div class='metric-value'>{avg_dpd:.1f}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### Recovery Overview")

    col1, col2 = st.columns(2)

    with col1:
        status_counts = df["Recovery_Status"].value_counts().reset_index()
        status_counts.columns = ["Recovery_Status", "Count"]
        fig = px.pie(status_counts, names="Recovery_Status", values="Count", title="Recovery Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        emp_counts = df["Employment_Type"].value_counts().reset_index()
        emp_counts.columns = ["Employment_Type", "Count"]
        fig = px.bar(emp_counts, x="Employment_Type", y="Count", title="Employment Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(df, x="Days_Past_Due", nbins=25, title="Days Past Due Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            df,
            x="Loan_Amount",
            y="Outstanding_Loan_Amount",
            color="Recovery_Status",
            title="Loan Amount vs Outstanding Balance"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Single Customer Prediction":
    st.markdown("### Customer Recovery Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        employment_type = st.selectbox("Employment Type", sorted(df["Employment_Type"].dropna().unique()))
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=150000.0)
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=500000.0)

    with col2:
        loan_tenure = st.number_input("Loan Tenure", min_value=1, max_value=600, value=24)
        interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=18.5)
        loan_type = st.selectbox("Loan Type", sorted(df["Loan_Type"].dropna().unique()))
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=300000.0)
        outstanding_loan_amount = st.number_input("Outstanding Loan Amount", min_value=0.0, value=250000.0)
        monthly_emi = st.number_input("Monthly EMI", min_value=0.0, value=22000.0)

    with col3:
        payment_history = st.selectbox("Payment History", sorted(df["Payment_History"].dropna().unique()))
        num_missed_payments = st.number_input("Missed Payments", min_value=0, max_value=100, value=3)
        days_past_due = st.number_input("Days Past Due", min_value=0, max_value=1000, value=45)
        collection_attempts = st.number_input("Collection Attempts", min_value=0, max_value=100, value=2)
        collection_method = st.selectbox("Collection Method", sorted(df["Collection_Method"].dropna().unique()))
        legal_action_taken = st.selectbox("Legal Action Taken", sorted(df["Legal_Action_Taken"].dropna().unique()))

    if st.button("Predict Recovery Outcome", use_container_width=True):
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Employment_Type": employment_type,
            "Monthly_Income": monthly_income,
            "Num_Dependents": num_dependents,
            "Loan_Amount": loan_amount,
            "Loan_Tenure": loan_tenure,
            "Interest_Rate": interest_rate,
            "Loan_Type": loan_type,
            "Collateral_Value": collateral_value,
            "Outstanding_Loan_Amount": outstanding_loan_amount,
            "Monthly_EMI": monthly_emi,
            "Payment_History": payment_history,
            "Num_Missed_Payments": num_missed_payments,
            "Days_Past_Due": days_past_due,
            "Collection_Attempts": collection_attempts,
            "Collection_Method": collection_method,
            "Legal_Action_Taken": legal_action_taken
        }])

        prediction = model.predict(input_data)
        prediction_label = label_encoder.inverse_transform(prediction)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = float(proba.max() * 100)

        risk_level = get_risk_level(confidence, days_past_due, num_missed_payments)
        action = get_recommended_action(risk_level, days_past_due, legal_action_taken)

        a, b = st.columns([1, 2])

        with a:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Prediction Confidence"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)

        with b:
            st.success(f"Predicted Recovery Status: {prediction_label}")
            st.info(f"Prediction Confidence: {confidence:.2f}%")
            st.warning(f"Risk Level: {risk_level}")
            st.write(f"**Recommended Action:** {action}")

            summary_df = pd.DataFrame({
                "Field": ["Days Past Due", "Missed Payments", "Collection Attempts", "Legal Action Taken"],
                "Value": [days_past_due, num_missed_payments, collection_attempts, legal_action_taken]
            })
            st.dataframe(summary_df, use_container_width=True)

elif page == "Batch Prediction":
    st.markdown("### Batch Recovery Prediction")
    uploaded_file = st.file_uploader("Upload CSV for batch scoring", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        required_columns = [
            "Age", "Gender", "Employment_Type", "Monthly_Income", "Num_Dependents",
            "Loan_Amount", "Loan_Tenure", "Interest_Rate", "Loan_Type",
            "Collateral_Value", "Outstanding_Loan_Amount", "Monthly_EMI",
            "Payment_History", "Num_Missed_Payments", "Days_Past_Due",
            "Collection_Attempts", "Collection_Method", "Legal_Action_Taken"
        ]

        missing_cols = [col for col in required_columns if col not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            preds = model.predict(batch_df[required_columns])
            pred_labels = label_encoder.inverse_transform(preds)
            probs = model.predict_proba(batch_df[required_columns]).max(axis=1) * 100

            batch_df["Predicted_Recovery_Status"] = pred_labels
            batch_df["Prediction_Confidence"] = probs

            risk_levels = []
            actions = []

            for _, row in batch_df.iterrows():
                risk = get_risk_level(
                    row["Prediction_Confidence"],
                    row["Days_Past_Due"],
                    row["Num_Missed_Payments"]
                )
                action = get_recommended_action(
                    risk,
                    row["Days_Past_Due"],
                    row["Legal_Action_Taken"]
                )
                risk_levels.append(risk)
                actions.append(action)

            batch_df["Risk_Level"] = risk_levels
            batch_df["Recommended_Action"] = actions

            st.dataframe(batch_df, use_container_width=True)

            result_counts = batch_df["Predicted_Recovery_Status"].value_counts().reset_index()
            result_counts.columns = ["Predicted_Recovery_Status", "Count"]
            fig = px.bar(result_counts, x="Predicted_Recovery_Status", y="Count", title="Batch Prediction Summary")
            st.plotly_chart(fig, use_container_width=True)

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Scored Portfolio",
                data=csv,
                file_name="scored_loan_recovery_portfolio.csv",
                mime="text/csv"
            )

elif page == "Portfolio Analytics":
    st.markdown("### Portfolio Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df,
            x="Recovery_Status",
            y="Outstanding_Loan_Amount",
            title="Outstanding Amount by Recovery Status"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df.groupby("Loan_Type", as_index=False)["Loan_Amount"].mean(),
            x="Loan_Type",
            y="Loan_Amount",
            title="Average Loan Amount by Loan Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            df,
            x="Days_Past_Due",
            y="Num_Missed_Payments",
            color="Recovery_Status",
            title="Delinquency Pattern Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df,
            x="Monthly_Income",
            color="Recovery_Status",
            barmode="overlay",
            title="Income Distribution by Recovery Status"
        )
        st.plotly_chart(fig, use_container_width=True)