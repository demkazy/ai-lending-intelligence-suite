import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Loan Approval AI Dashboard",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #1f2937;
}
.sub-text {
    font-size: 16px;
    color: #6b7280;
}
.card {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💳 Loan Approval AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict loan approval using a Random Forest model</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("loan_approval_app/loan_model.joblib")

model = load_model()

feature_names = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]

st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=100)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150, step=10)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0, value=360, step=12)
credit_history = st.sidebar.selectbox("Credit History", [1, 0], format_func=lambda x: "Good" if x == 1 else "Bad")
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

col1, col2, col3 = st.columns(3)

total_income = applicant_income + coapplicant_income
loan_income_ratio = (loan_amount / total_income * 100) if total_income > 0 else 0

col1.metric("Total Income", f"{total_income:,}")
col2.metric("Loan Amount", f"{loan_amount:,}")
col3.metric("Loan / Income Ratio", f"{loan_income_ratio:.2f}%")

if st.button("Predict Loan Status", use_container_width=True):
    input_data = pd.DataFrame([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        int(dependents.replace("3+", "3")),
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    ]], columns=feature_names)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Loan Approved")
    else:
        st.error(f"❌ Loan Not Approved")

    st.write(f"**Approval Probability:** {probability:.2%}")
    st.progress(float(probability))

    st.subheader("Applicant Summary")
    summary_df = pd.DataFrame({
        "Feature": [
            "Gender", "Married", "Dependents", "Education", "Self Employed",
            "Applicant Income", "Coapplicant Income", "Loan Amount",
            "Loan Amount Term", "Credit History", "Property Area"
        ],
        "Value": [
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_term, "Good" if credit_history == 1 else "Bad", property_area
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("Model Feature Importance")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    top_features = importance_df.sort_values(by="Importance", ascending=False).head(5)
    st.write("### Top 5 Important Features")
    st.dataframe(top_features, use_container_width=True, hide_index=True)
else:
    st.info("Feature importance is not available for this model.")