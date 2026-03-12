import pandas as pd
import joblib

model = joblib.load("recovery_model.joblib")
label_encoder = joblib.load("recovery_label_encoder.joblib")

sample_data = pd.DataFrame([{
    "Age": 35,
    "Gender": "Male",
    "Employment_Type": "Salaried",
    "Monthly_Income": 150000,
    "Num_Dependents": 2,
    "Loan_Amount": 500000,
    "Loan_Tenure": 24,
    "Interest_Rate": 18.5,
    "Loan_Type": "Personal",
    "Collateral_Value": 300000,
    "Outstanding_Loan_Amount": 250000,
    "Monthly_EMI": 22000,
    "Payment_History": "Average",
    "Num_Missed_Payments": 3,
    "Days_Past_Due": 45,
    "Collection_Attempts": 2,
    "Collection_Method": "Phone Call",
    "Legal_Action_Taken": "No"
}])

prediction = model.predict(sample_data)
prediction_label = label_encoder.inverse_transform(prediction)

prediction_proba = model.predict_proba(sample_data)
confidence = prediction_proba.max() * 100

print("Predicted Recovery Status:", prediction_label[0])
print("Prediction Confidence: {:.2f}%".format(confidence))