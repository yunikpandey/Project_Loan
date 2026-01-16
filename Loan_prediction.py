import streamlit as st
import pickle
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open('loan_approval_svc_model.pickle', 'rb') as f:
        pkg = pickle.load(f)
    return pkg['model'], pkg['scaler'], pkg['features']

model, scaler, features = load_model()

st.title("Loan Approve/Reject")
st.write("Enter applicant details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 850, 700)
    loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 25.0, 10.0, 0.1)
    income = st.number_input("Annual Income ($)", 0, 10000000, 60000)

with col2:
    loan_percent = st.number_input("Loan / Income Ratio", 0.0, 1.0, 0.25, 0.01)
    defaulted_before = st.selectbox("Previous Default?", ["No", "Yes"])

# Prepare input
input_dict = {
    'credit_score': credit_score,
    'loan_percent_income': loan_percent,
    'loan_int_rate': loan_int_rate,
    'previous_loan_defaults_on_file': 1 if defaulted_before == "Yes" else 0,
    'person_income': income,
    # Add dummy columns for home_ownership if you encoded it
}

input_df = pd.DataFrame([input_dict])[features]  # match training columns order

if st.button("Predict", type="primary"):
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    
    if prediction == 1:
        st.success("### LOAN APPROVED ✓")
    else:
        st.error("### LOAN REJECTED ✗")