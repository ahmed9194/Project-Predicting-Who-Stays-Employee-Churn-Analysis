import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load(r"C:\\Users\\user\\Downloads\\best_xgb_model_for_deployment.pkl")

# Define columns
feature_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                   'average_montly_hours', 'time_spend_company', 'Work_accident',
                   'promotion_last_5years', 'salary',
                   'department_IT', 'department_RandD', 'department_accounting',
                   'department_hr', 'department_management', 'department_marketing',
                   'department_product_mng', 'department_sales', 'department_support',
                   'department_technical', 'hours_level']

departments = ['IT', 'RandD', 'accounting', 'hr', 'management', 'marketing',
               'product_mng', 'sales', 'support', 'technical']

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("Employee Attrition Prediction")
st.write("Fill in the details to predict if the employee will leave.")

# Input form
with st.form("prediction_form"):
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
    number_project = st.number_input("Number of Projects", 1, 10, 3)
    average_montly_hours = st.number_input("Average Monthly Hours", 50, 400, 150)
    time_spend_company = st.number_input("Years at Company", 1, 20, 3)
    Work_accident = st.selectbox("Work Accident", [0, 1])
    promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
    salary = st.selectbox("Salary Level (0=Low,1=Medium,2=High)", [0,1,2])
    department = st.selectbox("Department", departments)
    hours_level = st.slider("Hours Level (Scaled)", 0.0, 1.0, 0.5)
    submitted = st.form_submit_button("Predict")

if submitted:
    # One-hot for department
    department_encoded = [1 if department == d else 0 for d in departments]
    
    # Combine features
    input_data = np.array([satisfaction_level, last_evaluation, number_project,
                           average_montly_hours, time_spend_company, Work_accident,
                           promotion_last_5years, salary] + department_encoded + [hours_level]).reshape(1, -1)
    
    # Prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ This employee is **likely to leave** the company.")
    else:
        st.success(f"✅ This employee is **likely to stay** at the company.")
    if proba is not None:
        st.write(f"**Probability of leaving:** {proba:.2f}")

    # SHAP Explainability
    st.subheader("Feature Impact (SHAP Explainability)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # --- Waterfall Plot ---
    st.write("### Waterfall Plot (Effect of each feature)")
    fig1 = plt.figure(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                         data=input_data[0], feature_names=feature_columns), max_display=10
    )
    st.pyplot(fig1)

    # --- Bar Plot ---
    st.write("### Bar Plot (Feature Importance)")
    fig2, ax = plt.subplots(figsize=(8, 4))
    shap.bar_plot(shap_values[0], feature_names=feature_columns, max_display=10, show=False)
    st.pyplot(fig2)

    # --- Force Plot ---
    st.write("### Force Plot (Interactive)")
    st.write("This plot shows how each feature pushes the prediction.")
    st.components.v1.html(shap.force_plot(
        explainer.expected_value, shap_values[0], input_data,
        feature_names=feature_columns, matplotlib=False
    ).html(), height=300)

