import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    df = pd.read_csv("data/data.csv", sep=";")
    df = df[df["Target"] != "Enrolled"].copy()
    df["Target"] = (df["Target"] == "Dropout").astype(int)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    explainer = shap.LinearExplainer(model, X_train_scaled)
    return model, scaler, explainer, X.columns.tolist(), df

model, scaler, explainer, feature_names, df = load_model()

st.title("🎓 Student Dropout Risk Predictor")
st.markdown(
    "Enter a student's profile to predict their dropout risk. "
    "The model uses academic performance, demographics, and financial data "
    "to identify at-risk students early enough for intervention."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Academic Performance")
    sem1_approved = st.slider("1st Semester — Courses Approved", 0, 26, 5)
    sem1_grade = st.slider("1st Semester — Average Grade", 0.0, 18.0, 12.0, 0.5)
    sem2_approved = st.slider("2nd Semester — Courses Approved", 0, 20, 5)
    sem2_grade = st.slider("2nd Semester — Average Grade", 0.0, 18.0, 12.0, 0.5)
    sem1_enrolled = st.slider("1st Semester — Courses Enrolled", 0, 26, 6)
    sem2_enrolled = st.slider("2nd Semester — Courses Enrolled", 0, 23, 6)

with col2:
    st.subheader("Student Background")
    age = st.slider("Age at Enrollment", 17, 70, 19)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Other"])
    displaced = st.selectbox("Displaced (from home region)?", ["No", "Yes"])
    international = st.selectbox("International Student?", ["No", "Yes"])
    admission_grade = st.slider("Admission Grade", 90.0, 190.0, 130.0, 1.0)

with col3:
    st.subheader("Financial & Other")
    tuition_up_to_date = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])
    scholarship = st.selectbox("Scholarship Holder?", ["No", "Yes"])
    debtor = st.selectbox("Has Tuition Debt?", ["No", "Yes"])
    prev_qual_grade = st.slider("Previous Qualification Grade", 90.0, 190.0, 130.0, 1.0)
    mothers_qual = st.slider("Mother's Qualification (education level)", 1, 44, 19)
    fathers_qual = st.slider("Father's Qualification (education level)", 1, 44, 12)

median_vals = df.drop(columns=["Target"]).median()
input_data = median_vals.copy()

input_data["Age at enrollment"] = age
input_data["Gender"] = 1 if gender == "Male" else 0
input_data["Marital status"] = {"Single": 1, "Married": 2, "Other": 6}[marital_status]
input_data["Displaced"] = 1 if displaced == "Yes" else 0
input_data["International"] = 1 if international == "Yes" else 0
input_data["Admission grade"] = admission_grade
input_data["Tuition fees up to date"] = 1 if tuition_up_to_date == "Yes" else 0
input_data["Scholarship holder"] = 1 if scholarship == "Yes" else 0
input_data["Debtor"] = 1 if debtor == "Yes" else 0
input_data["Previous qualification (grade)"] = prev_qual_grade
input_data["Mother's qualification"] = mothers_qual
input_data["Father's qualification"] = fathers_qual
input_data["Curricular units 1st sem (approved)"] = sem1_approved
input_data["Curricular units 1st sem (grade)"] = sem1_grade
input_data["Curricular units 1st sem (enrolled)"] = sem1_enrolled
input_data["Curricular units 2nd sem (approved)"] = sem2_approved
input_data["Curricular units 2nd sem (grade)"] = sem2_grade
input_data["Curricular units 2nd sem (enrolled)"] = sem2_enrolled

input_df = pd.DataFrame([input_data])[feature_names]
input_scaled = scaler.transform(input_df)

dropout_prob = model.predict_proba(input_scaled)[0][1]

st.divider()
st.subheader("Prediction")

result_col1, result_col2 = st.columns([1, 2])

with result_col1:
    if dropout_prob >= 0.7:
        st.error(f"### ⚠️ High Risk: {dropout_prob:.0%}")
        st.markdown("This student profile shows **high dropout risk**. Immediate advising intervention recommended.")
    elif dropout_prob >= 0.4:
        st.warning(f"### ⚡ Moderate Risk: {dropout_prob:.0%}")
        st.markdown("This student profile shows **moderate dropout risk**. Proactive check-in recommended.")
    else:
        st.success(f"### ✅ Low Risk: {dropout_prob:.0%}")
        st.markdown("This student profile shows **low dropout risk**. Standard monitoring sufficient.")

with result_col2:
    shap_values = explainer.shap_values(input_scaled)[0]
    feature_impact = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values,
        "Abs SHAP": np.abs(shap_values)
    }).sort_values("Abs SHAP", ascending=False).head(8)

    st.markdown("**Top factors influencing this prediction:**")
    for _, row in feature_impact.iterrows():
        direction = "↑ risk" if row["SHAP Value"] > 0 else "↓ risk"
        icon = "🔴" if row["SHAP Value"] > 0 else "🟢"
        st.markdown(f"{icon} **{row['Feature']}** — {direction} (impact: {row['SHAP Value']:+.2f})")

st.divider()
st.caption(
    "Built by Mirla Irias · Data: UCI Predict Students' Dropout and Academic Success · "
    "Model: Logistic Regression (AUC-ROC: 0.973)"
)
