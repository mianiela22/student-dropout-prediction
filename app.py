import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    df = pd.read_csv("data/data.csv", sep=";")
    df = df[df["Target"] != "Enrolled"].copy()
    df["Target"] = (df["Target"] == "Dropout").astype(int)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    sem2_cols = [c for c in X.columns if "2nd sem" in c]
    sem1_cols = [c for c in X.columns if "1st sem" in c]
    macro_cols = ["Unemployment rate", "Inflation rate", "GDP"]
    enrollment_cols = [c for c in X.columns if c not in sem1_cols + sem2_cols + macro_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {}
    scenarios = {
        "Early Warning (Semester 1)": enrollment_cols + sem1_cols,
        "Full Model (All Semesters)": list(X.columns),
    }

    for name, cols in scenarios.items():
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[cols])
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(X_tr, y_train)
        explainer = shap.LinearExplainer(model, X_tr)
        models[name] = {
            "model": model, "scaler": scaler, "explainer": explainer, "features": cols,
        }

    return models, X.columns.tolist(), df

models, all_features, df = load_models()

st.title("🎓 Student Dropout Risk Predictor")

model_choice = st.radio(
    "Select model",
    ["Early Warning (Semester 1)", "Full Model (All Semesters)"],
    horizontal=True,
    help="The early-warning model uses only data available after semester 1 — when there's still time to intervene. "
         "The full model adds semester 2 data for higher accuracy, but by then dropout is already underway."
)

is_early = model_choice == "Early Warning (Semester 1)"

if is_early:
    st.info(
        "**Early-warning model (AUC 0.948)** — Uses only enrollment data and semester 1 performance. "
        "This is the actionable model: it flags at-risk students *between semesters*, when advising teams can still intervene."
    )
else:
    st.warning(
        "**Full model (AUC 0.973)** — Uses all data including semester 2 performance. "
        "Higher accuracy, but semester 2 features partly describe dropout as it's happening, "
        "so this is better suited for retrospective analysis than early intervention."
    )

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Semester 1 Performance")
    sem1_approved = st.slider("1st Semester — Courses Approved", 0, 26, 5)
    sem1_grade = st.slider("1st Semester — Average Grade", 0.0, 18.0, 12.0, 0.5)
    sem1_enrolled = st.slider("1st Semester — Courses Enrolled", 0, 26, 6)

    if not is_early:
        st.subheader("Semester 2 Performance")
        sem2_approved = st.slider("2nd Semester — Courses Approved", 0, 20, 5)
        sem2_grade = st.slider("2nd Semester — Average Grade", 0.0, 18.0, 12.0, 0.5)
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

if not is_early:
    input_data["Curricular units 2nd sem (approved)"] = sem2_approved
    input_data["Curricular units 2nd sem (grade)"] = sem2_grade
    input_data["Curricular units 2nd sem (enrolled)"] = sem2_enrolled

selected = models[model_choice]
feature_cols = selected["features"]
input_df = pd.DataFrame([input_data])[feature_cols]
input_scaled = selected["scaler"].transform(input_df)

dropout_prob = selected["model"].predict_proba(input_scaled)[0][1]

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
    shap_values = selected["explainer"].shap_values(input_scaled)[0]
    feature_impact = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP Value": shap_values,
        "Abs SHAP": np.abs(shap_values)
    }).sort_values("Abs SHAP", ascending=False).head(8)

    st.markdown("**Top factors influencing this prediction:**")
    for _, row in feature_impact.iterrows():
        direction = "↑ risk" if row["SHAP Value"] > 0 else "↓ risk"
        icon = "🔴" if row["SHAP Value"] > 0 else "🟢"
        st.markdown(f"{icon} **{row['Feature']}** — {direction} (impact: {row['SHAP Value']:+.2f})")

st.divider()

auc_label = "0.948" if is_early else "0.973"
st.caption(
    f"Built by Mirla Irias · Data: UCI Predict Students' Dropout and Academic Success · "
    f"Model: Logistic Regression ({model_choice}, AUC-ROC: {auc_label})"
)
