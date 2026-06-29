# Student Dropout Prediction

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-dropout-prediction-mianiela22.streamlit.app)

Predicting which university students are at risk of dropping out using machine learning, so institutions can intervene early and improve retention.

## The Problem

Student dropout is one of the most pressing challenges in higher education. Roughly 30% of first-year students at US institutions don't return for their second year, costing billions in lost tuition and unrealized potential. If universities could identify at-risk students early — based on enrollment data, academic performance, and socioeconomic factors — they could target advising and support resources where they matter most.

## Dataset

The [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset contains **4,424 student records** with **36 features** spanning:

- **Demographics**: age, gender, marital status, nationality
- **Academic background**: previous qualifications, admission grade
- **Socioeconomic**: parental education and occupation, scholarship status, tuition payment
- **Academic performance**: curricular units enrolled, approved, and grades for semesters 1–2
- **Macroeconomic context**: unemployment rate, inflation, GDP at time of enrollment

**Target**: three-class (Dropout / Enrolled / Graduate), simplified to binary (Dropout vs. Graduate) for modeling.

## Approach

1. **Exploratory Data Analysis** — understanding feature distributions, class imbalance, and which variables correlate with dropout
2. **Preprocessing** — binary target (dropping "Enrolled" students whose outcome is unknown), feature scaling, and handling the ~39/61 class split
3. **Modeling** — comparing Logistic Regression, Random Forest, and XGBoost with proper cross-validation
4. **Evaluation** — precision, recall, F1, AUC-ROC (not just accuracy), plus confusion matrices
5. **Interpretability** — SHAP values to explain which factors drive dropout risk
6. **Temporal analysis** — testing what the model can predict at enrollment, after semester 1, and after semester 2, to determine the realistic intervention window
7. **Business framing** — translating model output into actionable recommendations for an advising team

## Results

### Model Comparison (full feature set)

All three models achieve strong performance (AUC > 0.96), with **Logistic Regression** slightly outperforming the ensemble methods — evidence that the relationship between features and dropout is largely linear, so added model complexity doesn't help.

| Model | Accuracy | F1 (Dropout) | AUC-ROC | Recall | Precision |
|-------|----------|-------------|---------|--------|-----------|
| **Logistic Regression** | 0.926 | 0.908 | **0.973** | **0.937** | 0.881 |
| Random Forest | 0.923 | 0.902 | 0.972 | 0.905 | 0.899 |
| XGBoost | 0.916 | 0.894 | 0.967 | 0.901 | 0.886 |

### When Can We Actually Intervene?

The full model's 0.97 AUC is partly driven by second-semester features that describe dropout *as it's happening* (a student with 0 approved courses in semester 2 is already most of the way out). The more important question is: **how early can we flag at-risk students?**

| Intervention Point | Features | AUC-ROC | Recall | Precision |
|---|---|---|---|---|
| At enrollment | 21 (demographics, background) | 0.824 | 0.736 | 0.661 |
| **After semester 1** | **27 (+ semester 1 grades)** | **0.948** | **0.891** | **0.843** |
| After semester 2 | 36 (all features) | 0.973 | 0.937 | 0.881 |

**The semester-1 model is the practically useful one** — it achieves AUC 0.948 using only data available between semesters, when there's still time to intervene. It catches **89% of at-risk students** before semester 2 even begins.

### Top Predictors of Dropout (via SHAP)

| Feature | Direction |
|---------|-----------|
| Curricular units approved (sem 1 & 2) | Fewer approved → higher dropout risk |
| Tuition fees up to date | Not current → higher dropout risk |
| Curricular units enrolled | Fewer enrolled → higher dropout risk |
| Scholarship holder | No scholarship → higher dropout risk |
| Semester grades | Lower grades → higher dropout risk |

The strongest signals are **first-semester academic performance** and **financial standing** — both actionable intervention points for universities.

<p align="center">
  <img src="figures/08_temporal_comparison.png" width="90%" alt="Temporal model comparison: ROC curves and metrics by intervention timing" />
</p>

## Interactive Demo

A Streamlit app lets you input a student's profile and get a real-time dropout risk prediction with SHAP explanations. Users can toggle between the **early-warning model** (semester 1 data only — the actionable one) and the **full model** (all semesters — higher accuracy but partly retrospective), demonstrating the accuracy vs. actionability tradeoff.

```bash
streamlit run app.py
```

## Project Structure

```
student-dropout-prediction/
├── app.py              # Streamlit demo app
├── data/               # Dataset (CSV)
├── notebooks/          # Jupyter notebooks (EDA, modeling)
├── src/                # Reusable Python modules
├── figures/            # Saved visualizations
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run notebooks
jupyter notebook

# Run the interactive demo
streamlit run app.py
```

## Author

**Mirla Irias** — Data Science & Mathematics, University of Miami | M.S. Data Science, Rice University (incoming)

- GitHub: [@mianiela22](https://github.com/mianiela22)
