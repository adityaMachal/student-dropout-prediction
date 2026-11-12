import json
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to sys.path so we can import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

st.set_page_config(
    page_title="Student Dropout Risk Early Warning Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Dropout Risk Early Warning Dashboard")
st.markdown("Distributed Machine Learning & Decision Support powered by **Apache Spark 4.2.0 (PySpark MLlib)**.")

# Official Course ID to Human-Readable Degree Mapping
COURSE_MAP = {
    "1": "Biofuel Production Tech",
    "2": "Animation & Multimedia Design",
    "3": "Social Service (Evening)",
    "4": "Agronomy",
    "5": "Communication Design",
    "6": "Veterinary Nursing",
    "7": "Informatics Engineering",
    "8": "Equiniculture",
    "9": "Management",
    "10": "Social Service",
    "11": "Tourism",
    "12": "Nursing",
    "13": "Oral Hygiene",
    "14": "Advertising & Marketing",
    "15": "Journalism & Communication",
    "16": "Basic Education",
    "17": "Management (Evening)"
}


@st.cache_data
def load_predictions_and_metrics():
    """Loads holdout test predictions and evaluation metrics."""
    pred_path = os.path.join(parent_dir, "output", "test_dropout_predictions.csv")
    metric_path = os.path.join(parent_dir, "output", "metrics.json")

    if not os.path.exists(pred_path):
        return None, None

    df = pd.read_csv(pred_path)
    # Ensure Course is string and add Program Name column
    df["Course_Code"] = df["Course"].astype(str)
    df["Program_Name"] = df["Course_Code"].map(COURSE_MAP).fillna(df["Course_Code"])
    df["Study_Mode_Label"] = df["Daytime/evening attendance"].map({1: "Daytime", 0: "Evening", "1": "Daytime", "0": "Evening"})
    df["Scholarship_Label"] = df["Scholarship holder"].map({1: "Yes", 0: "No", "1": "Yes", "0": "No"})
    df["Tuition_Label"] = df["Tuition fees up to date"].map({1: "Up to Date", 0: "Arrears", "1": "Up to Date", "0": "Arrears"})

    metrics = None
    if os.path.exists(metric_path):
        with open(metric_path, "r") as f:
            metrics = json.load(f)

    return df, metrics


df, metrics = load_predictions_and_metrics()

if df is None:
    st.warning("⚠️ No prediction data found in `output/test_dropout_predictions.csv`.")
    st.info("Please run the PySpark pipeline first:\n```bash\npython3 -m src.pipeline_runner\n```")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls & Cohort Filters
# ---------------------------------------------------------
st.sidebar.header("🔍 Cohort Filters")

all_programs = sorted(df["Program_Name"].unique())
selected_programs = st.sidebar.multiselect(
    "Academic Program (Major)",
    options=all_programs,
    default=all_programs
)

selected_study_modes = st.sidebar.multiselect(
    "Study Mode",
    options=["Daytime", "Evening"],
    default=["Daytime", "Evening"]
)

selected_scholarship = st.sidebar.multiselect(
    "Scholarship Status",
    options=["Yes", "No"],
    default=["Yes", "No"]
)

selected_tuition = st.sidebar.multiselect(
    "Tuition Fee Status",
    options=["Up to Date", "Arrears"],
    default=["Up to Date", "Arrears"]
)

# Apply cohort filters
filtered_df = df[
    (df["Program_Name"].isin(selected_programs)) &
    (df["Study_Mode_Label"].isin(selected_study_modes)) &
    (df["Scholarship_Label"].isin(selected_scholarship)) &
    (df["Tuition_Label"].isin(selected_tuition))
]

if filtered_df.empty:
    st.error("No students match the selected filter combination. Please expand your selections.")
    st.stop()

# ---------------------------------------------------------
# Top Header KPI Cards
# ---------------------------------------------------------
rf_m = metrics.get("random_forest", {}) if metrics else {}
high_risk_count = (filtered_df["risk_level"] == "High Risk").sum()
high_risk_pct = (high_risk_count / len(filtered_df)) * 100

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Evaluated Students", f"{len(filtered_df):,}")
kpi2.metric("Best Model", "Random Forest")
kpi3.metric("Model ROC-AUC", f"{rf_m.get('roc_auc', 0.9082):.4f}")
kpi4.metric("Dropout Sensitivity", f"{rf_m.get('dropout_recall', 0.6775) * 100:.1f}%")
kpi5.metric("🚨 High Risk Cohort", f"{high_risk_count:,} ({high_risk_pct:.1f}%)")

st.markdown("---")

# ---------------------------------------------------------
# Tabs: Covering  1, 2, 3 + Live Serving & Roster
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 1. Risk Categories",
    "⚖️ 2. Actual vs. Predicted Outcomes",
    "🏫 3. Dropout Rates by Academic Program",
    "🔮 4. Live Student Risk Calculator",
    "📋 5. Advisor Early Warning Roster"
])

# ---------------------------------------------------------
# Tab 1:  Requirement 1 — Risk Categories
# ---------------------------------------------------------
with tab1:
    st.subheader(" 1: Student Risk Categories & Distribution")
    st.caption("Early-Warning Risk Tiers: High Risk (P ≥ 0.60), Moderate Risk (0.35 ≤ P < 0.60), Low Risk (P < 0.35)")

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.markdown("#### Overall Cohort Risk Tier Breakdown")
        risk_counts = filtered_df["risk_level"].value_counts().reindex(["High Risk", "Moderate Risk", "Low Risk"]).fillna(0).reset_index()
        risk_counts.columns = ["Risk Tier", "Students"]

        fig_pie = px.pie(
            risk_counts,
            names="Risk Tier",
            values="Students",
            color="Risk Tier",
            color_discrete_map={
                "High Risk": "#d62728",      # Red
                "Moderate Risk": "#ff7f0e",  # Orange
                "Low Risk": "#2ca02c"        # Green
            },
            hole=0.45
        )
        fig_pie.update_traces(textinfo="label+percent+value", textfont_size=13)
        fig_pie.update_layout(height=420)
        st.plotly_chart(fig_pie, width="stretch")

    with col_t2:
        st.markdown("#### Predicted Dropout Probability Distribution")
        filtered_df["Actual_Status"] = filtered_df["label"].map({1.0: "Dropout (At Risk)", 0.0: "Retained"})
        fig_hist = px.histogram(
            filtered_df,
            x="dropout_probability",
            color="Actual_Status",
            nbins=35,
            labels={"dropout_probability": "Predicted Probability of Dropping Out", "Actual_Status": "Actual Outcome"},
            color_discrete_map={"Dropout (At Risk)": "#d62728", "Retained": "#1f77b4"},
            marginal="box"
        )
        fig_hist.add_vline(x=0.60, line_dash="dash", line_color="red", annotation_text="High Risk Threshold (0.60)")
        fig_hist.add_vline(x=0.35, line_dash="dash", line_color="orange", annotation_text="Moderate Risk (0.35)")
        fig_hist.update_layout(height=420)
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("#### Risk Level Segmented by Demographics & Study Mode")
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        fig_mode = px.histogram(
            filtered_df,
            x="Study_Mode_Label",
            color="risk_level",
            barmode="group",
            title="Risk Tier Distribution by Study Mode (Daytime vs. Evening)",
            color_discrete_map={"High Risk": "#d62728", "Moderate Risk": "#ff7f0e", "Low Risk": "#2ca02c"},
            labels={"Study_Mode_Label": "Study Mode"}
        )
        st.plotly_chart(fig_mode, width="stretch")

    with col_sub2:
        fig_schol = px.histogram(
            filtered_df,
            x="Scholarship_Label",
            color="risk_level",
            barmode="group",
            title="Risk Tier Distribution by Scholarship Status",
            color_discrete_map={"High Risk": "#d62728", "Moderate Risk": "#ff7f0e", "Low Risk": "#2ca02c"},
            labels={"Scholarship_Label": "Has Scholarship"}
        )
        st.plotly_chart(fig_schol, width="stretch")

# ---------------------------------------------------------
# Tab 2:  Requirement 2 — Actual vs. Predicted Outcomes
# ---------------------------------------------------------
with tab2:
    st.subheader(" 2: Actual vs. Predicted Dropout Outcomes")
    st.caption("Fidelity evaluation comparing actual student outcomes against model classifications.")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("#### Confusion Matrix Heatmap")
        tp = int(((filtered_df["label"] == 1.0) & (filtered_df["prediction"] == 1.0)).sum())
        fp = int(((filtered_df["label"] == 0.0) & (filtered_df["prediction"] == 1.0)).sum())
        tn = int(((filtered_df["label"] == 0.0) & (filtered_df["prediction"] == 0.0)).sum())
        fn = int(((filtered_df["label"] == 1.0) & (filtered_df["prediction"] == 0.0)).sum())

        cm_matrix = [[tn, fp], [fn, tp]]
        fig_cm = px.imshow(
            cm_matrix,
            x=["Predicted: Retained", "Predicted: Dropout"],
            y=["Actual: Retained", "Actual: Dropout"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        fig_cm.update_layout(height=420)
        st.plotly_chart(fig_cm, width="stretch")

        st.info(f"""
        **Diagnostic Summary:**
        - **True Positives (Dropouts Caught)**: `{tp:,}` students
        - **False Alarms (False Positives)**: `{fp:,}` students
        - **Missed Dropouts (False Negatives)**: `{fn:,}` students
        - **Correctly Retained (True Negatives)**: `{tn:,}` students
        """)

    with col_m2:
        st.markdown("#### Overall Outcome Counts: Actual vs. Predicted")
        actual_dropout = int((filtered_df["label"] == 1.0).sum())
        actual_retained = int((filtered_df["label"] == 0.0).sum())
        pred_dropout = int((filtered_df["prediction"] == 1.0).sum())
        pred_retained = int((filtered_df["prediction"] == 0.0).sum())

        outcome_comp = pd.DataFrame({
            "Outcome": ["Dropout (At Risk)", "Retained"],
            "Actual Counts": [actual_dropout, actual_retained],
            "Predicted Counts": [pred_dropout, pred_retained]
        })
        outcome_melted = pd.melt(outcome_comp, id_vars=["Outcome"], var_name="Type", value_name="Students")

        fig_bar = px.bar(
            outcome_melted,
            x="Outcome",
            y="Students",
            color="Type",
            barmode="group",
            text_auto=True,
            title="Total Student Counts: Actual vs. Predicted Outcomes",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        fig_bar.update_layout(height=420)
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("#### Model Performance Benchmark Comparison")
    if metrics:
        lr_m = metrics.get("logistic_regression", {})
        rf_m = metrics.get("random_forest", {})

        bench_table = pd.DataFrame({
            "Metric": ["ROC-AUC", "F1-Score", "Precision", "Recall", "Dropout Recall (Sensitivity)"],
            "Logistic Regression": [lr_m.get("roc_auc", 0), lr_m.get("f1_score", 0), lr_m.get("precision", 0), lr_m.get("recall", 0), lr_m.get("dropout_recall", 0)],
            "Random Forest Classifier": [rf_m.get("roc_auc", 0), rf_m.get("f1_score", 0), rf_m.get("precision", 0), rf_m.get("recall", 0), rf_m.get("dropout_recall", 0)]
        })
        model_cols = ["Logistic Regression", "Random Forest Classifier"]
        styled_bench = bench_table.style.format(
            "{:.4f}", subset=model_cols
        ).highlight_max(
            axis=1, subset=model_cols, color="#d4edda"
        )
        st.dataframe(styled_bench, hide_index=True, width="stretch")

# ---------------------------------------------------------
# Tab 3:  Requirement 3 — Dropout Rates by Academic Program
# ---------------------------------------------------------
with tab3:
    st.subheader(" 3: Dropout Rates by Academic Program")
    st.caption("Comparison of actual student attrition rate vs. model predicted attrition rate across degree programs.")

    # Calculate Actual and Predicted Dropout Rates by Academic Program
    prog_stats = filtered_df.groupby("Program_Name").agg(
        Total_Students=("label", "count"),
        Actual_Dropouts=("label", lambda s: (s == 1.0).sum()),
        Predicted_Dropouts=("prediction", lambda s: (s == 1.0).sum()),
        High_Risk_Students=("risk_level", lambda s: (s == "High Risk").sum())
    ).reset_index()

    prog_stats["Actual_Dropout_Rate_Pct"] = (prog_stats["Actual_Dropouts"] / prog_stats["Total_Students"]) * 100
    prog_stats["Predicted_Dropout_Rate_Pct"] = (prog_stats["Predicted_Dropouts"] / prog_stats["Total_Students"]) * 100
    prog_stats["High_Risk_Rate_Pct"] = (prog_stats["High_Risk_Students"] / prog_stats["Total_Students"]) * 100

    prog_stats = prog_stats.sort_values(by="Actual_Dropout_Rate_Pct", ascending=True)

    # Plot grouped horizontal bar chart
    fig_prog = go.Figure()
    fig_prog.add_trace(go.Bar(
        y=prog_stats["Program_Name"],
        x=prog_stats["Actual_Dropout_Rate_Pct"],
        name="Actual Dropout Rate (%)",
        orientation="h",
        marker_color="#1f77b4"
    ))
    fig_prog.add_trace(go.Bar(
        y=prog_stats["Program_Name"],
        x=prog_stats["Predicted_Dropout_Rate_Pct"],
        name="Predicted Dropout Rate (%)",
        orientation="h",
        marker_color="#ff7f0e"
    ))
    fig_prog.update_layout(
        barmode="group",
        title="Actual vs. Predicted Dropout Rate (%) across Academic Programs",
        xaxis_title="Dropout Rate (%)",
        yaxis_title="Academic Program",
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_prog, width="stretch")

    st.markdown("#### Academic Program Risk & Attrition Data Table")
    display_prog = prog_stats[[
        "Program_Name", "Total_Students", "Actual_Dropouts",
        "Actual_Dropout_Rate_Pct", "Predicted_Dropout_Rate_Pct", "High_Risk_Students"
    ]].sort_values(by="Actual_Dropout_Rate_Pct", ascending=False)
    
    st.dataframe(
        display_prog.style.format({
            "Actual_Dropout_Rate_Pct": "{:.1f}%",
            "Predicted_Dropout_Rate_Pct": "{:.1f}%"
        }),
        hide_index=True,
        width="stretch"
    )

# ---------------------------------------------------------
# Tab 4: Live PySpark Model Serving (Student Risk Estimator)
# ---------------------------------------------------------
with tab4:
    st.subheader("🔮 Live PySpark Model Inference: Student Risk Estimator")
    st.caption("Interact with the serialized native PySpark PipelineModel (`models/rf_dropout_model`) in real time.")

    @st.cache_resource
    def load_spark_and_model():
        """Loads SparkSession and the trained PipelineModel once."""
        try:
            from src.spark_utils import get_spark_session
            from src.model_trainer import load_model

            spark_instance = get_spark_session("Streamlit_Dropout_Serving")
            model_path = os.path.join(parent_dir, "models", "rf_dropout_model")

            if os.path.exists(model_path):
                model_instance = load_model(model_path)
                return spark_instance, model_instance
            return spark_instance, None
        except Exception as ex:
            st.error(f"Error loading Spark model: {ex}")
            return None, None

    spark_session, trained_model = load_spark_and_model()

    if trained_model is None:
        st.warning("⚠️ Saved model not found at `models/rf_dropout_model`.")
        st.info("Train the PySpark model first by running:\n```bash\npython3 -m src.pipeline_runner\n```")
    else:
        st.success("✅ Native PySpark PipelineModel loaded and ready for live student scoring.")

        # Invert COURSE_MAP for selectbox
        NAME_TO_COURSE = {v: k for k, v in COURSE_MAP.items()}

        with st.form("student_risk_calculator_form"):
            st.markdown("#### Student Academic & Demographic Profile")
            f1, f2, f3 = st.columns(3)

            with f1:
                sel_program = st.selectbox("Academic Program", options=sorted(NAME_TO_COURSE.keys()), index=8)
                sel_study_mode = st.selectbox("Study Mode", ["Daytime", "Evening"])
                sel_gender = st.selectbox("Gender", ["Female", "Male"])
                sel_marital = st.selectbox("Marital Status", ["Single", "Married", "Other"])

            with f2:
                sel_scholarship = st.selectbox("Scholarship Holder", ["No", "Yes"])
                sel_tuition = st.selectbox("Tuition Fees Up to Date", ["Up to Date", "Arrears"])
                sel_debtor = st.selectbox("Debtor Status", ["No Debt", "Has Debt"])
                sel_displaced = st.selectbox("Displaced Student", ["Yes", "No"])

            with f3:
                sel_age = st.slider("Age at Enrollment", min_value=17, max_value=60, value=20)
                sem1_eval = st.slider("1st Sem Evaluations Completed", 0, 15, 6)
                sem1_appr = st.slider("1st Sem Units Approved", 0, 10, 5)
                sem1_grade = st.slider("1st Sem Average Grade (0-20)", 0.0, 20.0, 12.5, step=0.5)

            st.markdown("#### 2nd Semester Progress & Economic Context")
            f4, f5, f6 = st.columns(3)
            with f4:
                sem2_eval = st.slider("2nd Sem Evaluations Completed", 0, 15, 6)
            with f5:
                sem2_appr = st.slider("2nd Sem Units Approved", 0, 10, 5)
            with f6:
                sem2_grade = st.slider("2nd Sem Average Grade (0-20)", 0.0, 20.0, 12.5, step=0.5)

            calc_btn = st.form_submit_button("🚀 Run PySpark Dropout Risk Assessment", width="stretch")

        if calc_btn:
            with st.spinner("Processing through PySpark MLlib Pipeline..."):
                student_record = {
                    "Course": str(NAME_TO_COURSE[sel_program]),
                    "Daytime/evening attendance": "1" if sel_study_mode == "Daytime" else "0",
                    "Marital status": "1" if sel_marital == "Single" else "2",
                    "Gender": "1" if sel_gender == "Male" else "0",
                    "Scholarship holder": "1" if sel_scholarship == "Yes" else "0",
                    "Tuition fees up to date": "1" if sel_tuition == "Up to Date" else "0",
                    "Debtor": "1" if sel_debtor == "Has Debt" else "0",
                    "Displaced": "1" if sel_displaced == "Yes" else "0",
                    "Curricular units 1st sem (evaluations)": sem1_eval,
                    "Curricular units 1st sem (approved)": sem1_appr,
                    "Curricular units 1st sem (grade)": float(sem1_grade),
                    "Curricular units 2nd sem (evaluations)": sem2_eval,
                    "Curricular units 2nd sem (approved)": sem2_appr,
                    "Curricular units 2nd sem (grade)": float(sem2_grade),
                    "Age at enrollment": sel_age,
                    "Unemployment rate": 10.8,
                    "Inflation rate": 1.4,
                    "GDP": 1.74,
                    "Target": "Dropout"  # Dummy for schema compatibility
                }

                patient_spark_df = spark_session.createDataFrame([student_record])
                scored_res = trained_model.transform(patient_spark_df)

                prob_vector = scored_res.select("probability").collect()[0]["probability"]
                dropout_prob = float(prob_vector[1])

                if dropout_prob >= 0.60:
                    risk_badge = "🔴 High Risk"
                    alert_type = st.error
                    advice = "**Urgent Action Required**: Recommend emergency academic advising, tutoring support, and financial counseling."
                elif dropout_prob >= 0.35:
                    risk_badge = "🟡 Moderate Risk"
                    alert_type = st.warning
                    advice = "**Active Monitoring Recommended**: Student shows signs of attrition risk. Schedule a mid-semester check-in."
                else:
                    risk_badge = "🟢 Low Risk"
                    alert_type = st.success
                    advice = "**On Track**: Student is performing well and shows high likelihood of retention and timely graduation."

                r_col1, r_col2, r_col3 = st.columns(3)
                r_col1.metric("Dropout Probability", f"{dropout_prob * 100:.1f}%")
                r_col2.metric("Assigned Risk Tier", risk_badge)
                r_col3.metric("Retention Probability", f"{(1 - dropout_prob) * 100:.1f}%")

                alert_type(f"**Assessment Result**: {risk_badge} ({dropout_prob * 100:.1f}% Dropout Risk)\n\n{advice}")

# ---------------------------------------------------------
# Tab 5: Advisor Actionable Early-Warning Roster
# ---------------------------------------------------------
with tab5:
    st.subheader("📋 Advisor Actionable Early-Warning Roster")
    st.caption("Actionable roster of High-Risk students (P ≥ 0.60) prioritized for academic interventions.")

    high_risk_roster = filtered_df[filtered_df["risk_level"] == "High Risk"].copy()
    high_risk_roster = high_risk_roster.sort_values(by="dropout_probability", ascending=False)

    st.markdown(f"**Total Priority Students Flagged**: `{len(high_risk_roster):,}`")

    display_roster = high_risk_roster[[
        "Program_Name", "Study_Mode_Label", "Scholarship_Label",
        "Tuition_Label", "Age at enrollment",
        "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
        "dropout_probability"
    ]].rename(columns={
        "Program_Name": "Academic Program",
        "Study_Mode_Label": "Study Mode",
        "Scholarship_Label": "Scholarship",
        "Tuition_Label": "Tuition Status",
        "Curricular units 1st sem (approved)": "1st Sem Approved",
        "Curricular units 2nd sem (approved)": "2nd Sem Approved",
        "dropout_probability": "Dropout Risk Prob"
    })

    st.dataframe(
        display_roster.style.format({"Dropout Risk Prob": "{:.1%}"}),
        hide_index=True,
        width="stretch"
    )

    # Download button for advisors
    csv_data = display_roster.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Risk Advisor Outreach Roster (CSV)",
        data=csv_data,
        file_name="high_risk_student_roster.csv",
        mime="text/csv"
    )
