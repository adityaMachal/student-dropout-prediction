# 🎓 Student Dropout Risk Prediction & Academic Early Warning System

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-E25A1C?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![PySpark](https://img.shields.io/badge/PySpark-MLlib-blue?style=flat&logo=python&logoColor=white)](https://spark.apache.org/docs/latest/api/python/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![PyTest](https://img.shields.io/badge/Tested%20with-PyTest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end distributed machine learning pipeline and interactive early warning decision support system to identify students at risk of academic dropout. Built with **Apache Spark 4.2.0 (PySpark MLlib)** for large-scale data ingestion, categorical encoding, vector assembly, and classification modeling, accompanied by an interactive **Streamlit + Plotly** web dashboard satisfying all evaluator requirements.

---

## 📌 Institutional Context & Objectives

Student attrition poses significant institutional, academic, and economic challenges in higher education. Identifying at-risk students before irreversible dropouts occur allows institutions to take targeted, proactive measures:
- **Targeted Advising & Tutoring**: Proactively dispatch academic counseling to students struggling in 1st/2nd semester curricular units.
- **Financial Aid & Bursary Outreach**: Detect students facing tuition arrears or lack of scholarship support before financial disenrollment.
- **Academic Program Interventions**: Diagnose structural attrition bottlenecks across specific degree courses (e.g., Informatics Engineering vs. Nursing).

Using the **Predict Students' Dropout and Academic Success** dataset (4,424 student records across 35 attributes), this system implements distributed classification pipelines in PySpark to score continuous dropout probabilities and stratify students into actionable early-warning risk tiers.

---

## 🎯 Compulsory Evaluator Requirements & Implementation

The dashboard directly fulfills the mandatory evaluator specifications:

| Compulsory Requirement | Implementation in System & Dashboard |
| :--- | :--- |
| **1. Risk Categories** | Stratifies student cohort into 3 actionable tiers based on calibrated PySpark class probabilities:<br>• 🔴 **High Risk** ($P \ge 0.60$): Immediate intervention required.<br>• 🟡 **Moderate Risk** ($0.35 \le P < 0.60$): Active academic monitoring advised.<br>• 🟢 **Low Risk** ($P < 0.35$): On track for retention and graduation.<br>*Features interactive donut breakdown, probability distribution histogram, and demographic cross-cuts.* |
| **2. Actual vs. Predicted Dropout Outcomes** | Compares ground truth labels against MLlib classifications:<br>• Interactive **Confusion Matrix Heatmap** (True Positives, False Positives, True Negatives, False Negatives).<br>• Grouped bar charts contrasting actual vs. predicted counts.<br>• Model benchmark metrics (**ROC-AUC**, **F1-Score**, **Precision**, **Recall**, and **Dropout Sensitivity**). |
| **3. Dropout Rates by Academic Program** | Decodes raw integer course IDs (`1`–`17`) into official degree titles (*Nursing, Informatics Engineering, Management, Social Service, Agronomy, etc.*).<br>• Visualizes actual vs. predicted attrition percentages across all programs.<br>• Highlights high-attrition majors to guide institutional curriculum reviews. |
| **+ Live Model Serving & Early Warning Roster** | • **Live PySpark Inference**: Clinician/Advisor input form calling the native serialized `PipelineModel` in real time.<br>• **Advisor Priority Roster**: Filterable early-warning table of all High-Risk students with direct **CSV Export** for counseling outreach. |

---

## 📋 Tasks Breakdown (Tasks 97–101)

| Task # | Scope | PySpark MLlib Implementation Details |
| :--- | :--- | :--- |
| **Task 97** | **Data Ingestion & Null Handling** | Loads student records via `SparkSession` with local `file://` URIs. Maps string `Target` to binary classification labels (`Dropout` ➔ `1.0`, `Graduate` / `Enrolled` ➔ `0.0`). Casts categorical integer columns to strings. Performs defensive null imputation (`0.0` for numerics, `"Unknown"` for categoricals). |
| **Task 98** | **Categorical Encoding & Vector Assembly** | Applies `StringIndexer` and `OneHotEncoder` across all categorical features (`Course`, `Daytime/evening attendance`, `Marital status`, `Gender`, `Scholarship holder`, `Tuition fees up to date`, `Debtor`, `Displaced`). Assembles encoded vectors and continuous academic metrics into a 43-dimensional `features` vector via `VectorAssembler`. |
| **Task 99** | **Classification Modeling** | Constructs two distinct Spark ML Pipelines: **Logistic Regression** (`maxIter=30`, `regParam=0.1`) and **Random Forest Classifier** (`numTrees=50`, `maxDepth=8`, `seed=42`). Trains models and serializes pipeline artifacts to `models/lr_dropout_model` and `models/rf_dropout_model`. |
| **Task 100** | **Model Evaluation & Benchmarking** | Uses `BinaryClassificationEvaluator` and `MulticlassClassificationEvaluator` to measure **ROC-AUC**, **F1-Score**, **Precision**, and **Recall**. Computes specialized **Dropout Recall (Sensitivity)** to ensure high capture rate of at-risk students. Exports metrics to `output/metrics.json`. |
| **Task 101** | **Holdout Test Scoring & Risk Tiering** | Generates class predictions and extracts continuous class probability vectors ($P(\text{Dropout})$). Stratifies the holdout test set (843 students) into **High Risk**, **Moderate Risk**, and **Low Risk** tiers. Exports results to `output/test_dropout_predictions.csv`. |

---

## 📊 Benchmark Results (Holdout Test Cohort)

Models were evaluated on a stratified 80/20 train-test split (843 test students):

| Metric | Logistic Regression | Random Forest Classifier | Performance Impact |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | `0.8938` | **`0.9082`** | **+0.0144** (Superior ranking of risk probabilities) |
| **F1-Score (Weighted)** | `0.8537` | **`0.8584`** | **+0.0047** (Higher overall balance across classes) |
| **Precision (Weighted)** | `0.8631` | **`0.8650`** | **+0.0019** (Fewer false alarms) |
| **Recall (Weighted)** | `0.8600` | **`0.8636`** | **+0.0036** (Stronger overall instance capture) |
| **Dropout Sensitivity** | `0.6558` | **`0.6775`** | **+0.0217** (Catches ~68% of dropouts early) |

> **Key Takeaway**: The **Random Forest Classifier** is selected as the production model due to its outstanding **0.9082 ROC-AUC** and superior sensitivity on the minority dropout class, capturing complex non-linear combinations of 1st/2nd semester course failures and tuition fee arrears.

---

## 🏗️ Project Architecture & Directory Layout

```
student-dropout-prediction/
├── notebooks/                        # Phase 1: Exploratory Analysis & Prototyping
│   └── 01_dropout_research_eda.ipynb # PySpark EDA, class imbalance analysis & visualizations
├── src/                              # Production PySpark MLlib Package
│   ├── __init__.py                   # Package initialization & PySpark sys.path bootstrap
│   ├── spark_utils.py                # SparkSession factory (forces local fs:///)
│   ├── data_loader.py                # Task 97: Loading, binary mapping & null handling
│   ├── feature_pipeline.py           # Task 98: StringIndexer, OneHotEncoder & VectorAssembler
│   ├── model_trainer.py              # Task 99: Pipeline training & model serialization
│   ├── evaluator.py                  # Task 100: ROC-AUC, F1, Precision, Recall & Dropout Sensitivity
│   └── pipeline_runner.py            # Task 101: End-to-end orchestration & risk scoring
├── dashboard/                        # Evaluator Requirements Dashboard
│   └── app.py                        # Multi-tab Streamlit dashboard with PySpark live inference
├── models/                           # Serialized PySpark PipelineModels (git-ignored)
│   ├── lr_dropout_model/             # Fitted Logistic Regression MLlib pipeline
│   └── rf_dropout_model/             # Fitted Random Forest Classifier MLlib pipeline
├── output/                           # Pre-computed evaluation metrics and scored test cohort
│   ├── metrics.json                  # Model comparison metrics
│   └── test_dropout_predictions.csv  # 843 scored test students with risk tiers (git-ignored)
├── test/                             # Automated PyTest Test Suite (14 unit & integration tests)
│   ├── conftest.py                   # Session-scoped SparkSession & synthetic student fixtures
│   ├── test_data_loader.py           # Target mapping & null imputation unit tests
│   ├── test_feature_pipeline.py      # Encoder & VectorAssembler dimension tests
│   ├── test_model_trainer.py         # Pipeline fitting & model persistence tests
│   ├── test_evaluator.py             # Metric computation & JSON export tests
│   └── test_pipeline.py              # End-to-end classification pipeline test
├── pytest.ini                        # PyTest configuration & warning filters
├── requirements.txt                  # Python runtime dependencies
├── run.sh                            # Automated all-in-one execution script
└── .gitignore                        # Git ignore rules for datasets, models & caches
```

---

## ⚡ Quick Start

### 1. Prerequisites
- **Java**: OpenJDK 17 or 21 (`java -version`)
- **Apache Spark**: Spark 4.2.0 installed (e.g. at `/opt/spark`)
- **Python**: Python 3.10+ / Conda environment (e.g., `conda activate ML`)

### 2. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/student-dropout-prediction.git
cd student-dropout-prediction
pip install -r requirements.txt
```

### 3. Run the PySpark Pipeline (Tasks 97–101)
Execute the complete distributed pipeline to train models, evaluate metrics, and score the test set:
```bash
python3 -m src.pipeline_runner
```
*Outputs generated: `output/metrics.json` and `output/test_dropout_predictions.csv`.*

### 4. Launch the Interactive Dashboard
Launch the Streamlit early warning web application:
```bash
streamlit run dashboard/app.py --server.port 8501
```
Open **`http://localhost:8501`** to explore the dashboard:
- **Tab 1: 🚨 Risk Categories**: Donut breakdown of High/Moderate/Low risk tiers, probability distribution histogram, and demographic breakdowns.
- **Tab 2: ⚖️ Actual vs. Predicted Outcomes**: Interactive Confusion Matrix Heatmap, outcome counts comparison, and full model benchmark metrics.
- **Tab 3: 🏫 Dropout Rates by Academic Program**: Grouped horizontal bar chart of actual vs. predicted attrition across all 17 decoded degree programs.
- **Tab 4: 🔮 Live Student Risk Calculator**: Interactive advisor input form running live inference against the loaded native PySpark `PipelineModel`.
- **Tab 5: 📋 Advisor Early Warning Roster**: Filterable table of all High-Risk students ($P \ge 0.60$) with one-click **CSV download** for outreach.

### 5. Run Automated Tests
Execute the comprehensive PyTest test suite (14 tests across unit and integration levels):
```bash
pytest -v
```

---

## 🧪 Testing & Verification

The test suite covers all pipeline stages using session-scoped Spark fixtures:
- **`test_data_loader.py`**: Verifies binary target mapping (`Dropout` ➔ `1.0`, `Graduate`/`Enrolled` ➔ `0.0`), categorical casting, and defensive null handling.
- **`test_feature_pipeline.py`**: Asserts that `OneHotEncoder` and `VectorAssembler` produce valid non-null feature vectors of the expected dimensionality (43 features).
- **`test_model_trainer.py`**: Validates MLlib pipeline building, model training, prediction generation, and disk serialization/re-loading.
- **`test_evaluator.py`**: Validates ROC-AUC, F1, Precision, Recall, and Dropout Sensitivity calculations against known test cases.
- **`test_pipeline.py`**: Runs an end-to-end integration test ensuring complete data flow from raw CSV to scored risk tiers.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
