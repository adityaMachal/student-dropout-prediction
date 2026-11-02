# 🤖 Master Prompt for Gemini on Web / Canva Docs

If you wish to use **Gemini on Web** or **Canva Docs** to generate slides, an executive summary, or alternative presentation formats, copy and paste the prompt below directly into Gemini on Web:

---

```markdown
You are an expert Professor of Computer Science and Distributed Systems specializing in Apache Spark and Cluster Computing.
I need you to write a comprehensive, publication-grade academic project report for my course:
Course Code: INT315 – Cluster Computing
Project Title: Distributed Early Warning System for Student Dropout Risk Prediction using Apache Spark MLlib on Cluster Infrastructure

Please format the report with clear headings, technical rigor, professional tables, and deep cluster computing explanations covering all 13 syllabus requirements below:

1. Project Title:
   Distributed Early Warning System for Student Dropout Risk Prediction using Apache Spark MLlib on Cluster Infrastructure

2. Problem Statement:
   Explain higher education student attrition, socioeconomic implications, and why traditional single-node machines suffer from memory saturation, CPU bottlenecks during combinatoric tree splitting, and class imbalance bias. Detail how Apache Spark's distributed computing architecture resolves these bottlenecks.

3. Objective of the Project:
   - Ingest and preprocess 4,424 student records with resilient partitioning using PySpark DataFrames.
   - Construct an MLlib feature engineering pipeline using StringIndexer, OneHotEncoder, and VectorAssembler to generate a 43-dimensional feature vector.
   - Train and serialize dual distributed classification models: Logistic Regression (distributed gradient descent with elastic-net regularization) and Random Forest Classifier (distributed parallel tree bagging with candidate split histogram aggregation).
   - Evaluate performance on a 20% holdout test partition (843 students) targeting high Dropout Sensitivity (Recall on positive class) alongside ROC-AUC, F1, Precision, and Recall.
   - Deploy an interactive Streamlit early-warning dashboard satisfying all 3 compulsory evaluator requirements (Risk Categories, Actual vs. Predicted Outcomes, and Program Dropout Rates).

4. Dataset Description:
   "Predict Students' Dropout and Academic Success" dataset (4,424 students, 35 attributes).
   Explain the 4 feature categories: Curricular Academic Performance (1st/2nd sem evaluations, approved units, grades), Demographic Indicators (Age, Gender, Marital Status), Socioeconomic/Institutional Indicators (Course, Study Mode, Scholarship, Tuition fee arrears, Debtor), and Macroeconomic Context (Unemployment, Inflation, GDP).
   Formulate the binary classification target: Y = 1.0 for Dropout (32.12%), Y = 0.0 for Graduate/Enrolled (67.88%).

5. Dataset Source:
   - Original Publication: Realinho et al. (2021), "Predicting Student Dropout and Academic Success", Data, 6(11), 110. DOI: 10.3390/data6110110.
   - UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
   - Kaggle Mirror Dataset reference.

6. Tools & Technologies Used (Cluster Computing Focus):
   Detail the roles of:
   - Apache Spark 4.2.0 (Driver, Executors, Worker Nodes, RDD lineage, Catalyst Optimizer, Tungsten off-heap memory and code generation).
   - PySpark MLlib (Distributed Pipelines, VectorAssembler, MLlib Classifiers).
   - Python 3.12, Py4J Gateway, OpenJDK 21, Streamlit (v1.66+), Plotly Express, and PyTest (14 automated unit/integration tests).

7. Implementation / Methodology:
   Detail the 5 project tasks:
   - Task 97: Data loading with file:// URI safety, binary target conversion, defensive null imputation.
   - Task 98: StringIndexer, OneHotEncoder, and VectorAssembler building a 43-dimensional vector.
   - Task 99: Mathematical formulation of distributed Logistic Regression and Random Forest tree splitting with Gini impurity. Model serialization to models/lr_dropout_model and models/rf_dropout_model.
   - Task 100: Evaluation via ROC-AUC, F1, Precision, Recall, and Dropout Sensitivity.
   - Task 101: Test scoring and 3-tier risk calibration: High Risk (P >= 0.60), Moderate Risk (0.35 <= P < 0.60), Low Risk (P < 0.35). Export to output/test_dropout_predictions.csv.

8. Complete Source Code:
   Organize code into modular sections:
   - src/spark_utils.py (SparkSession builder forcing local fs:///)
   - src/data_loader.py (Data ingestion & preprocessing)
   - src/feature_pipeline.py (Feature stages & VectorAssembler)
   - src/model_trainer.py (Pipeline construction & model fitting)
   - src/evaluator.py (Evaluation metrics & JSON persistence)
   - src/pipeline_runner.py (End-to-end orchestration)
   - dashboard/app.py (Streamlit multi-tab decision support app)
   - test/test_pipeline.py (Automated PyTest test suite)

9. Execution Screenshots:
   Provide structured callout boxes indicating where to insert screenshots for:
   - Pipeline runner ingestion & 80/20 train/test split.
   - Model training completion and disk serialization.
   - Benchmark evaluation output in terminal.
   - PyTest passing all 14 tests in 18.98s.

10. Output / Result Screenshots:
    Provide structured callout boxes for the 5 dashboard views:
    - Tab 1: Compulsory 1 - Risk Categories Donut & Probability Histogram.
    - Tab 2: Compulsory 2 - Actual vs. Predicted Confusion Matrix Heatmap (TP: 191, FP: 36, TN: 536, FN: 80).
    - Tab 3: Compulsory 3 - Dropout Rates across 17 Decoded Academic Programs.
    - Tab 4: Live PySpark Model Serving interactive scoring calculator.
    - Tab 5: Advisor Actionable Early-Warning Outreach Roster with CSV download.

11. Result Analysis:
    Discuss the holdout test benchmark metrics:
    - Random Forest (Winner): ROC-AUC 0.9082, F1-Score 0.8584, Precision 0.8650, Recall 0.8636, Dropout Recall 0.6775.
    - Logistic Regression: ROC-AUC 0.8938, F1-Score 0.8537, Precision 0.8631, Recall 0.8600, Dropout Recall 0.6558.
    - Explain why Random Forest excelled (non-linear interaction modeling, handling skewed enrollment age, high sensitivity on dropouts).
    - Analyze the Confusion Matrix (191 caught dropouts, 84.14% positive predictive value).
    - Analyze Program Disparities (Biofuel Tech ~58% attrition, Informatics Engineering ~45% vs Nursing <14%).

12. Conclusion:
    Summarize achievements in cluster computing scalability, predictive accuracy, and advisor utility. Propose future enhancements using Spark Structured Streaming for LMS clickstreams and distributed hyperparameter tuning via CrossValidator and ParamGridBuilder.

13. Plagiarism Report:
    Include an academic integrity declaration certifying <7% similarity with a placeholder frame for attaching the Turnitin/DrillBit similarity certificate.
```
