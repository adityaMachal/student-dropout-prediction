import os
import sys

from src.spark_utils import get_spark_session
from pyspark.sql.functions import col, when
from pyspark.ml.functions import vector_to_array

from src.data_loader import load_and_preprocess
from src.model_trainer import train_models
from src.evaluator import evaluate_classification_predictions, compare_and_save_metrics


def run_pipeline(
    data_path: str = "src/data/dataset.csv",
    output_dir: str = "output",
    models_dir: str = "models"
) -> None:
    """
    Executes Tasks 97 through 101 end-to-end:
    - Task 97: Load data & handle missing values
    - Task 98: Feature vector assembly
    - Task 99: Train Logistic Regression & Random Forest Classifier
    - Task 100: Evaluate using Precision, Recall, F1-Score, and ROC-AUC
    - Task 101: Score holdout test set & categorize dropout risk tiers
    """
    print("=" * 70)
    print("🎓 Starting Student Dropout Risk PySpark Pipeline (Tasks 97-101)")
    print("=" * 70)

    # Resolve data path (supports src/data/dataset.csv or data/dataset.csv)
    if not os.path.exists(data_path) and os.path.exists("data/dataset.csv"):
        data_path = "data/dataset.csv"

    # 1. Initialize SparkSession
    spark = get_spark_session("Student_Dropout_Runner")

    # 2. Task 97: Data Loading & Preprocessing
    print(f"\n[Task 97] Loading student records from: {data_path}")
    df_clean = load_and_preprocess(spark, data_path)
    total_count = df_clean.count()
    dropout_count = df_clean.filter(col("label") == 1.0).count()
    print(f"[Task 97] Cleaned records: {total_count:,} (At-Risk Dropouts: {dropout_count:,} / {dropout_count/total_count*100:.1f}%)")

    # 3. Train-Test Split (80 / 20)
    print("\nSplitting dataset into 80% Train and 20% Holdout Test...")
    train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"Train samples: {train_count:,} | Test samples: {test_count:,}")

    # 4. Tasks 98 & 99: Train Classification Models
    print("\n[Tasks 98 & 99] Training classification models via Spark MLlib...")
    models = train_models(train_df, save_dir=models_dir)

    # 5. Task 100: Comprehensive Model Evaluation
    print("\n[Task 100] Evaluating models on holdout test set...")
    preds_lr = models["logistic_regression"].transform(test_df)
    preds_rf = models["random_forest"].transform(test_df)

    metrics_lr = evaluate_classification_predictions(preds_lr)
    metrics_rf = evaluate_classification_predictions(preds_rf)

    print("\n" + "=" * 65)
    print("🎯 Model Performance Metrics (Holdout Test Set)")
    print("=" * 65)
    print(f"{'Metric':<38} {'Logistic Reg':<14} {'Random Forest':<14}")
    print("-" * 65)
    print(f"{'ROC-AUC (Discrimination Power)':<38} {metrics_lr['roc_auc']:<14.4f} {metrics_rf['roc_auc']:<14.4f}")
    print(f"{'F1-Score (Balanced Accuracy)':<38} {metrics_lr['f1_score']:<14.4f} {metrics_rf['f1_score']:<14.4f}")
    print(f"{'Precision (Positive Predictive)':<38} {metrics_lr['precision']:<14.4f} {metrics_rf['precision']:<14.4f}")
    print(f"{'Recall (Overall Sensitivity)':<38} {metrics_lr['recall']:<14.4f} {metrics_rf['recall']:<14.4f}")
    print(f"{'Dropout Recall (At-Risk Catch)':<38} {metrics_lr['dropout_recall']:<14.4f} {metrics_rf['dropout_recall']:<14.4f}")
    print("=" * 65)

    all_metrics = {
        "logistic_regression": metrics_lr,
        "random_forest": metrics_rf,
        "total_students": total_count,
        "train_samples": train_count,
        "test_samples": test_count
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    compare_and_save_metrics(all_metrics, save_path=metrics_path)

    # 6. Task 101: Test Set Risk Scoring & Tier Categorization
    print("\n[Task 101] Scoring holdout students and stratifying into early warning risk tiers...")
    scored_rf = preds_rf.withColumn("dropout_probability", vector_to_array(col("probability"))[1])

    # 3-Tier Risk Assignment
    scored_rf = scored_rf.withColumn(
        "risk_level",
        when(col("dropout_probability") >= 0.60, "High Risk")
        .when(col("dropout_probability") >= 0.35, "Moderate Risk")
        .otherwise("Low Risk")
    )

    export_cols = [
        "Course",
        "Daytime/evening attendance",
        "Scholarship holder",
        "Tuition fees up to date",
        "Debtor",
        "Age at enrollment",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "dropout_probability",
        "risk_level",
        "prediction",
        "label"
    ]

    os.makedirs(output_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, "test_dropout_predictions.csv")
    pdf_export = scored_rf.select(export_cols).toPandas()
    pdf_export.to_csv(csv_out_path, index=False)
    print(f"[Task 101] Successfully exported {len(pdf_export):,} scored student predictions to: {csv_out_path}")

    # Risk Distribution Breakdown
    risk_summary = pdf_export["risk_level"].value_counts()
    print("\n📋 Early Warning Cohort Distribution:")
    for risk_tier, count in risk_summary.items():
        print(f"   • {risk_tier:<14}: {count:,} students ({count/len(pdf_export)*100:.1f}%)")

    print("\n✅ PySpark Pipeline execution complete!")


if __name__ == "__main__":
    run_pipeline()
