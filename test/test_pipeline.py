import pytest
from pyspark.sql.functions import col, when
from pyspark.ml.functions import vector_to_array
from src.data_loader import preprocess_student_data
from src.model_trainer import build_logistic_regression_pipeline
from src.evaluator import evaluate_classification_predictions


class TestPipelineEndToEnd:
    """Integration test suite covering Tasks 97 through 101 end-to-end."""

    def test_full_classification_pipeline_flow(self, mock_raw_df):
        """
        Tests complete data flow:
        1. Ingestion & Preprocessing (Task 97)
        2. Feature Assembly (Task 98)
        3. Model Training (Task 99)
        4. Model Evaluation (Task 100)
        5. Test Scoring & Risk Tier Categorization (Task 101)
        """
        # 1. Clean data & map binary label
        clean_df = preprocess_student_data(mock_raw_df)
        assert clean_df.count() == mock_raw_df.count()

        # 2. Build Pipeline
        pipeline = build_logistic_regression_pipeline(max_iter=10)

        # 3. Train Model
        model = pipeline.fit(clean_df)
        assert model is not None

        # 4. Generate Predictions on Holdout Test Data
        predictions = model.transform(clean_df)
        assert "prediction" in predictions.columns
        assert "probability" in predictions.columns

        # 5. Evaluate Metrics
        metrics = evaluate_classification_predictions(predictions)
        assert metrics["roc_auc"] > 0.50
        assert metrics["f1_score"] > 0.50

        # 6. Extract Probability & Assign 3-Tier Risk (Task 101)
        scored = predictions.withColumn("dropout_probability", vector_to_array(col("probability"))[1])
        scored = scored.withColumn(
            "risk_level",
            when(col("dropout_probability") >= 0.60, "High Risk")
            .when(col("dropout_probability") >= 0.35, "Moderate Risk")
            .otherwise("Low Risk")
        )
        assert "risk_level" in scored.columns
        risk_values = [row["risk_level"] for row in scored.select("risk_level").collect()]
        assert all(r in ["High Risk", "Moderate Risk", "Low Risk"] for r in risk_values)
