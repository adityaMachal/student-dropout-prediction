import json
import pytest
from pyspark.ml.linalg import Vectors
from src.evaluator import evaluate_classification_predictions, compare_and_save_metrics


class TestEvaluator:
    """Unit tests for Task 100: Model Evaluation (Precision, Recall, F1, ROC-AUC)."""

    def test_evaluate_classification_metrics(self, spark):
        """Verifies Precision, Recall, F1, and ROC-AUC computation."""
        # Simulated predictions:
        # label, prediction, rawPrediction, probability
        data = [
            (1.0, 1.0, Vectors.dense([-1.5, 1.5]), Vectors.dense([0.18, 0.82])),
            (1.0, 1.0, Vectors.dense([-2.0, 2.0]), Vectors.dense([0.12, 0.88])),
            (0.0, 0.0, Vectors.dense([2.0, -2.0]), Vectors.dense([0.88, 0.12])),
            (0.0, 1.0, Vectors.dense([-0.5, 0.5]), Vectors.dense([0.38, 0.62])),  # False positive
            (0.0, 0.0, Vectors.dense([1.5, -1.5]), Vectors.dense([0.82, 0.18]))
        ]
        df = spark.createDataFrame(data, ["label", "prediction", "rawPrediction", "probability"])
        metrics = evaluate_classification_predictions(df)

        assert "roc_auc" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "dropout_recall" in metrics

        assert metrics["roc_auc"] > 0.80
        assert metrics["f1_score"] > 0.70
        assert metrics["dropout_recall"] == 1.0  # Caught both dropouts

    def test_save_and_read_metrics_json(self, tmp_path):
        """Validates that evaluation metrics serialize cleanly to a JSON file."""
        sample_metrics = {
            "logistic_regression": {"roc_auc": 0.88, "f1_score": 0.82},
            "random_forest": {"roc_auc": 0.90, "f1_score": 0.85}
        }
        json_path = str(tmp_path / "metrics_test.json")
        compare_and_save_metrics(sample_metrics, save_path=json_path)

        with open(json_path, "r") as f:
            loaded = json.load(f)

        assert loaded["random_forest"]["roc_auc"] == 0.90
