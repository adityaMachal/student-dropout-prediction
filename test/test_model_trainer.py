import os
import pytest
from pyspark.ml import PipelineModel
from src.model_trainer import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    load_model
)


class TestModelTrainer:
    """Unit tests for Task 99: Classification Modeling."""

    def test_build_logistic_regression_pipeline(self):
        """Validates that Logistic Regression pipeline is properly constructed."""
        pipeline = build_logistic_regression_pipeline(max_iter=10)
        stages = pipeline.getStages()
        assert len(stages) == 4
        assert stages[-1].__class__.__name__ == "LogisticRegression"

    def test_build_random_forest_pipeline(self):
        """Validates that Random Forest Classifier pipeline is properly constructed."""
        pipeline = build_random_forest_pipeline(num_trees=10, max_depth=5)
        stages = pipeline.getStages()
        assert len(stages) == 4
        assert stages[-1].__class__.__name__ == "RandomForestClassifier"

    def test_model_fit_predict_and_persistence(self, mock_clean_df, tmp_path):
        """Verifies training, prediction generation, model serialization and reloading."""
        pipeline = build_logistic_regression_pipeline(max_iter=5)
        model = pipeline.fit(mock_clean_df)

        predictions = model.transform(mock_clean_df)
        assert "prediction" in predictions.columns
        assert "probability" in predictions.columns

        # Verify predictions match binary labels (0.0 or 1.0)
        pred_labels = [row["prediction"] for row in predictions.select("prediction").collect()]
        assert all(p in [0.0, 1.0] for p in pred_labels)

        # Test model persistence to disk
        save_path = str(tmp_path / "test_lr_model")
        model.write().overwrite().save(f"file://{save_path}")
        assert os.path.exists(save_path)

        # Test reloading serialized model
        reloaded = load_model(save_path)
        assert isinstance(reloaded, PipelineModel)
