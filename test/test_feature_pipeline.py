import pytest
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector
from src.feature_pipeline import (
    build_feature_stages,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL
)


class TestFeaturePipeline:
    """Unit tests for Task 98: Feature Vector Assembly."""

    def test_build_feature_stages_count_and_types(self):
        """Verifies that feature engineering contains StringIndexer, OneHotEncoder, and VectorAssembler."""
        stages, feature_col = build_feature_stages()
        assert len(stages) == 3
        assert feature_col == "features"

        indexer, encoder, assembler = stages
        assert hasattr(indexer, "setInputCols")
        assert hasattr(encoder, "setInputCols")
        assert hasattr(assembler, "setInputCols")

    def test_feature_transformation_pipeline(self, mock_clean_df):
        """Validates that patient attributes transform into a dense/sparse vector without NaNs."""
        stages, _ = build_feature_stages()
        pipeline = Pipeline(stages=stages)
        fitted_pipeline = pipeline.fit(mock_clean_df)
        transformed_df = fitted_pipeline.transform(mock_clean_df)

        assert "features" in transformed_df.columns

        first_row = transformed_df.select("features", TARGET_COL).first()
        vec = first_row["features"]
        assert isinstance(vec, Vector)
        assert vec.size > len(NUMERIC_COLS)  # Includes encoded categorical dimensions

        # Verify no NaN values in assembled vector
        assert not any(float(val) != float(val) for val in vec.toArray())
