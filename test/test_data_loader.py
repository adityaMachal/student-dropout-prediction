import pytest
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col
from src.data_loader import load_student_data, preprocess_student_data


class TestDataLoader:
    """Unit tests for Task 97: Data Loading & Preprocessing."""

    @pytest.mark.parametrize(
        "target_status, expected_label",
        [
            ("Dropout", 1.0),
            ("Graduate", 0.0),
            ("Enrolled", 0.0)
        ]
    )
    def test_binary_target_mapping(self, spark, target_status, expected_label):
        """Verifies that Dropout maps to 1.0 and non-dropouts map to 0.0."""
        schema = StructType([
            StructField("Target", StringType(), True),
            StructField("Course", StringType(), True)
        ])
        df_single = spark.createDataFrame([(target_status, "2")], schema=schema)
        cleaned = preprocess_student_data(df_single)
        row = cleaned.select("label").first()
        assert row["label"] == expected_label

    def test_categorical_column_casting(self, mock_clean_df):
        """Verifies that categorical columns are cast to StringType for StringIndexer."""
        schema_dict = dict(mock_clean_df.dtypes)
        assert schema_dict["Course"] == "string"
        assert schema_dict["Daytime/evening attendance"] == "string"
        assert schema_dict["Scholarship holder"] == "string"

    def test_missing_value_handling(self, spark):
        """Ensures missing categorical entries receive 'Unknown' fallback."""
        schema = StructType([
            StructField("Course", StringType(), True),
            StructField("Target", StringType(), True)
        ])
        df_nulls = spark.createDataFrame([(None, "Dropout")], schema=schema)
        cleaned = preprocess_student_data(df_nulls)
        row = cleaned.select("Course", "label").first()
        assert row["Course"] == "Unknown"
        assert row["label"] == 1.0

    def test_file_not_found_raises(self, spark):
        """Ensures FileNotFoundError is raised when pointing to a missing CSV file."""
        with pytest.raises(FileNotFoundError):
            load_student_data(spark, "invalid_dir/no_such_dataset.csv")
