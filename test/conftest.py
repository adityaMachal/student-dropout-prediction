import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark_utils import get_spark_session
from src.data_loader import preprocess_student_data


@pytest.fixture(scope="session")
def spark():
    """
    Session-scoped PySpark fixture.
    Initializes a single local SparkSession shared across test modules.
    """
    spark_session = get_spark_session(app_name="PyTest_Student_Dropout_Suite")
    yield spark_session


@pytest.fixture
def sample_student_columns():
    """Returns standard columns matching the student dataset."""
    return [
        "Course",
        "Daytime/evening attendance",
        "Marital status",
        "Gender",
        "Scholarship holder",
        "Tuition fees up to date",
        "Debtor",
        "Displaced",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Age at enrollment",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
        "Target"
    ]


@pytest.fixture
def sample_student_data():
    """Provides representative mock student records."""
    return [
        (2, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0.0, 0, 0, 0.0, 20, 10.8, 1.4, 1.74, "Dropout"),
        (11, 1, 1, 1, 0, 0, 0, 1, 6, 6, 14.0, 6, 6, 13.6, 19, 13.9, -0.3, 0.79, "Graduate"),
        (5, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0.0, 0, 0, 0.0, 19, 10.8, 1.4, 1.74, "Dropout"),
        (15, 1, 1, 0, 0, 1, 0, 1, 8, 6, 13.4, 10, 5, 12.4, 20, 9.4, -0.8, -3.12, "Graduate"),
        (3, 0, 1, 1, 1, 1, 0, 0, 5, 4, 11.5, 5, 4, 11.0, 22, 12.0, 0.5, 1.2, "Enrolled")
    ]


@pytest.fixture
def mock_raw_df(spark, sample_student_data, sample_student_columns):
    """Returns a Spark DataFrame populated with mock student records."""
    return spark.createDataFrame(sample_student_data, sample_student_columns)


@pytest.fixture
def mock_clean_df(mock_raw_df):
    """Returns preprocessed DataFrame ready for feature engineering."""
    return preprocess_student_data(mock_raw_df)
