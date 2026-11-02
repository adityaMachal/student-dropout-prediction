import os
from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when

# Categorical columns to cast to StringType for StringIndexer
CATEGORICAL_COLS = [
    "Course",
    "Daytime/evening attendance",
    "Marital status",
    "Gender",
    "Scholarship holder",
    "Tuition fees up to date",
    "Debtor",
    "Displaced"
]

TARGET_COL = "label"


def load_student_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Task 97: Loads the student dataset into a Spark DataFrame
    using an explicit absolute file:// path.
    """
    abs_path = os.path.abspath(file_path)
    file_uri = f"file://{abs_path}" if not abs_path.startswith("file://") else abs_path

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Student dataset not found at: {abs_path}")

    df = spark.read.csv(file_uri, header=True, inferSchema=True)
    return df


def preprocess_student_data(df: DataFrame) -> DataFrame:
    """
    Task 97: Converts Target to binary label (1.0 = Dropout, 0.0 = Retained),
    casts categorical columns to StringType, and handles missing/null values.
    """
    # 1. Binary target mapping
    if "Target" in df.columns:
        df_transformed = df.withColumn(
            "label",
            when(col("Target") == "Dropout", 1.0).otherwise(0.0)
        )
    else:
        df_transformed = df

    # 2. Cast categorical integer codes to strings for StringIndexer
    for c in CATEGORICAL_COLS:
        if c in df_transformed.columns:
            df_transformed = df_transformed.withColumn(c, col(c).cast("string"))

    # 3. Defensive null handling
    fill_string_dict = {c: "Unknown" for c in CATEGORICAL_COLS if c in df_transformed.columns}
    if fill_string_dict:
        df_clean = df_transformed.na.fill(fill_string_dict)
    else:
        df_clean = df_transformed

    # Filter out null targets if present
    if "label" in df_clean.columns:
        df_clean = df_clean.filter(col("label").isNotNull())

    return df_clean


def load_and_preprocess(spark: SparkSession, file_path: str) -> DataFrame:
    """Convenience helper combining data loading and preprocessing."""
    df_raw = load_student_data(spark, file_path)
    return preprocess_student_data(df_raw)
