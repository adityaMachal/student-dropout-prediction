from typing import List, Tuple, Any
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from src.data_loader import CATEGORICAL_COLS, TARGET_COL

NUMERIC_COLS = [
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]


def build_feature_stages() -> Tuple[List[Any], str]:
    """
    Task 98: Constructs the feature engineering pipeline stages:
    1. StringIndexer: Encodes categorical features (course, study mode, demographics).
    2. OneHotEncoder: Transforms indexed categories into sparse one-hot vectors.
    3. VectorAssembler: Combines numeric metrics (evaluations, approved units, grades)
       and encoded vectors into a single 'features' vector.

    Returns:
        stages: List of feature pipeline stages.
        features_col: Name of the assembled features vector column ('features').
    """
    indexed_cols = [f"{c}_idx" for c in CATEGORICAL_COLS]
    encoded_cols = [f"{c}_vec" for c in CATEGORICAL_COLS]

    # StringIndexer stage
    indexer = StringIndexer(
        inputCols=CATEGORICAL_COLS,
        outputCols=indexed_cols,
        handleInvalid="keep"
    )

    # OneHotEncoder stage
    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols
    )

    # VectorAssembler stage
    assembler_inputs = NUMERIC_COLS + encoded_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="skip"
    )

    stages = [indexer, encoder, assembler]
    return stages, "features"
