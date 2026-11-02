import os
from typing import Dict
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import DataFrame

from src.feature_pipeline import build_feature_stages, TARGET_COL


def build_logistic_regression_pipeline(
    max_iter: int = 30,
    reg_param: float = 0.01,
    elastic_net_param: float = 0.5
) -> Pipeline:
    """Task 99: Builds a Spark ML Pipeline with Logistic Regression."""
    feature_stages, features_col = build_feature_stages()
    lr = LogisticRegression(
        featuresCol=features_col,
        labelCol=TARGET_COL,
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net_param
    )
    return Pipeline(stages=feature_stages + [lr])


def build_random_forest_pipeline(
    num_trees: int = 80,
    max_depth: int = 8,
    seed: int = 42
) -> Pipeline:
    """Task 99: Builds a Spark ML Pipeline with Random Forest Classifier."""
    feature_stages, features_col = build_feature_stages()
    rf = RandomForestClassifier(
        featuresCol=features_col,
        labelCol=TARGET_COL,
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=seed
    )
    return Pipeline(stages=feature_stages + [rf])


def train_models(
    train_df: DataFrame,
    save_dir: str = "models"
) -> Dict[str, PipelineModel]:
    """
    Task 99: Trains both Logistic Regression and Random Forest models
    on training data and serializes them to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Training Logistic Regression Pipeline...")
    pipeline_lr = build_logistic_regression_pipeline()
    model_lr = pipeline_lr.fit(train_df)
    lr_path = os.path.join(save_dir, "lr_dropout_model")
    model_lr.write().overwrite().save(f"file://{os.path.abspath(lr_path)}")
    print(f"Saved Logistic Regression model to: {lr_path}")

    print("Training Random Forest Classifier Pipeline...")
    pipeline_rf = build_random_forest_pipeline()
    model_rf = pipeline_rf.fit(train_df)
    rf_path = os.path.join(save_dir, "rf_dropout_model")
    model_rf.write().overwrite().save(f"file://{os.path.abspath(rf_path)}")
    print(f"Saved Random Forest model to: {rf_path}")

    return {
        "logistic_regression": model_lr,
        "random_forest": model_rf
    }


def load_model(model_path: str) -> PipelineModel:
    """Loads a serialized PySpark PipelineModel from disk."""
    abs_path = os.path.abspath(model_path)
    return PipelineModel.load(f"file://{abs_path}")
