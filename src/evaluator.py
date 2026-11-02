import json
import os
from typing import Dict, Any
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import DataFrame

from src.data_loader import TARGET_COL


def evaluate_classification_predictions(
    predictions_df: DataFrame,
    label_col: str = TARGET_COL,
    prediction_col: str = "prediction",
    raw_pred_col: str = "rawPrediction"
) -> Dict[str, float]:
    """
    Task 100: Evaluates classification performance using:
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - Dropout-specific Recall (Sensitivity on class 1.0)
    """
    eval_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="f1"
    )
    eval_precision = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="weightedPrecision"
    )
    eval_recall = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="weightedRecall"
    )
    eval_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol=raw_pred_col,
        metricName="areaUnderROC"
    )

    f1 = eval_f1.evaluate(predictions_df)
    precision = eval_precision.evaluate(predictions_df)
    recall = eval_recall.evaluate(predictions_df)
    auc = eval_auc.evaluate(predictions_df)

    # Dropout-specific recall (Sensitivity on at-risk class 1.0)
    eval_dropout_recall = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="recallByLabel",
        metricLabel=1.0
    )
    dropout_recall = eval_dropout_recall.evaluate(predictions_df)

    return {
        "roc_auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "dropout_recall": round(dropout_recall, 4)
    }


def compare_and_save_metrics(
    eval_results: Dict[str, Any],
    save_path: str = "output/metrics.json"
) -> None:
    """Saves model evaluation comparison to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Metrics saved to: {save_path}")
