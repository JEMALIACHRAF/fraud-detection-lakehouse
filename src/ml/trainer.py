"""
ML Training — Fraud Detection Model with MLflow on Databricks.

Pipeline:
1. Load Gold features from Delta Lake
2. Preprocessing (encoding, scaling, imbalance handling)
3. Train XGBoost classifier
4. Evaluate (precision, recall, F1, AUC-ROC, AP)
5. Log everything to MLflow (params, metrics, model, feature importance)
6. Register best model in MLflow Model Registry
7. Promote to Production if performance threshold met
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.common.config import PipelineConfig, MLConfig
from src.common.exceptions import MLflowError
from src.common.logger import get_logger

logger = get_logger(__name__)

# ── Feature groups ────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "amount", "tx_hour", "tx_dow",
    "tx_count_1h", "tx_amount_1h",
    "tx_count_6h", "tx_amount_6h",
    "tx_count_24h", "tx_amount_24h", "tx_merchants_24h",
    "tx_count_7d", "tx_amount_7d", "tx_merchants_7d",
    "amount_mean_30d", "amount_std_30d", "amount_zscore",
    "amount_ratio_to_max",
    "time_since_last_tx_seconds",
    "merchant_category_diversity_7d",
    "account_total_tx", "account_avg_amount",
    "account_historical_fraud_rate",
    "account_distinct_countries", "account_distinct_categories",
    "account_age_days",
]

CATEGORICAL_FEATURES = [
    "currency", "transaction_type", "channel",
    "country_code", "merchant_category",
]

BOOLEAN_FEATURES = [
    "is_night_transaction", "is_weekend", "is_new_merchant",
]

TARGET = "is_fraud"

# Minimum performance thresholds to promote model to Production
PRODUCTION_THRESHOLDS = {
    "roc_auc": 0.85,
    "average_precision": 0.70,
    "f1_fraud": 0.65,
}


class FraudModelTrainer:
    """
    End-to-end fraud model trainer with MLflow tracking on Databricks.

    Uses XGBoost with SMOTE oversampling (fraud is typically < 1% of transactions).
    Logs all experiments to MLflow and registers the best model.
    """

    def __init__(self, spark: SparkSession, config: PipelineConfig) -> None:
        self.spark = spark
        self.config = config
        self.ml_config: MLConfig = config.ml
        self._gold_path = f"gs://{config.gcs.bucket}/{config.gcs.gold_prefix}"
        self._client = MlflowClient(tracking_uri=config.databricks.mlflow_tracking_uri)
        self._label_encoders: dict[str, LabelEncoder] = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_training_data(self, lookback_days: int = 180) -> pd.DataFrame:
        """
        Load Gold features for training.

        Args:
            lookback_days: History window for training data

        Returns:
            Pandas DataFrame with features and target
        """
        logger.info("Loading training data", extra={"lookback_days": lookback_days})

        df = (
            self.spark.read
            .format("delta")
            .load(self._gold_path)
            .filter(
                F.col("feature_date") >= F.date_sub(F.current_date(), lookback_days)
            )
            .filter(F.col(TARGET).isNotNull())
            .select(NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES + [TARGET])
        )

        pdf = df.toPandas()
        logger.info(
            "Training data loaded",
            extra={
                "rows": len(pdf),
                "fraud_rate": f"{pdf[TARGET].mean():.4f}",
                "fraud_count": int(pdf[TARGET].sum()),
            }
        )
        return pdf

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode categoricals, cast booleans, fill nulls.

        Args:
            df: Raw feature DataFrame

        Returns:
            (X, y) numpy arrays ready for training
        """
        df = df.copy()

        # Encode categoricals
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
            self._label_encoders[col] = le

        # Cast booleans
        for col in BOOLEAN_FEATURES:
            df[col] = df[col].astype(int)

        # Fill remaining nulls
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(-1)

        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
        X = df[feature_cols].values
        y = df[TARGET].astype(int).values

        return X, y

    def apply_smote(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE oversampling to handle class imbalance.

        Only applied when fraud rate < 5% to avoid over-sampling
        when data is already balanced.
        """
        fraud_rate = y_train.mean()
        logger.info("Applying SMOTE", extra={"fraud_rate_before": f"{fraud_rate:.4f}"})

        if fraud_rate >= 0.05:
            logger.info("Fraud rate >= 5% — skipping SMOTE")
            return X_train, y_train

        smote = SMOTE(
            sampling_strategy=0.1,  # target 10% fraud ratio
            random_state=self.ml_config.random_state,
            k_neighbors=5,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE complete",
            extra={"fraud_rate_after": f"{y_res.mean():.4f}", "samples_added": len(X_res) - len(X_train)},
        )
        return X_res, y_res

    # ── Training ──────────────────────────────────────────────────────────────

    def build_model(self, scale_pos_weight: float = 1.0) -> XGBClassifier:
        """Build XGBoost classifier with production-ready hyperparameters."""
        return XGBClassifier(
            n_estimators=self.ml_config.n_estimators,
            max_depth=self.ml_config.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=self.ml_config.random_state,
            n_jobs=-1,
        )

    def evaluate(
        self,
        model: XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Returns:
            Dict of metric name → value
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.ml_config.fraud_threshold).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "average_precision": average_precision_score(y_test, y_pred_proba),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "f1_fraud": report.get("1", {}).get("f1-score", 0.0),
            "precision_fraud": report.get("1", {}).get("precision", 0.0),
            "recall_fraud": report.get("1", {}).get("recall", 0.0),
            "true_positives": int(cm[1][1]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_negatives": int(cm[0][0]),
        }

        logger.info("Evaluation complete", extra=metrics)
        return metrics

    # ── MLflow run ────────────────────────────────────────────────────────────

    def train_with_mlflow(self, pdf: pd.DataFrame) -> tuple[str, dict]:
        """
        Run full training pipeline inside an MLflow experiment run.

        Returns:
            (run_id, metrics) tuple
        """
        mlflow.set_tracking_uri(self.config.databricks.mlflow_tracking_uri)
        mlflow.set_experiment(self.ml_config.experiment_name)

        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            logger.info("MLflow run started", extra={"run_id": run_id})

            # Preprocess
            X, y = self.preprocess(pdf)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.ml_config.test_size,
                random_state=self.ml_config.random_state,
                stratify=y,
            )

            # SMOTE
            X_train_res, y_train_res = self.apply_smote(X_train, y_train)

            # Compute scale_pos_weight from original (pre-SMOTE) distribution
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            # Log params
            params = {
                "n_estimators": self.ml_config.n_estimators,
                "max_depth": self.ml_config.max_depth,
                "fraud_threshold": self.ml_config.fraud_threshold,
                "test_size": self.ml_config.test_size,
                "smote_applied": True,
                "scale_pos_weight": round(scale_pos_weight, 2),
                "train_samples": len(X_train_res),
                "test_samples": len(X_test),
                "train_fraud_rate": round(y_train.mean(), 4),
            }
            mlflow.log_params(params)

            # Train
            start = time.time()
            model = self.build_model(scale_pos_weight=scale_pos_weight)
            model.fit(
                X_train_res, y_train_res,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
            train_time = time.time() - start
            mlflow.log_metric("training_time_seconds", round(train_time, 2))
            logger.info("Model trained", extra={"training_time_s": round(train_time, 2)})

            # Evaluate
            metrics = self.evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Feature importance
            feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

            importance_path = "/tmp/feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path, "feature_importance")

            # Log top 10 importances as metrics
            for _, row in importance_df.head(10).iterrows():
                mlflow.log_metric(f"fi_{row['feature']}", round(float(row["importance"]), 6))

            # Log model
            feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.ml_config.model_name,
                input_example=pd.DataFrame(X_test[:5], columns=feature_names),
            )

            logger.info("MLflow run complete", extra={"run_id": run_id, **metrics})
            return run_id, metrics

    # ── Model Registry ────────────────────────────────────────────────────────

    def promote_to_production(self, run_id: str, metrics: dict) -> bool:
        """
        Promote model to Production in MLflow Model Registry if thresholds met.

        Args:
            run_id: MLflow run ID
            metrics: Evaluation metrics dict

        Returns:
            True if promoted, False otherwise
        """
        passes = all(
            metrics.get(metric, 0) >= threshold
            for metric, threshold in PRODUCTION_THRESHOLDS.items()
        )

        if not passes:
            failing = {
                k: f"{metrics.get(k, 0):.3f} < {v}"
                for k, v in PRODUCTION_THRESHOLDS.items()
                if metrics.get(k, 0) < v
            }
            logger.warning(
                "Model did NOT meet production thresholds",
                extra={"failing_metrics": failing},
            )
            return False

        try:
            # Get latest version
            versions = self._client.search_model_versions(
                f"name='{self.ml_config.model_name}'"
            )
            latest = max(versions, key=lambda v: int(v.version))

            # Archive current Production
            prod_versions = [
                v for v in versions if v.current_stage == "Production"
            ]
            for v in prod_versions:
                self._client.transition_model_version_stage(
                    name=self.ml_config.model_name,
                    version=v.version,
                    stage="Archived",
                )

            # Promote new version
            self._client.transition_model_version_stage(
                name=self.ml_config.model_name,
                version=latest.version,
                stage="Production",
            )

            self._client.update_model_version(
                name=self.ml_config.model_name,
                version=latest.version,
                description=(
                    f"Promoted on {datetime.now().isoformat()} | "
                    f"ROC-AUC={metrics['roc_auc']:.3f} | "
                    f"AP={metrics['average_precision']:.3f} | "
                    f"F1-fraud={metrics['f1_fraud']:.3f}"
                ),
            )

            logger.info(
                "Model promoted to Production",
                extra={"version": latest.version, **metrics},
            )
            return True

        except Exception as e:
            raise MLflowError(f"Failed to promote model: {e}") from e

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self, lookback_days: int = 180) -> dict:
        """
        Run full training pipeline.

        Returns:
            Summary dict with run_id, metrics, and promotion status
        """
        logger.info("Starting model training pipeline")

        try:
            pdf = self.load_training_data(lookback_days=lookback_days)

            if len(pdf) < 1000:
                raise MLflowError(
                    f"Insufficient training data: {len(pdf)} rows (minimum 1000)"
                )

            run_id, metrics = self.train_with_mlflow(pdf)
            promoted = self.promote_to_production(run_id, metrics)

            summary = {
                "status": "success",
                "run_id": run_id,
                "promoted_to_production": promoted,
                **metrics,
            }
            logger.info("Training pipeline complete", extra=summary)
            return summary

        except MLflowError as e:
            logger.error("Training pipeline failed", extra={"error": str(e)})
            raise
