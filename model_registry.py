
"""
Model Registry & Automated Training Pipeline
Uses MLflow for experiment tracking and model versioning.
"""
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import json, os, hashlib, time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ── Model Governance Metadata ────────────────────────────────────────────────
MODEL_CARDS = {
    'PD_Model': {
        'purpose': 'Predict probability of default for loan applicants',
        'features': 8,
        'training_data': 'lending_features/master_feature_table.csv',
        'fairness_groups': ['gender', 'employment_type', 'city'],
        'performance_thresholds': {'min_auc': 0.75, 'max_brier': 0.20},
        'retraining_triggers': ['psi > 0.20', 'auc_drop > 0.05', 'monthly'],
        'regulatory_requirement': 'RBI Master Direction on Credit',
        'explainability': 'SHAP values per decision',
        'approver': 'CRO',
    },
    'Fraud_Model': {
        'purpose': 'Detect fraudulent loan applications and synthetic identities',
        'features': 12,
        'training_data': 'fraud_detection/data/fraud_labeled.csv',
        'fairness_groups': ['employment_type'],
        'performance_thresholds': {'min_auc': 0.85, 'min_precision_red': 0.70},
        'retraining_triggers': ['fraud_rate_change > 20%', 'psi > 0.15', 'weekly'],
        'regulatory_requirement': 'PMLA Anti-Money Laundering',
        'explainability': 'Rule-based + SHAP',
        'approver': 'Fraud Head',
    },
    'EWS_Model': {
        'purpose': 'Early warning detection for loan delinquency',
        'features': 10,
        'training_data': 'early_warning/data/ews_labeled.csv',
        'fairness_groups': ['employment_type'],
        'performance_thresholds': {'min_auc': 0.72, 'min_recall': 0.65},
        'retraining_triggers': ['psi > 0.20', 'recall_drop > 0.08', 'monthly'],
        'regulatory_requirement': 'RBI Prudential Norms on NPA',
        'explainability': 'SHAP + counterfactuals',
        'approver': 'CRO',
    },
}


# ── MLflow Registry Manager ──────────────────────────────────────────────────
class ModelRegistryManager:
    def __init__(self, tracking_uri: str = './mlruns',
                 registry_uri: Optional[str] = None):
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        self.client = MlflowClient()

    def get_or_create_experiment(self, name: str, tags: dict = None) -> str:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name, tags=tags or {})
        else:
            exp_id = exp.experiment_id
        return exp_id

    def register_model(self, run_id: str, model_name: str,
                       metrics: dict, tags: dict = None) -> str:
        """Register a trained model in the MLflow Model Registry."""
        model_uri = f'runs:/{run_id}/model'
        try:
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            # Add governance tags
            self.client.set_model_version_tag(
                model_name, version, 'governance_status', 'PENDING_REVIEW'
            )
            self.client.set_model_version_tag(
                model_name, version, 'registered_at', datetime.now().isoformat()
            )
            for k, v in (tags or {}).items():
                self.client.set_model_version_tag(model_name, version, k, str(v))
            return version
        except Exception as e:
            return f'REGISTRY_ERROR: {e}'

    def promote_to_staging(self, model_name: str, version: str):
        self.client.transition_model_version_stage(
            model_name, version, 'Staging',
            archive_existing_versions=False
        )
        self.client.set_model_version_tag(
            model_name, version, 'governance_status', 'STAGING'
        )

    def promote_to_production(self, model_name: str, version: str,
                               approver: str = 'CRO'):
        self.client.transition_model_version_stage(
            model_name, version, 'Production',
            archive_existing_versions=True
        )
        self.client.set_model_version_tag(
            model_name, version, 'governance_status', 'APPROVED'
        )
        self.client.set_model_version_tag(
            model_name, version, 'approved_by', approver
        )
        self.client.set_model_version_tag(
            model_name, version, 'approved_at', datetime.now().isoformat()
        )

    def rollback(self, model_name: str, target_version: str):
        """Emergency rollback to a specified version."""
        self.client.transition_model_version_stage(
            model_name, target_version, 'Production',
            archive_existing_versions=True
        )
        self.client.set_model_version_tag(
            model_name, target_version, 'governance_status', 'ROLLBACK'
        )
        self.client.set_model_version_tag(
            model_name, target_version, 'rollback_at', datetime.now().isoformat()
        )

    def get_production_model(self, model_name: str):
        try:
            versions = self.client.get_latest_versions(
                model_name, stages=['Production']
            )
            return versions[0] if versions else None
        except Exception:
            return None


# ── Training Pipeline ────────────────────────────────────────────────────────
class LendingModelTrainer:
    """
    Automated training pipeline for all lending models.
    Supports experiment tracking, validation gates, and governance logging.
    """

    # Rule-based feature weights for fallback PD model
    RULE_WEIGHTS = {
        'missed_payment_ratio':     0.22,
        'worst_delinquency_stage':  0.20,
        'credit_score_norm':        0.12,
        'financial_stress_index':   0.15,
        'avg_delay_norm':           0.10,
        'debt_to_income_norm':      0.08,
        'spending_volatility_index':0.08,
        'history_norm':             0.05,
    }

    def __init__(self, registry_manager: ModelRegistryManager):
        self.registry = registry_manager

    def prepare_pd_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for PD model training."""
        feature_cols = [
            'credit_score', 'missed_payment_ratio',
            'worst_delinquency_stage', 'financial_stress_index',
            'debt_to_income_ratio', 'spending_volatility_index',
            'income_stability_score', 'pd_score',
        ]
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(df[available].median())
        y = df['default_flag'].fillna(0).astype(int) if 'default_flag' in df.columns             else (df.get('pd_score', pd.Series(np.random.beta(2, 8, len(df)))) > 0.3).astype(int)
        return X, y

    def train_pd_model(self, df: pd.DataFrame,
                       experiment_name: str = 'PD_Model_Training') -> dict:
        """Train and register a PD model."""
        exp_id = self.registry.get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        X, y = self.prepare_pd_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Choose best available algorithm
        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                random_state=42, verbosity=-1,
                class_weight='balanced',
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=42,
            )

        with mlflow.start_run(run_name=f'PD_Train_{datetime.now().strftime("%Y%m%d_%H%M%S")}') as run:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > 0.30).astype(int)

            metrics = {
                'auc_roc':  float(roc_auc_score(y_test, y_prob)),
                'auc_pr':   float(average_precision_score(y_test, y_prob)),
                'brier':    float(brier_score_loss(y_test, y_prob)),
                'precision':float(precision_score(y_test, y_pred, zero_division=0)),
                'recall':   float(recall_score(y_test, y_pred, zero_division=0)),
                'f1':       float(f1_score(y_test, y_pred, zero_division=0)),
                'n_train':  len(X_train),
                'n_test':   len(X_test),
                'default_rate': float(y.mean()),
            }

            mlflow.log_metrics(metrics)
            mlflow.log_params({
                'algorithm': type(model).__name__,
                'n_features': X.shape[1],
                'feature_names': ','.join(X.columns.tolist()),
            })
            mlflow.log_dict({'model_card': MODEL_CARDS.get('PD_Model', {})}, 'model_card.json')

            if HAS_LGB:
                mlflow.lightgbm.log_model(model, 'model')
            else:
                mlflow.sklearn.log_model(model, 'model')

            run_id = run.info.run_id

        # Governance gate
        passed = (
            metrics['auc_roc'] >= MODEL_CARDS['PD_Model']['performance_thresholds']['min_auc']
            and metrics['brier'] <= MODEL_CARDS['PD_Model']['performance_thresholds']['max_brier']
        )

        return {
            'run_id':  run_id,
            'metrics': metrics,
            'model':   model,
            'features': X.columns.tolist(),
            'governance_passed': passed,
            'gate_reason': 'AUC and Brier score within thresholds' if passed
                           else f'FAILED: AUC={metrics["auc_roc"]:.3f}',
        }

    def validate_data(self, df: pd.DataFrame) -> dict:
        """Data validation gate before training."""
        checks = {}
        checks['min_rows']     = len(df) >= 1000
        checks['has_target']   = 'default_flag' in df.columns or 'pd_score' in df.columns
        checks['low_nulls']    = df.isnull().mean().max() < 0.30
        checks['no_constants'] = all(
            df[c].nunique() > 1
            for c in df.select_dtypes(include=[np.number]).columns
        )
        checks['passed'] = all(checks[k] for k in checks if k != 'passed')
        return checks
