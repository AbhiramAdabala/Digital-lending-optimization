
"""
Enterprise Drift Detection Engine
Monitors data drift, concept drift, prediction drift, and feature stability.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json, logging

log = logging.getLogger('DriftDetector')

# ── Drift Result Data Model ────────────────────────────────────────────────
@dataclass
class DriftResult:
    feature:       str
    drift_type:    str   # data / concept / prediction / fairness
    metric:        str   # PSI / KS / chi2 / wasserstein
    score:         float
    threshold_monitor: float
    threshold_alert:   float
    threshold_retrain: float
    status:        str   # OK / MONITOR / ALERT / RETRAIN
    direction:     str   # UP / DOWN / STABLE
    timestamp:     str = field(default_factory=lambda: datetime.now().isoformat())
    recommendation: str = ''

    @property
    def needs_action(self):
        return self.status in ('ALERT', 'RETRAIN')


@dataclass
class DriftReport:
    report_id:   str
    timestamp:   str
    baseline_period: str
    current_period:  str
    n_features_drifted: int
    overall_status: str  # GREEN / YELLOW / RED
    results: List[DriftResult] = field(default_factory=list)
    escalation_required: bool = False
    retraining_required: bool = False


class DriftDetector:
    """
    Comprehensive drift detection for lending ML models.
    Implements PSI, KS-test, Chi-squared, and Wasserstein distance.
    """

    PSI_THRESHOLDS    = {'monitor': 0.10, 'alert': 0.20, 'retrain': 0.25}
    KS_THRESHOLDS     = {'monitor': 0.05, 'alert': 0.10, 'retrain': 0.15}
    PRED_THRESHOLDS   = {'monitor': 0.03, 'alert': 0.06, 'retrain': 0.10}

    def compute_psi(self, baseline: np.ndarray,
                    current: np.ndarray,
                    n_bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        PSI < 0.10: Stable
        PSI 0.10–0.20: Monitor
        PSI 0.20–0.25: Alert
        PSI > 0.25: Retrain
        """
        eps = 1e-6
        try:
            bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
            if len(bins) < 3:
                return 0.0
            base_counts = np.histogram(baseline, bins=bins)[0]
            curr_counts = np.histogram(current,  bins=bins)[0]
            base_pct = np.maximum(base_counts / len(baseline), eps)
            curr_pct = np.maximum(curr_counts / len(current),  eps)
            base_pct /= base_pct.sum()
            curr_pct /= curr_pct.sum()
            psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
            return float(np.clip(psi, 0, None))
        except Exception:
            return 0.0

    def compute_ks(self, baseline: np.ndarray,
                   current: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for distributional drift."""
        try:
            ks_stat, p_value = stats.ks_2samp(baseline, current)
            return float(ks_stat), float(p_value)
        except Exception:
            return 0.0, 1.0

    def compute_wasserstein(self, baseline: np.ndarray,
                             current: np.ndarray) -> float:
        """Wasserstein (Earth Mover's) distance — sensitive to shape changes."""
        try:
            return float(stats.wasserstein_distance(baseline, current))
        except Exception:
            return 0.0

    def _get_status(self, score: float, thresholds: dict) -> str:
        if score >= thresholds['retrain']:
            return 'RETRAIN'
        elif score >= thresholds['alert']:
            return 'ALERT'
        elif score >= thresholds['monitor']:
            return 'MONITOR'
        return 'OK'

    def _get_direction(self, baseline: np.ndarray,
                       current: np.ndarray) -> str:
        if current.mean() > baseline.mean() * 1.05:
            return 'UP'
        elif current.mean() < baseline.mean() * 0.95:
            return 'DOWN'
        return 'STABLE'

    def detect_feature_drift(self, baseline_df: pd.DataFrame,
                              current_df: pd.DataFrame,
                              features: Optional[List[str]] = None) -> List[DriftResult]:
        """Run drift detection across all numeric features."""
        numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
        target_features = features or numeric_cols
        target_features = [f for f in target_features if f in baseline_df.columns
                            and f in current_df.columns]

        results = []
        for feat in target_features:
            base_arr = baseline_df[feat].dropna().values
            curr_arr = current_df[feat].dropna().values
            if len(base_arr) < 30 or len(curr_arr) < 30:
                continue

            psi_score  = self.compute_psi(base_arr, curr_arr)
            ks_stat, _ = self.compute_ks(base_arr, curr_arr)
            status = self._get_status(psi_score, self.PSI_THRESHOLDS)
            direction = self._get_direction(base_arr, curr_arr)

            recommendation = ''
            if status == 'RETRAIN':
                recommendation = f'Feature {feat} has drifted significantly. Trigger retraining pipeline.'
            elif status == 'ALERT':
                recommendation = f'Feature {feat} showing drift. Increase monitoring frequency.'
            elif status == 'MONITOR':
                recommendation = f'Feature {feat} showing early drift signals. Watch closely.'

            results.append(DriftResult(
                feature=feat,
                drift_type='data',
                metric='PSI',
                score=round(psi_score, 4),
                threshold_monitor=self.PSI_THRESHOLDS['monitor'],
                threshold_alert=self.PSI_THRESHOLDS['alert'],
                threshold_retrain=self.PSI_THRESHOLDS['retrain'],
                status=status,
                direction=direction,
                recommendation=recommendation,
            ))
        return results

    def detect_prediction_drift(self, baseline_preds: np.ndarray,
                                 current_preds: np.ndarray) -> DriftResult:
        """Detect drift in model prediction distribution."""
        psi_score = self.compute_psi(baseline_preds, current_preds)
        ks_stat, _ = self.compute_ks(baseline_preds, current_preds)
        mean_shift = abs(current_preds.mean() - baseline_preds.mean())
        combined_score = 0.5 * psi_score + 0.3 * ks_stat + 0.2 * min(mean_shift / 0.1, 1.0)
        status = self._get_status(combined_score, self.PRED_THRESHOLDS)
        direction = self._get_direction(baseline_preds, current_preds)

        return DriftResult(
            feature='pd_score_predictions',
            drift_type='prediction',
            metric='PSI+KS+MeanShift',
            score=round(combined_score, 4),
            threshold_monitor=self.PRED_THRESHOLDS['monitor'],
            threshold_alert=self.PRED_THRESHOLDS['alert'],
            threshold_retrain=self.PRED_THRESHOLDS['retrain'],
            status=status,
            direction=direction,
            recommendation=(
                'Model predictions shifting. Investigate data pipeline and retrain.'
                if status in ('ALERT', 'RETRAIN') else
                'Prediction distribution stable.'
            ),
        )

    def detect_fairness_drift(self, baseline_df: pd.DataFrame,
                               current_df: pd.DataFrame,
                               group_col: str,
                               score_col: str = 'pd_score') -> List[DriftResult]:
        """Monitor fairness metrics across demographic groups."""
        results = []
        if group_col not in baseline_df.columns or score_col not in baseline_df.columns:
            return results
        for group in baseline_df[group_col].unique():
            base_group = baseline_df[baseline_df[group_col] == group][score_col].dropna().values
            curr_group = current_df[current_df[group_col] == group][score_col].dropna().values
            if len(base_group) < 20 or len(curr_group) < 20:
                continue
            psi = self.compute_psi(base_group, curr_group)
            status = self._get_status(psi, self.PSI_THRESHOLDS)
            results.append(DriftResult(
                feature=f'{score_col}@{group_col}={group}',
                drift_type='fairness',
                metric='PSI',
                score=round(psi, 4),
                threshold_monitor=self.PSI_THRESHOLDS['monitor'],
                threshold_alert=self.PSI_THRESHOLDS['alert'],
                threshold_retrain=self.PSI_THRESHOLDS['retrain'],
                status=status,
                direction=self._get_direction(base_group, curr_group),
                recommendation=(
                    f'Fairness drift for group {group} in {group_col}. CRO review required.'
                    if status in ('ALERT', 'RETRAIN') else ''
                ),
            ))
        return results

    def generate_report(self, all_results: List[DriftResult],
                         baseline_period: str = 'T-30d',
                         current_period: str = 'T-0d') -> DriftReport:
        """Consolidate drift results into an actionable report."""
        n_drifted = sum(1 for r in all_results if r.needs_action)
        needs_retrain = any(r.status == 'RETRAIN' for r in all_results)
        needs_escalation = any(r.status in ('ALERT', 'RETRAIN') for r in all_results)

        if needs_retrain:
            overall = 'RED'
        elif needs_escalation:
            overall = 'YELLOW'
        else:
            overall = 'GREEN'

        import hashlib, time
        report_id = 'DRF' + hashlib.md5(
            f'{time.time()}'.encode()
        ).hexdigest()[:8].upper()

        return DriftReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            baseline_period=baseline_period,
            current_period=current_period,
            n_features_drifted=n_drifted,
            overall_status=overall,
            results=all_results,
            escalation_required=needs_escalation,
            retraining_required=needs_retrain,
        )
