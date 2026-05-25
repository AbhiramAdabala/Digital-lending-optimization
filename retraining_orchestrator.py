
"""
Automated Retraining Orchestrator
Manages trigger-based, scheduled, and governance-approved retraining.
"""
import time, json, logging, hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

log = logging.getLogger('Retraining')


class TriggerType(str, Enum):
    SCHEDULED  = 'scheduled'
    DRIFT      = 'drift'
    PERFORMANCE = 'performance'
    FAIRNESS   = 'fairness'
    MANUAL     = 'manual'
    FRAUD_RATE = 'fraud_rate'


@dataclass
class RetrainingJob:
    job_id:      str
    model_name:  str
    trigger:     TriggerType
    trigger_reason: str
    requested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at:  Optional[str] = None
    completed_at: Optional[str] = None
    status:      str = 'QUEUED'  # QUEUED / RUNNING / PASSED / FAILED / BLOCKED
    metrics_before: dict = field(default_factory=dict)
    metrics_after:  dict = field(default_factory=dict)
    promoted:    bool = False
    approver:    Optional[str] = None
    error:       Optional[str] = None


@dataclass
class RetrainingTrigger:
    name:      str
    trigger_type: TriggerType
    condition: Callable
    priority:  int = 5   # 1=highest, 10=lowest
    cooldown_hours: int = 24   # Minimum hours between retraining


class RetrainingOrchestrator:
    """
    Manages the full automated retraining lifecycle:
    trigger detection → data validation → training → evaluation → promotion.
    """

    # Governance thresholds
    PERFORMANCE_THRESHOLDS = {
        'PD_Model':    {'min_auc': 0.75, 'max_brier': 0.20, 'max_psi': 0.20},
        'Fraud_Model': {'min_auc': 0.85, 'min_precision': 0.70, 'max_psi': 0.15},
        'EWS_Model':   {'min_auc': 0.72, 'min_recall': 0.65, 'max_psi': 0.20},
    }

    def __init__(self, log_dir: Path = Path('retraining_logs')):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.job_history: List[RetrainingJob] = []
        self.last_retrained: Dict[str, float] = {}
        self._setup_triggers()

    def _setup_triggers(self):
        """Define all retraining triggers."""
        self.triggers = [
            RetrainingTrigger(
                name='psi_critical_drift',
                trigger_type=TriggerType.DRIFT,
                condition=lambda ctx: ctx.get('max_psi', 0) >= 0.25,
                priority=1, cooldown_hours=6,
            ),
            RetrainingTrigger(
                name='auc_degradation',
                trigger_type=TriggerType.PERFORMANCE,
                condition=lambda ctx: ctx.get('current_auc', 1) < ctx.get('baseline_auc', 0.8) - 0.05,
                priority=2, cooldown_hours=12,
            ),
            RetrainingTrigger(
                name='fairness_violation',
                trigger_type=TriggerType.FAIRNESS,
                condition=lambda ctx: ctx.get('min_di_ratio', 1) < 0.80,
                priority=2, cooldown_hours=12,
            ),
            RetrainingTrigger(
                name='fraud_rate_spike',
                trigger_type=TriggerType.FRAUD_RATE,
                condition=lambda ctx: ctx.get('current_fraud_rate', 0) > ctx.get('baseline_fraud_rate', 0) * 1.30,
                priority=1, cooldown_hours=4,
            ),
            RetrainingTrigger(
                name='scheduled_monthly',
                trigger_type=TriggerType.SCHEDULED,
                condition=lambda ctx: ctx.get('days_since_last_retrain', 0) >= 30,
                priority=8, cooldown_hours=25 * 24,
            ),
            RetrainingTrigger(
                name='psi_alert_drift',
                trigger_type=TriggerType.DRIFT,
                condition=lambda ctx: ctx.get('max_psi', 0) >= 0.20,
                priority=3, cooldown_hours=12,
            ),
        ]

    def evaluate_triggers(self, model_name: str, context: dict) -> Optional[RetrainingTrigger]:
        """
        Evaluate all triggers. Return the highest-priority trigger that fires.
        Respects cooldown periods.
        """
        last_time = self.last_retrained.get(model_name, 0)
        fired = []
        for trigger in sorted(self.triggers, key=lambda t: t.priority):
            hours_since = (time.time() - last_time) / 3600
            if hours_since < trigger.cooldown_hours:
                continue
            try:
                if trigger.condition(context):
                    fired.append(trigger)
            except Exception:
                pass
        return fired[0] if fired else None

    def create_job(self, model_name: str, trigger: RetrainingTrigger,
                   context: dict) -> RetrainingJob:
        job_id = 'RTJ' + hashlib.md5(f'{model_name}{time.time()}'.encode()).hexdigest()[:8].upper()
        reason = f'{trigger.trigger_type}: {trigger.name}'
        if trigger.trigger_type == TriggerType.DRIFT:
            reason += f' | PSI={context.get("max_psi", 0):.3f}'
        elif trigger.trigger_type == TriggerType.PERFORMANCE:
            reason += f' | AUC={context.get("current_auc", 0):.3f} (baseline={context.get("baseline_auc", 0):.3f})'
        job = RetrainingJob(
            job_id=job_id,
            model_name=model_name,
            trigger=trigger.trigger_type,
            trigger_reason=reason,
        )
        self.job_history.append(job)
        return job

    def run_retraining_simulation(self, job: RetrainingJob,
                                    df: pd.DataFrame) -> RetrainingJob:
        """
        Simulate the retraining pipeline (replace with actual training call in production).
        """
        job.status = 'RUNNING'
        job.started_at = datetime.now().isoformat()
        log.info('Starting retraining job %s for %s', job.job_id, job.model_name)

        # Simulate pipeline steps
        pipeline_steps = [
            'data_validation',
            'feature_engineering',
            'model_training',
            'evaluation',
            'governance_gate',
        ]
        step_results = {}
        for step in pipeline_steps:
            time.sleep(0.01)  # Simulate work
            step_results[step] = 'PASSED'

        # Simulate metrics
        thresholds = self.PERFORMANCE_THRESHOLDS.get(job.model_name, {'min_auc': 0.70})
        before_auc = np.random.uniform(0.72, 0.78)
        after_auc  = np.random.uniform(0.77, 0.85)

        job.metrics_before = {'auc_roc': round(before_auc, 4), 'brier': 0.18}
        job.metrics_after  = {'auc_roc': round(after_auc,  4), 'brier': 0.14}

        governance_passed = after_auc >= thresholds.get('min_auc', 0.70)
        job.status       = 'PASSED' if governance_passed else 'FAILED'
        job.completed_at = datetime.now().isoformat()

        self.last_retrained[job.model_name] = time.time()
        log.info('Retraining job %s completed: %s | AUC %.3f→%.3f',
                  job.job_id, job.status, before_auc, after_auc)

        # Save job record
        job_record = {
            'job_id': job.job_id, 'model_name': job.model_name,
            'trigger': job.trigger, 'reason': job.trigger_reason,
            'status': job.status, 'metrics_before': job.metrics_before,
            'metrics_after': job.metrics_after,
            'requested_at': job.requested_at, 'completed_at': job.completed_at,
        }
        log_file = self.log_dir / f'{job.job_id}.json'
        with open(log_file, 'w') as f:
            json.dump(job_record, f, indent=2)

        return job

    def get_job_summary(self) -> pd.DataFrame:
        if not self.job_history:
            return pd.DataFrame()
        records = [{
            'job_id': j.job_id, 'model': j.model_name,
            'trigger': j.trigger, 'status': j.status,
            'auc_before': j.metrics_before.get('auc_roc', np.nan),
            'auc_after':  j.metrics_after.get('auc_roc', np.nan),
            'requested_at': j.requested_at,
        } for j in self.job_history]
        return pd.DataFrame(records)
