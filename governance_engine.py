
"""
AI Governance & Auditability Engine
Manages model approvals, prediction logging, fairness monitoring,
audit trails, and regulatory compliance workflows.
"""
import sqlite3, json, hashlib, time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

GOV_DB_PATH = Path('governance.db')

@dataclass
class GovernanceDecision:
    decision_id:    str
    model_name:     str
    model_version:  str
    applicant_id:   str
    prediction:     float
    decision:       str
    confidence:     float
    top_features:   dict
    explanation:    str
    fairness_group: str
    timestamp:      str
    latency_ms:     float
    reviewed:       bool = False


class GovernanceEngine:
    """
    Central governance registry for all model decisions.
    Implements RBI-grade audit trail and fairness monitoring.
    """

    REGULATORY_CHECKS = {
        'RBI_FPC': 'Fair Practices Code — Explanation required for all declinations',
        'RBI_KYC': 'Know Your Customer — fraud score must be logged',
        'IND_AS_109': 'Expected Credit Loss — ECL must be computed and logged',
        'PMLA': 'Anti-Money Laundering — suspicious patterns must be flagged',
        'DI_RULE': 'Disparate Impact — 80% rule monitoring required',
    }

    def __init__(self, db_path: Path = GOV_DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        conn = self._connect()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS decision_audit (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id     TEXT UNIQUE,
            model_name      TEXT,
            model_version   TEXT,
            applicant_id    TEXT,
            prediction      REAL,
            decision        TEXT,
            confidence      REAL,
            top_features    TEXT,
            explanation     TEXT,
            fairness_group  TEXT,
            timestamp       TEXT,
            latency_ms      REAL,
            reviewed        INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS model_approvals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT,
            version         TEXT,
            action          TEXT,
            approver        TEXT,
            rationale       TEXT,
            metrics_json    TEXT,
            approved_at     TEXT,
            effective_date  TEXT
        );
        CREATE TABLE IF NOT EXISTS fairness_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT,
            evaluation_date TEXT,
            group_col       TEXT,
            group_value     TEXT,
            approval_rate   REAL,
            avg_pd_score    REAL,
            di_ratio        REAL,
            di_pass         INTEGER,
            notes           TEXT
        );
        CREATE TABLE IF NOT EXISTS governance_alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id        TEXT,
            alert_type      TEXT,
            severity        TEXT,
            description     TEXT,
            triggered_at    TEXT,
            resolved        INTEGER DEFAULT 0,
            resolved_at     TEXT
        );
        """)
        conn.commit()
        conn.close()

    def log_decision(self, gov_decision: GovernanceDecision):
        conn = self._connect()
        conn.execute("""
        INSERT OR IGNORE INTO decision_audit
        (decision_id, model_name, model_version, applicant_id,
         prediction, decision, confidence, top_features, explanation,
         fairness_group, timestamp, latency_ms, reviewed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            gov_decision.decision_id, gov_decision.model_name,
            gov_decision.model_version, gov_decision.applicant_id,
            gov_decision.prediction, gov_decision.decision,
            gov_decision.confidence,
            json.dumps(gov_decision.top_features),
            gov_decision.explanation, gov_decision.fairness_group,
            gov_decision.timestamp, gov_decision.latency_ms,
            int(gov_decision.reviewed),
        ))
        conn.commit()
        conn.close()

    def log_model_approval(self, model_name: str, version: str,
                            action: str, approver: str,
                            rationale: str, metrics: dict):
        conn = self._connect()
        conn.execute("""
        INSERT INTO model_approvals
        (model_name, version, action, approver, rationale, metrics_json, approved_at, effective_date)
        VALUES (?,?,?,?,?,?,?,?)
        """, (
            model_name, version, action, approver, rationale,
            json.dumps(metrics),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

    def compute_fairness(self, df: pd.DataFrame,
                          group_col: str,
                          score_col: str = 'pd_score',
                          decision_col: str = 'decision',
                          model_name: str = 'PD_Model') -> pd.DataFrame:
        """
        Compute disparate impact (4/5ths rule) across demographic groups.
        Groups with approval_rate < 80% of highest group flag as CONCERN.
        """
        if group_col not in df.columns:
            return pd.DataFrame()
        results = []
        if decision_col in df.columns:
            group_stats = df.groupby(group_col).apply(lambda g: pd.Series({
                'approval_rate': (g[decision_col] == 'APPROVE').mean() if decision_col in g else np.nan,
                'avg_pd_score':  g[score_col].mean() if score_col in g else np.nan,
                'count':         len(g),
            })).reset_index()
        else:
            group_stats = df.groupby(group_col).apply(lambda g: pd.Series({
                'approval_rate': (g[score_col] < 0.30).mean() if score_col in g else np.nan,
                'avg_pd_score':  g[score_col].mean() if score_col in g else np.nan,
                'count':         len(g),
            })).reset_index()

        max_rate = group_stats['approval_rate'].max()
        group_stats['di_ratio'] = group_stats['approval_rate'] / max(max_rate, 0.001)
        group_stats['di_pass']  = group_stats['di_ratio'] >= 0.80
        group_stats['model_name'] = model_name
        group_stats['evaluation_date'] = datetime.now().isoformat()

        # Persist to DB
        conn = self._connect()
        for _, row in group_stats.iterrows():
            conn.execute("""
            INSERT INTO fairness_log
            (model_name, evaluation_date, group_col, group_value,
             approval_rate, avg_pd_score, di_ratio, di_pass)
            VALUES (?,?,?,?,?,?,?,?)
            """, (
                model_name, row['evaluation_date'], group_col, str(row[group_col]),
                row['approval_rate'], row['avg_pd_score'],
                row['di_ratio'], int(row['di_pass']),
            ))
        conn.commit()
        conn.close()
        return group_stats

    def raise_alert(self, alert_type: str, severity: str, description: str):
        alert_id = 'ALT' + hashlib.md5(f'{time.time()}'.encode()).hexdigest()[:8].upper()
        conn = self._connect()
        conn.execute("""
        INSERT INTO governance_alerts (alert_id, alert_type, severity, description, triggered_at)
        VALUES (?,?,?,?,?)
        """, (alert_id, alert_type, severity, description, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return alert_id

    def get_audit_summary(self, days: int = 30) -> dict:
        conn = self._connect()
        total   = conn.execute('SELECT COUNT(*) FROM decision_audit').fetchone()[0]
        by_dec  = dict(conn.execute(
            'SELECT decision, COUNT(*) FROM decision_audit GROUP BY decision'
        ).fetchall())
        avg_lat = conn.execute(
            'SELECT AVG(latency_ms) FROM decision_audit'
        ).fetchone()[0] or 0
        alerts  = conn.execute(
            'SELECT COUNT(*) FROM governance_alerts WHERE resolved=0'
        ).fetchone()[0]
        conn.close()
        return {
            'total_decisions': total,
            'by_decision': by_dec,
            'avg_latency_ms': round(avg_lat, 2),
            'open_alerts': alerts,
        }
