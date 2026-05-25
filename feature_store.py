
"""
Enterprise Feature Store — Digital Lending Platform
Supports offline (batch) and online (real-time) feature serving.
"""
import sqlite3, json, hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

FEATURE_STORE_PATH = Path("feature_store.db")

# ── Feature Registry ────────────────────────────────────────────────────────
FEATURE_REGISTRY = {
    "credit_features": [
        "credit_score", "credit_history_length", "missed_payment_ratio",
        "avg_delay_days", "worst_delinquency_stage", "existing_loans",
    ],
    "financial_features": [
        "monthly_income", "debt_to_income_ratio", "financial_stress_index",
        "income_stability_score", "bank_balance_avg",
    ],
    "behavioral_features": [
        "spending_volatility_index", "digital_engagement_score",
        "app_usage_mean", "digital_activity_score",
    ],
    "fraud_features": [
        "income_inflation_ratio", "document_risk_proxy",
        "identity_consistency", "synthetic_id_probability",
    ],
    "loan_features": [
        "loan_amount", "loan_tenure_months", "loan_purpose",
        "acquisition_channel",
    ],
    "derived_features": [
        "pd_score", "risk_grade", "expected_loss",
        "fraud_risk_score", "health_score",
    ],
}

FEATURE_METADATA = {
    "credit_score":               {"type": "integer", "range": [300, 900], "importance": "HIGH", "sensitivity": "PII"},
    "missed_payment_ratio":       {"type": "float",   "range": [0, 1],     "importance": "HIGH", "sensitivity": "PII"},
    "monthly_income":             {"type": "float",   "range": [0, None],  "importance": "HIGH", "sensitivity": "PII"},
    "debt_to_income_ratio":       {"type": "float",   "range": [0, None],  "importance": "HIGH", "sensitivity": None},
    "financial_stress_index":     {"type": "float",   "range": [0, 1],     "importance": "HIGH", "sensitivity": None},
    "spending_volatility_index":  {"type": "float",   "range": [0, 1],     "importance": "MED",  "sensitivity": None},
    "digital_engagement_score":   {"type": "float",   "range": [0, 100],   "importance": "MED",  "sensitivity": None},
    "worst_delinquency_stage":    {"type": "integer", "range": [0, 4],     "importance": "HIGH", "sensitivity": "PII"},
    "income_stability_score":     {"type": "float",   "range": [0, 1],     "importance": "MED",  "sensitivity": None},
    "loan_amount":                {"type": "float",   "range": [0, None],  "importance": "MED",  "sensitivity": None},
}


class OfflineFeatureStore:
    """
    Batch feature store backed by SQLite.
    In production: replace with Hive/BigQuery/Feast.
    """
    def __init__(self, db_path: Path = FEATURE_STORE_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        conn = self._connect()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feature_vectors (
            entity_id    TEXT NOT NULL,
            feature_group TEXT NOT NULL,
            feature_name  TEXT NOT NULL,
            feature_value REAL,
            version       TEXT,
            created_at    TEXT,
            PRIMARY KEY (entity_id, feature_group, feature_name, version)
        );
        CREATE TABLE IF NOT EXISTS feature_versions (
            version       TEXT PRIMARY KEY,
            description   TEXT,
            created_at    TEXT,
            author        TEXT,
            is_active     INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS feature_stats (
            feature_name  TEXT,
            version       TEXT,
            mean          REAL, std REAL, min_val REAL, max_val REAL,
            p25 REAL, p50 REAL, p75 REAL, null_rate REAL,
            computed_at   TEXT,
            PRIMARY KEY (feature_name, version)
        );
        """)
        conn.commit()
        conn.close()

    def write_features(self, df: pd.DataFrame, feature_group: str,
                       entity_col: str = 'loan_id',
                       version: str = 'v1') -> int:
        """Write a feature group to the offline store."""
        conn = self._connect()
        records = []
        ts = datetime.now().isoformat()
        feature_cols = [c for c in df.columns if c != entity_col]
        for _, row in df.iterrows():
            for feat in feature_cols:
                val = row[feat]
                records.append((
                    str(row[entity_col]), feature_group, feat,
                    float(val) if pd.notna(val) and isinstance(val, (int, float)) else None,
                    version, ts
                ))
        conn.executemany("""
        INSERT OR REPLACE INTO feature_vectors
        (entity_id, feature_group, feature_name, feature_value, version, created_at)
        VALUES (?,?,?,?,?,?)
        """, records)
        conn.commit()
        conn.close()
        return len(records)

    def read_features(self, entity_ids: List[str],
                      feature_group: str, version: str = 'v1') -> pd.DataFrame:
        """Retrieve features for a list of entities."""
        conn = self._connect()
        placeholders = ','.join('?' * len(entity_ids))
        rows = conn.execute(f"""
        SELECT entity_id, feature_name, feature_value
        FROM feature_vectors
        WHERE entity_id IN ({placeholders})
          AND feature_group = ?
          AND version = ?
        """, (*entity_ids, feature_group, version)).fetchall()
        conn.close()
        if not rows:
            return pd.DataFrame()
        records = [dict(r) for r in rows]
        df = pd.DataFrame(records).pivot(
            index='entity_id', columns='feature_name', values='feature_value'
        ).reset_index()
        return df

    def compute_stats(self, df: pd.DataFrame, version: str = 'v1') -> pd.DataFrame:
        """Compute and store feature statistics for monitoring."""
        stats_records = []
        ts = datetime.now().isoformat()
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            stats_records.append({
                'feature_name': col,
                'version': version,
                'mean':  float(s.mean()),
                'std':   float(s.std()),
                'min_val': float(s.min()),
                'max_val': float(s.max()),
                'p25': float(s.quantile(0.25)),
                'p50': float(s.quantile(0.50)),
                'p75': float(s.quantile(0.75)),
                'null_rate': float(df[col].isna().mean()),
                'computed_at': ts,
            })
        stats_df = pd.DataFrame(stats_records)
        conn = self._connect()
        for rec in stats_records:
            conn.execute("""
            INSERT OR REPLACE INTO feature_stats
            (feature_name, version, mean, std, min_val, max_val,
             p25, p50, p75, null_rate, computed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                rec['feature_name'], rec['version'],
                rec['mean'], rec['std'], rec['min_val'], rec['max_val'],
                rec['p25'], rec['p50'], rec['p75'],
                rec['null_rate'], rec['computed_at'],
            ))
        conn.commit()
        conn.close()
        return stats_df


class OnlineFeatureStore:
    """
    In-memory online feature cache (simulates Redis/DynamoDB in production).
    Provides sub-millisecond feature retrieval for real-time inference.
    """
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._ttl: Dict[str, float] = {}
        self.ttl_seconds = 3600

    def set(self, entity_id: str, features: dict):
        self._cache[entity_id] = features
        self._ttl[entity_id] = time.time() + self.ttl_seconds

    def get(self, entity_id: str) -> Optional[dict]:
        if entity_id in self._cache:
            if time.time() < self._ttl.get(entity_id, 0):
                return self._cache[entity_id]
            else:
                del self._cache[entity_id]
        return None

    def bulk_set(self, entity_df: pd.DataFrame, entity_col: str = 'loan_id'):
        for _, row in entity_df.iterrows():
            self.set(str(row[entity_col]), row.to_dict())
        return len(entity_df)

    def stats(self) -> dict:
        return {'cached_entities': len(self._cache), 'ttl_seconds': self.ttl_seconds}
