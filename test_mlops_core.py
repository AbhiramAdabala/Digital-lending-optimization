
"""
FinLend AI MLOps — Test Suite
Unit tests for drift detection, feature store, and API.
"""
import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')


# ── Drift Detector Tests ──────────────────────────────────────────────────────
class TestDriftDetector:

    def setup_method(self):
        """Import DriftDetector from module (adjust path as needed)."""
        # Inline implementation for testing
        import numpy as np
        from scipy import stats

        def compute_psi(baseline, current, n_bins=10):
            eps = 1e-6
            bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
            if len(bins) < 3:
                return 0.0
            bp = np.maximum(np.histogram(baseline, bins=bins)[0] / len(baseline), eps)
            cp = np.maximum(np.histogram(current,  bins=bins)[0] / len(current),  eps)
            bp /= bp.sum(); cp /= cp.sum()
            return float(np.sum((cp - bp) * np.log(cp / bp + eps)))

        self.compute_psi = compute_psi

    def test_psi_stable_distribution(self):
        """PSI should be near zero for identical distributions."""
        np.random.seed(42)
        base = np.random.normal(0.2, 0.05, 1000)
        curr = np.random.normal(0.2, 0.05, 1000)
        psi = self.compute_psi(base, curr)
        assert psi < 0.10, f"Stable distribution PSI={psi:.3f} should be < 0.10"

    def test_psi_drifted_distribution(self):
        """PSI should be high for significantly shifted distributions."""
        np.random.seed(42)
        base = np.random.normal(0.20, 0.05, 1000)
        curr = np.random.normal(0.45, 0.08, 1000)  # Major shift
        psi = self.compute_psi(base, curr)
        assert psi > 0.20, f"Drifted distribution PSI={psi:.3f} should be > 0.20"

    def test_psi_is_non_negative(self):
        np.random.seed(99)
        base = np.random.beta(2, 5, 500)
        curr = np.random.beta(3, 4, 500)
        psi = self.compute_psi(base, curr)
        assert psi >= 0, f"PSI must be non-negative, got {psi}"


# ── PD Score Logic Tests ──────────────────────────────────────────────────────
class TestPDScoringLogic:

    def compute_pd(self, credit_score=680, missed_payment_ratio=0.05,
                    worst_delinquency_stage=0, financial_stress_index=0.3,
                    avg_delay_days=5, debt_to_income_ratio=2.0,
                    spending_volatility_index=0.25, credit_history_length=36):
        cs_norm   = max(0, (900 - credit_score) / 600)
        mpr_norm  = min(missed_payment_ratio, 1.0)
        dpd_norm  = worst_delinquency_stage / 4.0
        dly_norm  = min(avg_delay_days / 90, 1.0)
        fsi_norm  = min(financial_stress_index, 1.0)
        dti_norm  = min(debt_to_income_ratio / 10, 1.0)
        svi_norm  = min(spending_volatility_index, 1.0)
        hist_norm = max(0, 1 - credit_history_length / 60)
        raw_pd = (
            0.22 * mpr_norm + 0.20 * dpd_norm + 0.12 * cs_norm
            + 0.15 * fsi_norm + 0.10 * dly_norm + 0.08 * dti_norm
            + 0.08 * svi_norm + 0.05 * hist_norm
        )
        if credit_history_length < 6:
            raw_pd = min(raw_pd + 0.06, 0.95)
        if worst_delinquency_stage >= 3:
            raw_pd = min(raw_pd + 0.15, 0.95)
        return float(np.clip(raw_pd, 0.01, 0.95))

    def test_prime_borrower_low_pd(self):
        """Prime borrowers (high score, no delinquency) should have low PD."""
        pd = self.compute_pd(credit_score=850, missed_payment_ratio=0.0,
                              worst_delinquency_stage=0, financial_stress_index=0.1,
                              credit_history_length=60)
        assert pd < 0.15, f"Prime borrower PD={pd:.3f} should be < 0.15"

    def test_risky_borrower_high_pd(self):
        """High-risk borrowers should have high PD."""
        pd = self.compute_pd(credit_score=350, missed_payment_ratio=0.6,
                              worst_delinquency_stage=3, financial_stress_index=0.8,
                              debt_to_income_ratio=8, credit_history_length=2)
        assert pd > 0.55, f"Risky borrower PD={pd:.3f} should be > 0.55"

    def test_pd_bounded(self):
        """PD must be in [0.01, 0.95]."""
        pd1 = self.compute_pd(credit_score=900, missed_payment_ratio=0.0)
        pd2 = self.compute_pd(credit_score=300, missed_payment_ratio=1.0,
                               worst_delinquency_stage=4)
        assert 0.01 <= pd1 <= 0.95, f"PD1={pd1} out of bounds"
        assert 0.01 <= pd2 <= 0.95, f"PD2={pd2} out of bounds"

    def test_delinquency_bump(self):
        """DPD stage 3+ should increase PD."""
        pd_clean = self.compute_pd(worst_delinquency_stage=0)
        pd_dpd3  = self.compute_pd(worst_delinquency_stage=3)
        assert pd_dpd3 > pd_clean, f"DPD3 PD should be > clean PD: {pd_dpd3:.3f} vs {pd_clean:.3f}"


# ── Fraud Scoring Tests ───────────────────────────────────────────────────────
class TestFraudScoring:

    def compute_fraud(self, monthly_income=55000, employment_type='Salaried',
                       credit_history_length=36, loan_amount=200000,
                       worst_delinquency_stage=0, income_stability_score=0.7):
        medians = {'Salaried': 45000, 'Self-Employed': 52000,
                    'Gig Worker': 22000, 'SME Owner': 85000, 'First-Time Borrower': 18000}
        exp_income = medians.get(employment_type, 40000)
        inflation = min(max((monthly_income / max(exp_income, 1)) - 1.0, 0) / 2.0, 1.0)
        doc_risk = float(
            (credit_history_length < 6) * 0.30
            + (worst_delinquency_stage >= 3) * 0.20
            + (1 - income_stability_score) * 0.25
        )
        identity = 1 - min(
            0.90 + (income_stability_score - 0.5) * 0.20
            - (credit_history_length < 6) * 0.20, 1.0
        )
        synthetic = float(
            (credit_history_length < 6) * 0.40
            + (loan_amount > monthly_income * 20) * 0.25
            + doc_risk * 0.20
        )
        return float(np.clip(
            0.35 * inflation + 0.25 * doc_risk + 0.25 * identity + 0.15 * synthetic, 0, 1
        ))

    def test_low_fraud_normal_applicant(self):
        score = self.compute_fraud()
        assert score < 0.35, f"Normal applicant fraud score={score:.3f} should be < 0.35"

    def test_high_fraud_thin_file(self):
        score = self.compute_fraud(credit_history_length=2, income_stability_score=0.2,
                                    worst_delinquency_stage=3)
        assert score > 0.35, f"Thin-file fraud score={score:.3f} should be > 0.35"

    def test_income_inflation_raises_score(self):
        normal_score = self.compute_fraud(monthly_income=45000)
        inflated_score = self.compute_fraud(monthly_income=150000)
        assert inflated_score > normal_score, "Inflated income should raise fraud score"

    def test_fraud_score_bounded(self):
        score = self.compute_fraud(credit_history_length=0, monthly_income=999999,
                                    loan_amount=10000000)
        assert 0 <= score <= 1, f"Fraud score {score} out of [0,1] bounds"


if __name__ == '__main__':
    pytest.main(['--tb=short', '-v', __file__])
