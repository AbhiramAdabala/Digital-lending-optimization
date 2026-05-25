
"""
FinLend AI — Production Model Serving API
FastAPI application exposing underwriting, pricing, fraud, and EWS models.

Run: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import numpy as np
import time, json, logging, hashlib, os
from datetime import datetime
from pathlib import Path

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinLend AI — Model Serving API",
    description="Production inference layer for digital lending ML models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("finlend_api")

# ── Request / Response Models ────────────────────────────────────────────────
class UnderwritingRequest(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier")
    monthly_income: float = Field(..., gt=0, description="Monthly income in INR")
    employment_type: str = Field(..., description="Employment category")
    credit_score: int = Field(..., ge=300, le=900)
    credit_history_length: int = Field(default=36, ge=0)
    loan_amount: float = Field(..., gt=0)
    loan_tenure_months: int = Field(..., ge=6, le=84)
    missed_payment_ratio: float = Field(default=0.05, ge=0, le=1)
    avg_delay_days: float = Field(default=5.0, ge=0)
    worst_delinquency_stage: int = Field(default=0, ge=0, le=4)
    financial_stress_index: float = Field(default=0.3, ge=0, le=1)
    debt_to_income_ratio: float = Field(default=2.0, ge=0)
    income_stability_score: float = Field(default=0.7, ge=0, le=1)
    spending_volatility_index: float = Field(default=0.25, ge=0, le=1)
    digital_engagement_score: float = Field(default=60.0, ge=0, le=100)

    @validator('employment_type')
    def validate_employment(cls, v):
        valid = ['Salaried', 'Self-Employed', 'Gig Worker', 'SME Owner', 'First-Time Borrower']
        if v not in valid:
            raise ValueError(f'employment_type must be one of {valid}')
        return v


class FraudRequest(BaseModel):
    applicant_id: str
    monthly_income: float
    employment_type: str
    credit_history_length: int
    loan_amount: float
    worst_delinquency_stage: int = 0
    income_stability_score: float = 0.7


class EWSRequest(BaseModel):
    loan_id: str
    current_dpd_stage: int = Field(default=0, ge=0, le=4)
    missed_payment_ratio: float = Field(default=0.05, ge=0, le=1)
    financial_stress_index: float = Field(default=0.3, ge=0, le=1)
    income_stability_score: float = Field(default=0.7, ge=0, le=1)
    spending_volatility_index: float = Field(default=0.25, ge=0, le=1)
    bank_balance_avg: float = Field(default=50000.0)


class UnderwritingResponse(BaseModel):
    applicant_id: str
    request_id: str
    timestamp: str
    pd_score: float
    risk_grade: str
    decision: str
    confidence: float
    interest_rate: float
    approved_amount: float
    emi_amount: float
    expected_loss: float
    lgd: float
    latency_ms: float
    model_version: str


class FraudResponse(BaseModel):
    applicant_id: str
    request_id: str
    fraud_score: float
    fraud_severity: str
    alert_level: str
    recommended_action: str
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: dict
    uptime_seconds: float
    request_count: int


# ── Inference Engine (rule-based fallback) ────────────────────────────────────
GRADE_THRESHOLDS = {
    'A': (0.00, 0.05), 'B': (0.05, 0.10),
    'C': (0.10, 0.18), 'D': (0.18, 0.30), 'E': (0.30, 1.00),
}
LGD_BY_GRADE = {'A': 0.50, 'B': 0.55, 'C': 0.60, 'D': 0.65, 'E': 0.70}
RATE_FLOORS  = {'A': 10.5, 'B': 12.5, 'C': 15.0, 'D': 18.0, 'E': 22.0}
RATE_CEILS   = {'A': 14.0, 'B': 16.5, 'C': 20.0, 'D': 24.0, 'E': 32.0}
INCOME_MEDIANS = {
    'Salaried': 45000, 'Self-Employed': 52000, 'Gig Worker': 22000,
    'SME Owner': 85000, 'First-Time Borrower': 18000,
}

START_TIME   = time.time()
REQUEST_COUNT = 0
MODEL_VERSION = 'v1.0.0-rule-based'


def compute_pd(req: UnderwritingRequest) -> float:
    cs_norm   = max(0, (900 - req.credit_score) / 600)
    mpr_norm  = min(req.missed_payment_ratio, 1.0)
    dpd_norm  = req.worst_delinquency_stage / 4.0
    dly_norm  = min(req.avg_delay_days / 90, 1.0)
    fsi_norm  = min(req.financial_stress_index, 1.0)
    dti_norm  = min(req.debt_to_income_ratio / 10, 1.0)
    svi_norm  = min(req.spending_volatility_index, 1.0)
    hist_norm = max(0, 1 - req.credit_history_length / 60)
    raw_pd = (
        0.22 * mpr_norm + 0.20 * dpd_norm + 0.12 * cs_norm
        + 0.15 * fsi_norm + 0.10 * dly_norm + 0.08 * dti_norm
        + 0.08 * svi_norm + 0.05 * hist_norm
    )
    if req.credit_history_length < 6:
        raw_pd = min(raw_pd + 0.06, 0.95)
    if req.worst_delinquency_stage >= 3:
        raw_pd = min(raw_pd + 0.15, 0.95)
    return float(np.clip(raw_pd, 0.01, 0.95))


def get_grade(pd_score: float) -> str:
    for g, (lo, hi) in GRADE_THRESHOLDS.items():
        if lo <= pd_score < hi:
            return g
    return 'E'


def compute_emi(principal: float, rate_pct: float, months: int) -> float:
    mr = rate_pct / 100 / 12
    if mr == 0 or months == 0:
        return principal / max(months, 1)
    return principal * mr * (1 + mr)**months / ((1 + mr)**months - 1)


def generate_request_id(applicant_id: str) -> str:
    seed = f'{applicant_id}{time.time()}'
    return 'REQ' + hashlib.md5(seed.encode()).hexdigest()[:10].upper()


# ── Logging utilities ────────────────────────────────────────────────────────
PREDICTION_LOG = []

def log_prediction(request_id: str, model: str, inputs: dict,
                   outputs: dict, latency: float):
    PREDICTION_LOG.append({
        'request_id': request_id, 'model': model,
        'timestamp': datetime.now().isoformat(),
        'inputs': inputs, 'outputs': outputs, 'latency_ms': latency,
    })
    if len(PREDICTION_LOG) > 10000:
        PREDICTION_LOG.pop(0)


# ── API Routes ───────────────────────────────────────────────────────────────
@app.get('/', summary='Root')
async def root():
    return {
        'platform': 'FinLend AI Model Serving API',
        'version': '1.0.0',
        'docs': '/docs',
        'health': '/health',
        'status': 'operational',
    }


@app.get('/health', response_model=HealthResponse, summary='Health check')
async def health_check():
    global REQUEST_COUNT
    return HealthResponse(
        status='healthy',
        timestamp=datetime.now().isoformat(),
        models_loaded={
            'underwriting': MODEL_VERSION,
            'fraud': MODEL_VERSION,
            'ews': MODEL_VERSION,
        },
        uptime_seconds=round(time.time() - START_TIME, 1),
        request_count=REQUEST_COUNT,
    )


@app.post('/v1/underwriting/score', response_model=UnderwritingResponse,
          summary='Real-time loan underwriting score')
async def underwriting_score(req: UnderwritingRequest,
                              background_tasks: BackgroundTasks):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    t0 = time.time()
    request_id = generate_request_id(req.applicant_id)

    try:
        pd_score  = compute_pd(req)
        grade     = get_grade(pd_score)
        lgd       = LGD_BY_GRADE[grade]
        tenure_yr = req.loan_tenure_months / 12
        cum_pd    = 1 - (1 - pd_score) ** tenure_yr
        ecl       = cum_pd * lgd * req.loan_amount

        base_rate  = 8.5 + 2.0 + 2.0
        rate = float(np.clip(
            base_rate + pd_score * 0.55 * 100,
            RATE_FLOORS[grade], RATE_CEILS[grade]
        ))

        # Decision logic
        if pd_score > 0.55 or req.debt_to_income_ratio > 7:
            decision, confidence = 'DECLINE', 0.92
        elif grade == 'E' or pd_score > 0.30:
            decision, confidence = 'MANUAL_REVIEW', 0.70
        elif req.credit_history_length < 6:
            decision, confidence = 'MANUAL_REVIEW', 0.60
        else:
            decision = 'APPROVE'
            confidence = float(np.clip(1 - pd_score * 2, 0.55, 0.99))

        grade_caps = {'A': 1.0, 'B': 0.90, 'C': 0.80, 'D': 0.65, 'E': 0.45}
        max_emi    = req.monthly_income * 0.40
        mr_m       = rate / 100 / 12
        n_m        = req.loan_tenure_months
        max_afford = (max_emi * ((1 + mr_m)**n_m - 1) / (mr_m * (1 + mr_m)**n_m)
                      if mr_m > 0 else max_emi * n_m)
        approved   = min(req.loan_amount, max_afford * grade_caps[grade])
        emi        = compute_emi(approved, rate, n_m)
        latency    = round((time.time() - t0) * 1000, 2)

        response = UnderwritingResponse(
            applicant_id=req.applicant_id,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            pd_score=round(pd_score, 4),
            risk_grade=grade,
            decision=decision,
            confidence=round(confidence, 3),
            interest_rate=round(rate, 2),
            approved_amount=round(approved, 2),
            emi_amount=round(emi, 2),
            expected_loss=round(ecl, 2),
            lgd=lgd,
            latency_ms=latency,
            model_version=MODEL_VERSION,
        )

        background_tasks.add_task(
            log_prediction, request_id, 'underwriting',
            req.dict(), response.dict(), latency
        )
        log.info('UW score %s | PD=%.3f | %s | %.1fms',
                 request_id, pd_score, decision, latency)
        return response

    except Exception as e:
        log.error('Scoring error %s: %s', request_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/v1/fraud/score', response_model=FraudResponse,
          summary='Real-time fraud risk score')
async def fraud_score(req: FraudRequest, background_tasks: BackgroundTasks):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    t0 = time.time()
    request_id = generate_request_id(req.applicant_id)

    exp_income = INCOME_MEDIANS.get(req.employment_type, 40000)
    income_ratio = req.monthly_income / max(exp_income, 1)
    inflation_score = min(max(income_ratio - 1.0, 0) / 2.0, 1.0)

    doc_risk = float(
        (req.credit_history_length < 6) * 0.30
        + (req.worst_delinquency_stage >= 3) * 0.20
        + (1 - req.income_stability_score) * 0.25
    )
    identity_risk = 1 - min(
        0.90 + (req.income_stability_score - 0.5) * 0.20
        - (req.credit_history_length < 6) * 0.20, 1.0
    )
    synthetic_id = float(
        (req.credit_history_length < 6) * 0.40
        + (req.loan_amount > req.monthly_income * 20) * 0.25
        + doc_risk * 0.20
    )
    fraud_score_val = float(np.clip(
        0.35 * inflation_score + 0.25 * doc_risk
        + 0.25 * identity_risk + 0.15 * synthetic_id, 0, 1
    ))

    if fraud_score_val >= 0.75:
        severity, alert, action = 'Red', 'CRITICAL', 'Block & investigate immediately'
    elif fraud_score_val >= 0.55:
        severity, alert, action = 'Orange', 'HIGH', 'Enhanced KYC verification required'
    elif fraud_score_val >= 0.35:
        severity, alert, action = 'Yellow', 'MEDIUM', 'Manual review recommended'
    else:
        severity, alert, action = 'Green', 'LOW', 'Proceed with standard processing'

    latency = round((time.time() - t0) * 1000, 2)
    response = FraudResponse(
        applicant_id=req.applicant_id,
        request_id=request_id,
        fraud_score=round(fraud_score_val, 4),
        fraud_severity=severity,
        alert_level=alert,
        recommended_action=action,
        latency_ms=latency,
        model_version=MODEL_VERSION,
    )
    background_tasks.add_task(
        log_prediction, request_id, 'fraud',
        req.dict(), response.dict(), latency
    )
    return response


@app.get('/v1/metrics/predictions', summary='Recent prediction metrics')
async def prediction_metrics():
    if not PREDICTION_LOG:
        return {'message': 'No predictions yet', 'count': 0}
    latencies = [p['latency_ms'] for p in PREDICTION_LOG]
    return {
        'total_requests': REQUEST_COUNT,
        'logged_predictions': len(PREDICTION_LOG),
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'models': list({p['model'] for p in PREDICTION_LOG}),
    }


@app.get('/v1/models', summary='List deployed models')
async def list_models():
    return {
        'models': [
            {'name': 'underwriting', 'version': MODEL_VERSION, 'status': 'production',
             'endpoint': '/v1/underwriting/score'},
            {'name': 'fraud',        'version': MODEL_VERSION, 'status': 'production',
             'endpoint': '/v1/fraud/score'},
            {'name': 'ews',          'version': MODEL_VERSION, 'status': 'production',
             'endpoint': '/v1/ews/score'},
        ]
    }
