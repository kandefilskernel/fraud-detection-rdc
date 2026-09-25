from pydantic import BaseModel


class FeatureContribution(BaseModel):
    feature: str
    shap: float
    value: float


class Explanation(BaseModel):
    top_features: list[FeatureContribution]
    branch_scores: dict[str, float]
    branch_contributions: dict[str, float]
    attention_on_history: list[float]


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud_predicted: bool
    threshold: float
    action: str                 # APPROVE | VERIFY | BLOCK
    risk_level: str             # FAIBLE | MOYEN | ELEVE | CRITIQUE
    reason: str
    expected_costs_usd: dict[str, float]
    rules_triggered: list[str]
    known_user: bool
    explanation: Explanation
    degraded: bool
    degraded_branches: list[str]
    model_version: str
    latency_ms: float
    features: dict[str, float]
