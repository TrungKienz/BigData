from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fraud_pipeline.config import PipelineConfig


DEFAULT_SCHEMA_VERSION = PipelineConfig().schema_version


class ScoreRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    event_id: Optional[str] = Field(default=None, description="Optional transaction identifier")
    event_time: Optional[datetime] = Field(default=None, description="Optional event timestamp in ISO-8601")
    producer_ts: Optional[datetime] = Field(default=None, description="Optional producer timestamp in ISO-8601")
    step: int = Field(..., ge=0)
    txn_type: str = Field(..., alias="type", min_length=1)
    amount: float = Field(..., ge=0)
    name_orig: str = Field(..., alias="nameOrig", min_length=1)
    oldbalance_org: float = Field(..., alias="oldbalanceOrg")
    name_dest: str = Field(..., alias="nameDest", min_length=1)
    oldbalance_dest: float = Field(..., alias="oldbalanceDest")
    newbalance_orig: Optional[float] = Field(default=None, alias="newbalanceOrig")
    newbalance_dest: Optional[float] = Field(default=None, alias="newbalanceDest")
    is_fraud: int = Field(default=0, alias="isFraud", ge=0, le=1)
    schema_version: int = Field(default=DEFAULT_SCHEMA_VERSION, ge=1)


class ScoreResponse(BaseModel):
    event_id: str
    is_alert: bool
    decision: str
    risk_score: float
    rule_score: float
    ml_score: float
    hybrid_score: float
    threshold: float
    severity: str
    ml_model_version: str
    model_version: str
    model_tag: str
    feature_configuration: str
    rule_weight: float
    ml_weight: float
    triggered_rules: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_type: str
    prediction_logging_enabled: bool


class ModelInfoResponse(BaseModel):
    artifact_path: str
    model_loaded: bool
    model_version: str
    model_tag: str
    feature_configuration: str
    feature_count: int
    hybrid_threshold: float
    rule_weight: float
    ml_weight: float
    feature_columns: Optional[List[str]] = None
    ml_threshold: Optional[float] = None


class BatchScoreRequest(BaseModel):
    transactions: List[ScoreRequest] = Field(..., min_length=1)


class BatchScoreResponse(BaseModel):
    predictions: List[ScoreResponse]
