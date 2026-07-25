from fastapi import FastAPI, HTTPException

from .schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ModelInfoResponse,
    ScoreRequest,
    ScoreResponse,
)
from .service import KafkaPublishError, health_payload, model_info_payload, score_transaction


app = FastAPI(
    title="Fraud Scoring API",
    version="1.0.0",
    description="Real-time transaction scoring API for the fraud detection pipeline.",
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(**health_payload())


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_info_payload())


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest) -> ScoreResponse:
    try:
        return score_transaction(payload)
    except (KafkaPublishError, RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(payload: BatchScoreRequest) -> BatchScoreResponse:
    try:
        return BatchScoreResponse(predictions=[score_transaction(item) for item in payload.transactions])
    except (KafkaPublishError, RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
