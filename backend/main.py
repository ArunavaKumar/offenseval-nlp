# ============================================================
# main.py
# FastAPI application entry point
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
import logging

from fastapi import FastAPI, HTTPException, status

from backend.inference import model_is_ready, predict_sentence
from backend.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse
)

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
# Configure application logging for debugging and monitoring.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------
# Initialize the FastAPI application with API metadata.

app = FastAPI(
    title="OffensEval NLP API",
    description="REST API for Offensive Language Detection",
    version="1.0.0"
)

# ============================================================
# Home Endpoint
# ============================================================

@app.get("/")
def home() -> dict:
    """
    Root endpoint.

    Returns
    -------
    dict
        Basic API information and documentation link.
    """

    return {
        "message": "Welcome to OffensEval NLP API",
        "status": "running",
        "docs": "/docs"
    }

# ============================================================
# Health Check Endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health() -> HealthResponse:
    """
    Verify whether the API and prediction model
    are ready to serve requests.

    Returns
    -------
    HealthResponse
        Current application health status.
    """

    ready = model_is_ready()

    return HealthResponse(
        status="healthy" if ready else "unhealthy",
        model_ready=ready
    )

# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
def predict(
    request: PredictionRequest
) -> PredictionResponse:
    """
    Predict whether the supplied text contains
    offensive language.

    Parameters
    ----------
    request : PredictionRequest
        Input request containing the text to classify.

    Returns
    -------
    PredictionResponse
        Predicted label and confidence score.
    """

    try:
        # Generate prediction
        result = predict_sentence(request.text)

        return PredictionResponse(**result)

    except ValueError as error:
        # Invalid user input
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        ) from error

    except Exception as error:
        # Unexpected server error
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal server error."
        ) from error