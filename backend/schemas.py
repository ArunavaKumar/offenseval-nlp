# ============================================================
# schemas.py
# Pydantic request and response models
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Prediction Request Schema
# ============================================================

class PredictionRequest(BaseModel):
    """
    Request model for the prediction endpoint.

    Attributes
    ----------
    text : str
        Input text to be classified.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to classify"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """
        Remove leading and trailing spaces and ensure
        that the submitted text is not empty.
        """

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Text cannot be empty.")

        return cleaned_value


# ============================================================
# Prediction Response Schema
# ============================================================

class PredictionResponse(BaseModel):
    """
    Response model returned by the prediction endpoint.

    Attributes
    ----------
    prediction : str
        Predicted offensive-language class label.

    confidence : float
        Confidence score for the predicted class.

    cleaned_text : str
        Text after preprocessing and normalization.

    sentiment_score : float
        VADER compound sentiment score.

    sentiment_category : str
        Overall sentiment category.

    positive_score : float
        Positive sentiment proportion.

    neutral_score : float
        Neutral sentiment proportion.

    negative_score : float
        Negative sentiment proportion.
    """

    prediction: str = Field(
        ...,
        description="Predicted offensive-language class"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted class"
    )

    cleaned_text: str = Field(
        ...,
        description="Preprocessed version of the input text"
    )

    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="VADER compound sentiment score"
    )

    sentiment_category: str = Field(
        ...,
        description="Positive, Neutral or Negative sentiment category"
    )

    positive_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Positive sentiment proportion"
    )

    neutral_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Neutral sentiment proportion"
    )

    negative_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Negative sentiment proportion"
    )


# ============================================================
# Health Check Response Schema
# ============================================================

class HealthResponse(BaseModel):
    """
    Response model for the health-check endpoint.

    Attributes
    ----------
    status : str
        Current API health status.

    model_ready : bool
        Indicates whether the trained prediction model
        is loaded and ready for inference.
    """

    status: str = Field(
        ...,
        description="Current API health status"
    )

    model_ready: bool = Field(
        ...,
        description="Whether the prediction model is ready"
    )