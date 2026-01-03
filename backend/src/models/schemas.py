"""
Pydantic Models/Schemas for T1D-AI
Defines data structures for glucose readings, treatments, users, etc.
"""
from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from enum import Enum


# ==================== Enums ====================

class GlucoseRange(str, Enum):
    CRITICAL_LOW = "critical_low"    # < 54 mg/dL
    LOW = "low"                      # 54-70 mg/dL
    IN_RANGE = "in_range"            # 70-180 mg/dL
    HIGH = "high"                    # 180-250 mg/dL
    CRITICAL_HIGH = "critical_high"  # > 250 mg/dL


class TrendDirection(str, Enum):
    DOUBLE_UP = "DoubleUp"
    SINGLE_UP = "SingleUp"
    FORTY_FIVE_UP = "FortyFiveUp"
    FLAT = "Flat"
    FORTY_FIVE_DOWN = "FortyFiveDown"
    SINGLE_DOWN = "SingleDown"
    DOUBLE_DOWN = "DoubleDown"
    NOT_COMPUTABLE = "NotComputable"
    RATE_OUT_OF_RANGE = "RateOutOfRange"


class TreatmentType(str, Enum):
    INSULIN = "insulin"
    CARBS = "carbs"
    CORRECTION_BOLUS = "Correction Bolus"
    CARB_CORRECTION = "Carb Correction"


# ==================== Glucose Models ====================

class GlucoseReading(BaseModel):
    """A single glucose reading from CGM."""
    id: str = Field(..., description="Unique identifier")
    userId: str = Field(..., description="User ID for partitioning")
    timestamp: datetime = Field(..., description="Reading timestamp")
    value: int = Field(..., ge=20, le=600, description="Glucose value in mg/dL")
    trend: Optional[TrendDirection] = Field(None, description="Trend direction")
    source: str = Field(default="gluroo", description="Data source")
    sourceId: Optional[str] = Field(None, description="Original ID from source")

    # Calculated fields (populated after processing)
    iob: Optional[float] = Field(None, description="Insulin on Board at this time")
    cob: Optional[float] = Field(None, description="Carbs on Board at this time")
    isf: Optional[float] = Field(None, description="Predicted ISF at this time")

    class Config:
        use_enum_values = True


class GlucosePrediction(BaseModel):
    """ML predictions for a glucose reading."""
    timestamp: datetime
    linear: List[float] = Field(default_factory=list, description="Linear predictions [5m, 10m, 15m]")
    lstm: List[float] = Field(default_factory=list, description="LSTM predictions [5m, 10m, 15m]")


class GlucoseWithPredictions(GlucoseReading):
    """Glucose reading with ML predictions."""
    predictions: Optional[GlucosePrediction] = None


# ==================== Treatment Models ====================

class Treatment(BaseModel):
    """An insulin or carb treatment."""
    id: str = Field(..., description="Unique identifier")
    userId: str = Field(..., description="User ID for partitioning")
    timestamp: datetime = Field(..., description="Treatment timestamp")
    type: TreatmentType = Field(..., description="Type of treatment")
    insulin: Optional[float] = Field(None, ge=0, description="Insulin units")
    carbs: Optional[float] = Field(None, ge=0, description="Carbs in grams")
    protein: Optional[float] = Field(None, ge=0, description="Protein in grams")
    fat: Optional[float] = Field(None, ge=0, description="Fat in grams")
    notes: Optional[str] = Field(None, description="User notes")
    source: str = Field(default="gluroo", description="Data source")
    sourceId: Optional[str] = Field(None, description="Original ID from source")

    class Config:
        use_enum_values = True


# ==================== User Models ====================

class UserSettings(BaseModel):
    """User-specific settings for diabetes management."""
    timezone: str = Field(default="UTC")
    targetBg: int = Field(default=100, ge=70, le=150)
    insulinSensitivity: float = Field(default=50.0, ge=10, le=200)
    carbRatio: float = Field(default=10.0, ge=1, le=50)
    insulinDuration: int = Field(default=180, ge=120, le=360)
    carbAbsorptionDuration: int = Field(default=180, ge=60, le=360)

    # Alert thresholds
    highThreshold: int = Field(default=180)
    lowThreshold: int = Field(default=70)
    criticalHighThreshold: int = Field(default=250)
    criticalLowThreshold: int = Field(default=54)

    # Feature toggles
    enableAlerts: bool = Field(default=True)
    enablePredictiveAlerts: bool = Field(default=True)
    showInsights: bool = Field(default=True)


class User(BaseModel):
    """User account model."""
    id: str = Field(..., description="Unique user ID")
    email: str = Field(..., description="User email")
    displayName: Optional[str] = Field(None)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    settings: UserSettings = Field(default_factory=UserSettings)


# ==================== Data Source Models ====================

class GlurooCredentials(BaseModel):
    """Encrypted Gluroo connection credentials."""
    url: str = Field(..., description="Gluroo Nightscout URL")
    apiSecretEncrypted: str = Field(..., description="Encrypted API secret")
    lastSyncAt: Optional[datetime] = Field(None)
    syncEnabled: bool = Field(default=True)


class DataSource(BaseModel):
    """User data source configuration."""
    id: str
    userId: str
    type: Literal["gluroo"] = "gluroo"
    credentials: GlurooCredentials
    createdAt: datetime = Field(default_factory=datetime.utcnow)


# ==================== Metrics Models ====================

class CurrentMetrics(BaseModel):
    """Current calculated metrics for display."""
    iob: float = Field(default=0.0, description="Insulin on Board (units)")
    cob: float = Field(default=0.0, description="Carbs on Board (grams)")
    isf: float = Field(default=50.0, description="Predicted ISF")
    recommendedDose: float = Field(default=0.0, description="Recommended correction dose")
    effectiveBg: int = Field(default=0, description="BG adjusted for IOB/COB")


class PredictionAccuracy(BaseModel):
    """Accuracy tracking for predictions."""
    linearWins: int = Field(default=0)
    lstmWins: int = Field(default=0)
    totalComparisons: int = Field(default=0)


# ==================== API Response Models ====================

class GlucoseCurrentResponse(BaseModel):
    """Response for current glucose endpoint."""
    glucose: GlucoseWithPredictions
    metrics: CurrentMetrics
    accuracy: PredictionAccuracy


class GlucoseHistoryResponse(BaseModel):
    """Response for glucose history endpoint."""
    readings: List[GlucoseReading]
    totalCount: int
    startTime: datetime
    endTime: datetime


# ==================== Insight Models ====================

class AIInsight(BaseModel):
    """GPT-generated insight."""
    id: str
    userId: str
    content: str
    category: Literal["pattern", "recommendation", "warning", "achievement"]
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    expiresAt: Optional[datetime] = None
