"""
Pydantic Models/Schemas for T1D-AI
Defines data structures for glucose readings, treatments, users, etc.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator
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


# Numeric to string mapping for trends (Dexcom API uses numbers)
TREND_NUMBER_MAP = {
    0: None,  # No data
    1: "DoubleUp",
    2: "SingleUp",
    3: "FortyFiveUp",
    4: "Flat",
    5: "FortyFiveDown",
    6: "SingleDown",
    7: "DoubleDown",
}


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
    trend: Optional[str] = Field(None, description="Trend direction")
    source: str = Field(default="gluroo", description="Data source")
    sourceId: Optional[str] = Field(None, description="Original ID from source")

    # Calculated fields (populated after processing)
    iob: Optional[float] = Field(None, description="Insulin on Board at this time")
    cob: Optional[float] = Field(None, description="Carbs on Board at this time")
    isf: Optional[float] = Field(None, description="Predicted ISF at this time")

    @field_validator('trend', mode='before')
    @classmethod
    def convert_trend(cls, v):
        """Convert numeric trends to string values."""
        if v is None:
            return None
        if isinstance(v, int):
            return TREND_NUMBER_MAP.get(v)
        if isinstance(v, str):
            # Already a string, return as-is
            return v
        return None

    class Config:
        use_enum_values = True


class GlucosePrediction(BaseModel):
    """ML predictions for a glucose reading."""
    timestamp: datetime
    linear: List[float] = Field(default_factory=list, description="Linear predictions [5m, 10m, 15m]")
    lstm: List[float] = Field(default_factory=list, description="LSTM predictions [5m, 10m, 15m]")


class TFTPrediction(BaseModel):
    """TFT prediction with uncertainty bounds for long-horizon prediction."""
    timestamp: datetime = Field(..., description="Prediction timestamp")
    horizon: int = Field(..., description="Prediction horizon in minutes (30, 45, 60)")
    value: float = Field(..., description="Median prediction (50th percentile)")
    lower: float = Field(..., description="Lower bound (10th percentile)")
    upper: float = Field(..., description="Upper bound (90th percentile)")
    tftDelta: Optional[float] = Field(None, description="TFT modifier delta (mg/dL) - how much TFT adjusted the physics baseline")


class EffectPoint(BaseModel):
    """IOB/COB/POB effect projection at a point in time."""
    minutesAhead: int = Field(..., description="Minutes ahead from current time")
    iobEffect: float = Field(..., description="BG lowering from IOB (negative mg/dL)")
    cobEffect: float = Field(..., description="BG raising from COB (positive mg/dL)")
    pobEffect: float = Field(default=0.0, description="BG raising from POB (positive mg/dL, delayed)")
    netEffect: float = Field(..., description="Combined IOB+COB+POB effect (mg/dL)")
    remainingIOB: float = Field(..., description="IOB remaining at this time (units)")
    remainingCOB: float = Field(..., description="COB remaining at this time (grams)")
    remainingPOB: float = Field(default=0.0, description="POB remaining at this time (grams)")
    insulinActivity: float = Field(default=0.0, description="Insulin activity level (0-1, bell-shaped curve)")
    carbActivity: float = Field(default=0.0, description="Carb absorption activity level (0-1, bell-shaped curve)")
    proteinActivity: float = Field(default=0.0, description="Protein absorption activity level (0-1, delayed curve)")
    expectedBg: Optional[float] = Field(default=None, description="Projected BG at this time accounting for IOB/COB/POB effects")
    bgWithIobOnly: Optional[float] = Field(default=None, description="BG trajectory with only insulin effect (pull-down floor)")
    bgWithCobOnly: Optional[float] = Field(default=None, description="BG trajectory with only carb effect (push-up ceiling)")


class GlucoseWithPredictions(GlucoseReading):
    """Glucose reading with ML predictions."""
    predictions: Optional[GlucosePrediction] = None


# ==================== Treatment Models ====================

class AbsorptionRate(str, Enum):
    """Carb absorption rate based on food type and GI."""
    NONE = "none"            # Not specified or unknown
    VERY_FAST = "very_fast"  # Glucose, juice, chocolate milk - spike in 3-8 min
    FAST = "fast"            # White bread, candy, potatoes - spike in 8-15 min
    MEDIUM = "medium"        # Rice, pasta, fruits - spike in 15-30 min
    SLOW = "slow"            # Whole grains, beans - spike in 30-60 min
    VERY_SLOW = "very_slow"  # High-fat meals (pizza, burgers) - extended 1-4 hours


class FatContent(str, Enum):
    """Fat content level affecting absorption delay."""
    NONE = "none"      # No fat (pure carbs like juice, candy)
    LOW = "low"        # < 5g fat
    MEDIUM = "medium"  # 5-15g fat
    HIGH = "high"      # > 15g fat (pizza, fried foods - delays absorption)


class InferenceStatus(str, Enum):
    """Confirmation status for ML-inferred treatments."""
    PENDING = "pending"      # Not yet confirmed by user
    CONFIRMED = "confirmed"  # User confirmed this inference
    DISMISSED = "dismissed"  # User dismissed as incorrect


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
    fiber: Optional[float] = Field(None, ge=0, description="Fiber in grams (delays absorption)")
    notes: Optional[str] = Field(None, description="User notes (food description)")
    source: str = Field(default="gluroo", description="Data source")
    sourceId: Optional[str] = Field(None, description="Original ID from source")

    # LLM-enriched food features (populated async after logging)
    glycemicIndex: Optional[int] = Field(None, ge=0, le=120, description="Glycemic index (0-120, precise value from AI)")
    glycemicLoad: Optional[float] = Field(None, ge=0, description="Glycemic load = carbs * GI / 100")
    absorptionRate: Optional[AbsorptionRate] = Field(None, description="Carb absorption rate from AI analysis")
    fatContent: Optional[FatContent] = Field(None, description="Fat content level affecting absorption delay")
    isLiquid: Optional[bool] = Field(None, description="True for drinks - liquids absorb 40% faster")
    enrichedAt: Optional[datetime] = Field(None, description="When food features were enriched")

    # Treatment Inference fields (for ML-inferred treatments to explain unexplained BG movements)
    # Inferred treatments are shown in UI with visual distinction and require user confirmation
    isInferred: bool = Field(
        default=False,
        description="Whether this treatment was inferred by ML to explain unexplained BG movement"
    )
    inferenceConfidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Confidence level of inference (0-1). Higher = more certain the inference is correct."
    )
    inferenceReason: Optional[str] = Field(
        None,
        description="Human-readable explanation of why this treatment was inferred (e.g., 'BG rose 45 mg/dL without logged carbs')"
    )
    confirmationStatus: Optional[InferenceStatus] = Field(
        None,
        description="User confirmation status. Only confirmed inferred treatments are used for future training."
    )
    inferredAt: Optional[datetime] = Field(
        None,
        description="When this treatment was inferred by the ML system"
    )

    # User edit tracking (prevents Gluroo sync from overwriting)
    userEdited: bool = Field(
        default=False,
        description="True if user manually edited this treatment. Prevents Gluroo sync from overwriting."
    )
    lastGlurooSync: Optional[datetime] = Field(
        None,
        description="Last time this treatment was synced from Gluroo"
    )

    @field_validator('absorptionRate', mode='before')
    @classmethod
    def validate_absorption_rate(cls, v):
        """Convert invalid absorption rate values to None."""
        if v is None:
            return None
        # Handle string values
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Map 'n/a', 'na', 'unknown', empty string to None
            if v_lower in ('n/a', 'na', 'unknown', '', 'null'):
                return None
            # Check if it's a valid enum value
            valid_values = {'none', 'very_fast', 'fast', 'medium', 'slow', 'very_slow'}
            if v_lower in valid_values:
                return v_lower
        # If already an enum or valid, return as-is
        return v

    @field_validator('fatContent', mode='before')
    @classmethod
    def validate_fat_content(cls, v):
        """Convert invalid fat content values to None."""
        if v is None:
            return None
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ('n/a', 'na', 'unknown', '', 'null'):
                return None
            valid_values = {'none', 'low', 'medium', 'high'}
            if v_lower in valid_values:
                return v_lower
        return v

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

    # Protein and ratio settings
    proteinRatio: float = Field(default=25.0, ge=10, le=100, description="Manual PIR - grams protein per unit insulin")
    useLearnedICR: bool = Field(default=False, description="Use AI-learned ICR instead of manual carbRatio")
    useLearnedPIR: bool = Field(default=False, description="Use AI-learned PIR instead of manual proteinRatio")
    includeProteinInBolus: bool = Field(default=False, description="Include protein in bolus calculations")

    # Prediction settings
    useTFTModifiers: bool = Field(default=True, description="Enable TFT neural network modifiers on physics predictions")
    trackPredictionAccuracy: bool = Field(default=True, description="Track prediction accuracy over time")


class User(BaseModel):
    """User account model.

    Supports multiple account types:
    - personal: Direct T1D user managing their own data
    - parent: Parent/guardian monitoring a child with T1D
    - child: Child T1D user monitored by a parent
    """
    id: str = Field(..., description="Unique user ID")
    email: str = Field(..., description="User email")
    displayName: Optional[str] = Field(None)
    passwordHash: Optional[str] = Field(None, description="Hashed password for email/password auth")
    authProvider: Literal["email", "microsoft", "google"] = Field(default="email")
    microsoftId: Optional[str] = Field(None, description="Microsoft account ID for OAuth")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    settings: UserSettings = Field(default_factory=UserSettings)

    # Email verification (for email/password auth)
    emailVerified: bool = Field(default=False, description="Whether email has been verified")
    emailVerificationToken: Optional[str] = Field(None, description="Token for email verification")
    emailVerificationExpires: Optional[datetime] = Field(None, description="When verification token expires")

    # Onboarding status
    onboardingCompleted: bool = Field(default=False, description="Whether user has completed onboarding")
    onboardingCompletedAt: Optional[datetime] = Field(None, description="When onboarding was completed")
    preferredDataSources: List[str] = Field(
        default_factory=list,
        description="Data sources user wants supported (for feature prioritization)"
    )

    # Account type and relationships
    accountType: Literal["personal", "parent", "child"] = Field(
        default="personal",
        description="Type of account: personal (self-managed T1D), parent (monitoring child), or child (monitored by parent)"
    )

    # For child accounts - link to parent
    parentId: Optional[str] = Field(None, description="Parent user ID for child accounts")
    guardianEmail: Optional[str] = Field(None, description="Guardian email for child accounts")

    # For parent accounts - list of monitored children
    linkedChildIds: List[str] = Field(
        default_factory=list,
        description="List of child user IDs this parent monitors"
    )

    # Profile information
    dateOfBirth: Optional[datetime] = Field(None, description="Date of birth for age-appropriate features")
    diagnosisDate: Optional[datetime] = Field(None, description="Date of T1D diagnosis")

    # For parents/guardians who also have T1D
    hasT1D: bool = Field(default=True, description="Whether this user has T1D themselves")

    # Avatar/profile customization (especially for children)
    avatarUrl: Optional[str] = Field(None, description="Profile avatar URL")
    theme: Optional[str] = Field(None, description="UI theme preference")

    # Admin and tracking
    isAdmin: bool = Field(default=False, description="Whether user has admin privileges")
    lastLoginAt: Optional[datetime] = Field(None, description="Last login timestamp")


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
    pob: float = Field(default=0.0, description="Protein on Board (grams)")
    isf: float = Field(default=50.0, description="Predicted ISF")
    recommendedDose: float = Field(default=0.0, description="Recommended correction dose")
    effectiveBg: int = Field(default=0, description="BG adjusted for IOB/COB/POB")
    proteinDoseNow: float = Field(default=0.0, description="Protein insulin to give NOW (with time decay)")
    proteinDoseLater: float = Field(default=0.0, description="Protein insulin to give LATER (remaining)")


class PredictionAccuracy(BaseModel):
    """Accuracy tracking for predictions."""
    linearWins: int = Field(default=0)
    lstmWins: int = Field(default=0)
    totalComparisons: int = Field(default=0)


# ==================== API Response Models ====================

class HistoricalIobCobPoint(BaseModel):
    """Historical IOB/COB/POB and BG pressure at a specific timestamp."""
    timestamp: datetime = Field(..., description="Point in time")
    iob: float = Field(..., description="Insulin on Board at this time (units)")
    cob: float = Field(..., description="Carbs on Board at this time (grams)")
    pob: float = Field(default=0.0, description="Protein on Board at this time (grams)")
    bgPressure: Optional[float] = Field(None, description="Net BG pressure: where BG is heading based on IOB+COB+POB")
    actualBg: Optional[float] = Field(None, description="Actual BG reading at this time (for comparison)")


class GlucoseCurrentResponse(BaseModel):
    """Response for current glucose endpoint."""
    glucose: GlucoseWithPredictions
    metrics: CurrentMetrics
    accuracy: PredictionAccuracy
    # New ML predictions and effect curves
    tftPredictions: List["TFTPrediction"] = Field(
        default_factory=list,
        description="TFT long-horizon predictions with uncertainty (30, 45, 60 min)"
    )
    effectCurve: List["EffectPoint"] = Field(
        default_factory=list,
        description="Projected IOB/COB effect on BG over next 60 minutes"
    )
    # Historical IOB/COB for continuous plotting
    historicalIobCob: List["HistoricalIobCobPoint"] = Field(
        default_factory=list,
        description="Historical IOB/COB at each glucose reading timestamp for continuous plotting"
    )


class GlucoseHistoryResponse(BaseModel):
    """Response for glucose history endpoint."""
    readings: List[GlucoseReading]
    totalCount: int
    startTime: datetime
    endTime: datetime


# ==================== Insight Models ====================

class AIInsight(BaseModel):
    """GPT-generated insight."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    userId: str
    content: str
    category: Literal["pattern", "recommendation", "warning", "achievement"]
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    expiresAt: Optional[datetime] = None


# ==================== ISF Learning Models ====================

class ISFDataPoint(BaseModel):
    """Single ISF observation from bolus analysis."""
    timestamp: datetime
    value: float = Field(..., ge=10, le=200, description="ISF value (mg/dL per unit)")
    bgBefore: int = Field(..., description="BG before bolus")
    bgAfter: int = Field(..., description="BG after bolus (2.5h later)")
    insulinUnits: float = Field(..., description="Insulin given")
    hoursAfterMeal: Optional[float] = Field(None, description="Hours since last carbs (None if fasting)")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Data quality score")


class LearnedISF(BaseModel):
    """Learned ISF data for a user - tracks fasting and meal ISF separately."""
    id: str = Field(..., description="Format: {userId}_{isfType}")
    userId: str
    isfType: Literal["fasting", "meal"] = Field(..., description="Type of ISF - fasting or with meals")
    value: float = Field(..., ge=10, le=200, description="Current best ISF estimate (mg/dL per unit)")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Confidence in estimate")
    sampleCount: int = Field(default=0, description="Number of observations used")
    lastUpdated: datetime = Field(default_factory=datetime.utcnow)

    # Historical observations (last 30)
    history: List[ISFDataPoint] = Field(default_factory=list, description="Recent ISF observations")

    # Time-of-day patterns (ISF varies throughout the day)
    timeOfDayPattern: dict = Field(
        default_factory=lambda: {
            "morning": None,    # 6am - 11am
            "afternoon": None,  # 11am - 5pm
            "evening": None,    # 5pm - 10pm
            "night": None       # 10pm - 6am
        },
        description="ISF values by time of day"
    )

    # Statistics
    meanISF: Optional[float] = Field(None, description="Mean ISF across observations")
    stdISF: Optional[float] = Field(None, description="Standard deviation of ISF")
    minISF: Optional[float] = Field(None, description="Minimum observed ISF")
    maxISF: Optional[float] = Field(None, description="Maximum observed ISF")


# ==================== ICR Learning Models ====================

class ICRDataPoint(BaseModel):
    """Single ICR observation from meal+bolus analysis."""
    timestamp: datetime
    value: float = Field(..., ge=1, le=50, description="ICR value (grams carbs per unit insulin)")
    bgBefore: int = Field(..., description="BG before meal")
    bgAfter: int = Field(..., description="BG after meal (2-3h later)")
    insulinUnits: float = Field(..., description="Total insulin given")
    carbsGrams: float = Field(..., description="Carbs eaten")
    proteinGrams: Optional[float] = Field(None, description="Protein eaten")
    fatGrams: Optional[float] = Field(None, description="Fat eaten")
    glycemicIndex: Optional[int] = Field(None, description="Estimated GI of meal")
    mealType: Optional[Literal["breakfast", "lunch", "dinner", "snack"]] = Field(None)
    correctionComponent: Optional[float] = Field(None, description="Insulin used for correction (subtracted)")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Data quality score")


class LearnedICR(BaseModel):
    """Learned Insulin-to-Carb Ratio for a user."""
    id: str = Field(..., description="Format: {userId}_icr")
    userId: str
    value: float = Field(..., ge=1, le=50, description="Current best ICR estimate (grams per unit)")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Confidence in estimate")
    sampleCount: int = Field(default=0, description="Number of observations used")
    lastUpdated: datetime = Field(default_factory=datetime.utcnow)

    # Historical observations (last 30)
    history: List[ICRDataPoint] = Field(default_factory=list, description="Recent ICR observations")

    # Meal-type patterns (ICR varies by meal)
    mealTypePattern: dict = Field(
        default_factory=lambda: {
            "breakfast": None,
            "lunch": None,
            "dinner": None,
            "snack": None
        },
        description="ICR values by meal type"
    )

    # Time-of-day patterns
    timeOfDayPattern: dict = Field(
        default_factory=lambda: {
            "morning": None,
            "afternoon": None,
            "evening": None,
            "night": None
        },
        description="ICR values by time of day"
    )

    # Statistics
    meanICR: Optional[float] = Field(None, description="Mean ICR across observations")
    stdICR: Optional[float] = Field(None, description="Standard deviation of ICR")
    minICR: Optional[float] = Field(None, description="Minimum observed ICR")
    maxICR: Optional[float] = Field(None, description="Maximum observed ICR")


# ==================== PIR Learning Models ====================

class PIRDataPoint(BaseModel):
    """Single PIR observation from protein impact analysis."""
    timestamp: datetime
    value: float = Field(..., ge=5, le=20, description="PIR value (grams protein per unit insulin)")
    bgBefore: int = Field(..., description="BG before meal")
    bgPeak: int = Field(..., description="Peak BG from protein effect")
    bgAfter: int = Field(..., description="BG after protein effect ended")
    insulinForProtein: float = Field(..., description="Insulin attributed to protein")
    proteinGrams: float = Field(..., description="Protein eaten")
    fatGrams: Optional[float] = Field(None, description="Fat eaten (affects timing)")
    carbsGrams: float = Field(..., description="Carbs eaten (should be low for clean events)")
    mealDescription: Optional[str] = Field(None, description="Food description")
    proteinOnsetMin: int = Field(..., description="When protein started affecting BG (minutes)")
    proteinPeakMin: int = Field(..., description="When protein effect peaked (minutes)")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Data quality score")


class LearnedPIR(BaseModel):
    """Learned Protein-to-Insulin Ratio for a user."""
    id: str = Field(..., description="Format: {userId}_pir")
    userId: str
    value: float = Field(..., ge=8, le=25, description="Current best PIR estimate (grams per unit)")

    # Learned protein timing
    proteinOnsetMin: float = Field(default=120, description="Learned: when protein starts affecting BG")
    proteinPeakMin: float = Field(default=210, description="Learned: when protein effect peaks")
    proteinDurationMin: float = Field(default=300, description="Learned: total protein effect duration")

    confidence: float = Field(default=0.0, ge=0, le=1, description="Confidence in estimate")
    sampleCount: int = Field(default=0, description="Number of observations used")
    lastUpdated: datetime = Field(default_factory=datetime.utcnow)

    # Historical observations (last 30)
    history: List[PIRDataPoint] = Field(default_factory=list, description="Recent PIR observations")

    # Meal-type patterns (protein timing varies by food type)
    mealTypeTimingPattern: dict = Field(
        default_factory=lambda: {
            "high_fat_protein": {"onset": 150, "peak": 270},  # Pizza, burgers
            "lean_protein": {"onset": 90, "peak": 180},       # Chicken, fish
            "dairy_protein": {"onset": 60, "peak": 150},      # Cheese, eggs
            "mixed": {"onset": 120, "peak": 210}              # Default
        },
        description="Protein timing by food type"
    )

    # Statistics
    meanPIR: Optional[float] = Field(None, description="Mean PIR across observations")
    stdPIR: Optional[float] = Field(None, description="Standard deviation of PIR")
    minPIR: Optional[float] = Field(None, description="Minimum observed PIR")
    maxPIR: Optional[float] = Field(None, description="Maximum observed PIR")


# ==================== Personalized Absorption Curve Models ====================

class AbsorptionCurveDataPoint(BaseModel):
    """Single observation of absorption timing from BG response."""
    timestamp: datetime = Field(..., description="When treatment was given")
    treatmentId: str = Field(..., description="Associated treatment ID")
    treatmentType: str = Field(..., description="insulin, carbs, or protein")
    detectedOnsetMin: float = Field(..., description="Detected onset time in minutes")
    detectedPeakMin: float = Field(..., description="Detected peak activity time in minutes")
    detectedHalfLifeMin: Optional[float] = Field(None, description="Detected half-life in minutes")
    dataQuality: float = Field(default=0.5, ge=0, le=1, description="Quality score (clean window, stable baseline, etc)")


class UserAbsorptionProfile(BaseModel):
    """Personalized absorption curve parameters learned from user data.

    These parameters replace the hardcoded defaults in activity curve calculations:
    - Insulin: default peak_min=75 → learned insulinPeakMin
    - Carbs: default peak_min=45 → learned carbPeakMin
    - Protein: default peak_min=180 → learned proteinPeakMin
    """
    id: str = Field(..., description="Format: {userId}_absorption_profile")
    userId: str

    # Learned insulin timing (replaces hardcoded peak_min=75)
    insulinOnsetMin: float = Field(default=15, ge=5, le=60, description="When insulin starts working")
    insulinPeakMin: float = Field(default=75, ge=30, le=150, description="When insulin is most active")
    insulinDurationMin: float = Field(default=240, ge=120, le=360, description="Total insulin action duration")

    # Learned carb timing (replaces hardcoded peak_min=45)
    carbOnsetMin: float = Field(default=10, ge=0, le=45, description="When carbs start affecting BG")
    carbPeakMin: float = Field(default=45, ge=15, le=120, description="When carb effect peaks")
    carbDurationMin: float = Field(default=180, ge=90, le=300, description="Total carb effect duration")

    # Learned protein timing (replaces hardcoded peak_min=180)
    proteinOnsetMin: float = Field(default=90, ge=30, le=180, description="When protein starts affecting BG")
    proteinPeakMin: float = Field(default=180, ge=90, le=300, description="When protein effect peaks")
    proteinDurationMin: float = Field(default=300, ge=180, le=480, description="Total protein effect duration")

    # Learning metadata
    confidence: float = Field(default=0.0, ge=0, le=1, description="Overall confidence in learned values")
    insulinSampleCount: int = Field(default=0, description="Number of insulin observations")
    carbSampleCount: int = Field(default=0, description="Number of carb observations")
    proteinSampleCount: int = Field(default=0, description="Number of protein observations")

    lastUpdated: datetime = Field(default_factory=datetime.utcnow)

    # Historical observations for each type (keep last 20 each)
    insulinHistory: List[AbsorptionCurveDataPoint] = Field(default_factory=list)
    carbHistory: List[AbsorptionCurveDataPoint] = Field(default_factory=list)
    proteinHistory: List[AbsorptionCurveDataPoint] = Field(default_factory=list)

    # Time-of-day variations (morning insulin might be faster due to dawn phenomenon)
    timeOfDayAdjustments: dict = Field(
        default_factory=lambda: {
            "morning": {"insulinPeakMultiplier": 0.9, "carbPeakMultiplier": 1.0},  # Faster insulin
            "afternoon": {"insulinPeakMultiplier": 1.0, "carbPeakMultiplier": 1.0},
            "evening": {"insulinPeakMultiplier": 1.1, "carbPeakMultiplier": 1.1},  # Slower
            "night": {"insulinPeakMultiplier": 1.2, "carbPeakMultiplier": 1.0}     # Slower insulin
        },
        description="Time-of-day multipliers for peak timing"
    )


# ==================== Account Sharing Models ====================

class ShareRole(str, Enum):
    """Role for shared access."""
    VIEWER = "viewer"          # Can only view data
    CAREGIVER = "caregiver"    # Can view and add treatments
    ADMIN = "admin"            # Full access except settings


class SharePermission(str, Enum):
    """Individual permissions that can be granted."""
    VIEW_GLUCOSE = "view_glucose"
    VIEW_TREATMENTS = "view_treatments"
    VIEW_PREDICTIONS = "view_predictions"
    VIEW_INSIGHTS = "view_insights"
    ADD_TREATMENTS = "add_treatments"
    RECEIVE_ALERTS = "receive_alerts"


class AccountShare(BaseModel):
    """Sharing access between users."""
    id: str = Field(..., description="Unique share ID")
    ownerId: str = Field(..., description="User who owns the data (host)")
    sharedWithId: str = Field(..., description="User granted access (viewer)")
    sharedWithEmail: str = Field(..., description="Email of shared user")
    profileId: Optional[str] = Field(None, description="Specific profile being shared (if None, shares all profiles)")
    profileName: Optional[str] = Field(None, description="Profile display name for UI")
    role: ShareRole = Field(default=ShareRole.VIEWER)
    permissions: List[SharePermission] = Field(
        default_factory=lambda: [
            SharePermission.VIEW_GLUCOSE,
            SharePermission.VIEW_TREATMENTS,
            SharePermission.VIEW_PREDICTIONS,
            SharePermission.VIEW_INSIGHTS
        ]
    )
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    expiresAt: Optional[datetime] = Field(None, description="Optional expiration")
    isActive: bool = Field(default=True)

    class Config:
        use_enum_values = True


class ShareInvitation(BaseModel):
    """Pending invitation to share access."""
    id: str = Field(..., description="Unique invitation ID (used as token)")
    ownerId: str = Field(..., description="User sending the invitation")
    ownerEmail: str = Field(..., description="Owner's email for display")
    ownerName: Optional[str] = Field(None, description="Owner's display name")
    profileId: Optional[str] = Field(None, description="Specific profile to share (if None, shares all)")
    profileName: Optional[str] = Field(None, description="Profile display name")
    inviteeEmail: str = Field(..., description="Email to send invitation to")
    role: ShareRole = Field(default=ShareRole.VIEWER)
    permissions: List[SharePermission] = Field(default_factory=list)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    expiresAt: datetime = Field(..., description="Invitation expiration time")
    acceptedAt: Optional[datetime] = Field(None)
    isUsed: bool = Field(default=False)

    class Config:
        use_enum_values = True


# ==================== User Model Tracking ====================

class UserModelStatus(str, Enum):
    """Status of a user's trained model."""
    PENDING = "pending"        # Not enough data for training
    TRAINING = "training"      # Currently training
    ACTIVE = "active"          # Model is ready to use
    FAILED = "failed"          # Training failed


class UserModel(BaseModel):
    """Tracks ML models trained for individual users."""
    id: str = Field(..., description="Unique model ID")
    userId: str = Field(..., description="User this model belongs to")
    modelType: Literal["tft", "isf", "iob", "cob"] = Field(..., description="Type of model")
    version: int = Field(default=1, description="Model version number")
    blobPath: Optional[str] = Field(None, description="Azure Blob Storage path")
    trainedAt: Optional[datetime] = Field(None, description="When model was trained")
    metrics: dict = Field(default_factory=dict, description="Training metrics (RMSE, MAE, etc)")
    dataPoints: int = Field(default=0, description="Number of data points used for training")
    status: UserModelStatus = Field(default=UserModelStatus.PENDING)
    lastError: Optional[str] = Field(None, description="Last error message if failed")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class TrainingJob(BaseModel):
    """Training job for a user's model."""
    id: str = Field(..., description="Unique job ID")
    userId: str = Field(..., description="User to train model for")
    modelType: str = Field(..., description="Type of model to train")
    status: Literal["queued", "running", "completed", "failed"] = Field(default="queued")
    startedAt: Optional[datetime] = Field(None)
    completedAt: Optional[datetime] = Field(None)
    error: Optional[str] = Field(None)
    metrics: dict = Field(default_factory=dict)
    createdAt: datetime = Field(default_factory=datetime.utcnow)


# ==================== ML Training Data Models ====================

class MLTrainingDataPoint(BaseModel):
    """
    Clean training data point for ML model improvement.

    Captures prediction vs actual BG at +30, +60, +90 min after a meal.
    This data is used to:
    1. Learn per-food absorption profiles
    2. Train personalized absorption modifier models
    3. Detect model drift and trigger retraining

    Only "clean" meals are captured (no overlapping treatments within the window).
    """
    # Core identifiers
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique data point ID")
    userId: str = Field(..., description="User ID for partitioning")
    treatmentId: str = Field(..., description="Source treatment ID")
    timestamp: datetime = Field(..., description="When the meal was logged")

    # Food data
    foodId: str = Field(..., description="Hash of normalized food description for grouping")
    foodDescription: str = Field(..., description="Original food description")
    carbs: float = Field(..., description="Carbs in grams")
    protein: float = Field(default=0, description="Protein in grams")
    fat: float = Field(default=0, description="Fat in grams")
    fiber: float = Field(default=0, description="Fiber in grams")
    glycemicIndex: int = Field(default=55, description="Glycemic index (0-100)")
    isLiquid: bool = Field(default=False, description="True for drinks")

    # Predictions made at meal time
    predictedOnsetMin: float = Field(..., description="Predicted onset delay in minutes")
    predictedHalfLifeMin: float = Field(..., description="Predicted absorption half-life")
    predictedBg30: float = Field(..., description="Predicted BG at +30 min")
    predictedBg60: float = Field(..., description="Predicted BG at +60 min")
    predictedBg90: float = Field(..., description="Predicted BG at +90 min")

    # Actuals (filled in after data collection at each checkpoint)
    actualBg30: Optional[float] = Field(None, description="Actual BG at +30 min")
    actualBg60: Optional[float] = Field(None, description="Actual BG at +60 min")
    actualBg90: Optional[float] = Field(None, description="Actual BG at +90 min")

    # Prediction errors (computed when actuals are filled)
    error30: Optional[float] = Field(None, description="Prediction error at +30 min (actual - predicted)")
    error60: Optional[float] = Field(None, description="Prediction error at +60 min")
    error90: Optional[float] = Field(None, description="Prediction error at +90 min")

    # Context features for future ML models
    hourOfDay: int = Field(..., ge=0, le=23, description="Hour of meal (0-23)")
    dayOfWeek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    dayOfYear: int = Field(..., ge=1, le=366, description="Day of year for seasonality")
    lunarPhaseSin: float = Field(default=0.0, description="Sine component of lunar phase")
    lunarPhaseCos: float = Field(default=0.0, description="Cosine component of lunar phase")

    # Weather context (optional - use when available)
    weatherTempC: Optional[float] = Field(None, description="Temperature in Celsius")
    weatherHumidity: Optional[float] = Field(None, description="Humidity percentage")
    weatherPressure: Optional[float] = Field(None, description="Atmospheric pressure (hPa)")

    # Activity context (optional - use when available)
    activityLevel: Optional[float] = Field(None, ge=0, le=1, description="Activity level 0-1")

    # Metabolic context at meal time
    bgAtMeal: float = Field(..., description="BG at time of meal")
    bgTrend: int = Field(default=0, description="BG trend direction (-3 to +3)")
    iobAtMeal: float = Field(default=0, description="IOB at meal time")
    cobAtMeal: float = Field(default=0, description="COB from previous meals")
    minutesSinceLastMeal: Optional[int] = Field(None, description="Minutes since last carb intake")

    # Quality flags
    isCleanMeal: bool = Field(default=True, description="No overlapping treatments in window")
    dataQualityScore: float = Field(default=1.0, ge=0, le=1, description="Quality score (1.0 = perfect)")

    # Collection status
    collectedAt30: Optional[datetime] = Field(None, description="When +30 min data was collected")
    collectedAt60: Optional[datetime] = Field(None, description="When +60 min data was collected")
    collectedAt90: Optional[datetime] = Field(None, description="When +90 min data was collected")
    isComplete: bool = Field(default=False, description="All checkpoints collected")

    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class FoodAbsorptionProfile(BaseModel):
    """
    Learned absorption profile for a specific food.

    Built from MLTrainingDataPoint observations for the same foodId.
    Used to provide personalized predictions for foods the user has eaten before.
    """
    id: str = Field(..., description="Format: {userId}_{foodId}")
    userId: str = Field(..., description="User ID")
    foodId: str = Field(..., description="Hash of normalized food description")
    foodDescription: str = Field(..., description="Representative food description")

    # Learned multipliers relative to baseline formula
    onsetMultiplier: float = Field(default=1.0, description="Learned onset adjustment (1.0 = no change)")
    halfLifeMultiplier: float = Field(default=1.0, description="Learned half-life adjustment")

    # Statistics
    sampleCount: int = Field(default=0, description="Number of observations")
    meanError: float = Field(default=0, description="Mean prediction error (mg/dL)")
    stdError: float = Field(default=0, description="Standard deviation of error")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Confidence based on samples/consistency")

    # Source data
    source: Literal["learned", "similar", "gpt_estimated", "formula"] = Field(
        default="formula",
        description="How this profile was derived"
    )

    lastUpdated: datetime = Field(default_factory=datetime.utcnow)
    createdAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# ==================== Multi-Profile Management Models ====================

class ProfileRelationship(str, Enum):
    """Relationship of profile to account holder."""
    SELF = "self"          # User's own diabetes data
    CHILD = "child"        # Child's diabetes data
    SPOUSE = "spouse"      # Spouse/partner's diabetes data
    PARENT = "parent"      # Parent's diabetes data
    OTHER = "other"        # Other relationship


class DiabetesType(str, Enum):
    """Type of diabetes."""
    T1D = "T1D"
    T2D = "T2D"
    LADA = "LADA"
    GESTATIONAL = "gestational"
    OTHER = "other"


class DataSourceType(str, Enum):
    """Type of data source."""
    GLUROO = "gluroo"
    DEXCOM = "dexcom"
    NIGHTSCOUT = "nightscout"
    TIDEPOOL = "tidepool"
    MANUAL = "manual"


class SyncStatus(str, Enum):
    """Status of data source synchronization."""
    OK = "ok"
    ERROR = "error"
    PENDING = "pending"
    DISABLED = "disabled"


class ProfileDataSource(BaseModel):
    """
    A data source connected to a profile.

    Each profile can have multiple data sources (e.g., Gluroo for treatments
    AND Dexcom for CGM). This allows flexible data collection from multiple
    sources with configurable priority for conflict resolution.
    """
    id: str = Field(..., description="Format: {profileId}_{sourceType}")
    profileId: str = Field(..., description="Profile this source belongs to")
    sourceType: DataSourceType = Field(..., description="Type of data source")

    # Encrypted credentials - specific structure depends on sourceType
    # For Gluroo: {"url": "...", "apiSecret": "..."}
    # For Dexcom: {"accessToken": "...", "refreshToken": "..."}
    credentialsEncrypted: str = Field(..., description="Encrypted JSON credentials")

    # Sync configuration
    isActive: bool = Field(default=True, description="Whether this source is enabled")
    syncEnabled: bool = Field(default=True, description="Whether auto-sync is enabled")
    lastSyncAt: Optional[datetime] = Field(None, description="Last successful sync time")
    syncStatus: SyncStatus = Field(default=SyncStatus.PENDING)
    syncErrorMessage: Optional[str] = Field(None, description="Last error message if sync failed")

    # Priority for conflict resolution (lower = higher priority)
    priority: int = Field(default=1, ge=1, le=10, description="Priority for conflict resolution")

    # What data types this source provides
    providesGlucose: bool = Field(default=True, description="Source provides CGM data")
    providesTreatments: bool = Field(default=True, description="Source provides treatment data")

    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class ProfileSettings(BaseModel):
    """
    Settings specific to a managed profile.

    These override the account-level settings for this specific profile,
    allowing different targets, alert thresholds, etc. for each person.
    """
    # Target glucose ranges
    targetBgLow: int = Field(default=70, ge=50, le=100, description="Low target BG")
    targetBgHigh: int = Field(default=180, ge=100, le=250, description="High target BG")
    targetBg: int = Field(default=100, ge=70, le=150, description="Target BG for corrections")

    # Alert thresholds
    highBgThreshold: int = Field(default=180, description="High alert threshold")
    lowBgThreshold: int = Field(default=70, description="Low alert threshold")
    criticalHighThreshold: int = Field(default=250, description="Urgent high threshold")
    criticalLowThreshold: int = Field(default=54, description="Urgent low threshold")

    # Personalized insulin parameters (learned or manually set)
    insulinHalfLifeMinutes: float = Field(default=54.0, description="Insulin half-life (child ~54, adult ~81)")
    insulinActionDurationMinutes: int = Field(default=300, description="Total insulin action duration")
    carbHalfLifeMinutes: float = Field(default=45.0, description="Carb absorption half-life")
    carbAbsorptionDurationMinutes: int = Field(default=180, description="Total carb absorption duration")

    # Default ratios (can be overridden by learned values)
    defaultIsf: float = Field(default=50.0, description="Default ISF if not learned")
    defaultIcr: float = Field(default=10.0, description="Default ICR if not learned")
    defaultPir: float = Field(default=20.0, description="Default PIR if not learned")

    # Alert preferences
    enableAlerts: bool = Field(default=True)
    enablePushNotifications: bool = Field(default=True)
    quietHoursStart: Optional[str] = Field(None, description="Quiet hours start (HH:MM)")
    quietHoursEnd: Optional[str] = Field(None, description="Quiet hours end (HH:MM)")


class ManagedProfile(BaseModel):
    """
    A person whose diabetes data is managed by an account.

    This enables one account (e.g., a parent) to manage multiple people's
    diabetes data (e.g., self + two children). Each profile has its own:
    - Data sources (Gluroo, Dexcom, etc.)
    - Settings (targets, alert thresholds)
    - Learned parameters (ISF, ICR, PIR)
    - Historical data (glucose, treatments, predictions)

    The accountId serves as the partition key for CosmosDB, ensuring all
    profiles for an account are stored together for efficient queries.
    """
    id: str = Field(..., description="Unique profile ID (UUID)")
    accountId: str = Field(..., description="Account that manages this profile (partition key)")

    # Profile identity
    displayName: str = Field(..., min_length=1, max_length=100, description="Display name for this profile")
    relationship: ProfileRelationship = Field(..., description="Relationship to account holder")
    avatarUrl: Optional[str] = Field(None, description="Profile avatar URL")

    # Medical information
    diabetesType: DiabetesType = Field(default=DiabetesType.T1D)
    dateOfBirth: Optional[datetime] = Field(None, description="Date of birth")
    diagnosisDate: Optional[datetime] = Field(None, description="Date of T1D diagnosis")

    # Settings specific to this profile
    settings: ProfileSettings = Field(default_factory=ProfileSettings)

    # Data sources - references to ProfileDataSource documents
    # Note: Full source data stored in separate ProfileDataSource documents
    dataSourceIds: List[str] = Field(default_factory=list, description="IDs of connected data sources")

    # Primary sources for each data type (for conflict resolution)
    primaryGlucoseSourceId: Optional[str] = Field(None, description="Primary source for CGM data")
    primaryTreatmentSourceId: Optional[str] = Field(None, description="Primary source for treatments")

    # Status
    isActive: bool = Field(default=True, description="Whether profile is active")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    lastDataAt: Optional[datetime] = Field(None, description="Last time data was received for this profile")

    class Config:
        use_enum_values = True


class ManagedProfileCreate(BaseModel):
    """Request model for creating a new managed profile."""
    displayName: str = Field(..., min_length=1, max_length=100)
    relationship: ProfileRelationship
    diabetesType: DiabetesType = Field(default=DiabetesType.T1D)
    dateOfBirth: Optional[datetime] = None
    diagnosisDate: Optional[datetime] = None
    settings: Optional[ProfileSettings] = None


class ManagedProfileUpdate(BaseModel):
    """Request model for updating a managed profile."""
    displayName: Optional[str] = Field(None, min_length=1, max_length=100)
    relationship: Optional[ProfileRelationship] = None
    avatarUrl: Optional[str] = None
    diabetesType: Optional[DiabetesType] = None
    dateOfBirth: Optional[datetime] = None
    diagnosisDate: Optional[datetime] = None
    settings: Optional[ProfileSettings] = None
    isActive: Optional[bool] = None
    primaryGlucoseSourceId: Optional[str] = None
    primaryTreatmentSourceId: Optional[str] = None


class ProfileDataSourceCreate(BaseModel):
    """Request model for adding a data source to a profile."""
    profileId: str = Field(..., description="Profile to add source to")
    sourceType: DataSourceType = Field(..., description="Type of data source")

    # Source-specific credentials (will be encrypted before storage)
    # For Gluroo: url and apiSecret
    # For Dexcom: handled via OAuth flow
    credentials: dict = Field(..., description="Source-specific credentials")

    priority: int = Field(default=1, ge=1, le=10)
    providesGlucose: bool = Field(default=True)
    providesTreatments: bool = Field(default=True)


class ProfileDataSourceUpdate(BaseModel):
    """Request model for updating a data source."""
    isActive: Optional[bool] = None
    syncEnabled: Optional[bool] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    providesGlucose: Optional[bool] = None
    providesTreatments: Optional[bool] = None
    # Credentials can be updated separately via dedicated endpoint


class ProfileSummary(BaseModel):
    """Summary of a profile for display in profile selector."""
    id: str
    displayName: str
    relationship: ProfileRelationship
    avatarUrl: Optional[str]
    diabetesType: DiabetesType
    isActive: bool
    lastDataAt: Optional[datetime]
    dataSourceCount: int = Field(default=0, description="Number of connected data sources")
    syncStatus: SyncStatus = Field(default=SyncStatus.PENDING, description="Overall sync status")

    class Config:
        use_enum_values = True