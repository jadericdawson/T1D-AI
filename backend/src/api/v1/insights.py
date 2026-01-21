"""
Insights API Endpoints
AI-generated insights and pattern analysis.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Literal

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from database.repositories import GlucoseRepository, TreatmentRepository, InsightRepository
from services.insight_service import insight_service
from models.schemas import AIInsight, User
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["insights"])

# Repository instances
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
insight_repo = InsightRepository()


# Response Models
class InsightResponse(BaseModel):
    """AI insight response."""
    id: str
    content: str
    category: str
    createdAt: datetime
    expiresAt: Optional[datetime] = None


class InsightsListResponse(BaseModel):
    """List of insights response."""
    insights: List[InsightResponse]
    totalCount: int
    hasMore: bool


class PatternResponse(BaseModel):
    """Detected pattern response."""
    type: str
    description: str
    timeOfDay: Optional[str] = None
    dayOfWeek: Optional[str] = None
    avgValue: Optional[float] = None
    frequency: str
    confidence: float
    recommendation: Optional[str] = None


class MealImpactResponse(BaseModel):
    """Meal impact analysis response."""
    avgBgRise: float = Field(..., description="Average BG rise after meals")
    avgPeakTime: int = Field(..., description="Average time to peak in minutes")
    mealsAnalyzed: int
    problematicMeals: List[dict]
    recommendations: List[str]
    mealDetails: Optional[List[dict]] = None
    periodDays: int


class AnomalyResponse(BaseModel):
    """Anomaly detection response."""
    type: str
    severity: Literal["info", "warning", "critical"]
    value: float
    expectedRange: tuple
    timestamp: datetime
    context: Optional[str] = None


class WeeklySummaryResponse(BaseModel):
    """Weekly summary response."""
    stats: dict
    comparison: dict
    summary: dict
    generatedAt: str


class GenerateInsightsResponse(BaseModel):
    """Response for insight generation."""
    cached: bool
    insights: List[dict]
    insightCount: Optional[int] = None
    generatedAt: Optional[str] = None
    lastGenerated: Optional[str] = None
    nextRefresh: Optional[str] = None
    dataPoints: Optional[int] = None


# Endpoints
@router.get("/", response_model=InsightsListResponse)
async def get_insights(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """
    Get AI-generated insights for the user.

    Categories:
    - pattern: Detected glucose patterns
    - recommendation: Suggested adjustments
    - warning: Alerts about concerning trends
    - achievement: Positive feedback
    """
    user_id = current_user.id
    try:
        insights = await insight_repo.get_by_user(
            user_id=user_id,
            category=category,
            limit=limit + 1,  # Get one extra to check hasMore
            offset=offset
        )

        has_more = len(insights) > limit
        if has_more:
            insights = insights[:limit]

        return InsightsListResponse(
            insights=[
                InsightResponse(
                    id=i.id,
                    content=i.content,
                    category=i.category,
                    createdAt=i.createdAt,
                    expiresAt=i.expiresAt
                ) for i in insights
            ],
            totalCount=len(insights),
            hasMore=has_more
        )

    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_patterns(
    days: int = Query(default=14, ge=7, le=90),
    current_user: User = Depends(get_current_user)
):
    """
    Detect glucose patterns over the specified period.

    Analyzes:
    - Time-of-day patterns
    - Day-of-week patterns
    - Dawn phenomenon
    - Nocturnal hypoglycemia
    - Glucose variability
    """
    user_id = current_user.id
    try:
        result = await insight_service.detect_patterns(user_id, days)
        return result

    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meal-impact", response_model=MealImpactResponse)
async def get_meal_impact(
    days: int = Query(default=14, ge=7, le=90),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze the impact of meals on blood glucose.

    Examines:
    - Average BG rise after meals
    - Time to peak
    - Problematic meal types
    - Personalized recommendations
    """
    user_id = current_user.id
    try:
        result = await insight_service.analyze_meal_impact(user_id, days)

        # Handle insufficient data case
        if "message" in result and "avgBgRise" not in result:
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Insufficient data for meal analysis")
            )

        return MealImpactResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing meal impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_anomalies(
    hours: int = Query(default=24, ge=1, le=168),
    current_user: User = Depends(get_current_user)
):
    """
    Detect anomalies in recent glucose data.

    Detects:
    - Critical highs/lows
    - Rapid changes
    - Compression lows
    - Unusual patterns
    """
    user_id = current_user.id
    try:
        anomalies = await insight_service.detect_anomalies(user_id, hours)

        return {
            "anomalies": [
                {
                    "type": a.type,
                    "severity": a.severity,
                    "value": a.value,
                    "expectedRange": a.expected_range,
                    "timestamp": a.timestamp.isoformat(),
                    "context": a.context
                }
                for a in anomalies
            ],
            "count": len(anomalies),
            "periodHours": hours,
            "analyzedAt": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=GenerateInsightsResponse)
async def generate_insights(
    force: bool = Query(default=False, description="Force regeneration"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """
    Generate new AI insights for the user.

    Uses GPT-4.1 to analyze glucose patterns and provide personalized recommendations.
    Insights are cached for 1 hour.
    """
    user_id = current_user.id
    try:
        result = await insight_service.generate_insights(user_id, force)
        return GenerateInsightsResponse(**result)

    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly-summary", response_model=WeeklySummaryResponse)
async def get_weekly_summary(
    current_user: User = Depends(get_current_user)
):
    """
    Get a weekly summary with GPT-powered analysis.

    Includes:
    - Time in range statistics
    - Week-over-week comparison
    - AI-generated summary and recommendations
    """
    user_id = current_user.id
    try:
        result = await insight_service.get_weekly_summary(user_id)

        if "message" in result and "stats" not in result:
            raise HTTPException(status_code=400, detail=result["message"])

        return WeeklySummaryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating weekly summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/expired")
async def cleanup_expired_insights(
    current_user: User = Depends(get_current_user)
):
    """
    Clean up expired insights for a user.
    """
    user_id = current_user.id
    try:
        deleted_count = await insight_repo.delete_expired(user_id)
        return {
            "message": f"Cleaned up {deleted_count} expired insights",
            "deletedCount": deleted_count
        }

    except Exception as e:
        logger.error(f"Error cleaning up insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RealtimeInsightRequest(BaseModel):
    """Request for real-time AI insight."""
    currentBg: float = Field(..., description="Current blood glucose")
    trend: str = Field(default="Flat", description="Current trend")
    iob: float = Field(default=0, description="Insulin on board")
    cob: float = Field(default=0, description="Carbs on board")
    recentFood: Optional[str] = Field(default=None, description="Recent food eaten")
    recentInsulin: Optional[float] = Field(default=None, description="Recent insulin dose")
    # All diabetes metrics for comprehensive AI advice
    isf: Optional[float] = Field(default=None, description="Insulin Sensitivity Factor")
    icr: Optional[float] = Field(default=None, description="Insulin to Carb Ratio")
    pir: Optional[float] = Field(default=None, description="Protein to Insulin Ratio")
    dose: Optional[float] = Field(default=None, description="Recommended correction dose")
    bgPressure: Optional[float] = Field(default=None, description="BG Pressure - where BG is heading")
    tftPredictions: Optional[list] = Field(default=None, description="TFT predictions with confidence")
    recentGI: Optional[float] = Field(default=None, description="GI of recent food")
    absorptionRate: Optional[str] = Field(default=None, description="Absorption rate of recent food")


class RealtimeInsightResponse(BaseModel):
    """Response with real-time AI insight."""
    insight: str
    urgency: str  # low, normal, high, critical
    action: Optional[str] = None
    reasoning: str
    generatedAt: str


@router.post("/realtime", response_model=RealtimeInsightResponse)
async def get_realtime_insight(
    request: RealtimeInsightRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get real-time AI insight based on current diabetes state.

    Uses GPT to analyze:
    - Current BG and trend
    - Active IOB/COB
    - ML predictions
    - Recent food (fat, protein, carbs all affect BG differently)
    - Recent insulin

    Returns immediate, actionable advice.
    """
    user_id = current_user.id
    try:
        result = await insight_service.get_realtime_insight(
            user_id=user_id,
            current_bg=request.currentBg,
            trend=request.trend,
            iob=request.iob,
            cob=request.cob,
            predictions={},  # Deprecated - using TFT predictions
            recent_food=request.recentFood,
            recent_insulin=request.recentInsulin,
            # All diabetes metrics for comprehensive AI advice
            isf=request.isf,
            icr=request.icr,
            pir=request.pir,
            dose=request.dose,
            bg_pressure=request.bgPressure,
            tft_predictions=request.tftPredictions,
            recent_gi=request.recentGI,
            absorption_rate=request.absorptionRate
        )

        return RealtimeInsightResponse(
            insight=result.get("insight", ""),
            urgency=result.get("urgency", "normal"),
            action=result.get("action"),
            reasoning=result.get("reasoning", ""),
            generatedAt=result.get("generatedAt", datetime.utcnow().isoformat())
        )

    except Exception as e:
        logger.error(f"Error getting real-time insight: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    """Request for AI chat/what-if questions."""
    question: str = Field(..., description="User's question in natural language")
    # All diabetes metrics for accurate calculations
    currentBg: Optional[float] = Field(default=None, description="Current blood glucose")
    trend: Optional[str] = Field(default=None, description="Current trend")
    iob: Optional[float] = Field(default=None, description="Insulin on board")
    cob: Optional[float] = Field(default=None, description="Carbs on board")
    isf: Optional[float] = Field(default=None, description="Insulin Sensitivity Factor")
    icr: Optional[float] = Field(default=None, description="Insulin to Carb Ratio")
    pir: Optional[float] = Field(default=None, description="Protein to Insulin Ratio")
    dose: Optional[float] = Field(default=None, description="Current recommended correction dose")
    bgPressure: Optional[float] = Field(default=None, description="BG Pressure")
    tftPredictions: Optional[list] = Field(default=None, description="TFT predictions")


class ChatResponse(BaseModel):
    """Response from AI chat."""
    response: str
    prediction: Optional[dict] = None
    recommendation: Optional[dict] = None
    calculation: Optional[str] = None
    confidence: str
    generatedAt: str


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Chat with AI for "what-if" scenario questions.

    Allows users to ask questions like:
    - "What will happen if I eat 37 carbs at 11:30am?"
    - "How much insulin do I need for a 50g meal?"
    - "Should I take a correction now or wait?"

    The AI uses current diabetes state (BG, IOB, COB, ISF, etc.)
    to provide specific, calculated answers.
    """
    try:
        # Build context from request - AI gets access to all current data
        context = {
            "currentBg": request.currentBg or 120,
            "trend": request.trend or "Flat",
            "iob": request.iob or 0,
            "cob": request.cob or 0,
            "isf": request.isf or 50,
            "icr": request.icr or 10,
            "pir": request.pir or 14,
            "dose": request.dose or 0,
            "bgPressure": request.bgPressure or 0,
            "tftPredictions": request.tftPredictions or [],
            "currentTime": datetime.utcnow().strftime("%H:%M"),
        }

        # Call OpenAI service for chat response
        from services.openai_service import openai_service
        result = await openai_service.chat_what_if(
            question=request.question,
            context=context
        )

        return ChatResponse(
            response=result.get("response", ""),
            prediction=result.get("prediction"),
            recommendation=result.get("recommendation"),
            calculation=result.get("calculation"),
            confidence=result.get("confidence", "medium"),
            generatedAt=result.get("generatedAt", datetime.utcnow().isoformat())
        )

    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_insight_categories():
    """
    Get available insight categories with descriptions.
    """
    return {
        "categories": [
            {
                "id": "pattern",
                "name": "Patterns",
                "description": "Detected glucose patterns and trends",
                "icon": "trending-up"
            },
            {
                "id": "recommendation",
                "name": "Recommendations",
                "description": "Actionable suggestions for improvement",
                "icon": "lightbulb"
            },
            {
                "id": "warning",
                "name": "Warnings",
                "description": "Alerts about concerning trends",
                "icon": "alert-triangle"
            },
            {
                "id": "achievement",
                "name": "Achievements",
                "description": "Positive feedback and milestones",
                "icon": "trophy"
            }
        ]
    }
