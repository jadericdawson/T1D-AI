"""
Insights API Endpoints
AI-generated insights and pattern analysis.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Literal

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from database.repositories import GlucoseRepository, TreatmentRepository, InsightRepository
from services.insight_service import insight_service
from models.schemas import AIInsight

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["insights"])

# Repository instances
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
insight_repo = InsightRepository()

TEMP_USER_ID = "demo_user"


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
    user_id: str = Query(default=TEMP_USER_ID),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0)
):
    """
    Get AI-generated insights for the user.

    Categories:
    - pattern: Detected glucose patterns
    - recommendation: Suggested adjustments
    - warning: Alerts about concerning trends
    - achievement: Positive feedback
    """
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
    user_id: str = Query(default=TEMP_USER_ID),
    days: int = Query(default=14, ge=7, le=90)
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
    try:
        result = await insight_service.detect_patterns(user_id, days)
        return result

    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meal-impact", response_model=MealImpactResponse)
async def get_meal_impact(
    user_id: str = Query(default=TEMP_USER_ID),
    days: int = Query(default=14, ge=7, le=90)
):
    """
    Analyze the impact of meals on blood glucose.

    Examines:
    - Average BG rise after meals
    - Time to peak
    - Problematic meal types
    - Personalized recommendations
    """
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
    user_id: str = Query(default=TEMP_USER_ID),
    hours: int = Query(default=24, ge=1, le=168)
):
    """
    Detect anomalies in recent glucose data.

    Detects:
    - Critical highs/lows
    - Rapid changes
    - Compression lows
    - Unusual patterns
    """
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
    user_id: str = Query(default=TEMP_USER_ID),
    force: bool = Query(default=False, description="Force regeneration"),
    background_tasks: BackgroundTasks = None
):
    """
    Generate new AI insights for the user.

    Uses GPT-4.1 to analyze glucose patterns and provide personalized recommendations.
    Insights are cached for 1 hour.
    """
    try:
        result = await insight_service.generate_insights(user_id, force)
        return GenerateInsightsResponse(**result)

    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly-summary", response_model=WeeklySummaryResponse)
async def get_weekly_summary(
    user_id: str = Query(default=TEMP_USER_ID)
):
    """
    Get a weekly summary with GPT-powered analysis.

    Includes:
    - Time in range statistics
    - Week-over-week comparison
    - AI-generated summary and recommendations
    """
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
    user_id: str = Query(default=TEMP_USER_ID)
):
    """
    Clean up expired insights for a user.
    """
    try:
        deleted_count = await insight_repo.delete_expired(user_id)
        return {
            "message": f"Cleaned up {deleted_count} expired insights",
            "deletedCount": deleted_count
        }

    except Exception as e:
        logger.error(f"Error cleaning up insights: {e}")
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
