"""
Insight Generation Service
Coordinates AI insight generation and pattern detection.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import statistics

from database.repositories import GlucoseRepository, TreatmentRepository, InsightRepository, UserRepository, MLTrainingDataRepository
from services.openai_service import openai_service
from models.schemas import AIInsight
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Repositories
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
insight_repo = InsightRepository()
user_repo = UserRepository()
ml_training_repo = MLTrainingDataRepository()


@dataclass
class AnomalyDetection:
    """Detected anomaly in glucose data."""
    type: str
    severity: str  # info, warning, critical
    value: float
    expected_range: tuple
    timestamp: datetime
    context: str


class InsightService:
    """Service for generating and managing AI insights."""

    def __init__(self):
        # TTL cache for realtime insights: {cache_key: (monotonic_ts, result)}
        self._realtime_cache: Dict[str, tuple] = {}
        self._realtime_cache_ttl = 300  # 5 minutes

    def _realtime_cache_key(self, user_id: str, current_bg: float, trend: str) -> str:
        """Cache key from user + rounded BG (nearest 5 mg/dL) + trend."""
        rounded_bg = round(current_bg / 5) * 5
        return f"{user_id}:{rounded_bg}:{trend}"

    def _clean_realtime_cache(self):
        """Remove expired entries."""
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._realtime_cache.items()
                   if now - ts > self._realtime_cache_ttl]
        for k in expired:
            del self._realtime_cache[k]

    async def generate_insights(
        self,
        user_id: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Generate AI insights for a user.

        Args:
            user_id: User identifier
            force: Force regeneration even if recent insights exist

        Returns:
            Generated insights and metadata
        """
        # Check for recent insights
        if not force:
            recent = await insight_repo.get_by_user(user_id, limit=1)
            if recent:
                age_seconds = (datetime.utcnow() - recent[0].createdAt).total_seconds()
                if age_seconds < 3600:  # 1 hour cache
                    return {
                        "cached": True,
                        "insights": [{"content": i.content, "category": i.category} for i in recent[:5]],
                        "lastGenerated": recent[0].createdAt.isoformat(),
                        "nextRefresh": (recent[0].createdAt + timedelta(hours=1)).isoformat()
                    }

        try:
            # Gather data for analysis
            start_time = datetime.utcnow() - timedelta(hours=24)

            # Get glucose readings
            readings = await glucose_repo.get_history(user_id, start_time)
            glucose_data = [
                {"value": r.value, "trend": r.trend, "timestamp": r.timestamp.isoformat()}
                for r in readings
            ]

            # Get treatments
            treatments = await treatment_repo.get_by_user(user_id, start_time)
            treatment_data = [
                {
                    "type": t.type,
                    "insulin": t.insulin,
                    "carbs": t.carbs,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in treatments
            ]

            # Get user settings
            user = await user_repo.get_by_id(user_id)
            user_settings = user.settings.model_dump() if user and user.settings else {
                "targetBg": settings.target_bg,
                "highThreshold": settings.high_bg_threshold,
                "lowThreshold": settings.low_bg_threshold,
                "criticalHighThreshold": settings.critical_high_threshold,
                "criticalLowThreshold": settings.critical_low_threshold,
            }

            # Detect patterns (rule-based)
            patterns = await self.detect_patterns(user_id, days=7)

            # Generate AI insights
            insights = await openai_service.generate_glucose_insights(
                glucose_data=glucose_data,
                treatments=treatment_data,
                patterns=patterns,
                user_settings=user_settings
            )

            # Store insights
            stored_insights = []
            for insight in insights:
                ai_insight = AIInsight(
                    userId=user_id,
                    content=insight["content"],
                    category=insight.get("category", "recommendation"),
                    createdAt=datetime.utcnow(),
                    expiresAt=datetime.utcnow() + timedelta(days=7)
                )
                stored = await insight_repo.create(ai_insight)
                stored_insights.append(stored)

            return {
                "cached": False,
                "insights": insights,
                "insightCount": len(insights),
                "generatedAt": datetime.utcnow().isoformat(),
                "dataPoints": len(glucose_data)
            }

        except Exception as e:
            logger.error(f"Error generating insights for user {user_id}: {e}")
            raise

    async def detect_patterns(
        self,
        user_id: str,
        days: int = 14
    ) -> Dict[str, Any]:
        """
        Detect glucose patterns using rule-based analysis.

        Detects:
        - Time-of-day patterns
        - Dawn phenomenon
        - Post-meal spikes
        - Nocturnal hypoglycemia
        """
        start_time = datetime.utcnow() - timedelta(days=days)
        readings = await glucose_repo.get_history(user_id, start_time)

        if len(readings) < 50:
            return {
                "patterns": [],
                "message": "Insufficient data for pattern detection",
                "readingsAnalyzed": len(readings)
            }

        patterns = []

        # Group by hour of day
        hourly_data = {}
        for r in readings:
            hour = r.timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(r.value)

        # Detect time-of-day patterns
        for hour, values in hourly_data.items():
            if len(values) < 5:
                continue

            avg = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0

            time_str = f"{hour:02d}:00"

            if avg > 180:
                patterns.append({
                    "type": "high_time",
                    "description": f"Consistently elevated glucose around {time_str}",
                    "timeOfDay": time_str,
                    "avgValue": round(avg, 0),
                    "frequency": f"{len(values)} readings",
                    "confidence": min(len(values) / 10, 1.0),
                    "recommendation": f"Consider reviewing insulin timing around {time_str}"
                })
            elif avg < 80:
                patterns.append({
                    "type": "low_time",
                    "description": f"Glucose tends to run low around {time_str}",
                    "timeOfDay": time_str,
                    "avgValue": round(avg, 0),
                    "frequency": f"{len(values)} readings",
                    "confidence": min(len(values) / 10, 1.0),
                    "recommendation": f"Consider reducing insulin or adding a snack around {time_str}"
                })

        # Dawn phenomenon (4-7 AM rise)
        early_morning = [r.value for r in readings if 4 <= r.timestamp.hour < 7]
        late_night = [r.value for r in readings if 1 <= r.timestamp.hour < 4]

        if early_morning and late_night:
            em_avg = statistics.mean(early_morning)
            ln_avg = statistics.mean(late_night)

            if em_avg - ln_avg > 30 and em_avg > 140:
                patterns.append({
                    "type": "dawn_phenomenon",
                    "description": "Blood glucose rises significantly in early morning hours",
                    "timeOfDay": "04:00-07:00",
                    "avgRise": round(em_avg - ln_avg, 0),
                    "frequency": "Daily pattern",
                    "confidence": 0.8,
                    "recommendation": "Consider adjusting nighttime basal insulin timing or dose"
                })

        # Nocturnal hypoglycemia (12-4 AM lows)
        night_readings = [r for r in readings if 0 <= r.timestamp.hour < 4]
        night_lows = [r for r in night_readings if r.value < 70]

        if len(night_lows) >= 3:
            patterns.append({
                "type": "nocturnal_hypoglycemia",
                "description": f"Low blood glucose detected during sleep ({len(night_lows)} episodes)",
                "timeOfDay": "00:00-04:00",
                "frequency": f"{len(night_lows)} episodes in {days} days",
                "confidence": 0.9,
                "recommendation": "Consider reducing evening basal or having a bedtime snack"
            })

        # High variability detection
        all_values = [r.value for r in readings]
        if len(all_values) > 20:
            cv = (statistics.stdev(all_values) / statistics.mean(all_values)) * 100
            if cv > 36:  # >36% CV indicates high variability
                patterns.append({
                    "type": "high_variability",
                    "description": f"Glucose variability is elevated (CV: {cv:.0f}%)",
                    "avgValue": round(statistics.mean(all_values), 0),
                    "stdDev": round(statistics.stdev(all_values), 0),
                    "frequency": "Ongoing pattern",
                    "confidence": 0.85,
                    "recommendation": "Focus on consistent meal timing and carb amounts"
                })

        return {
            "patterns": patterns,
            "readingsAnalyzed": len(readings),
            "periodDays": days,
            "generatedAt": datetime.utcnow().isoformat()
        }

    async def detect_anomalies(
        self,
        user_id: str,
        hours: int = 24
    ) -> List[AnomalyDetection]:
        """
        Detect anomalies in recent glucose data.

        Detects:
        - Critical values
        - Rapid changes
        - Compression lows (false lows)
        - Sensor noise
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        readings = await glucose_repo.get_history(user_id, start_time)

        if len(readings) < 5:
            return []

        anomalies = []
        sorted_readings = sorted(readings, key=lambda x: x.timestamp)

        for i, r in enumerate(sorted_readings):
            # Critical lows
            if r.value < settings.critical_low_threshold:
                anomalies.append(AnomalyDetection(
                    type="critical_low",
                    severity="critical",
                    value=r.value,
                    expected_range=(70, 180),
                    timestamp=r.timestamp,
                    context=f"Blood glucose {r.value} mg/dL is dangerously low. Treat immediately with fast-acting glucose."
                ))

            # Critical highs
            elif r.value > settings.critical_high_threshold + 50:  # >300
                anomalies.append(AnomalyDetection(
                    type="critical_high",
                    severity="critical",
                    value=r.value,
                    expected_range=(70, 180),
                    timestamp=r.timestamp,
                    context=f"Blood glucose {r.value} mg/dL is very high. Check for ketones and consider correction."
                ))

            # Rapid changes
            if i > 0:
                prev = sorted_readings[i - 1]
                time_diff_minutes = (r.timestamp - prev.timestamp).total_seconds() / 60

                if time_diff_minutes > 0 and time_diff_minutes <= 10:
                    rate = (r.value - prev.value) / time_diff_minutes

                    if abs(rate) > 3:  # >3 mg/dL per minute
                        direction = "rising" if rate > 0 else "falling"
                        severity = "critical" if abs(rate) > 5 else "warning"

                        anomalies.append(AnomalyDetection(
                            type="rapid_change",
                            severity=severity,
                            value=abs(rate),
                            expected_range=(-2, 2),
                            timestamp=r.timestamp,
                            context=f"Glucose is {direction} rapidly at {abs(rate):.1f} mg/dL per minute"
                        ))

            # Potential compression low (sudden drop followed by quick recovery at night)
            if i >= 2 and 0 <= r.timestamp.hour < 6:
                prev2 = sorted_readings[i - 2]
                prev1 = sorted_readings[i - 1]

                # Pattern: normal -> sudden low -> quick recovery
                if prev2.value > 100 and prev1.value < 60 and r.value > 90:
                    anomalies.append(AnomalyDetection(
                        type="compression_low",
                        severity="info",
                        value=prev1.value,
                        expected_range=(70, 180),
                        timestamp=prev1.timestamp,
                        context="Possible compression low detected - glucose drop may be from sleeping on sensor"
                    ))

        # Remove duplicates within 30-minute windows
        filtered = []
        for anomaly in anomalies:
            is_duplicate = False
            for existing in filtered:
                if (existing.type == anomaly.type and
                    abs((existing.timestamp - anomaly.timestamp).total_seconds()) < 1800):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(anomaly)

        return filtered[:20]  # Limit to 20 anomalies

    async def get_prediction_accuracy(
        self,
        user_id: str,
        days: int = 14
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy stats from historical data.

        Returns metrics showing how well the ML predictions have matched
        actual BG outcomes at +30, +60, +90 minute horizons.
        """
        try:
            # Get completed training data points
            data_points = await ml_training_repo.get_recent_complete(user_id, days=days, limit=200)

            if len(data_points) < 5:
                return {
                    "available": False,
                    "message": "Insufficient data for accuracy calculation",
                    "dataPoints": len(data_points),
                    "minimumRequired": 5
                }

            # Calculate accuracy stats for each horizon
            errors_30 = [dp.error30 for dp in data_points if dp.error30 is not None]
            errors_60 = [dp.error60 for dp in data_points if dp.error60 is not None]
            errors_90 = [dp.error90 for dp in data_points if dp.error90 is not None]

            def calc_stats(errors: List[float]) -> Dict[str, float]:
                if not errors:
                    return {"available": False}
                abs_errors = [abs(e) for e in errors]
                return {
                    "available": True,
                    "count": len(errors),
                    "meanError": round(statistics.mean(errors), 1),  # Bias (+ = under-predicting)
                    "meanAbsError": round(statistics.mean(abs_errors), 1),  # MAE
                    "medianAbsError": round(statistics.median(abs_errors), 1),
                    "stdError": round(statistics.stdev(errors), 1) if len(errors) > 1 else 0,
                    "within20": round(sum(1 for e in abs_errors if e <= 20) / len(errors) * 100, 0),  # % within 20 mg/dL
                    "within30": round(sum(1 for e in abs_errors if e <= 30) / len(errors) * 100, 0),  # % within 30 mg/dL
                }

            stats_30 = calc_stats(errors_30)
            stats_60 = calc_stats(errors_60)
            stats_90 = calc_stats(errors_90)

            # Overall accuracy assessment
            overall_mae = None
            if errors_30 and errors_60:
                all_abs_errors = [abs(e) for e in errors_30 + errors_60]
                overall_mae = round(statistics.mean(all_abs_errors), 1)

            # Determine accuracy quality
            accuracy_quality = "unknown"
            if overall_mae is not None:
                if overall_mae <= 15:
                    accuracy_quality = "excellent"
                elif overall_mae <= 25:
                    accuracy_quality = "good"
                elif overall_mae <= 40:
                    accuracy_quality = "moderate"
                else:
                    accuracy_quality = "needs_improvement"

            # Check for systematic bias
            bias_direction = "neutral"
            if errors_30:
                mean_bias = statistics.mean(errors_30)
                if mean_bias > 10:
                    bias_direction = "under_predicting"  # Actual higher than predicted
                elif mean_bias < -10:
                    bias_direction = "over_predicting"  # Actual lower than predicted

            return {
                "available": True,
                "dataPoints": len(data_points),
                "periodDays": days,
                "horizon30": stats_30,
                "horizon60": stats_60,
                "horizon90": stats_90,
                "overallMAE": overall_mae,
                "accuracyQuality": accuracy_quality,
                "biasDirection": bias_direction,
                "generatedAt": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.warning(f"Error calculating prediction accuracy: {e}")
            return {
                "available": False,
                "error": str(e)
            }

    async def analyze_meal_impact(
        self,
        user_id: str,
        days: int = 14
    ) -> Dict[str, Any]:
        """
        Analyze the impact of meals on blood glucose.
        """
        start_time = datetime.utcnow() - timedelta(days=days)

        # Get meals
        treatments = await treatment_repo.get_by_user(user_id, start_time)
        meals = [t for t in treatments if (t.carbs or 0) > 10]  # Only significant meals

        if len(meals) < 5:
            return {
                "message": "Not enough meal data for analysis",
                "mealsAnalyzed": len(meals),
                "minimumRequired": 5
            }

        # Get glucose readings
        readings = await glucose_repo.get_history(user_id, start_time)

        meal_impacts = []
        problematic_meals = []

        for meal in meals:
            # Get readings 30 min before to 3 hours after meal
            pre_meal = [
                r for r in readings
                if -30 <= (r.timestamp - meal.timestamp).total_seconds() / 60 <= 0
            ]
            post_meal = [
                r for r in readings
                if 0 < (r.timestamp - meal.timestamp).total_seconds() / 60 <= 180
            ]

            if not pre_meal or len(post_meal) < 3:
                continue

            pre_bg = statistics.mean([r.value for r in pre_meal])
            peak_bg = max(r.value for r in post_meal)
            rise = peak_bg - pre_bg

            # Time to peak
            peak_reading = max(post_meal, key=lambda x: x.value)
            time_to_peak = (peak_reading.timestamp - meal.timestamp).total_seconds() / 60

            meal_impacts.append({
                "timestamp": meal.timestamp.isoformat(),
                "carbs": meal.carbs,
                "insulin": meal.insulin,
                "preBg": round(pre_bg, 0),
                "peakBg": round(peak_bg, 0),
                "rise": round(rise, 0),
                "timeToPeak": round(time_to_peak, 0)
            })

            if rise > 80:  # Significant spike
                problematic_meals.append({
                    "timestamp": meal.timestamp.isoformat(),
                    "carbs": meal.carbs,
                    "insulin": meal.insulin,
                    "rise": round(rise, 0),
                    "notes": meal.notes or "",
                    "suggestion": self._get_meal_suggestion(rise, meal.carbs, meal.insulin, time_to_peak)
                })

        if not meal_impacts:
            return {
                "message": "Could not analyze meal impacts",
                "mealsFound": len(meals),
                "issue": "Insufficient glucose data around meal times"
            }

        # Calculate averages
        avg_rise = statistics.mean([m["rise"] for m in meal_impacts])
        avg_peak_time = statistics.mean([m["timeToPeak"] for m in meal_impacts])

        # Generate recommendations
        recommendations = []
        if avg_rise > 60:
            recommendations.append("Consider pre-bolusing 15-20 minutes before meals to reduce post-meal spikes")
        if avg_peak_time < 30:
            recommendations.append("Glucose peaks quickly after eating - faster-acting insulin or earlier bolusing may help")
        if len(problematic_meals) > len(meal_impacts) * 0.3:
            recommendations.append("Review carb counting accuracy - many meals cause significant spikes")

        # Use GPT for additional analysis if available
        try:
            gpt_analysis = await openai_service.generate_meal_analysis(
                meal_data={
                    "avgCarbs": statistics.mean([m["carbs"] for m in meal_impacts if m["carbs"]]),
                    "avgRise": avg_rise,
                    "problematicCount": len(problematic_meals)
                },
                post_meal_readings=[{"value": m["peakBg"]} for m in meal_impacts],
                user_settings={"lowThreshold": 70, "highThreshold": 180}
            )
            if gpt_analysis.get("suggestions"):
                recommendations.extend(gpt_analysis["suggestions"][:2])
        except Exception as e:
            logger.debug(f"GPT meal analysis unavailable: {e}")

        return {
            "avgBgRise": round(avg_rise, 0),
            "avgPeakTime": round(avg_peak_time, 0),
            "mealsAnalyzed": len(meal_impacts),
            "problematicMeals": problematic_meals[:5],
            "recommendations": recommendations,
            "mealDetails": meal_impacts[:10],  # Last 10 meals
            "periodDays": days
        }

    def _get_meal_suggestion(
        self,
        rise: float,
        carbs: Optional[float],
        insulin: Optional[float],
        time_to_peak: float
    ) -> str:
        """Generate a specific suggestion for a problematic meal."""
        if rise > 120:
            return "Significant spike - consider splitting bolus or increasing pre-bolus time"
        elif time_to_peak < 30 and rise > 80:
            return "Fast rise suggests pre-bolusing earlier would help"
        elif carbs and insulin and (carbs / insulin if insulin else 0) > 15:
            return "May need more aggressive carb ratio for this meal type"
        else:
            return "Review meal composition and timing"

    async def get_weekly_summary(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Generate a weekly summary with GPT analysis.
        """
        # Current week stats
        week_start = datetime.utcnow() - timedelta(days=7)
        readings = await glucose_repo.get_history(user_id, week_start)

        if not readings:
            return {"message": "No data for weekly summary"}

        values = [r.value for r in readings]
        avg_bg = statistics.mean(values)
        std_bg = statistics.stdev(values) if len(values) > 1 else 0

        low_count = sum(1 for v in values if v < 70)
        high_count = sum(1 for v in values if v > 180)
        in_range = sum(1 for v in values if 70 <= v <= 180)
        tir = (in_range / len(values)) * 100

        # Previous week for comparison
        prev_week_start = week_start - timedelta(days=7)
        prev_readings = await glucose_repo.get_history(user_id, prev_week_start, week_start)

        comparison = {}
        if prev_readings:
            prev_values = [r.value for r in prev_readings]
            prev_avg = statistics.mean(prev_values)
            prev_in_range = sum(1 for v in prev_values if 70 <= v <= 180)
            prev_tir = (prev_in_range / len(prev_values)) * 100

            comparison = {
                "avgBgChange": avg_bg - prev_avg,
                "tirChange": tir - prev_tir
            }

        weekly_stats = {
            "avgBg": avg_bg,
            "stdBg": std_bg,
            "timeInRange": tir,
            "totalReadings": len(values),
            "lows": low_count,
            "highs": high_count
        }

        # Generate GPT summary
        try:
            summary = await openai_service.generate_weekly_summary(weekly_stats, comparison)
        except Exception as e:
            logger.debug(f"GPT weekly summary unavailable: {e}")
            summary = {
                "summary": f"This week: {tir:.0f}% time in range with an average of {avg_bg:.0f} mg/dL.",
                "highlight": "Keep up the good monitoring!",
                "focus": "Continue reviewing patterns",
                "motivation": "Every day is progress!"
            }

        return {
            "stats": weekly_stats,
            "comparison": comparison,
            "summary": summary,
            "generatedAt": datetime.utcnow().isoformat()
        }


    async def get_realtime_insight(
        self,
        user_id: str,
        current_bg: float,
        trend: str,
        iob: float,
        cob: float,
        predictions: Dict[str, Any],
        recent_food: Optional[str] = None,
        recent_insulin: Optional[float] = None,
        # All diabetes metrics for comprehensive AI advice
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        pir: Optional[float] = None,
        dose: Optional[float] = None,
        bg_pressure: Optional[float] = None,
        tft_predictions: Optional[list] = None,
        recent_gi: Optional[float] = None,
        absorption_rate: Optional[str] = None,
        # NEW: Metabolic state context for illness/sensitivity detection
        pob: float = 0.0,
        metabolic_state: Optional[str] = None,
        isf_deviation: Optional[float] = None,
        absorption_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate real-time AI insight based on current state.

        Provides immediate actionable advice based on:
        - Current BG and trend
        - Active IOB/COB/POB
        - LEARNED metabolic parameters: ISF, ICR, PIR (with deviation tracking)
        - Metabolic state: sick, resistant, normal, sensitive
        - Absorption state: very_slow, slow, normal, fast
        - TFT predictions with confidence intervals
        - BG Pressure (where BG is heading based on IOB/COB)
        - Recent food (including GI and absorption rate)
        - Recent insulin

        Args:
            user_id: User identifier
            current_bg: Current blood glucose
            trend: Current trend direction (Flat, FortyFiveUp, etc.)
            iob: Insulin on board
            cob: Carbs on board
            predictions: Dict with linear, lstm predictions (deprecated)
            recent_food: Description of recently eaten food
            recent_insulin: Amount of recent insulin dose
            isf: Insulin Sensitivity Factor (learned value)
            icr: Insulin to Carb Ratio (learned value)
            pir: Protein to Insulin Ratio (learned value)
            dose: Currently recommended correction dose
            bg_pressure: BG Pressure - net effect of IOB/COB
            tft_predictions: TFT predictions with confidence intervals
            recent_gi: Glycemic Index of recent food
            absorption_rate: Absorption rate of recent food
            pob: Protein on board
            metabolic_state: Current metabolic state (sick/resistant/normal/sensitive)
            isf_deviation: ISF deviation percentage from baseline
            absorption_state: Absorption state (very_slow/slow/normal/fast)

        Returns:
            Real-time insight with advice
        """
        # Check realtime cache — return cached result if BG hasn't changed meaningfully
        cache_key = self._realtime_cache_key(user_id, current_bg, trend)
        now = time.monotonic()
        cached = self._realtime_cache.get(cache_key)
        if cached:
            cached_time, cached_result = cached
            if now - cached_time < self._realtime_cache_ttl:
                logger.debug(f"Realtime insight cache HIT for {user_id} (bg~{round(current_bg/5)*5}, {trend})")
                return cached_result
            else:
                del self._realtime_cache[cache_key]
        if len(self._realtime_cache) > 100:
            self._clean_realtime_cache()

        try:
            # Fetch prediction accuracy stats (non-blocking, fails gracefully)
            prediction_accuracy = None
            try:
                prediction_accuracy = await self.get_prediction_accuracy(user_id, days=14)
            except Exception as e:
                logger.debug(f"Could not fetch prediction accuracy: {e}")

            # Build context for GPT with all available data including metabolic state
            context = {
                "currentBg": current_bg,
                "trend": trend,
                "iob": iob,
                "cob": cob,
                "pob": pob,
                "recentFood": recent_food,
                "recentInsulin": recent_insulin,
                # LEARNED diabetes metrics (NOT defaults)
                "isf": isf if isf else 50,
                "icr": icr if icr else 10,
                "pir": pir if pir else 25,
                "dose": dose if dose else 0,
                "bgPressure": bg_pressure if bg_pressure else 0,
                "tftPredictions": tft_predictions or [],
                "recentGI": recent_gi,
                "absorptionRate": absorption_rate,
                # Metabolic state context for illness-aware recommendations
                "metabolicState": metabolic_state or "normal",
                "isfDeviation": isf_deviation or 0.0,
                "absorptionState": absorption_state or "normal",
                # Prediction accuracy - helps AI calibrate confidence in predictions
                "predictionAccuracy": prediction_accuracy
            }

            # Use GPT to generate real-time advice
            insight = await openai_service.generate_realtime_insight(context)

            result = {
                "insight": insight.get("message", ""),
                "urgency": insight.get("urgency", "normal"),  # low, normal, high, critical
                "action": insight.get("action"),  # Suggested action if any
                "reasoning": insight.get("reasoning", ""),
                "generatedAt": datetime.utcnow().isoformat()
            }
            # Cache for 5 min keyed on user + rounded BG + trend
            self._realtime_cache[cache_key] = (time.monotonic(), result)
            return result

        except Exception as e:
            logger.error(f"Error generating real-time insight: {e}")
            # Fallback to rule-based insight
            return self._generate_fallback_insight(current_bg, trend, iob, cob, predictions)

    def _generate_fallback_insight(
        self,
        current_bg: float,
        trend: str,
        iob: float,
        cob: float,
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate rule-based insight when GPT is unavailable."""
        message = ""
        urgency = "normal"
        action = None

        # Critical situations
        if current_bg < 70:
            message = f"Blood glucose is low ({current_bg} mg/dL). Consider fast-acting carbs."
            urgency = "high"
            action = "Eat 15-20g fast carbs"
        elif current_bg < 54:
            message = f"Blood glucose is critically low ({current_bg} mg/dL)! Treat immediately."
            urgency = "critical"
            action = "Treat with glucose tabs NOW"
        elif current_bg > 250:
            message = f"Blood glucose is high ({current_bg} mg/dL). Active IOB: {iob:.1f}U."
            urgency = "high" if iob < 1 else "normal"
            if iob < 1:
                action = "Consider correction dose"

        # Prediction-based insights
        elif predictions:
            pred_15 = predictions.get("lstm", predictions.get("linear", [None]))[2] if len(predictions.get("lstm", predictions.get("linear", []))) > 2 else None
            if pred_15:
                if pred_15 < 70:
                    message = f"Prediction shows BG dropping to {pred_15:.0f} mg/dL in 15 min."
                    urgency = "high"
                    action = "Have carbs ready"
                elif pred_15 > 200 and iob < 1:
                    message = f"BG rising toward {pred_15:.0f} mg/dL. IOB: {iob:.1f}U."
                    urgency = "normal"

        # Default message
        if not message:
            if iob > 0.5 and cob > 10:
                message = f"IOB ({iob:.1f}U) and COB ({cob:.0f}g) active. Monitoring."
            elif iob > 2:
                message = f"Significant IOB ({iob:.1f}U) still active. Watch for lows."
            elif cob > 20:
                message = f"Carbs still absorbing ({cob:.0f}g COB). BG may rise."
            else:
                message = f"Current BG: {current_bg} mg/dL. Stable."

        return {
            "insight": message,
            "urgency": urgency,
            "action": action,
            "reasoning": "Rule-based analysis",
            "generatedAt": datetime.utcnow().isoformat()
        }


# Singleton instance
insight_service = InsightService()
