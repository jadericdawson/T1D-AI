"""
Azure OpenAI Service
Provides GPT-4.1 integration for insight generation.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from openai import AsyncAzureOpenAI
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIService:
    """Azure OpenAI client for T1D-AI insights."""

    def __init__(self):
        self.client: Optional[AsyncAzureOpenAI] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the Azure OpenAI client."""
        if self._initialized:
            return

        try:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
            )
            self._initialized = True
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    async def _ensure_initialized(self):
        """Ensure client is initialized before use."""
        if not self._initialized:
            await self.initialize()

    async def generate_glucose_insights(
        self,
        glucose_data: List[Dict[str, Any]],
        treatments: List[Dict[str, Any]],
        patterns: Dict[str, Any],
        user_settings: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate personalized glucose insights using GPT-4.

        Args:
            glucose_data: Recent glucose readings with timestamps
            treatments: Recent insulin/carb treatments
            patterns: Detected patterns from analysis
            user_settings: User's target ranges and preferences

        Returns:
            List of insights with content and category
        """
        await self._ensure_initialized()

        # Prepare context summary
        recent_bg = glucose_data[-1] if glucose_data else {}
        avg_bg = sum(r.get("value", 0) for r in glucose_data) / len(glucose_data) if glucose_data else 0
        time_in_range = self._calculate_time_in_range(glucose_data, user_settings)

        # Count highs and lows
        high_threshold = user_settings.get("highThreshold", 180)
        low_threshold = user_settings.get("lowThreshold", 70)
        highs = sum(1 for r in glucose_data if r.get("value", 0) > high_threshold)
        lows = sum(1 for r in glucose_data if r.get("value", 0) < low_threshold)

        # Recent treatments summary
        total_insulin = sum(t.get("insulin", 0) or 0 for t in treatments)
        total_carbs = sum(t.get("carbs", 0) or 0 for t in treatments)

        system_prompt = """You are a helpful diabetes management assistant for T1D-AI, a Type 1 Diabetes monitoring app.
Your role is to analyze glucose data and provide actionable, personalized insights.

Guidelines:
- Be encouraging but honest about concerning patterns
- Focus on actionable recommendations
- Use simple, non-medical language
- Keep insights concise (1-2 sentences each)
- Never provide medical dosing advice - only suggest consulting healthcare provider
- Consider the user's target ranges in your analysis
- Acknowledge good patterns as well as areas for improvement

Response format: Return a JSON array of insights, each with "content" and "category" fields.
Categories: "pattern", "recommendation", "warning", "achievement"
"""

        user_prompt = f"""Analyze this diabetes data and provide 3-5 personalized insights:

**Current Status:**
- Current BG: {recent_bg.get('value', 'N/A')} mg/dL ({recent_bg.get('trend', 'stable')})
- Average BG (last 24h): {avg_bg:.0f} mg/dL
- Time in Range: {time_in_range:.0f}%
- High readings: {highs} | Low readings: {lows}

**Recent Treatments (24h):**
- Total Insulin: {total_insulin:.1f} units
- Total Carbs: {total_carbs:.0f} grams

**User Settings:**
- Target BG: {user_settings.get('targetBg', 100)} mg/dL
- High Threshold: {high_threshold} mg/dL
- Low Threshold: {low_threshold} mg/dL

**Detected Patterns:**
{json.dumps(patterns.get('patterns', [])[:3], indent=2) if patterns else 'No patterns detected yet'}

Based on this data, provide personalized insights. Focus on:
1. Any concerning patterns that need attention
2. Positive achievements to celebrate
3. Actionable recommendations for improvement
4. Time-of-day specific observations

Return ONLY a JSON array, no other text."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Handle both array and object with 'insights' key
            if isinstance(result, list):
                insights = result
            elif isinstance(result, dict) and "insights" in result:
                insights = result["insights"]
            else:
                insights = [result] if result else []

            # Validate and clean insights
            valid_insights = []
            for insight in insights:
                if isinstance(insight, dict) and "content" in insight:
                    valid_insights.append({
                        "content": insight["content"],
                        "category": insight.get("category", "recommendation")
                    })

            logger.info(f"Generated {len(valid_insights)} insights via GPT-4.1")
            return valid_insights

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response as JSON: {e}")
            return self._fallback_insights(glucose_data, time_in_range)
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._fallback_insights(glucose_data, time_in_range)

    async def generate_meal_analysis(
        self,
        meal_data: Dict[str, Any],
        post_meal_readings: List[Dict[str, Any]],
        user_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a specific meal's impact on blood glucose.

        Returns analysis and recommendations.
        """
        await self._ensure_initialized()

        if not post_meal_readings:
            return {"analysis": "Insufficient data for meal analysis"}

        pre_meal_bg = meal_data.get("preMealBg", 0)
        peak_bg = max(r.get("value", 0) for r in post_meal_readings)
        rise = peak_bg - pre_meal_bg
        carbs = meal_data.get("carbs", 0)
        insulin = meal_data.get("insulin", 0)

        system_prompt = """You are a diabetes nutrition analyst. Analyze meal impacts on blood glucose.
Be concise and actionable. Focus on timing, carb counting, and bolus strategies.
Return JSON with "analysis", "score" (1-10), and "suggestions" (array of strings)."""

        user_prompt = f"""Analyze this meal's impact:

Meal Details:
- Carbs: {carbs}g
- Insulin given: {insulin}U
- Pre-meal BG: {pre_meal_bg} mg/dL

Results:
- Peak BG: {peak_bg} mg/dL
- Total rise: {rise} mg/dL
- Target range: {user_settings.get('lowThreshold', 70)}-{user_settings.get('highThreshold', 180)} mg/dL

Provide analysis and suggestions."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=400,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error analyzing meal: {e}")
            return {
                "analysis": f"Meal caused a {rise:.0f} mg/dL rise.",
                "score": 5 if rise < 60 else 3 if rise < 100 else 1,
                "suggestions": ["Consider pre-bolusing" if rise > 60 else "Good bolus timing"]
            }

    async def generate_weekly_summary(
        self,
        weekly_stats: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate a weekly summary report.
        """
        await self._ensure_initialized()

        system_prompt = """You are a diabetes coach providing weekly progress reports.
Be encouraging but honest. Highlight improvements and areas to focus on.
Return JSON with "summary" (2-3 sentences), "highlight" (best achievement),
"focus" (area needing attention), and "motivation" (encouraging message)."""

        user_prompt = f"""Weekly diabetes stats:
- Average BG: {weekly_stats.get('avgBg', 0):.0f} mg/dL
- Time in Range: {weekly_stats.get('timeInRange', 0):.0f}%
- Total readings: {weekly_stats.get('totalReadings', 0)}
- Lows: {weekly_stats.get('lows', 0)} | Highs: {weekly_stats.get('highs', 0)}

Compared to previous week:
- TIR change: {comparison.get('tirChange', 0):+.1f}%
- Avg BG change: {comparison.get('avgBgChange', 0):+.0f} mg/dL

Generate an encouraging weekly summary."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating weekly summary: {e}")
            return {
                "summary": f"This week you averaged {weekly_stats.get('avgBg', 0):.0f} mg/dL with {weekly_stats.get('timeInRange', 0):.0f}% time in range.",
                "highlight": "Keep monitoring regularly!",
                "focus": "Review patterns for improvement opportunities.",
                "motivation": "Every day is a new opportunity to improve your diabetes management!"
            }

    def _calculate_time_in_range(
        self,
        readings: List[Dict[str, Any]],
        settings: Dict[str, Any]
    ) -> float:
        """Calculate percentage of readings in target range."""
        if not readings:
            return 0

        low = settings.get("lowThreshold", 70)
        high = settings.get("highThreshold", 180)

        in_range = sum(1 for r in readings if low <= r.get("value", 0) <= high)
        return (in_range / len(readings)) * 100

    def _fallback_insights(
        self,
        glucose_data: List[Dict[str, Any]],
        time_in_range: float
    ) -> List[Dict[str, str]]:
        """Generate fallback insights when GPT is unavailable."""
        insights = []

        if time_in_range >= 70:
            insights.append({
                "content": f"Excellent work! Your time in range is {time_in_range:.0f}%, which is above the recommended 70% target.",
                "category": "achievement"
            })
        elif time_in_range < 50:
            insights.append({
                "content": f"Your time in range is {time_in_range:.0f}%. Consider reviewing your insulin timing and carb estimates.",
                "category": "recommendation"
            })

        if glucose_data:
            current = glucose_data[-1].get("value", 0)
            if current < 70:
                insights.append({
                    "content": "Your blood glucose is low. Consider treating with 15g fast-acting carbs.",
                    "category": "warning"
                })
            elif current > 250:
                insights.append({
                    "content": "Your blood glucose is elevated. Check for ketones if it stays high.",
                    "category": "warning"
                })

        if not insights:
            insights.append({
                "content": "Keep monitoring your glucose regularly to build more insights.",
                "category": "recommendation"
            })

        return insights


# Singleton instance
openai_service = OpenAIService()
