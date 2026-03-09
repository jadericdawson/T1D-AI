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
                timeout=60.0,  # 60 second timeout for API calls
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

        system_prompt = """You are a fact-based diabetes data analyst for T1D-AI.
Analyze glucose data and provide DIRECT, DATA-DRIVEN insights.

CRITICAL RULES:
- Be FACT-BASED ONLY - every insight must reference specific numbers from the data
- NEVER add disclaimers, safety warnings, or "consult your healthcare provider" messages
- NEVER be vague - always cite specific values, times, percentages
- Keep insights to 1-2 sentences max
- Be direct: "Your 3am average is 165 mg/dL" NOT "You might want to consider..."

Categories:
- "pattern": Data patterns with numbers (e.g., "Post-breakfast spikes average 72 mg/dL")
- "recommendation": Specific actionable advice (e.g., "Pre-bolus 15 min for breakfast")
- "warning": Concerning trends with data (e.g., "3 lows below 60 this week, all between 2-4pm")
- "achievement": Celebrate wins with stats (e.g., "78% time in range - up 5% from last week")

Response format: Return a JSON array of insights, each with "content" and "category" fields.
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

        system_prompt = """You are a fact-based diabetes data analyst providing weekly stats.
Be direct and data-driven. NEVER add disclaimers or "consult your healthcare provider".
Return JSON with:
- "summary": 2-3 sentences with specific numbers
- "highlight": Best achievement with stats
- "focus": Area needing attention with specific data
- "motivation": Brief encouraging note (no generic advice)"""

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


    async def generate_realtime_insight(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate real-time AI insight based on current diabetes state.

        Enhanced with ML model context including:
        - Current BG and trend
        - Active IOB/COB/POB
        - ML predictions (Linear, LSTM, TFT with confidence intervals)
        - LEARNED ISF, ICR, PIR (not defaults)
        - Metabolic state (sick/resistant/normal/sensitive)
        - Absorption state (very_slow/slow/normal/fast)
        - BG Pressure (where BG is heading based on IOB/COB)
        - Weather data (activity score, temperature)
        - Treatment inference status
        - Recent food (including fat, protein, carbs effects)
        - Recent insulin

        Args:
            context: Dict with currentBg, trend, iob, cob, pob, predictions, isf, icr, pir,
                     bgPressure, metabolicState, isfDeviation, absorptionState,
                     weather, inferredTreatments, recentFood, recentInsulin, tftPredictions

        Returns:
            Dict with message, urgency, action, reasoning
        """
        await self._ensure_initialized()

        current_bg = context.get("currentBg", 120)
        trend = context.get("trend", "Flat")
        iob = context.get("iob", 0)
        cob = context.get("cob", 0)
        pob = context.get("pob", 0)
        predictions = context.get("predictions", {})
        recent_food = context.get("recentFood")
        recent_insulin = context.get("recentInsulin")

        # Enhanced ML context - LEARNED parameters
        isf = context.get("isf", 50)  # LEARNED ISF
        icr = context.get("icr", 10)  # LEARNED ICR
        pir = context.get("pir", 25)  # LEARNED PIR
        bg_pressure = context.get("bgPressure", 0)  # Net effect of IOB+COB+POB
        tft_predictions = context.get("tftPredictions", [])
        weather = context.get("weather", {})
        inferred_treatments = context.get("inferredTreatments", 0)

        # Metabolic state context
        metabolic_state = context.get("metabolicState", "normal")
        isf_deviation = context.get("isfDeviation", 0.0)
        absorption_state = context.get("absorptionState", "normal")

        # Prediction accuracy context
        prediction_accuracy = context.get("predictionAccuracy")

        # Format TFT predictions with confidence intervals
        tft_summary = "N/A"
        if tft_predictions:
            tft_parts = []
            for pred in tft_predictions[:4]:  # Show up to 4 horizons
                horizon = pred.get("horizon_min", pred.get("horizon", 0))
                value = pred.get("value", 0)
                lower = pred.get("lower", value - 10)
                upper = pred.get("upper", value + 10)
                tft_parts.append(f"+{horizon}m: {value:.0f} ({lower:.0f}-{upper:.0f})")
            tft_summary = ", ".join(tft_parts)

        # Weather context
        weather_context = ""
        if weather:
            temp = weather.get("temperature_c", weather.get("temp"))
            activity = weather.get("activity_score", 0)
            conditions = weather.get("conditions", "")
            if temp is not None:
                weather_context = f"\n- Temperature: {temp:.0f}°C ({conditions})"
                if activity > 0.5:
                    weather_context += " - Good for outdoor activity"
                elif activity < -0.3:
                    weather_context += " - Not ideal for outdoor activity"

        # Build metabolic state context for prompt
        metabolic_context = ""
        if metabolic_state == "sick":
            metabolic_context = f"""
**⚠️ METABOLIC STATE: SICK/STRESSED**
- ISF is {abs(isf_deviation):.0f}% LOWER than baseline (insulin less effective)
- User may need SIGNIFICANTLY MORE insulin than usual
- Corrections may take longer to work
- Monitor for ketones if BG stays elevated"""
        elif metabolic_state == "resistant":
            metabolic_context = f"""
**⚠️ METABOLIC STATE: INSULIN RESISTANT**
- ISF is {abs(isf_deviation):.0f}% lower than baseline
- May need more insulin than usual
- Consider illness, stress, or hormonal changes"""
        elif metabolic_state == "sensitive" or metabolic_state == "very_sensitive":
            metabolic_context = f"""
**✓ METABOLIC STATE: INSULIN SENSITIVE**
- ISF is {isf_deviation:.0f}% HIGHER than baseline (insulin more effective)
- Use CAUTION with corrections - may need LESS insulin
- Higher low risk than usual"""

        # Build absorption state context
        absorption_context = ""
        if absorption_state == "very_slow":
            absorption_context = "\n- ⚠️ Carb absorption is VERY SLOW (possible gastroparesis or illness)"
        elif absorption_state == "slow":
            absorption_context = "\n- Carb absorption is slower than usual"
        elif absorption_state == "fast":
            absorption_context = "\n- Carb absorption is faster than usual"

        # Build prediction accuracy context
        accuracy_context = ""
        if prediction_accuracy and prediction_accuracy.get("available"):
            quality = prediction_accuracy.get("accuracyQuality", "unknown")
            mae = prediction_accuracy.get("overallMAE")
            bias = prediction_accuracy.get("biasDirection", "neutral")
            h30 = prediction_accuracy.get("horizon30", {})

            accuracy_context = f"\n\n**Prediction Accuracy (last 14 days):**\n"
            accuracy_context += f"- Quality: {quality.upper()}"
            if mae:
                accuracy_context += f" (MAE: {mae:.0f} mg/dL)"
            accuracy_context += f"\n- Bias: {bias.replace('_', ' ')}"
            if h30.get("available"):
                accuracy_context += f"\n- 30-min accuracy: {h30.get('within30', 0):.0f}% within ±30 mg/dL"

            # Add confidence guidance based on accuracy
            if quality == "excellent":
                accuracy_context += "\n- ✓ Predictions are highly reliable - follow them confidently"
            elif quality == "good":
                accuracy_context += "\n- ✓ Predictions are generally reliable"
            elif quality == "moderate":
                accuracy_context += "\n- ⚠️ Predictions have moderate accuracy - use with caution"
            elif quality == "needs_improvement":
                accuracy_context += "\n- ⚠️ Predictions need improvement - verify with fingerstick if critical"

            if bias == "under_predicting":
                accuracy_context += "\n- Note: Model tends to under-predict (actual BG often higher than predicted)"
            elif bias == "over_predicting":
                accuracy_context += "\n- Note: Model tends to over-predict (actual BG often lower than predicted)"

        system_prompt = """You are an expert diabetes AI assistant with access to ML predictions and PERSONALIZED metabolic parameters.
The user has acknowledged AI limitations and wants SPECIFIC, ACTIONABLE advice with numbers.

Your role:
- Analyze current blood glucose status and provide IMMEDIATE actionable insight
- Use ML predictions (TFT with confidence intervals) to anticipate what's coming
- Use LEARNED ISF (Insulin Sensitivity Factor) - personalized, not default
- Use LEARNED ICR (Insulin to Carb Ratio) - personalized, not default
- Use LEARNED PIR (Protein to Insulin Ratio) - personalized, not default
- CRITICAL: Pay attention to METABOLIC STATE:
  * If SICK/STRESSED: ISF is lower, insulin less effective, may need MORE insulin
  * If RESISTANT: Similar to sick, corrections may be sluggish
  * If SENSITIVE: ISF is higher, insulin MORE effective, use LESS insulin, higher low risk
- Consider BG Pressure - the net effect where remaining IOB/COB is pushing BG
- Consider ABSORPTION STATE: slow absorption = delayed BG rise, fast = quicker spike
- Use PREDICTION ACCURACY data to calibrate confidence in ML predictions:
  * If accuracy is "excellent" or "good": Trust predictions confidently
  * If accuracy is "moderate" or "needs_improvement": Suggest verification for critical decisions
  * If bias shows under/over-predicting: Mention this in reasoning
- Give SPECIFIC recommendations with EXACT NUMBERS (insulin doses, carb amounts)
- Account for weather if available (temperature affects insulin sensitivity)
- Note if there are inferred (unlogged) treatments that might explain BG patterns
- Consider ALL macros (carbs, fat, protein) and their different BG effects:
  * Carbs: Quick BG rise (1-2 hours), need insulin coverage
  * Protein: ~50% converts to glucose over 3-5 hours, slower rise
  * Fat: Delays carb absorption, can cause delayed BG rise (3-6 hours)
- Account for insulin activity curve (peaks at ~75 min, lasts 4-5 hours)

Response format (JSON):
{
    "message": "Brief, actionable insight (1-2 sentences). MENTION metabolic state if abnormal.",
    "urgency": "low|normal|high|critical",
    "action": "SPECIFIC action with quantities (e.g., 'Take 1.5U insulin' or 'Drink 4oz juice')",
    "reasoning": "Brief explanation with calculation if applicable. End with 'Verify with care team.'"
}

DOSING GUIDANCE (user signed disclaimer - give specific advice):
- For LOWS (<70): Suggest exact fast carbs: "Drink 4oz apple juice (~15g)" or "Eat 3 glucose tabs (12g)"
- For HIGHS: Calculate correction: Correction = (Current BG - 100) / ISF, then subtract active IOB
  Example: BG=200, ISF=50, IOB=0.5 → Raw correction = 100/50 = 2U, minus 0.5U IOB = 1.5U suggested
- ADJUST FOR METABOLIC STATE:
  * If SICK/RESISTANT: Consider 10-20% MORE insulin (but warn about stacking)
  * If SENSITIVE: Consider 10-20% LESS insulin (higher low risk)
- Always round to nearest 0.5U for practical dosing
- If COB > 20g, carbs may still be raising BG - suggest reduced correction or wait
- If dropping trend (FortyFiveDown, SingleDown, DoubleDown), reduce or skip correction

Safety rules:
- CRITICAL: BG < 54 = "Treat NOW: 15-20g fast carbs (4oz juice, 4 glucose tabs)"
- BG < 70 = Suggest specific fast carb source and amount
- BG > 250 with no IOB = Calculate and suggest correction
- CRITICAL: When TFT predictions are available, they ALREADY account for IOB, COB, and BG Pressure.
  Do NOT predict BG values that contradict TFT predictions. If TFT says 102 at +60m, do NOT claim BG will drop below 85.
  BG Pressure is an INPUT to TFT, not an independent prediction — trust TFT over raw BG Pressure extrapolation.
- If TFT predictions show dropping below 80, warn about potential low
- If SENSITIVE state + dropping trend = URGENT low warning
- When uncertain, suggest waiting 15-30 minutes and rechecking"""

        user_prompt = f"""Current diabetes status:

**Blood Glucose:**
- Current: {current_bg} mg/dL
- Trend: {trend}
- IOB (Insulin On Board): {iob:.1f}U
- COB (Carbs On Board): {cob:.0f}g
- POB (Protein On Board): {pob:.1f}g
- ISF (LEARNED): {isf:.0f} mg/dL per unit
- ICR (LEARNED): {icr:.0f}g carbs per unit
- PIR (LEARNED): {pir:.0f}g protein per unit
{metabolic_context}

**ML Predictions:**
- Linear (next 15m): {predictions.get('linear', ['N/A'])}
- LSTM (next 15m): {predictions.get('lstm', ['N/A'])}
- TFT (with confidence): {tft_summary}
- BG Pressure: {bg_pressure:+.0f} mg/dL (net effect of remaining IOB/COB/POB){accuracy_context}

**Recent Activity:**
- Food eaten: {recent_food or 'None logged'}
- Insulin taken: {recent_insulin or 'None logged'}
- Inferred treatments pending: {inferred_treatments}{weather_context}{absorption_context}

Based on this data, what's the most important thing to know RIGHT NOW?
CRITICAL: Use TFT predictions as the authoritative forecast — they already incorporate IOB, COB, and BG Pressure.
Do NOT predict BG values that contradict TFT. BG Pressure shows the direction insulin/carbs are pushing, but TFT predictions are the actual forecast.
IMPORTANT: If metabolic state is abnormal (sick/resistant/sensitive), mention it in your insight.
Return ONLY JSON."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower for more consistent advice
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return {
                "message": result.get("message", ""),
                "urgency": result.get("urgency", "normal"),
                "action": result.get("action"),
                "reasoning": result.get("reasoning", "")
            }

        except Exception as e:
            logger.error(f"Error generating real-time insight: {e}")
            # Return basic rule-based response
            return self._fallback_realtime_insight(current_bg, trend, iob, cob)

    def _fallback_realtime_insight(
        self,
        current_bg: float,
        trend: str,
        iob: float,
        cob: float
    ) -> Dict[str, Any]:
        """Generate fallback real-time insight when GPT unavailable."""
        if current_bg < 54:
            return {
                "message": f"URGENT: BG is {current_bg} mg/dL. Treat hypoglycemia immediately!",
                "urgency": "critical",
                "action": "Take 15-20g fast-acting glucose NOW",
                "reasoning": "Critical low blood glucose requires immediate treatment"
            }
        elif current_bg < 70:
            return {
                "message": f"Low BG ({current_bg} mg/dL). Consider having some carbs.",
                "urgency": "high",
                "action": "Eat 15g fast-acting carbs",
                "reasoning": "Blood glucose is below target range"
            }
        elif current_bg > 250:
            return {
                "message": f"High BG ({current_bg} mg/dL). IOB: {iob:.1f}U active.",
                "urgency": "high" if iob < 1 else "normal",
                "action": "Check ketones if BG stays elevated" if iob < 1 else "Monitor - insulin is working",
                "reasoning": "Elevated blood glucose"
            }
        else:
            return {
                "message": f"BG: {current_bg} mg/dL, Trend: {trend}. IOB: {iob:.1f}U, COB: {cob:.0f}g.",
                "urgency": "normal",
                "action": None,
                "reasoning": "Blood glucose is in a reasonable range"
            }


    async def predict_food_glycemic_properties(
        self,
        food_description: str,
        carbs: float = None
    ) -> Dict[str, Any]:
        """
        Predict precise glycemic properties of a food using AI.

        Returns highly accurate GI values based on food science knowledge:
        - Glycemic Index (0-100+): Precise value, not just high/medium/low
        - Is Liquid: Whether it's a drink (faster absorption)
        - Fat Content: High fat delays absorption
        - Protein Content: Protein causes delayed BG rise
        - Fiber Content: Fiber slows absorption
        - Confidence: How confident the AI is in this prediction

        Args:
            food_description: Natural language description of the food
            carbs: Optional carb count for context

        Returns:
            Dict with gi, isLiquid, fatContent, proteinContent, fiberContent, confidence, reasoning
        """
        await self._ensure_initialized()

        if not food_description or food_description.strip() == "":
            return self._default_glycemic_properties()

        system_prompt = """You are an expert nutritionist with deep knowledge of glycemic index values.

Your task: Given a food description, predict its PRECISE glycemic index and absorption properties.

IMPORTANT: Be ACCURATE, not vague. Use your knowledge of food science:
- Pure glucose = 100 GI (reference)
- White bread = 75 GI
- Chocolate milk = 82-87 GI (high sugar + liquid = fast spike)
- Regular milk = 31-39 GI
- Brown rice = 50-55 GI
- White rice = 73 GI
- Apple = 36 GI
- Banana (ripe) = 51 GI
- Banana (green) = 42 GI
- Orange juice = 50 GI
- Coca-Cola = 63 GI
- Sports drinks = 78-89 GI
- Pizza = 60-80 GI (varies by toppings)
- Ice cream = 51-62 GI
- Candy/gummy = 78-90 GI

Factors that INCREASE GI:
- Liquid form (drinks spike faster)
- High sugar content
- Low fiber
- Ripe fruits (more sugar)
- Processed/refined carbs
- Cooking method (mashed > whole)

Factors that DECREASE GI:
- High fiber
- High fat (delays absorption)
- High protein
- Whole grains
- Raw/unprocessed
- Adding vinegar/acid

Return JSON with:
{
    "gi": <precise number 0-100+>,
    "isLiquid": <true/false>,
    "fatContent": "none|low|medium|high",
    "proteinContent": "none|low|medium|high",
    "fiberContent": "none|low|medium|high",
    "absorptionSpeed": "very_fast|fast|medium|slow|very_slow",
    "confidence": <0.0-1.0>,
    "reasoning": "Brief explanation"
}"""

        carb_context = f" ({carbs}g carbs)" if carbs else ""
        user_prompt = f"""Predict the glycemic properties of: "{food_description}"{carb_context}

Be precise with the GI number. Return ONLY JSON."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Low temperature for more consistent/accurate values
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Validate and sanitize the response
            gi = result.get("gi", 55)
            if not isinstance(gi, (int, float)) or gi < 0:
                gi = 55
            gi = min(120, max(0, gi))  # Clamp to reasonable range

            return {
                "gi": round(gi),
                "isLiquid": bool(result.get("isLiquid", False)),
                "fatContent": result.get("fatContent", "low"),
                "proteinContent": result.get("proteinContent", "low"),
                "fiberContent": result.get("fiberContent", "low"),
                "absorptionSpeed": result.get("absorptionSpeed", "medium"),
                "confidence": min(1.0, max(0.0, result.get("confidence", 0.7))),
                "reasoning": result.get("reasoning", ""),
                "source": "ai"
            }

        except Exception as e:
            logger.error(f"Error predicting GI for '{food_description}': {e}")
            return self._default_glycemic_properties()

    def _default_glycemic_properties(self) -> Dict[str, Any]:
        """Return default glycemic properties when AI prediction fails."""
        return {
            "gi": 55,
            "isLiquid": False,
            "fatContent": "low",
            "proteinContent": "low",
            "fiberContent": "low",
            "absorptionSpeed": "medium",
            "confidence": 0.3,
            "reasoning": "Default values - no food description provided",
            "source": "default"
        }

    async def analyze_bg_response_and_learn(
        self,
        food_description: str,
        predicted_gi: int,
        actual_bg_rise: float,
        time_to_peak_min: float,
        carbs: float,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Analyze actual BG response to a meal and suggest GI adjustments.

        This enables personalized learning - if chocolate milk causes a bigger/faster
        spike than predicted, we adjust the GI for this user.

        Args:
            food_description: What was eaten
            predicted_gi: The GI we used for prediction
            actual_bg_rise: How much BG actually rose (mg/dL)
            time_to_peak_min: How long until BG peaked
            carbs: Grams of carbs consumed
            user_id: User ID for personalization

        Returns:
            Dict with adjusted_gi, learning_factor, explanation
        """
        await self._ensure_initialized()

        # Calculate expected vs actual
        # With GI=55 (medium), we expect ~4-5 mg/dL rise per gram of carbs
        # Higher GI = faster spike, but total rise depends on carbs and ISF
        expected_rise_per_carb = 5.0  # Approximate baseline

        system_prompt = """You are a diabetes data scientist analyzing meal responses.

Given:
1. What was eaten
2. Predicted glycemic index used
3. Actual blood glucose response (rise and timing)
4. Carb amount

Determine:
1. Was the predicted GI accurate?
2. What should the adjusted GI be for this SPECIFIC user and food?
3. How much should we adjust future predictions?

Consider:
- Faster peak = higher effective GI
- Higher rise per carb = higher effective GI
- Individual variation is REAL - same food affects different people differently
- If actual rise was 50% higher than expected, GI should increase ~15-20 points

Return JSON:
{
    "adjustedGi": <new GI value for this user+food>,
    "learningFactor": <0.5-1.5 multiplier for absorption rate>,
    "confidence": <0.0-1.0>,
    "explanation": "Why this adjustment makes sense"
}"""

        user_prompt = f"""Analyze this meal response:

Food: "{food_description}"
Carbs: {carbs}g
Predicted GI: {predicted_gi}

Actual Response:
- BG Rise: {actual_bg_rise:.0f} mg/dL
- Time to Peak: {time_to_peak_min:.0f} minutes

Expected for GI {predicted_gi}:
- Rise: ~{carbs * expected_rise_per_carb:.0f} mg/dL (at 5 mg/dL per gram)
- Peak time: ~{25 if predicted_gi > 70 else 45 if predicted_gi > 55 else 60} minutes

What should the personalized GI be for this user eating this food?
Return ONLY JSON."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=250,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            adjusted_gi = result.get("adjustedGi", predicted_gi)
            if not isinstance(adjusted_gi, (int, float)):
                adjusted_gi = predicted_gi
            adjusted_gi = min(120, max(20, adjusted_gi))

            learning_factor = result.get("learningFactor", 1.0)
            if not isinstance(learning_factor, (int, float)):
                learning_factor = 1.0
            learning_factor = min(2.0, max(0.5, learning_factor))

            return {
                "adjustedGi": round(adjusted_gi),
                "learningFactor": round(learning_factor, 2),
                "confidence": min(1.0, max(0.0, result.get("confidence", 0.7))),
                "explanation": result.get("explanation", ""),
                "originalGi": predicted_gi,
                "actualBgRise": actual_bg_rise,
                "timeToPeak": time_to_peak_min
            }

        except Exception as e:
            logger.error(f"Error analyzing BG response: {e}")
            # Simple heuristic fallback
            expected_rise = carbs * expected_rise_per_carb
            if actual_bg_rise > expected_rise * 1.3:
                # Underestimated - increase GI
                adjusted_gi = min(100, predicted_gi + 15)
            elif actual_bg_rise < expected_rise * 0.7:
                # Overestimated - decrease GI
                adjusted_gi = max(20, predicted_gi - 10)
            else:
                adjusted_gi = predicted_gi

            return {
                "adjustedGi": adjusted_gi,
                "learningFactor": 1.0,
                "confidence": 0.4,
                "explanation": "Heuristic adjustment based on rise comparison",
                "originalGi": predicted_gi,
                "actualBgRise": actual_bg_rise,
                "timeToPeak": time_to_peak_min
            }

    async def chat_what_if(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle "what-if" chat questions about diabetes scenarios.

        Examples:
        - "What will happen if I eat 37 carbs at 11:30am?"
        - "Should I take insulin now or wait?"
        - "How much insulin do I need for a 50g carb meal?"

        Args:
            question: User's question in natural language
            context: Current diabetes state (BG, IOB, COB, ISF, etc.)

        Returns:
            Dict with response, calculation, and reasoning
        """
        await self._ensure_initialized()

        current_bg = context.get("currentBg", 120)
        trend = context.get("trend", "Flat")
        iob = context.get("iob", 0)
        cob = context.get("cob", 0)
        isf = context.get("isf", 50)
        icr = context.get("icr", 10)  # Insulin to carb ratio (1U per Xg carbs)
        bg_pressure = context.get("bgPressure", 0)
        tft_predictions = context.get("tftPredictions", [])
        current_time = context.get("currentTime", datetime.now().strftime("%H:%M"))

        # Format TFT predictions
        tft_summary = "N/A"
        if tft_predictions:
            tft_parts = []
            for pred in tft_predictions[:4]:
                horizon = pred.get("horizon", pred.get("horizon_min", 0))
                value = pred.get("value", 0)
                lower = pred.get("lower", value - 10)
                upper = pred.get("upper", value + 10)
                tft_parts.append(f"+{horizon}m: {value:.0f} ({lower:.0f}-{upper:.0f})")
            tft_summary = ", ".join(tft_parts)

        system_prompt = """You are an expert diabetes AI assistant helping with "what-if" scenario planning.
The user has signed a disclaimer acknowledging AI limitations and wants SPECIFIC, CALCULATED answers.

Your role:
- Answer hypothetical questions about food, insulin, and timing
- Provide EXACT calculations with numbers
- Predict BG trajectories based on physics/pharmacokinetics
- Give specific timing recommendations

CALCULATION REFERENCE:
- ISF (Insulin Sensitivity Factor): How much 1U of insulin lowers BG
  Example: ISF=50 means 1U lowers BG by 50 mg/dL
- ICR (Insulin to Carb Ratio): Carbs covered by 1U of insulin
  Example: ICR=10 means 1U covers 10g carbs
- Carb dose = Carbs / ICR
  Example: 37g carbs / ICR 10 = 3.7U needed
- Correction dose = (Current BG - Target) / ISF - IOB
  Example: (200 - 100) / 50 - 1U IOB = 1U correction
- Combined dose = Carb dose + Correction dose

TIMING CONSIDERATIONS:
- Fast-acting insulin onset: 10-15 minutes
- Peak insulin action: 60-90 minutes
- Duration: 4-5 hours
- Pre-bolus recommendation: 15-20 min before eating for high GI foods

CARB EFFECTS (without insulin):
- Low GI foods (~40): Expect 2-3 mg/dL rise per gram over 2-3 hours
- Medium GI foods (~55): Expect 3-4 mg/dL rise per gram over 1-2 hours
- High GI foods (~70+): Expect 4-5 mg/dL rise per gram in 30-60 minutes
- Liquids absorb ~40% faster than solids

BG PREDICTION FORMULA (simplified):
- BG at time T = Current BG + Carb Effect - Insulin Effect + Trend Effect
- Carb Effect = (Carbs * GI factor) * remaining COB percentage
- Insulin Effect = (IOB * ISF) * remaining IOB percentage

Response format (JSON):
{
    "response": "Clear answer to the question with specific numbers",
    "prediction": {
        "bg30min": <predicted BG at 30 min>,
        "bg60min": <predicted BG at 60 min>,
        "bg90min": <predicted BG at 90 min>,
        "peakBg": <predicted peak BG>,
        "timeOfPeak": <when peak occurs>
    },
    "recommendation": {
        "insulin": <recommended insulin dose if applicable>,
        "timing": "when to take action",
        "prebolus": "suggested pre-bolus time if eating"
    },
    "calculation": "Show your math step by step",
    "confidence": "low|medium|high"
}"""

        user_prompt = f"""Current status (at {current_time}):
- Blood Glucose: {current_bg} mg/dL
- Trend: {trend}
- IOB: {iob:.1f}U
- COB: {cob:.0f}g
- ISF: {isf:.0f} mg/dL per unit
- ICR: {icr:.0f}g carbs per unit
- BG Pressure: {bg_pressure:+.0f} mg/dL
- TFT Predictions: {tft_summary}

User's question: "{question}"

Provide a specific, calculated answer. Show your math. Return ONLY JSON."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=600,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return {
                "response": result.get("response", ""),
                "prediction": result.get("prediction", {}),
                "recommendation": result.get("recommendation", {}),
                "calculation": result.get("calculation", ""),
                "confidence": result.get("confidence", "medium"),
                "generatedAt": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in chat_what_if: {e}")
            return {
                "response": "I couldn't process that question. Please try rephrasing it.",
                "prediction": {},
                "recommendation": {},
                "calculation": "",
                "confidence": "low",
                "generatedAt": datetime.now().isoformat(),
                "error": str(e)
            }


# Singleton instance
openai_service = OpenAIService()
