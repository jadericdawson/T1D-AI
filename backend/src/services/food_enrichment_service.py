"""
Food Enrichment Service for T1D-AI
Uses comprehensive GI database lookup with GPT fallback for unknown foods.
This enriches treatment data for more accurate COB predictions.

Lookup priority:
1. Comprehensive GI database (300+ foods with precise values)
2. GPT-4.1 AI prediction (for unknown foods)
3. Macro-based estimation (fallback when AI unavailable)
"""
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from openai import AsyncAzureOpenAI
from config import get_settings

logger = logging.getLogger(__name__)

# Import comprehensive GI database
try:
    from data.glycemic_index_db import lookup_gi, GLYCEMIC_INDEX_DATABASE
    GI_DATABASE_AVAILABLE = True
    logger.info(f"GI database loaded with {len(GLYCEMIC_INDEX_DATABASE)} foods")
except ImportError:
    GI_DATABASE_AVAILABLE = False
    lookup_gi = None
    logger.warning("GI database not available, falling back to AI-only mode")
settings = get_settings()


@dataclass
class FoodFeatures:
    """Extracted food features from GPT analysis."""
    glycemic_index: int  # 0-100 (PRECISE value, not just high/medium/low)
    glycemic_load: float  # carbs * GI / 100
    absorption_rate: str  # "very_fast", "fast", "medium", "slow", "very_slow"
    fat_content: str  # "none", "low", "medium", "high"
    is_liquid: bool  # True for drinks - absorb 40% faster
    confidence: float  # 0-1 confidence in analysis
    reasoning: Optional[str] = None


@dataclass
class MacroEstimate:
    """Full macro estimation from food description."""
    carbs_g: float
    protein_g: float
    fat_g: float
    glycemic_index: int
    absorption_rate: str  # "fast", "medium", "slow"
    absorption_half_life_min: float  # Estimated carb absorption half-life
    confidence: float
    reasoning: Optional[str] = None


class FoodEnrichmentService:
    """
    Service for enriching food descriptions with ML-relevant features.

    Uses GPT to analyze food descriptions (e.g., "hot dog", "birthday cake", "oatmeal")
    and extract:
    - Glycemic Index (0-100): How fast the food raises blood sugar
    - Glycemic Load: Total blood sugar impact (carbs * GI / 100)
    - Absorption Rate: fast/medium/slow based on food composition
    - Fat Content: low/medium/high - affects absorption delay

    These features improve COB ML model accuracy.
    """

    # PRECISE GI lookup with liquid detection (research-based values)
    # is_liquid=True means 40% faster absorption
    KNOWN_FOODS = {
        # === VERY FAST (GI > 85, liquid) - spike in 3-8 min ===
        "glucose tabs": {"gi": 100, "rate": "very_fast", "fat": "none", "liquid": False},
        "glucose": {"gi": 100, "rate": "very_fast", "fat": "none", "liquid": False},
        "dextrose": {"gi": 100, "rate": "very_fast", "fat": "none", "liquid": False},
        "glucose gel": {"gi": 95, "rate": "very_fast", "fat": "none", "liquid": True},
        "juice box": {"gi": 90, "rate": "very_fast", "fat": "none", "liquid": True},
        "apple juice": {"gi": 44, "rate": "fast", "fat": "none", "liquid": True},
        "orange juice": {"gi": 50, "rate": "fast", "fat": "none", "liquid": True},
        "grape juice": {"gi": 56, "rate": "fast", "fat": "none", "liquid": True},
        "sports drink": {"gi": 89, "rate": "very_fast", "fat": "none", "liquid": True},
        "gatorade": {"gi": 89, "rate": "very_fast", "fat": "none", "liquid": True},
        "chocolate milk": {"gi": 84, "rate": "very_fast", "fat": "low", "liquid": True},
        "choc milk": {"gi": 84, "rate": "very_fast", "fat": "low", "liquid": True},
        "strawberry milk": {"gi": 82, "rate": "very_fast", "fat": "low", "liquid": True},
        "milkshake": {"gi": 75, "rate": "fast", "fat": "medium", "liquid": True},
        "smoothie": {"gi": 68, "rate": "fast", "fat": "low", "liquid": True},
        "soda": {"gi": 63, "rate": "fast", "fat": "none", "liquid": True},
        "coca cola": {"gi": 63, "rate": "fast", "fat": "none", "liquid": True},
        "coke": {"gi": 63, "rate": "fast", "fat": "none", "liquid": True},
        "sprite": {"gi": 59, "rate": "fast", "fat": "none", "liquid": True},
        "fanta": {"gi": 68, "rate": "fast", "fat": "none", "liquid": True},

        # === FAST (GI 70-85) - spike in 8-15 min ===
        "white bread": {"gi": 75, "rate": "fast", "fat": "low", "liquid": False},
        "bagel": {"gi": 72, "rate": "fast", "fat": "low", "liquid": False},
        "rice krispies": {"gi": 82, "rate": "fast", "fat": "low", "liquid": False},
        "corn flakes": {"gi": 81, "rate": "fast", "fat": "low", "liquid": False},
        "cheerios": {"gi": 74, "rate": "fast", "fat": "low", "liquid": False},
        "gummy": {"gi": 78, "rate": "fast", "fat": "low", "liquid": False},
        "gummies": {"gi": 78, "rate": "fast", "fat": "low", "liquid": False},
        "candy": {"gi": 80, "rate": "fast", "fat": "low", "liquid": False},
        "skittles": {"gi": 70, "rate": "fast", "fat": "low", "liquid": False},
        "jelly beans": {"gi": 78, "rate": "fast", "fat": "low", "liquid": False},
        "potato": {"gi": 78, "rate": "fast", "fat": "low", "liquid": False},
        "mashed potato": {"gi": 83, "rate": "fast", "fat": "low", "liquid": False},
        "french fries": {"gi": 75, "rate": "medium", "fat": "high", "liquid": False},
        "white rice": {"gi": 73, "rate": "fast", "fat": "low", "liquid": False},
        "instant rice": {"gi": 87, "rate": "fast", "fat": "low", "liquid": False},
        "watermelon": {"gi": 76, "rate": "fast", "fat": "none", "liquid": False},
        "pineapple": {"gi": 66, "rate": "fast", "fat": "none", "liquid": False},

        # === MEDIUM (GI 55-70) - spike in 15-30 min ===
        "rice": {"gi": 65, "rate": "medium", "fat": "low", "liquid": False},
        "brown rice": {"gi": 50, "rate": "medium", "fat": "low", "liquid": False},
        "pasta": {"gi": 55, "rate": "medium", "fat": "low", "liquid": False},
        "spaghetti": {"gi": 49, "rate": "medium", "fat": "low", "liquid": False},
        "mac and cheese": {"gi": 64, "rate": "medium", "fat": "medium", "liquid": False},
        "banana": {"gi": 51, "rate": "medium", "fat": "low", "liquid": False},
        "ripe banana": {"gi": 62, "rate": "medium", "fat": "low", "liquid": False},
        "oatmeal": {"gi": 55, "rate": "medium", "fat": "low", "liquid": False},
        "cereal": {"gi": 60, "rate": "medium", "fat": "low", "liquid": False},
        "granola": {"gi": 55, "rate": "medium", "fat": "medium", "liquid": False},
        "sandwich": {"gi": 55, "rate": "medium", "fat": "medium", "liquid": False},
        "bread": {"gi": 65, "rate": "medium", "fat": "low", "liquid": False},
        "pancake": {"gi": 66, "rate": "medium", "fat": "medium", "liquid": False},
        "waffle": {"gi": 76, "rate": "medium", "fat": "medium", "liquid": False},
        "popcorn": {"gi": 65, "rate": "medium", "fat": "low", "liquid": False},
        "tortilla": {"gi": 52, "rate": "medium", "fat": "low", "liquid": False},
        "honey": {"gi": 61, "rate": "fast", "fat": "none", "liquid": False},
        "grapes": {"gi": 59, "rate": "medium", "fat": "none", "liquid": False},
        "orange": {"gi": 43, "rate": "medium", "fat": "none", "liquid": False},
        "mango": {"gi": 56, "rate": "medium", "fat": "none", "liquid": False},

        # === SLOW (GI < 55) - spike in 30-60 min ===
        "apple": {"gi": 36, "rate": "slow", "fat": "low", "liquid": False},
        "pear": {"gi": 38, "rate": "slow", "fat": "low", "liquid": False},
        "strawberries": {"gi": 41, "rate": "slow", "fat": "low", "liquid": False},
        "blueberries": {"gi": 53, "rate": "slow", "fat": "low", "liquid": False},
        "milk": {"gi": 39, "rate": "slow", "fat": "low", "liquid": True},
        "yogurt": {"gi": 41, "rate": "slow", "fat": "medium", "liquid": False},
        "whole grain": {"gi": 45, "rate": "slow", "fat": "low", "liquid": False},
        "beans": {"gi": 32, "rate": "slow", "fat": "low", "liquid": False},
        "lentils": {"gi": 32, "rate": "slow", "fat": "low", "liquid": False},
        "chickpeas": {"gi": 28, "rate": "slow", "fat": "low", "liquid": False},
        "hummus": {"gi": 25, "rate": "slow", "fat": "medium", "liquid": False},

        # === VERY SLOW (high fat) - extended absorption 1-4 hours ===
        "pizza": {"gi": 60, "rate": "very_slow", "fat": "high", "liquid": False},
        "pepperoni pizza": {"gi": 60, "rate": "very_slow", "fat": "high", "liquid": False},
        "cheese pizza": {"gi": 60, "rate": "very_slow", "fat": "high", "liquid": False},
        "burger": {"gi": 66, "rate": "very_slow", "fat": "high", "liquid": False},
        "cheeseburger": {"gi": 66, "rate": "very_slow", "fat": "high", "liquid": False},
        "hot dog": {"gi": 52, "rate": "slow", "fat": "high", "liquid": False},
        "ice cream": {"gi": 51, "rate": "slow", "fat": "high", "liquid": False},
        "chocolate": {"gi": 40, "rate": "slow", "fat": "high", "liquid": False},
        "chocolate bar": {"gi": 44, "rate": "slow", "fat": "high", "liquid": False},
        "donut": {"gi": 76, "rate": "slow", "fat": "high", "liquid": False},
        "cake": {"gi": 67, "rate": "slow", "fat": "medium", "liquid": False},
        "birthday cake": {"gi": 67, "rate": "slow", "fat": "medium", "liquid": False},
        "brownie": {"gi": 72, "rate": "slow", "fat": "high", "liquid": False},
        "cookie": {"gi": 55, "rate": "slow", "fat": "medium", "liquid": False},
        "cheese": {"gi": 0, "rate": "very_slow", "fat": "high", "liquid": False},
        "nuts": {"gi": 15, "rate": "very_slow", "fat": "high", "liquid": False},
        "peanut butter": {"gi": 14, "rate": "very_slow", "fat": "high", "liquid": False},
    }

    def __init__(self):
        self.client: Optional[AsyncAzureOpenAI] = None
        self._initialized = False
        self._cache: Dict[str, FoodFeatures] = {}

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
            logger.info("Food enrichment service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize food enrichment service: {e}")
            # Don't raise - we have fallback

    async def _ensure_initialized(self):
        """Ensure client is initialized before use."""
        if not self._initialized:
            await self.initialize()

    async def extract_food_features(
        self,
        food_text: str,
        carbs: float = 0,
        protein: float = 0,
        fat: float = 0
    ) -> FoodFeatures:
        """
        Extract food features using database lookup with AI fallback.

        Lookup priority:
        1. Local cache (instant)
        2. Comprehensive GI database (300+ foods, instant)
        3. KNOWN_FOODS hardcoded table (instant)
        4. GPT-4.1 AI prediction (network call)
        5. Macro-based estimation (fallback)

        Args:
            food_text: Description of food (e.g., "hot dog with bun", "birthday cake")
            carbs: Known carb amount in grams (helps with glycemic load calculation)
            protein: Known protein in grams
            fat: Known fat in grams

        Returns:
            FoodFeatures with precise glycemic index, load, absorption rate, and fat content
        """
        # If no food text but we have macros, use AI to infer food type
        if (not food_text or not food_text.strip()) and (carbs > 0 or protein > 0 or fat > 0):
            return await self._infer_from_macros(carbs, protein, fat)

        if not food_text or not food_text.strip():
            return self._default_features(carbs)

        # Normalize text for cache lookup
        normalized = food_text.lower().strip()

        # 1. Check cache first (instant)
        if normalized in self._cache:
            cached = self._cache[normalized]
            return FoodFeatures(
                glycemic_index=cached.glycemic_index,
                glycemic_load=carbs * cached.glycemic_index / 100 if carbs > 0 else cached.glycemic_load,
                absorption_rate=cached.absorption_rate,
                fat_content=cached.fat_content,
                is_liquid=cached.is_liquid,
                confidence=cached.confidence,
                reasoning=cached.reasoning
            )

        # 2. Check comprehensive GI database (300+ foods, instant)
        if GI_DATABASE_AVAILABLE and lookup_gi:
            db_result = lookup_gi(normalized)
            if db_result:
                gi = db_result["gi"]
                is_liquid = db_result.get("is_liquid", False)
                category = db_result.get("category", "unknown")

                # Determine absorption rate from GI
                if gi >= 85:
                    rate = "very_fast"
                elif gi >= 70:
                    rate = "fast"
                elif gi >= 55:
                    rate = "medium"
                elif gi >= 40:
                    rate = "slow"
                else:
                    rate = "very_slow"

                # Adjust for fat content
                fat_level = "none" if fat == 0 else "low" if fat <= 5 else "medium" if fat <= 15 else "high"
                if fat > 15 and rate in ("very_fast", "fast"):
                    rate = "medium"
                elif fat > 15 and rate == "medium":
                    rate = "slow"

                result = FoodFeatures(
                    glycemic_index=gi,
                    glycemic_load=carbs * gi / 100 if carbs > 0 else 0,
                    absorption_rate=rate,
                    fat_content=fat_level,
                    is_liquid=is_liquid,
                    confidence=0.95,  # High confidence for database matches
                    reasoning=f"GI Database: matched '{db_result.get('matched', normalized)}' ({category})"
                )
                self._cache[normalized] = result
                logger.info(f"GI lookup: '{normalized}' -> GI={gi}, liquid={is_liquid} (from database)")
                return result

        # 3. Try hardcoded KNOWN_FOODS lookup (fallback for items not in database)
        known_result = self._lookup_known_food(normalized, carbs, fat)
        if known_result:
            self._cache[normalized] = known_result
            return known_result

        # 4. Use GPT for unknown foods (network call)
        await self._ensure_initialized()

        if not self.client:
            return self._estimate_from_macros(carbs, protein, fat)

        try:
            logger.info(f"GI lookup: '{normalized}' not in database, using AI prediction")
            result = await self._query_gpt(food_text, carbs, protein, fat)
            self._cache[normalized] = result
            return result
        except Exception as e:
            logger.error(f"GPT food analysis failed: {e}")
            return self._estimate_from_macros(carbs, protein, fat)

    async def _query_gpt(
        self,
        food_text: str,
        carbs: float,
        protein: float,
        fat: float
    ) -> FoodFeatures:
        """Query GPT for PRECISE food GI analysis with liquid detection."""

        system_prompt = """You are a nutrition expert with PRECISE knowledge of glycemic index values.

IMPORTANT: Provide EXACT GI numbers, not just categories. Examples:
- Pure glucose = 100
- Chocolate milk = 84 (high sugar + liquid = very fast spike)
- White bread = 75
- White rice = 73
- Banana = 51
- Apple = 36
- Regular milk = 39

For each food, provide:
1. glycemicIndex (0-100): PRECISE value based on food science research
2. isLiquid (true/false): Is this a drink? Liquids absorb 40% faster!
3. absorptionRate: How quickly carbs enter bloodstream
   - "very_fast": glucose, juice, chocolate milk, sports drinks
   - "fast": white bread, candy, potatoes
   - "medium": fruits, rice, pasta
   - "slow": whole grains, legumes
   - "very_slow": high-fat meals (pizza, burgers)
4. fatContent: "none", "low", "medium", or "high"

Return ONLY valid JSON, no other text."""

        user_prompt = f"""Analyze this food for PRECISE blood glucose impact:

Food: "{food_text}"
Known macros: {carbs}g carbs, {protein}g protein, {fat}g fat

Return JSON with PRECISE GI value:
{{
    "glycemicIndex": <EXACT number 0-100>,
    "isLiquid": <true if drink/liquid, false otherwise>,
    "absorptionRate": "<very_fast|fast|medium|slow|very_slow>",
    "fatContent": "<none|low|medium|high>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation with GI source>"
}}"""

        response = await self.client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Lower temp for more consistent/accurate GI values
            max_completion_tokens=250,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        gi = int(data.get("glycemicIndex", 55))
        gi = max(0, min(120, gi))  # Clamp to valid range (allow up to 120 for maltodextrin)

        return FoodFeatures(
            glycemic_index=gi,
            glycemic_load=carbs * gi / 100 if carbs > 0 else 0,
            absorption_rate=data.get("absorptionRate", "medium"),
            fat_content=data.get("fatContent", "low"),
            is_liquid=bool(data.get("isLiquid", False)),
            confidence=float(data.get("confidence", 0.7)),
            reasoning=data.get("reasoning")
        )

    def _lookup_known_food(
        self,
        text: str,
        carbs: float,
        fat: float
    ) -> Optional[FoodFeatures]:
        """Look up food in known database with precise GI and liquid detection."""
        text_lower = text.lower()

        for food_name, props in self.KNOWN_FOODS.items():
            if food_name in text_lower:
                gi = props["gi"]
                rate = props["rate"]
                fat_level = props["fat"]
                is_liquid = props.get("liquid", False)

                # Adjust absorption rate if high fat content logged
                if fat > 15:
                    if rate in ("very_fast", "fast"):
                        rate = "medium"
                    elif rate == "medium":
                        rate = "slow"
                    fat_level = "high"
                elif fat > 5:
                    fat_level = "medium"
                    if rate == "very_fast":
                        rate = "fast"

                return FoodFeatures(
                    glycemic_index=gi,
                    glycemic_load=carbs * gi / 100 if carbs > 0 else 0,
                    absorption_rate=rate,
                    fat_content=fat_level,
                    is_liquid=is_liquid,
                    confidence=0.9,  # High confidence for known foods with precise GI
                    reasoning=f"Matched known food: {food_name} (GI={gi}, liquid={is_liquid})"
                )

        return None

    def _estimate_from_macros(
        self,
        carbs: float,
        protein: float,
        fat: float
    ) -> FoodFeatures:
        """Estimate features from macro composition when GPT unavailable."""
        # Default medium GI
        gi = 55
        rate = "medium"
        fat_level = "low"

        # High fat delays absorption
        if fat > 15:
            rate = "slow"
            fat_level = "high"
            gi = 50  # Fat lowers effective GI
        elif fat > 5:
            fat_level = "medium"
            gi = 52

        # High protein also slows absorption
        if protein > 20:
            rate = "slow"
            gi = max(40, gi - 5)

        return FoodFeatures(
            glycemic_index=gi,
            glycemic_load=carbs * gi / 100 if carbs > 0 else 0,
            absorption_rate=rate,
            fat_content=fat_level,
            is_liquid=False,  # Can't detect from macros alone
            confidence=0.5,  # Lower confidence for estimates
            reasoning="Estimated from macro composition"
        )

    async def _infer_from_macros(
        self,
        carbs: float,
        protein: float,
        fat: float
    ) -> FoodFeatures:
        """Use GPT to infer food type and GI from macro composition."""
        await self._ensure_initialized()

        if not self.client:
            return self._estimate_from_macros(carbs, protein, fat)

        # Create cache key from macros
        cache_key = f"macros_{int(carbs)}_{int(protein)}_{int(fat)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            system_prompt = """You are a nutrition expert. Given only macro nutrients, identify the most likely food and its glycemic properties.

Common macro patterns:
- 29g carbs, 13g protein, 10g fat → Breakfast sandwich (egg, cheese, bread/muffin)
- 30g carbs, 0g protein, 0g fat → Juice, soda, or candy
- 45g carbs, 3g protein, 1g fat → Rice, pasta, or bread
- 20g carbs, 8g protein, 5g fat → Chocolate milk
- 40g carbs, 5g protein, 20g fat → Pizza slice, burger, or donut

Return ONLY valid JSON with your analysis."""

            user_prompt = f"""Based on these macros, identify the most likely food:
- Carbs: {carbs}g
- Protein: {protein}g
- Fat: {fat}g

Return JSON:
{{
    "likelyFood": "<most likely food based on macros>",
    "glycemicIndex": <0-100>,
    "isLiquid": <true/false>,
    "absorptionRate": "<very_fast|fast|medium|slow|very_slow>",
    "fatContent": "<none|low|medium|high>",
    "confidence": <0.0-1.0>,
    "reasoning": "<why you think it's this food>"
}}"""

            response = await self.client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=250,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            likely_food = data.get("likelyFood", "unknown")
            gi = int(data.get("glycemicIndex", 55))
            gi = max(0, min(100, gi))

            result = FoodFeatures(
                glycemic_index=gi,
                glycemic_load=carbs * gi / 100 if carbs > 0 else 0,
                absorption_rate=data.get("absorptionRate", "medium"),
                fat_content=data.get("fatContent", "medium"),
                is_liquid=data.get("isLiquid", False),
                confidence=float(data.get("confidence", 0.6)),
                reasoning=f"Inferred from macros: likely {likely_food} - {data.get('reasoning', '')}"
            )

            self._cache[cache_key] = result
            logger.info(
                f"Macro inference: {carbs}g C/{protein}g P/{fat}g F -> "
                f"likely '{likely_food}', GI={gi}"
            )
            return result

        except Exception as e:
            logger.error(f"Macro inference failed: {e}")
            return self._estimate_from_macros(carbs, protein, fat)

    def _default_features(self, carbs: float) -> FoodFeatures:
        """Return default features when no food info available."""
        return FoodFeatures(
            glycemic_index=55,
            glycemic_load=carbs * 55 / 100 if carbs > 0 else 0,
            absorption_rate="medium",
            fat_content="low",
            is_liquid=False,
            confidence=0.3,
            reasoning="Default values - no food description provided"
        )

    async def enrich_treatment(
        self,
        treatment_id: str,
        food_text: str,
        carbs: float = 0,
        protein: float = 0,
        fat: float = 0
    ) -> Dict[str, Any]:
        """
        Enrich a treatment with precise food features including GI and liquid detection.

        Returns dict of fields to update on the treatment.
        """
        features = await self.extract_food_features(food_text, carbs, protein, fat)

        return {
            "glycemicIndex": features.glycemic_index,
            "glycemicLoad": features.glycemic_load,
            "absorptionRate": features.absorption_rate,
            "fatContent": features.fat_content,
            "isLiquid": features.is_liquid,
            "enrichedAt": datetime.utcnow().isoformat()
        }

    def clear_cache(self):
        """Clear the food features cache."""
        self._cache.clear()

    async def estimate_macros_from_description(
        self,
        food_description: str,
        known_carbs: Optional[float] = None,
    ) -> MacroEstimate:
        """
        Estimate ALL macros (carbs, protein, fat) from a food description using GPT-4.1.

        This is the key method for enriching treatment data - it takes a food note
        like "2 slices pepperoni pizza" and estimates the full nutritional profile.

        Args:
            food_description: User's food note (e.g., "hot dog with bun", "birthday cake slice")
            known_carbs: If carbs were already logged, use this to improve estimates

        Returns:
            MacroEstimate with carbs, protein, fat, GI, and absorption characteristics
        """
        if not food_description or not food_description.strip():
            return self._default_macro_estimate(known_carbs)

        await self._ensure_initialized()

        if not self.client:
            logger.warning("GPT client not available for macro estimation")
            return self._default_macro_estimate(known_carbs)

        try:
            return await self._query_gpt_for_macros(food_description, known_carbs)
        except Exception as e:
            logger.error(f"GPT macro estimation failed: {e}")
            return self._default_macro_estimate(known_carbs)

    async def _query_gpt_for_macros(
        self,
        food_description: str,
        known_carbs: Optional[float] = None,
    ) -> MacroEstimate:
        """Query GPT-4.1 to estimate macros from food description."""

        system_prompt = """You are a nutrition expert for diabetes management.
Analyze food descriptions and estimate nutritional content for blood glucose prediction.

IMPORTANT: Users may add shorthand notation for macros they know:
- "(10gP)" or "(10P)" = 10 grams of protein - USE THIS VALUE
- "(15gF)" or "(15F)" = 15 grams of fat - USE THIS VALUE
- "10g protein" or "with 10g protein" = 10 grams protein - USE THIS VALUE
- Numbers in parentheses with P/F/C typically indicate known macro values

For each food, estimate:
1. Carbohydrates (grams): Total digestible carbs
2. Protein (grams): Protein content - USE USER-PROVIDED VALUE IF PRESENT
3. Fat (grams): Total fat content - THIS IS CRITICAL as fat slows carb absorption
4. Glycemic Index (0-100): How fast carbs raise blood sugar
5. Absorption Rate: "fast", "medium", or "slow"
   - fast: sugary drinks, candy, white bread (absorption half-life ~30 min)
   - medium: most meals, fruits, rice (absorption half-life ~45 min)
   - slow: high-fat/protein meals like pizza (absorption half-life ~75+ min)

Consider portion sizes mentioned in the description.
If a carb amount is provided, use it but still estimate protein and fat.
If user provides explicit macro values in notation like (10gP), USE THOSE EXACT VALUES.

Return ONLY valid JSON with no extra text."""

        carb_context = f"\nUser logged {known_carbs}g carbs - use this for carb estimate but estimate protein/fat." if known_carbs else ""

        user_prompt = f"""Analyze this food and estimate its nutritional content:

Food description: "{food_description}"{carb_context}

Return JSON:
{{
    "carbs_g": <estimated carbohydrates in grams>,
    "protein_g": <estimated protein in grams>,
    "fat_g": <estimated fat in grams>,
    "glycemic_index": <0-100>,
    "absorption_rate": "<fast|medium|slow>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation of estimate>"
}}"""

        response = await self.client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Low temp for consistent nutrition estimates
            max_completion_tokens=300,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Parse and validate
        carbs = float(data.get("carbs_g", known_carbs or 30))
        protein = float(data.get("protein_g", 10))
        fat = float(data.get("fat_g", 5))
        gi = int(data.get("glycemic_index", 55))
        gi = max(0, min(100, gi))
        rate = data.get("absorption_rate", "medium")
        confidence = float(data.get("confidence", 0.7))

        # Calculate absorption half-life based on rate and fat content
        if rate == "fast":
            half_life = 30.0
        elif rate == "slow":
            half_life = 75.0
        else:
            half_life = 45.0

        # Fat further slows absorption
        if fat > 20:
            half_life += 20
        elif fat > 10:
            half_life += 10

        return MacroEstimate(
            carbs_g=carbs,
            protein_g=protein,
            fat_g=fat,
            glycemic_index=gi,
            absorption_rate=rate,
            absorption_half_life_min=half_life,
            confidence=confidence,
            reasoning=data.get("reasoning")
        )

    def _default_macro_estimate(self, known_carbs: Optional[float] = None) -> MacroEstimate:
        """Return default macro estimate when GPT unavailable."""
        return MacroEstimate(
            carbs_g=known_carbs or 30.0,
            protein_g=10.0,
            fat_g=5.0,
            glycemic_index=55,
            absorption_rate="medium",
            absorption_half_life_min=45.0,
            confidence=0.3,
            reasoning="Default estimate - GPT unavailable"
        )

    async def batch_enrich_treatments(
        self,
        treatments: list,
        progress_callback=None,
    ) -> list:
        """
        Batch enrich multiple treatments with macro estimates.

        Args:
            treatments: List of treatment dicts with 'notes' and optionally 'carbs'
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of MacroEstimate results in same order as input
        """
        results = []
        total = len(treatments)

        for i, treatment in enumerate(treatments):
            notes = treatment.get('notes', '')
            carbs = treatment.get('carbs')

            if notes and notes.strip():
                estimate = await self.estimate_macros_from_description(notes, carbs)
            else:
                estimate = self._default_macro_estimate(carbs)

            results.append(estimate)

            if progress_callback:
                progress_callback(i + 1, total)

        return results


# Singleton instance
food_enrichment_service = FoodEnrichmentService()
