"""
Food Absorption Learner Service for T1D-AI

Learns per-food absorption patterns from actual BG response data.
Uses Bayesian updates to improve estimates over time.

Cold Start Strategy (use ALL approaches):
1. Formula baseline (always available)
2. Similar food matching (find similar foods with learned data)
3. GPT-assisted estimation (ask GPT based on food composition)

Learning threshold: 5 samples per food minimum
"""
import math
import logging
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass

from models.schemas import (
    MLTrainingDataPoint, FoodAbsorptionProfile
)
from database.repositories import (
    MLTrainingDataRepository, FoodAbsorptionProfileRepository
)

logger = logging.getLogger(__name__)


# Minimum samples required before using learned adjustments
MIN_SAMPLES_FOR_LEARNING = 5

# Maximum adjustment from baseline (prevents runaway learning)
MAX_ADJUSTMENT_FACTOR = 2.0
MIN_ADJUSTMENT_FACTOR = 0.5


@dataclass
class AbsorptionEstimate:
    """Result of absorption estimation."""
    onset_multiplier: float
    half_life_multiplier: float
    confidence: float
    source: str  # "learned", "similar", "gpt_estimated", "formula"
    sample_count: int = 0


class FoodAbsorptionLearner:
    """
    Learns personalized food absorption patterns from actual BG responses.

    This service:
    1. Tracks prediction errors for each food
    2. Updates absorption multipliers using Bayesian learning
    3. Provides estimates via multiple cold-start strategies
    4. Maintains per-user, per-food profiles
    """

    def __init__(
        self,
        profile_repo: Optional[FoodAbsorptionProfileRepository] = None,
        training_repo: Optional[MLTrainingDataRepository] = None,
        openai_service = None
    ):
        self.profile_repo = profile_repo or FoodAbsorptionProfileRepository()
        self.training_repo = training_repo or MLTrainingDataRepository()
        self._openai_service = openai_service

    async def get_absorption_multiplier(
        self,
        user_id: str,
        food_id: str,
        food_description: str = ""
    ) -> AbsorptionEstimate:
        """
        Get the best available absorption estimate for a food.

        Uses cascading cold-start strategy:
        1. Learned profile (if 5+ samples)
        2. Similar food matching (if any similar foods have data)
        3. GPT estimation (if description available)
        4. Formula baseline (always)

        Args:
            user_id: User ID
            food_id: Hash of normalized food description
            food_description: Original food description (for GPT/similar matching)

        Returns:
            AbsorptionEstimate with multipliers and source
        """
        # Strategy 1: Check for learned profile
        profile = await self.profile_repo.get_by_food(user_id, food_id)
        if profile and profile.sampleCount >= MIN_SAMPLES_FOR_LEARNING:
            confidence = self._calculate_confidence(profile)
            return AbsorptionEstimate(
                onset_multiplier=profile.onsetMultiplier,
                half_life_multiplier=profile.halfLifeMultiplier,
                confidence=confidence,
                source="learned",
                sample_count=profile.sampleCount
            )

        # Strategy 2: Try similar food matching
        if food_description:
            similar_estimate = await self._find_similar_food_estimate(
                user_id, food_description
            )
            if similar_estimate:
                return similar_estimate

        # Strategy 3: Try GPT estimation (if available and description provided)
        if self._openai_service and food_description:
            gpt_estimate = await self._get_gpt_estimate(food_description)
            if gpt_estimate:
                return gpt_estimate

        # Strategy 4: Formula baseline
        return AbsorptionEstimate(
            onset_multiplier=1.0,
            half_life_multiplier=1.0,
            confidence=0.0,
            source="formula",
            sample_count=0
        )

    async def update_from_training_data(
        self,
        data_point: MLTrainingDataPoint
    ) -> Optional[FoodAbsorptionProfile]:
        """
        Update the food absorption profile based on a completed training data point.

        Uses Bayesian updating to incorporate the new observation:
        - Prior: existing profile multipliers
        - Likelihood: new observation's implied multiplier
        - Posterior: weighted combination based on confidence

        Args:
            data_point: Completed MLTrainingDataPoint with actual BGs

        Returns:
            Updated FoodAbsorptionProfile or None if update failed
        """
        if not data_point.isComplete:
            logger.warning(f"Cannot update from incomplete data point: {data_point.id}")
            return None

        user_id = data_point.userId
        food_id = data_point.foodId

        # Calculate implied multipliers from prediction errors
        implied_onset, implied_half_life = self._calculate_implied_multipliers(data_point)

        # Get existing profile or create new one
        profile = await self.profile_repo.get_by_food(user_id, food_id)

        if profile is None:
            # Create new profile
            profile = FoodAbsorptionProfile(
                id=f"{user_id}_{food_id}",
                userId=user_id,
                foodId=food_id,
                foodDescription=data_point.foodDescription,
                onsetMultiplier=implied_onset,
                halfLifeMultiplier=implied_half_life,
                sampleCount=1,
                meanError=self._calculate_mean_error(data_point),
                stdError=0.0,
                confidence=0.1,
                source="learned"
            )
        else:
            # Bayesian update of existing profile
            n = profile.sampleCount
            new_n = n + 1

            # Weighted average (more weight to existing as sample count grows)
            prior_weight = n / (n + 1)
            likelihood_weight = 1 / (n + 1)

            profile.onsetMultiplier = (
                prior_weight * profile.onsetMultiplier +
                likelihood_weight * implied_onset
            )
            profile.halfLifeMultiplier = (
                prior_weight * profile.halfLifeMultiplier +
                likelihood_weight * implied_half_life
            )

            # Clamp to reasonable bounds
            profile.onsetMultiplier = max(
                MIN_ADJUSTMENT_FACTOR,
                min(MAX_ADJUSTMENT_FACTOR, profile.onsetMultiplier)
            )
            profile.halfLifeMultiplier = max(
                MIN_ADJUSTMENT_FACTOR,
                min(MAX_ADJUSTMENT_FACTOR, profile.halfLifeMultiplier)
            )

            # Update statistics
            new_error = self._calculate_mean_error(data_point)
            profile.meanError = (
                prior_weight * profile.meanError +
                likelihood_weight * new_error
            )

            # Update std error (running approximation)
            if n > 1:
                old_variance = profile.stdError ** 2
                new_variance = old_variance + (new_error - profile.meanError) ** 2 / new_n
                profile.stdError = math.sqrt(new_variance)
            else:
                profile.stdError = abs(new_error - profile.meanError)

            profile.sampleCount = new_n
            profile.confidence = self._calculate_confidence(profile)
            profile.source = "learned"
            profile.lastUpdated = datetime.utcnow()

        # Save the updated profile
        updated = await self.profile_repo.upsert(profile)
        logger.info(
            f"Updated food profile {food_id}: "
            f"onset={profile.onsetMultiplier:.2f}x, "
            f"half_life={profile.halfLifeMultiplier:.2f}x, "
            f"samples={profile.sampleCount}, "
            f"confidence={profile.confidence:.2f}"
        )

        return updated

    async def process_completed_training_data(self, user_id: str) -> int:
        """
        Process all completed training data points that haven't been used for learning.

        This should be called periodically to incorporate new observations.

        Args:
            user_id: User ID

        Returns:
            Number of profiles updated
        """
        # Get recent completed training data
        completed = await self.training_repo.get_recent_complete(user_id, days=30)

        updated = 0
        for dp in completed:
            try:
                result = await self.update_from_training_data(dp)
                if result:
                    updated += 1
            except Exception as e:
                logger.error(f"Failed to process training data {dp.id}: {e}")

        logger.info(f"Processed {len(completed)} training points, updated {updated} profiles")
        return updated

    async def get_all_learned_foods(
        self,
        user_id: str,
        min_samples: int = MIN_SAMPLES_FOR_LEARNING
    ) -> List[FoodAbsorptionProfile]:
        """
        Get all foods that have learned absorption profiles.

        Args:
            user_id: User ID
            min_samples: Minimum samples to be considered "learned"

        Returns:
            List of FoodAbsorptionProfile
        """
        return await self.profile_repo.get_learned_profiles(user_id, min_samples)

    def _calculate_implied_multipliers(
        self,
        data_point: MLTrainingDataPoint
    ) -> Tuple[float, float]:
        """
        Calculate what the absorption multipliers should be to match actual BG.

        Uses the prediction errors to estimate the implied adjustment:
        - Positive error (actual > predicted) means absorption was faster
        - Negative error (actual < predicted) means absorption was slower
        """
        # Use 30-min error for onset estimation
        error_30 = data_point.error30 or 0
        error_60 = data_point.error60 or 0

        # Positive error means BG rose faster than expected
        # This implies faster onset (lower onset multiplier) and faster absorption
        bg_rise_expected = data_point.predictedBg60 - data_point.bgAtMeal
        if bg_rise_expected == 0:
            bg_rise_expected = 1  # Prevent division by zero

        # Calculate adjustment factor
        # If actual rise was 20% more than expected, absorption was ~20% faster
        error_ratio_30 = 1.0 + (error_30 / max(30, abs(bg_rise_expected)))
        error_ratio_60 = 1.0 + (error_60 / max(30, abs(bg_rise_expected)))

        # Onset multiplier: positive error means faster onset (lower multiplier)
        # If BG rose faster at 30min, onset was quicker
        implied_onset = 1.0 / max(0.5, min(2.0, error_ratio_30))

        # Half-life multiplier: compare 30 vs 60 min errors
        # If error grows from 30→60, decay is slower than expected
        if error_60 > error_30:
            # More error at 60min means slower decay
            implied_half_life = 1.0 + (error_60 - error_30) / max(30, abs(bg_rise_expected))
        else:
            # Less error at 60min means faster decay
            implied_half_life = 1.0 - (error_30 - error_60) / max(30, abs(bg_rise_expected))

        # Clamp to reasonable bounds
        implied_onset = max(MIN_ADJUSTMENT_FACTOR, min(MAX_ADJUSTMENT_FACTOR, implied_onset))
        implied_half_life = max(MIN_ADJUSTMENT_FACTOR, min(MAX_ADJUSTMENT_FACTOR, implied_half_life))

        return implied_onset, implied_half_life

    def _calculate_mean_error(self, data_point: MLTrainingDataPoint) -> float:
        """Calculate the mean absolute error for a data point."""
        errors = []
        if data_point.error30 is not None:
            errors.append(abs(data_point.error30))
        if data_point.error60 is not None:
            errors.append(abs(data_point.error60))
        if data_point.error90 is not None:
            errors.append(abs(data_point.error90))

        return sum(errors) / len(errors) if errors else 0

    def _calculate_confidence(self, profile: FoodAbsorptionProfile) -> float:
        """
        Calculate confidence score for a profile.

        Based on:
        - Sample count (more samples = higher confidence)
        - Consistency (lower std error = higher confidence)
        - Recency (not implemented yet)
        """
        # Sample count contribution (saturates around 20 samples)
        sample_factor = min(1.0, profile.sampleCount / 20.0)

        # Consistency contribution (lower error = higher confidence)
        # Assume 20 mg/dL is acceptable error, 50+ is poor
        if profile.stdError < 20:
            consistency_factor = 1.0
        elif profile.stdError > 50:
            consistency_factor = 0.3
        else:
            consistency_factor = 1.0 - (profile.stdError - 20) / 50

        # Combined confidence
        confidence = sample_factor * 0.7 + consistency_factor * 0.3

        return round(min(1.0, max(0.0, confidence)), 2)

    async def _find_similar_food_estimate(
        self,
        user_id: str,
        food_description: str
    ) -> Optional[AbsorptionEstimate]:
        """
        Find a similar food with learned data and use its estimate.

        Uses keyword matching to find foods with similar descriptions.
        The estimate is weighted by the similar food's confidence.
        """
        # Extract keywords from description
        keywords = self._extract_food_keywords(food_description)
        if not keywords:
            return None

        # Search for similar foods
        similar_profiles = await self.profile_repo.search_similar(user_id, keywords)

        if not similar_profiles:
            return None

        # Use the best match (highest confidence)
        best = similar_profiles[0]

        # Reduce confidence for similar match (not exact)
        adjusted_confidence = best.confidence * 0.7

        return AbsorptionEstimate(
            onset_multiplier=best.onsetMultiplier,
            half_life_multiplier=best.halfLifeMultiplier,
            confidence=adjusted_confidence,
            source="similar",
            sample_count=best.sampleCount
        )

    def _extract_food_keywords(self, description: str) -> List[str]:
        """Extract meaningful food keywords from description."""
        # Common food categories
        food_categories = [
            'pizza', 'pasta', 'rice', 'bread', 'sandwich', 'burger', 'taco',
            'chicken', 'beef', 'fish', 'salad', 'soup', 'cereal', 'oatmeal',
            'yogurt', 'fruit', 'apple', 'banana', 'orange', 'milk', 'juice',
            'candy', 'chocolate', 'ice cream', 'cake', 'cookie', 'donut',
            'french fries', 'potato', 'beans', 'vegetables'
        ]

        description_lower = description.lower()
        found_keywords = []

        for category in food_categories:
            if category in description_lower:
                found_keywords.append(category)

        return found_keywords

    async def _get_gpt_estimate(
        self,
        food_description: str
    ) -> Optional[AbsorptionEstimate]:
        """
        Get absorption estimate from GPT based on food composition.

        This is a fallback for completely new foods with no similar matches.
        """
        if not self._openai_service:
            return None

        try:
            # Use GPT to estimate absorption characteristics
            # This would call the OpenAI service with a prompt like:
            # "Given the food '{description}', estimate absorption speed..."
            # For now, return None until integrated with OpenAI service
            return None
        except Exception as e:
            logger.warning(f"GPT estimation failed: {e}")
            return None


# Singleton instance
_learner: Optional[FoodAbsorptionLearner] = None


def get_food_absorption_learner() -> FoodAbsorptionLearner:
    """Get the singleton food absorption learner instance."""
    global _learner
    if _learner is None:
        _learner = FoodAbsorptionLearner()
    return _learner
