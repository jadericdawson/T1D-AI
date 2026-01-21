"""
Hybrid Absorption Model for T1D-AI

Three-tier absorption calculation:
1. Physics formula (baseline, always available)
2. Macro modifiers (fat/protein/fiber delays)
3. Per-food learned adjustments (when 5+ samples exist)

This model provides more accurate carb absorption predictions by:
- Using continuous GI-based formula (no hard-coded buckets)
- Applying physiological modifiers for macronutrients
- Learning from actual BG response data for frequently eaten foods
"""
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionParams:
    """Parameters describing carb absorption curve."""
    onset_min: float      # Time before carbs start affecting BG
    ramp_min: float       # Duration of effect build-up phase
    half_life_min: float  # Decay rate after peak
    duration_min: float   # Total duration of absorption
    gi_factor: float      # Combined GI and modifiers factor
    source: str           # How these params were derived


class HybridAbsorptionModel:
    """
    Three-tier absorption model for accurate carb absorption prediction.

    Tier 1: Physics Formula (always available)
    - Continuous GI-based calculation (no hard-coded buckets)
    - Based on physiological research on carbohydrate absorption kinetics

    Tier 2: Macro Modifiers (applied when macros are known)
    - Fat: Delays gastric emptying (up to 50% delay for high-fat meals)
    - Protein: Moderate delay effect (up to 30% delay)
    - Fiber: Slows absorption (up to 20% delay)

    Tier 3: Per-Food Learned (when 5+ samples exist)
    - Bayesian-updated multipliers from actual BG response data
    - Gradually replaces formula estimates as confidence increases
    """

    # Baseline parameters for medium GI (55), solid food
    BASE_ONSET = 10.0       # minutes until carbs start affecting BG
    BASE_RAMP = 15.0        # minutes to ramp up to peak
    BASE_HALF_LIFE = 40.0   # minutes for exponential decay
    BASE_DURATION = 180.0   # total duration of effect

    # Minimum physiological bounds (can't absorb faster than this)
    MIN_ONSET = 2.0
    MIN_RAMP = 3.0
    MIN_HALF_LIFE = 10.0
    MIN_DURATION = 60.0

    # Macro modifier coefficients (derived from research)
    FAT_DELAY_COEFF = 0.5       # Up to 50% delay for high fat
    FAT_SATURATION_G = 30.0     # Grams at which delay saturates
    PROTEIN_DELAY_COEFF = 0.3   # Up to 30% delay for high protein
    PROTEIN_SATURATION_G = 40.0
    FIBER_DELAY_COEFF = 0.2     # Up to 20% delay for high fiber
    FIBER_SATURATION_G = 15.0

    def __init__(self, food_profile_getter=None):
        """
        Initialize the hybrid absorption model.

        Args:
            food_profile_getter: Async function to get learned food profiles
                                 Signature: async def get_profile(user_id, food_id) -> Optional[FoodAbsorptionProfile]
        """
        self._get_food_profile = food_profile_getter

    def calculate_physics_baseline(
        self,
        glycemic_index: float,
        is_liquid: bool = False
    ) -> AbsorptionParams:
        """
        Tier 1: Physics-based absorption parameters.

        Uses continuous GI-based formula (no hard-coded buckets).
        The relationship is NON-LINEAR - high GI foods spike dramatically faster.

        GI Reference:
        - Very high (>85): Glucose, juice, sugary drinks - onset 2-5 min
        - High (70-85): White bread, potatoes - onset 5-10 min
        - Medium (55-70): Rice, pasta - onset 10-15 min
        - Low (<55): Beans, vegetables - onset 15-25 min

        Args:
            glycemic_index: GI value (0-100+)
            is_liquid: True for liquid carbs (absorb 40% faster)

        Returns:
            AbsorptionParams with baseline physics calculation
        """
        # Clamp GI to reasonable range
        gi = max(20, min(100, glycemic_index))

        # Non-linear scaling: exponential relationship for high GI foods
        # At GI=55 (medium), factor=1.0
        # At GI=85 (high), factor≈2.0 (twice as fast)
        # At GI=35 (low), factor≈0.6 (40% slower)
        gi_factor = math.exp((gi - 55) / 30)

        # Liquid carbs absorb 40% faster (no mechanical digestion needed)
        liquid_factor = 1.4 if is_liquid else 1.0

        combined_factor = gi_factor * liquid_factor

        # Apply factors (higher factor = faster absorption = lower times)
        onset = self.BASE_ONSET / combined_factor
        ramp = self.BASE_RAMP / combined_factor
        half_life = self.BASE_HALF_LIFE / combined_factor
        duration = self.BASE_DURATION / (combined_factor ** 0.5)  # Duration scales slower

        # Enforce physiological minimums
        onset = max(self.MIN_ONSET, onset)
        ramp = max(self.MIN_RAMP, ramp)
        half_life = max(self.MIN_HALF_LIFE, half_life)
        duration = max(self.MIN_DURATION, duration)

        return AbsorptionParams(
            onset_min=round(onset, 1),
            ramp_min=round(ramp, 1),
            half_life_min=round(half_life, 1),
            duration_min=round(duration, 1),
            gi_factor=round(combined_factor, 2),
            source="physics"
        )

    def calculate_macro_modifiers(
        self,
        fat_g: float = 0,
        protein_g: float = 0,
        fiber_g: float = 0
    ) -> float:
        """
        Tier 2: Macro-based absorption delay modifier.

        Fat, protein, and fiber all delay gastric emptying and slow
        carbohydrate absorption. This returns a multiplier that
        increases absorption times (> 1.0 = slower).

        Based on research:
        - Fat: Strong effect on gastric emptying, saturates around 30g
        - Protein: Moderate effect, saturates around 40g
        - Fiber: Slows intestinal absorption, saturates around 15g

        Args:
            fat_g: Fat content in grams
            protein_g: Protein content in grams
            fiber_g: Fiber content in grams

        Returns:
            Multiplier for absorption times (1.0 = no change, >1.0 = slower)
        """
        # Each macro contributes a delay factor that saturates
        # Using sigmoid-like saturation: delay = coeff * (amount / (amount + saturation))

        fat_delay = 1.0 + self.FAT_DELAY_COEFF * (fat_g / (fat_g + self.FAT_SATURATION_G)) if fat_g > 0 else 1.0
        protein_delay = 1.0 + self.PROTEIN_DELAY_COEFF * (protein_g / (protein_g + self.PROTEIN_SATURATION_G)) if protein_g > 0 else 1.0
        fiber_delay = 1.0 + self.FIBER_DELAY_COEFF * (fiber_g / (fiber_g + self.FIBER_SATURATION_G)) if fiber_g > 0 else 1.0

        # Combine multiplicatively
        combined_modifier = fat_delay * protein_delay * fiber_delay

        logger.debug(
            f"Macro modifiers - fat:{fat_g}g→{fat_delay:.2f}x, "
            f"protein:{protein_g}g→{protein_delay:.2f}x, "
            f"fiber:{fiber_g}g→{fiber_delay:.2f}x = {combined_modifier:.2f}x"
        )

        return combined_modifier

    async def get_learned_adjustment(
        self,
        user_id: str,
        food_id: str
    ) -> Tuple[float, float, str]:
        """
        Tier 3: Per-food learned adjustment from actual BG response data.

        Returns multipliers learned from the user's actual BG responses
        to this specific food. Only returns meaningful adjustments when
        there are 5+ clean observations for this food.

        Args:
            user_id: User ID
            food_id: Hash of normalized food description

        Returns:
            Tuple of (onset_multiplier, half_life_multiplier, source)
            - source: "learned" if 5+ samples, "similar" for similar food match,
                      "gpt_estimated" for GPT estimate, "formula" for baseline
        """
        if self._get_food_profile is None:
            return 1.0, 1.0, "formula"

        try:
            profile = await self._get_food_profile(user_id, food_id)

            if profile and profile.sampleCount >= 5:
                # Good learned profile - use it with confidence weighting
                confidence_weight = min(1.0, profile.sampleCount / 20.0)  # Full confidence at 20+ samples
                onset_mult = 1.0 + (profile.onsetMultiplier - 1.0) * confidence_weight
                half_life_mult = 1.0 + (profile.halfLifeMultiplier - 1.0) * confidence_weight

                logger.info(
                    f"Using learned profile for food {food_id}: "
                    f"onset={onset_mult:.2f}x, half_life={half_life_mult:.2f}x "
                    f"(samples={profile.sampleCount}, confidence={profile.confidence:.2f})"
                )
                return onset_mult, half_life_mult, "learned"

            # No learned profile - return baseline
            return 1.0, 1.0, "formula"

        except Exception as e:
            logger.warning(f"Failed to get learned profile: {e}")
            return 1.0, 1.0, "formula"

    async def calculate_absorption_params(
        self,
        glycemic_index: float,
        is_liquid: bool = False,
        fat_g: float = 0,
        protein_g: float = 0,
        fiber_g: float = 0,
        food_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AbsorptionParams:
        """
        Calculate final absorption parameters using all three tiers.

        This is the main entry point for absorption calculation.
        It combines:
        1. Physics-based baseline from GI
        2. Macro modifiers from fat/protein/fiber
        3. Per-food learned adjustments (if available)

        Args:
            glycemic_index: GI value (0-100+)
            is_liquid: True for liquid carbs
            fat_g: Fat content in grams
            protein_g: Protein content in grams
            fiber_g: Fiber content in grams
            food_id: Optional food ID for learned profile lookup
            user_id: Optional user ID for learned profile lookup

        Returns:
            Final AbsorptionParams combining all tiers
        """
        # Tier 1: Physics baseline
        baseline = self.calculate_physics_baseline(glycemic_index, is_liquid)

        # Tier 2: Macro modifiers
        macro_modifier = self.calculate_macro_modifiers(fat_g, protein_g, fiber_g)

        # Tier 3: Learned adjustment (if user/food provided)
        onset_learned = 1.0
        half_life_learned = 1.0
        source = "physics"

        if food_id and user_id:
            onset_learned, half_life_learned, source = await self.get_learned_adjustment(
                user_id, food_id
            )
            if source == "learned":
                source = "learned"
            elif macro_modifier > 1.01:
                source = "physics+macros"
            else:
                source = "physics"

        # Combine all factors
        final_onset = baseline.onset_min * macro_modifier * onset_learned
        final_ramp = baseline.ramp_min * macro_modifier
        final_half_life = baseline.half_life_min * macro_modifier * half_life_learned
        final_duration = baseline.duration_min * (macro_modifier ** 0.5)

        # Enforce physiological bounds
        final_onset = max(self.MIN_ONSET, round(final_onset, 1))
        final_ramp = max(self.MIN_RAMP, round(final_ramp, 1))
        final_half_life = max(self.MIN_HALF_LIFE, round(final_half_life, 1))
        final_duration = max(self.MIN_DURATION, round(final_duration, 1))

        logger.info(
            f"Absorption params (source={source}): "
            f"onset={final_onset}min, ramp={final_ramp}min, "
            f"half_life={final_half_life}min, duration={final_duration}min"
        )

        return AbsorptionParams(
            onset_min=final_onset,
            ramp_min=final_ramp,
            half_life_min=final_half_life,
            duration_min=final_duration,
            gi_factor=baseline.gi_factor,
            source=source
        )

    def calculate_cob_at_time(
        self,
        initial_carbs: float,
        time_elapsed_min: float,
        params: AbsorptionParams
    ) -> float:
        """
        Calculate remaining COB at a specific time using the absorption params.

        Uses three-phase pharmacokinetic model:
        1. Onset phase: Very slow decay (carbs not yet absorbed)
        2. Ramp phase: Moderate decay as carbs are absorbed
        3. Decay phase: Full exponential decay

        Args:
            initial_carbs: Carbs eaten in grams
            time_elapsed_min: Minutes since eating
            params: AbsorptionParams for this food

        Returns:
            Remaining COB in grams
        """
        if time_elapsed_min < 0 or time_elapsed_min > params.duration_min:
            return 0.0

        if time_elapsed_min < params.onset_min:
            # Pre-onset: Very slow decay (~5% during onset)
            decay_factor = 1.0 - (0.05 * time_elapsed_min / params.onset_min)
        elif time_elapsed_min < (params.onset_min + params.ramp_min):
            # Ramp phase: Decay from 95% to 50%
            ramp_progress = (time_elapsed_min - params.onset_min) / params.ramp_min
            decay_factor = 0.95 - (0.45 * ramp_progress)
        else:
            # Full decay phase: Exponential decay of remaining 50%
            decay_time = time_elapsed_min - params.onset_min - params.ramp_min
            decay_factor = 0.5 * (0.5 ** (decay_time / params.half_life_min))

        return initial_carbs * max(0, decay_factor)

    def calculate_absorbed_carbs(
        self,
        initial_carbs: float,
        time_elapsed_min: float,
        params: AbsorptionParams
    ) -> float:
        """
        Calculate carbs absorbed (available to affect BG) at a specific time.

        This is the complement of COB - how much has entered the bloodstream.

        Args:
            initial_carbs: Carbs eaten in grams
            time_elapsed_min: Minutes since eating
            params: AbsorptionParams for this food

        Returns:
            Absorbed carbs in grams
        """
        remaining = self.calculate_cob_at_time(initial_carbs, time_elapsed_min, params)
        return initial_carbs - remaining


def create_hybrid_model_with_profiles(food_profile_repo) -> HybridAbsorptionModel:
    """
    Factory function to create HybridAbsorptionModel with food profile repository.

    Args:
        food_profile_repo: FoodAbsorptionProfileRepository instance

    Returns:
        HybridAbsorptionModel configured with profile lookup
    """
    async def get_profile(user_id: str, food_id: str):
        return await food_profile_repo.get_by_food(user_id, food_id)

    return HybridAbsorptionModel(food_profile_getter=get_profile)
