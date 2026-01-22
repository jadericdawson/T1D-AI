"""
Glucose API Endpoints for T1D-AI
Provides glucose data, predictions, and metrics.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Depends

from models.schemas import (
    GlucoseReading, GlucoseWithPredictions, GlucoseCurrentResponse,
    GlucoseHistoryResponse, CurrentMetrics, PredictionAccuracy,
    GlucosePrediction, TFTPrediction, EffectPoint, HistoricalIobCobPoint,
    TrendDirection
)
from database.repositories import GlucoseRepository, TreatmentRepository, UserRepository, LearnedISFRepository, UserAbsorptionProfileRepository, LearnedPIRRepository, SharingRepository
from services.iob_cob_service import IOBCOBService
from models.schemas import UserAbsorptionProfile
from services.prediction_service import get_prediction_service, PredictionService
from services.dexcom_service import DexcomShareService, DexcomGlucoseReading
from services.metabolic_params_service import get_metabolic_params_service, MetabolicState
from config import get_settings
from auth.routes import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


def get_data_user_id(profile_id: str) -> str:
    """
    Convert a profile ID to the actual data user ID.

    For 'self' profiles, the profile ID is 'profile_{user_id}' but data
    is stored with just the raw user_id. This function strips the prefix.

    For non-self profiles (like children), the profile ID is a regular UUID
    that should be used as-is.
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix (8 chars)
    return profile_id


# Repository instances
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
user_repo = UserRepository()
learned_isf_repo = LearnedISFRepository()
learned_pir_repo = LearnedPIRRepository()
absorption_profile_repo = UserAbsorptionProfileRepository()
sharing_repo = SharingRepository()
iob_cob_service = IOBCOBService.from_settings()
metabolic_params_service = get_metabolic_params_service()

# Dexcom service for direct CGM data
dexcom_service: Optional[DexcomShareService] = None

def get_dexcom_service() -> Optional[DexcomShareService]:
    """Get or create Dexcom service (lazy init)."""
    global dexcom_service
    if dexcom_service is None:
        try:
            dexcom_service = DexcomShareService()
            logger.info("Initialized Dexcom Share service")
        except Exception as e:
            logger.warning(f"Failed to initialize Dexcom service: {e}")
    return dexcom_service

def dexcom_to_glucose_reading(dexcom_reading: DexcomGlucoseReading, user_id: str) -> GlucoseReading:
    """Convert Dexcom reading to GlucoseReading schema."""
    # Map trend description to TrendDirection
    trend_map = {
        "rising quickly": TrendDirection.DOUBLE_UP,
        "rising": TrendDirection.SINGLE_UP,
        "rising slightly": TrendDirection.FORTY_FIVE_UP,
        "steady": TrendDirection.FLAT,
        "falling slightly": TrendDirection.FORTY_FIVE_DOWN,
        "falling": TrendDirection.SINGLE_DOWN,
        "falling quickly": TrendDirection.DOUBLE_DOWN,
    }
    trend = trend_map.get(dexcom_reading.trend_description.lower(), TrendDirection.FLAT)

    return GlucoseReading(
        id=f"dexcom-{int(dexcom_reading.timestamp.timestamp() * 1000)}",
        userId=user_id,
        value=dexcom_reading.value,
        timestamp=dexcom_reading.timestamp,
        trend=trend,
        source="dexcom"
    )


async def validate_user_access(requester_id: str, target_user_id: str) -> bool:
    """
    Check if requester has access to target user's data.
    Access is granted if:
    - requester_id == target_user_id (viewing own data)
    - requester owns the target profile (managed profile)
    - requester is a parent with target_user_id in their linkedChildIds
    - target_user has shared their data with requester via the sharing system
    """
    try:
        # Normalize both IDs for comparison
        # Profile IDs may have 'profile_' prefix but user IDs don't
        normalized_requester = get_data_user_id(requester_id)
        normalized_target = get_data_user_id(target_user_id)

        logger.info(f"validate_user_access: requester_id={requester_id}, target_user_id={target_user_id}")
        logger.info(f"validate_user_access: normalized_requester={normalized_requester}, normalized_target={normalized_target}")

        if normalized_requester == normalized_target:
            logger.info(f"Access granted: normalized IDs match")
            return True

        # Also check raw IDs in case one has prefix and one doesn't
        if requester_id == target_user_id:
            logger.info(f"Access granted: raw IDs match")
            return True

        # Check if requester OWNS this profile (managed profile system)
        # This handles cases where a parent creates a profile for their child
        try:
            from database.repositories import ProfileRepository
            profile_repo = ProfileRepository()
            # Check if target_user_id is a profile owned by requester
            logger.info(f"Checking profile ownership: profile_id={target_user_id}, account_id={normalized_requester}")
            profile = await profile_repo.get_by_id(target_user_id, normalized_requester)
            if profile:
                logger.info(f"Access granted via profile ownership: {normalized_requester} owns profile {target_user_id}")
                return True
            else:
                logger.info(f"Profile not found: profile_id={target_user_id}, account_id={normalized_requester}")
        except Exception as e:
            logger.warning(f"Error checking profile ownership: {e}")

        # Check if requester is a parent of the target
        try:
            requester = await user_repo.get_by_id(requester_id)
            if requester and requester.linkedChildIds:
                # Check both normalized and raw target IDs
                if target_user_id in requester.linkedChildIds or normalized_target in requester.linkedChildIds:
                    return True
        except Exception as e:
            logger.warning(f"Error checking parent-child access: {e}")

        # Check if target user has shared data with requester
        # This handles two cases:
        # 1. Direct share: ownerId = target_user_id (user shares their own data)
        # 2. Profile share: profileId = target_user_id (parent shares child's data)
        try:
            # Try with original target_user_id first (for profile shares)
            share = await sharing_repo.get_share_for_profile(target_user_id, requester_id)
            if share and share.isActive:
                # Handle role as enum or string
                role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                logger.info(f"Access granted via share: {target_user_id} shared with {requester_id} (role: {role_str})")
                return True
            # Also try with normalized target (for direct user shares)
            if normalized_target != target_user_id:
                share = await sharing_repo.get_share_for_profile(normalized_target, requester_id)
                if share and share.isActive:
                    role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                    logger.info(f"Access granted via share: {normalized_target} shared with {requester_id} (role: {role_str})")
                    return True
        except Exception as e:
            logger.warning(f"Error checking share access: {e}")

        return False
    except Exception as e:
        logger.error(f"Unexpected error in validate_user_access: {e}")
        return False


def get_pred_service() -> PredictionService:
    """Get prediction service dependency."""
    settings = get_settings()
    models_dir = None
    for path in [Path("./models"), Path("./data/models"), Path("/app/models")]:
        if path.exists():
            models_dir = path
            break
    return get_prediction_service(models_dir, settings.model_device)


@router.get("/current", response_model=GlucoseCurrentResponse)
async def get_current_glucose(
    user_id: str = Query(..., description="User ID whose data to view"),
    current_user = Depends(get_current_user)
):
    """
    Get current glucose reading with predictions and metrics.

    Returns the latest glucose value along with:
    - ML predictions (Linear and LSTM) for 5, 10, 15 minutes ahead
    - IOB (Insulin on Board)
    - COB (Carbs on Board)
    - ISF (Insulin Sensitivity Factor)
    - Recommended correction dose

    Requires JWT authentication. Users can view their own data or their children's data.
    """
    try:
        # SECURITY: Always validate access using authenticated user
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")

        # Normalize profile ID to data user ID
        # For 'self' profiles, strip the 'profile_' prefix (data stored without it)
        data_user_id = get_data_user_id(user_id)
        logger.debug(f"Normalized user_id {user_id} -> data_user_id {data_user_id}")

        # Try Dexcom first for freshest data - BUT ONLY if this user has Dexcom configured
        # CRITICAL: The global Dexcom service has hardcoded credentials for one user (Emrys)
        # We must NOT return Dexcom data for users who don't have it set up
        latest: Optional[GlucoseReading] = None
        dexcom_svc = get_dexcom_service()

        if dexcom_svc:
            # Check if user has CGM data (dexcom OR gluroo source)
            # Dexcom is PRIMARY source, Gluroo is backup
            try:
                recent_dexcom = await glucose_repo.get_recent_by_source(data_user_id, "dexcom", hours=48)
                recent_gluroo = await glucose_repo.get_recent_by_source(data_user_id, "gluroo", hours=48)
                user_has_cgm = len(recent_dexcom) > 0 or len(recent_gluroo) > 0

                if user_has_cgm:
                    try:
                        dexcom_reading = await dexcom_svc.get_latest_reading_async()
                        if dexcom_reading:
                            latest = dexcom_to_glucose_reading(dexcom_reading, data_user_id)
                            logger.info(f"Got fresh reading from Dexcom (PRIMARY): {latest.value} mg/dL")
                    except Exception as e:
                        logger.warning(f"Dexcom fetch failed, will use Gluroo backup: {e}")
                else:
                    logger.debug(f"User {user_id} has no CGM data, skipping Dexcom fetch")
            except Exception as e:
                logger.warning(f"Error checking CGM status for user: {e}")

        # Fall back to Gluroo/database if Dexcom unavailable or stale
        gluroo_latest = await glucose_repo.get_latest(data_user_id)

        if gluroo_latest:
            # Use Gluroo if no Dexcom data, or if Gluroo is actually fresher
            if not latest:
                latest = gluroo_latest
                logger.info(f"Using Gluroo data: {latest.value} mg/dL")
            elif gluroo_latest.timestamp > latest.timestamp:
                latest = gluroo_latest
                logger.info(f"Gluroo data is fresher, using: {latest.value} mg/dL")
            else:
                logger.info(f"Using Dexcom (fresher than Gluroo)")

        if not latest:
            raise HTTPException(status_code=404, detail="No glucose data found")

        # Get recent treatments for IOB/COB calculation
        # Fetch 30 hours to cover 24hr chart view + 6hr insulin action window
        treatments = await treatment_repo.get_recent(data_user_id, hours=30)

        # Load user's learned absorption profile for personalized activity curves
        # This replaces hardcoded peak times (75/45/180 min) with learned values
        absorption_profile: Optional[UserAbsorptionProfile] = None
        try:
            absorption_profile = await absorption_profile_repo.get(data_user_id)
            if absorption_profile and absorption_profile.confidence > 0.3:
                logger.info(f"Using learned absorption profile: insulin_peak={absorption_profile.insulinPeakMin}min, "
                           f"carb_peak={absorption_profile.carbPeakMin}min, protein_peak={absorption_profile.proteinPeakMin}min "
                           f"(confidence={absorption_profile.confidence:.2f})")
            else:
                absorption_profile = None  # Low confidence, use defaults
                logger.debug("No learned absorption profile (or low confidence), using defaults")
        except Exception as e:
            logger.debug(f"Could not load absorption profile: {e}")

        # Determine peak times (learned or default)
        insulin_peak_min = absorption_profile.insulinPeakMin if absorption_profile else 75.0
        carb_peak_min = absorption_profile.carbPeakMin if absorption_profile else 45.0
        protein_peak_min = absorption_profile.proteinPeakMin if absorption_profile else 180.0

        # Calculate IOB, COB, and POB for metric display (instant, no absorption delay)
        iob = iob_cob_service.calculate_iob(treatments, include_absorption_ramp=False)
        cob = iob_cob_service.calculate_cob(treatments, include_absorption_ramp=False)
        pob = iob_cob_service.calculate_pob(treatments)

        # Get predictions from ML service
        pred_service = get_pred_service()

        # Get glucose history for LSTM predictions and historical IOB/COB/BG Pressure visualization
        # Use 24 hours to support full chart time range (1hr, 3hr, 6hr, 12hr, 24hr)
        # This allows historical IOB/COB and BG Pressure lines to span the entire viewable period
        start_time = datetime.utcnow() - timedelta(hours=24)
        glucose_history = await glucose_repo.get_history(data_user_id, start_time)

        # Convert trend to int for prediction
        trend_val = 0
        if latest.trend:
            trend_map = {
                "DoubleDown": -3, "SingleDown": -2, "FortyFiveDown": -1,
                "Flat": 0, "FortyFiveUp": 1, "SingleUp": 2, "DoubleUp": 3
            }
            trend_val = trend_map.get(str(latest.trend), 0)

        # Generate predictions
        logger.info(f"Calling predict with {len(glucose_history)} glucose readings, {len(treatments)} treatments, IOB={iob:.2f}, COB={cob:.0f}")
        logger.info(f"LSTM available: {pred_service.lstm_available}, TFT available: {pred_service.tft_available}")

        prediction_result = pred_service.predict(
            current_bg=float(latest.value),
            trend=trend_val,
            iob=iob,
            cob=cob,
            glucose_history=[r.model_dump() for r in glucose_history],
            treatments=[t.model_dump() for t in treatments]
        )

        logger.info(f"Prediction result - linear: {prediction_result.linear}, lstm: {prediction_result.lstm}, method: {prediction_result.method}")

        # Get effective ISF with short-term illness detection
        # Uses metabolic_params_service which combines long-term baseline + recent deviation
        isf = prediction_result.isf  # Default to model prediction
        isf_deviation = 0.0
        metabolic_state = MetabolicState.NORMAL
        try:
            # Determine if fasting based on recent COB
            is_fasting = cob < 5
            effective_isf = await metabolic_params_service.get_effective_isf(
                data_user_id, is_fasting=is_fasting, include_short_term=True
            )
            isf = effective_isf.value
            isf_deviation = effective_isf.deviation_percent

            # Track metabolic state for response
            if effective_isf.is_sick:
                metabolic_state = MetabolicState.SICK
                logger.warning(f"Illness detected for user {user_id}: ISF deviation {isf_deviation:.1f}%")
            elif effective_isf.is_resistant:
                metabolic_state = MetabolicState.RESISTANT
                logger.info(f"Insulin resistance detected: ISF deviation {isf_deviation:.1f}%")

            if effective_isf.source == "default":
                # Fall back to user settings if no learned data
                user = await user_repo.get_by_id(data_user_id)
                if user and user.settings and user.settings.insulinSensitivity != 50.0:
                    isf = user.settings.insulinSensitivity
                    logger.info(f"Using user settings ISF: {isf:.1f}")
                else:
                    logger.debug(f"Using default ISF: {isf:.1f}")
            else:
                logger.info(f"Using effective ISF: {isf:.1f} (baseline: {effective_isf.baseline:.1f}, deviation: {isf_deviation:.1f}%, source: {effective_isf.source})")
        except Exception as e:
            logger.warning(f"Failed to get effective ISF, using model prediction: {e}")

        # Get effective PIR (Protein-to-Insulin Ratio) with timing info
        pir = 25.0  # Default PIR
        try:
            effective_pir = await metabolic_params_service.get_effective_pir(user_id)
            pir = effective_pir.value
            logger.debug(f"Using effective PIR: {pir:.0f} (source: {effective_pir.source})")
        except Exception as e:
            logger.warning(f"Failed to get effective PIR, using default: {e}")

        # Convert ML predictions to dict format for food recommendation
        ml_predictions = None
        if prediction_result.tft:
            ml_predictions = [
                {
                    'horizon_min': p.horizon_min,
                    'value': p.value,
                    'lower': p.lower,
                    'upper': p.upper
                }
                for p in prediction_result.tft
            ]
        # Also include LSTM predictions for near-term lows
        if prediction_result.lstm:
            lstm_preds = [
                {'horizon_min': h, 'value': v, 'lower': v, 'upper': v}
                for h, v in zip([5, 10, 15], prediction_result.lstm)
            ]
            if ml_predictions:
                ml_predictions = lstm_preds + ml_predictions
            else:
                ml_predictions = lstm_preds

        # Calculate metrics with predicted ISF and PIR - pass ML predictions for food recommendations
        metrics = iob_cob_service.get_current_metrics(
            current_bg=latest.value,
            treatments=treatments,
            isf=isf,
            pir=pir,
            ml_predictions=ml_predictions
        )

        # Build predictions response
        predictions = GlucosePrediction(
            timestamp=prediction_result.timestamp,
            linear=prediction_result.linear,
            lstm=prediction_result.lstm or []
        )

        glucose_with_predictions = GlucoseWithPredictions(
            **latest.model_dump(),
            predictions=predictions
        )

        # Get accuracy stats
        accuracy_stats = pred_service.get_accuracy_stats()
        accuracy = PredictionAccuracy(
            linearWins=accuracy_stats.linear_count,
            lstmWins=accuracy_stats.lstm_count,
            totalComparisons=accuracy_stats.linear_count + accuracy_stats.lstm_count
        )

        # Calculate IOB/COB effect curve for visualization
        user = await user_repo.get_by_id(data_user_id)
        user_icr = user.settings.carbRatio if user else 10.0

        # Calculate effect curve for IOB/COB decay visualization
        # Use 180 min (3hr) which is clinically relevant for decision making
        # Beyond 3 hours, IOB/COB effects are minimal and predictions become unreliable
        current_bg_value = latest.value if latest else 120
        base_time = latest.timestamp.replace(tzinfo=None) if latest else datetime.utcnow()
        effect_curve_raw = iob_cob_service.calculate_bg_effect_curve(
            current_iob=iob,
            current_cob=cob,
            current_pob=pob,  # Pass current POB for protein effect on expected BG
            isf=isf,
            icr=user_icr,
            duration_min=180,  # 3 hours - clinically relevant prediction window
            step_min=5,
            current_bg=float(current_bg_value),  # Pass current BG for expected trajectory
            treatments=treatments,  # Enable treatment-based formula for continuity
            base_time=base_time  # Use latest reading timestamp as base
        )

        # Convert to EffectPoint schema
        effect_curve = [
            EffectPoint(**point) for point in effect_curve_raw
        ]

        # TFT predictions from prediction service
        tft_predictions: List[TFTPrediction] = []

        # Get TFT predictions from prediction result
        if prediction_result.tft:
            now = datetime.now(timezone.utc)  # Use timezone-aware UTC
            for tft_pred in prediction_result.tft:
                tft_predictions.append(TFTPrediction(
                    timestamp=now + timedelta(minutes=tft_pred.horizon_min),
                    horizon=tft_pred.horizon_min,
                    value=tft_pred.value,
                    lower=tft_pred.lower,
                    upper=tft_pred.upper,
                    tftDelta=tft_pred.tft_delta  # TFT modifier delta (physics + delta = final)
                ))

        # INCORPORATE TFT DELTA INTO EFFECT CURVE (Predicted BG line)
        # The TFT model provides delta adjustments at specific horizons (30, 45, 60 min)
        # Interpolate these deltas across all effect curve time points
        if tft_predictions and effect_curve:
            # Build horizon -> delta mapping
            tft_deltas = {p.horizon: p.tftDelta for p in tft_predictions if p.tftDelta is not None}

            if tft_deltas:
                # Get sorted horizons for interpolation
                sorted_horizons = sorted(tft_deltas.keys())
                logger.info(f"TFT delta adjustment: horizons={sorted_horizons}, deltas={[tft_deltas[h] for h in sorted_horizons]}")

                # Apply interpolated delta to each effect curve point's expectedBg
                for point in effect_curve:
                    if point.expectedBg is None:
                        continue

                    t = point.minutesAhead

                    # Interpolate delta for this time point
                    if t <= 0:
                        # Before first horizon: no delta
                        delta = 0.0
                    elif t < sorted_horizons[0]:
                        # Before first TFT prediction: linear ramp from 0
                        delta = tft_deltas[sorted_horizons[0]] * (t / sorted_horizons[0])
                    elif t >= sorted_horizons[-1]:
                        # After last TFT prediction: use last delta (constant)
                        delta = tft_deltas[sorted_horizons[-1]]
                    else:
                        # Between TFT predictions: linear interpolation
                        for i in range(len(sorted_horizons) - 1):
                            h1, h2 = sorted_horizons[i], sorted_horizons[i + 1]
                            if h1 <= t < h2:
                                d1, d2 = tft_deltas[h1], tft_deltas[h2]
                                frac = (t - h1) / (h2 - h1)
                                delta = d1 + (d2 - d1) * frac
                                break
                        else:
                            delta = 0.0

                    # Apply delta to expectedBg
                    if delta != 0.0:
                        adjusted_bg = max(40, min(400, point.expectedBg + delta))
                        point.expectedBg = round(adjusted_bg, 0)

                logger.info(f"Applied TFT delta to {len(effect_curve)} effect curve points")

        # Calculate historical IOB/COB/POB and BG pressure at each glucose reading timestamp
        # This provides continuous curves for plotting:
        # - IOB/COB/POB decay over time
        # - BG Pressure: where BG is heading based on REMAINING IOB (down) + COB (up) + POB (up, delayed)
        historical_iob_cob: List[HistoricalIobCobPoint] = []

        # Use user's ICR for carb-to-BG conversion (user already fetched above)
        bg_per_gram_carb = isf / user_icr  # BG rise per gram of carbs
        # Protein has about 50% the effect of carbs on BG (2.0 vs 4.0 bg_factor)
        bg_per_gram_protein = iob_cob_service.protein_bg_factor  # BG rise per gram of protein

        # Log protein in treatments for debugging
        protein_treatments = [t for t in treatments if getattr(t, 'protein', None) and t.protein > 0]
        if protein_treatments:
            logger.info(f"[POB Debug] Found {len(protein_treatments)} treatments with protein")
            for pt in protein_treatments[:3]:  # Log first 3
                logger.info(f"[POB Debug] Treatment: {pt.timestamp}, protein={pt.protein}g, type={pt.type}")
        else:
            logger.debug("[POB Debug] No treatments with protein > 0 found")

        for reading in glucose_history:
            reading_time = reading.timestamp.replace(tzinfo=None)
            hist_iob = iob_cob_service.calculate_iob(treatments, at_time=reading_time)
            hist_cob = iob_cob_service.calculate_cob(treatments, at_time=reading_time)
            hist_pob = iob_cob_service.calculate_pob(treatments, at_time=reading_time)

            # Calculate INSTANTANEOUS PRESSURE using ACTIVITY RATES
            # Key insight: Pressure should be a LEADING indicator
            # - High insulin ACTIVITY (absorbing now) → pressure drops BEFORE BG falls
            # - High carb ACTIVITY (absorbing now) → pressure rises BEFORE BG rises
            #
            # Use activity curves (0-1) at each moment, not total remaining amounts

            # Calculate instantaneous activity for each treatment
            total_insulin_activity_effect = 0.0
            total_carb_activity_effect = 0.0
            total_protein_activity_effect = 0.0

            for treatment in treatments:
                t_time = treatment.timestamp.replace(tzinfo=None)
                time_since_dose = (reading_time - t_time).total_seconds() / 60.0

                if time_since_dose < 0:
                    continue  # Future treatment

                # Insulin activity
                if treatment.insulin and treatment.insulin > 0 and time_since_dose < iob_cob_service.insulin_duration_min:
                    from services.iob_cob_service import insulin_activity_curve
                    activity = insulin_activity_curve(time_since_dose, peak_min=insulin_peak_min, dia_min=iob_cob_service.insulin_duration_min)
                    # Activity * amount * ISF = instantaneous BG lowering force
                    total_insulin_activity_effect += activity * treatment.insulin * isf

                # Carb activity
                if treatment.carbs and treatment.carbs > 0:
                    from services.iob_cob_service import carb_activity_curve, gi_to_absorption_params
                    gi = getattr(treatment, 'glycemicIndex', None) or 55
                    is_liquid = getattr(treatment, 'isLiquid', False) or False
                    # Get carb duration from GI
                    gi_params = gi_to_absorption_params(gi, is_liquid)
                    duration = gi_params['duration_min']

                    if time_since_dose < duration:
                        activity = carb_activity_curve(time_since_dose, peak_min=carb_peak_min, duration_min=duration,
                                                      glycemic_index=gi, is_liquid=is_liquid)
                        total_carb_activity_effect += activity * treatment.carbs * bg_per_gram_carb

                # Protein activity (delayed, like slow carbs)
                if treatment.protein and treatment.protein > 0 and time_since_dose < 300:  # 5 hour duration
                    from services.iob_cob_service import carb_activity_curve
                    # Protein peaks much later (2-5 hours) - use learned peak if available
                    activity = carb_activity_curve(time_since_dose, peak_min=protein_peak_min, duration_min=300,
                                                  glycemic_index=30, is_liquid=False)  # GI=30 for slow absorption
                    total_protein_activity_effect += activity * treatment.protein * bg_per_gram_protein

            # Pressure offset based on NET ACTIVITY
            # Positive = carbs absorbing faster → pressure above BG
            # Negative = insulin absorbing faster → pressure below BG
            activity_scale = 0.3  # Scale for visual clarity
            pressure_offset = (total_carb_activity_effect + total_protein_activity_effect - total_insulin_activity_effect) * activity_scale

            # Debug logging for first few readings
            if len(historical_iob_cob) < 3:
                logger.info(f"[BG Pressure Debug] Time: {reading.timestamp}, BG: {reading.value}")
                logger.info(f"  Insulin activity effect: {total_insulin_activity_effect:.1f} mg/dL")
                logger.info(f"  Carb activity effect: {total_carb_activity_effect:.1f} mg/dL")
                logger.info(f"  Protein activity effect: {total_protein_activity_effect:.1f} mg/dL")
                logger.info(f"  Net offset (scaled): {pressure_offset:.1f} mg/dL")
                logger.info(f"  Final pressure: {reading.value + pressure_offset:.1f} mg/dL")
                logger.info(f"  (IOB={hist_iob:.2f}, COB={hist_cob:.1f}, POB={hist_pob:.1f})")

            # BG Pressure = current BG + offset (leading indicator)
            bg_pressure = reading.value + pressure_offset

            historical_iob_cob.append(HistoricalIobCobPoint(
                timestamp=reading.timestamp,
                iob=hist_iob,
                cob=hist_cob,
                pob=hist_pob,
                bgPressure=round(bg_pressure, 0),  # Instantaneous pressure at this moment
                actualBg=reading.value
            ))

        # Log POB summary for debugging
        pob_values = [p.pob for p in historical_iob_cob]
        max_pob = max(pob_values) if pob_values else 0
        non_zero_pob = sum(1 for p in pob_values if p > 0)
        logger.info(f"[POB Debug] Historical POB summary: max={max_pob:.1f}g, non_zero_count={non_zero_pob}, total_points={len(pob_values)}")

        return GlucoseCurrentResponse(
            glucose=glucose_with_predictions,
            metrics=metrics,
            accuracy=accuracy,
            tftPredictions=tft_predictions,
            effectCurve=effect_curve,
            historicalIobCob=historical_iob_cob
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current glucose: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history", response_model=GlucoseHistoryResponse)
async def get_glucose_history(
    user_id: str = Query(..., description="User ID whose data to view"),
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history (1-168)"),
    limit: int = Query(default=1000, ge=1, le=5000),
    current_user = Depends(get_current_user)
):
    """
    Get historical glucose readings.

    Returns glucose readings for the specified time period.
    Requires JWT authentication. Users can view their own data or their children's data.
    """
    try:
        # SECURITY: Always validate access using authenticated user
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")

        # Normalize profile ID to data user ID
        data_user_id = get_data_user_id(user_id)

        start_time = datetime.utcnow() - timedelta(hours=hours)
        end_time = datetime.utcnow()

        readings = await glucose_repo.get_history(
            user_id=data_user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        return GlucoseHistoryResponse(
            readings=readings,
            totalCount=len(readings),
            startTime=start_time,
            endTime=end_time
        )

    except Exception as e:
        logger.error(f"Error getting glucose history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/range-stats")
async def get_range_stats(
    user_id: str = Query(..., description="User ID whose data to view"),
    hours: int = Query(default=24, ge=1, le=168),
    current_user = Depends(get_current_user)
):
    """
    Get time-in-range statistics.

    Returns percentage of time in each glucose range:
    - Critical Low (<54)
    - Low (54-70)
    - In Range (70-180)
    - High (180-250)
    - Critical High (>250)

    Requires JWT authentication. Users can view their own data or their children's data.
    """
    try:
        # SECURITY: Always validate access using authenticated user
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")

        # Normalize profile ID to data user ID
        data_user_id = get_data_user_id(user_id)

        start_time = datetime.utcnow() - timedelta(hours=hours)
        readings = await glucose_repo.get_history(data_user_id, start_time)

        if not readings:
            return {
                "totalReadings": 0,
                "criticalLow": 0,
                "low": 0,
                "inRange": 0,
                "high": 0,
                "criticalHigh": 0,
                "averageBg": None,
                "estimatedA1c": None
            }

        # Count readings in each range
        critical_low = sum(1 for r in readings if r.value < 54)
        low = sum(1 for r in readings if 54 <= r.value < 70)
        in_range = sum(1 for r in readings if 70 <= r.value <= 180)
        high = sum(1 for r in readings if 180 < r.value <= 250)
        critical_high = sum(1 for r in readings if r.value > 250)

        total = len(readings)
        avg_bg = sum(r.value for r in readings) / total

        # Estimate A1C from average BG
        # Formula: A1C = (average_bg + 46.7) / 28.7
        estimated_a1c = (avg_bg + 46.7) / 28.7

        return {
            "totalReadings": total,
            "criticalLow": round(critical_low / total * 100, 1),
            "low": round(low / total * 100, 1),
            "inRange": round(in_range / total * 100, 1),
            "high": round(high / total * 100, 1),
            "criticalHigh": round(critical_high / total * 100, 1),
            "averageBg": round(avg_bg, 0),
            "estimatedA1c": round(estimated_a1c, 1)
        }

    except Exception as e:
        logger.error(f"Error getting range stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
