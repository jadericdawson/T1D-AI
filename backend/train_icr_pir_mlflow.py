#!/usr/bin/env python3
"""
Improved ICR/PIR Training with MLflow Tracking

Key improvements:
1. Train/validation splits (70/30)
2. Tighter selection criteria
3. Use median instead of mean (more robust to outliers)
4. Retrospective ICR calculation (what ICR SHOULD have been)
5. Better protein event detection

Expected values: ICR ~10:1, PIR ~12:1
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import mlflow

from database.repositories import (
    GlucoseRepository, TreatmentRepository,
    LearnedISFRepository, LearnedICRRepository, LearnedPIRRepository
)
from models.schemas import LearnedICR, ICRDataPoint, LearnedPIR, PIRDataPoint
from ml.mlflow_tracking import ModelTracker, MLFLOW_TRACKING_URI

USER_ID = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"


class ImprovedICRLearner:
    """
    Improved ICR learner with tighter criteria.

    Key changes:
    - Uses retrospective ICR (what would have hit target)
    - Requires good BG outcomes (ending 70-160)
    - Uses median for robustness
    - Groups treatments within 30min as same meal
    """

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.isf_repo = LearnedISFRepository()
        self.icr_repo = LearnedICRRepository()

        # Tighter configuration
        self.min_carbs = 15           # Higher threshold for cleaner signal
        self.min_insulin = 0.5
        self.target_bg = 100
        self.min_icr = 5              # Tighter range
        self.max_icr = 20             # Tighter range
        self.good_outcome_range = (70, 160)  # BG range for "good" outcome

    async def collect_meal_events(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Collect all meal events with full data."""
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Get ISF
        isf_record = await self.isf_repo.get(user_id, "meal")
        isf = isf_record.value if isf_record else 50.0

        # Get treatments
        treatments = await self.treatment_repo.get_by_user(
            user_id, start_time=start_time, limit=2000
        )

        # Group treatments by meal (within 30min)
        meal_groups = self._group_treatments(treatments)

        events = []
        for meal_ts, group_treatments in meal_groups.items():
            total_carbs = sum(t.carbs or 0 for t in group_treatments)
            total_insulin = sum(t.insulin or 0 for t in group_treatments)
            notes = "; ".join([t.notes for t in group_treatments if t.notes])

            if total_carbs < self.min_carbs or total_insulin < self.min_insulin:
                continue

            # Get BG readings
            bg_before = await self._get_nearest_bg(user_id, meal_ts, 15)
            bg_after = await self._get_nearest_bg(
                user_id, meal_ts + timedelta(hours=2.5), 45
            )

            if not bg_before or not bg_after:
                continue

            # Calculate correction component
            correction_insulin = max(0, (bg_before - self.target_bg) / isf)
            meal_insulin = max(0.1, total_insulin - correction_insulin)

            # Raw ICR (what was used)
            raw_icr = total_carbs / meal_insulin

            # Retrospective ICR (what SHOULD have been used to hit target)
            # target = bg_before - (insulin * isf) + (carbs / true_icr * isf)
            # Solving for true_icr:
            bg_change_from_insulin = total_insulin * isf
            needed_bg_change = bg_before - self.target_bg
            carb_bg_rise = bg_after - bg_before + bg_change_from_insulin

            if carb_bg_rise > 0:
                # How much insulin would have been needed for this carb rise?
                needed_carb_insulin = carb_bg_rise / isf
                retro_icr = total_carbs / needed_carb_insulin if needed_carb_insulin > 0.1 else raw_icr
            else:
                retro_icr = raw_icr

            # Determine meal type
            hour = meal_ts.hour
            if 5 <= hour < 10:
                meal_type = "breakfast"
            elif 10 <= hour < 15:
                meal_type = "lunch"
            elif 15 <= hour < 21:
                meal_type = "dinner"
            else:
                meal_type = "snack"

            events.append({
                'timestamp': meal_ts,
                'carbs': total_carbs,
                'insulin': total_insulin,
                'bg_before': bg_before,
                'bg_after': bg_after,
                'correction': correction_insulin,
                'meal_insulin': meal_insulin,
                'raw_icr': raw_icr,
                'retro_icr': retro_icr,
                'meal_type': meal_type,
                'notes': notes,
                'isf': isf,
                'outcome': 'good' if self.good_outcome_range[0] <= bg_after <= self.good_outcome_range[1] else 'bad'
            })

        return events

    def _group_treatments(self, treatments) -> Dict[datetime, List]:
        """Group treatments within 30min as same meal."""
        if not treatments:
            return {}

        def normalize_ts(ts):
            if ts.tzinfo:
                return ts.astimezone(timezone.utc).replace(tzinfo=None)
            return ts

        sorted_treatments = sorted(treatments, key=lambda t: normalize_ts(t.timestamp))

        groups = {}
        current_ts = None
        current_group = []

        for t in sorted_treatments:
            ts = normalize_ts(t.timestamp)

            if current_ts is None:
                current_ts = ts
                current_group = [t]
            elif (ts - current_ts).total_seconds() <= 1800:  # 30 min
                current_group.append(t)
            else:
                if current_group:
                    groups[current_ts] = current_group
                current_ts = ts
                current_group = [t]

        if current_group and current_ts:
            groups[current_ts] = current_group

        return groups

    async def _get_nearest_bg(
        self, user_id: str, target_time: datetime, window_minutes: int
    ) -> Optional[float]:
        """Get nearest BG reading."""
        start = target_time - timedelta(minutes=window_minutes)
        end = target_time + timedelta(minutes=window_minutes)

        readings = await self.glucose_repo.get_history(
            user_id, start_time=start, end_time=end, limit=10
        )

        if not readings:
            return None

        def time_diff(r):
            r_ts = r.timestamp.replace(tzinfo=None) if r.timestamp.tzinfo else r.timestamp
            t_ts = target_time.replace(tzinfo=None) if target_time.tzinfo else target_time
            return abs((r_ts - t_ts).total_seconds())

        closest = min(readings, key=time_diff)
        return closest.value

    def train_and_validate(
        self,
        events: List[Dict],
        use_retro: bool = True,
        filter_good_outcomes: bool = False
    ) -> Dict[str, Any]:
        """
        Train on 70% of data, validate on 30%.

        Args:
            events: All collected meal events
            use_retro: Use retrospective ICR (what should have been)
            filter_good_outcomes: Only use meals with good BG outcome
        """
        if not events:
            return {'error': 'No events'}

        # Filter if requested
        if filter_good_outcomes:
            events = [e for e in events if e['outcome'] == 'good']

        if len(events) < 5:
            return {'error': f'Not enough events after filtering: {len(events)}'}

        # Select ICR type
        icr_key = 'retro_icr' if use_retro else 'raw_icr'

        # Filter valid ICR range
        valid_events = [e for e in events if self.min_icr <= e[icr_key] <= self.max_icr]

        if len(valid_events) < 5:
            return {'error': f'Not enough valid ICR values: {len(valid_events)}'}

        # Sort by timestamp for temporal split
        valid_events.sort(key=lambda e: e['timestamp'])

        # 70/30 split
        split_idx = int(len(valid_events) * 0.7)
        train_events = valid_events[:split_idx]
        val_events = valid_events[split_idx:]

        # Calculate ICR from training set
        train_icrs = [e[icr_key] for e in train_events]

        # Use median for robustness
        learned_icr = float(np.median(train_icrs))
        mean_icr = float(np.mean(train_icrs))
        std_icr = float(np.std(train_icrs))

        # Validate on validation set
        val_errors = []
        for e in val_events:
            # Predict what dose would have been with learned ICR
            predicted_dose = e['carbs'] / learned_icr + e['correction']
            actual_dose = e['insulin']
            error = abs(predicted_dose - actual_dose)
            val_errors.append(error)

        mae = float(np.mean(val_errors)) if val_errors else 0

        # Meal type breakdown
        meal_type_icr = {}
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            mt_events = [e for e in train_events if e['meal_type'] == meal_type]
            if len(mt_events) >= 2:
                mt_icrs = [e[icr_key] for e in mt_events]
                meal_type_icr[meal_type] = float(np.median(mt_icrs))

        return {
            'learned_icr': learned_icr,
            'mean_icr': mean_icr,
            'std_icr': std_icr,
            'min_icr': float(np.min(train_icrs)),
            'max_icr': float(np.max(train_icrs)),
            'train_samples': len(train_events),
            'val_samples': len(val_events),
            'val_mae': mae,
            'meal_type_icr': meal_type_icr,
            'method': 'retro' if use_retro else 'raw',
            'filtered': filter_good_outcomes,
        }


class ImprovedPIRLearner:
    """
    Improved PIR learner with tighter criteria.

    Key changes:
    - Uses median for robustness
    - Better late rise detection window
    - Filters meals with too many carbs (mask protein effect)
    - Improved protein estimation from notes
    """

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.isf_repo = LearnedISFRepository()
        self.pir_repo = LearnedPIRRepository()

        # Tighter configuration
        self.min_protein = 10         # Lower threshold to catch more events
        self.max_carbs = 40           # Filter high-carb meals
        self.min_late_rise = 15       # Lower threshold for better detection
        self.min_pir = 8              # Tighter range
        self.max_pir = 25             # Tighter range (expect ~12)

    async def collect_protein_events(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Collect all protein events with late rise data."""
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Get ISF
        isf_record = await self.isf_repo.get(user_id, "fasting")
        isf = isf_record.value if isf_record else 50.0

        # Get treatments
        treatments = await self.treatment_repo.get_by_user(
            user_id, start_time=start_time, limit=2000
        )

        events = []
        for t in treatments:
            protein = t.protein or 0
            carbs = t.carbs or 0
            notes = t.notes or ''

            # Estimate protein from notes if not logged
            if protein < self.min_protein:
                protein = self._estimate_protein(notes)

            if protein < self.min_protein:
                continue

            ts = t.timestamp
            if ts.tzinfo:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

            # Get BG at multiple time points
            bg_meal = await self._get_nearest_bg(user_id, ts, 15)
            bg_90 = await self._get_nearest_bg(user_id, ts + timedelta(minutes=90), 30)
            bg_120 = await self._get_nearest_bg(user_id, ts + timedelta(minutes=120), 30)
            bg_180 = await self._get_nearest_bg(user_id, ts + timedelta(minutes=180), 30)
            bg_240 = await self._get_nearest_bg(user_id, ts + timedelta(minutes=240), 30)

            if not bg_90 or not bg_180:
                continue

            # Calculate late rise (from 90min baseline to peak in 120-240 window)
            baseline = bg_90
            peak_bg = max(filter(None, [bg_120, bg_180, bg_240]), default=bg_90)
            late_rise = max(0, peak_bg - baseline)

            # Determine peak time
            peak_time = 90
            if bg_180 and bg_180 >= (bg_120 or 0):
                peak_time = 180
            elif bg_240 and bg_240 >= (bg_180 or 0):
                peak_time = 240
            elif bg_120:
                peak_time = 120

            # Calculate PIR if late rise is significant
            if late_rise >= self.min_late_rise:
                insulin_equiv = late_rise / isf
                pir = protein / insulin_equiv if insulin_equiv > 0.1 else 0
            else:
                pir = None  # Not enough rise to calculate

            events.append({
                'timestamp': ts,
                'protein': protein,
                'carbs': carbs,
                'insulin': t.insulin or 0,
                'notes': notes,
                'bg_meal': bg_meal,
                'bg_90': bg_90,
                'bg_120': bg_120,
                'bg_180': bg_180,
                'bg_240': bg_240,
                'late_rise': late_rise,
                'peak_time': peak_time,
                'pir': pir,
                'isf': isf,
                'is_clean': carbs <= self.max_carbs,  # Low carb = cleaner protein signal
            })

        return events

    def _estimate_protein(self, notes: str) -> float:
        """Estimate protein from meal notes with improved matching."""
        if not notes:
            return 0

        notes_lower = notes.lower()

        # More comprehensive protein keywords
        protein_foods = {
            # High protein
            "steak": 40, "ribeye": 45, "filet": 35, "beef": 30,
            "chicken breast": 30, "chicken": 25, "turkey": 25,
            "salmon": 30, "tuna": 30, "fish": 25, "shrimp": 20, "tilapia": 25,
            "pork chop": 30, "pork": 25, "ham": 20, "lamb": 25,

            # Medium protein
            "burger": 20, "cheeseburger": 22, "patty": 15,
            "hot dog": 7, "hotdog": 7, "sausage": 15, "bacon": 10,
            "eggs": 12, "egg": 6, "omelet": 18, "omelette": 18,
            "meatball": 8, "meat": 15,

            # Dairy/other protein
            "cheese": 7, "cottage cheese": 15, "greek yogurt": 15,
            "milk": 8, "chocolate milk": 8,
            "protein shake": 25, "protein bar": 20, "protein": 20,
            "sandwich": 10, "bologna": 5, "deli": 8,

            # Low protein (but still counts)
            "pizza": 12, "taco": 8, "tacos": 16, "burrito": 15,
        }

        estimated = 0
        for food, protein in protein_foods.items():
            if food in notes_lower:
                # Try to detect quantity
                import re
                quantity_match = re.search(rf'(\d+)\s*{food}', notes_lower)
                if quantity_match:
                    qty = int(quantity_match.group(1))
                    estimated += protein * qty
                else:
                    estimated += protein

        return estimated

    async def _get_nearest_bg(
        self, user_id: str, target_time: datetime, window_minutes: int
    ) -> Optional[float]:
        """Get nearest BG reading."""
        start = target_time - timedelta(minutes=window_minutes)
        end = target_time + timedelta(minutes=window_minutes)

        readings = await self.glucose_repo.get_history(
            user_id, start_time=start, end_time=end, limit=10
        )

        if not readings:
            return None

        def time_diff(r):
            r_ts = r.timestamp.replace(tzinfo=None) if r.timestamp.tzinfo else r.timestamp
            t_ts = target_time.replace(tzinfo=None) if target_time.tzinfo else target_time
            return abs((r_ts - t_ts).total_seconds())

        closest = min(readings, key=time_diff)
        return closest.value

    def train_and_validate(
        self,
        events: List[Dict],
        filter_clean: bool = True
    ) -> Dict[str, Any]:
        """
        Train on 70% of data, validate on 30%.

        Args:
            events: All collected protein events
            filter_clean: Only use low-carb meals for cleaner signal
        """
        # Filter events with valid PIR
        valid_events = [e for e in events if e['pir'] is not None]

        if filter_clean:
            valid_events = [e for e in valid_events if e['is_clean']]

        # Filter by PIR range
        valid_events = [e for e in valid_events if self.min_pir <= e['pir'] <= self.max_pir]

        if len(valid_events) < 3:
            return {'error': f'Not enough valid events: {len(valid_events)}'}

        # Sort by timestamp
        valid_events.sort(key=lambda e: e['timestamp'])

        # 70/30 split
        split_idx = max(1, int(len(valid_events) * 0.7))
        train_events = valid_events[:split_idx]
        val_events = valid_events[split_idx:]

        # Calculate PIR from training set
        train_pirs = [e['pir'] for e in train_events]

        # Use median for robustness
        learned_pir = float(np.median(train_pirs))
        mean_pir = float(np.mean(train_pirs))
        std_pir = float(np.std(train_pirs)) if len(train_pirs) > 1 else 0

        # Calculate average timing
        onset_times = [90]  # Baseline
        peak_times = [e['peak_time'] for e in train_events if e['peak_time']]
        avg_onset = 90
        avg_peak = int(np.median(peak_times)) if peak_times else 180

        # Validate
        val_errors = []
        for e in val_events:
            # Predict insulin needed for protein
            predicted_protein_insulin = e['protein'] / learned_pir
            actual_late_rise = e['late_rise']
            actual_protein_insulin = actual_late_rise / e['isf'] if e['isf'] > 0 else 0
            error = abs(predicted_protein_insulin - actual_protein_insulin)
            val_errors.append(error)

        mae = float(np.mean(val_errors)) if val_errors else 0

        return {
            'learned_pir': learned_pir,
            'mean_pir': mean_pir,
            'std_pir': std_pir,
            'min_pir': float(np.min(train_pirs)),
            'max_pir': float(np.max(train_pirs)),
            'train_samples': len(train_events),
            'val_samples': len(val_events),
            'val_mae': mae,
            'avg_onset_min': avg_onset,
            'avg_peak_min': avg_peak,
            'filtered': filter_clean,
        }


async def run_experiments():
    """Run ICR and PIR experiments with MLflow tracking."""
    print("="*70)
    print("ICR/PIR TRAINING WITH MLFLOW")
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"User: {USER_ID}")
    print("="*70)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ==================== ICR EXPERIMENTS ====================
    print("\n" + "="*70)
    print("ICR EXPERIMENTS")
    print("="*70)

    icr_learner = ImprovedICRLearner()
    icr_events = await icr_learner.collect_meal_events(USER_ID, days=30)
    print(f"Collected {len(icr_events)} meal events")

    # Create ICR experiment
    mlflow.set_experiment("T1D-AI/ICR-Personalized")

    # Experiment 1: Raw ICR, all outcomes
    with mlflow.start_run(run_name="icr_raw_all"):
        mlflow.log_param("method", "raw")
        mlflow.log_param("filter_outcomes", False)
        mlflow.log_param("user_id", USER_ID)

        result = icr_learner.train_and_validate(icr_events, use_retro=False, filter_good_outcomes=False)
        if 'error' not in result:
            mlflow.log_metrics({
                "icr_median": result['learned_icr'],
                "icr_mean": result['mean_icr'],
                "icr_std": result['std_icr'],
                "train_samples": result['train_samples'],
                "val_samples": result['val_samples'],
                "val_mae": result['val_mae'],
            })
            print(f"\n[Raw ICR, All]: {result['learned_icr']:.1f} (n={result['train_samples']}, MAE={result['val_mae']:.2f})")
        else:
            print(f"\n[Raw ICR, All]: {result['error']}")

    # Experiment 2: Retrospective ICR, all outcomes
    with mlflow.start_run(run_name="icr_retro_all"):
        mlflow.log_param("method", "retro")
        mlflow.log_param("filter_outcomes", False)
        mlflow.log_param("user_id", USER_ID)

        result = icr_learner.train_and_validate(icr_events, use_retro=True, filter_good_outcomes=False)
        if 'error' not in result:
            mlflow.log_metrics({
                "icr_median": result['learned_icr'],
                "icr_mean": result['mean_icr'],
                "icr_std": result['std_icr'],
                "train_samples": result['train_samples'],
                "val_samples": result['val_samples'],
                "val_mae": result['val_mae'],
            })
            print(f"[Retro ICR, All]: {result['learned_icr']:.1f} (n={result['train_samples']}, MAE={result['val_mae']:.2f})")
            for mt, icr in result['meal_type_icr'].items():
                print(f"  {mt}: {icr:.1f}")
        else:
            print(f"[Retro ICR, All]: {result['error']}")

    # Experiment 3: Retrospective ICR, good outcomes only
    with mlflow.start_run(run_name="icr_retro_good"):
        mlflow.log_param("method", "retro")
        mlflow.log_param("filter_outcomes", True)
        mlflow.log_param("user_id", USER_ID)

        result = icr_learner.train_and_validate(icr_events, use_retro=True, filter_good_outcomes=True)
        if 'error' not in result:
            mlflow.log_metrics({
                "icr_median": result['learned_icr'],
                "icr_mean": result['mean_icr'],
                "icr_std": result['std_icr'],
                "train_samples": result['train_samples'],
                "val_samples": result['val_samples'],
                "val_mae": result['val_mae'],
            })
            print(f"[Retro ICR, Good]: {result['learned_icr']:.1f} (n={result['train_samples']}, MAE={result['val_mae']:.2f})")
        else:
            print(f"[Retro ICR, Good]: {result['error']}")

    # ==================== PIR EXPERIMENTS ====================
    print("\n" + "="*70)
    print("PIR EXPERIMENTS")
    print("="*70)

    pir_learner = ImprovedPIRLearner()
    pir_events = await pir_learner.collect_protein_events(USER_ID, days=30)
    print(f"Collected {len(pir_events)} protein events")

    # Show protein events summary
    events_with_pir = [e for e in pir_events if e['pir'] is not None]
    print(f"Events with detectable late rise: {len(events_with_pir)}")

    # Create PIR experiment
    mlflow.set_experiment("T1D-AI/PIR-Personalized")

    # Experiment 1: All events
    with mlflow.start_run(run_name="pir_all"):
        mlflow.log_param("filter_clean", False)
        mlflow.log_param("user_id", USER_ID)

        result = pir_learner.train_and_validate(pir_events, filter_clean=False)
        if 'error' not in result:
            mlflow.log_metrics({
                "pir_median": result['learned_pir'],
                "pir_mean": result['mean_pir'],
                "pir_std": result['std_pir'],
                "train_samples": result['train_samples'],
                "val_samples": result['val_samples'],
                "val_mae": result['val_mae'],
                "avg_onset_min": result['avg_onset_min'],
                "avg_peak_min": result['avg_peak_min'],
            })
            print(f"\n[PIR, All]: {result['learned_pir']:.1f} (n={result['train_samples']}, MAE={result['val_mae']:.2f})")
            print(f"  Timing: onset={result['avg_onset_min']}min, peak={result['avg_peak_min']}min")
        else:
            print(f"\n[PIR, All]: {result['error']}")

    # Experiment 2: Clean events only (low carb)
    with mlflow.start_run(run_name="pir_clean"):
        mlflow.log_param("filter_clean", True)
        mlflow.log_param("user_id", USER_ID)

        result = pir_learner.train_and_validate(pir_events, filter_clean=True)
        if 'error' not in result:
            mlflow.log_metrics({
                "pir_median": result['learned_pir'],
                "pir_mean": result['mean_pir'],
                "pir_std": result['std_pir'],
                "train_samples": result['train_samples'],
                "val_samples": result['val_samples'],
                "val_mae": result['val_mae'],
                "avg_onset_min": result['avg_onset_min'],
                "avg_peak_min": result['avg_peak_min'],
            })
            print(f"[PIR, Clean]: {result['learned_pir']:.1f} (n={result['train_samples']}, MAE={result['val_mae']:.2f})")
            print(f"  Timing: onset={result['avg_onset_min']}min, peak={result['avg_peak_min']}min")
        else:
            print(f"[PIR, Clean]: {result['error']}")

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print(f"View results at: {MLFLOW_TRACKING_URI}")
    print("="*70)


async def main():
    """Run all experiments."""
    await run_experiments()


if __name__ == "__main__":
    asyncio.run(main())
