#!/usr/bin/env python3
"""
Quick ICR/PIR Training - No MLflow, faster execution.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from database.repositories import (
    GlucoseRepository, TreatmentRepository, LearnedISFRepository
)

USER_ID = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"


async def get_nearest_bg(glucose_repo, user_id: str, target_time: datetime, window_minutes: int):
    """Get nearest BG reading."""
    start = target_time - timedelta(minutes=window_minutes)
    end = target_time + timedelta(minutes=window_minutes)

    readings = await glucose_repo.get_history(
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


def create_meal_groups(treatments):
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


def estimate_protein(notes: str) -> float:
    """Estimate protein from meal notes."""
    if not notes:
        return 0

    notes_lower = notes.lower()

    protein_foods = {
        "steak": 40, "beef": 30, "chicken": 25, "turkey": 25,
        "salmon": 30, "tuna": 30, "fish": 25, "shrimp": 20,
        "burger": 20, "cheeseburger": 22, "patty": 15,
        "hot dog": 7, "hotdog": 7, "sausage": 15, "bacon": 10,
        "eggs": 12, "egg": 6, "omelet": 18,
        "meatball": 8, "meat": 15,
        "cheese": 7, "milk": 8, "chocolate milk": 8,
        "sandwich": 10, "bologna": 5,
        "pizza": 12, "taco": 8, "tacos": 16,
    }

    estimated = 0
    for food, protein in protein_foods.items():
        if food in notes_lower:
            estimated += protein

    return estimated


async def train_icr():
    """Train ICR with improved criteria."""
    print("\n" + "="*60)
    print("ICR TRAINING")
    print("="*60)

    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()
    isf_repo = LearnedISFRepository()

    # Get ISF
    isf_record = await isf_repo.get(USER_ID, "meal")
    isf = isf_record.value if isf_record else 50.0
    print(f"Using ISF: {isf:.1f}")

    # Get treatments
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    treatments = await treatment_repo.get_by_user(USER_ID, start_time=start_time, limit=2000)
    print(f"Total treatments: {len(treatments)}")

    # Group into meals
    meal_groups = create_meal_groups(treatments)
    print(f"Meal groups: {len(meal_groups)}")

    # Collect meal events
    target_bg = 100
    events = []

    for meal_ts, group_treatments in meal_groups.items():
        total_carbs = sum(t.carbs or 0 for t in group_treatments)
        total_insulin = sum(t.insulin or 0 for t in group_treatments)

        if total_carbs < 15 or total_insulin < 0.5:
            continue

        # Get BG
        bg_before = await get_nearest_bg(glucose_repo, USER_ID, meal_ts, 15)
        bg_after = await get_nearest_bg(glucose_repo, USER_ID, meal_ts + timedelta(hours=2.5), 45)

        if not bg_before or not bg_after:
            continue

        # Calculate correction
        correction = max(0, (bg_before - target_bg) / isf)
        meal_insulin = max(0.1, total_insulin - correction)

        # Raw ICR
        raw_icr = total_carbs / meal_insulin

        # Retrospective ICR (what would have hit target)
        carb_bg_rise = bg_after - bg_before + (total_insulin * isf)
        if carb_bg_rise > 0:
            needed_insulin = carb_bg_rise / isf
            retro_icr = total_carbs / needed_insulin if needed_insulin > 0.1 else raw_icr
        else:
            retro_icr = raw_icr

        # Filter range
        if 5 <= retro_icr <= 20:
            events.append({
                'carbs': total_carbs,
                'insulin': total_insulin,
                'bg_before': bg_before,
                'bg_after': bg_after,
                'raw_icr': raw_icr,
                'retro_icr': retro_icr,
                'outcome': 'good' if 70 <= bg_after <= 160 else 'bad'
            })

    print(f"Valid meal events: {len(events)}")

    if not events:
        print("No valid events!")
        return

    # Train/validate split
    split_idx = int(len(events) * 0.7)
    train = events[:split_idx]
    val = events[split_idx:]

    # Calculate ICR (median)
    train_icrs = [e['retro_icr'] for e in train]
    learned_icr = float(np.median(train_icrs))

    # Good outcomes only
    good_train = [e for e in train if e['outcome'] == 'good']
    good_icr = float(np.median([e['retro_icr'] for e in good_train])) if good_train else learned_icr

    print(f"\n--- RESULTS ---")
    print(f"Retro ICR (median, all): {learned_icr:.1f} g/U (n={len(train)})")
    print(f"Retro ICR (median, good outcomes): {good_icr:.1f} g/U (n={len(good_train)})")
    print(f"Raw ICR (median): {float(np.median([e['raw_icr'] for e in train])):.1f}")
    print(f"Range: {np.min(train_icrs):.1f} - {np.max(train_icrs):.1f}")

    # Validation
    val_errors = []
    for e in val:
        pred_dose = e['carbs'] / learned_icr
        actual_meal_dose = e['insulin'] - max(0, (e['bg_before'] - 100) / isf)
        error = abs(pred_dose - actual_meal_dose)
        val_errors.append(error)

    print(f"Validation MAE: {np.mean(val_errors):.2f} U")

    return learned_icr


async def train_pir():
    """Train PIR with improved criteria."""
    print("\n" + "="*60)
    print("PIR TRAINING")
    print("="*60)

    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()
    isf_repo = LearnedISFRepository()

    # Get ISF
    isf_record = await isf_repo.get(USER_ID, "fasting")
    isf = isf_record.value if isf_record else 50.0
    print(f"Using ISF: {isf:.1f}")

    # Get treatments
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    treatments = await treatment_repo.get_by_user(USER_ID, start_time=start_time, limit=2000)

    # Find protein events
    events = []

    for t in treatments:
        protein = t.protein or 0
        carbs = t.carbs or 0
        notes = t.notes or ''

        # Estimate protein if not logged
        if protein < 10:
            protein = estimate_protein(notes)

        if protein < 10:
            continue

        ts = t.timestamp
        if ts.tzinfo:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

        # Get BG at multiple times
        bg_90 = await get_nearest_bg(glucose_repo, USER_ID, ts + timedelta(minutes=90), 30)
        bg_180 = await get_nearest_bg(glucose_repo, USER_ID, ts + timedelta(minutes=180), 30)

        if not bg_90 or not bg_180:
            continue

        # Late rise
        late_rise = max(0, bg_180 - bg_90)

        if late_rise >= 15:  # Minimum rise
            insulin_equiv = late_rise / isf
            pir = protein / insulin_equiv if insulin_equiv > 0.1 else None

            if pir and 8 <= pir <= 25:  # Tighter range
                events.append({
                    'protein': protein,
                    'carbs': carbs,
                    'late_rise': late_rise,
                    'pir': pir,
                    'is_clean': carbs <= 40
                })

    print(f"Valid protein events: {len(events)}")

    if not events:
        print("No valid events!")
        return

    # Train/validate split
    split_idx = max(1, int(len(events) * 0.7))
    train = events[:split_idx]
    val = events[split_idx:]

    # Calculate PIR (median)
    train_pirs = [e['pir'] for e in train]
    learned_pir = float(np.median(train_pirs))

    # Clean events only
    clean_train = [e for e in train if e['is_clean']]
    clean_pir = float(np.median([e['pir'] for e in clean_train])) if clean_train else learned_pir

    print(f"\n--- RESULTS ---")
    print(f"PIR (median, all): {learned_pir:.1f} g/U (n={len(train)})")
    print(f"PIR (median, clean <40g carbs): {clean_pir:.1f} g/U (n={len(clean_train)})")
    print(f"Range: {np.min(train_pirs):.1f} - {np.max(train_pirs):.1f}")

    return learned_pir


async def main():
    print("="*60)
    print("QUICK ICR/PIR TRAINING")
    print(f"User: {USER_ID}")
    print("Expected: ICR ~10, PIR ~12")
    print("="*60)

    icr = await train_icr()
    pir = await train_pir()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Learned ICR: {icr:.1f} g/U (expected ~10)")
    print(f"Learned PIR: {pir:.1f} g/U (expected ~12)")


if __name__ == "__main__":
    asyncio.run(main())
