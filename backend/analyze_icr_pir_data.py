#!/usr/bin/env python3
"""
Analyze ICR and PIR data to understand why learned values are high.
Expected: ICR ~10:1, PIR ~12:1
Actual: ICR 13.1, PIR 30.8
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

USER_ID = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"


async def analyze_meal_data():
    """Analyze meal bolus data to understand ICR calculation."""
    print("\n" + "="*70)
    print("ICR DATA ANALYSIS")
    print("="*70)

    from database.repositories import GlucoseRepository, TreatmentRepository, LearnedISFRepository

    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()
    isf_repo = LearnedISFRepository()

    # Get ISF
    isf_record = await isf_repo.get(USER_ID, "meal")
    isf = isf_record.value if isf_record else 50.0
    print(f"\nUsing ISF: {isf:.1f} mg/dL/U")

    # Get treatments from last 30 days
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    treatments = await treatment_repo.get_by_user(USER_ID, start_time=start_time, limit=2000)

    print(f"Total treatments: {len(treatments)}")

    # Find meal events (carbs + insulin)
    meal_events = []

    for t in treatments:
        carbs = t.carbs or 0
        insulin = t.insulin or 0

        if carbs >= 10 and insulin >= 0.5:
            # Get BG around meal time
            ts = t.timestamp
            if ts.tzinfo:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

            bg_before = await get_nearest_bg(glucose_repo, USER_ID, ts, 15)
            bg_after = await get_nearest_bg(glucose_repo, USER_ID, ts + timedelta(hours=2.5), 45)

            if bg_before and bg_after:
                # Calculate correction component
                target_bg = 100
                correction_insulin = max(0, (bg_before - target_bg) / isf)
                meal_insulin = max(0.1, insulin - correction_insulin)
                icr = carbs / meal_insulin

                meal_events.append({
                    'timestamp': ts,
                    'carbs': carbs,
                    'insulin': insulin,
                    'bg_before': bg_before,
                    'bg_after': bg_after,
                    'correction': correction_insulin,
                    'meal_insulin': meal_insulin,
                    'icr': icr,
                    'notes': t.notes or ''
                })

    print(f"\nMeal events with BG data: {len(meal_events)}")

    if not meal_events:
        print("No meal events found!")
        return

    # Analyze the data
    df = pd.DataFrame(meal_events)

    print("\n--- RAW ICR Distribution ---")
    print(f"Mean ICR: {df['icr'].mean():.1f}")
    print(f"Median ICR: {df['icr'].median():.1f}")
    print(f"Std ICR: {df['icr'].std():.1f}")
    print(f"Min ICR: {df['icr'].min():.1f}")
    print(f"Max ICR: {df['icr'].max():.1f}")

    # ICR by outcome
    print("\n--- ICR by BG Outcome ---")
    good_outcome = df[(df['bg_after'] >= 70) & (df['bg_after'] <= 140)]
    high_outcome = df[df['bg_after'] > 140]
    low_outcome = df[df['bg_after'] < 70]

    print(f"Good outcome (70-140): n={len(good_outcome)}, ICR={good_outcome['icr'].mean():.1f}" if len(good_outcome) > 0 else "Good outcome: 0")
    print(f"High outcome (>140): n={len(high_outcome)}, ICR={high_outcome['icr'].mean():.1f}" if len(high_outcome) > 0 else "High outcome: 0")
    print(f"Low outcome (<70): n={len(low_outcome)}, ICR={low_outcome['icr'].mean():.1f}" if len(low_outcome) > 0 else "Low outcome: 0")

    # Events where outcome was high suggest ICR should be LOWER (need more insulin)
    print("\n--- Analysis ---")
    print("If BG ends high after meal -> ICR should be LOWER (gave too little insulin)")
    print("If BG ends low after meal -> ICR should be HIGHER (gave too much insulin)")

    # Calculate what ICR SHOULD have been for each meal
    print("\n--- Retrospective ICR Calculation ---")
    # For each meal, what ICR would have resulted in target BG?
    # final_bg = bg_before - (insulin * isf) + carb_effect
    # carb_effect = carbs / true_icr * isf
    # We want final_bg = target, so:
    # target = bg_before - (insulin * isf) + (carbs / true_icr * isf)
    # carbs / true_icr * isf = target - bg_before + insulin * isf
    # true_icr = carbs * isf / (target - bg_before + insulin * isf)

    target_bg = 100
    retrospective_icrs = []
    for _, row in df.iterrows():
        # What ICR would have hit target?
        numerator = row['carbs'] * isf
        denominator = target_bg - row['bg_before'] + row['insulin'] * isf

        if denominator > 0:
            retro_icr = numerator / denominator
            if 3 <= retro_icr <= 30:
                retrospective_icrs.append(retro_icr)

    if retrospective_icrs:
        print(f"Retrospective ICR (to hit target):")
        print(f"  Mean: {np.mean(retrospective_icrs):.1f}")
        print(f"  Median: {np.median(retrospective_icrs):.1f}")

    # Show sample events
    print("\n--- Sample Meal Events ---")
    print(f"{'Carbs':>6} {'Ins':>5} {'BG1':>5} {'BG2':>5} {'Corr':>5} {'Meal':>5} {'ICR':>6} Notes")
    for _, row in df.head(20).iterrows():
        print(f"{row['carbs']:>6.0f} {row['insulin']:>5.1f} {row['bg_before']:>5.0f} {row['bg_after']:>5.0f} {row['correction']:>5.1f} {row['meal_insulin']:>5.1f} {row['icr']:>6.1f} {row['notes'][:30]}")

    return df


async def analyze_protein_data():
    """Analyze protein data to understand PIR calculation."""
    print("\n" + "="*70)
    print("PIR DATA ANALYSIS")
    print("="*70)

    from database.repositories import GlucoseRepository, TreatmentRepository, LearnedISFRepository

    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()
    isf_repo = LearnedISFRepository()

    # Get ISF
    isf_record = await isf_repo.get(USER_ID, "fasting")
    isf = isf_record.value if isf_record else 50.0
    print(f"\nUsing ISF: {isf:.1f} mg/dL/U")

    # Get treatments
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    treatments = await treatment_repo.get_by_user(USER_ID, start_time=start_time, limit=2000)

    # Find protein-containing meals
    protein_events = []

    for t in treatments:
        protein = t.protein or 0
        carbs = t.carbs or 0
        notes = t.notes or ''

        # Check if has protein or might have protein based on notes
        has_protein = protein > 0
        might_have_protein = any(kw in notes.lower() for kw in [
            'steak', 'beef', 'chicken', 'turkey', 'fish', 'salmon', 'tuna',
            'egg', 'bacon', 'sausage', 'pork', 'ham', 'lamb', 'cheese',
            'hot dog', 'burger', 'meatball', 'protein', 'meat'
        ])

        if has_protein or might_have_protein:
            ts = t.timestamp
            if ts.tzinfo:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

            # Get BG at various times after meal
            bg_meal = await get_nearest_bg(glucose_repo, USER_ID, ts, 15)
            bg_90min = await get_nearest_bg(glucose_repo, USER_ID, ts + timedelta(minutes=90), 30)
            bg_120min = await get_nearest_bg(glucose_repo, USER_ID, ts + timedelta(minutes=120), 30)
            bg_180min = await get_nearest_bg(glucose_repo, USER_ID, ts + timedelta(minutes=180), 30)
            bg_240min = await get_nearest_bg(glucose_repo, USER_ID, ts + timedelta(minutes=240), 30)

            protein_events.append({
                'timestamp': ts,
                'protein': protein,
                'carbs': carbs,
                'insulin': t.insulin or 0,
                'notes': notes,
                'bg_meal': bg_meal,
                'bg_90min': bg_90min,
                'bg_120min': bg_120min,
                'bg_180min': bg_180min,
                'bg_240min': bg_240min,
            })

    print(f"\nProtein-related events: {len(protein_events)}")

    if not protein_events:
        print("No protein events found!")
        return

    # Analyze late rise patterns
    print("\n--- Late BG Rise Analysis ---")
    print(f"{'Protein':>8} {'Carbs':>6} {'BGmeal':>6} {'BG90':>6} {'BG120':>6} {'BG180':>6} {'BG240':>6} {'Rise':>6} Notes")

    late_rises = []
    for ev in protein_events:
        bg_meal = ev['bg_meal']
        bg_90 = ev['bg_90min']
        bg_180 = ev['bg_180min']

        if bg_90 and bg_180:
            # Late rise = peak after 90min minus value at 90min (baseline after carbs)
            late_rise = max(0, (ev['bg_180min'] or 0) - bg_90)

            protein_est = ev['protein'] if ev['protein'] > 0 else estimate_protein(ev['notes'])

            print(f"{protein_est:>8.0f} {ev['carbs']:>6.0f} {bg_meal or 0:>6.0f} {bg_90:>6.0f} {ev['bg_120min'] or 0:>6.0f} {ev['bg_180min'] or 0:>6.0f} {ev['bg_240min'] or 0:>6.0f} {late_rise:>6.0f} {ev['notes'][:25]}")

            if protein_est > 0 and late_rise > 10:
                # Calculate PIR
                insulin_equiv = late_rise / isf
                pir = protein_est / insulin_equiv if insulin_equiv > 0 else 0
                late_rises.append({
                    'protein': protein_est,
                    'late_rise': late_rise,
                    'pir': pir
                })

    if late_rises:
        print("\n--- PIR from Late Rises ---")
        pirs = [x['pir'] for x in late_rises if 5 <= x['pir'] <= 100]
        if pirs:
            print(f"Mean PIR: {np.mean(pirs):.1f}")
            print(f"Median PIR: {np.median(pirs):.1f}")

        # The issue: PIR of 30 means 30g protein = 1U insulin effect
        # Expected 12 means 12g protein = 1U insulin effect
        # So we're detecting LESS protein effect than expected
        print("\n--- Analysis ---")
        print("PIR too high = not detecting enough protein effect on BG")
        print("Possible causes:")
        print("  1. Not looking at the right time window for late rise")
        print("  2. Protein estimation from notes is too high")
        print("  3. Late rise detection threshold too high")
        print("  4. Carb-induced rise masking protein rise")


def estimate_protein(notes: str) -> float:
    """Estimate protein from meal notes."""
    if not notes:
        return 0

    notes_lower = notes.lower()
    protein_keywords = {
        "steak": 40, "beef": 30, "chicken": 25, "turkey": 25,
        "fish": 25, "salmon": 30, "tuna": 30, "shrimp": 20,
        "egg": 6, "eggs": 12, "bacon": 10, "sausage": 15,
        "pork": 25, "ham": 20, "lamb": 25,
        "cheese": 10, "greek yogurt": 15, "cottage cheese": 15,
        "protein shake": 25, "protein bar": 20,
        "hot dog": 7, "burger": 20, "meatball": 15,
    }

    estimated = 0
    for keyword, protein in protein_keywords.items():
        if keyword in notes_lower:
            estimated += protein

    return estimated


async def get_nearest_bg(repo, user_id: str, target_time: datetime, window_minutes: int) -> float:
    """Get nearest BG reading."""
    start = target_time - timedelta(minutes=window_minutes)
    end = target_time + timedelta(minutes=window_minutes)

    readings = await repo.get_history(user_id, start_time=start, end_time=end, limit=10)
    if not readings:
        return None

    def time_diff(r):
        r_ts = r.timestamp
        if r_ts.tzinfo:
            r_ts = r_ts.replace(tzinfo=None)
        t_ts = target_time
        if t_ts.tzinfo:
            t_ts = t_ts.replace(tzinfo=None)
        return abs((r_ts - t_ts).total_seconds())

    closest = min(readings, key=time_diff)
    return closest.value


async def main():
    print("="*70)
    print("ICR/PIR DATA ANALYSIS")
    print(f"User: {USER_ID}")
    print("Expected ICR: ~10:1, Expected PIR: ~12:1")
    print("="*70)

    await analyze_meal_data()
    await analyze_protein_data()


if __name__ == "__main__":
    asyncio.run(main())
