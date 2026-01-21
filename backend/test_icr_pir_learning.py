#!/usr/bin/env python3
"""
Test ICR and PIR Learning locally against CosmosDB
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# User ID from the conversation
USER_ID = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"


async def test_icr_learning():
    """Test ICR (Carb-to-Insulin Ratio) learning."""
    print("\n" + "="*60)
    print("Testing ICR Learning")
    print("="*60)

    from ml.training.icr_learner import ICRLearner

    learner = ICRLearner()

    # Learn all ICR values
    print(f"\nLearning ICR for user {USER_ID} over 30 days...")
    results = await learner.learn_all_icr(USER_ID, days=30)

    # Display results
    overall = results.get("overall")
    breakfast = results.get("breakfast")
    lunch = results.get("lunch")
    dinner = results.get("dinner")
    default = results.get("default", 10.0)

    print("\n--- ICR Results ---")
    if overall:
        print(f"Overall ICR: {overall.value:.1f} g/U")
        print(f"  Samples: {overall.sampleCount}")
        print(f"  Confidence: {overall.confidence:.2f}")
        print(f"  Range: {overall.minICR:.1f} - {overall.maxICR:.1f} g/U")
        if overall.mealTypePattern:
            pattern_str = ", ".join(f"{k}={v:.1f}" for k, v in overall.mealTypePattern.items() if v)
            if pattern_str:
                print(f"  Meal patterns: {pattern_str}")
    else:
        print("Overall ICR: Not enough data")

    if breakfast:
        print(f"\nBreakfast ICR: {breakfast.value:.1f} g/U (n={breakfast.sampleCount})")
    if lunch:
        print(f"Lunch ICR: {lunch.value:.1f} g/U (n={lunch.sampleCount})")
    if dinner:
        print(f"Dinner ICR: {dinner.value:.1f} g/U (n={dinner.sampleCount})")

    print(f"\nDefault ICR to use: {default:.1f} g/U")

    return results


async def test_pir_learning():
    """Test PIR (Protein-to-Insulin Ratio) learning."""
    print("\n" + "="*60)
    print("Testing PIR Learning")
    print("="*60)

    from ml.training.pir_learner import PIRLearner

    learner = PIRLearner()

    # Learn all PIR values
    print(f"\nLearning PIR for user {USER_ID} over 30 days...")
    results = await learner.learn_all_pir(USER_ID, days=30)

    # Display results
    overall = results.get("overall")
    breakfast = results.get("breakfast")
    lunch = results.get("lunch")
    dinner = results.get("dinner")
    default = results.get("default", 25.0)
    timing = results.get("timing")

    print("\n--- PIR Results ---")
    if overall:
        print(f"Overall PIR: {overall.value:.1f} g/U")
        print(f"  Samples: {overall.sampleCount}")
        print(f"  Confidence: {overall.confidence:.2f}")
        print(f"  Range: {overall.minPIR:.1f} - {overall.maxPIR:.1f} g/U")
        if overall.proteinOnsetMin:
            print(f"  Protein onset: {overall.proteinOnsetMin:.0f} min")
        if overall.proteinPeakMin:
            print(f"  Protein peak: {overall.proteinPeakMin:.0f} min")
    else:
        print("Overall PIR: Not enough protein data")
        print("  (Need high-protein meals with detectable late BG rise)")

    if breakfast:
        print(f"\nBreakfast PIR: {breakfast.value:.1f} g/U (n={breakfast.sampleCount})")
    if lunch:
        print(f"Lunch PIR: {lunch.value:.1f} g/U (n={lunch.sampleCount})")
    if dinner:
        print(f"Dinner PIR: {dinner.value:.1f} g/U (n={dinner.sampleCount})")

    print(f"\nDefault PIR to use: {default:.1f} g/U")

    if timing:
        print(f"\nProtein Timing:")
        print(f"  Onset: {timing.get('avgOnsetMinutes', 120)} min")
        print(f"  Peak: {timing.get('avgPeakMinutes', 180)} min")

    return results


async def test_meal_dose():
    """Test meal dose calculation with ICR/PIR."""
    print("\n" + "="*60)
    print("Testing Meal Dose Calculation")
    print("="*60)

    from database.repositories import LearnedISFRepository, LearnedICRRepository, LearnedPIRRepository

    isf_repo = LearnedISFRepository()
    icr_repo = LearnedICRRepository()
    pir_repo = LearnedPIRRepository()

    # Get learned values
    isf = await isf_repo.get(USER_ID, "meal")
    icr = await icr_repo.get(USER_ID, "overall")
    pir = await pir_repo.get(USER_ID, "overall")

    # Example meal
    current_bg = 180
    target_bg = 100
    carbs = 45
    protein = 30

    print(f"\n--- Example Meal ---")
    print(f"Current BG: {current_bg} mg/dL")
    print(f"Target BG: {target_bg} mg/dL")
    print(f"Carbs: {carbs}g")
    print(f"Protein: {protein}g")

    # Use learned values or defaults
    isf_val = isf.value if isf else 50.0
    icr_val = icr.value if icr else 10.0
    pir_val = pir.value if pir else 25.0
    protein_upfront_pct = 40  # 40% upfront, 60% delayed

    print(f"\n--- Using Ratios ---")
    print(f"ISF: {isf_val:.1f} mg/dL/U {'(learned)' if isf else '(default)'}")
    print(f"ICR: {icr_val:.1f} g/U {'(learned)' if icr else '(default)'}")
    print(f"PIR: {pir_val:.1f} g/U {'(learned)' if pir else '(default)'}")

    # Calculate doses
    correction_dose = max(0, (current_bg - target_bg) / isf_val)
    carb_dose = carbs / icr_val
    protein_dose_total = protein / pir_val
    protein_dose_immediate = protein_dose_total * (protein_upfront_pct / 100)
    protein_dose_delayed = protein_dose_total * (1 - protein_upfront_pct / 100)

    immediate_total = correction_dose + carb_dose + protein_dose_immediate
    delayed_total = protein_dose_delayed
    grand_total = immediate_total + delayed_total

    print(f"\n--- Dose Breakdown ---")
    print(f"Correction: {correction_dose:.2f}U (for BG {current_bg} -> {target_bg})")
    print(f"Carbs:      {carb_dose:.2f}U ({carbs}g / {icr_val:.1f})")
    print(f"Protein:    {protein_dose_total:.2f}U total ({protein}g / {pir_val:.1f})")
    print(f"  - NOW:    {protein_dose_immediate:.2f}U ({protein_upfront_pct}%)")
    print(f"  - LATER:  {protein_dose_delayed:.2f}U ({100-protein_upfront_pct}%)")

    print(f"\n--- Totals ---")
    print(f"Give NOW:   {immediate_total:.2f}U (correction + carbs + immediate protein)")
    print(f"Give LATER: {delayed_total:.2f}U (delayed protein - extend or give at 2h)")
    print(f"GRAND TOTAL: {grand_total:.2f}U")

    if pir and pir.proteinOnsetMin and pir.proteinPeakMin:
        print(f"\nTiming advice: Give delayed dose as extended bolus over {int(pir.proteinOnsetMin)}-{int(pir.proteinPeakMin)} min")


async def main():
    """Run all tests."""
    print("="*60)
    print("ICR/PIR Learning Test")
    print(f"User ID: {USER_ID}")
    print("="*60)

    try:
        # Test ICR learning
        icr_results = await test_icr_learning()

        # Test PIR learning
        pir_results = await test_pir_learning()

        # Test meal dose calculation
        await test_meal_dose()

        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
