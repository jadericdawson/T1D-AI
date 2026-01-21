#!/usr/bin/env python3
"""
Test ICR and PIR Learners - Final Validation

This script tests the actual learner classes to verify
the improvements (retrospective ICR, median) work correctly.

Expected results:
- ICR: ~10 g/U (was 13.1)
- PIR: ~12 g/U (was 30.8)
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from ml.training.icr_learner import ICRLearner, learn_icr_for_user
from ml.training.pir_learner import PIRLearner, learn_pir_for_user

USER_ID = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"


async def test_icr():
    """Test ICR learner."""
    print("\n" + "=" * 60)
    print("TESTING ICR LEARNER")
    print("=" * 60)

    learner = ICRLearner()
    print(f"Configuration:")
    print(f"  min_carbs: {learner.min_carbs}")
    print(f"  min_icr: {learner.min_icr}")
    print(f"  max_icr: {learner.max_icr}")
    print(f"  use_retrospective: {learner.use_retrospective}")

    # Learn ICR
    results = await learn_icr_for_user(USER_ID, days=30)

    print("\nResults:")
    for meal_type, learned in results.items():
        if meal_type == "default":
            continue
        if learned:
            print(f"  {meal_type}: {learned.value:.1f} g/U (n={learned.sampleCount}, "
                  f"conf={learned.confidence:.2f})")
            print(f"    range: {learned.minICR:.1f} - {learned.maxICR:.1f}")
        else:
            print(f"  {meal_type}: No data")

    overall = results.get("overall")
    if overall:
        print(f"\n*** LEARNED ICR: {overall.value:.1f} g/U (expected ~10) ***")
        return overall.value
    return None


async def test_pir():
    """Test PIR learner."""
    print("\n" + "=" * 60)
    print("TESTING PIR LEARNER")
    print("=" * 60)

    learner = PIRLearner()
    print(f"Configuration:")
    print(f"  min_protein: {learner.min_protein}")
    print(f"  min_late_rise: {learner.min_late_rise}")
    print(f"  min_pir: {learner.min_pir}")
    print(f"  max_pir: {learner.max_pir}")

    # Learn PIR
    results = await learn_pir_for_user(USER_ID, days=30)

    print("\nResults:")
    overall = results.get("overall")
    if overall:
        print(f"  value: {overall.value:.1f} g/U (n={overall.sampleCount})")
        print(f"  range: {overall.minPIR:.1f} - {overall.maxPIR:.1f}")
        print(f"  timing: onset={overall.proteinOnsetMin:.0f}min, peak={overall.proteinPeakMin:.0f}min")
        print(f"\n*** LEARNED PIR: {overall.value:.1f} g/U (expected ~12) ***")
        return overall.value
    else:
        print("  No PIR learned")
        # Check for timing info
        if "timing" in results:
            timing = results["timing"]
            print(f"  timing info: onset={timing.get('avgOnsetMinutes')}min")

    return None


async def main():
    print("=" * 60)
    print("FINAL LEARNER VALIDATION")
    print(f"User: {USER_ID}")
    print("Expected: ICR ~10, PIR ~12")
    print("=" * 60)

    icr = await test_icr()
    pir = await test_pir()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if icr:
        icr_status = "GOOD" if 9 <= icr <= 11 else "NEEDS ADJUSTMENT"
        print(f"ICR: {icr:.1f} g/U (expected ~10) - {icr_status}")
    else:
        print("ICR: Not learned")

    if pir:
        pir_status = "GOOD" if 10 <= pir <= 14 else "NEEDS ADJUSTMENT"
        print(f"PIR: {pir:.1f} g/U (expected ~12) - {pir_status}")
    else:
        print("PIR: Not learned")


if __name__ == "__main__":
    asyncio.run(main())
