#!/usr/bin/env python3
"""
Run ISF learning for a user and store the learned ISF in CosmosDB.

Usage:
    python scripts/learn_isf.py --user-id <user_id> [--days 30]

The ISF learner analyzes clean correction boluses (no carbs nearby) to calculate
the actual ISF = (BG_before - BG_after) / insulin_units
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.training.isf_learner import ISFLearner, learn_isf_for_user

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Learn ISF from glucose and treatment data")
    parser.add_argument("--user-id", required=True, help="User ID to learn ISF for")
    parser.add_argument("--days", type=int, default=30, help="Days of history to analyze (default: 30)")
    args = parser.parse_args()

    logger.info(f"Learning ISF for user {args.user_id} using {args.days} days of history")

    try:
        result = await learn_isf_for_user(args.user_id, args.days)

        print("\n" + "=" * 50)
        print("ISF LEARNING RESULTS")
        print("=" * 50)

        if result["fasting"]:
            fasting = result["fasting"]
            print(f"\nFasting ISF: {fasting.value:.1f} mg/dL per unit")
            print(f"  - Sample count: {fasting.sampleCount}")
            print(f"  - Confidence: {fasting.confidence:.2f}")
            print(f"  - Range: {fasting.minISF:.1f} - {fasting.maxISF:.1f}")
            print(f"  - Std dev: {fasting.stdISF:.1f}")
            if any(fasting.timeOfDayPattern.values()):
                print("  - Time of day patterns:")
                for tod, val in fasting.timeOfDayPattern.items():
                    if val:
                        print(f"      {tod}: {val:.1f}")
        else:
            print("\nFasting ISF: Not enough clean correction boluses")

        if result["meal"]:
            meal = result["meal"]
            print(f"\nMeal ISF: {meal.value:.1f} mg/dL per unit")
            print(f"  - Sample count: {meal.sampleCount}")
            print(f"  - Confidence: {meal.confidence:.2f}")
        else:
            print("\nMeal ISF: Not enough meal boluses")

        print(f"\nDefault ISF (for calculations): {result['default']:.1f}")
        print("=" * 50 + "\n")

        # Show some history
        if result["fasting"] and result["fasting"].history:
            print("Recent fasting ISF observations:")
            for dp in result["fasting"].history[-10:]:
                print(f"  {dp.timestamp}: BG {dp.bgBefore} -> {dp.bgAfter} with {dp.insulinUnits}U = ISF {dp.value:.1f}")

    except Exception as e:
        logger.error(f"Failed to learn ISF: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
