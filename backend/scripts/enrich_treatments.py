#!/usr/bin/env python3
"""
Batch Enrich Treatments with GPT-4.1 Macro Estimates

This script:
1. Connects to CosmosDB
2. Fetches treatments with food notes
3. Uses GPT-4.1 to estimate fat/protein from notes
4. Updates treatments in CosmosDB with enriched data

This enriched data will be used for:
- Better COB modeling (fat slows absorption)
- Improved TFT model training
- More accurate glucose predictions

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python3 scripts/enrich_treatments.py [--dry-run] [--limit N]

Author: T1D-AI
Date: 2026-01-07
"""

import sys
import os

# Load .env file
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from services.food_enrichment_service import FoodEnrichmentService, MacroEstimate
from database.cosmos_client import CosmosDBManager
from config import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_treatments_needing_enrichment(container, limit: int = 1000) -> List[Dict]:
    """Fetch treatments that need enrichment (have carbs or notes, but no GI)."""

    # Query for treatments with carbs that don't have glycemicIndex field
    query = """
        SELECT TOP @limit *
        FROM c
        WHERE IS_DEFINED(c.carbs)
          AND c.carbs > 0
          AND (NOT IS_DEFINED(c.glycemicIndex) OR c.glycemicIndex = null)
        ORDER BY c.timestamp DESC
    """

    items = list(container.query_items(
        query=query,
        parameters=[{"name": "@limit", "value": limit}],
        enable_cross_partition_query=True
    ))

    logger.info(f"Found {len(items)} treatments with carbs that need enrichment")
    return items


def estimate_from_carb_count(carbs: float) -> Dict[str, Any]:
    """
    Estimate macro content from carb count when no food notes available.
    Uses heuristics based on typical meal patterns:
    - 5-15g carbs: Snack (high GI, low fat/protein)
    - 20-40g carbs: Light meal (medium GI, medium fat/protein)
    - 50+g carbs: Full meal (lower GI, higher fat/protein)
    """
    if carbs <= 15:
        # Small snack - likely quick carbs
        return {
            'proteinG': 2.0,
            'fatG': 2.0,
            'glycemicIndex': 65,
            'absorptionRate': 'fast',
            'absorptionHalfLifeMin': 30.0,
            'enrichmentConfidence': 0.4,
            'enrichmentReasoning': f'Estimated from carb count ({carbs}g) - small snack pattern',
        }
    elif carbs <= 40:
        # Light meal
        return {
            'proteinG': 15.0,
            'fatG': 10.0,
            'glycemicIndex': 55,
            'absorptionRate': 'medium',
            'absorptionHalfLifeMin': 45.0,
            'enrichmentConfidence': 0.4,
            'enrichmentReasoning': f'Estimated from carb count ({carbs}g) - light meal pattern',
        }
    else:
        # Full meal - likely has more fat/protein
        return {
            'proteinG': 25.0,
            'fatG': 20.0,
            'glycemicIndex': 50,
            'absorptionRate': 'slow',
            'absorptionHalfLifeMin': 60.0,
            'enrichmentConfidence': 0.4,
            'enrichmentReasoning': f'Estimated from carb count ({carbs}g) - full meal pattern',
        }


async def enrich_treatment(
    service: FoodEnrichmentService,
    treatment: Dict
) -> Dict[str, Any]:
    """Enrich a single treatment with macro estimates."""

    notes = treatment.get('notes', '')
    carbs = treatment.get('carbs', 0)

    # If no notes, estimate from carb count
    if not notes or not notes.strip():
        if carbs > 0:
            enrichment = estimate_from_carb_count(carbs)
            enrichment['enrichedAt'] = datetime.now(timezone.utc).isoformat()
            return enrichment
        return {}

    try:
        estimate = await service.estimate_macros_from_description(notes, carbs)

        return {
            'proteinG': estimate.protein_g,
            'fatG': estimate.fat_g,
            'glycemicIndex': estimate.glycemic_index,
            'absorptionRate': estimate.absorption_rate,
            'absorptionHalfLifeMin': estimate.absorption_half_life_min,
            'enrichmentConfidence': estimate.confidence,
            'enrichmentReasoning': estimate.reasoning,
            'enrichedAt': datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to enrich treatment {treatment.get('id')}: {e}")
        # Fallback to carb-based estimate
        if carbs > 0:
            enrichment = estimate_from_carb_count(carbs)
            enrichment['enrichedAt'] = datetime.now(timezone.utc).isoformat()
            return enrichment
        return {}


async def update_treatment(container, treatment: Dict, enrichment: Dict, dry_run: bool = False):
    """Update a treatment in CosmosDB with enrichment data."""

    if not enrichment:
        return False

    # Merge enrichment into treatment
    updated = {**treatment, **enrichment}

    if dry_run:
        logger.info(
            f"[DRY RUN] Would update treatment {treatment.get('id')}: "
            f"notes='{treatment.get('notes', '')[:50]}...' → "
            f"protein={enrichment.get('proteinG', 0):.0f}g, "
            f"fat={enrichment.get('fatG', 0):.0f}g, "
            f"GI={enrichment.get('glycemicIndex', 0)}"
        )
        return True

    try:
        container.upsert_item(body=updated)
        logger.info(
            f"Updated treatment {treatment.get('id')}: "
            f"protein={enrichment.get('proteinG', 0):.0f}g, "
            f"fat={enrichment.get('fatG', 0):.0f}g"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to update treatment {treatment.get('id')}: {e}")
        return False


async def main(dry_run: bool = False, limit: int = 100):
    """Main enrichment pipeline."""

    logger.info("=" * 60)
    logger.info("T1D-AI Treatment Enrichment Pipeline")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
    logger.info(f"Limit: {limit} treatments")

    # Connect to CosmosDB using manager (creates container if needed)
    logger.info("Connecting to CosmosDB...")
    cosmos_manager = CosmosDBManager()
    treatments_container = cosmos_manager.get_container("treatments", "/userId")

    # Initialize food enrichment service
    logger.info("Initializing GPT-4.1 food enrichment service...")
    enrichment_service = FoodEnrichmentService()
    await enrichment_service.initialize()

    # Fetch treatments
    treatments = await fetch_treatments_needing_enrichment(treatments_container, limit)

    if not treatments:
        logger.info("No treatments found that need enrichment")
        return

    # Process treatments
    success_count = 0
    error_count = 0

    for i, treatment in enumerate(treatments, 1):
        logger.info(f"Processing {i}/{len(treatments)}: {treatment.get('notes', '')[:60]}...")

        # Get enrichment
        enrichment = await enrich_treatment(enrichment_service, treatment)

        if enrichment:
            # Update in CosmosDB
            if await update_treatment(treatments_container, treatment, enrichment, dry_run):
                success_count += 1
            else:
                error_count += 1
        else:
            error_count += 1

        # Rate limiting - avoid overwhelming GPT-4.1
        if not dry_run and i % 10 == 0:
            logger.info(f"Rate limiting pause (processed {i}/{len(treatments)})...")
            await asyncio.sleep(1)

    # Summary
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed: {len(treatments)}")
    logger.info(f"Successfully enriched: {success_count}")
    logger.info(f"Errors: {error_count}")

    if dry_run:
        logger.info("\nThis was a DRY RUN - no changes were made to CosmosDB")
        logger.info("Run without --dry-run to apply changes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enrich treatments with GPT-4.1 macro estimates')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without updating CosmosDB')
    parser.add_argument('--limit', type=int, default=100, help='Maximum treatments to process')

    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, limit=args.limit))
