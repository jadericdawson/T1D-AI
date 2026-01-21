#!/usr/bin/env python3
"""
Test script for TFT Training Pipeline

Validates that the comprehensive TFT training pipeline works correctly with:
1. Data quality filtering (skips windows with missing treatments)
2. Per-timestep IOB/COB computation
3. Time exclusion patterns (e.g., school hours)
4. Temporal train/val/test splitting
5. Weighted quantile loss

Usage:
    python scripts/test_tft_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(
    n_days: int = 7,
    include_gaps: bool = True,
    include_school_hours: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic glucose and treatment data for testing.

    Args:
        n_days: Number of days of data
        include_gaps: If True, simulate missing treatment periods
        include_school_hours: If True, add school hours with no treatments

    Returns:
        Tuple of (glucose_df, treatments_df)
    """
    np.random.seed(42)

    start_time = datetime(2024, 1, 1, 0, 0, 0)

    # Generate glucose readings every 5 minutes
    n_readings = n_days * 288  # 288 readings per day (24h * 60/5)

    glucose_records = []
    current_bg = 120.0  # Start at 120 mg/dL

    for i in range(n_readings):
        timestamp = start_time + timedelta(minutes=i * 5)

        # Simulate glucose with some variability
        change = np.random.normal(0, 3)

        # Add meal-like patterns (rises after 8am, 12pm, 6pm)
        hour = timestamp.hour
        if hour in [8, 12, 18]:
            change += np.random.uniform(5, 15)

        # Add insulin effect patterns
        if hour in [9, 13, 19]:
            change -= np.random.uniform(3, 8)

        # Natural drift toward 110
        change += (110 - current_bg) * 0.01

        current_bg = np.clip(current_bg + change, 50, 350)

        # Add trend
        if i > 0:
            trend = 4 if change > 2 else (5 if change < -2 else 0)
        else:
            trend = 0

        glucose_records.append({
            'timestamp': timestamp,
            'value': round(current_bg, 1),
            'trend': trend
        })

    glucose_df = pd.DataFrame(glucose_records)

    # Generate treatments
    treatment_records = []

    for day in range(n_days):
        day_start = start_time + timedelta(days=day)
        day_of_week = day_start.weekday()

        # Breakfast: 7:30 AM
        meal_time = day_start + timedelta(hours=7, minutes=30)
        treatment_records.append({
            'timestamp': meal_time,
            'type': 'carbs',
            'carbs': np.random.uniform(30, 50),
            'protein': np.random.uniform(10, 20),
            'fat': np.random.uniform(5, 15),
            'glycemicIndex': np.random.choice([50, 55, 65, 75])
        })
        treatment_records.append({
            'timestamp': meal_time + timedelta(minutes=5),
            'type': 'insulin',
            'insulin': np.random.uniform(3, 6)
        })

        # School hours (8am-3pm on weekdays): Include gaps if testing
        if include_school_hours and day_of_week < 5:
            # On school days, no lunch treatment logged (simulating missing data)
            if not include_gaps:
                # If we're testing with complete data, add lunch
                lunch_time = day_start + timedelta(hours=12)
                treatment_records.append({
                    'timestamp': lunch_time,
                    'type': 'carbs',
                    'carbs': np.random.uniform(40, 60),
                    'protein': np.random.uniform(15, 25),
                    'fat': np.random.uniform(10, 20),
                    'glycemicIndex': 60
                })
        else:
            # Weekend lunch
            lunch_time = day_start + timedelta(hours=12, minutes=30)
            treatment_records.append({
                'timestamp': lunch_time,
                'type': 'carbs',
                'carbs': np.random.uniform(40, 60),
                'protein': np.random.uniform(15, 25),
                'fat': np.random.uniform(10, 20),
                'glycemicIndex': 55
            })
            treatment_records.append({
                'timestamp': lunch_time + timedelta(minutes=5),
                'type': 'insulin',
                'insulin': np.random.uniform(4, 7)
            })

        # Dinner: 6:30 PM
        dinner_time = day_start + timedelta(hours=18, minutes=30)
        treatment_records.append({
            'timestamp': dinner_time,
            'type': 'carbs',
            'carbs': np.random.uniform(50, 80),
            'protein': np.random.uniform(20, 35),
            'fat': np.random.uniform(15, 30),
            'glycemicIndex': np.random.choice([45, 55, 65])
        })
        treatment_records.append({
            'timestamp': dinner_time + timedelta(minutes=5),
            'type': 'insulin',
            'insulin': np.random.uniform(5, 9)
        })

        # Optional bedtime snack
        if np.random.random() > 0.5:
            snack_time = day_start + timedelta(hours=21)
            treatment_records.append({
                'timestamp': snack_time,
                'type': 'carbs',
                'carbs': np.random.uniform(15, 25),
                'glycemicIndex': 50
            })

    treatments_df = pd.DataFrame(treatment_records)

    # Add insulin column to treatments with carbs (set to 0)
    if 'insulin' not in treatments_df.columns:
        treatments_df['insulin'] = 0.0
    treatments_df['insulin'] = treatments_df['insulin'].fillna(0.0)
    treatments_df['carbs'] = treatments_df.get('carbs', 0).fillna(0.0)

    return glucose_df, treatments_df


def test_data_quality_filter():
    """Test the DataQualityFilter with missing treatment detection."""
    from ml.training.tft_trainer import (
        DataQualityFilter, TFTTrainingConfig, TimeExclusionPattern
    )

    print("\n" + "="*60)
    print("TEST 1: DataQualityFilter")
    print("="*60)

    # Generate data with gaps
    glucose_df, treatments_df = generate_sample_data(n_days=3, include_gaps=True)

    # Create config with school hours exclusion
    config = TFTTrainingConfig.with_school_hours_excluded()
    data_filter = DataQualityFilter(config)

    # Test completeness score for a specific window
    window_start = datetime(2024, 1, 2, 10, 0)  # During school hours
    window_end = datetime(2024, 1, 2, 13, 0)  # 3-hour window

    score, reason = data_filter.calculate_completeness_score(
        glucose_df, treatments_df, window_start, window_end
    )

    print(f"\nSchool hours window (10am-1pm on Tuesday):")
    print(f"  Score: {score:.2f}")
    print(f"  Reason: {reason}")
    assert score == 0.0, "School hours should be excluded"
    assert "school_hours" in reason, "Should be excluded for school hours"

    # Test a weekend window
    window_start = datetime(2024, 1, 6, 10, 0)  # Saturday
    window_end = datetime(2024, 1, 6, 13, 0)

    score, reason = data_filter.calculate_completeness_score(
        glucose_df, treatments_df, window_start, window_end
    )

    print(f"\nWeekend window (10am-1pm on Saturday):")
    print(f"  Score: {score:.2f}")
    print(f"  Reason: {reason if reason else 'Valid'}")

    print("\n✅ DataQualityFilter test passed")


def test_per_timestep_iob_cob():
    """Test per-timestep IOB/COB computation."""
    from ml.feature_engineering import compute_per_timestep_iob, compute_per_timestep_cob

    print("\n" + "="*60)
    print("TEST 2: Per-Timestep IOB/COB Computation")
    print("="*60)

    # Create simple test data
    glucose_df, treatments_df = generate_sample_data(n_days=1, include_gaps=False)

    # Compute IOB for all timesteps
    iob_values = compute_per_timestep_iob(glucose_df, treatments_df)
    cob_values = compute_per_timestep_cob(glucose_df, treatments_df)

    print(f"\nIOB statistics:")
    print(f"  Min: {iob_values.min():.2f} U")
    print(f"  Max: {iob_values.max():.2f} U")
    print(f"  Mean: {iob_values.mean():.2f} U")
    print(f"  Non-zero values: {(iob_values > 0).sum()}")

    print(f"\nCOB statistics:")
    print(f"  Min: {cob_values.min():.2f} g")
    print(f"  Max: {cob_values.max():.2f} g")
    print(f"  Mean: {cob_values.mean():.2f} g")
    print(f"  Non-zero values: {(cob_values > 0).sum()}")

    # Verify IOB peaks after insulin boluses
    insulin_times = treatments_df[treatments_df['insulin'] > 0]['timestamp']
    print(f"\n{len(insulin_times)} insulin boluses in data")

    # IOB should be non-zero for some period after each bolus
    assert iob_values.max() > 0, "IOB should have non-zero values after boluses"
    assert cob_values.max() > 0, "COB should have non-zero values after meals"

    print("\n✅ Per-timestep IOB/COB test passed")


def test_feature_engineering():
    """Test extended feature engineering."""
    from ml.feature_engineering import engineer_extended_features

    print("\n" + "="*60)
    print("TEST 3: Extended Feature Engineering")
    print("="*60)

    glucose_df, treatments_df = generate_sample_data(n_days=1, include_gaps=False)

    # Convert glucose to expected format
    df = glucose_df.copy()
    df['carbs'] = 0.0
    df['protein'] = 0.0
    df['fat'] = 0.0
    df['insulin'] = 0.0

    # Merge treatments
    for _, treat in treatments_df.iterrows():
        closest_idx = (df['timestamp'] - treat['timestamp']).abs().idxmin()
        if treat.get('carbs', 0) > 0:
            df.loc[closest_idx, 'carbs'] += treat.get('carbs', 0)
            df.loc[closest_idx, 'protein'] += treat.get('protein', 0)
            df.loc[closest_idx, 'fat'] += treat.get('fat', 0)
        if treat.get('insulin', 0) > 0:
            df.loc[closest_idx, 'insulin'] += treat.get('insulin', 0)

    # Engineer features with per-timestep IOB/COB
    df_extended = engineer_extended_features(
        df,
        treatments_df=treatments_df,
        isf=50.0,
        icr=10.0,
        compute_per_timestep=True
    )

    print(f"\nDataFrame shape: {df_extended.shape}")
    print(f"Features created: {len(df_extended.columns)}")

    # Check key features exist
    key_features = ['ml_iob', 'ml_cob', 'ml_iob_effect', 'net_effect', 'rate_of_change']
    for feat in key_features:
        assert feat in df_extended.columns, f"Missing feature: {feat}"
        print(f"  {feat}: min={df_extended[feat].min():.2f}, max={df_extended[feat].max():.2f}")

    # Verify ml_iob varies (not static)
    unique_iob = df_extended['ml_iob'].nunique()
    print(f"\nUnique ml_iob values: {unique_iob}")
    assert unique_iob > 10, "ml_iob should vary across timesteps, not be static"

    print("\n✅ Extended feature engineering test passed")


def test_temporal_splitter():
    """Test temporal data splitting."""
    from ml.training.tft_trainer import TemporalDataSplitter

    print("\n" + "="*60)
    print("TEST 4: Temporal Data Splitting")
    print("="*60)

    splitter = TemporalDataSplitter(val_split=0.15, test_split=0.15)

    n_samples = 1000
    train_idx, val_idx, test_idx = splitter.split(n_samples)

    print(f"\nTotal samples: {n_samples}")
    print(f"Train: {len(train_idx)} ({len(train_idx)/n_samples:.1%})")
    print(f"Val: {len(val_idx)} ({len(val_idx)/n_samples:.1%})")
    print(f"Test: {len(test_idx)} ({len(test_idx)/n_samples:.1%})")

    # Verify temporal ordering
    assert train_idx.max() < val_idx.min(), "Train should be before val"
    assert val_idx.max() < test_idx.min(), "Val should be before test"

    # Verify gap between splits
    gap = val_idx.min() - train_idx.max()
    print(f"Gap between train/val: {gap} samples")

    print("\n✅ Temporal splitting test passed")


def test_weighted_quantile_loss():
    """Test weighted quantile loss."""
    import torch
    from ml.training.tft_trainer import WeightedQuantileLoss

    print("\n" + "="*60)
    print("TEST 5: Weighted Quantile Loss")
    print("="*60)

    loss_fn = WeightedQuantileLoss(
        quantiles=[0.1, 0.5, 0.9],
        quantile_weights=[1.5, 1.0, 0.8]
    )

    # Create sample predictions and targets
    batch_size = 32
    n_horizons = 6
    n_quantiles = 3

    predictions = torch.randn(batch_size, n_horizons, n_quantiles) * 20 + 100
    targets = torch.randn(batch_size, n_horizons) * 20 + 100

    loss = loss_fn(predictions, targets)

    print(f"\nLoss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("\n✅ Weighted quantile loss test passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TFT TRAINING PIPELINE TESTS")
    print("="*60)

    tests = [
        ("Data Quality Filter", test_data_quality_filter),
        ("Per-Timestep IOB/COB", test_per_timestep_iob_cob),
        ("Feature Engineering", test_feature_engineering),
        ("Temporal Splitter", test_temporal_splitter),
        ("Weighted Quantile Loss", test_weighted_quantile_loss),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
