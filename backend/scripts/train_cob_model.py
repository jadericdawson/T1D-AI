"""
Train Personalized COB Model using enriched treatment data.

Uses GPT-4.1 enriched fat/protein/GI data to learn individual carb
absorption patterns based on food composition.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_enriched_treatments():
    """Load enriched carb treatments from CosmosDB."""
    endpoint = os.environ.get('COSMOS_ENDPOINT')
    key = os.environ.get('COSMOS_KEY')

    client = CosmosClient(endpoint, key)
    database = client.get_database_client('T1D-AI-DB')
    container = database.get_container_client("treatments")

    query = """
        SELECT c.id, c.userId, c.timestamp, c.carbs, c.protein, c.fat,
               c.glycemicIndex, c.absorptionRate, c.notes
        FROM c
        WHERE IS_DEFINED(c.enrichedAt) AND c.enrichedAt != null
          AND c.carbs > 0
        ORDER BY c.timestamp DESC
    """

    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    logger.info(f"Loaded {len(items)} enriched treatments")
    return items


def load_glucose_readings():
    """Load glucose readings from CosmosDB."""
    endpoint = os.environ.get('COSMOS_ENDPOINT')
    key = os.environ.get('COSMOS_KEY')

    client = CosmosClient(endpoint, key)
    database = client.get_database_client('T1D-AI-DB')
    container = database.get_container_client("glucose_readings")

    # Load last 30 days of readings
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    query = f"""
        SELECT c.userId, c.timestamp, c.value, c.trend
        FROM c
        WHERE c.timestamp > '{cutoff}'
        ORDER BY c.timestamp
    """

    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    logger.info(f"Loaded {len(items)} glucose readings")
    return items


def load_insulin_treatments():
    """Load insulin treatments for IOB calculation."""
    endpoint = os.environ.get('COSMOS_ENDPOINT')
    key = os.environ.get('COSMOS_KEY')

    client = CosmosClient(endpoint, key)
    database = client.get_database_client('T1D-AI-DB')
    container = database.get_container_client("treatments")

    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    query = f"""
        SELECT c.userId, c.timestamp, c.insulin
        FROM c
        WHERE c.timestamp > '{cutoff}'
          AND c.insulin > 0
        ORDER BY c.timestamp
    """

    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    logger.info(f"Loaded {len(items)} insulin treatments")
    return items


def calculate_iob_at_time(insulin_treatments, at_time, user_id, half_life_min=54):
    """Calculate IOB at a specific time."""
    iob = 0.0
    duration_min = 240

    for t in insulin_treatments:
        if t.get('userId') != user_id:
            continue

        t_time = datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
        elapsed_min = (at_time - t_time).total_seconds() / 60

        if 0 <= elapsed_min < duration_min:
            insulin = t.get('insulin', 0)
            remaining = insulin * (0.5 ** (elapsed_min / half_life_min))
            iob += remaining

    return iob


def get_bg_at_time(glucose_readings, target_time, user_id, tolerance_min=10):
    """Get BG reading closest to target time."""
    best_reading = None
    best_diff = float('inf')

    for r in glucose_readings:
        if r.get('userId') != user_id:
            continue

        r_time = datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
        diff = abs((r_time - target_time).total_seconds() / 60)

        if diff < tolerance_min and diff < best_diff:
            best_diff = diff
            best_reading = r

    return best_reading


def create_training_data(treatments, glucose_readings, insulin_treatments, isf=50.0, icr=10.0):
    """
    Create training samples from enriched treatments and BG responses.

    For each carb treatment:
    1. Get BG at time of meal (t=0)
    2. Get BG at t+30, t+60, t+90, t+120 min
    3. Calculate absorbed carbs from BG rise (minus insulin effect)
    4. Create training samples with features and remaining fraction label
    """
    samples = []
    bg_per_gram = isf / icr  # BG rise per gram of carbs

    for treatment in treatments:
        carbs = treatment.get('carbs', 0)
        if carbs <= 0:
            continue

        user_id = treatment.get('userId')
        t_time = datetime.fromisoformat(treatment['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)

        # Get food composition from enrichment
        protein = treatment.get('protein', 0) or 0
        fat = treatment.get('fat', 0) or 0
        gi = treatment.get('glycemicIndex', 55) or 55
        absorption_rate = treatment.get('absorptionRate', 'medium')

        # Get BG at meal time
        bg_at_meal = get_bg_at_time(glucose_readings, t_time, user_id)
        if not bg_at_meal:
            continue

        bg_start = bg_at_meal.get('value', 120)
        meal_hour = t_time.hour

        # Create samples at different time points
        for minutes_after in [30, 45, 60, 90, 120, 150, 180]:
            check_time = t_time + timedelta(minutes=minutes_after)

            # Get BG at check time
            bg_check = get_bg_at_time(glucose_readings, check_time, user_id)
            if not bg_check:
                continue

            bg_value = bg_check.get('value', bg_start)

            # Calculate IOB effect (how much insulin has lowered BG)
            iob_at_meal = calculate_iob_at_time(insulin_treatments, t_time, user_id)
            iob_at_check = calculate_iob_at_time(insulin_treatments, check_time, user_id)
            insulin_effect = (iob_at_meal - iob_at_check) * isf  # BG drop from insulin

            # Calculate actual BG change
            actual_bg_change = bg_value - bg_start

            # Adjust for insulin effect to isolate carb impact
            carb_induced_rise = actual_bg_change + insulin_effect

            # Calculate expected rise if all carbs absorbed
            expected_rise = carbs * bg_per_gram

            # Estimate absorbed fraction
            if expected_rise > 0:
                absorbed_fraction = carb_induced_rise / expected_rise
                absorbed_fraction = max(0, min(1.5, absorbed_fraction))  # Allow some overshoot
                remaining_fraction = 1.0 - absorbed_fraction
                remaining_fraction = max(0, min(1, remaining_fraction))
            else:
                remaining_fraction = 0.5  # Default

            # Build feature vector (matching COB model input)
            hour_sin = np.sin(2 * np.pi * meal_hour / 24)
            hour_cos = np.cos(2 * np.pi * meal_hour / 24)

            features = np.array([
                carbs / 100.0,
                protein / 50.0,
                fat / 50.0,
                gi / 100.0,
                minutes_after / 360.0,
                hour_sin,
                hour_cos
            ], dtype=np.float32)

            target = np.array([remaining_fraction], dtype=np.float32)

            samples.append({
                'features': features,
                'target': target,
                'carbs': carbs,
                'minutes': minutes_after,
                'remaining_fraction': remaining_fraction,
                'absorption_rate': absorption_rate,
                'fat': fat,
                'gi': gi
            })

    logger.info(f"Created {len(samples)} training samples")
    return samples


def train_cob_model(samples, epochs=100, lr=0.001, batch_size=32):
    """Train the COB model."""
    from ml.models.cob_model import PersonalizedCOBModel

    if len(samples) < 30:
        logger.warning(f"Only {len(samples)} samples - need at least 30 for training")
        return None

    # Prepare data
    X = np.array([s['features'] for s in samples])
    y = np.array([s['target'] for s in samples])

    # Split train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model (7 features for basic model)
    model = PersonalizedCOBModel(input_size=7, hidden_sizes=[64, 32], dropout=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None

    logger.info(f"Training COB model with {len(X_train)} training samples...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    return model, best_val_loss


def evaluate_model(model, samples):
    """Evaluate model performance by absorption rate."""
    results = {'fast': [], 'medium': [], 'slow': []}

    model.eval()
    with torch.no_grad():
        for s in samples:
            features = torch.tensor([s['features']], dtype=torch.float32)
            predicted = model(features).item()
            actual = s['target'][0]
            error = abs(predicted - actual)

            rate = s.get('absorption_rate', 'medium')
            if rate in results:
                results[rate].append(error)

    logger.info("\nPerformance by absorption rate:")
    for rate, errors in results.items():
        if errors:
            mae = np.mean(errors)
            logger.info(f"  {rate}: MAE = {mae:.3f} (n={len(errors)})")

    return results


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting COB Model Training")
    logger.info("=" * 60)

    # Load data
    treatments = load_enriched_treatments()
    glucose_readings = load_glucose_readings()
    insulin_treatments = load_insulin_treatments()

    if len(treatments) < 10:
        logger.error("Not enough enriched treatments. Run enrichment first.")
        return

    # Create training data
    samples = create_training_data(treatments, glucose_readings, insulin_treatments)

    if len(samples) < 30:
        logger.error(f"Only {len(samples)} samples created. Need more data.")
        return

    # Train model
    model, val_loss = train_cob_model(samples)

    if model is None:
        logger.error("Training failed")
        return

    # Evaluate
    evaluate_model(model, samples)

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'personalized_cob_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': 7,
        'hidden_sizes': [64, 32],
        'val_loss': val_loss,
        'num_samples': len(samples),
        'trained_at': datetime.utcnow().isoformat()
    }, model_path)

    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Training samples: {len(samples)}")
    logger.info(f"Validation loss: {val_loss:.4f}")

    # Log to MLflow if available
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5002")
        mlflow.set_experiment("T1D-AI/COB-Personalized")

        with mlflow.start_run(run_name=f"cob_v1_{len(treatments)}_samples"):
            mlflow.log_param("num_enriched_treatments", len(treatments))
            mlflow.log_param("num_training_samples", len(samples))
            mlflow.log_param("input_size", 7)
            mlflow.log_param("hidden_sizes", "[64, 32]")
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_artifact(model_path)

        logger.info("Logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
