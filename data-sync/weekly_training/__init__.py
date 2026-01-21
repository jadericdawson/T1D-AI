"""
Weekly Training Azure Function
Timer-triggered function that trains personalized models for all users.
Runs every Sunday at 3 AM UTC.
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List

import azure.functions as func
from azure.cosmos import CosmosClient

# Add backend src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'src'))

logger = logging.getLogger(__name__)

# Environment variables
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("COSMOS_KEY")
COSMOS_DATABASE = os.environ.get("COSMOS_DATABASE", "T1D-AI-DB")


async def get_users_to_train(cosmos_client: CosmosClient) -> List[str]:
    """Get all users with connected data sources who need training."""
    database = cosmos_client.get_database_client(COSMOS_DATABASE)
    datasource_container = database.get_container_client("datasources")

    # Query users with connected Gluroo
    query = """
    SELECT DISTINCT c.userId
    FROM c
    WHERE c.sourceType = 'gluroo' AND c.status = 'connected'
    """
    results = list(datasource_container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))

    user_ids = [r['userId'] for r in results]
    logger.info(f"Found {len(user_ids)} users for training")
    return user_ids


async def train_user_with_timeout(user_id: str, timeout_minutes: int = 30) -> dict:
    """Train a single user with timeout."""
    try:
        # Import here to avoid cold start issues
        from ml.training.trainer import UserModelTrainer

        trainer = UserModelTrainer(device="cpu", min_days=7)

        # Train with timeout
        result = await asyncio.wait_for(
            trainer.train_user_model(user_id, days=30, learn_isf=True),
            timeout=timeout_minutes * 60
        )

        await trainer.close()
        return result

    except asyncio.TimeoutError:
        logger.error(f"Training timed out for user {user_id}")
        return {
            "success": False,
            "user_id": user_id,
            "error": f"Training timed out after {timeout_minutes} minutes"
        }
    except Exception as e:
        logger.error(f"Training failed for user {user_id}: {e}")
        return {
            "success": False,
            "user_id": user_id,
            "error": str(e)
        }


async def send_training_report(results: List[dict]) -> None:
    """Send training report via email."""
    try:
        from azure.communication.email import EmailClient

        connection_string = os.environ.get("COMMUNICATION_SERVICES_CONNECTION_STRING")
        admin_email = os.environ.get("ADMIN_EMAIL")

        if not connection_string or not admin_email:
            logger.warning("Email not configured, skipping report")
            return

        # Create summary
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        body = f"""
T1D-AI Weekly Training Report
=============================
Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

Summary:
- Total users: {len(results)}
- Successful: {len(successful)}
- Failed: {len(failed)}

Successful Training:
"""
        for r in successful:
            metrics = r.get("metrics", {})
            body += f"- {r['user_id']}: MAE={metrics.get('mae', 'N/A'):.1f} mg/dL\n"

        if failed:
            body += "\nFailed Training:\n"
            for r in failed:
                body += f"- {r['user_id']}: {r.get('error', 'Unknown error')}\n"

        # Send email
        email_client = EmailClient.from_connection_string(connection_string)

        message = {
            "content": {
                "subject": f"T1D-AI Training Report: {len(successful)}/{len(results)} Successful",
                "plainText": body,
            },
            "recipients": {
                "to": [{"address": admin_email}]
            },
            "senderAddress": os.environ.get("EMAIL_SENDER_ADDRESS")
        }

        poller = email_client.begin_send(message)
        poller.result()
        logger.info("Training report sent successfully")

    except Exception as e:
        logger.error(f"Failed to send training report: {e}")


async def run_weekly_training():
    """Main training function."""
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        logger.error("Missing Cosmos DB configuration")
        return {"error": "Missing configuration"}

    # Get users to train
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    user_ids = await get_users_to_train(cosmos_client)

    if not user_ids:
        logger.info("No users to train")
        return {"total": 0, "successful": 0, "failed": 0}

    # Train each user
    results = []
    for user_id in user_ids:
        logger.info(f"Training model for user {user_id}")
        result = await train_user_with_timeout(user_id, timeout_minutes=30)
        results.append(result)

        # Log progress
        completed = len(results)
        logger.info(f"Progress: {completed}/{len(user_ids)} users")

    # Send report
    await send_training_report(results)

    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"Weekly training complete: {successful}/{len(results)} successful")

    return {
        "total": len(results),
        "successful": successful,
        "failed": len(results) - successful,
        "results": results
    }


def main(mytimer: func.TimerRequest) -> None:
    """Azure Function entry point - Timer triggered weekly."""
    utc_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    if mytimer.past_due:
        logger.info('Timer is past due!')

    logger.info(f'Weekly training function started at {utc_timestamp}')

    # Run async training
    result = asyncio.run(run_weekly_training())

    logger.info(f'Weekly training function completed: {result}')
