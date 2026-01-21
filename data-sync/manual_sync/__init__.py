"""
Manual Sync HTTP Trigger
Allows triggering a sync for a specific user via HTTP POST.
"""
import asyncio
import json
import logging
import os

import azure.functions as func

# Import from gluroo_sync module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gluroo_sync import sync_user, COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE
from azure.cosmos import CosmosClient

logger = logging.getLogger(__name__)


async def run_manual_sync(user_id: str) -> dict:
    """Run sync for a specific user."""
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        return {"error": "Missing Cosmos DB configuration"}

    cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    database = cosmos_client.get_database_client(COSMOS_DATABASE)
    datasource_container = database.get_container_client("datasources")

    # Get user's datasource
    query = f"SELECT * FROM c WHERE c.userId = '{user_id}' AND c.sourceType = 'gluroo'"
    datasources = list(datasource_container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))

    if not datasources:
        return {"error": f"No Gluroo datasource found for user {user_id}"}

    result = await sync_user(cosmos_client, user_id, datasources[0])
    return result


def main(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for manual sync."""
    logging.info('Manual sync function triggered')

    # Get user_id from request body or query params
    user_id = req.params.get('user_id')
    if not user_id:
        try:
            req_body = req.get_json()
            user_id = req_body.get('user_id')
        except ValueError:
            pass

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "user_id is required"}),
            mimetype="application/json",
            status_code=400
        )

    # Run sync
    result = asyncio.run(run_manual_sync(user_id))

    status_code = 200 if result.get("success", True) else 500

    return func.HttpResponse(
        json.dumps(result),
        mimetype="application/json",
        status_code=status_code
    )
