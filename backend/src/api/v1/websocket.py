"""
WebSocket API for Real-time Glucose Updates
Provides real-time streaming of glucose data and predictions.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

from database.repositories import GlucoseRepository, TreatmentRepository
from services.iob_cob_service import IOBCOBService
from services.prediction_service import get_prediction_service
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for real-time glucose updates.

    Supports multiple connections per user and broadcasts updates
    to all connected clients for a given user.
    """

    def __init__(self):
        # user_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Track last update time per user
        self.last_update: Dict[str, datetime] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"WebSocket connected for user {user_id}. Total connections: {len(self.active_connections[user_id])}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                if user_id in self.last_update:
                    del self.last_update[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast_to_user(self, user_id: str, message: dict):
        """Broadcast a message to all connections for a user."""
        if user_id not in self.active_connections:
            return

        disconnected = set()
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.add(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            self.active_connections[user_id].discard(ws)

    def get_connection_count(self, user_id: str) -> int:
        """Get number of active connections for a user."""
        return len(self.active_connections.get(user_id, set()))


# Global connection manager
manager = ConnectionManager()

# Service instances
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
iob_cob_service = IOBCOBService.from_settings()


async def get_glucose_update(user_id: str) -> dict:
    """
    Get current glucose data with predictions for WebSocket broadcast.
    """
    try:
        # Get latest glucose reading
        latest = await glucose_repo.get_latest(user_id)
        if not latest:
            return {"type": "error", "message": "No glucose data"}

        # Get recent treatments
        treatments = await treatment_repo.get_recent(user_id, hours=6)

        # Calculate IOB/COB
        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments)

        # Get predictions
        settings = get_settings()
        models_dir = None
        for path in [Path("./models"), Path("./data/models"), Path("/app/models")]:
            if path.exists():
                models_dir = path
                break

        pred_service = get_prediction_service(models_dir, settings.model_device)

        # Get glucose history for LSTM
        start_time = datetime.utcnow() - timedelta(minutes=120)
        glucose_history = await glucose_repo.get_history(user_id, start_time)

        # Convert trend
        trend_val = 0
        if latest.trend:
            trend_map = {
                "DoubleDown": -3, "SingleDown": -2, "FortyFiveDown": -1,
                "Flat": 0, "FortyFiveUp": 1, "SingleUp": 2, "DoubleUp": 3
            }
            trend_val = trend_map.get(str(latest.trend), 0)

        # Generate predictions
        prediction = pred_service.predict(
            current_bg=float(latest.value),
            trend=trend_val,
            iob=iob,
            glucose_history=[r.model_dump() for r in glucose_history],
            treatments=[t.model_dump() for t in treatments]
        )

        # Build response
        return {
            "type": "glucose_update",
            "data": {
                "timestamp": latest.timestamp.isoformat(),
                "value": latest.value,
                "trend": str(latest.trend) if latest.trend else "Flat",
                "trendArrow": get_trend_arrow(trend_val),
                "predictions": {
                    "linear": prediction.linear,
                    "lstm": prediction.lstm,
                    "horizons": [5, 10, 15]
                },
                "metrics": {
                    "iob": round(iob, 2),
                    "cob": round(cob, 1),
                    "isf": round(prediction.isf, 1)
                },
                "modelAvailable": pred_service.lstm_available
            },
            "serverTime": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting glucose update: {e}")
        return {"type": "error", "message": str(e)}


def get_trend_arrow(trend: int) -> str:
    """Convert trend int to arrow symbol."""
    arrows = {
        -3: "⇊", -2: "↓", -1: "↘",
        0: "→",
        1: "↗", 2: "↑", 3: "⇈"
    }
    return arrows.get(trend, "→")


@router.websocket("/ws/glucose/{user_id}")
async def glucose_websocket(
    websocket: WebSocket,
    user_id: str,
    interval: int = Query(default=60, ge=10, le=300)
):
    """
    WebSocket endpoint for real-time glucose updates.

    Connects and streams glucose data at the specified interval (seconds).
    Default is 60 seconds (matching CGM update frequency).

    Messages sent:
    - glucose_update: Current glucose with predictions
    - ping: Keep-alive message
    - error: Error message

    Client can send:
    - {"type": "ping"}: Keep-alive
    - {"type": "refresh"}: Request immediate update
    """
    await manager.connect(websocket, user_id)

    try:
        # Send initial data
        initial_data = await get_glucose_update(user_id)
        await manager.send_personal_message(initial_data, websocket)

        # Start update loop
        last_update = datetime.utcnow()

        while True:
            try:
                # Wait for message or timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=interval
                )

                # Parse message
                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "")

                    if msg_type == "ping":
                        await manager.send_personal_message(
                            {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                            websocket
                        )
                    elif msg_type == "refresh":
                        update = await get_glucose_update(user_id)
                        await manager.send_personal_message(update, websocket)
                        last_update = datetime.utcnow()

                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Timeout - send scheduled update
                now = datetime.utcnow()
                if (now - last_update).total_seconds() >= interval:
                    update = await get_glucose_update(user_id)
                    await manager.send_personal_message(update, websocket)
                    last_update = now

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)


@router.websocket("/ws/glucose/stream/{user_id}")
async def glucose_stream(
    websocket: WebSocket,
    user_id: str
):
    """
    Continuous glucose stream WebSocket.

    Streams glucose data as fast as updates are available.
    Useful for dashboards that need immediate updates.
    """
    await manager.connect(websocket, user_id)

    try:
        # Send initial data
        initial_data = await get_glucose_update(user_id)
        await manager.send_personal_message(initial_data, websocket)

        last_reading_id = None

        while True:
            # Check for new readings every 5 seconds
            await asyncio.sleep(5)

            try:
                latest = await glucose_repo.get_latest(user_id)
                if latest and latest.id != last_reading_id:
                    # New reading available
                    update = await get_glucose_update(user_id)
                    await manager.send_personal_message(update, websocket)
                    last_reading_id = latest.id

            except Exception as e:
                logger.error(f"Stream error: {e}")
                await manager.send_personal_message(
                    {"type": "error", "message": "Update failed"},
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"Stream error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)


@router.get("/ws/status")
async def websocket_status():
    """
    Get WebSocket connection status.

    Returns the number of active connections and users.
    """
    total_connections = sum(
        len(conns) for conns in manager.active_connections.values()
    )

    return {
        "activeUsers": len(manager.active_connections),
        "totalConnections": total_connections,
        "status": "healthy"
    }
