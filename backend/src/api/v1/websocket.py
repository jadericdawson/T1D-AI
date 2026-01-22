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
from services.dexcom_service import DexcomShareService
from models.schemas import GlucoseReading, TrendDirection
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


def get_data_user_id(profile_id: str) -> str:
    """
    Convert a profile ID to the actual data user ID.

    For 'self' profiles, the profile ID is 'profile_{user_id}' but data
    is stored with just the raw user_id. This function strips the prefix.
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix (8 chars)
    return profile_id


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

# Dexcom service for direct CGM data (lazy init)
_dexcom_service: DexcomShareService | None = None

def get_dexcom_service() -> DexcomShareService | None:
    """Get or create Dexcom service (lazy init)."""
    global _dexcom_service
    if _dexcom_service is None:
        try:
            _dexcom_service = DexcomShareService()
            logger.info("WebSocket: Initialized Dexcom Share service")
        except Exception as e:
            logger.warning(f"WebSocket: Failed to initialize Dexcom service: {e}")
    return _dexcom_service

def dexcom_to_glucose_reading(dexcom_reading, user_id: str) -> GlucoseReading:
    """Convert Dexcom reading to GlucoseReading schema."""
    trend_map = {
        "rising quickly": TrendDirection.DOUBLE_UP,
        "rising": TrendDirection.SINGLE_UP,
        "rising slightly": TrendDirection.FORTY_FIVE_UP,
        "steady": TrendDirection.FLAT,
        "falling slightly": TrendDirection.FORTY_FIVE_DOWN,
        "falling": TrendDirection.SINGLE_DOWN,
        "falling quickly": TrendDirection.DOUBLE_DOWN,
    }
    trend = trend_map.get(dexcom_reading.trend_description.lower(), TrendDirection.FLAT)

    return GlucoseReading(
        id=f"dexcom-{int(dexcom_reading.timestamp.timestamp() * 1000)}",
        userId=user_id,
        value=dexcom_reading.value,
        timestamp=dexcom_reading.timestamp,
        trend=trend,
        source="dexcom"
    )


async def get_glucose_update(user_id: str) -> dict:
    """
    Get current glucose data with predictions for WebSocket broadcast.
    Tries Dexcom first for freshest data, falls back to database.
    """
    try:
        # Normalize profile ID to data user ID
        # For 'self' profiles, strip the 'profile_' prefix (data stored without it)
        data_user_id = get_data_user_id(user_id)
        logger.debug(f"WebSocket: Normalized user_id {user_id} -> data_user_id {data_user_id}")

        # Try Dexcom first for freshest data
        latest: GlucoseReading | None = None
        dexcom_svc = get_dexcom_service()

        if dexcom_svc:
            # Check if user has CGM data (dexcom OR gluroo source)
            # Dexcom is PRIMARY source, Gluroo is backup
            try:
                recent_dexcom = await glucose_repo.get_recent_by_source(data_user_id, "dexcom", hours=48)
                recent_gluroo = await glucose_repo.get_recent_by_source(data_user_id, "gluroo", hours=48)
                user_has_cgm = len(recent_dexcom) > 0 or len(recent_gluroo) > 0

                if user_has_cgm:
                    try:
                        dexcom_reading = await dexcom_svc.get_latest_reading_async()
                        if dexcom_reading:
                            latest = dexcom_to_glucose_reading(dexcom_reading, data_user_id)
                            logger.info(f"WebSocket: Got fresh reading from Dexcom (PRIMARY): {latest.value} mg/dL")
                    except Exception as e:
                        logger.warning(f"WebSocket: Dexcom fetch failed, will use Gluroo backup: {e}")
                else:
                    logger.debug(f"WebSocket: User {user_id} has no CGM data, skipping Dexcom fetch")
            except Exception as e:
                logger.warning(f"WebSocket: Error checking CGM status for user: {e}")

        # Fall back to database if Dexcom unavailable
        db_latest = await glucose_repo.get_latest(data_user_id)

        if db_latest:
            if not latest:
                latest = db_latest
                logger.info(f"WebSocket: Using database data: {latest.value} mg/dL")
            elif db_latest.timestamp > latest.timestamp:
                latest = db_latest
                logger.info(f"WebSocket: Database data is fresher, using: {latest.value} mg/dL")

        if not latest:
            return {"type": "error", "message": "No glucose data"}

        # Get recent treatments
        treatments = await treatment_repo.get_recent(data_user_id, hours=6)

        # Get predictions first to get ISF
        settings = get_settings()
        models_dir = None
        for path in [Path("./models"), Path("./data/models"), Path("/app/models")]:
            if path.exists():
                models_dir = path
                break

        pred_service = get_prediction_service(models_dir, settings.model_device)

        # Get glucose history for LSTM
        start_time = datetime.utcnow() - timedelta(minutes=120)
        glucose_history = await glucose_repo.get_history(data_user_id, start_time)

        # Convert trend
        trend_val = 0
        if latest.trend:
            trend_map = {
                "DoubleDown": -3, "SingleDown": -2, "FortyFiveDown": -1,
                "Flat": 0, "FortyFiveUp": 1, "SingleUp": 2, "DoubleUp": 3
            }
            trend_val = trend_map.get(str(latest.trend), 0)

        # Calculate IOB/COB first for predictions
        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments)

        # Generate predictions
        prediction = pred_service.predict(
            current_bg=float(latest.value),
            trend=trend_val,
            iob=iob,
            cob=cob,
            glucose_history=[r.model_dump() for r in glucose_history],
            treatments=[t.model_dump() for t in treatments]
        )

        # Get PIR from settings (default 25g protein per unit)
        pir = getattr(settings, 'default_pir', 25.0)

        # Calculate complete metrics using get_current_metrics (includes POB, dose, protein dose)
        metrics = iob_cob_service.get_current_metrics(
            current_bg=latest.value,
            treatments=treatments,
            isf=prediction.isf,
            pir=pir
        )

        # Build response
        return {
            "type": "glucose_update",
            "data": {
                "timestamp": latest.timestamp.isoformat(),
                "value": latest.value,
                "trend": str(latest.trend) if latest.trend else "Flat",
                "trendArrow": get_trend_arrow(trend_val),
                "source": latest.source if hasattr(latest, 'source') and latest.source else "gluroo",
                "predictions": {
                    "linear": prediction.linear,
                    "lstm": prediction.lstm,
                    "horizons": [5, 10, 15]
                },
                "metrics": {
                    "iob": round(metrics.iob, 2),
                    "cob": round(metrics.cob, 1),
                    "pob": round(metrics.pob, 1),
                    "isf": round(metrics.isf, 1),
                    "recommendedDose": round(metrics.recommendedDose, 2),
                    "proteinDoseNow": round(metrics.proteinDoseNow, 2),
                    "proteinDoseLater": round(metrics.proteinDoseLater, 2),
                    "effectiveBg": metrics.effectiveBg,
                    # Food recommendation fields
                    "actionType": metrics.actionType,
                    "recommendedCarbs": round(metrics.recommendedCarbs, 0),
                    "foodSuggestions": [
                        {
                            "name": f.name,
                            "carbs": f.carbs,
                            "typical_portion": f.typical_portion,
                            "glycemic_index": f.glycemic_index,
                            "times_eaten": f.times_eaten
                        }
                        for f in metrics.foodSuggestions
                    ],
                    "predictedBgWithoutAction": metrics.predictedBgWithoutAction,
                    "predictedBgWithAction": metrics.predictedBgWithAction,
                    "recommendationReasoning": metrics.recommendationReasoning,
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
