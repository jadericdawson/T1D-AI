"""
Bolus Mapper - Maps Tandem bolus events to T1D-AI Treatment format.

Mapping:
  Standard bolus    -> type='insulin',          deliveryMethod='pump_bolus'
  Extended bolus    -> type='insulin',          deliveryMethod='pump_bolus' (with duration)
  Combo bolus       -> type='insulin',          deliveryMethod='pump_bolus' (with duration)
  Auto correction   -> type='auto_correction',  deliveryMethod='pump_auto_correction'
  Carbs from calc   -> type='carbs'             (separate treatment)
"""
import logging
from typing import List

from services.tandem.tandem_models import TandemBolusEvent
from services.tandem.t1dai_models import T1DAITreatment

logger = logging.getLogger(__name__)

# Mapping of Tandem bolus types to T1D-AI treatment types and delivery methods
_BOLUS_TYPE_MAP = {
    "standard":        ("insulin",          "pump_bolus"),
    "extended":        ("insulin",          "pump_bolus"),
    "combo":           ("insulin",          "pump_bolus"),
    "auto_correction": ("auto_correction",  "pump_auto_correction"),
}


def map_bolus_events(
    events: List[TandemBolusEvent],
    user_id: str
) -> List[T1DAITreatment]:
    """
    Map Tandem bolus events to T1D-AI treatments.

    Each bolus becomes an insulin treatment.
    If carbs were entered in the bolus calculator, a separate carb treatment is also created.
    """
    treatments = []

    for event in events:
        if event.completion_status not in ("completed",) and event.insulin <= 0:
            logger.debug(f"Skipping {event.completion_status} bolus {event.event_id}")
            continue

        treatment_type, delivery_method = _BOLUS_TYPE_MAP.get(
            event.bolus_type,
            ("insulin", "pump_bolus")
        )

        notes_parts = []
        if event.bolus_type == "extended":
            notes_parts.append(f"Extended {event.duration_seconds // 60}min")
        elif event.bolus_type == "combo":
            notes_parts.append(f"Combo bolus ({event.duration_seconds // 60}min extended)")
        elif event.bolus_type == "auto_correction":
            notes_parts.append("Control-IQ auto correction")

        if event.requested_insulin and event.requested_insulin != event.insulin:
            notes_parts.append(f"Requested {event.requested_insulin}U, delivered {event.insulin}U")

        if event.completion_status != "completed":
            notes_parts.append(f"Status: {event.completion_status}")

        insulin_treatment = T1DAITreatment(
            id=f"{user_id}_tandem_{event.event_id}",
            userId=user_id,
            timestamp=event.timestamp.isoformat(),
            type=treatment_type,
            insulin=event.insulin,
            source="tandem",
            sourceId=f"tandem_{event.event_id}",
            bolusType=event.bolus_type,
            deliveryMethod=delivery_method,
            pumpSource="tandem_mobi",
            durationMinutes=event.duration_seconds // 60 if event.duration_seconds > 0 else None,
            notes=", ".join(notes_parts) if notes_parts else None,
        )
        treatments.append(insulin_treatment)

        if event.carbs and event.carbs > 0:
            carb_treatment = T1DAITreatment(
                id=f"{user_id}_tandem_{event.event_id}_carbs",
                userId=user_id,
                timestamp=event.timestamp.isoformat(),
                type="carbs",
                carbs=event.carbs,
                source="tandem",
                sourceId=f"tandem_{event.event_id}_carbs",
                pumpSource="tandem_mobi",
                notes="Carbs from pump bolus calculator",
            )
            treatments.append(carb_treatment)

    insulin_count = sum(1 for t in treatments if t.type != "carbs")
    carb_count = sum(1 for t in treatments if t.type == "carbs")
    logger.info(f"Mapped {len(events)} bolus events -> {insulin_count} insulin + {carb_count} carb treatments")

    return treatments
