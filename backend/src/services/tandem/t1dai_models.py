"""
T1D-AI Treatment schema models for Tandem sync.
Matches the Treatment model in backend/src/models/schemas.py.

Timestamps MUST be stored in UTC (matching Gluroo convention) because
CosmosDB uses lexicographic string comparison for timestamp queries.
"""
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class T1DAITreatment(BaseModel):
    """Treatment document matching T1D-AI CosmosDB schema."""
    id: str = Field(description="Document ID: {userId}_tandem_{event_id}")
    userId: str = Field(description="Partition key")
    timestamp: str = Field(description="ISO 8601 timestamp")
    type: str = Field(description="'insulin', 'carbs', 'basal', 'auto_correction'")
    insulin: Optional[float] = Field(default=None, description="Insulin units")
    carbs: Optional[float] = Field(default=None, description="Carbs in grams")
    notes: Optional[str] = Field(default=None)
    source: str = Field(default="tandem", description="Data source identifier")
    sourceId: str = Field(description="Source-unique ID: tandem_{event_id}")

    # Pump-specific fields
    basalRate: Optional[float] = Field(default=None, description="Basal rate U/hr")
    bolusType: Optional[str] = Field(default=None, description="standard|extended|combo|auto_correction")
    deliveryMethod: Optional[str] = Field(default=None, description="pump_basal|pump_bolus|pump_auto_correction")
    pumpSource: Optional[str] = Field(default="tandem_mobi")

    # Duration for basal windows and extended boluses
    durationMinutes: Optional[int] = Field(default=None, description="Duration in minutes")

    @field_validator("timestamp", mode="before")
    @classmethod
    def normalize_timestamp_to_utc(cls, v):
        """Ensure timestamp is stored in UTC ISO format for CosmosDB compatibility."""
        if isinstance(v, datetime):
            utc_dt = v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
            return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                utc_dt = dt.astimezone(timezone.utc)
                return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            except ValueError:
                pass
        return v

    def to_cosmos_dict(self) -> dict:
        """Convert to dict for CosmosDB upsert, excluding None values."""
        data = self.model_dump(mode="json")
        return {k: v for k, v in data.items() if v is not None}
