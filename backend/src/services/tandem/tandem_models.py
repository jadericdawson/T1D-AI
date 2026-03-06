"""
Pydantic models for raw Tandem pump data.
Maps to structures returned by tconnectsync.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TandemBasalEvent(BaseModel):
    """A single basal delivery event from the pump (typically every 5 min)."""
    timestamp: datetime
    duration_seconds: int = Field(default=300, description="Duration of this basal segment")
    rate: float = Field(description="Basal rate in U/hr at this time")
    delivered_units: float = Field(description="Actual insulin delivered in this segment")
    delivery_type: str = Field(description="'profile', 'temp', 'controliq_adjustment'")
    event_id: Optional[str] = Field(default=None, description="Unique event ID from pump")


class TandemBolusEvent(BaseModel):
    """A bolus event from the pump."""
    timestamp: datetime
    insulin: float = Field(description="Bolus amount in units")
    bolus_type: str = Field(description="'standard', 'extended', 'combo', 'auto_correction'")
    duration_seconds: int = Field(default=0, description="Duration for extended boluses")
    carbs: Optional[float] = Field(default=None, description="Carbs entered in bolus calculator")
    event_id: str = Field(description="Unique event ID from pump")
    completion_status: str = Field(default="completed", description="'completed', 'cancelled', 'interrupted'")
    requested_insulin: Optional[float] = Field(default=None, description="Originally requested amount")


class TandemPumpEvent(BaseModel):
    """General pump event (suspend, resume, alert)."""
    timestamp: datetime
    event_type: str = Field(description="'suspend', 'resume', 'alert', 'cartridge_change', 'site_change'")
    event_id: str
    details: Optional[str] = Field(default=None)
