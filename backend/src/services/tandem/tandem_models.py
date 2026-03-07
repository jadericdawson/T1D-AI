"""
Pydantic models for raw Tandem pump data.
Maps to structures returned by tconnectsync.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
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


# ===================== Expanded Event Models =====================

class TandemDailyBasalEvent(BaseModel):
    """Daily basal summary with battery/IOB info (LidDailyBasal)."""
    timestamp: datetime
    battery_percent: Optional[float] = Field(default=None, description="Battery % (computed from MSB/LSB)")
    battery_millivolts: Optional[int] = Field(default=None, description="Battery voltage in mV")
    daily_basal_units: Optional[float] = Field(default=None, description="Total basal delivered today")
    daily_bolus_units: Optional[float] = Field(default=None, description="Total bolus delivered today")
    daily_total_insulin: Optional[float] = Field(default=None, description="Total insulin today")
    pump_iob: Optional[float] = Field(default=None, description="Pump-calculated IOB")
    event_id: str = ""


class TandemModeChangeEvent(BaseModel):
    """Control-IQ mode change (LidAaUserModeChange)."""
    timestamp: datetime
    previous_mode: str = Field(description="Normal, Sleeping, Exercising, EatingSoon")
    current_mode: str = Field(description="Normal, Sleeping, Exercising, EatingSoon")
    event_id: str = ""


class TandemPcmChangeEvent(BaseModel):
    """Pump control mode change (LidAaPcmChange)."""
    timestamp: datetime
    previous_pcm: str = Field(description="NoControl, OpenLoop, Pining, ClosedLoop")
    current_pcm: str = Field(description="NoControl, OpenLoop, Pining, ClosedLoop")
    event_id: str = ""


class TandemSuspendEvent(BaseModel):
    """Pump suspend or resume (LidPumpingSuspended / LidPumpingResumed)."""
    timestamp: datetime
    action: str = Field(description="'suspended' or 'resumed'")
    reason: Optional[str] = Field(default=None, description="User, Alarm, Malfunction, PLGS, etc.")
    event_id: str = ""


class TandemAlertEvent(BaseModel):
    """Pump alert (LidAlertActivated)."""
    timestamp: datetime
    alert_type: str = Field(description="Alert name/code")
    alert_id: Optional[int] = Field(default=None)
    event_id: str = ""


class TandemAlarmEvent(BaseModel):
    """Pump alarm (LidAlarmActivated)."""
    timestamp: datetime
    alarm_type: str = Field(description="Alarm name/code")
    alarm_id: Optional[int] = Field(default=None)
    event_id: str = ""


class TandemCartridgeEvent(BaseModel):
    """Cartridge fill (LidCartridgeFilled)."""
    timestamp: datetime
    volume: Optional[float] = Field(default=None, description="Cartridge volume in units")
    event_id: str = ""


class TandemSiteChangeEvent(BaseModel):
    """Infusion site change (LidCannulaFilled)."""
    timestamp: datetime
    prime_volume: Optional[float] = Field(default=None, description="Cannula prime volume")
    event_id: str = ""


class TandemTubingEvent(BaseModel):
    """Tubing fill (LidTubingFilled)."""
    timestamp: datetime
    volume: Optional[float] = Field(default=None, description="Tubing fill volume")
    event_id: str = ""


class TandemBgReadingEvent(BaseModel):
    """Manual BG reading from pump (LidBgReadingTaken)."""
    timestamp: datetime
    bg_value: int = Field(description="BG reading in mg/dL")
    event_id: str = ""


class TandemBolusDetailEvent(BaseModel):
    """Detailed bolus info from Msg2+Msg3 (LidBolusRequestedMsg2/Msg3)."""
    timestamp: datetime
    bolus_id: int
    food_insulin: Optional[float] = Field(default=None, description="Food portion of bolus")
    correction_insulin: Optional[float] = Field(default=None, description="Correction portion")
    isf: Optional[float] = Field(default=None, description="ISF used by pump calculator")
    target_bg: Optional[float] = Field(default=None, description="Target BG used by pump")
    current_bg: Optional[float] = Field(default=None, description="BG at time of bolus calc")
    icr: Optional[float] = Field(default=None, description="ICR used by pump calculator")
    event_id: str = ""


class TandemDailyStatusEvent(BaseModel):
    """Daily algo status (LidAaDailyStatus)."""
    timestamp: datetime
    auto_corrections_today: Optional[int] = Field(default=None)
    sensor_type: Optional[str] = Field(default=None)
    event_id: str = ""


@dataclass
class TandemFetchResult:
    """Complete result from expanded Tandem data extraction."""
    basal_events: List[TandemBasalEvent] = field(default_factory=list)
    bolus_events: List[TandemBolusEvent] = field(default_factory=list)
    daily_basal_events: List[TandemDailyBasalEvent] = field(default_factory=list)
    mode_changes: List[TandemModeChangeEvent] = field(default_factory=list)
    pcm_changes: List[TandemPcmChangeEvent] = field(default_factory=list)
    suspend_events: List[TandemSuspendEvent] = field(default_factory=list)
    alerts: List[TandemAlertEvent] = field(default_factory=list)
    alarms: List[TandemAlarmEvent] = field(default_factory=list)
    cartridge_events: List[TandemCartridgeEvent] = field(default_factory=list)
    site_changes: List[TandemSiteChangeEvent] = field(default_factory=list)
    tubing_events: List[TandemTubingEvent] = field(default_factory=list)
    bg_readings: List[TandemBgReadingEvent] = field(default_factory=list)
    bolus_details: List[TandemBolusDetailEvent] = field(default_factory=list)
    daily_status: List[TandemDailyStatusEvent] = field(default_factory=list)
