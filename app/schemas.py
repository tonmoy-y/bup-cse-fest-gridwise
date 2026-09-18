from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


class HourEntry(BaseModel):
    hour: int = Field(..., ge=0, le=23, strict=True)
    demand_kwh: float = Field(..., ge=0, strict=True)
    solar_kwh: float = Field(..., ge=0, strict=True)
    tariff_bdt_per_kwh: float = Field(..., ge=0, strict=True)


class BatterySpec(BaseModel):
    capacity_kwh: float = Field(..., gt=0, strict=True)
    initial_energy_kwh: float = Field(..., ge=0, strict=True)
    minimum_energy_kwh: float = Field(..., ge=0, strict=True)
    max_charge_kwh_per_hour: float = Field(..., ge=0, strict=True)
    max_discharge_kwh_per_hour: float = Field(..., ge=0, strict=True)

    @model_validator(mode="after")
    def validate_consistency(self) -> "BatterySpec":
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        return self


class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourEntry]
    battery: BatterySpec

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, hours: List[HourEntry]) -> List[HourEntry]:
        if len(hours) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        seen = set()
        for h in hours:
            if h.hour < 0 or h.hour > 23:
                raise ValueError("hour must be between 0 and 23")
            if h.hour in seen:
                raise ValueError("duplicate hour in hours array")
            seen.add(h.hour)
        if seen != set(range(24)):
            raise ValueError("hours must cover every hour 0..23 exactly once")
        return hours

    @field_validator("operator_notes")
    @classmethod
    def validate_notes(cls, notes: List[str]) -> List[str]:
        if not (1 <= len(notes) <= 3):
            raise ValueError("operator_notes must contain 1 to 3 entries")
        for n in notes:
            if not isinstance(n, str) or not n.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return notes


class SolarReductionAdjustment(BaseModel):
    hours: List[int]
    factor: float


class MinimumBatteryReserveAdjustment(BaseModel):
    hours: List[int]
    minimum_energy_kwh: float


class NoChargeWindowAdjustment(BaseModel):
    hours: List[int]


class NoDischargeWindowAdjustment(BaseModel):
    hours: List[int]


class MaxGridWindowAdjustment(BaseModel):
    hours: List[int]
    max_grid_kwh: float


StructuredAdjustment = Optional[
    Union[
        SolarReductionAdjustment,
        MinimumBatteryReserveAdjustment,
        NoChargeWindowAdjustment,
        NoDischargeWindowAdjustment,
        MaxGridWindowAdjustment,
    ]
]


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict]
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
