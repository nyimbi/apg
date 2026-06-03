"""
Pydantic v2 models for APG Fleet Management (transport_fle).

Entities: Vehicle, Driver, VehicleAssignment, Trip, FuelRecord,
Maintenance, Inspection, Incident, InsurancePolicy, Registration,
TachographRecord, COFInspection, Telematics.

All IDs are uuid7 strings.  All timestamps are UTC ISO-8601.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


def _non_empty(v: str) -> str:
	assert v and v.strip(), "Value must be non-empty"
	return v.strip()


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]


# ──────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────

class VehicleType(str, Enum):
	RIGID_TRUCK = "rigid_truck"
	ARTICULATED_TRUCK = "articulated_truck"
	VAN = "van"
	PICKUP = "pickup"
	TRACTOR_UNIT = "tractor_unit"
	TRAILER = "trailer"
	TANKER = "tanker"
	REFRIGERATED = "refrigerated_vehicle"
	FLATBED = "flatbed"
	TIPPER = "tipper"
	MINIBUS = "minibus"
	MOTORCYCLE = "motorcycle"
	ELECTRIC_VEHICLE = "electric_vehicle"
	BUS = "bus"
	CRANE_TRUCK = "crane_truck"


class VehicleStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	IN_MAINTENANCE = "in_maintenance"
	OUT_OF_SERVICE = "out_of_service"
	DISPOSED = "disposed"
	ON_HIRE = "on_hire"
	AWAITING_INSPECTION = "awaiting_inspection"
	BREAKDOWN = "breakdown"
	IMPOUNDED = "impounded"


class FuelType(str, Enum):
	DIESEL = "diesel"
	PETROL = "petrol"
	CNG = "cng"
	LNG = "lng"
	ELECTRIC = "electric"
	HYBRID = "hybrid"
	HYDROGEN = "hydrogen"
	BIODIESEL = "biodiesel"
	HVO = "hvo"


class OwnershipType(str, Enum):
	OWNED = "owned"
	LEASED = "leased"
	HIRED = "hired"
	CONTRACT_HIRE = "contract_hire"
	FINANCE_LEASE = "finance_lease"
	HIRE_PURCHASE = "hire_purchase"


class DriverStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	ON_LEAVE = "on_leave"
	SUSPENDED = "suspended"
	TRAINING = "training"
	PROBATION = "probation"
	TERMINATED = "terminated"


class LicenceClass(str, Enum):
	AM = "am"
	A1 = "a1"
	A2 = "a2"
	A = "a"
	B = "b"
	BE = "be"
	C1 = "c1"
	C1E = "c1e"
	C = "c"
	CE = "ce"
	D1 = "d1"
	D1E = "d1e"
	D = "d"
	DE = "de"


class TripStatus(str, Enum):
	PLANNED = "planned"
	DISPATCHED = "dispatched"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	BREAKDOWN = "breakdown"
	DELAYED = "delayed"


class MaintenanceType(str, Enum):
	SCHEDULED = "scheduled"
	CORRECTIVE = "corrective"
	PREDICTIVE = "predictive"
	EMERGENCY = "emergency"


class MaintenanceStatus(str, Enum):
	SCHEDULED = "scheduled"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	OVERDUE = "overdue"
	CANCELLED = "cancelled"
	DEFERRED = "deferred"


class InspectionType(str, Enum):
	PRE_TRIP = "pre_trip"
	POST_TRIP = "post_trip"
	PERIODIC = "periodic"
	COF = "cof"
	ROADSIDE = "roadside"
	ANNUAL = "annual"


class InspectionResult(str, Enum):
	PASS = "pass"
	FAIL = "fail"
	ADVISORY = "advisory"
	CONDITIONAL_PASS = "conditional_pass"


class IncidentSeverity(str, Enum):
	MINOR = "minor"
	MODERATE = "moderate"
	MAJOR = "major"
	CRITICAL = "critical"
	FATAL = "fatal"


class IncidentStatus(str, Enum):
	REPORTED = "reported"
	UNDER_INVESTIGATION = "under_investigation"
	RESOLVED = "resolved"
	CLOSED = "closed"
	DISPUTED = "disputed"


class TachographMode(str, Enum):
	DRIVING = "driving"
	REST = "rest"
	OTHER_WORK = "other_work"
	AVAILABILITY = "availability"
	UNKNOWN = "unknown"


class ComplianceStandard(str, Enum):
	DVLA = "dvla"
	C_TPAT = "c_tpat"
	EURO6 = "euro6"
	EURO5 = "euro5"
	ADR = "adr"
	GDPR_TELEMATICS = "gdpr_telematics"
	TACHOGRAPH = "tachograph_regulation"
	OPERATOR_LICENCE = "operator_licence"
	COF = "cof"
	AXLE_LOAD = "axle_load"


# ──────────────────────────────────────────────────────────────────
# Base model
# ──────────────────────────────────────────────────────────────────

class FleBase(BaseModel):
	"""Common audit fields on every FLE entity."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		populate_by_name=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(default="system")
	is_deleted: bool = Field(default=False)


# ──────────────────────────────────────────────────────────────────
# Vehicle
# ──────────────────────────────────────────────────────────────────

class VehicleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_type: VehicleType
	registration: NonEmptyStr
	vin: NonEmptyStr
	make: NonEmptyStr
	model: NonEmptyStr
	year: Annotated[int, AfterValidator(lambda v: v)] = Field(ge=1980, le=2100)
	fuel_type: FuelType
	ownership_type: OwnershipType
	gross_vehicle_weight_kg: Decimal = Field(default=Decimal("0"), ge=0)
	payload_capacity_kg: Decimal = Field(default=Decimal("0"), ge=0)
	axle_count: int = Field(default=2, ge=1, le=12)
	odometer_km: Decimal = Field(default=Decimal("0"), ge=0)
	colour: str = Field(default="")
	depot_id: str | None = None
	notes: str = Field(default="")


class VehicleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: VehicleStatus | None = None
	odometer_km: Decimal | None = None
	depot_id: str | None = None
	notes: str | None = None
	colour: str | None = None


class VehicleResponse(FleBase):
	vehicle_type: VehicleType
	registration: str
	vin: str
	make: str
	model: str
	year: int
	fuel_type: FuelType
	ownership_type: OwnershipType
	status: VehicleStatus = VehicleStatus.ACTIVE
	gross_vehicle_weight_kg: Decimal = Decimal("0")
	payload_capacity_kg: Decimal = Decimal("0")
	axle_count: int = 2
	odometer_km: Decimal = Decimal("0")
	colour: str = ""
	depot_id: str | None = None
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────

class DriverCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	name: NonEmptyStr
	employee_number: str = Field(default="")
	licence_number: NonEmptyStr
	licence_class: LicenceClass
	licence_expiry: datetime
	tacho_card_number: str = Field(default="")
	cpc_expiry: datetime | None = None
	medical_expiry: datetime | None = None
	phone: str = Field(default="")
	email: str = Field(default="")
	depot_id: str | None = None
	notes: str = Field(default="")


class DriverUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: DriverStatus | None = None
	licence_expiry: datetime | None = None
	cpc_expiry: datetime | None = None
	medical_expiry: datetime | None = None
	phone: str | None = None
	email: str | None = None
	depot_id: str | None = None
	notes: str | None = None


class DriverResponse(FleBase):
	name: str
	employee_number: str = ""
	licence_number: str
	licence_class: LicenceClass
	licence_expiry: datetime
	status: DriverStatus = DriverStatus.ACTIVE
	tacho_card_number: str = ""
	cpc_expiry: datetime | None = None
	medical_expiry: datetime | None = None
	phone: str = ""
	email: str = ""
	depot_id: str | None = None
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# VehicleAssignment
# ──────────────────────────────────────────────────────────────────

class VehicleAssignmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	driver_id: NonEmptyStr
	assigned_at: datetime = Field(default_factory=datetime.utcnow)
	released_at: datetime | None = None
	assignment_reason: str = Field(default="")
	trip_id: str | None = None


class VehicleAssignmentResponse(FleBase):
	vehicle_id: str
	driver_id: str
	assigned_at: datetime
	released_at: datetime | None = None
	assignment_reason: str = ""
	trip_id: str | None = None
	is_active: bool = True


# ──────────────────────────────────────────────────────────────────
# Trip
# ──────────────────────────────────────────────────────────────────

class TripCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	driver_id: NonEmptyStr
	origin: NonEmptyStr
	destination: NonEmptyStr
	origin_lat: float | None = None
	origin_lon: float | None = None
	dest_lat: float | None = None
	dest_lon: float | None = None
	planned_departure: datetime
	planned_arrival: datetime | None = None
	load_kg: Decimal = Field(default=Decimal("0"), ge=0)
	load_description: str = Field(default="")
	route_id: str | None = None
	customs_required: bool = False
	cross_border_countries: list[str] = Field(default_factory=list)
	notes: str = Field(default="")


class TripUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: TripStatus | None = None
	actual_departure: datetime | None = None
	actual_arrival: datetime | None = None
	driver_id: str | None = None  # driver change mid-trip
	odometer_start_km: Decimal | None = None
	odometer_end_km: Decimal | None = None
	fuel_consumed_l: Decimal | None = None
	delay_reason: str | None = None
	breakdown_at: datetime | None = None
	notes: str | None = None


class TripResponse(FleBase):
	vehicle_id: str
	driver_id: str
	origin: str
	destination: str
	origin_lat: float | None = None
	origin_lon: float | None = None
	dest_lat: float | None = None
	dest_lon: float | None = None
	planned_departure: datetime
	planned_arrival: datetime | None = None
	actual_departure: datetime | None = None
	actual_arrival: datetime | None = None
	status: TripStatus = TripStatus.PLANNED
	load_kg: Decimal = Decimal("0")
	load_description: str = ""
	odometer_start_km: Decimal | None = None
	odometer_end_km: Decimal | None = None
	fuel_consumed_l: Decimal | None = None
	distance_km: Decimal | None = None
	route_id: str | None = None
	customs_required: bool = False
	cross_border_countries: list[str] = Field(default_factory=list)
	delay_reason: str = ""
	breakdown_at: datetime | None = None
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# FuelRecord
# ──────────────────────────────────────────────────────────────────

class FuelRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	driver_id: str | None = None
	trip_id: str | None = None
	fuelled_at: datetime = Field(default_factory=datetime.utcnow)
	litres: Decimal = Field(ge=Decimal("0.1"))
	cost_per_litre: Decimal = Field(ge=Decimal("0"))
	currency: str = Field(default="KES")
	station_name: str = Field(default="")
	station_lat: float | None = None
	station_lon: float | None = None
	odometer_km: Decimal = Field(ge=0)
	full_tank: bool = True
	receipt_ref: str = Field(default="")
	notes: str = Field(default="")


class FuelRecordResponse(FleBase):
	vehicle_id: str
	driver_id: str | None = None
	trip_id: str | None = None
	fuelled_at: datetime
	litres: Decimal
	cost_per_litre: Decimal
	currency: str = "KES"
	total_cost: Decimal = Decimal("0")
	station_name: str = ""
	station_lat: float | None = None
	station_lon: float | None = None
	odometer_km: Decimal
	full_tank: bool = True
	receipt_ref: str = ""
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────────────────────────

class MaintenanceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	maintenance_type: MaintenanceType
	description: NonEmptyStr
	scheduled_date: datetime
	due_odometer_km: Decimal | None = None
	supplier_id: str | None = None
	estimated_cost: Decimal = Field(default=Decimal("0"), ge=0)
	currency: str = Field(default="KES")
	notes: str = Field(default="")


class MaintenanceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: MaintenanceStatus | None = None
	completed_date: datetime | None = None
	actual_cost: Decimal | None = None
	odometer_at_service_km: Decimal | None = None
	work_order_ref: str | None = None
	parts_replaced: list[str] | None = None
	next_service_date: datetime | None = None
	next_service_odometer_km: Decimal | None = None
	notes: str | None = None


class MaintenanceResponse(FleBase):
	vehicle_id: str
	maintenance_type: MaintenanceType
	description: str
	status: MaintenanceStatus = MaintenanceStatus.SCHEDULED
	scheduled_date: datetime
	completed_date: datetime | None = None
	due_odometer_km: Decimal | None = None
	odometer_at_service_km: Decimal | None = None
	estimated_cost: Decimal = Decimal("0")
	actual_cost: Decimal | None = None
	currency: str = "KES"
	supplier_id: str | None = None
	work_order_ref: str = ""
	parts_replaced: list[str] = Field(default_factory=list)
	next_service_date: datetime | None = None
	next_service_odometer_km: Decimal | None = None
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Inspection
# ──────────────────────────────────────────────────────────────────

class InspectionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	driver_id: str | None = None
	inspection_type: InspectionType
	inspected_at: datetime = Field(default_factory=datetime.utcnow)
	inspected_by: str = Field(default="")
	result: InspectionResult
	defects: list[str] = Field(default_factory=list)
	advisory_notes: list[str] = Field(default_factory=list)
	odometer_km: Decimal | None = None
	next_inspection_due: datetime | None = None
	certificate_ref: str = Field(default="")
	notes: str = Field(default="")


class InspectionResponse(FleBase):
	vehicle_id: str
	driver_id: str | None = None
	inspection_type: InspectionType
	inspected_at: datetime
	inspected_by: str = ""
	result: InspectionResult
	defects: list[str] = Field(default_factory=list)
	advisory_notes: list[str] = Field(default_factory=list)
	odometer_km: Decimal | None = None
	next_inspection_due: datetime | None = None
	certificate_ref: str = ""
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Incident
# ──────────────────────────────────────────────────────────────────

class IncidentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	driver_id: str | None = None
	trip_id: str | None = None
	occurred_at: datetime
	severity: IncidentSeverity
	description: NonEmptyStr
	location: str = Field(default="")
	lat: float | None = None
	lon: float | None = None
	injuries_count: int = Field(default=0, ge=0)
	fatalities_count: int = Field(default=0, ge=0)
	third_party_involved: bool = False
	police_ref: str = Field(default="")
	estimated_damage_cost: Decimal = Field(default=Decimal("0"), ge=0)
	currency: str = Field(default="KES")
	overloading_fine_allocated: Decimal = Field(default=Decimal("0"), ge=0)
	notes: str = Field(default="")


class IncidentResponse(FleBase):
	vehicle_id: str
	driver_id: str | None = None
	trip_id: str | None = None
	occurred_at: datetime
	severity: IncidentSeverity
	status: IncidentStatus = IncidentStatus.REPORTED
	description: str
	location: str = ""
	lat: float | None = None
	lon: float | None = None
	injuries_count: int = 0
	fatalities_count: int = 0
	third_party_involved: bool = False
	police_ref: str = ""
	estimated_damage_cost: Decimal = Decimal("0")
	actual_damage_cost: Decimal | None = None
	currency: str = "KES"
	overloading_fine_allocated: Decimal = Decimal("0")
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Insurance Policy
# ──────────────────────────────────────────────────────────────────

class InsurancePolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	policy_number: NonEmptyStr
	insurer: NonEmptyStr
	policy_type: str  # comprehensive, third_party, fleet_blanket
	cover_start: datetime
	cover_end: datetime
	premium: Decimal = Field(ge=0)
	currency: str = Field(default="KES")
	excess: Decimal = Field(default=Decimal("0"), ge=0)
	sum_insured: Decimal = Field(default=Decimal("0"), ge=0)
	notes: str = Field(default="")


class InsurancePolicyResponse(FleBase):
	vehicle_id: str
	policy_number: str
	insurer: str
	policy_type: str
	cover_start: datetime
	cover_end: datetime
	premium: Decimal
	currency: str = "KES"
	excess: Decimal = Decimal("0")
	sum_insured: Decimal = Decimal("0")
	is_active: bool = True
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────

class RegistrationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	registration_number: NonEmptyStr
	registration_authority: str = Field(default="NTSA")
	issued_at: datetime
	expires_at: datetime
	certificate_ref: str = Field(default="")
	road_worthiness_ref: str = Field(default="")
	notes: str = Field(default="")


class RegistrationResponse(FleBase):
	vehicle_id: str
	registration_number: str
	registration_authority: str = "NTSA"
	issued_at: datetime
	expires_at: datetime
	certificate_ref: str = ""
	road_worthiness_ref: str = ""
	is_current: bool = True
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Tachograph Record (EU / EAC tachograph regulation)
# ──────────────────────────────────────────────────────────────────

class TachographRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	driver_id: NonEmptyStr
	trip_id: str | None = None
	period_start: datetime
	period_end: datetime
	mode: TachographMode
	distance_km: Decimal = Field(default=Decimal("0"), ge=0)
	max_speed_kmh: float = 0.0
	avg_speed_kmh: float = 0.0
	driving_minutes: int = Field(default=0, ge=0)
	break_minutes: int = Field(default=0, ge=0)
	rest_minutes: int = Field(default=0, ge=0)
	infringement_code: str | None = None
	notes: str = Field(default="")


class TachographRecordResponse(FleBase):
	vehicle_id: str
	driver_id: str
	trip_id: str | None = None
	period_start: datetime
	period_end: datetime
	mode: TachographMode
	distance_km: Decimal = Decimal("0")
	max_speed_kmh: float = 0.0
	avg_speed_kmh: float = 0.0
	driving_minutes: int = 0
	break_minutes: int = 0
	rest_minutes: int = 0
	infringement_code: str | None = None
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# COF Inspection (Certificate of Fitness — East Africa / Kenya)
# ──────────────────────────────────────────────────────────────────

class COFInspectionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	inspected_at: datetime
	inspection_station: str = Field(default="")
	inspector_id: str = Field(default="")
	result: InspectionResult
	cof_number: str = Field(default="")
	issued_at: datetime | None = None
	expires_at: datetime | None = None
	defects_found: list[str] = Field(default_factory=list)
	rectification_deadline: datetime | None = None
	notes: str = Field(default="")


class COFInspectionResponse(FleBase):
	vehicle_id: str
	inspected_at: datetime
	inspection_station: str = ""
	inspector_id: str = ""
	result: InspectionResult
	cof_number: str = ""
	issued_at: datetime | None = None
	expires_at: datetime | None = None
	defects_found: list[str] = Field(default_factory=list)
	rectification_deadline: datetime | None = None
	notes: str = ""


# ──────────────────────────────────────────────────────────────────
# Telematics Event
# ──────────────────────────────────────────────────────────────────

class TelematicsEventCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	vehicle_id: NonEmptyStr
	driver_id: str | None = None
	trip_id: str | None = None
	provider: str = Field(default="custom")
	event_type: NonEmptyStr  # position, harsh_braking, speeding, geofence_enter, geofence_exit, idle, …
	occurred_at: datetime = Field(default_factory=datetime.utcnow)
	lat: float
	lon: float
	speed_kmh: float = 0.0
	heading_deg: float | None = None
	altitude_m: float | None = None
	odometer_km: Decimal | None = None
	engine_on: bool | None = None
	fuel_level_pct: float | None = None
	payload: dict[str, Any] = Field(default_factory=dict)


class TelematicsEventResponse(FleBase):
	vehicle_id: str
	driver_id: str | None = None
	trip_id: str | None = None
	provider: str = "custom"
	event_type: str
	occurred_at: datetime
	lat: float
	lon: float
	speed_kmh: float = 0.0
	heading_deg: float | None = None
	altitude_m: float | None = None
	odometer_km: Decimal | None = None
	engine_on: bool | None = None
	fuel_level_pct: float | None = None
	payload: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Report / Aggregation models
# ──────────────────────────────────────────────────────────────────

class TCOBreakdown(BaseModel):
	"""Total Cost of Ownership components for a vehicle over a period."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	vehicle_id: str
	tenant_id: str
	period_start: datetime
	period_end: datetime
	currency: str = "KES"
	fuel_cost: Decimal = Decimal("0")
	maintenance_cost: Decimal = Decimal("0")
	insurance_cost: Decimal = Decimal("0")
	registration_cost: Decimal = Decimal("0")
	depreciation: Decimal = Decimal("0")
	driver_cost: Decimal = Decimal("0")
	toll_cost: Decimal = Decimal("0")
	fine_cost: Decimal = Decimal("0")
	total_cost: Decimal = Decimal("0")
	distance_km: Decimal = Decimal("0")
	cost_per_km: Decimal = Decimal("0")
	utilisation_pct: float = 0.0


class DriverBehaviourScore(BaseModel):
	"""Driver behaviour aggregated score."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	driver_id: str
	tenant_id: str
	period_start: datetime
	period_end: datetime
	overall_score: float = Field(ge=0, le=100)
	speeding_score: float = Field(ge=0, le=100)
	harsh_braking_score: float = Field(ge=0, le=100)
	harsh_acceleration_score: float = Field(ge=0, le=100)
	cornering_score: float = Field(ge=0, le=100)
	idle_score: float = Field(ge=0, le=100)
	seatbelt_score: float = Field(ge=0, le=100)
	distraction_score: float = Field(ge=0, le=100)
	fatigue_score: float = Field(ge=0, le=100)
	incidents_count: int = 0
	trips_count: int = 0
	distance_km: Decimal = Decimal("0")
	grade: str = "C"  # A, B, C, D, F


class FleetUtilisationReport(BaseModel):
	"""Fleet-wide utilisation analytics."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	period_start: datetime
	period_end: datetime
	total_vehicles: int = 0
	active_vehicles: int = 0
	avg_utilisation_pct: float = 0.0
	total_distance_km: Decimal = Decimal("0")
	total_trips: int = 0
	total_fuel_l: Decimal = Decimal("0")
	avg_fuel_efficiency_l100km: float = 0.0
	vehicles_in_maintenance: int = 0
	vehicles_awaiting_inspection: int = 0
	overdue_maintenance_count: int = 0
	compliance_alert_count: int = 0
	top_performers: list[str] = Field(default_factory=list)
	underperformers: list[str] = Field(default_factory=list)


class ComplianceCalendarEntry(BaseModel):
	"""A single due-date compliance event for display in calendar."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	entity_id: str
	entity_type: str  # vehicle, driver
	tenant_id: str
	due_date: datetime
	event_type: str  # cof_renewal, insurance_renewal, licence_expiry, cpc_expiry, maintenance_due, …
	description: str
	days_until_due: int
	is_overdue: bool = False
	severity: str = "info"  # info, warning, critical


class PredictiveMaintenanceAlert(BaseModel):
	"""Predictive maintenance signal."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	vehicle_id: str
	tenant_id: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	component: str
	predicted_failure_date: datetime | None = None
	confidence_pct: float = Field(ge=0, le=100)
	recommended_action: str
	urgency: str  # low, medium, high, critical
	supporting_signals: list[str] = Field(default_factory=list)


class DashboardKPIs(BaseModel):
	"""Fleet management dashboard KPIs."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime = Field(default_factory=datetime.utcnow)
	total_vehicles: int = 0
	active_vehicles: int = 0
	vehicles_on_trip: int = 0
	vehicles_in_maintenance: int = 0
	total_drivers: int = 0
	active_drivers: int = 0
	drivers_on_duty: int = 0
	trips_today: int = 0
	trips_in_progress: int = 0
	fuel_spend_mtd: Decimal = Decimal("0")
	maintenance_spend_mtd: Decimal = Decimal("0")
	fleet_utilisation_pct: float = 0.0
	compliance_alerts: int = 0
	overdue_maintenance: int = 0
	active_incidents: int = 0
	avg_driver_score: float = 0.0
