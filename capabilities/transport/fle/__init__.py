"""
APG Fleet Management Capability (transport_fle) v2.0.0

Provides end-to-end fleet lifecycle management:
vehicles, drivers, trips, fuel, maintenance, inspections,
incidents, compliance, tachograph, COF, telematics, TCO.

Standalone:
    from capabilities.transport.fle import FleetService, VehicleCreate
    from capabilities.transport.fle.api import fle_bp
    from capabilities.transport.fle.views import fle_views_bp

APG platform:
    from capabilities.transport.fle import register_capability
    register_capability(app)
"""

from __future__ import annotations

__version__ = "2.0.0"
CAPABILITY_ID = "transport_fle"
CAPABILITY_NAME = "Fleet Management"

from .models import (
	COFInspectionCreate, COFInspectionResponse,
	ComplianceCalendarEntry, DashboardKPIs, DriverBehaviourScore,
	DriverCreate, DriverResponse, DriverStatus, DriverUpdate,
	FleetUtilisationReport, FuelRecordCreate, FuelRecordResponse,
	IncidentCreate, IncidentResponse, IncidentSeverity, IncidentStatus,
	InspectionCreate, InspectionResponse, InspectionResult, InspectionType,
	InsurancePolicyCreate, InsurancePolicyResponse,
	MaintenanceCreate, MaintenanceResponse, MaintenanceStatus, MaintenanceType,
	PredictiveMaintenanceAlert, RegistrationCreate, RegistrationResponse,
	TCOBreakdown, TachographRecordCreate, TachographRecordResponse,
	TelematicsEventCreate, TelematicsEventResponse,
	TripCreate, TripResponse, TripStatus, TripUpdate,
	VehicleAssignmentCreate, VehicleAssignmentResponse,
	VehicleCreate, VehicleResponse, VehicleStatus, VehicleUpdate,
	VehicleType, FuelType, OwnershipType, LicenceClass, TachographMode,
	uuid7str,
)
from .service import FleetService
from .api import fle_bp
from .views import fle_views_bp


def register_capability(app, appbuilder=None) -> dict:
	"""Register Fleet Management blueprints with a Flask app."""
	app.register_blueprint(fle_bp)
	app.register_blueprint(fle_views_bp)
	result = {
		"capability_id": CAPABILITY_ID,
		"version": __version__,
		"api_prefix": fle_bp.url_prefix,
		"views_prefix": fle_views_bp.url_prefix,
	}
	app.logger.info("[FLE] registered — %s", result)
	return result


__all__ = [
	"CAPABILITY_ID", "CAPABILITY_NAME", "__version__",
	"FleetService",
	"VehicleCreate", "VehicleUpdate", "VehicleResponse", "VehicleStatus", "VehicleType",
	"DriverCreate", "DriverUpdate", "DriverResponse", "DriverStatus", "LicenceClass",
	"VehicleAssignmentCreate", "VehicleAssignmentResponse",
	"TripCreate", "TripUpdate", "TripResponse", "TripStatus",
	"FuelRecordCreate", "FuelRecordResponse",
	"MaintenanceCreate", "MaintenanceResponse", "MaintenanceStatus", "MaintenanceType",
	"InspectionCreate", "InspectionResponse", "InspectionResult", "InspectionType",
	"IncidentCreate", "IncidentResponse", "IncidentSeverity", "IncidentStatus",
	"InsurancePolicyCreate", "InsurancePolicyResponse",
	"RegistrationCreate", "RegistrationResponse",
	"TachographRecordCreate", "TachographRecordResponse", "TachographMode",
	"COFInspectionCreate", "COFInspectionResponse",
	"TelematicsEventCreate", "TelematicsEventResponse",
	"TCOBreakdown", "DriverBehaviourScore", "FleetUtilisationReport",
	"ComplianceCalendarEntry", "PredictiveMaintenanceAlert", "DashboardKPIs",
	"FuelType", "OwnershipType", "uuid7str",
	"fle_bp", "fle_views_bp", "register_capability",
]
