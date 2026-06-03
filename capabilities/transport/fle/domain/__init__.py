"""Domain logic for APG Fleet Management (transport_fle)."""

from .adapters import (
	AuthAdapter, NullAuthAdapter,
	AuditAdapter, NullAuditAdapter,
	NotifyAdapter, NullNotifyAdapter,
	WorkflowAdapter, NullWorkflowAdapter,
	get_auth_adapter, get_audit_adapter, get_notify_adapter, get_workflow_adapter,
)
from .rules import RuleViolation
from .calculations import (
	calculate_fuel_cost, calculate_fuel_efficiency_l100km, calculate_tco,
	calculate_cost_per_km, calculate_driver_score, calculate_depreciation_straight_line,
	days_until, compliance_severity, predict_oil_change_due,
)

__all__ = [
	"AuthAdapter", "NullAuthAdapter",
	"AuditAdapter", "NullAuditAdapter",
	"NotifyAdapter", "NullNotifyAdapter",
	"WorkflowAdapter", "NullWorkflowAdapter",
	"get_auth_adapter", "get_audit_adapter", "get_notify_adapter", "get_workflow_adapter",
	"RuleViolation",
	"calculate_fuel_cost", "calculate_fuel_efficiency_l100km", "calculate_tco",
	"calculate_cost_per_km", "calculate_driver_score", "calculate_depreciation_straight_line",
	"days_until", "compliance_severity", "predict_oil_change_due",
]
