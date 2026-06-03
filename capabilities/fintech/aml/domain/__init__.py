"""Domain logic for APG Anti-Money Laundering.

Exports:
- calculations: risk scoring, structuring/velocity/round-trip/layering detection
- rules: assertion functions and RuleViolation exception
- adapters: auth/audit/notify/workflow adapter interfaces
"""
from .adapters import (
	AuthAdapter,
	NullAuthAdapter,
	AuditAdapter,
	NullAuditAdapter,
	NotifyAdapter,
	NullNotifyAdapter,
	WorkflowAdapter,
	NullWorkflowAdapter,
	get_auth_adapter,
	get_audit_adapter,
	get_notify_adapter,
	get_workflow_adapter,
)
from .rules import RuleViolation, evaluate_rules
from .calculations import (
	calculate_risk_score,
	severity_from_score,
	risk_segment_from_score,
	detect_structuring,
	detect_velocity_anomaly,
	detect_round_trip,
	detect_layering,
	calculate_network_risk_score,
	requires_ctr,
	calculate_sar_priority,
	calculate_false_positive_rate,
)

__all__ = [
	# adapters
	"AuthAdapter", "NullAuthAdapter",
	"AuditAdapter", "NullAuditAdapter",
	"NotifyAdapter", "NullNotifyAdapter",
	"WorkflowAdapter", "NullWorkflowAdapter",
	"get_auth_adapter", "get_audit_adapter",
	"get_notify_adapter", "get_workflow_adapter",
	# rules
	"RuleViolation", "evaluate_rules",
	# calculations
	"calculate_risk_score", "severity_from_score", "risk_segment_from_score",
	"detect_structuring", "detect_velocity_anomaly",
	"detect_round_trip", "detect_layering",
	"calculate_network_risk_score",
	"requires_ctr", "calculate_sar_priority", "calculate_false_positive_rate",
]
