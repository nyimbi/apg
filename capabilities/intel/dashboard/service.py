"""Executable service layer for APG Intelligence Dashboard."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DASHBOARD_TYPES, SUPPORTED_FILTER_TYPES, SUPPORTED_METRIC_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SHARE_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_VIEW_TYPES, SUPPORTED_WIDGET_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .dashboard_runtime import bounded_score, normalize_code, positive_int, present
	from .models import DashboardAgent, DashboardAuthority, DashboardBoard, DashboardDataSource, DashboardFilter, DashboardMetric, DashboardReview, DashboardShare, DashboardView, DashboardWidget, DashboardWorkspace
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DASHBOARD_TYPES, SUPPORTED_FILTER_TYPES, SUPPORTED_METRIC_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SHARE_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_VIEW_TYPES, SUPPORTED_WIDGET_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from dashboard_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import DashboardAgent, DashboardAuthority, DashboardBoard, DashboardDataSource, DashboardFilter, DashboardMetric, DashboardReview, DashboardShare, DashboardView, DashboardWidget, DashboardWorkspace  # type: ignore


class IntelligenceDashboardService:
	"""Tenant-scoped dashboard runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], DashboardAuthority] = {}
		self.workspaces: dict[tuple[str, str], DashboardWorkspace] = {}
		self.dashboards: dict[tuple[str, str], DashboardBoard] = {}
		self.sources: dict[tuple[str, str], DashboardDataSource] = {}
		self.metrics: dict[tuple[str, str], DashboardMetric] = {}
		self.widgets: dict[tuple[str, str], DashboardWidget] = {}
		self.filters: dict[tuple[str, str], DashboardFilter] = {}
		self.views: dict[tuple[str, str], DashboardView] = {}
		self.shares: dict[tuple[str, str], DashboardShare] = {}
		self.reviews: dict[tuple[str, str], DashboardReview] = {}
		self.agents: dict[tuple[str, str], DashboardAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = DashboardAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "dashboard_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = DashboardWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "dashboard_workspace_recorded", workspace_id)
		return item.to_dict()

	def record_dashboard(self, dashboard_id: str, tenant_id: str, workspace_id: str, dashboard_type: str, title: str, owner_id: str, classification: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		dashboard_type = normalize_code(dashboard_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dashboard", "workspace_present": workspace is not None, "dashboard_type_supported": dashboard_type in SUPPORTED_DASHBOARD_TYPES, "title_present": present(title), "owner_present": present(owner_id), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "evidence_present": present(evidence_reference)})
		item = DashboardBoard(dashboard_id, tenant_id, workspace_id, dashboard_type, title, owner_id, classification, evidence_reference)
		self.dashboards[self._tenant_key(tenant_id, dashboard_id)] = item
		self._audit(tenant_id, "dashboard_recorded", dashboard_id)
		return item.to_dict()

	def record_source(self, source_id: str, tenant_id: str, dashboard_id: str, source_type: str, source_reference: str, custodian_id: str, evidence_reference: str) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_source", "dashboard_present": dashboard is not None, "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "source_reference_present": present(source_reference), "custodian_present": present(custodian_id), "evidence_present": present(evidence_reference)})
		item = DashboardDataSource(source_id, tenant_id, dashboard_id, source_type, source_reference, custodian_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "dashboard_source_recorded", source_id)
		return item.to_dict()

	def record_metric(self, metric_id: str, tenant_id: str, source_id: str, metric_type: str, metric_reference: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		metric_type = normalize_code(metric_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_metric", "source_present": source is not None, "metric_type_supported": metric_type in SUPPORTED_METRIC_TYPES, "metric_reference_present": present(metric_reference), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = DashboardMetric(metric_id, tenant_id, source_id, metric_type, metric_reference, float(confidence_score), evidence_reference)
		self.metrics[self._tenant_key(tenant_id, metric_id)] = item
		self._audit(tenant_id, "dashboard_metric_recorded", metric_id)
		return item.to_dict()

	def record_widget(self, widget_id: str, tenant_id: str, dashboard_id: str, widget_type: str, widget_reference: str, metric_id: str, evidence_reference: str) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		metric = self._tenant_metric_or_none(metric_id, tenant_id)
		widget_type = normalize_code(widget_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_widget", "dashboard_present": dashboard is not None, "metric_present": metric is not None, "widget_type_supported": widget_type in SUPPORTED_WIDGET_TYPES, "widget_reference_present": present(widget_reference), "evidence_present": present(evidence_reference)})
		item = DashboardWidget(widget_id, tenant_id, dashboard_id, widget_type, widget_reference, metric_id, evidence_reference)
		self.widgets[self._tenant_key(tenant_id, widget_id)] = item
		self._audit(tenant_id, "dashboard_widget_recorded", widget_id)
		return item.to_dict()

	def record_filter(self, filter_id: str, tenant_id: str, dashboard_id: str, filter_type: str, filter_reference: str, evidence_reference: str) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		filter_type = normalize_code(filter_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_filter", "dashboard_present": dashboard is not None, "filter_type_supported": filter_type in SUPPORTED_FILTER_TYPES, "filter_reference_present": present(filter_reference), "evidence_present": present(evidence_reference)})
		item = DashboardFilter(filter_id, tenant_id, dashboard_id, filter_type, filter_reference, evidence_reference)
		self.filters[self._tenant_key(tenant_id, filter_id)] = item
		self._audit(tenant_id, "dashboard_filter_recorded", filter_id)
		return item.to_dict()

	def record_view(self, view_id: str, tenant_id: str, dashboard_id: str, view_type: str, view_reference: str, viewer_role: str, evidence_reference: str) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		view_type = normalize_code(view_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_view", "dashboard_present": dashboard is not None, "view_type_supported": view_type in SUPPORTED_VIEW_TYPES, "view_reference_present": present(view_reference), "viewer_role_present": present(viewer_role), "evidence_present": present(evidence_reference)})
		item = DashboardView(view_id, tenant_id, dashboard_id, view_type, view_reference, viewer_role, evidence_reference)
		self.views[self._tenant_key(tenant_id, view_id)] = item
		self._audit(tenant_id, "dashboard_view_recorded", view_id)
		return item.to_dict()

	def record_share(self, share_id: str, tenant_id: str, dashboard_id: str, share_type: str, recipient_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		share_type = normalize_code(share_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_share", "dashboard_present": dashboard is not None, "share_type_supported": share_type in SUPPORTED_SHARE_TYPES, "recipient_present": present(recipient_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = DashboardShare(share_id, tenant_id, dashboard_id, share_type, recipient_reference, approval_reference, evidence_reference)
		self.shares[self._tenant_key(tenant_id, share_id)] = item
		self._audit(tenant_id, "dashboard_share_recorded", share_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = DashboardReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "dashboard_review_recorded", reference_id)
		return item.to_dict()

	def register_dashboard_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_dashboard_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES, "agent_name_present": present(name), "agent_scope_present": present(scope)})
		item = DashboardAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "dashboard_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, uncited_metric_scope: bool = False, classification_leak_scope: bool = False, source_tampering_scope: bool = False, privacy_bypass_scope: bool = False, autonomous_share_scope: bool = False, unapproved_public_view_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "dashboard_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "uncited_metric_scope": uncited_metric_scope, "classification_leak_scope": classification_leak_scope, "source_tampering_scope": source_tampering_scope, "privacy_bypass_scope": privacy_bypass_scope, "autonomous_share_scope": autonomous_share_scope, "unapproved_public_view_scope": unapproved_public_view_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "dashboard_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.dashboard.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "dashboard_count": self._count(self.dashboards, tenant_id), "source_count": self._count(self.sources, tenant_id), "metric_count": self._count(self.metrics, tenant_id), "widget_count": self._count(self.widgets, tenant_id), "filter_count": self._count(self.filters, tenant_id), "view_count": self._count(self.views, tenant_id), "share_count": self._count(self.shares, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> DashboardAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> DashboardWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_dashboard_or_none(self, item_id: str, tenant_id: str) -> DashboardBoard | None:
		return self.dashboards.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> DashboardDataSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_metric_or_none(self, item_id: str, tenant_id: str) -> DashboardMetric | None:
		return self.metrics.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "dashboard_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "dashboard_policy_denied")


IntelDashboardService = IntelligenceDashboardService

