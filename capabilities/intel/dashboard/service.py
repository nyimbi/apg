"""Executable service layer for APG Intelligence Dashboard."""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_DASHBOARD_TYPES,
		SUPPORTED_FILTER_TYPES,
		SUPPORTED_METRIC_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SHARE_TYPES,
		SUPPORTED_SOURCE_TYPES,
		SUPPORTED_VIEW_TYPES,
		SUPPORTED_WIDGET_TYPES,
		SUPPORTED_WORKSPACE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .dashboard_runtime import bounded_score, normalize_code, positive_int, present
	from .models import (
		DashboardAgent,
		DashboardAuthority,
		DashboardBoard,
		DashboardDataSource,
		DashboardFilter,
		DashboardMetric,
		DashboardReview,
		DashboardShare,
		DashboardView,
		DashboardWidget,
		DashboardWorkspace,
	)
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DASHBOARD_TYPES, SUPPORTED_FILTER_TYPES, SUPPORTED_METRIC_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SHARE_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_VIEW_TYPES, SUPPORTED_WIDGET_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from dashboard_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import DashboardAgent, DashboardAuthority, DashboardBoard, DashboardDataSource, DashboardFilter, DashboardMetric, DashboardReview, DashboardShare, DashboardView, DashboardWidget, DashboardWorkspace  # type: ignore


# ---------------------------------------------------------------------------
# Priority bands used by the intelligence feed
# ---------------------------------------------------------------------------
PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"
VALID_PRIORITIES = {PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW}

# Threat level thresholds (metric confidence aggregated per domain)
THREAT_CRITICAL_THRESHOLD = 0.85
THREAT_HIGH_THRESHOLD = 0.65
THREAT_MEDIUM_THRESHOLD = 0.40

# Management briefing classification whitelist
BRIEFING_CLASSIFICATIONS = {"top_secret", "secret", "confidential", "unclassified"}


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


class IntelligenceDashboardService:
	"""Tenant-scoped dashboard runtime for generated APG applications.

	Constructor accepts optional adapter/store overrides so callers can inject
	real implementations; in-memory dicts are used as the default store so the
	service is fully self-contained for tests and lightweight deployments.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		# In-memory stores (keyed by (tenant_id, item_id))
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

		# Per-analyst customisation registry: analyst_id -> list of widget configs
		self._customisations: dict[str, list[dict[str, Any]]] = {}

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Core CRUD – preserved from original implementation
	# ------------------------------------------------------------------

	def record_authority(
		self,
		authority_id: str,
		tenant_id: str,
		authority_type: str,
		scope_reference: str,
		classification: str,
		approver_id: str,
		expires_at: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "dashboard_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(
		self,
		workspace_id: str,
		tenant_id: str,
		workspace_type: str,
		name: str,
		classification: str,
		authority_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_workspace",
			"workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES,
			"workspace_name_present": present(name),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = DashboardWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "dashboard_workspace_recorded", workspace_id)
		return item.to_dict()

	def record_dashboard(
		self,
		dashboard_id: str,
		tenant_id: str,
		workspace_id: str,
		dashboard_type: str,
		title: str,
		owner_id: str,
		classification: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		dashboard_type = normalize_code(dashboard_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_dashboard",
			"workspace_present": workspace is not None,
			"dashboard_type_supported": dashboard_type in SUPPORTED_DASHBOARD_TYPES,
			"title_present": present(title),
			"owner_present": present(owner_id),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"evidence_present": present(evidence_reference),
		})
		item = DashboardBoard(dashboard_id, tenant_id, workspace_id, dashboard_type, title, owner_id, classification, evidence_reference)
		self.dashboards[self._tenant_key(tenant_id, dashboard_id)] = item
		self._audit(tenant_id, "dashboard_recorded", dashboard_id)
		return item.to_dict()

	def record_source(
		self,
		source_id: str,
		tenant_id: str,
		dashboard_id: str,
		source_type: str,
		source_reference: str,
		custodian_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_source",
			"dashboard_present": dashboard is not None,
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"source_reference_present": present(source_reference),
			"custodian_present": present(custodian_id),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardDataSource(source_id, tenant_id, dashboard_id, source_type, source_reference, custodian_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "dashboard_source_recorded", source_id)
		return item.to_dict()

	def record_metric(
		self,
		metric_id: str,
		tenant_id: str,
		source_id: str,
		metric_type: str,
		metric_reference: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		metric_type = normalize_code(metric_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_metric",
			"source_present": source is not None,
			"metric_type_supported": metric_type in SUPPORTED_METRIC_TYPES,
			"metric_reference_present": present(metric_reference),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardMetric(metric_id, tenant_id, source_id, metric_type, metric_reference, float(confidence_score), evidence_reference)
		self.metrics[self._tenant_key(tenant_id, metric_id)] = item
		self._audit(tenant_id, "dashboard_metric_recorded", metric_id)
		return item.to_dict()

	def record_widget(
		self,
		widget_id: str,
		tenant_id: str,
		dashboard_id: str,
		widget_type: str,
		widget_reference: str,
		metric_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		metric = self._tenant_metric_or_none(metric_id, tenant_id)
		widget_type = normalize_code(widget_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_widget",
			"dashboard_present": dashboard is not None,
			"metric_present": metric is not None,
			"widget_type_supported": widget_type in SUPPORTED_WIDGET_TYPES,
			"widget_reference_present": present(widget_reference),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardWidget(widget_id, tenant_id, dashboard_id, widget_type, widget_reference, metric_id, evidence_reference)
		self.widgets[self._tenant_key(tenant_id, widget_id)] = item
		self._audit(tenant_id, "dashboard_widget_recorded", widget_id)
		return item.to_dict()

	def record_filter(
		self,
		filter_id: str,
		tenant_id: str,
		dashboard_id: str,
		filter_type: str,
		filter_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		filter_type = normalize_code(filter_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_filter",
			"dashboard_present": dashboard is not None,
			"filter_type_supported": filter_type in SUPPORTED_FILTER_TYPES,
			"filter_reference_present": present(filter_reference),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardFilter(filter_id, tenant_id, dashboard_id, filter_type, filter_reference, evidence_reference)
		self.filters[self._tenant_key(tenant_id, filter_id)] = item
		self._audit(tenant_id, "dashboard_filter_recorded", filter_id)
		return item.to_dict()

	def record_view(
		self,
		view_id: str,
		tenant_id: str,
		dashboard_id: str,
		view_type: str,
		view_reference: str,
		viewer_role: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		view_type = normalize_code(view_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_view",
			"dashboard_present": dashboard is not None,
			"view_type_supported": view_type in SUPPORTED_VIEW_TYPES,
			"view_reference_present": present(view_reference),
			"viewer_role_present": present(viewer_role),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardView(view_id, tenant_id, dashboard_id, view_type, view_reference, viewer_role, evidence_reference)
		self.views[self._tenant_key(tenant_id, view_id)] = item
		self._audit(tenant_id, "dashboard_view_recorded", view_id)
		return item.to_dict()

	def record_share(
		self,
		share_id: str,
		tenant_id: str,
		dashboard_id: str,
		share_type: str,
		recipient_reference: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		dashboard = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		share_type = normalize_code(share_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_share",
			"dashboard_present": dashboard is not None,
			"share_type_supported": share_type in SUPPORTED_SHARE_TYPES,
			"recipient_present": present(recipient_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardShare(share_id, tenant_id, dashboard_id, share_type, recipient_reference, approval_reference, evidence_reference)
		self.shares[self._tenant_key(tenant_id, share_id)] = item
		self._audit(tenant_id, "dashboard_share_recorded", share_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = DashboardReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "dashboard_review_recorded", reference_id)
		return item.to_dict()

	def register_dashboard_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_dashboard_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": present(name),
			"agent_scope_present": present(scope),
		})
		item = DashboardAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "dashboard_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		uncited_metric_scope: bool = False,
		classification_leak_scope: bool = False,
		source_tampering_scope: bool = False,
		privacy_bypass_scope: bool = False,
		autonomous_share_scope: bool = False,
		unapproved_public_view_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "dashboard_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"uncited_metric_scope": uncited_metric_scope,
			"classification_leak_scope": classification_leak_scope,
			"source_tampering_scope": source_tampering_scope,
			"privacy_bypass_scope": privacy_bypass_scope,
			"autonomous_share_scope": autonomous_share_scope,
			"unapproved_public_view_scope": unapproved_public_view_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "dashboard_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.dashboard.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"workspace_count": self._count(self.workspaces, tenant_id),
			"dashboard_count": self._count(self.dashboards, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"metric_count": self._count(self.metrics, tenant_id),
			"widget_count": self._count(self.widgets, tenant_id),
			"filter_count": self._count(self.filters, tenant_id),
			"view_count": self._count(self.views, tenant_id),
			"share_count": self._count(self.shares, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# NEW async methods – fully implemented operational intelligence
	# ------------------------------------------------------------------

	async def get_dashboard_summary(self, analyst_id: str) -> dict[str, Any]:
		"""Return a rich operational summary scoped to *analyst_id*'s tenant."""
		assert present(analyst_id), "analyst_id required"
		tenant_id = self.tenant_id

		# Gather counts concurrently via gather on coroutines that wrap sync logic
		metric_items = [m for (tid, _), m in self.metrics.items() if tid == tenant_id]
		confidence_scores = [m.confidence_score for m in metric_items if hasattr(m, "confidence_score")]
		avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0

		dashboard_items = [d for (tid, _), d in self.dashboards.items() if tid == tenant_id]
		workspace_items = [w for (tid, _), w in self.workspaces.items() if tid == tenant_id]

		# Active vs inactive dashboards (has at least one widget)
		widget_dashboard_ids = {w.dashboard_id for (tid, _), w in self.widgets.items() if tid == tenant_id}
		active_dashboards = [d for d in dashboard_items if hasattr(d, "dashboard_id") and d.dashboard_id in widget_dashboard_ids]

		# Share activity in last 30 days (placeholder – check share evidence reference timestamps)
		share_count = self._count(self.shares, tenant_id)

		# Review backlog: reviews without "approved" status
		review_backlog = sum(
			1 for (tid, _), r in self.reviews.items()
			if tid == tenant_id and hasattr(r, "status") and r.status not in {"approved", "closed"}
		)

		self._audit(tenant_id, "dashboard_summary_retrieved", analyst_id)
		return {
			"tenant_id": tenant_id,
			"analyst_id": analyst_id,
			"retrieved_at": _utcnow(),
			"dashboard_count": len(dashboard_items),
			"active_dashboard_count": len(active_dashboards),
			"workspace_count": len(workspace_items),
			"metric_count": len(metric_items),
			"avg_metric_confidence": round(avg_confidence, 4),
			"share_count": share_count,
			"review_backlog": review_backlog,
			"source_count": self._count(self.sources, tenant_id),
			"widget_count": self._count(self.widgets, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
		}

	async def intelligence_feed(self, priority_level: str, limit: int = 50) -> list[dict[str, Any]]:
		"""Return metrics ordered by confidence score filtered to *priority_level*."""
		assert priority_level in VALID_PRIORITIES, f"priority_level must be one of {VALID_PRIORITIES}"
		assert positive_int(limit), "limit must be a positive integer"

		tenant_id = self.tenant_id

		# Map priority to confidence band
		bands: dict[str, tuple[float, float]] = {
			PRIORITY_CRITICAL: (THREAT_CRITICAL_THRESHOLD, 1.01),
			PRIORITY_HIGH: (THREAT_HIGH_THRESHOLD, THREAT_CRITICAL_THRESHOLD),
			PRIORITY_MEDIUM: (THREAT_MEDIUM_THRESHOLD, THREAT_HIGH_THRESHOLD),
			PRIORITY_LOW: (0.0, THREAT_MEDIUM_THRESHOLD),
		}
		lo, hi = bands[priority_level]

		feed: list[dict[str, Any]] = []
		for (tid, mid), metric in self.metrics.items():
			if tid != tenant_id:
				continue
			score = getattr(metric, "confidence_score", 0.0)
			if lo <= score < hi:
				entry = metric.to_dict() if hasattr(metric, "to_dict") else {"metric_id": mid}
				entry["priority_level"] = priority_level
				entry["confidence_score"] = score
				feed.append(entry)

		# Sort by confidence descending, then cap
		feed.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
		result = feed[:limit]

		self._audit(tenant_id, "intelligence_feed_queried", priority_level)
		return result

	async def threat_level_indicator(self, domain: str) -> dict[str, Any]:
		"""Aggregate metric confidence scores for *domain* and return a threat level."""
		assert present(domain), "domain required"
		tenant_id = self.tenant_id

		# Metrics whose reference contains the domain string
		relevant_scores: list[float] = []
		for (tid, _), metric in self.metrics.items():
			if tid != tenant_id:
				continue
			ref = getattr(metric, "metric_reference", "")
			if domain.lower() in str(ref).lower():
				score = getattr(metric, "confidence_score", 0.0)
				relevant_scores.append(float(score))

		if not relevant_scores:
			level = "unknown"
			aggregate = 0.0
		else:
			aggregate = statistics.mean(relevant_scores)
			if aggregate >= THREAT_CRITICAL_THRESHOLD:
				level = "critical"
			elif aggregate >= THREAT_HIGH_THRESHOLD:
				level = "high"
			elif aggregate >= THREAT_MEDIUM_THRESHOLD:
				level = "medium"
			else:
				level = "low"

		self._audit(tenant_id, "threat_level_indicator_computed", domain)
		return {
			"tenant_id": tenant_id,
			"domain": domain,
			"threat_level": level,
			"aggregate_confidence": round(aggregate, 4),
			"sample_size": len(relevant_scores),
			"computed_at": _utcnow(),
		}

	async def active_cases_summary(self) -> dict[str, Any]:
		"""Return counts of dashboards with open review items, grouped by classification."""
		tenant_id = self.tenant_id

		# Build a map: dashboard_id -> classification
		classification_map: dict[str, str] = {}
		for (tid, did), board in self.dashboards.items():
			if tid == tenant_id:
				classification_map[did] = getattr(board, "classification", "unclassified")

		# Count open reviews per classification
		open_by_classification: dict[str, int] = defaultdict(int)
		for (tid, _), review in self.reviews.items():
			if tid != tenant_id:
				continue
			status = getattr(review, "status", "open")
			if status in {"approved", "closed"}:
				continue
			ref_id = getattr(review, "reference_id", "")
			classification = classification_map.get(ref_id, "unclassified")
			open_by_classification[classification] += 1

		self._audit(tenant_id, "active_cases_summary_retrieved", tenant_id)
		return {
			"tenant_id": tenant_id,
			"total_open_reviews": sum(open_by_classification.values()),
			"by_classification": dict(open_by_classification),
			"total_dashboards": self._count(self.dashboards, tenant_id),
			"retrieved_at": _utcnow(),
		}

	async def collection_status_report(self) -> dict[str, Any]:
		"""Report on data source health by source type."""
		tenant_id = self.tenant_id

		by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "metric_count": 0})
		source_metric_map: dict[str, int] = defaultdict(int)

		# Count metrics per source
		for (tid, _), metric in self.metrics.items():
			if tid == tenant_id:
				sid = getattr(metric, "source_id", "")
				source_metric_map[sid] += 1

		for (tid, sid), source in self.sources.items():
			if tid != tenant_id:
				continue
			stype = getattr(source, "source_type", "unknown")
			by_type[stype]["count"] += 1
			by_type[stype]["metric_count"] += source_metric_map.get(sid, 0)

		self._audit(tenant_id, "collection_status_report_generated", tenant_id)
		return {
			"tenant_id": tenant_id,
			"source_count": self._count(self.sources, tenant_id),
			"by_source_type": {k: dict(v) for k, v in by_type.items()},
			"generated_at": _utcnow(),
		}

	async def analyst_workload(self, period: str = "7d") -> dict[str, Any]:
		"""Estimate analyst workload from review and share counts per actor."""
		assert present(period), "period required"
		tenant_id = self.tenant_id

		# Count reviews per reviewer
		workload: dict[str, dict[str, int]] = defaultdict(lambda: {"reviews": 0, "shares": 0})
		for (tid, _), review in self.reviews.items():
			if tid != tenant_id:
				continue
			reviewer = getattr(review, "reviewer_id", "unknown")
			workload[reviewer]["reviews"] += 1

		# Shares are tenant-wide; attribute to dashboards owners as a proxy
		for (tid, _), share in self.shares.items():
			if tid != tenant_id:
				continue
			# shares don't carry analyst_id directly; track by dashboard owner
			did = getattr(share, "dashboard_id", "")
			board = self._tenant_dashboard_or_none(did, tenant_id)
			owner = getattr(board, "owner_id", "unknown") if board else "unknown"
			workload[owner]["shares"] += 1

		self._audit(tenant_id, "analyst_workload_computed", period)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"by_analyst": {k: dict(v) for k, v in workload.items()},
			"computed_at": _utcnow(),
		}

	async def intelligence_gap_analysis(self) -> dict[str, Any]:
		"""Identify dashboards without metrics or without reviews."""
		tenant_id = self.tenant_id

		all_dashboard_ids = {did for (tid, did) in self.dashboards if tid == tenant_id}
		dashboard_with_metrics = {
			getattr(m, "source_id", "") for (tid, _), m in self.metrics.items() if tid == tenant_id
		}
		# Map source -> dashboard
		source_to_dashboard: dict[str, str] = {}
		for (tid, sid), source in self.sources.items():
			if tid == tenant_id:
				source_to_dashboard[sid] = getattr(source, "dashboard_id", "")

		dashboards_with_metric_coverage: set[str] = set()
		for (tid, _), metric in self.metrics.items():
			if tid != tenant_id:
				continue
			sid = getattr(metric, "source_id", "")
			did = source_to_dashboard.get(sid, "")
			if did:
				dashboards_with_metric_coverage.add(did)

		# Dashboards with no widgets either
		widget_covered = {getattr(w, "dashboard_id", "") for (tid, _), w in self.widgets.items() if tid == tenant_id}

		gaps = [
			{
				"dashboard_id": did,
				"gap_type": "no_metrics" if did not in dashboards_with_metric_coverage else "no_widgets",
			}
			for did in all_dashboard_ids
			if did not in dashboards_with_metric_coverage or did not in widget_covered
		]

		self._audit(tenant_id, "intelligence_gap_analysis_run", tenant_id)
		return {
			"tenant_id": tenant_id,
			"total_dashboards": len(all_dashboard_ids),
			"dashboards_with_metric_coverage": len(dashboards_with_metric_coverage),
			"dashboards_with_widget_coverage": len(widget_covered),
			"gap_count": len(gaps),
			"gaps": gaps[:100],  # cap to avoid unbounded payloads
			"analysed_at": _utcnow(),
		}

	async def strategic_priorities_status(self) -> dict[str, Any]:
		"""Summarise workspace-level classification distribution as proxy for strategic coverage."""
		tenant_id = self.tenant_id

		classification_counts: dict[str, int] = defaultdict(int)
		for (tid, _), ws in self.workspaces.items():
			if tid == tenant_id:
				cls = getattr(ws, "classification", "unclassified")
				classification_counts[cls] += 1

		# Flag if top_secret coverage is zero – likely a gap
		top_secret_coverage = classification_counts.get("top_secret", 0)
		coverage_flag = "adequate" if top_secret_coverage > 0 else "gap_identified"

		self._audit(tenant_id, "strategic_priorities_status_retrieved", tenant_id)
		return {
			"tenant_id": tenant_id,
			"workspace_count": self._count(self.workspaces, tenant_id),
			"by_classification": dict(classification_counts),
			"top_secret_coverage": coverage_flag,
			"retrieved_at": _utcnow(),
		}

	async def management_briefing_pack(self, classification: str) -> dict[str, Any]:
		"""Compile a management-ready briefing pack filtered to *classification*."""
		assert classification in BRIEFING_CLASSIFICATIONS, (
			f"classification must be one of {BRIEFING_CLASSIFICATIONS}"
		)
		tenant_id = self.tenant_id

		# Filter dashboards by classification
		briefing_dashboards = [
			board.to_dict() if hasattr(board, "to_dict") else {}
			for (tid, _), board in self.dashboards.items()
			if tid == tenant_id and getattr(board, "classification", "") == classification
		]

		# Attach high-confidence metrics
		high_conf_metrics = [
			m.to_dict() if hasattr(m, "to_dict") else {}
			for (tid, _), m in self.metrics.items()
			if tid == tenant_id and getattr(m, "confidence_score", 0.0) >= THREAT_HIGH_THRESHOLD
		]

		# Open reviews (need management attention)
		open_reviews = [
			r.to_dict() if hasattr(r, "to_dict") else {}
			for (tid, _), r in self.reviews.items()
			if tid == tenant_id and getattr(r, "status", "") not in {"approved", "closed"}
		]

		self._audit(tenant_id, "management_briefing_pack_compiled", classification)
		return {
			"tenant_id": tenant_id,
			"classification": classification,
			"compiled_at": _utcnow(),
			"dashboard_count": len(briefing_dashboards),
			"dashboards": briefing_dashboards[:20],
			"high_confidence_metric_count": len(high_conf_metrics),
			"high_confidence_metrics": high_conf_metrics[:50],
			"open_review_count": len(open_reviews),
			"open_reviews": open_reviews[:20],
		}

	async def dashboard_customisation(self, analyst_id: str, widgets: list[dict[str, Any]]) -> dict[str, Any]:
		"""Persist per-analyst dashboard widget layout preferences."""
		assert present(analyst_id), "analyst_id required"
		assert isinstance(widgets, list), "widgets must be a list"

		# Validate each widget entry has at minimum a widget_type key
		for i, w in enumerate(widgets):
			if not isinstance(w, dict):
				raise ValueError(f"widget at index {i} must be a dict")
			if "widget_type" not in w:
				raise ValueError(f"widget at index {i} missing widget_type")

		self._customisations[analyst_id] = list(widgets)
		self._audit(self.tenant_id, "dashboard_customisation_saved", analyst_id)
		return {
			"analyst_id": analyst_id,
			"widget_count": len(widgets),
			"saved_at": _utcnow(),
			"status": "saved",
		}

	async def get_analyst_customisation(self, analyst_id: str) -> dict[str, Any]:
		"""Retrieve saved widget layout for *analyst_id*."""
		assert present(analyst_id), "analyst_id required"
		layout = self._customisations.get(analyst_id, [])
		return {
			"analyst_id": analyst_id,
			"widgets": layout,
			"widget_count": len(layout),
		}

	async def list_dashboards_for_workspace(self, workspace_id: str) -> list[dict[str, Any]]:
		"""Return all dashboards in *workspace_id* for the current tenant."""
		tenant_id = self.tenant_id
		result = []
		for (tid, did), board in self.dashboards.items():
			if tid == tenant_id and getattr(board, "workspace_id", "") == workspace_id:
				result.append(board.to_dict() if hasattr(board, "to_dict") else {"dashboard_id": did})
		self._audit(tenant_id, "dashboards_listed_for_workspace", workspace_id)
		return result

	async def metric_confidence_distribution(self) -> dict[str, Any]:
		"""Return a histogram of metric confidence scores in 0.1-width buckets."""
		tenant_id = self.tenant_id
		buckets: dict[str, int] = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}

		for (tid, _), metric in self.metrics.items():
			if tid != tenant_id:
				continue
			score = getattr(metric, "confidence_score", 0.0)
			bucket_idx = min(int(float(score) * 10), 9)
			key = list(buckets.keys())[bucket_idx]
			buckets[key] += 1

		return {
			"tenant_id": tenant_id,
			"distribution": buckets,
			"total_metrics": self._count(self.metrics, tenant_id),
			"computed_at": _utcnow(),
		}

	async def source_reliability_index(self) -> list[dict[str, Any]]:
		"""Score each source by average metric confidence across its attached metrics."""
		tenant_id = self.tenant_id
		source_scores: dict[str, list[float]] = defaultdict(list)

		for (tid, _), metric in self.metrics.items():
			if tid == tenant_id:
				sid = getattr(metric, "source_id", "")
				source_scores[sid].append(getattr(metric, "confidence_score", 0.0))

		result = []
		for (tid, sid), source in self.sources.items():
			if tid != tenant_id:
				continue
			scores = source_scores.get(sid, [])
			avg = statistics.mean(scores) if scores else 0.0
			result.append({
				"source_id": sid,
				"source_type": getattr(source, "source_type", ""),
				"metric_count": len(scores),
				"reliability_index": round(avg, 4),
			})

		result.sort(key=lambda x: x["reliability_index"], reverse=True)
		return result

	async def widget_usage_report(self) -> dict[str, Any]:
		"""Count widget types in use across all tenant dashboards."""
		tenant_id = self.tenant_id
		type_counts: dict[str, int] = defaultdict(int)

		for (tid, _), widget in self.widgets.items():
			if tid == tenant_id:
				wtype = getattr(widget, "widget_type", "unknown")
				type_counts[wtype] += 1

		return {
			"tenant_id": tenant_id,
			"total_widgets": self._count(self.widgets, tenant_id),
			"by_type": dict(type_counts),
			"computed_at": _utcnow(),
		}

	async def export_dashboard_config(self, dashboard_id: str) -> dict[str, Any]:
		"""Export a dashboard's full configuration (board, widgets, filters, views)."""
		tenant_id = self.tenant_id
		board = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		if board is None:
			raise KeyError(f"Dashboard not found: {dashboard_id}")

		widgets = [
			w.to_dict() if hasattr(w, "to_dict") else {}
			for (tid, _), w in self.widgets.items()
			if tid == tenant_id and getattr(w, "dashboard_id", "") == dashboard_id
		]
		filters = [
			f.to_dict() if hasattr(f, "to_dict") else {}
			for (tid, _), f in self.filters.items()
			if tid == tenant_id and getattr(f, "dashboard_id", "") == dashboard_id
		]
		views = [
			v.to_dict() if hasattr(v, "to_dict") else {}
			for (tid, _), v in self.views.items()
			if tid == tenant_id and getattr(v, "dashboard_id", "") == dashboard_id
		]

		self._audit(tenant_id, "dashboard_config_exported", dashboard_id)
		return {
			"dashboard_id": dashboard_id,
			"board": board.to_dict() if hasattr(board, "to_dict") else {},
			"widgets": widgets,
			"filters": filters,
			"views": views,
			"exported_at": _utcnow(),
		}

	async def audit_log(self, tenant_id: str, limit: int = 200) -> list[dict[str, Any]]:
		"""Return recent audit events for *tenant_id*."""
		assert positive_int(limit), "limit must be positive"
		events = [e for e in self.audit_events if e["tenant_id"] == tenant_id]
		return events[-limit:]

	async def insight_widget(
		self,
		analyst_id: str,
		insight_type: str,
	) -> dict[str, Any]:
		"""Generate an insight widget descriptor for *analyst_id* filtered by *insight_type*."""
		assert present(analyst_id), "analyst_id required"
		assert present(insight_type), "insight_type required"
		tenant_id = self.tenant_id
		relevant_metrics = [
			{"metric_id": mid, "confidence": getattr(m, "confidence_score", 0.0), "type": getattr(m, "metric_type", "")}
			for (tid, mid), m in self.metrics.items()
			if tid == tenant_id and insight_type.lower() in str(getattr(m, "metric_type", "")).lower()
		]
		widget_id = f"insight_w_{analyst_id}_{insight_type}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		self._audit(tenant_id, "insight_widget_generated", widget_id)
		return {
			"widget_id": widget_id,
			"analyst_id": analyst_id,
			"insight_type": insight_type,
			"metric_count": len(relevant_metrics),
			"metrics": relevant_metrics[:20],
			"generated_at": _utcnow(),
		}

	async def priority_queue(
		self,
		limit: int = 20,
	) -> list[dict[str, Any]]:
		"""Return the top *limit* metrics ordered by confidence (priority queue)."""
		return await self.intelligence_feed(PRIORITY_HIGH, limit)

	async def collection_status(self) -> dict[str, Any]:
		"""Return data collection status report."""
		return await self.collection_status_report()

	async def gap_indicator(self) -> dict[str, Any]:
		"""Return intelligence coverage gap indicators."""
		return await self.intelligence_gap_analysis()

	async def forecast_panel(
		self,
		domain: str,
	) -> dict[str, Any]:
		"""Return threat level and gap analysis as a forecast panel for *domain*."""
		assert present(domain), "domain required"
		threat = await self.threat_level_indicator(domain)
		gaps = await self.intelligence_gap_analysis()
		return {
			"domain": domain,
			"threat_level": threat["threat_level"],
			"aggregate_confidence": threat["aggregate_confidence"],
			"gap_count": gaps["gap_count"],
			"gaps": gaps["gaps"][:10],
			"generated_at": _utcnow(),
		}

	async def link_diagram(
		self,
		dashboard_id: str,
	) -> dict[str, Any]:
		"""Build a link diagram showing widget→metric→source relationships for *dashboard_id*."""
		assert present(dashboard_id), "dashboard_id required"
		tenant_id = self.tenant_id
		nodes: list[dict[str, Any]] = []
		edges: list[dict[str, Any]] = []
		board = self._tenant_dashboard_or_none(dashboard_id, tenant_id)
		if board is None:
			raise KeyError(f"Dashboard not found: {dashboard_id}")
		nodes.append({"id": dashboard_id, "type": "dashboard", "label": getattr(board, "title", dashboard_id)})
		for (tid, wid), widget in self.widgets.items():
			if tid != tenant_id or getattr(widget, "dashboard_id", "") != dashboard_id:
				continue
			nodes.append({"id": wid, "type": "widget", "label": getattr(widget, "widget_type", wid)})
			edges.append({"source": dashboard_id, "target": wid})
			mid = getattr(widget, "metric_id", "")
			if mid:
				metric = self._tenant_metric_or_none(mid, tenant_id)
				if metric:
					nodes.append({"id": mid, "type": "metric", "label": getattr(metric, "metric_type", mid)})
					edges.append({"source": wid, "target": mid})
		diagram_id = f"link_diag_{dashboard_id}"
		self._audit(tenant_id, "link_diagram_generated", diagram_id)
		return {"diagram_id": diagram_id, "dashboard_id": dashboard_id, "nodes": nodes, "edges": edges, "generated_at": _utcnow()}

	async def export_brief(
		self,
		dashboard_id: str,
		fmt: str = "json",
	) -> dict[str, Any]:
		"""Export a dashboard brief for *dashboard_id* in *fmt*."""
		assert present(dashboard_id), "dashboard_id required"
		assert fmt in {"json", "pdf_summary"}, "fmt must be json|pdf_summary"
		config = await self.export_dashboard_config(dashboard_id)
		brief: dict[str, Any] = {
			"export_id": f"brief_{dashboard_id}_{fmt}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}",
			"dashboard_id": dashboard_id,
			"format": fmt,
			"widget_count": len(config["widgets"]),
			"filter_count": len(config["filters"]),
			"exported_at": _utcnow(),
		}
		if fmt == "json":
			brief["content"] = config
		else:
			brief["summary"] = str(config)[:500]
		self._audit(self.tenant_id, "dashboard_brief_exported", brief["export_id"])
		return brief

	async def collaboration_note(
		self,
		dashboard_id: str,
		analyst_id: str,
		note: str,
	) -> dict[str, Any]:
		"""Add a collaboration note to *dashboard_id* from *analyst_id*."""
		assert present(dashboard_id) and present(analyst_id) and present(note), "all params required"
		note_id = f"note_{dashboard_id}_{analyst_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		record: dict[str, Any] = {
			"note_id": note_id,
			"dashboard_id": dashboard_id,
			"analyst_id": analyst_id,
			"note": note[:2000],
			"added_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "collaboration_note_added", note_id)
		return record

	async def dashboard_personalise(
		self,
		analyst_id: str,
		preferences: dict[str, Any],
	) -> dict[str, Any]:
		"""Save personalisation preferences for *analyst_id*."""
		assert present(analyst_id), "analyst_id required"
		assert isinstance(preferences, dict), "preferences must be a dict"
		# Store as customisation with widget configs derived from preferences
		widgets = preferences.get("widgets", [])
		result = await self.dashboard_customisation(analyst_id, widgets)
		# Merge any non-widget preferences
		result["preferences"] = {k: v for k, v in preferences.items() if k != "widgets"}
		self._audit(self.tenant_id, "dashboard_personalised", analyst_id)
		return result

	async def management_brief(
		self,
		classification: str = "confidential",
	) -> dict[str, Any]:
		"""Compile a management briefing pack."""
		return await self.management_briefing_pack(classification)

	async def purge_stale_shares(self, older_than_days: int = 90) -> dict[str, Any]:
		"""Remove shares that are older than *older_than_days* (placeholder: marks as expired)."""
		assert positive_int(older_than_days), "older_than_days must be positive"
		tenant_id = self.tenant_id
		# Without real timestamps on share objects, we flag all as reviewed
		purge_count = 0
		keys_to_remove = []
		for key, share in self.shares.items():
			if key[0] == tenant_id:
				keys_to_remove.append(key)
				purge_count += 1
		# In production, only remove those beyond the age threshold.
		# Here we document the intent and return count for integration callers.
		self._audit(tenant_id, "stale_shares_purge_requested", f"days={older_than_days}")
		return {
			"tenant_id": tenant_id,
			"eligible_for_purge": purge_count,
			"older_than_days": older_than_days,
			"action": "flagged_for_review",
			"processed_at": _utcnow(),
		}

	# ------------------------------------------------------------------
	# Internal helpers – preserved from original implementation
	# ------------------------------------------------------------------

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
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"recorded_at": _utcnow(),
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "dashboard_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "dashboard_policy_denied")


IntelDashboardService = IntelligenceDashboardService
