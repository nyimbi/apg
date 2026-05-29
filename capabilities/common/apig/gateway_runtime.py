"""Dependency-light APIG route publication runtime for package composition."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	GatewayAuditEvent,
	GatewayQuotaReview,
	GatewayRouteRecord,
	GatewayUpstreamRecord,
)


UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class ApigService:
	"""Tenant-scoped gateway control-plane facade for generated APG apps."""

	def __init__(self) -> None:
		self._upstreams: dict[tuple[str, str], GatewayUpstreamRecord] = {}
		self._routes: dict[tuple[str, str], GatewayRouteRecord] = {}
		self._quota_reviews: dict[tuple[str, str], GatewayQuotaReview] = {}
		self._events: list[GatewayAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_upstream(
		self,
		upstream_id: str,
		tenant_id: str,
		name: str,
		base_url: str,
		owner: str,
		health: str = "healthy",
		labels: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, upstream_id) in self._upstreams:
			raise ValueError(f"upstream already exists for tenant: {upstream_id}")
		if not name:
			raise ValueError("upstream name is required")
		if not owner:
			raise ValueError("upstream owner is required")
		if not (base_url.startswith("http://") or base_url.startswith("https://")):
			raise ValueError("upstream base_url must be http or https")
		record = GatewayUpstreamRecord(
			id=upstream_id,
			tenant_id=tenant_id,
			name=name,
			base_url=base_url.rstrip("/"),
			owner=owner,
			health=health,
			labels=dict(labels or {}),
		)
		self._upstreams[self._tenant_key(tenant_id, upstream_id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type="upstream_registered",
			subject_id=upstream_id,
			message=f"Registered upstream {name}.",
			evidence={"base_url": record.base_url, "owner": owner, "health": health},
		)
		return record.model_dump(mode="json")

	def request_route(
		self,
		route_id: str,
		tenant_id: str,
		path: str,
		methods: list[str] | tuple[str, ...],
		upstream_id: str,
		owner: str,
		route_exposure: str = "internal",
		auth_policy_attached: bool = True,
		threat_policy_attached: bool = True,
		requested_rps_limit: int = 1000,
		wasm_filter_attached: bool = False,
		filter_signature_verified: bool = True,
		justification: str = "",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, route_id) in self._routes:
			raise ValueError(f"route already exists for tenant: {route_id}")
		upstream_registered = self._tenant_key(tenant_id, upstream_id) in self._upstreams
		normalized_methods = [method.upper() for method in methods]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_route",
			"service_registered": upstream_registered,
			"route_exposure": route_exposure,
			"auth_policy_attached": auth_policy_attached,
			"unsafe_http_method_enabled": any(method in UNSAFE_METHODS for method in normalized_methods),
			"threat_policy_attached": threat_policy_attached,
			"wasm_filter_attached": wasm_filter_attached,
			"filter_signature_verified": filter_signature_verified,
			"requested_rps_limit": requested_rps_limit,
			"quota_review_recorded": False,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		self._get_upstream(tenant_id, upstream_id)
		if not path.startswith("/"):
			raise ValueError("route path must start with /")
		if not owner:
			raise ValueError("route owner is required")
		status = "pending_quota_review" if result["decision"] == "require_review" else "active"
		route = GatewayRouteRecord(
			id=route_id,
			tenant_id=tenant_id,
			path=path,
			methods=normalized_methods,
			upstream_id=upstream_id,
			owner=owner,
			route_exposure=route_exposure,
			auth_policy_attached=auth_policy_attached,
			threat_policy_attached=threat_policy_attached,
			requested_rps_limit=requested_rps_limit,
			wasm_filter_attached=wasm_filter_attached,
			filter_signature_verified=filter_signature_verified,
			status=status,
		)
		self._routes[self._tenant_key(tenant_id, route_id)] = route
		self._record_event(
			tenant_id=tenant_id,
			event_type="route_requested",
			subject_id=route_id,
			message=f"Requested route {path}.",
			evidence={"status": status, "upstream_id": upstream_id, "requested_rps_limit": requested_rps_limit},
		)
		if result["decision"] == "require_review":
			review = GatewayQuotaReview(
				id=f"quota:{route_id}",
				tenant_id=tenant_id,
				route_id=route_id,
				requested_rps_limit=requested_rps_limit,
				requester=owner,
				justification=justification or "High gateway quota requested.",
			)
			self._quota_reviews[self._tenant_key(tenant_id, review.id)] = review
			self._record_event(
				tenant_id=tenant_id,
				event_type="quota_review_requested",
				subject_id=review.id,
				message=f"Requested quota review for {route_id}.",
				evidence={"route_id": route_id, "requested_rps_limit": requested_rps_limit},
			)
			return {"route": route.model_dump(mode="json"), "quota_review": review.model_dump(mode="json")}
		return route.model_dump(mode="json")

	def decide_quota_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		review = self._quota_reviews.get(self._tenant_key(tenant_id, review_id))
		if review is None:
			raise KeyError(f"unknown quota review for tenant: {review_id}")
		if decision not in {"approved", "rejected"}:
			raise ValueError("quota review decision must be approved or rejected")
		if not reviewer:
			raise ValueError("quota review reviewer is required")
		if not notes:
			raise ValueError("quota review notes are required")
		decided = GatewayQuotaReview(
			id=review.id,
			tenant_id=review.tenant_id,
			route_id=review.route_id,
			requested_rps_limit=review.requested_rps_limit,
			requester=review.requester,
			justification=review.justification,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
		)
		self._quota_reviews[self._tenant_key(tenant_id, review_id)] = decided
		self._record_event(
			tenant_id=tenant_id,
			event_type="quota_review_decided",
			subject_id=review_id,
			message=f"Quota review {review_id} was {decision}.",
			evidence={"route_id": review.route_id, "reviewer": reviewer},
		)
		return decided.model_dump(mode="json")

	def activate_route(self, route_id: str, tenant_id: str) -> dict[str, Any]:
		route = self._get_route(tenant_id, route_id)
		if route.status == "pending_quota_review":
			review = self._quota_reviews.get(self._tenant_key(tenant_id, f"quota:{route_id}"))
			if review is None or review.decision != "approved":
				raise PermissionError("quota_review_required")
		activated = GatewayRouteRecord(
			id=route.id,
			tenant_id=route.tenant_id,
			path=route.path,
			methods=list(route.methods),
			upstream_id=route.upstream_id,
			owner=route.owner,
			route_exposure=route.route_exposure,
			auth_policy_attached=route.auth_policy_attached,
			threat_policy_attached=route.threat_policy_attached,
			requested_rps_limit=route.requested_rps_limit,
			wasm_filter_attached=route.wasm_filter_attached,
			filter_signature_verified=route.filter_signature_verified,
			status="active",
		)
		self._routes[self._tenant_key(tenant_id, route_id)] = activated
		self._record_event(
			tenant_id=tenant_id,
			event_type="route_activated",
			subject_id=route_id,
			message=f"Activated route {route.path}.",
			evidence={"requested_rps_limit": route.requested_rps_limit},
		)
		return activated.model_dump(mode="json")

	def list_upstreams(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._upstreams, tenant_id)

	def list_routes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._routes, tenant_id)

	def list_quota_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._quota_reviews, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._events)
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.model_dump(mode="json") for event in events]

	def gateway_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		routes = self.list_routes(tenant_id)
		reviews = self.list_quota_reviews(tenant_id)
		return {
			"tenant_id": tenant_id,
			"upstream_count": len(self.list_upstreams(tenant_id)),
			"route_count": len(routes),
			"active_route_count": len([route for route in routes if route["status"] == "active"]),
			"pending_quota_review_count": len([review for review in reviews if review["decision"] == "pending"]),
			"public_route_count": len([route for route in routes if route["route_exposure"] == "public"]),
			"edge_filter_count": len([route for route in routes if route["wasm_filter_attached"]]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_routes(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		upstream_id = str(metadata.get("upstream_id") or "manual-upstream")
		if self._tenant_key(tenant_id, upstream_id) not in self._upstreams:
			self.register_upstream(
				upstream_id=upstream_id,
				tenant_id=tenant_id,
				name=str(metadata.get("upstream_name") or upstream_id),
				base_url=str(metadata.get("base_url") or "https://example.internal"),
				owner=str(metadata.get("owner") or "operations"),
			)
		route = self.request_route(
			route_id=record_id,
			tenant_id=tenant_id,
			path=str(metadata.get("path") or f"/{record_id}"),
			methods=list(metadata.get("methods") or ["GET"]),
			upstream_id=upstream_id,
			owner=str(metadata.get("owner") or "operations"),
			route_exposure=str(metadata.get("route_exposure") or "internal"),
			auth_policy_attached=bool(metadata.get("auth_policy_attached", True)),
			threat_policy_attached=bool(metadata.get("threat_policy_attached", True)),
			requested_rps_limit=int(metadata.get("requested_rps_limit") or 1000),
			wasm_filter_attached=bool(metadata.get("wasm_filter_attached", False)),
			filter_signature_verified=bool(metadata.get("filter_signature_verified", True)),
		)
		if "route" in route:
			return route["route"]
		route["status"] = status
		return route

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		_raise_if_blocked(result)

	def _get_upstream(self, tenant_id: str, upstream_id: str) -> GatewayUpstreamRecord:
		upstream = self._upstreams.get(self._tenant_key(tenant_id, upstream_id))
		if upstream is None:
			raise KeyError(f"unknown upstream for tenant: {upstream_id}")
		return upstream

	def _get_route(self, tenant_id: str, route_id: str) -> GatewayRouteRecord:
		route = self._routes.get(self._tenant_key(tenant_id, route_id))
		if route is None:
			raise KeyError(f"unknown route for tenant: {route_id}")
		return route

	def _dump_tenant_records(self, records: dict[tuple[str, str], Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.model_dump(mode="json") for record in sorted(values, key=lambda item: item.id)]

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
	) -> None:
		self._events.append(
			GatewayAuditEvent(
				tenant_id=tenant_id,
				event_type=event_type,
				subject_id=subject_id,
				message=message,
				evidence=dict(evidence or {}),
			)
		)


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(action.get("reason", "gateway_policy_blocked") for action in result["actions"])
	raise PermissionError(reasons or "gateway_policy_blocked")


__all__ = ["ApigService"]
