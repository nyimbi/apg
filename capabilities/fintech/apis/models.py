"""Dependency-light data models for APG Banking APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class APIProduct:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	product_type: str
	environment: str
	scopes: list[str]
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "owner_id": self.owner_id, "product_type": self.product_type, "environment": self.environment, "scopes": list(self.scopes), "status": self.status}


@dataclass
class DeveloperOrganization:
	id: str
	tenant_id: str
	name: str
	kyb_reference: str
	security_review_reference: str
	risk_clearance_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "kyb_reference": self.kyb_reference, "security_review_reference": self.security_review_reference, "risk_clearance_reference": self.risk_clearance_reference, "status": self.status}


@dataclass
class DeveloperApplication:
	id: str
	tenant_id: str
	developer_id: str
	name: str
	environment: str
	redirect_uri: str
	terms_reference: str
	status: str = "registered"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "developer_id": self.developer_id, "name": self.name, "environment": self.environment, "redirect_uri": self.redirect_uri, "terms_reference": self.terms_reference, "status": self.status}


@dataclass
class ConsentGrant:
	id: str
	tenant_id: str
	application_id: str
	customer_reference: str
	scopes: list[str]
	expiry_date: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "application_id": self.application_id, "customer_reference": self.customer_reference, "scopes": list(self.scopes), "expiry_date": self.expiry_date, "status": self.status}


@dataclass
class APIClient:
	id: str
	tenant_id: str
	application_id: str
	auth_flow: str
	key_reference: str
	scopes: list[str]
	status: str = "issued"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "application_id": self.application_id, "auth_flow": self.auth_flow, "key_reference": self.key_reference, "scopes": list(self.scopes), "status": self.status}


@dataclass
class EndpointPolicy:
	id: str
	tenant_id: str
	product_id: str
	route: str
	required_scope: str
	throttle_policy_reference: str
	risk_policy_reference: str
	status: str = "published"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "product_id": self.product_id, "route": self.route, "required_scope": self.required_scope, "throttle_policy_reference": self.throttle_policy_reference, "risk_policy_reference": self.risk_policy_reference, "status": self.status}


@dataclass
class WebhookSubscription:
	id: str
	tenant_id: str
	application_id: str
	event_type: str
	endpoint: str
	signing_secret_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "application_id": self.application_id, "event_type": self.event_type, "endpoint": self.endpoint, "signing_secret_reference": self.signing_secret_reference, "status": self.status}


@dataclass
class APICallRecord:
	id: str
	tenant_id: str
	client_id: str
	product_id: str
	endpoint_id: str
	status_code: int
	call_count: int
	risk_reference: str
	human_approval: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "client_id": self.client_id, "product_id": self.product_id, "endpoint_id": self.endpoint_id, "status_code": self.status_code, "call_count": self.call_count, "risk_reference": self.risk_reference, "human_approval": self.human_approval}


@dataclass
class RateLimitBucket:
	id: str
	tenant_id: str
	client_id: str
	limit: int
	window_seconds: int
	remaining: int
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "client_id": self.client_id, "limit": self.limit, "window_seconds": self.window_seconds, "remaining": self.remaining, "status": self.status}


@dataclass
class SLAIncident:
	id: str
	tenant_id: str
	severity: str
	owner_id: str
	evidence_references: list[str]
	human_approval: str = ""
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "severity": self.severity, "owner_id": self.owner_id, "evidence_references": list(self.evidence_references), "human_approval": self.human_approval, "status": self.status}


@dataclass
class APIEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
