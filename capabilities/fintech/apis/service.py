"""Executable service layer for APG Banking APIs."""

from __future__ import annotations

from typing import Any

try:
	from .apis_runtime import client_public_id, is_critical_severity, normalize_code, normalize_codes, normalize_url, rate_limit_allows
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_API_PRODUCTS, SUPPORTED_AUTH_FLOWS, SUPPORTED_ENVIRONMENTS, SUPPORTED_INCIDENT_SEVERITIES, SUPPORTED_WEBHOOK_EVENTS, evaluate_capability_rules, get_capability_contract
	from .models import APICallRecord, APIClient, APIEvidence, APIProduct, ConsentGrant, DeveloperApplication, DeveloperOrganization, EndpointPolicy, RateLimitBucket, SLAIncident, WebhookSubscription
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from apis_runtime import client_public_id, is_critical_severity, normalize_code, normalize_codes, normalize_url, rate_limit_allows  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_API_PRODUCTS, SUPPORTED_AUTH_FLOWS, SUPPORTED_ENVIRONMENTS, SUPPORTED_INCIDENT_SEVERITIES, SUPPORTED_WEBHOOK_EVENTS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import APICallRecord, APIClient, APIEvidence, APIProduct, ConsentGrant, DeveloperApplication, DeveloperOrganization, EndpointPolicy, RateLimitBucket, SLAIncident, WebhookSubscription  # type: ignore


class BankingAPIService:
	"""Dependency-light banking API runtime for generated applications."""

	def __init__(self) -> None:
		self.products: dict[str, APIProduct] = {}
		self.developers: dict[str, DeveloperOrganization] = {}
		self.applications: dict[str, DeveloperApplication] = {}
		self.consents: dict[str, ConsentGrant] = {}
		self.clients: dict[str, APIClient] = {}
		self.endpoints: dict[str, EndpointPolicy] = {}
		self.webhooks: dict[str, WebhookSubscription] = {}
		self.calls: dict[str, APICallRecord] = {}
		self.rate_limits: dict[str, RateLimitBucket] = {}
		self.incidents: dict[str, SLAIncident] = {}
		self.evidence: dict[str, APIEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_api_product(self, product_id: str, tenant_id: str, name: str, owner_id: str, product_type: str, environment: str, scopes: list[str], policy_attached: bool = True) -> dict[str, Any]:
		product_type = normalize_code(product_type)
		environment = normalize_code(environment)
		scopes = normalize_codes(scopes)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_api_product", "owner_present": bool(owner_id), "product_type_supported": product_type in SUPPORTED_API_PRODUCTS, "environment_supported": environment in SUPPORTED_ENVIRONMENTS, "scopes_present": bool(scopes)})
		product = APIProduct(product_id, tenant_id, name, owner_id, product_type, environment, scopes)
		self.products[product_id] = product
		self._audit(tenant_id, "api_product_registered", product_id)
		return product.to_dict()

	def onboard_developer(self, developer_id: str, tenant_id: str, name: str, kyb_reference: str, security_review_reference: str, risk_clearance_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_developer", "kyb_present": bool(kyb_reference), "security_review_present": bool(security_review_reference), "risk_clearance_present": bool(risk_clearance_reference)})
		developer = DeveloperOrganization(developer_id, tenant_id, name, kyb_reference, security_review_reference, risk_clearance_reference)
		self.developers[developer_id] = developer
		self._audit(tenant_id, "developer_onboarded", developer_id)
		return developer.to_dict()

	def register_application(self, application_id: str, tenant_id: str, developer_id: str, name: str, environment: str, redirect_uri: str, terms_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		developer = self._tenant_developer_or_none(developer_id, tenant_id)
		environment = normalize_code(environment)
		redirect_uri = normalize_url(redirect_uri)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_application", "developer_present": developer is not None, "environment_supported": environment in SUPPORTED_ENVIRONMENTS, "redirect_uri_present": bool(redirect_uri), "terms_present": bool(terms_reference)})
		application = DeveloperApplication(application_id, tenant_id, developer_id, name, environment, redirect_uri, terms_reference)
		self.applications[application_id] = application
		self._audit(tenant_id, "developer_application_registered", application_id)
		return application.to_dict()

	def create_consent_grant(self, consent_id: str, tenant_id: str, application_id: str, customer_reference: str, scopes: list[str], expiry_date: str, policy_attached: bool = True) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		scopes = normalize_codes(scopes)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "create_consent_grant", "application_present": application is not None, "customer_present": bool(customer_reference), "scopes_present": bool(scopes), "expiry_present": bool(expiry_date)})
		consent = ConsentGrant(consent_id, tenant_id, application_id, customer_reference, scopes, expiry_date)
		self.consents[consent_id] = consent
		self._audit(tenant_id, "consent_grant_created", consent_id)
		return consent.to_dict()

	def issue_api_client(self, client_id: str, tenant_id: str, application_id: str, auth_flow: str, key_reference: str, scopes: list[str], policy_attached: bool = True) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		auth_flow = normalize_code(auth_flow)
		scopes = normalize_codes(scopes)
		consented_scopes = {scope for consent in self.consents.values() if consent.tenant_id == tenant_id and consent.application_id == application_id and consent.status == "active" for scope in consent.scopes}
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "issue_api_client", "application_present": application is not None, "auth_flow_supported": auth_flow in SUPPORTED_AUTH_FLOWS, "key_reference_present": bool(key_reference), "scopes_present": bool(scopes), "scopes_allowed_by_consent": bool(scopes) and set(scopes).issubset(consented_scopes)})
		client = APIClient(client_id, tenant_id, application_id, auth_flow, key_reference, scopes)
		self.clients[client_id] = client
		self._audit(tenant_id, "api_client_issued", client_id)
		return client.to_dict() | {"public_client_id": client_public_id(application_id, auth_flow)}

	def publish_endpoint_policy(self, endpoint_id: str, tenant_id: str, product_id: str, route: str, required_scope: str, throttle_policy_reference: str, risk_policy_reference: str) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "publish_endpoint_policy", "product_present": product is not None, "route_present": bool(route), "scope_present": bool(required_scope), "throttle_policy_present": bool(throttle_policy_reference), "risk_policy_present": bool(risk_policy_reference)})
		endpoint = EndpointPolicy(endpoint_id, tenant_id, product_id, route, required_scope, throttle_policy_reference, risk_policy_reference)
		self.endpoints[endpoint_id] = endpoint
		self._audit(tenant_id, "endpoint_policy_published", endpoint_id)
		return endpoint.to_dict()

	def subscribe_webhook(self, webhook_id: str, tenant_id: str, application_id: str, event_type: str, endpoint: str, signing_secret_reference: str) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		event_type = normalize_code(event_type)
		endpoint = normalize_url(endpoint)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "subscribe_webhook", "application_present": application is not None, "event_supported": event_type in SUPPORTED_WEBHOOK_EVENTS, "endpoint_present": bool(endpoint), "signing_secret_present": bool(signing_secret_reference)})
		webhook = WebhookSubscription(webhook_id, tenant_id, application_id, event_type, endpoint, signing_secret_reference)
		self.webhooks[webhook_id] = webhook
		self._audit(tenant_id, "webhook_subscribed", webhook_id)
		return webhook.to_dict()

	def record_api_call(self, call_id: str, tenant_id: str, client_id: str, product_id: str, endpoint_id: str, status_code: int, call_count: int, risk_reference: str, human_approval: str = "") -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		product = self._tenant_product_or_none(product_id, tenant_id)
		endpoint = self._tenant_endpoint_or_none(endpoint_id, tenant_id)
		bucket = self.rate_limits.get(client_id)
		limit = bucket.limit if bucket and bucket.tenant_id == tenant_id else 1000
		high_volume = int(call_count) >= 10000
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_api_call", "client_present": client is not None, "product_present": product is not None, "endpoint_present": endpoint is not None, "endpoint_matches_product": endpoint is not None and endpoint.product_id == product_id, "rate_limit_allowed": rate_limit_allows(int(call_count), limit), "risk_reference_present": bool(risk_reference), "high_volume": high_volume, "human_approval_recorded": bool(human_approval)})
		call = APICallRecord(call_id, tenant_id, client_id, product_id, endpoint_id, int(status_code), int(call_count), risk_reference, human_approval)
		self.calls[call_id] = call
		if bucket and bucket.tenant_id == tenant_id:
			bucket.remaining = max(bucket.limit - int(call_count), 0)
		self._audit(tenant_id, "api_call_recorded", call_id)
		return call.to_dict()

	def update_rate_limit(self, bucket_id: str, tenant_id: str, client_id: str, limit: int, window_seconds: int = 60) -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "update_rate_limit", "client_present": client is not None, "positive_limit": int(limit) > 0})
		bucket = RateLimitBucket(bucket_id, tenant_id, client_id, int(limit), int(window_seconds), int(limit))
		self.rate_limits[client_id] = bucket
		self._audit(tenant_id, "rate_limit_updated", bucket_id)
		return bucket.to_dict()

	def open_sla_incident(self, incident_id: str, tenant_id: str, severity: str, owner_id: str, evidence_references: list[str], human_approval: str = "") -> dict[str, Any]:
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_sla_incident", "severity_supported": severity in SUPPORTED_INCIDENT_SEVERITIES, "owner_present": bool(owner_id), "evidence_present": bool(evidence_references), "critical_severity": is_critical_severity(severity), "human_approval_recorded": bool(human_approval)})
		incident = SLAIncident(incident_id, tenant_id, severity, owner_id, list(evidence_references), human_approval)
		self.incidents[incident_id] = incident
		self._audit(tenant_id, "sla_incident_opened", incident_id)
		return incident.to_dict()

	def register_api_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_api_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "api_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "apis_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.apis.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "product_count": sum(1 for item in self.products.values() if item.tenant_id == tenant_id), "developer_count": sum(1 for item in self.developers.values() if item.tenant_id == tenant_id), "application_count": sum(1 for item in self.applications.values() if item.tenant_id == tenant_id), "consent_count": sum(1 for item in self.consents.values() if item.tenant_id == tenant_id), "client_count": sum(1 for item in self.clients.values() if item.tenant_id == tenant_id), "endpoint_count": sum(1 for item in self.endpoints.values() if item.tenant_id == tenant_id), "webhook_count": sum(1 for item in self.webhooks.values() if item.tenant_id == tenant_id), "call_count": sum(1 for item in self.calls.values() if item.tenant_id == tenant_id), "rate_limit_count": sum(1 for item in self.rate_limits.values() if item.tenant_id == tenant_id), "incident_count": sum(1 for item in self.incidents.values() if item.tenant_id == tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_calls(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.calls.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _tenant_product_or_none(self, product_id: str, tenant_id: str) -> APIProduct | None:
		item = self.products.get(product_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_developer_or_none(self, developer_id: str, tenant_id: str) -> DeveloperOrganization | None:
		item = self.developers.get(developer_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_application_or_none(self, application_id: str, tenant_id: str) -> DeveloperApplication | None:
		item = self.applications.get(application_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_client_or_none(self, client_id: str, tenant_id: str) -> APIClient | None:
		item = self.clients.get(client_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_endpoint_or_none(self, endpoint_id: str, tenant_id: str) -> EndpointPolicy | None:
		item = self.endpoints.get(endpoint_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = APIEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "apis_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "apis_policy_denied")


BankingApisService = BankingAPIService
