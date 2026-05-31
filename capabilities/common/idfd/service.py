"""Executable service layer for APG Identity Federation."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .federation_runtime import (
	FederationHealthInspector,
	FederationSessionIssuer,
	MetadataFreshnessInspector,
	iso_hours_ago,
	iso_hours_from_now,
)
from .models import (
	CertificateRecord,
	ClaimMapping,
	FederatedSession,
	FederationAuditEvent,
	FederationAgentRecord,
	FederationHealthReport,
	FederationProvider,
	IdfdLifecycleBatchRecord,
	ProviderProtocol,
	ProviderStatus,
	SessionStatus,
	utc_now_iso,
)


StoreKey = tuple[str, str]


class IdfdService:
	"""Tenant-aware federation provider, mapping, session, certificate, and audit runtime."""

	def __init__(self) -> None:
		self._providers: dict[StoreKey, FederationProvider] = {}
		self._mappings: dict[StoreKey, ClaimMapping] = {}
		self._sessions: dict[StoreKey, FederatedSession] = {}
		self._certificates: dict[StoreKey, CertificateRecord] = {}
		self._audit_events: dict[StoreKey, FederationAuditEvent] = {}
		self._health_reports: dict[StoreKey, FederationHealthReport] = {}
		self._federation_agents: dict[StoreKey, FederationAgentRecord] = {}
		self._lifecycle_batches: dict[StoreKey, IdfdLifecycleBatchRecord] = {}
		self._counter = count(1)
		self._metadata = MetadataFreshnessInspector()
		self._session_issuer = FederationSessionIssuer()
		self._health = FederationHealthInspector()
		contract = get_capability_contract()
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_provider(
		self,
		provider_id: str,
		tenant_id: str,
		name: str,
		protocol: str,
		owner_id: str,
		signing_key_id: str,
		metadata_url: str = "",
		assertion_encrypted: bool = True,
		redirect_allowlist: list[str] | None = None,
		pkce_required: bool = True,
		metadata_signed: bool = True,
		response_signature_validated: bool = True,
		tls_enabled: bool = True,
		metadata_refresh_completed: bool = True,
		metadata_age_hours: int | float = 0,
		status: str = ProviderStatus.ACTIVE.value,
	) -> dict[str, Any]:
		self._ensure_new(self._providers, tenant_id, provider_id)
		protocol_value = ProviderProtocol(protocol)
		config = DEFAULT_CONFIGURATION
		enabled = set(config["providers"]["enabled_provider_types"])
		if protocol_value.value not in enabled:
			raise ValueError("provider_protocol_not_enabled")
		if config["providers"]["provider_owner_required"] and not owner_id:
			raise PermissionError("provider_owner_required")
		self._enforce_federation_policy(
			tenant_id=tenant_id,
			operation="register_provider",
			owner_present=bool(owner_id),
			signing_key_present=bool(signing_key_id),
			protocol=protocol_value.value,
			protocol_enabled=protocol_value.value in enabled,
			metadata_present=bool(metadata_url),
			metadata_signed=metadata_signed,
			assertion_encrypted=assertion_encrypted,
			redirect_allowlist_configured=bool(redirect_allowlist),
			pkce_required=pkce_required,
			response_signature_validated=response_signature_validated,
			tls_enabled=tls_enabled,
			metadata_age_hours=float(metadata_age_hours),
			metadata_refresh_completed=metadata_refresh_completed,
		)
		provider = FederationProvider(
			id=provider_id,
			tenant_id=tenant_id,
			name=name,
			protocol=protocol_value,
			owner_id=owner_id,
			signing_key_id=signing_key_id,
			metadata_url=metadata_url,
			assertion_encrypted=assertion_encrypted,
			redirect_allowlist=list(redirect_allowlist or []),
			pkce_required=pkce_required,
			metadata_refreshed_at=iso_hours_ago(metadata_age_hours),
			status=ProviderStatus(status),
		)
		self._providers[self._key(tenant_id, provider_id)] = provider
		self._audit(tenant_id, "provider_registered", provider_id=provider_id, reason=protocol_value.value)
		return provider.to_dict()

	def refresh_provider_metadata(
		self,
		provider_id: str,
		tenant_id: str,
		metadata_refreshed_at: str | None = None,
	) -> dict[str, Any]:
		provider = self._require_provider(provider_id, tenant_id)
		provider.metadata_refreshed_at = metadata_refreshed_at or utc_now_iso()
		if provider.status == ProviderStatus.STALE:
			provider.status = ProviderStatus.ACTIVE
		provider.updated_at = utc_now_iso()
		self._audit(tenant_id, "metadata_refreshed", provider_id=provider_id)
		return provider.to_dict()

	def add_claim_mapping(
		self,
		mapping_id: str,
		tenant_id: str,
		provider_id: str,
		source_claim: str,
		target_claim: str,
		transform: str = "copy",
		reviewed: bool = True,
	) -> dict[str, Any]:
		self._ensure_new(self._mappings, tenant_id, mapping_id)
		self._require_provider(provider_id, tenant_id)
		self._enforce_federation_policy(
			tenant_id=tenant_id,
			operation="add_claim_mapping",
			source_claim_present=bool(source_claim),
			target_claim_present=bool(target_claim),
			claim_mapping_reviewed=reviewed,
			sensitive_claim=target_claim.lower() in {"role", "roles", "groups", "entitlements", "department"},
			privacy_review_recorded=reviewed,
		)
		mapping = ClaimMapping(
			id=mapping_id,
			tenant_id=tenant_id,
			provider_id=provider_id,
			source_claim=source_claim,
			target_claim=target_claim,
			transform=transform,
			reviewed=reviewed,
		)
		self._mappings[self._key(tenant_id, mapping_id)] = mapping
		self._audit(tenant_id, "claim_mapping_added", provider_id=provider_id, reason=f"{source_claim}->{target_claim}")
		return mapping.to_dict()

	def issue_session(
		self,
		session_id: str,
		tenant_id: str,
		provider_id: str,
		subject_id: str,
		session_privilege: str = "standard",
		mfa_completed: bool = True,
		risk_score: float = 0.0,
		max_session_hours: int | None = None,
		reauth_completed: bool = True,
	) -> dict[str, Any]:
		self._ensure_new(self._sessions, tenant_id, session_id)
		provider = self._require_provider(provider_id, tenant_id)
		hours = max_session_hours or int(DEFAULT_CONFIGURATION["sessions"]["max_session_hours"])
		self._enforce_federation_policy(
			tenant_id=tenant_id,
			operation="issue_session",
			provider_active=provider.status == ProviderStatus.ACTIVE,
			protocol=provider.protocol.value,
			assertion_encrypted=provider.assertion_encrypted,
			redirect_allowlist_configured=bool(provider.redirect_allowlist),
			session_privilege=session_privilege,
			mfa_completed=mfa_completed,
			session_hours=hours,
			risk_score=float(risk_score),
			reauth_completed=reauth_completed,
		)
		session = self._session_issuer.issue(
			session_id=session_id,
			tenant_id=tenant_id,
			provider_id=provider_id,
			subject_id=subject_id,
			session_privilege=session_privilege,
			mfa_completed=mfa_completed,
			max_session_hours=hours,
			risk_score=risk_score,
		)
		self._sessions[self._key(tenant_id, session_id)] = session
		self._audit(tenant_id, "session_issued", provider_id=provider_id, subject_id=subject_id)
		return session.to_dict()

	def revoke_session(self, session_id: str, tenant_id: str, reason: str = "manual") -> dict[str, Any]:
		session = self._require_session(session_id, tenant_id)
		self._enforce_federation_policy(tenant_id=tenant_id, operation="revoke_session", reason_present=bool(reason))
		session.status = SessionStatus.REVOKED
		session.revoked_at = utc_now_iso()
		session.revocation_reason = reason
		self._audit(tenant_id, "session_revoked", provider_id=session.provider_id, subject_id=session.subject_id, reason=reason)
		return session.to_dict()

	def register_certificate(
		self,
		certificate_id: str,
		tenant_id: str,
		provider_id: str,
		key_id: str,
		expires_at: str,
		active: bool = True,
		rotated_at: str | None = None,
	) -> dict[str, Any]:
		self._ensure_new(self._certificates, tenant_id, certificate_id)
		self._require_provider(provider_id, tenant_id)
		self._enforce_federation_policy(
			tenant_id=tenant_id,
			operation="register_certificate",
			provider_present=bool(provider_id),
			key_present=bool(key_id),
		)
		certificate = CertificateRecord(
			id=certificate_id,
			tenant_id=tenant_id,
			provider_id=provider_id,
			key_id=key_id,
			expires_at=expires_at,
			active=active,
			rotated_at=rotated_at,
		)
		self._certificates[self._key(tenant_id, certificate_id)] = certificate
		self._audit(tenant_id, "certificate_registered", provider_id=provider_id, reason=key_id)
		return certificate.to_dict()

	def health_report(self, report_id: str, tenant_id: str) -> dict[str, Any]:
		config = DEFAULT_CONFIGURATION
		summary = self._health.summarize(
			tenant_id=tenant_id,
			providers=list(self._providers.values()),
			sessions=list(self._sessions.values()),
			certificates=list(self._certificates.values()),
			metadata_refresh_hours=int(config["providers"]["metadata_refresh_hours"]),
			certificate_rotation_days=int(config["governance"]["certificate_rotation_days"]),
		)
		report = FederationHealthReport(id=report_id, tenant_id=tenant_id, **summary)
		self._health_reports[self._key(tenant_id, report_id)] = report
		self._audit(tenant_id, "health_report_generated", reason=report_id)
		return report.to_dict()

	def register_federation_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		self._ensure_new(self._federation_agents, tenant_id, agent_id)
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_federation_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "federation_agent_denied") for action in result["actions"])
			raise PermissionError(reasons)
		if not str(name or "").strip():
			raise ValueError("federation_agent_name_required")
		agent = FederationAgentRecord(
			id=str(agent_id).strip(),
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="pending_review" if result["decision"] == "require_review" else "active",
		)
		self._federation_agents[self._key(tenant_id, agent.id)] = agent
		self._audit(tenant_id, "federation_agent_registered", reason=f"{runtime_value}:{role_value}", decision=result["decision"])
		return agent.to_dict()

	def validate_idfd_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "federation_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		if not str(tenant_id or "").strip():
			self._enforce_federation_policy(tenant_id=tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("idfd_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_idfd_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "validate_idfd_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
		})
		accepted = result["decision"] == "allow"
		record = IdfdLifecycleBatchRecord(
			id=batch_id or f"idfdbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, f"idfd_lifecycle_batch_{record.status}", reason=f"{operation_value}:{stream_value}", decision=result["decision"])
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "idfd_lifecycle_batch_denied") for action in result["actions"])
			raise PermissionError(reasons)
		return record.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper for generated package probes."""
		data = dict(metadata or {})
		protocol = str(data.get("protocol") or ProviderProtocol.SAML.value)
		redirect_allowlist = list(data.get("redirect_allowlist") or ["https://example.test/callback"])
		return self.register_provider(
			provider_id=record_id,
			tenant_id=tenant_id,
			name=str(data.get("name") or "Compatibility identity provider"),
			protocol=protocol,
			owner_id=str(data.get("owner_id") or "system"),
			signing_key_id=str(data.get("signing_key_id") or "signing-key"),
			metadata_url=str(data.get("metadata_url") or "https://idp.example.test/metadata"),
			assertion_encrypted=bool(data.get("assertion_encrypted", True)),
			redirect_allowlist=redirect_allowlist,
			pkce_required=bool(data.get("pkce_required", True)),
			metadata_refresh_completed=bool(data.get("metadata_refresh_completed", True)),
			metadata_age_hours=float(data.get("metadata_age_hours") or 0),
			status=status,
		)

	def list_providers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._providers, tenant_id)

	def list_claim_mappings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._mappings, tenant_id)

	def list_sessions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sessions, tenant_id)

	def list_certificates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._certificates, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_health_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._health_reports, tenant_id)

	def list_federation_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._federation_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_providers(tenant_id)

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		config = DEFAULT_CONFIGURATION
		stale = self._metadata.stale_providers(
			list(self._providers.values()),
			tenant_id,
			int(config["providers"]["metadata_refresh_hours"]),
		)
		active_sessions = [
			session
			for session in self._sessions.values()
			if session.tenant_id == tenant_id and self._session_issuer.effective_status(session) == SessionStatus.ACTIVE
		]
		return {
			"tenant_id": tenant_id,
			"provider_count": len(self.list_providers(tenant_id)),
			"claim_mapping_count": len(self.list_claim_mappings(tenant_id)),
			"active_session_count": len(active_sessions),
			"certificate_count": len(self.list_certificates(tenant_id)),
			"stale_provider_count": len(stale),
			"federation_agent_count": len(self.list_federation_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_federation_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"routes": len(self.describe(tenant_id)["ui"]["routes"]),
			"theme": self.describe(tenant_id)["theme"]["name"],
		}

	def _enforce_federation_policy(self, tenant_id: str, **context: Any) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			**context,
		})
		if result["decision"] != "allow":
			reasons = ", ".join(action.get("reason", "federation_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "federation_policy_blocked")

	def _require_provider(self, provider_id: str, tenant_id: str) -> FederationProvider:
		provider = self._providers.get(self._key(tenant_id, provider_id))
		if provider is None:
			self._raise_cross_tenant_if_present(self._providers, provider_id, tenant_id)
			raise KeyError("provider_missing")
		return provider

	def _require_session(self, session_id: str, tenant_id: str) -> FederatedSession:
		session = self._sessions.get(self._key(tenant_id, session_id))
		if session is None:
			self._raise_cross_tenant_if_present(self._sessions, session_id, tenant_id)
			raise KeyError("session_missing")
		return session

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		provider_id: str | None = None,
		subject_id: str | None = None,
		decision: str = "allow",
		reason: str = "",
	) -> None:
		if not DEFAULT_CONFIGURATION["governance"]["audit_federation_events"]:
			return
		event_id = f"audit-{next(self._counter)}"
		self._audit_events[self._key(tenant_id, event_id)] = FederationAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			provider_id=provider_id,
			subject_id=subject_id,
			decision=decision,
			reason=reason,
		)

	def _list(self, records: dict[StoreKey, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _ensure_new(self, records: dict[StoreKey, Any], tenant_id: str, record_id: str) -> None:
		if not record_id:
			raise ValueError("federation_record_id_required")
		if self._key(tenant_id, record_id) in records:
			raise ValueError(f"federation_record_already_exists:{record_id}")

	def _raise_cross_tenant_if_present(self, records: dict[StoreKey, Any], record_id: str, tenant_id: str) -> None:
		if any(record.id == record_id and record.tenant_id != tenant_id for record in records.values()):
			result = self.evaluate({"tenant_context_present": bool(tenant_id), "cross_tenant_access": True})
			reasons = ", ".join(action.get("reason", "cross_tenant_federation_access_denied") for action in result["actions"])
			raise PermissionError(reasons or "cross_tenant_federation_access_denied")

	def _key(self, tenant_id: str, record_id: str) -> StoreKey:
		return (tenant_id, record_id)


def expires_in_days(days: int) -> str:
	"""Return an ISO timestamp used by tests and generated probes."""
	return iso_hours_from_now(days * 24)


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
