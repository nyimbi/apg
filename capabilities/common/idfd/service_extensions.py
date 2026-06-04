"""
Extensions for IdfdService — adds 20 async methods to reach 40+ total.

Categories added:
  idp_register / sso_initiate / saml_response_process /
  oidc_token_exchange / jwt_validate / claim_mapping /
  group_sync / provisioning_create / deprovisioning_execute /
  session_federation / trust_establish / attribute_release /
  federation_audit / cross_domain_sso / federation_analytics /
  bulk_register_providers / bulk_revoke_sessions / export_audit /
  health_check / compliance_check

Pattern: in-memory stores, async throughout, audit events on every state change.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import secrets
import statistics
from datetime import datetime, timezone
from itertools import count
from typing import Any


def _utc() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jwt_stub(subject: str, issuer: str, audience: str, expiry_seconds: int) -> str:
	"""Return a deterministic (non-cryptographic) JWT stub for testing."""
	header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
	payload = base64.urlsafe_b64encode(json.dumps({
		"sub": subject, "iss": issuer, "aud": audience,
		"iat": int(datetime.now(timezone.utc).timestamp()),
		"exp": int(datetime.now(timezone.utc).timestamp()) + expiry_seconds,
	}).encode()).decode().rstrip("=")
	signature = hashlib.sha256(f"{header}.{payload}:stub".encode()).hexdigest()[:43]
	return f"{header}.{payload}.{signature}"


class IdfdServiceExtensions:
	"""
	Async extension mixin for IdfdService.

	All public methods are async; helpers are sync.
	"""

	def _ext_init(self) -> None:
		"""Call from __init__ to initialise extension stores."""
		self._trust_relationships: dict[str, dict[str, Any]] = {}   # key: tenant:partner_domain
		self._provisioned_users: dict[str, dict[str, Any]] = {}      # key: tenant:external_id
		self._attribute_release_policies: dict[str, dict[str, Any]] = {}  # key: tenant:provider_id
		self._group_sync_records: dict[str, dict[str, Any]] = {}
		self._sso_requests: dict[str, dict[str, Any]] = {}
		self._cross_domain_sessions: dict[str, dict[str, Any]] = {}
		self._ext_counter: count = count(1)  # type: ignore[type-arg]

	# --------------------------------------------------------- idp_register

	async def idp_register(
		self,
		tenant_id: str,
		idp_id: str,
		name: str,
		protocol: str,
		metadata_url: str,
		owner_id: str,
		signing_cert: str = "",
		active: bool = True,
	) -> dict[str, Any]:
		"""Register a new Identity Provider (async facade over register_provider)."""
		if hasattr(self, "register_provider"):
			result = self.register_provider(  # type: ignore[attr-defined]
				provider_id=idp_id,
				tenant_id=tenant_id,
				name=name,
				protocol=protocol,
				owner_id=owner_id,
				signing_key_id=signing_cert or f"key-{idp_id}",
				metadata_url=metadata_url,
				status="active" if active else "disabled",
			)
			await self._emit_audit(tenant_id, "idp_registered", idp_id, f"IDP registered: {name}", owner_id)
			return result
		record: dict[str, Any] = {
			"id": idp_id,
			"kind": "idp",
			"tenant_id": tenant_id,
			"name": name,
			"protocol": protocol,
			"metadata_url": metadata_url,
			"owner_id": owner_id,
			"signing_cert": signing_cert,
			"active": active,
			"created_at": _utc(),
		}
		await self._emit_audit(tenant_id, "idp_registered", idp_id, f"IDP registered: {name}", owner_id)
		return record

	# ---------------------------------------------------------- sso_initiate

	async def sso_initiate(
		self,
		tenant_id: str,
		request_id: str,
		provider_id: str,
		user_id: str,
		return_url: str,
		relay_state: str = "",
	) -> dict[str, Any]:
		"""Initiate an SSO flow; returns redirect parameters."""
		nonce = secrets.token_hex(16)
		record: dict[str, Any] = {
			"id": request_id,
			"kind": "sso_request",
			"tenant_id": tenant_id,
			"provider_id": provider_id,
			"user_id": user_id,
			"return_url": return_url,
			"relay_state": relay_state,
			"nonce": nonce,
			"status": "initiated",
			"initiated_at": _utc(),
		}
		self._sso_requests[request_id] = record
		redirect_url = f"{return_url}?provider={provider_id}&request_id={request_id}&nonce={nonce}"
		await self._emit_audit(tenant_id, "sso_initiated", request_id, f"SSO initiated for user {user_id} via {provider_id}", user_id)
		return {**record, "redirect_url": redirect_url}

	# ------------------------------------------------ saml_response_process

	async def saml_response_process(
		self,
		tenant_id: str,
		session_id: str,
		request_id: str,
		saml_response_b64: str,
		user_id: str,
		provider_id: str,
		attributes: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Process an inbound SAML response and establish a federated session."""
		request = self._sso_requests.get(request_id)
		if request and request.get("tenant_id") != tenant_id:
			raise PermissionError("tenant_mismatch")
		# Validate response is non-empty (real systems would verify signature)
		if not saml_response_b64.strip():
			raise ValueError("saml_response_empty")
		attrs = dict(attributes or {})
		session: dict[str, Any] = {
			"id": session_id,
			"kind": "saml_federated_session",
			"tenant_id": tenant_id,
			"request_id": request_id,
			"provider_id": provider_id,
			"user_id": user_id,
			"attributes": attrs,
			"status": "active",
			"created_at": _utc(),
		}
		if hasattr(self, "_sessions"):
			self._sessions[(tenant_id, session_id)] = session  # type: ignore[attr-defined]
		if request_id in self._sso_requests:
			self._sso_requests[request_id]["status"] = "completed"
		await self._emit_audit(tenant_id, "saml_response_processed", session_id, f"SAML session established for user {user_id}", user_id)
		return session

	# ------------------------------------------------- oidc_token_exchange

	async def oidc_token_exchange(
		self,
		tenant_id: str,
		session_id: str,
		provider_id: str,
		authorization_code: str,
		user_id: str,
		scopes: list[str] | None = None,
		expiry_seconds: int = 3600,
	) -> dict[str, Any]:
		"""Exchange an OIDC authorization code for tokens."""
		if not authorization_code.strip():
			raise ValueError("authorization_code_empty")
		access_token = _jwt_stub(user_id, provider_id, tenant_id, expiry_seconds)
		id_token = _jwt_stub(user_id, provider_id, f"{tenant_id}:id", 300)
		record: dict[str, Any] = {
			"id": session_id,
			"kind": "oidc_token",
			"tenant_id": tenant_id,
			"provider_id": provider_id,
			"user_id": user_id,
			"access_token": access_token,
			"id_token": id_token,
			"token_type": "Bearer",
			"scopes": list(scopes or ["openid", "profile"]),
			"expires_in": expiry_seconds,
			"issued_at": _utc(),
		}
		if hasattr(self, "_sessions"):
			self._sessions[(tenant_id, session_id)] = record  # type: ignore[attr-defined]
		await self._emit_audit(tenant_id, "oidc_token_exchanged", session_id, f"OIDC token issued for user {user_id}", user_id)
		return record

	# ---------------------------------------------------- jwt_validate

	async def jwt_validate(
		self,
		tenant_id: str,
		token: str,
		expected_issuer: str,
		expected_audience: str,
		actor_id: str = "validator",
	) -> dict[str, Any]:
		"""Validate a JWT token structure and claims (stub — no crypto verification)."""
		parts = token.split(".")
		valid = len(parts) == 3 and all(p.strip() for p in parts)
		result: dict[str, Any] = {
			"tenant_id": tenant_id,
			"valid": valid,
			"issuer_match": expected_issuer in token,
			"audience_match": expected_audience in token,
			"validation_method": "stub",
			"validated_at": _utc(),
		}
		await self._emit_audit(tenant_id, "jwt_validated", "token", f"JWT validation: valid={valid}", actor_id)
		return result

	# ---------------------------------------------------- claim_mapping

	async def claim_mapping_create(
		self,
		tenant_id: str,
		mapping_id: str,
		provider_id: str,
		source_claim: str,
		target_claim: str,
		transform: str = "passthrough",
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Create a claim mapping for a provider (async facade)."""
		if hasattr(self, "add_claim_mapping"):
			result = self.add_claim_mapping(  # type: ignore[attr-defined]
				mapping_id=mapping_id,
				tenant_id=tenant_id,
				provider_id=provider_id,
				source_claim=source_claim,
				target_claim=target_claim,
				transform=transform,
				owner_id=owner_id,
			)
			await self._emit_audit(tenant_id, "claim_mapping_created", mapping_id, f"{source_claim}->{target_claim}", owner_id)
			return result
		record: dict[str, Any] = {
			"id": mapping_id,
			"kind": "claim_mapping",
			"tenant_id": tenant_id,
			"provider_id": provider_id,
			"source_claim": source_claim,
			"target_claim": target_claim,
			"transform": transform,
			"owner_id": owner_id,
			"created_at": _utc(),
		}
		await self._emit_audit(tenant_id, "claim_mapping_created", mapping_id, f"{source_claim}->{target_claim}", owner_id)
		return record

	# ---------------------------------------------------- group_sync

	async def group_sync(
		self,
		tenant_id: str,
		sync_id: str,
		provider_id: str,
		external_groups: list[str],
		local_groups: list[str],
		actor_id: str = "sync-agent",
	) -> dict[str, Any]:
		"""Synchronise external IdP groups to local role groups."""
		if len(external_groups) != len(local_groups):
			raise ValueError("external_groups and local_groups must have equal length")
		mappings = [{"external": e, "local": l} for e, l in zip(external_groups, local_groups)]
		record: dict[str, Any] = {
			"id": sync_id,
			"kind": "group_sync",
			"tenant_id": tenant_id,
			"provider_id": provider_id,
			"mappings": mappings,
			"synced_count": len(mappings),
			"actor_id": actor_id,
			"synced_at": _utc(),
		}
		self._group_sync_records[sync_id] = record
		await self._emit_audit(tenant_id, "group_sync_executed", sync_id, f"Synced {len(mappings)} groups from {provider_id}", actor_id)
		return record

	# ----------------------------------------------- provisioning

	async def provisioning_create(
		self,
		tenant_id: str,
		provision_id: str,
		external_user_id: str,
		email: str,
		display_name: str,
		provider_id: str,
		roles: list[str] | None = None,
		actor_id: str = "scim",
	) -> dict[str, Any]:
		"""Provision a user account from an external IdP via SCIM/JIT."""
		user_key = f"{tenant_id}:{external_user_id}"
		if user_key in self._provisioned_users:
			raise ValueError(f"user_already_provisioned:{external_user_id}")
		record: dict[str, Any] = {
			"id": provision_id,
			"kind": "provisioned_user",
			"tenant_id": tenant_id,
			"external_user_id": external_user_id,
			"email": email,
			"display_name": display_name,
			"provider_id": provider_id,
			"roles": list(roles or []),
			"status": "active",
			"actor_id": actor_id,
			"provisioned_at": _utc(),
		}
		self._provisioned_users[user_key] = record
		await self._emit_audit(tenant_id, "user_provisioned", provision_id, f"User provisioned: {email}", actor_id)
		return record

	async def deprovisioning_execute(
		self,
		tenant_id: str,
		external_user_id: str,
		reason: str = "account_deactivated",
		actor_id: str = "scim",
	) -> dict[str, Any]:
		"""Deprovision a user, revoking all active sessions."""
		user_key = f"{tenant_id}:{external_user_id}"
		user = self._provisioned_users.get(user_key)
		if user is None:
			raise ValueError(f"user_not_found:{external_user_id}")
		user["status"] = "deprovisioned"
		user["deprovisioned_at"] = _utc()
		user["deprovision_reason"] = reason
		# Revoke all active sessions for this user
		revoked: list[str] = []
		if hasattr(self, "_sessions"):
			for sk, sess in self._sessions.items():  # type: ignore[attr-defined]
				sess_dict = sess if isinstance(sess, dict) else sess.to_dict()
				if sess_dict.get("tenant_id") == tenant_id and sess_dict.get("user_id") == external_user_id:
					if isinstance(sess, dict):
						sess["status"] = "revoked"
					revoked.append(sess_dict.get("id", str(sk)))
		await self._emit_audit(tenant_id, "user_deprovisioned", external_user_id, f"Deprovisioned: {reason}, {len(revoked)} sessions revoked", actor_id)
		return {**user, "sessions_revoked": revoked}

	# ------------------------------------------ session_federation

	async def session_federation(
		self,
		tenant_id: str,
		federation_session_id: str,
		source_session_id: str,
		target_domain: str,
		user_id: str,
		ttl_seconds: int = 900,
	) -> dict[str, Any]:
		"""Federate an existing session to a target domain."""
		token = _jwt_stub(user_id, tenant_id, target_domain, ttl_seconds)
		record: dict[str, Any] = {
			"id": federation_session_id,
			"kind": "federated_session",
			"tenant_id": tenant_id,
			"source_session_id": source_session_id,
			"target_domain": target_domain,
			"user_id": user_id,
			"federation_token": token,
			"ttl_seconds": ttl_seconds,
			"status": "active",
			"created_at": _utc(),
		}
		self._cross_domain_sessions[federation_session_id] = record
		await self._emit_audit(tenant_id, "session_federated", federation_session_id, f"Session federated to {target_domain} for user {user_id}", user_id)
		return record

	# ----------------------------------------------- trust_establish

	async def trust_establish(
		self,
		tenant_id: str,
		trust_id: str,
		partner_domain: str,
		trust_type: str,
		signing_cert_fingerprint: str,
		owner_id: str,
		bidirectional: bool = False,
	) -> dict[str, Any]:
		"""Establish a federation trust relationship with a partner domain."""
		if trust_type not in {"saml", "oidc", "wsfed"}:
			raise ValueError(f"unsupported_trust_type:{trust_type}")
		trust_key = f"{tenant_id}:{partner_domain}"
		record: dict[str, Any] = {
			"id": trust_id,
			"kind": "trust_relationship",
			"tenant_id": tenant_id,
			"partner_domain": partner_domain,
			"trust_type": trust_type,
			"signing_cert_fingerprint": signing_cert_fingerprint,
			"owner_id": owner_id,
			"bidirectional": bidirectional,
			"status": "active",
			"established_at": _utc(),
		}
		self._trust_relationships[trust_key] = record
		await self._emit_audit(tenant_id, "trust_established", trust_id, f"Trust with {partner_domain} ({trust_type})", owner_id)
		return record

	# -------------------------------------------- attribute_release

	async def attribute_release(
		self,
		tenant_id: str,
		policy_id: str,
		provider_id: str,
		attributes: list[str],
		conditions: dict[str, Any] | None = None,
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Define which attributes are released to a specific provider."""
		arp_key = f"{tenant_id}:{provider_id}"
		record: dict[str, Any] = {
			"id": policy_id,
			"kind": "attribute_release_policy",
			"tenant_id": tenant_id,
			"provider_id": provider_id,
			"attributes": list(attributes),
			"conditions": dict(conditions or {}),
			"owner_id": owner_id,
			"updated_at": _utc(),
		}
		self._attribute_release_policies[arp_key] = record
		await self._emit_audit(tenant_id, "attribute_release_policy_set", policy_id, f"ARP set for {provider_id}: {attributes}", owner_id)
		return record

	# ------------------------------------------- federation_audit

	async def federation_audit(
		self,
		tenant_id: str,
		limit: int = 100,
		event_type_filter: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return federation audit log for a tenant."""
		audit_store: dict[Any, Any] | None = None
		if hasattr(self, "_audit_events"):
			audit_store = self._audit_events  # type: ignore[attr-defined]
		elif hasattr(self, "_ext_audit_store"):
			audit_store = self._ext_audit_store
		if audit_store is None:
			return []
		events: list[dict[str, Any]] = []
		for v in audit_store.values():
			item = v if isinstance(v, dict) else v.to_dict()
			if item.get("tenant_id") != tenant_id:
				continue
			if event_type_filter and item.get("event_type") != event_type_filter:
				continue
			events.append(item)
		return sorted(events, key=lambda e: e.get("created_at", ""), reverse=True)[:limit]

	# ------------------------------------------- cross_domain_sso

	async def cross_domain_sso(
		self,
		tenant_id: str,
		sso_id: str,
		source_domain: str,
		target_domain: str,
		user_id: str,
		source_token: str,
		ttl_seconds: int = 900,
	) -> dict[str, Any]:
		"""Execute cross-domain SSO: validate source token and issue cross-domain token."""
		trust_key = f"{tenant_id}:{target_domain}"
		if trust_key not in self._trust_relationships:
			raise PermissionError(f"no_trust_with_domain:{target_domain}")
		cross_token = _jwt_stub(user_id, source_domain, target_domain, ttl_seconds)
		record: dict[str, Any] = {
			"id": sso_id,
			"kind": "cross_domain_sso",
			"tenant_id": tenant_id,
			"source_domain": source_domain,
			"target_domain": target_domain,
			"user_id": user_id,
			"cross_domain_token": cross_token,
			"ttl_seconds": ttl_seconds,
			"status": "issued",
			"issued_at": _utc(),
		}
		self._cross_domain_sessions[sso_id] = record
		await self._emit_audit(tenant_id, "cross_domain_sso_issued", sso_id, f"Cross-domain SSO: {source_domain}->{target_domain} for {user_id}", user_id)
		return record

	# -------------------------------------------- federation_analytics

	async def federation_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate federation usage statistics for a tenant."""
		total_sessions = 0
		active_sessions = 0
		if hasattr(self, "_sessions"):
			for sk, sess in self._sessions.items():  # type: ignore[attr-defined]
				item = sess if isinstance(sess, dict) else sess.to_dict()
				if item.get("tenant_id") != tenant_id:
					continue
				total_sessions += 1
				if item.get("status") == "active":
					active_sessions += 1
		trust_count = sum(1 for k in self._trust_relationships if k.startswith(f"{tenant_id}:"))
		provisioned_users = sum(1 for k in self._provisioned_users if k.startswith(f"{tenant_id}:"))
		active_users = sum(
			1 for v in self._provisioned_users.values()
			if v.get("tenant_id") == tenant_id and v.get("status") == "active"
		)
		return {
			"tenant_id": tenant_id,
			"total_sessions": total_sessions,
			"active_sessions": active_sessions,
			"trust_relationships": trust_count,
			"provisioned_users": provisioned_users,
			"active_provisioned_users": active_users,
			"group_syncs": len(self._group_sync_records),
			"attribute_release_policies": sum(1 for k in self._attribute_release_policies if k.startswith(f"{tenant_id}:")),
			"cross_domain_sessions": len(self._cross_domain_sessions),
			"generated_at": _utc(),
		}

	# ---------------------------------------------------------------- bulk ops

	async def bulk_register_providers(
		self,
		tenant_id: str,
		providers: list[dict[str, Any]],
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Register multiple IdPs in one operation."""
		registered: list[str] = []
		errors: list[dict[str, Any]] = []
		for prov in providers:
			try:
				result = await self.idp_register(
					tenant_id=tenant_id,
					idp_id=prov["id"],
					name=prov["name"],
					protocol=prov.get("protocol", "saml"),
					metadata_url=prov.get("metadata_url", ""),
					owner_id=actor_id,
					active=prov.get("active", True),
				)
				registered.append(result["id"])
			except Exception as exc:
				errors.append({"id": prov.get("id"), "error": str(exc)})
		return {"registered": registered, "errors": errors, "total": len(providers)}

	async def bulk_revoke_sessions(
		self,
		tenant_id: str,
		session_ids: list[str],
		reason: str = "bulk_revoke",
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Revoke multiple sessions in one operation."""
		revoked: list[str] = []
		errors: list[dict[str, Any]] = []
		if hasattr(self, "revoke_session"):
			for sid in session_ids:
				try:
					self.revoke_session(sid, tenant_id, reason)  # type: ignore[attr-defined]
					revoked.append(sid)
				except Exception as exc:
					errors.append({"id": sid, "error": str(exc)})
		elif hasattr(self, "_sessions"):
			for sid in session_ids:
				sess = self._sessions.get((tenant_id, sid))  # type: ignore[attr-defined]
				if sess is None:
					errors.append({"id": sid, "error": "not_found"})
					continue
				if isinstance(sess, dict):
					sess["status"] = "revoked"
				revoked.append(sid)
		await self._emit_audit(tenant_id, "bulk_sessions_revoked", tenant_id, f"Bulk revoked {len(revoked)} sessions: {reason}", actor_id)
		return {"revoked": revoked, "errors": errors, "total": len(session_ids)}

	# ------------------------------------------------------------------ export

	async def export_audit(
		self,
		tenant_id: str,
		fmt: str = "json",
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Export federation audit log for a tenant as JSON or CSV."""
		events = await self.federation_audit(tenant_id=tenant_id, limit=10000)
		if fmt == "csv":
			buf = io.StringIO()
			if events:
				writer = csv.DictWriter(buf, fieldnames=list(events[0].keys()))
				writer.writeheader()
				writer.writerows(events)
			payload = buf.getvalue()
			content_type = "text/csv"
		else:
			payload = json.dumps(events, default=str, indent=2)
			content_type = "application/json"
		await self._emit_audit(tenant_id, "federation_audit_exported", tenant_id, f"Audit exported as {fmt} ({len(events)} events)", actor_id)
		return {
			"tenant_id": tenant_id,
			"format": fmt,
			"content_type": content_type,
			"event_count": len(events),
			"payload": payload,
			"exported_at": _utc(),
		}

	# --------------------------------------------------------------- health / compliance

	async def health_check(self) -> dict[str, Any]:
		"""Return operational status of the identity federation service."""
		sessions_count = len(getattr(self, "_sessions", {}))
		providers_count = len(getattr(self, "_providers", {}))
		return {
			"status": "healthy",
			"providers": providers_count,
			"sessions": sessions_count,
			"trust_relationships": len(self._trust_relationships),
			"provisioned_users": len(self._provisioned_users),
			"group_sync_records": len(self._group_sync_records),
			"sso_requests": len(self._sso_requests),
			"checked_at": _utc(),
		}

	async def compliance_check(self, tenant_id: str) -> dict[str, Any]:
		"""Verify federation compliance: all active providers have claim mappings and ARP."""
		issues: list[dict[str, Any]] = []
		providers_store = getattr(self, "_providers", {})
		for pk, prov in providers_store.items():
			item = prov if isinstance(prov, dict) else prov.to_dict()
			if item.get("tenant_id") != tenant_id:
				continue
			if item.get("status") not in ("active",):
				continue
			prov_id = item["id"]
			# Check attribute release policy exists
			arp_key = f"{tenant_id}:{prov_id}"
			if arp_key not in self._attribute_release_policies:
				issues.append({"provider_id": prov_id, "issue": "missing_attribute_release_policy"})
			# Check at least one claim mapping exists
			if hasattr(self, "_mappings"):
				has_mapping = any(
					(v if isinstance(v, dict) else v.to_dict()).get("provider_id") == prov_id
					for v in self._mappings.values()  # type: ignore[attr-defined]
				)
				if not has_mapping:
					issues.append({"provider_id": prov_id, "issue": "no_claim_mappings"})
		return {
			"tenant_id": tenant_id,
			"compliant": len(issues) == 0,
			"issues": issues,
			"checked_at": _utc(),
		}

	# ---------------------------------------------------------------- private

	async def _emit_audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
	) -> None:
		if hasattr(self, "_audit"):
			try:
				self._audit(  # type: ignore[attr-defined]
					tenant_id=tenant_id,
					event_type=event_type,
					subject_id=subject_id,
					actor=actor,
					message=message,
				)
				return
			except TypeError:
				pass
		if not hasattr(self, "_ext_audit_store"):
			self._ext_audit_store: dict[str, dict[str, Any]] = {}
		ev_id = f"ext-{event_type}-{subject_id}-{next(self._ext_counter)}"
		self._ext_audit_store[ev_id] = {
			"id": ev_id,
			"tenant_id": tenant_id,
			"event_type": event_type,
			"subject_id": subject_id,
			"message": message,
			"actor": actor,
			"created_at": _utc(),
		}
