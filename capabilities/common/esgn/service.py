"""APG Digital Forms and eSign Service — expanded async runtime (42+ methods).

All state in _Store. Every mutation emits an audit event.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac as _hmac
import io
import json
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:
		return str(uuid.uuid4())

import logging
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

logger = logging.getLogger(__name__)

SUPPORTED_CHANNELS: set[str] = {"email", "sms", "webhook", "audit_log"}


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(v: str) -> str:
	return str(v or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_future(ts: str) -> bool:
	try:
		dt = datetime.fromisoformat(ts)
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=timezone.utc)
		return dt > datetime.now(timezone.utc)
	except Exception:
		return False


def _is_expired(ts: str) -> bool:
	try:
		dt = datetime.fromisoformat(ts)
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=timezone.utc)
		return dt <= datetime.now(timezone.utc)
	except Exception:
		return True


class _Store:
	def __init__(self) -> None:
		self._data: dict[str, dict[str, Any]] = {}

	async def put(self, col: str, rec: dict[str, Any]) -> dict[str, Any]:
		self._data.setdefault(col, {})[rec["id"]] = rec
		return rec

	async def get(self, col: str, rid: str) -> dict[str, Any] | None:
		return self._data.get(col, {}).get(rid)

	async def list(self, col: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._data.get(col, {}).values())
		if tenant_id is not None:
			items = [i for i in items if i.get("tenant_id") == tenant_id]
		return sorted(items, key=lambda i: i.get("id", ""))

	async def delete(self, col: str, rid: str) -> bool:
		bucket = self._data.get(col, {})
		if rid in bucket:
			del bucket[rid]
			return True
		return False


class _Audit:
	def __init__(self, store: _Store) -> None:
		self._store = store

	async def log_event(self, event_type: str, actor_id: str, tenant_id: str, subject_id: str,
						details: dict[str, Any] | None = None, severity: str = "info") -> dict[str, Any]:
		rec = {
			"id": uuid7str(), "tenant_id": tenant_id, "event_type": event_type,
			"actor_id": actor_id, "subject_id": subject_id, "severity": severity,
			"details": details or {}, "recorded_at": _utc_now(),
		}
		await self._store.put("esgn_audit", rec)
		return rec


class _Notify:
	async def send(self, recipient: str, channel: str, subject: str, body: str) -> dict[str, Any]:
		if channel not in SUPPORTED_CHANNELS:
			raise ValueError(f"unsupported_channel:{channel}")
		return {"id": uuid7str(), "recipient": recipient, "channel": channel, "subject": subject, "sent_at": _utc_now()}


class EsgnService:
	"""Async Digital Forms and eSign service — 42+ methods."""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id
		self._store = _Store()
		self._audit = _Audit(self._store)
		self._notify = _Notify()

	# ------------------------------------------------------------------
	# 1. form_create
	# ------------------------------------------------------------------
	async def form_create(
		self,
		tenant_id: str,
		template_id: str,
		name: str,
		owner: str,
		schema_fields: list[str],
		compliance_framework: str = "",
		dlp_policy: str = "standard-dlp",
		retention_policy: str = "standard-7y",
		regulated_form: bool = False,
	) -> dict[str, Any]:
		"""Create a form template."""
		assert name and owner and schema_fields, "name, owner, schema_fields required"
		digest = hashlib.sha256(json.dumps(schema_fields, sort_keys=True).encode()).hexdigest()
		record = {
			"id": template_id, "tenant_id": tenant_id, "name": name, "owner": owner,
			"schema_fields": schema_fields, "schema_digest": digest,
			"compliance_framework": compliance_framework, "dlp_policy": dlp_policy,
			"retention_policy": retention_policy, "regulated_form": regulated_form,
			"status": "draft", "review_status": "approved", "created_at": _utc_now(),
		}
		await self._store.put("esgn_templates", record)
		await self._audit.log_event("form_created", self.actor_id, tenant_id, template_id, {"name": name, "field_count": len(schema_fields)})
		return record

	# ------------------------------------------------------------------
	# 2. form_publish
	# ------------------------------------------------------------------
	async def form_publish(
		self,
		tenant_id: str,
		template_id: str,
		approved_by: str,
	) -> dict[str, Any]:
		"""Publish a form template so it can receive submissions."""
		template = await self._require_template(tenant_id, template_id)
		if template["status"] == "published":
			return template
		template["status"] = "published"
		template["published_by"] = approved_by
		template["published_at"] = _utc_now()
		await self._store.put("esgn_templates", template)
		await self._audit.log_event("form_published", self.actor_id, tenant_id, template_id, {"approved_by": approved_by})
		return template

	# ------------------------------------------------------------------
	# 3. form_submit
	# ------------------------------------------------------------------
	async def form_submit(
		self,
		tenant_id: str,
		submission_id: str,
		template_id: str,
		submitted_by: str,
		data: dict[str, Any],
		evidence_ref: str = "",
	) -> dict[str, Any]:
		"""Submit a form."""
		template = await self._require_template(tenant_id, template_id)
		if template["status"] != "published":
			raise PermissionError("form_not_published")
		missing = [f for f in template["schema_fields"] if f not in data]
		if missing:
			raise ValueError(f"missing_required_fields:{missing}")
		record = {
			"id": submission_id, "tenant_id": tenant_id, "template_id": template_id,
			"submitted_by": submitted_by, "data": data, "evidence_ref": evidence_ref,
			"validation_hash": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
			"status": "submitted", "submitted_at": _utc_now(),
		}
		await self._store.put("esgn_submissions", record)
		await self._audit.log_event("form_submitted", self.actor_id, tenant_id, submission_id, {"template_id": template_id})
		return record

	# ------------------------------------------------------------------
	# 4. signature_request
	# ------------------------------------------------------------------
	async def signature_request(
		self,
		tenant_id: str,
		envelope_id: str,
		submission_id: str,
		subject: str,
		sender: str,
		recipients: list[dict[str, Any]],
		expires_days: int = 30,
		document_hash: str = "",
	) -> dict[str, Any]:
		"""Create a signature envelope and dispatch to recipients."""
		submission = await self._require_submission(tenant_id, submission_id)
		expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()
		tamper_seal = _hmac.new(
			hashlib.sha256((submission_id + envelope_id).encode()).digest(),
			json.dumps(recipients, sort_keys=True, default=str).encode(),
			hashlib.sha256,
		).hexdigest()
		record = {
			"id": envelope_id, "tenant_id": tenant_id, "submission_id": submission_id,
			"subject": subject, "sender": sender, "recipients": recipients,
			"document_hash": document_hash, "expires_at": expires_at,
			"tamper_seal": tamper_seal, "signature_intent": "approval",
			"status": "sent", "created_at": _utc_now(),
		}
		await self._store.put("esgn_envelopes", record)
		await self._audit.log_event("signature_requested", self.actor_id, tenant_id, envelope_id, {"recipient_count": len(recipients)})
		for r in recipients:
			await self._notify.send(r.get("email", ""), "email", f"Signature request: {subject}", f"Please sign envelope {envelope_id}")
		return record

	# ------------------------------------------------------------------
	# 5. sign_document
	# ------------------------------------------------------------------
	async def sign_document(
		self,
		tenant_id: str,
		ceremony_id: str,
		envelope_id: str,
		recipient_id: str,
		signature_intent: str,
		identity_verified: bool,
	) -> dict[str, Any]:
		"""Record a signing ceremony for a recipient."""
		envelope = await self._require_envelope(tenant_id, envelope_id)
		if envelope["status"] not in {"sent", "partially_signed"}:
			raise PermissionError("envelope_not_signable")
		if _is_expired(envelope["expires_at"]):
			raise PermissionError("envelope_expired")
		sig_hash = hashlib.sha256(f"{envelope_id}:{recipient_id}:{signature_intent}".encode()).hexdigest()
		ceremony = {
			"id": ceremony_id, "tenant_id": tenant_id, "envelope_id": envelope_id,
			"recipient_id": recipient_id, "signature_hash": sig_hash,
			"identity_verified": identity_verified, "signature_intent": signature_intent,
			"signed_at": _utc_now(),
		}
		await self._store.put("esgn_ceremonies", ceremony)
		# Update recipient signed status
		recipients = [
			{**r, "signed": True, "identity_verified": True} if r["id"] == recipient_id else r
			for r in envelope["recipients"]
		]
		all_signed = all(r.get("signed") for r in recipients)
		envelope["recipients"] = recipients
		envelope["status"] = "completed" if all_signed else "partially_signed"
		await self._store.put("esgn_envelopes", envelope)
		await self._audit.log_event("document_signed", self.actor_id, tenant_id, ceremony_id, {"envelope_id": envelope_id, "recipient_id": recipient_id})
		return ceremony

	# ------------------------------------------------------------------
	# 6. verify_signature
	# ------------------------------------------------------------------
	async def verify_signature(self, tenant_id: str, ceremony_id: str) -> dict[str, Any]:
		"""Verify a signing ceremony record."""
		ceremony = await self._store.get("esgn_ceremonies", ceremony_id)
		if ceremony is None or ceremony["tenant_id"] != tenant_id:
			raise KeyError(f"ceremony_not_found:{ceremony_id}")
		expected = hashlib.sha256(f"{ceremony['envelope_id']}:{ceremony['recipient_id']}:{ceremony['signature_intent']}".encode()).hexdigest()
		valid = ceremony["signature_hash"] == expected
		result = {"ceremony_id": ceremony_id, "valid": valid, "verified_at": _utc_now()}
		await self._audit.log_event("signature_verified", self.actor_id, tenant_id, ceremony_id, {"valid": valid})
		return result

	# ------------------------------------------------------------------
	# 7. witness_add
	# ------------------------------------------------------------------
	async def witness_add(
		self,
		tenant_id: str,
		witness_id: str,
		ceremony_id: str,
		witness_name: str,
		witness_email: str,
		witness_statement: str,
	) -> dict[str, Any]:
		"""Add a witness attestation to a signing ceremony."""
		ceremony = await self._store.get("esgn_ceremonies", ceremony_id)
		if ceremony is None or ceremony["tenant_id"] != tenant_id:
			raise KeyError(f"ceremony_not_found:{ceremony_id}")
		record = {
			"id": witness_id, "tenant_id": tenant_id, "ceremony_id": ceremony_id,
			"witness_name": witness_name, "witness_email": witness_email,
			"witness_statement": witness_statement,
			"attestation_hash": hashlib.sha256(f"{witness_id}:{ceremony_id}:{witness_statement}".encode()).hexdigest(),
			"attested_at": _utc_now(),
		}
		await self._store.put("esgn_witnesses", record)
		await self._audit.log_event("witness_added", self.actor_id, tenant_id, witness_id, {"ceremony_id": ceremony_id})
		return record

	# ------------------------------------------------------------------
	# 8. audit_trail
	# ------------------------------------------------------------------
	async def audit_trail(self, tenant_id: str, envelope_id: str) -> dict[str, Any]:
		"""Return the complete audit trail for an envelope."""
		ceremonies = [c for c in await self._store.list("esgn_ceremonies", tenant_id) if c["envelope_id"] == envelope_id]
		witnesses = [w for w in await self._store.list("esgn_witnesses", tenant_id) if w["ceremony_id"] in {c["id"] for c in ceremonies}]
		events = [e for e in await self._store.list("esgn_audit", tenant_id) if e.get("details", {}).get("envelope_id") == envelope_id or e.get("subject_id") == envelope_id]
		return {
			"envelope_id": envelope_id, "tenant_id": tenant_id,
			"ceremony_count": len(ceremonies), "witness_count": len(witnesses),
			"audit_event_count": len(events),
			"ceremonies": ceremonies, "witnesses": witnesses,
			"audit_events": events, "generated_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 9. form_analytics
	# ------------------------------------------------------------------
	async def form_analytics(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Compute form and signing analytics."""
		templates = await self._store.list("esgn_templates", tenant_id)
		submissions = await self._store.list("esgn_submissions", tenant_id)
		envelopes = await self._store.list("esgn_envelopes", tenant_id)
		ceremonies = await self._store.list("esgn_ceremonies", tenant_id)
		completion_rate = (
			sum(1 for e in envelopes if e["status"] == "completed") / max(len(envelopes), 1) * 100
		)
		return {
			"tenant_id": tenant_id, "period": period,
			"template_count": len(templates),
			"published_templates": sum(1 for t in templates if t["status"] == "published"),
			"submission_count": len(submissions),
			"envelope_count": len(envelopes),
			"completed_envelopes": sum(1 for e in envelopes if e["status"] == "completed"),
			"completion_rate_percent": round(completion_rate, 2),
			"ceremony_count": len(ceremonies),
			"computed_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 10. template_library
	# ------------------------------------------------------------------
	async def template_library(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all published templates available for use."""
		templates = await self._store.list("esgn_templates", tenant_id)
		return [t for t in templates if t["status"] == "published"]

	# ------------------------------------------------------------------
	# 11. conditional_logic
	# ------------------------------------------------------------------
	async def conditional_logic(
		self,
		tenant_id: str,
		rule_id: str,
		template_id: str,
		conditions: list[dict[str, Any]],
		actions: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Attach conditional logic rules to a form template."""
		await self._require_template(tenant_id, template_id)
		record = {
			"id": rule_id, "tenant_id": tenant_id, "template_id": template_id,
			"conditions": conditions, "actions": actions,
			"status": "active", "created_at": _utc_now(),
		}
		await self._store.put("esgn_conditional_rules", record)
		await self._audit.log_event("conditional_logic_added", self.actor_id, tenant_id, rule_id, {"template_id": template_id})
		return record

	# ------------------------------------------------------------------
	# 12. multi_step_workflow
	# ------------------------------------------------------------------
	async def multi_step_workflow(
		self,
		tenant_id: str,
		workflow_id: str,
		name: str,
		steps: list[dict[str, Any]],
		owner: str,
	) -> dict[str, Any]:
		"""Create a multi-step signing workflow."""
		assert steps, "steps required"
		record = {
			"id": workflow_id, "tenant_id": tenant_id, "name": name, "owner": owner,
			"steps": steps, "step_count": len(steps),
			"status": "active", "created_at": _utc_now(),
		}
		await self._store.put("esgn_workflows", record)
		await self._audit.log_event("workflow_created", self.actor_id, tenant_id, workflow_id, {"name": name, "steps": len(steps)})
		return record

	# ------------------------------------------------------------------
	# 13. deadline_reminder
	# ------------------------------------------------------------------
	async def deadline_reminder(
		self,
		tenant_id: str,
		envelope_id: str,
		reminder_message: str = "",
	) -> dict[str, Any]:
		"""Send deadline reminders to pending signers."""
		envelope = await self._require_envelope(tenant_id, envelope_id)
		pending = [r for r in envelope["recipients"] if not r.get("signed")]
		sent = []
		for r in pending:
			await self._notify.send(r.get("email", ""), "email", "Signature reminder", reminder_message or f"Please sign envelope {envelope_id} before {envelope['expires_at']}")
			sent.append(r.get("email", ""))
		result = {"envelope_id": envelope_id, "reminders_sent": len(sent), "recipients": sent, "sent_at": _utc_now()}
		await self._audit.log_event("deadline_reminder_sent", self.actor_id, tenant_id, envelope_id, {"count": len(sent)})
		return result

	# ------------------------------------------------------------------
	# 14. bulk_sign_request
	# ------------------------------------------------------------------
	async def bulk_sign_request(
		self,
		tenant_id: str,
		submission_ids: list[str],
		subject_template: str,
		sender: str,
		recipients: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Send signing requests for multiple submissions in parallel."""
		tasks = [
			self.signature_request(tenant_id, uuid7str(), sid, subject_template.format(sid=sid), sender, recipients)
			for sid in submission_ids
		]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		out = []
		for sid, res in zip(submission_ids, results):
			if isinstance(res, Exception):
				out.append({"submission_id": sid, "status": "failed", "error": str(res)})
			else:
				out.append({**res, "status": "ok"})  # type: ignore[arg-type]
		await self._audit.log_event("bulk_sign_requests_sent", self.actor_id, tenant_id, "bulk", {"count": len(submission_ids)})
		return out

	# ------------------------------------------------------------------
	# 15. legal_review
	# ------------------------------------------------------------------
	async def legal_review(
		self,
		tenant_id: str,
		review_id: str,
		envelope_id: str,
		reviewer: str,
		status: str,
		notes: str,
	) -> dict[str, Any]:
		"""Record a legal review decision for an envelope."""
		assert status in {"approved", "rejected", "pending"}, "invalid status"
		await self._require_envelope(tenant_id, envelope_id)
		record = {
			"id": review_id, "tenant_id": tenant_id, "envelope_id": envelope_id,
			"reviewer": reviewer, "status": status, "notes": notes,
			"reviewed_at": _utc_now(),
		}
		await self._store.put("esgn_legal_reviews", record)
		await self._audit.log_event("legal_review_recorded", self.actor_id, tenant_id, review_id, {"envelope_id": envelope_id, "status": status}, severity="medium")
		return record

	# ------------------------------------------------------------------
	# 16. create_evidence_package
	# ------------------------------------------------------------------
	async def create_evidence_package(
		self,
		tenant_id: str,
		evidence_id: str,
		envelope_id: str,
		encrypted: bool,
		retention_policy: str,
		audit_trail_ref: str = "",
	) -> dict[str, Any]:
		"""Create a tamper-evident evidence package for a completed envelope."""
		envelope = await self._require_envelope(tenant_id, envelope_id)
		ceremonies = [c for c in await self._store.list("esgn_ceremonies", tenant_id) if c["envelope_id"] == envelope_id]
		seal = hashlib.sha256(json.dumps([envelope, ceremonies], sort_keys=True, default=str).encode()).hexdigest()
		record = {
			"id": evidence_id, "tenant_id": tenant_id, "envelope_id": envelope_id,
			"certificate_id": uuid7str(), "audit_trail_ref": audit_trail_ref,
			"seal_digest": seal, "encrypted": encrypted, "retention_policy": retention_policy,
			"created_at": _utc_now(),
		}
		await self._store.put("esgn_evidence_packages", record)
		await self._audit.log_event("evidence_package_created", self.actor_id, tenant_id, evidence_id, {"envelope_id": envelope_id})
		return record

	# ------------------------------------------------------------------
	# 17. cancel_envelope
	# ------------------------------------------------------------------
	async def cancel_envelope(self, tenant_id: str, envelope_id: str, actor: str, reason: str) -> dict[str, Any]:
		"""Cancel a signing envelope."""
		return await self._change_envelope_status(tenant_id, envelope_id, "cancelled", actor, reason, "envelope_cancelled")

	# ------------------------------------------------------------------
	# 18. reject_envelope
	# ------------------------------------------------------------------
	async def reject_envelope(self, tenant_id: str, envelope_id: str, recipient_id: str, reason: str) -> dict[str, Any]:
		"""Reject a signing envelope."""
		return await self._change_envelope_status(tenant_id, envelope_id, "rejected", recipient_id, reason, "envelope_rejected")

	# ------------------------------------------------------------------
	# 19. register_signing_agent
	# ------------------------------------------------------------------
	async def register_signing_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope_ref: str,
		registered_by: str,
		contribution_disclosed: bool,
		purpose: str = "",
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		"""Register an AI signing agent."""
		record = {
			"id": agent_id, "tenant_id": tenant_id, "name": name,
			"runtime": _normalize(runtime), "role": _normalize(role), "scope_ref": scope_ref,
			"registered_by": registered_by, "contribution_disclosed": contribution_disclosed,
			"purpose": purpose, "human_approval_required": human_approval_required,
			"status": "active", "registered_at": _utc_now(),
		}
		await self._store.put("esgn_agents", record)
		await self._audit.log_event("signing_agent_registered", self.actor_id, tenant_id, agent_id, {"role": role})
		return record

	# ------------------------------------------------------------------
	# 20. verify_tamper_seal
	# ------------------------------------------------------------------
	async def verify_tamper_seal(self, tenant_id: str, envelope_id: str) -> bool:
		"""Verify the tamper seal of an envelope."""
		envelope = await self._require_envelope(tenant_id, envelope_id)
		submission = await self._require_submission(tenant_id, envelope["submission_id"])
		expected = _hmac.new(
			hashlib.sha256((envelope["submission_id"] + envelope_id).encode()).digest(),
			json.dumps(envelope["recipients"], sort_keys=True, default=str).encode(),
			hashlib.sha256,
		).hexdigest()
		valid = envelope.get("tamper_seal") == expected
		await self._audit.log_event("tamper_seal_verified", self.actor_id, tenant_id, envelope_id, {"valid": valid})
		return valid

	# ------------------------------------------------------------------
	# 21. template_archive
	# ------------------------------------------------------------------
	async def template_archive(self, tenant_id: str, template_id: str, actor: str) -> dict[str, Any]:
		"""Archive a form template."""
		template = await self._require_template(tenant_id, template_id)
		template["status"] = "archived"
		template["archived_by"] = actor
		template["archived_at"] = _utc_now()
		await self._store.put("esgn_templates", template)
		await self._audit.log_event("template_archived", self.actor_id, tenant_id, template_id, {"actor": actor})
		return template

	# ------------------------------------------------------------------
	# 22. submission_withdraw
	# ------------------------------------------------------------------
	async def submission_withdraw(self, tenant_id: str, submission_id: str, reason: str) -> dict[str, Any]:
		"""Withdraw a submitted form."""
		sub = await self._require_submission(tenant_id, submission_id)
		sub["status"] = "withdrawn"
		sub["withdrawal_reason"] = reason
		sub["withdrawn_at"] = _utc_now()
		await self._store.put("esgn_submissions", sub)
		await self._audit.log_event("submission_withdrawn", self.actor_id, tenant_id, submission_id, {"reason": reason})
		return sub

	# ------------------------------------------------------------------
	# 23. bulk_create_templates
	# ------------------------------------------------------------------
	async def bulk_create_templates(self, tenant_id: str, templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Bulk-create form templates in parallel."""
		tasks = [
			self.form_create(tenant_id, t["template_id"], t["name"], t["owner"], t["schema_fields"])
			for t in templates
		]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		out = []
		for t, res in zip(templates, results):
			if isinstance(res, Exception):
				out.append({"template_id": t["template_id"], "status": "failed", "error": str(res)})
			else:
				out.append({**res, "status": "ok"})  # type: ignore[arg-type]
		return out

	# ------------------------------------------------------------------
	# 24. compliance_check
	# ------------------------------------------------------------------
	async def compliance_check(self, tenant_id: str, framework: str = "eIDAS") -> dict[str, Any]:
		"""Check signing compliance posture against a framework."""
		envelopes = await self._store.list("esgn_envelopes", tenant_id)
		ceremonies = await self._store.list("esgn_ceremonies", tenant_id)
		issues: list[str] = []
		unverified = [c for c in ceremonies if not c.get("identity_verified")]
		if unverified:
			issues.append(f"{len(unverified)}_unverified_signers")
		expired = [e for e in envelopes if e["status"] not in {"completed", "cancelled"} and _is_expired(e.get("expires_at", ""))]
		if expired:
			issues.append(f"{len(expired)}_expired_open_envelopes")
		return {
			"tenant_id": tenant_id, "framework": framework,
			"passed": len(issues) == 0, "issues": issues,
			"envelope_count": len(envelopes), "ceremony_count": len(ceremonies),
			"checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 25. dashboard_summary
	# ------------------------------------------------------------------
	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		templates = await self._store.list("esgn_templates", tenant_id)
		submissions = await self._store.list("esgn_submissions", tenant_id)
		envelopes = await self._store.list("esgn_envelopes", tenant_id)
		ceremonies = await self._store.list("esgn_ceremonies", tenant_id)
		evidence = await self._store.list("esgn_evidence_packages", tenant_id)
		agents = await self._store.list("esgn_agents", tenant_id)
		return {
			"tenant_id": tenant_id,
			"template_count": len(templates),
			"published_templates": sum(1 for t in templates if t["status"] == "published"),
			"submission_count": len(submissions),
			"envelope_count": len(envelopes),
			"completed_envelopes": sum(1 for e in envelopes if e["status"] == "completed"),
			"cancelled_envelopes": sum(1 for e in envelopes if e["status"] == "cancelled"),
			"rejected_envelopes": sum(1 for e in envelopes if e["status"] == "rejected"),
			"ceremony_count": len(ceremonies),
			"evidence_packages": len(evidence),
			"signing_agents": len(agents),
			"audit_events": len(await self._store.list("esgn_audit", tenant_id)),
			"generated_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 26. health_check
	# ------------------------------------------------------------------
	async def health_check(self) -> dict[str, Any]:
		try:
			test_id = f"_health_{uuid7str()}"
			await self.form_create("_health", test_id, "HealthForm", "system", ["name"])
			await self._store.delete("esgn_templates", test_id)
			status = "healthy"
		except Exception as exc:
			status = f"degraded:{exc}"
		return {
			"service": "EsgnService", "status": status,
			"collections": {
				"templates": len(await self._store.list("esgn_templates")),
				"submissions": len(await self._store.list("esgn_submissions")),
				"envelopes": len(await self._store.list("esgn_envelopes")),
				"audit_events": len(await self._store.list("esgn_audit")),
			},
			"checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 27. export_csv
	# ------------------------------------------------------------------
	async def export_csv(self, tenant_id: str, collection: str = "esgn_submissions") -> str:
		records = await self._store.list(collection, tenant_id)
		if not records:
			return ""
		buf = io.StringIO()
		writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
		writer.writeheader()
		writer.writerows(records)
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 28. export_json
	# ------------------------------------------------------------------
	async def export_json(self, tenant_id: str, collection: str = "esgn_submissions") -> str:
		records = await self._store.list(collection, tenant_id)
		return json.dumps(records, indent=2, default=str)

	# ------------------------------------------------------------------
	# 29–42. list helpers
	# ------------------------------------------------------------------
	async def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_templates", tenant_id)

	async def list_submissions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_submissions", tenant_id)

	async def list_envelopes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_envelopes", tenant_id)

	async def list_ceremonies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_ceremonies", tenant_id)

	async def list_evidence_packages(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_evidence_packages", tenant_id)

	async def list_signing_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_agents", tenant_id)

	async def list_workflows(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_workflows", tenant_id)

	async def list_legal_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_legal_reviews", tenant_id)

	async def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_audit", tenant_id)

	async def list_witnesses(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_witnesses", tenant_id)

	async def list_conditional_rules(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_conditional_rules", tenant_id)

	async def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgn_lifecycle_batches", tenant_id)

	# compat
	async def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self.list_submissions(tenant_id)

	# ------------------------------------------------------------------
	# 43. create_template (compat alias)
	# ------------------------------------------------------------------
	async def create_template(
		self,
		template_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		schema_fields: list[str] | tuple[str, ...] | None = None,
		compliance_framework: str = "",
		dlp_policy: str = "standard-dlp",
		retention_policy: str = "standard-7y",
		regulated_form: bool = False,
		compliance_review_recorded: bool = True,
		schema: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		fields = list(schema_fields or (schema or {}).get("fields", []) or ["name"])
		return await self.form_create(tenant_id, template_id, name, owner, fields, compliance_framework, dlp_policy, retention_policy, regulated_form)

	# ------------------------------------------------------------------
	# 44. publish_template (compat alias)
	# ------------------------------------------------------------------
	async def publish_template(self, template_id: str, tenant_id: str, approved_by: str, publication_approved: bool) -> dict[str, Any]:
		return await self.form_publish(tenant_id, template_id, approved_by)

	# ------------------------------------------------------------------
	# 45. submit_form (compat alias)
	# ------------------------------------------------------------------
	async def submit_form(
		self,
		submission_id: str,
		tenant_id: str,
		template_id: str,
		submitted_by: str,
		data: dict[str, Any] | None = None,
		evidence_ref: str = "",
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		return await self.form_submit(tenant_id, submission_id, template_id, submitted_by, data or payload or {}, evidence_ref)

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------

	async def _require_template(self, tenant_id: str, template_id: str) -> dict[str, Any]:
		rec = await self._store.get("esgn_templates", template_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"template_not_found:{template_id}")
		return rec

	async def _require_submission(self, tenant_id: str, submission_id: str) -> dict[str, Any]:
		rec = await self._store.get("esgn_submissions", submission_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"submission_not_found:{submission_id}")
		return rec

	async def _require_envelope(self, tenant_id: str, envelope_id: str) -> dict[str, Any]:
		rec = await self._store.get("esgn_envelopes", envelope_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"envelope_not_found:{envelope_id}")
		return rec

	async def _change_envelope_status(
		self, tenant_id: str, envelope_id: str, status: str,
		actor: str, reason: str, event_type: str,
	) -> dict[str, Any]:
		envelope = await self._require_envelope(tenant_id, envelope_id)
		if envelope["status"] in {"completed", "cancelled", "rejected"}:
			raise PermissionError("envelope_already_finalized")
		envelope["status"] = status
		envelope["state_reason"] = reason
		envelope["updated_at"] = _utc_now()
		await self._store.put("esgn_envelopes", envelope)
		await self._audit.log_event(event_type, self.actor_id, tenant_id, envelope_id, {"reason": reason, "actor": actor})
		return envelope


__all__ = ["EsgnService"]
