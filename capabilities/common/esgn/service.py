"""Service layer for APG Digital Forms and eSign."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	EsgnAuditEvent,
	EvidencePackage,
	FormSubmission,
	FormTemplate,
	SignatureEnvelope,
	SignatureRecipient,
	SigningCeremony,
)
from .signing_engine import SigningEngine


class EsgnService:
	"""Digital form template, submission, envelope, signing, and evidence service."""

	def __init__(self) -> None:
		self._templates: dict[str, FormTemplate] = {}
		self._submissions: dict[str, FormSubmission] = {}
		self._envelopes: dict[str, SignatureEnvelope] = {}
		self._ceremonies: dict[str, SigningCeremony] = {}
		self._evidence_packages: dict[str, EvidencePackage] = {}
		self._audit_events: dict[str, EsgnAuditEvent] = {}
		self._engine = SigningEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_template(
		self,
		template_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		schema_fields: list[str] | tuple[str, ...] | None = None,
		compliance_framework: str = "",
		dlp_policy: str = "",
		retention_policy: str = "",
		regulated_form: bool = False,
		compliance_review_recorded: bool = True,
		schema: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		fields = self._normalize_schema_fields(schema_fields, schema)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_form_template",
			"template_owner_assigned": bool(owner),
			"regulated_form": bool(regulated_form),
			"compliance_review_recorded": bool(compliance_review_recorded),
		})
		self._raise_if_denied(result)
		if not fields:
			raise PermissionError("schema_validation_required")
		if not compliance_framework:
			raise PermissionError("compliance_framework_link_required")
		if regulated_form and not dlp_policy:
			raise PermissionError("regulated_field_dlp_required")
		if not retention_policy:
			raise PermissionError("retention_policy_required")
		review_required = result["decision"] == "require_review"
		status = "pending_review" if review_required else "draft"
		template = FormTemplate(
			id=template_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			schema_fields=tuple(fields),
			schema_digest=self._engine.validation_hash({"fields": fields}, {}),
			compliance_framework=compliance_framework,
			dlp_policy=dlp_policy,
			retention_policy=retention_policy,
			regulated_form=bool(regulated_form),
			status=status,
			review_status="required" if review_required else "approved",
		)
		self._templates[template_id] = template
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=template_id,
			event_type="template_created",
			actor=owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"regulated_form": regulated_form, "field_count": len(fields)},
		)
		return template.to_dict()

	def publish_template(self, template_id: str, tenant_id: str, approved_by: str, publication_approved: bool) -> dict[str, Any]:
		template = self._require_template(template_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_form",
			"publication_approved": bool(publication_approved),
		})
		self._raise_if_denied(result)
		if template.review_status == "required":
			raise PermissionError("compliance_review_required")
		published = FormTemplate(
			id=template.id,
			tenant_id=template.tenant_id,
			name=template.name,
			owner=template.owner,
			schema_fields=template.schema_fields,
			schema_digest=template.schema_digest,
			compliance_framework=template.compliance_framework,
			dlp_policy=template.dlp_policy,
			retention_policy=template.retention_policy,
			regulated_form=template.regulated_form,
			status="published",
			review_status=template.review_status,
		)
		self._templates[template_id] = published
		self._record_audit(tenant_id, template_id, "template_published", approved_by, result["decision"], metadata={"approved": publication_approved})
		return published.to_dict()

	def submit_form(
		self,
		submission_id: str,
		tenant_id: str,
		template_id: str,
		submitted_by: str,
		data: dict[str, Any] | None = None,
		evidence_ref: str = "",
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant_context(tenant_id)
		template = self._require_template(template_id, tenant_id)
		if template.status != "published":
			raise PermissionError("form_not_published")
		form_data = dict(data if data is not None else payload or {})
		self._validate_payload(template.schema_fields, form_data)
		if not evidence_ref:
			raise PermissionError("audit_trail_required")
		submission = FormSubmission(
			id=submission_id,
			tenant_id=tenant_id,
			template_id=template_id,
			submitted_by=submitted_by,
			data=form_data,
			validation_hash=self._engine.validation_hash({"fields": list(template.schema_fields)}, form_data),
			evidence_ref=evidence_ref,
		)
		self._submissions[submission_id] = submission
		self._record_audit(tenant_id, submission_id, "form_submitted", submitted_by, "allow", metadata={"template_id": template_id})
		return submission.to_dict()

	def create_envelope(
		self,
		envelope_id: str,
		tenant_id: str,
		submission_id: str,
		subject: str,
		recipients: list[dict[str, Any]],
		sender: str = "system",
		signature_intent: str = "approval",
		compliance_review_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant_context(tenant_id)
		submission = self._require_submission(submission_id, tenant_id)
		template = self._require_template(submission.template_id, tenant_id)
		if not recipients:
			raise PermissionError("recipient_required")
		if not signature_intent:
			raise PermissionError("signature_intent_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"regulated_form": bool(template.regulated_form),
			"compliance_review_recorded": bool(compliance_review_recorded),
		})
		self._raise_if_denied(result)
		recipient_models = tuple(self._recipient_from_payload(item) for item in recipients)
		if not all(recipient.consent_recorded for recipient in recipient_models):
			raise PermissionError("recipient_consent_required")
		for recipient in recipient_models:
			if recipient.role == "delegate" and not recipient.delegated_policy_ref:
				raise PermissionError("delegated_signing_policy_required")
		recipient_dicts = [recipient.to_dict() for recipient in recipient_models]
		status = "review_required" if result["decision"] == "require_review" else "sent"
		envelope = SignatureEnvelope(
			id=envelope_id,
			tenant_id=tenant_id,
			template_id=template.id,
			submission_id=submission_id,
			subject=subject,
			sender=sender,
			recipients=recipient_models,
			tamper_seal=self._engine.tamper_seal(submission.to_dict(), recipient_dicts),
			signature_intent=signature_intent,
			compliance_review_recorded=bool(compliance_review_recorded),
			status=status,
		)
		self._envelopes[envelope_id] = envelope
		self._record_audit(
			tenant_id,
			envelope_id,
			"envelope_sent",
			sender,
			result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"recipient_count": len(recipients), "submission_id": submission_id},
		)
		return envelope.to_dict()

	def sign_envelope(
		self,
		ceremony_id: str,
		tenant_id: str,
		envelope_id: str,
		recipient_id: str,
		signature_intent: str,
		identity_verified: bool,
		signature_intent_recorded: bool = True,
		signed_at: str | None = None,
	) -> dict[str, Any]:
		envelope = self._require_envelope(envelope_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "sign_envelope",
			"identity_verified": bool(identity_verified),
		})
		self._raise_if_denied(result)
		if envelope.status == "review_required":
			raise PermissionError("compliance_review_required")
		if not signature_intent_recorded or not signature_intent:
			raise PermissionError("signature_intent_required")
		if recipient_id not in {recipient.id for recipient in envelope.recipients}:
			raise KeyError(f"unknown recipient: {recipient_id}")
		timestamp = signed_at or datetime.now(timezone.utc).isoformat()
		ceremony = SigningCeremony(
			id=ceremony_id,
			tenant_id=tenant_id,
			envelope_id=envelope_id,
			recipient_id=recipient_id,
			signature_hash=self._engine.signature_hash(envelope_id, recipient_id, signature_intent),
			identity_verified=bool(identity_verified),
			signature_intent=signature_intent,
			signed_at=timestamp,
		)
		self._ceremonies[ceremony_id] = ceremony
		self._refresh_envelope_signing_state(envelope_id, tenant_id, recipient_id)
		self._record_audit(tenant_id, ceremony_id, "envelope_signed", recipient_id, result["decision"], metadata={"envelope_id": envelope_id})
		return ceremony.to_dict()

	def create_evidence_package(
		self,
		evidence_id: str,
		tenant_id: str,
		envelope_id: str,
		encrypted: bool,
		retention_policy: str,
		audit_trail_ref: str = "",
	) -> dict[str, Any]:
		envelope = self._require_envelope(envelope_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"evidence_package_created": True,
			"evidence_encrypted": bool(encrypted),
		})
		self._raise_if_denied(result)
		if envelope.status != "completed":
			raise PermissionError("envelope_not_completed")
		if not audit_trail_ref:
			raise PermissionError("audit_trail_required")
		if not retention_policy:
			raise PermissionError("retention_policy_required")
		ceremonies = [item for item in self.list_ceremonies(tenant_id) if item["envelope_id"] == envelope_id]
		seal_digest = self._engine.evidence_hash(envelope.to_dict(), ceremonies)
		evidence = EvidencePackage(
			id=evidence_id,
			tenant_id=tenant_id,
			envelope_id=envelope_id,
			certificate_id=self._engine.certificate_id(envelope_id, seal_digest),
			audit_trail_ref=audit_trail_ref,
			seal_digest=seal_digest,
			encrypted=bool(encrypted),
			retention_policy=retention_policy,
		)
		self._evidence_packages[evidence_id] = evidence
		self._record_audit(tenant_id, evidence_id, "evidence_package_created", "evidence-vault", result["decision"], metadata={"envelope_id": envelope_id})
		return evidence.to_dict()

	def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._templates, tenant_id)

	def list_submissions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._submissions, tenant_id)

	def list_envelopes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._envelopes, tenant_id)

	def list_ceremonies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._ceremonies, tenant_id)

	def list_evidence_packages(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._evidence_packages, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing submissions as ESGN records."""
		return self.list_submissions(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that creates a published template and submission."""
		metadata = dict(metadata or {})
		template_id = str(metadata.get("template_id") or "template-default")
		if template_id not in self._templates:
			self.create_template(
				template_id=template_id,
				tenant_id=tenant_id,
				name=str(metadata.get("template_name") or "Default form"),
				owner=str(metadata.get("owner") or "forms-admin"),
				schema_fields=list(metadata.get("schema_fields") or ["name"]),
				compliance_framework=str(metadata.get("compliance_framework") or "standard"),
				dlp_policy=str(metadata.get("dlp_policy") or "standard-dlp"),
				retention_policy=str(metadata.get("retention_policy") or "standard-7y"),
			)
			self.publish_template(template_id, tenant_id, "forms-admin", True)
		return self.submit_form(
			submission_id=record_id,
			tenant_id=tenant_id,
			template_id=template_id,
			submitted_by=str(metadata.get("submitted_by") or "system"),
			data=dict(metadata.get("data") or metadata.get("payload") or {"name": status}),
			evidence_ref=str(metadata.get("evidence_ref") or f"audit:{record_id}"),
		)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		envelopes = self.list_envelopes(tenant_id)
		return {
			"template_count": len(self.list_templates(tenant_id)),
			"published_template_count": len([item for item in self.list_templates(tenant_id) if item["status"] == "published"]),
			"submission_count": len(self.list_submissions(tenant_id)),
			"envelope_count": len(envelopes),
			"completed_envelope_count": len([item for item in envelopes if item["status"] == "completed"]),
			"ceremony_count": len(self.list_ceremonies(tenant_id)),
			"evidence_package_count": len(self.list_evidence_packages(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant_context(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _normalize_schema_fields(
		self,
		schema_fields: list[str] | tuple[str, ...] | None,
		schema: dict[str, Any] | None,
	) -> list[str]:
		fields = schema_fields
		if fields is None and schema is not None:
			fields = schema.get("fields")
		return [str(field) for field in fields or [] if str(field)]

	def _validate_payload(self, schema_fields: tuple[str, ...], data: dict[str, Any]) -> None:
		missing = [field for field in schema_fields if field not in data]
		if missing:
			raise PermissionError("schema_validation_required")

	def _recipient_from_payload(self, payload: dict[str, Any]) -> SignatureRecipient:
		return SignatureRecipient(
			id=str(payload["id"]),
			name=str(payload["name"]),
			email=str(payload["email"]),
			role=str(payload.get("role") or "signer"),
			routing_order=int(payload.get("routing_order") or 1),
			consent_recorded=bool(payload.get("consent_recorded", False)),
			delegated_policy_ref=str(payload.get("delegated_policy_ref") or ""),
			identity_verified=bool(payload.get("identity_verified", False)),
		)

	def _refresh_envelope_signing_state(self, envelope_id: str, tenant_id: str, recipient_id: str) -> None:
		envelope = self._require_envelope(envelope_id, tenant_id)
		updated_recipients = tuple(
			SignatureRecipient(
				id=recipient.id,
				name=recipient.name,
				email=recipient.email,
				role=recipient.role,
				routing_order=recipient.routing_order,
				consent_recorded=recipient.consent_recorded,
				delegated_policy_ref=recipient.delegated_policy_ref,
				identity_verified=True if recipient.id == recipient_id else recipient.identity_verified,
				signed=True if recipient.id == recipient_id else recipient.signed,
			)
			for recipient in envelope.recipients
		)
		status = "completed" if all(recipient.signed for recipient in updated_recipients) else "partially_signed"
		refreshed = SignatureEnvelope(
			id=envelope.id,
			tenant_id=envelope.tenant_id,
			template_id=envelope.template_id,
			submission_id=envelope.submission_id,
			subject=envelope.subject,
			sender=envelope.sender,
			recipients=updated_recipients,
			tamper_seal=envelope.tamper_seal,
			signature_intent=envelope.signature_intent,
			compliance_review_recorded=envelope.compliance_review_recorded,
			status=status,
		)
		self._envelopes[envelope_id] = refreshed

	def _list(self, values: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _require_template(self, template_id: str, tenant_id: str) -> FormTemplate:
		template = self._templates.get(template_id)
		if template is None or template.tenant_id != tenant_id:
			raise KeyError(f"unknown form template: {template_id}")
		return template

	def _require_submission(self, submission_id: str, tenant_id: str) -> FormSubmission:
		submission = self._submissions.get(submission_id)
		if submission is None or submission.tenant_id != tenant_id:
			raise KeyError(f"unknown form submission: {submission_id}")
		return submission

	def _require_envelope(self, envelope_id: str, tenant_id: str) -> SignatureEnvelope:
		envelope = self._envelopes.get(envelope_id)
		if envelope is None or envelope.tenant_id != tenant_id:
			raise KeyError(f"unknown signature envelope: {envelope_id}")
		return envelope

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> EsgnAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = EsgnAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[event_id] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "esgn_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "esgn_policy_blocked")
