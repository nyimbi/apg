"""Domain models for APG Digital Forms and eSign."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FormTemplate:
	"""Governed form template with schema, ownership, and compliance controls."""

	id: str
	tenant_id: str
	name: str
	owner: str
	schema_fields: tuple[str, ...]
	schema_digest: str
	compliance_framework: str
	dlp_policy: str
	retention_policy: str
	regulated_form: bool = False
	status: str = "draft"
	review_status: str = "approved"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"schema_fields": list(self.schema_fields),
			"schema_digest": self.schema_digest,
			"compliance_framework": self.compliance_framework,
			"dlp_policy": self.dlp_policy,
			"retention_policy": self.retention_policy,
			"regulated_form": self.regulated_form,
			"status": self.status,
			"review_status": self.review_status,
		}


@dataclass(frozen=True)
class FormSubmission:
	"""Validated submission captured against a published form template."""

	id: str
	tenant_id: str
	template_id: str
	submitted_by: str
	data: dict[str, Any]
	validation_hash: str
	evidence_ref: str
	status: str = "submitted"
	validation_status: str = "valid"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"template_id": self.template_id,
			"submitted_by": self.submitted_by,
			"data": dict(self.data),
			"validation_hash": self.validation_hash,
			"evidence_ref": self.evidence_ref,
			"status": self.status,
			"validation_status": self.validation_status,
		}


@dataclass(frozen=True)
class SignatureRecipient:
	"""Recipient in an envelope signing route."""

	id: str
	name: str
	email: str
	role: str
	routing_order: int
	consent_recorded: bool
	delegated_policy_ref: str = ""
	identity_verified: bool = False
	signed: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"name": self.name,
			"email": self.email,
			"role": self.role,
			"routing_order": self.routing_order,
			"consent_recorded": self.consent_recorded,
			"delegated_policy_ref": self.delegated_policy_ref,
			"identity_verified": self.identity_verified,
			"signed": self.signed,
		}


@dataclass(frozen=True)
class SignatureEnvelope:
	"""Envelope that binds a form submission to one or more recipients."""

	id: str
	tenant_id: str
	template_id: str
	submission_id: str
	subject: str
	sender: str
	recipients: tuple[SignatureRecipient, ...]
	tamper_seal: str
	signature_intent: str
	compliance_review_recorded: bool
	status: str = "sent"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"template_id": self.template_id,
			"submission_id": self.submission_id,
			"subject": self.subject,
			"sender": self.sender,
			"recipients": [recipient.to_dict() for recipient in self.recipients],
			"tamper_seal": self.tamper_seal,
			"signature_intent": self.signature_intent,
			"compliance_review_recorded": self.compliance_review_recorded,
			"status": self.status,
		}


@dataclass(frozen=True)
class SigningCeremony:
	"""Single signer ceremony with identity, intent, and seal evidence."""

	id: str
	tenant_id: str
	envelope_id: str
	recipient_id: str
	signature_hash: str
	identity_verified: bool
	signature_intent: str
	signed_at: str
	status: str = "signed"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"envelope_id": self.envelope_id,
			"recipient_id": self.recipient_id,
			"signature_hash": self.signature_hash,
			"identity_verified": self.identity_verified,
			"signature_intent": self.signature_intent,
			"signed_at": self.signed_at,
			"status": self.status,
		}


@dataclass(frozen=True)
class EvidencePackage:
	"""Encrypted evidence bundle for a completed envelope."""

	id: str
	tenant_id: str
	envelope_id: str
	certificate_id: str
	audit_trail_ref: str
	seal_digest: str
	encrypted: bool
	retention_policy: str
	status: str = "sealed"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"envelope_id": self.envelope_id,
			"certificate_id": self.certificate_id,
			"audit_trail_ref": self.audit_trail_ref,
			"seal_digest": self.seal_digest,
			"encrypted": self.encrypted,
			"retention_policy": self.retention_policy,
			"status": self.status,
		}


@dataclass(frozen=True)
class EsgnAuditEvent:
	"""Governance event emitted by digital-form and e-sign operations."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
		}
