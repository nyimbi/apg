"""Data Loss Prevention service for the APG DLPD capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .dlp_engine import (
	action_for,
	detect_classifier_hits,
	highest_sensitivity_label,
	highest_severity,
	stable_digest,
)
from .models import (
	DataClassifier,
	DlpAuditEvent,
	DlpIncident,
	DlpPolicy,
	EgressInspection,
	QuarantineItem,
	utc_now,
)


StoreKey = tuple[str, str]


class DlpdService:
	"""Tenant-scoped DLP policy, classifier, inspection, incident, and quarantine service."""

	def __init__(self) -> None:
		self._policies: dict[StoreKey, DlpPolicy] = {}
		self._classifiers: dict[StoreKey, DataClassifier] = {}
		self._inspections: dict[StoreKey, EgressInspection] = {}
		self._quarantine: dict[StoreKey, QuarantineItem] = {}
		self._incidents: dict[StoreKey, DlpIncident] = {}
		self._audit_events: list[DlpAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_policy(
		self,
		policy_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		channels: list[str],
		classifiers: list[str],
		default_action: str = "quarantine",
		egress_policy_attached: bool = True,
		large_export_review_required: bool = True,
	) -> dict[str, Any]:
		self._ensure_new(self._policies, tenant_id, policy_id)
		self._require_tenant(tenant_id)
		self._raise_if_denied(self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_policy",
			"owner_present": bool(owner),
			"channels_present": bool(channels),
			"classifiers_present": bool(classifiers),
			"egress_policy_attached": egress_policy_attached,
		}))
		for classifier_id in classifiers:
			self._require_classifier(classifier_id, tenant_id)
		if default_action not in {"allow", "alert", "block", "quarantine"}:
			raise ValueError("default_action must be one of allow, alert, block, quarantine")
		policy = DlpPolicy(
			id=policy_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			channels=list(channels),
			classifiers=list(classifiers),
			default_action=default_action,
			egress_policy_attached=egress_policy_attached,
			large_export_review_required=large_export_review_required,
		)
		self._policies[self._key(tenant_id, policy_id)] = policy
		self._record_audit(tenant_id, "policy_registered", policy_id, owner, policy.to_dict())
		return policy.to_dict()

	def register_classifier(
		self,
		classifier_id: str,
		tenant_id: str,
		name: str,
		classifier_type: str,
		sensitivity_label: str,
		pattern_keys: list[str],
		reviewed_by: str | None = None,
		confidence_threshold: float | None = None,
	) -> dict[str, Any]:
		self._ensure_new(self._classifiers, tenant_id, classifier_id)
		self._require_tenant(tenant_id)
		self._raise_if_denied(self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_classifier",
			"classifier_type": classifier_type,
			"sensitivity_label_present": bool(sensitivity_label),
			"pattern_keys_present": bool(pattern_keys),
			"classifier_review_recorded": bool(reviewed_by) or classifier_type != "custom",
		}))
		classifier = DataClassifier(
			id=classifier_id,
			tenant_id=tenant_id,
			name=name,
			classifier_type=classifier_type,
			sensitivity_label=sensitivity_label,
			pattern_keys=list(pattern_keys),
			reviewed_by=reviewed_by,
			confidence_threshold=confidence_threshold or DEFAULT_CONFIGURATION["data_patterns"]["minimum_classifier_confidence"],
		)
		self._classifiers[self._key(tenant_id, classifier_id)] = classifier
		self._record_audit(tenant_id, "classifier_registered", classifier_id, reviewed_by or "system", classifier.to_dict())
		return classifier.to_dict()

	def classify_content(
		self,
		tenant_id: str,
		content: str,
		classifier_ids: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		pattern_keys = self._pattern_keys_for(tenant_id, classifier_ids)
		hits = detect_classifier_hits(content, pattern_keys)
		confidence = max([hit["confidence"] for hit in hits], default=1.0)
		self._raise_if_denied(self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "classify_content",
			"classifier_confidence": confidence,
		}))
		return {
			"tenant_id": tenant_id,
			"content_hash": stable_digest(content),
			"classifier_hits": hits,
			"sensitive_content_detected": bool(hits),
			"classification_label": highest_sensitivity_label(hits),
			"severity": highest_severity(hits),
		}

	def inspect_egress(
		self,
		inspection_id: str,
		tenant_id: str,
		policy_id: str,
		channel: str,
		subject_id: str,
		destination: str,
		content: str,
		record_count: int = 1,
		classification_label: str | None = None,
		auto_classify: bool = True,
		review_recorded: bool = False,
		quarantine_encrypted: bool = True,
	) -> dict[str, Any]:
		self._ensure_new(self._inspections, tenant_id, inspection_id)
		self._require_tenant(tenant_id)
		policy = self._require_policy(policy_id, tenant_id)
		if channel not in policy.channels:
			raise PermissionError("channel_not_covered_by_policy")
		classification = self.classify_content(tenant_id, content, policy.classifiers)
		hits = classification["classifier_hits"]
		hit_names = {hit["classifier"] for hit in hits}
		effective_label = classification_label or (classification["classification_label"] if auto_classify else None)
		severity = classification["severity"]
		review_required = (
			policy.large_export_review_required
			and record_count > DEFAULT_CONFIGURATION["channels"]["bulk_export_threshold_records"]
			and not review_recorded
		)
		action = action_for(policy.default_action, severity, review_required)
		blocked = action == "block"
		quarantined = action == "quarantine"
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "inspect_egress",
			"egress_policy_attached": policy.egress_policy_attached,
			"policy_active": policy.status == "active",
			"channel_covered": channel in policy.channels,
			"destination_present": bool(destination),
			"sensitive_content_detected": bool(hits),
			"classification_label_present": bool(effective_label),
			"severity": severity,
			"blocked_or_quarantined": blocked or quarantined,
			"alerted_or_quarantined": action in {"alert", "block", "quarantine"},
			"secret_detected": "secrets" in hit_names,
			"source_code_detected": "source_code" in hit_names,
			"quarantine_requested": quarantined,
			"quarantine_encrypted": quarantine_encrypted,
			"export_record_count": record_count,
			"review_recorded": review_recorded,
		})
		self._raise_if_denied(result)
		inspection = EgressInspection(
			id=inspection_id,
			tenant_id=tenant_id,
			policy_id=policy_id,
			channel=channel,
			subject_id=subject_id,
			destination=destination,
			content_hash=classification["content_hash"],
			classification_label=effective_label,
			classifier_hits=hits,
			severity=severity,
			record_count=record_count,
			decision=result["decision"] if result["decision"] == "require_review" else action,
			blocked=blocked,
			quarantined=quarantined,
			review_required=result["decision"] == "require_review",
			reviewed_by="reviewed" if review_recorded else None,
		)
		self._inspections[self._key(tenant_id, inspection_id)] = inspection
		if quarantined:
			quarantine = self._create_quarantine_item(inspection, quarantine_encrypted)
			inspection.quarantine_id = quarantine.id
		if blocked or quarantined:
			incident = self._open_incident(inspection, policy, result)
			inspection.incident_id = incident.id
		self._record_audit(tenant_id, "egress_inspected", inspection_id, subject_id, inspection.to_dict())
		return inspection.to_dict()

	def review_export(self, inspection_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
		inspection = self._require_inspection(inspection_id, tenant_id)
		if not inspection.review_required:
			raise ValueError("inspection_does_not_require_review")
		self._raise_if_denied(self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "review_export",
			"reviewer_same_as_subject": reviewer == inspection.subject_id,
			"notes_present": True,
		}))
		inspection.review_required = False
		inspection.reviewed_by = reviewer
		inspection.decision = "reviewed"
		self._record_audit(tenant_id, "large_export_reviewed", inspection_id, reviewer, inspection.to_dict())
		return inspection.to_dict()

	def resolve_incident(self, incident_id: str, tenant_id: str, actor: str, resolution: str) -> dict[str, Any]:
		incident = self._require_incident(incident_id, tenant_id)
		self._raise_if_denied(self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "resolve_incident",
			"resolution_present": bool(resolution),
		}))
		incident.status = "resolved"
		incident.resolution = resolution
		incident.resolved_at = utc_now()
		self._record_audit(tenant_id, "incident_resolved", incident_id, actor, incident.to_dict())
		return incident.to_dict()

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		inspections = self.list_inspections(tenant_id)
		incidents = self.list_incidents(tenant_id)
		return {
			"tenant_id": tenant_id,
			"policy_count": len(self.list_policies(tenant_id)),
			"classifier_count": len(self.list_classifiers(tenant_id)),
			"inspection_count": len(inspections),
			"blocked_count": sum(1 for item in inspections if item["blocked"]),
			"quarantine_count": len(self.list_quarantine(tenant_id)),
			"open_incident_count": sum(1 for item in incidents if item["status"] == "open"),
			"review_required_count": sum(1 for item in inspections if item["review_required"]),
		}

	def list_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._policies, tenant_id)

	def list_classifiers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._classifiers, tenant_id)

	def list_inspections(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._inspections, tenant_id)

	def list_quarantine(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._quarantine, tenant_id)

	def list_incidents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._incidents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	def _pattern_keys_for(self, tenant_id: str, classifier_ids: list[str] | None) -> list[str]:
		if classifier_ids:
			classifiers = [self._require_classifier(classifier_id, tenant_id) for classifier_id in classifier_ids]
			keys = [key for classifier in classifiers for key in classifier.pattern_keys]
		else:
			keys = list(DEFAULT_CONFIGURATION["data_patterns"]["enabled_classifiers"])
		return list(dict.fromkeys(keys))

	def _create_quarantine_item(self, inspection: EgressInspection, encrypted: bool) -> QuarantineItem:
		result = self.evaluate({
			"tenant_context_present": bool(inspection.tenant_id),
			"operation": "create_quarantine_item",
			"quarantine_requested": True,
			"quarantine_encrypted": encrypted,
			"content_hash_present": bool(inspection.content_hash),
		})
		self._raise_if_denied(result)
		item = QuarantineItem(
			id=f"qrn-{inspection.id}",
			tenant_id=inspection.tenant_id,
			inspection_id=inspection.id,
			content_hash=inspection.content_hash,
			reason=inspection.severity,
			encrypted=encrypted,
			legal_hold=DEFAULT_CONFIGURATION["governance"]["legal_hold_supported"],
		)
		self._quarantine[self._key(inspection.tenant_id, item.id)] = item
		self._record_audit(inspection.tenant_id, "content_quarantined", item.id, inspection.subject_id, item.to_dict())
		return item

	def _open_incident(self, inspection: EgressInspection, policy: DlpPolicy, result: dict[str, Any]) -> DlpIncident:
		self._raise_if_denied(self.evaluate({
			"tenant_context_present": bool(inspection.tenant_id),
			"operation": "open_incident",
			"owner_present": bool(policy.owner),
			"severity_present": bool(inspection.severity),
			"duplicate_open_incident": False,
			"notification_sent": DEFAULT_CONFIGURATION["response"]["notification_required"],
		}))
		required_action = ",".join(action.get("required_action", "respond") for action in result["actions"]) or inspection.decision
		incident = DlpIncident(
			id=f"inc-{inspection.id}",
			tenant_id=inspection.tenant_id,
			inspection_id=inspection.id,
			severity=inspection.severity,
			owner=policy.owner,
			required_action=required_action,
			notifications_sent=DEFAULT_CONFIGURATION["response"]["notification_required"],
		)
		self._incidents[self._key(inspection.tenant_id, incident.id)] = incident
		self._record_audit(inspection.tenant_id, "incident_opened", incident.id, policy.owner, incident.to_dict())
		return incident

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_policy(self, policy_id: str, tenant_id: str) -> DlpPolicy:
		policy = self._policies.get(self._key(tenant_id, policy_id))
		if policy is None:
			self._raise_cross_tenant_if_present(self._policies, policy_id, tenant_id)
			raise KeyError(f"unknown_policy:{policy_id}")
		return policy

	def _require_classifier(self, classifier_id: str, tenant_id: str) -> DataClassifier:
		classifier = self._classifiers.get(self._key(tenant_id, classifier_id))
		if classifier is None:
			self._raise_cross_tenant_if_present(self._classifiers, classifier_id, tenant_id)
			raise KeyError(f"unknown_classifier:{classifier_id}")
		return classifier

	def _require_inspection(self, inspection_id: str, tenant_id: str) -> EgressInspection:
		inspection = self._inspections.get(self._key(tenant_id, inspection_id))
		if inspection is None:
			self._raise_cross_tenant_if_present(self._inspections, inspection_id, tenant_id)
			raise KeyError(f"unknown_inspection:{inspection_id}")
		return inspection

	def _require_incident(self, incident_id: str, tenant_id: str) -> DlpIncident:
		incident = self._incidents.get(self._key(tenant_id, incident_id))
		if incident is None:
			self._raise_cross_tenant_if_present(self._incidents, incident_id, tenant_id)
			raise KeyError(f"unknown_incident:{incident_id}")
		return incident

	def _record_audit(self, tenant_id: str, action: str, resource_id: str, actor: str, metadata: dict[str, Any]) -> None:
		payload = {
			"tenant_id": tenant_id,
			"action": action,
			"resource_id": resource_id,
			"actor": actor,
			"metadata": metadata,
		}
		self._audit_events.append(DlpAuditEvent(
			id=f"aud-{len(self._audit_events) + 1:06d}",
			tenant_id=tenant_id,
			action=action,
			resource_id=resource_id,
			actor=actor,
			digest=stable_digest(payload),
			metadata=dict(metadata),
		))

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(action.get("reason", "dlp_policy_blocked") for action in result["actions"]))

	def _list_for_tenant(self, records: dict[StoreKey, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _ensure_new(self, records: dict[StoreKey, Any], tenant_id: str, record_id: str) -> None:
		if not record_id:
			raise ValueError("dlp_record_id_required")
		if self._key(tenant_id, record_id) in records:
			raise ValueError(f"dlp_record_already_exists:{record_id}")

	def _raise_cross_tenant_if_present(self, records: dict[StoreKey, Any], record_id: str, tenant_id: str) -> None:
		if any(record.id == record_id and record.tenant_id != tenant_id for record in records.values()):
			result = self.evaluate({"tenant_context_present": bool(tenant_id), "cross_tenant_access": True})
			raise PermissionError(", ".join(action.get("reason", "cross_tenant_dlp_access_denied") for action in result["actions"]))

	def _key(self, tenant_id: str, record_id: str) -> StoreKey:
		return (tenant_id, record_id)
