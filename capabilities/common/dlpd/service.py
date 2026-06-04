"""Data Loss Prevention service for the APG DLPD capability."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
	DlpAgentRecord,
	DlpAuditEvent,
	DlpIncident,
	DlpPolicy,
	DlpdLifecycleBatchRecord,
	EgressInspection,
	QuarantineItem,
	utc_now,
)


StoreKey = tuple[str, str]


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DlpdService:
	"""Tenant-scoped DLP policy, classifier, inspection, incident, quarantine, and analytics service."""

	def __init__(self) -> None:
		self._policies: dict[StoreKey, DlpPolicy] = {}
		self._classifiers: dict[StoreKey, DataClassifier] = {}
		self._inspections: dict[StoreKey, EgressInspection] = {}
		self._quarantine: dict[StoreKey, QuarantineItem] = {}
		self._incidents: dict[StoreKey, DlpIncident] = {}
		self._dlp_agents: dict[StoreKey, DlpAgentRecord] = {}
		self._lifecycle_batches: dict[StoreKey, DlpdLifecycleBatchRecord] = {}
		self._audit_events: list[DlpAuditEvent] = []
		# Extended stores
		self._regex_patterns: dict[StoreKey, dict[str, Any]] = {}       # pattern library
		self._false_positives: dict[StoreKey, dict[str, Any]] = {}      # feedback records
		self._shadow_it: dict[StoreKey, dict[str, Any]] = {}            # shadow IT detections
		self._cloud_activities: dict[StoreKey, dict[str, Any]] = {}     # cloud activity events
		self._bulk_scan_jobs: dict[StoreKey, dict[str, Any]] = {}       # async bulk scan records
		self._quarantine_releases: dict[StoreKey, dict[str, Any]] = {}  # release records
		self._ml_models: dict[StoreKey, dict[str, Any]] = {}            # classifier model metadata
		self._agent_runtimes = set(DEFAULT_CONFIGURATION["agents"]["supported_runtimes"])
		self._agent_roles = set(DEFAULT_CONFIGURATION["agents"]["supported_roles"])
		self._privileged_agent_roles = set(DEFAULT_CONFIGURATION["agents"]["privileged_roles"])
		self._lifecycle_operations = set(DEFAULT_CONFIGURATION["streaming"]["required_operations"])

	# -------------------------------------------------------------------------
	# Contract helpers
	# -------------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# -------------------------------------------------------------------------
	# Policy management
	# -------------------------------------------------------------------------

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

	def create_policy(
		self,
		policy_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		channels: list[str],
		classifiers: list[str],
		default_action: str = "quarantine",
	) -> dict[str, Any]:
		"""Alias for register_policy with simpler signature."""
		return self.register_policy(
			policy_id=policy_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			channels=channels,
			classifiers=classifiers,
			default_action=default_action,
		)

	def update_policy(
		self,
		policy_id: str,
		tenant_id: str,
		actor: str,
		channels: list[str] | None = None,
		classifiers: list[str] | None = None,
		default_action: str | None = None,
		status: str | None = None,
	) -> dict[str, Any]:
		"""Mutate an existing policy's fields."""
		policy = self._require_policy(policy_id, tenant_id)
		self._require_tenant(tenant_id)
		if channels is not None:
			policy.channels = list(channels)
		if classifiers is not None:
			for cid in classifiers:
				self._require_classifier(cid, tenant_id)
			policy.classifiers = list(classifiers)
		if default_action is not None:
			if default_action not in {"allow", "alert", "block", "quarantine"}:
				raise ValueError(f"invalid_default_action:{default_action}")
			policy.default_action = default_action
		if status is not None:
			policy.status = status
		self._record_audit(tenant_id, "policy_updated", policy_id, actor, policy.to_dict())
		return policy.to_dict()

	def policy_effectiveness(self, tenant_id: str, policy_id: str) -> dict[str, Any]:
		"""Compute effectiveness metrics for a policy based on inspection history."""
		self._require_tenant(tenant_id)
		policy = self._require_policy(policy_id, tenant_id)
		inspections = [
			v for v in self._inspections.values()
			if v.tenant_id == tenant_id and v.policy_id == policy_id
		]
		total = len(inspections)
		blocked = sum(1 for i in inspections if i.blocked)
		quarantined = sum(1 for i in inspections if i.quarantined)
		reviewed = sum(1 for i in inspections if i.review_required)
		fp_count = sum(
			1 for fp in self._false_positives.values()
			if fp["tenant_id"] == tenant_id and fp.get("policy_id") == policy_id
		)
		tp = max(0, blocked + quarantined - fp_count)
		precision = tp / max(blocked + quarantined, 1)
		return {
			"policy_id": policy_id,
			"policy_name": policy.name,
			"total_inspections": total,
			"blocked_count": blocked,
			"quarantined_count": quarantined,
			"review_required_count": reviewed,
			"false_positive_count": fp_count,
			"true_positive_estimate": tp,
			"precision": round(precision, 4),
			"action_rate": round((blocked + quarantined) / max(total, 1), 4),
		}

	# -------------------------------------------------------------------------
	# Classifier management
	# -------------------------------------------------------------------------

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

	def regex_pattern_library(
		self,
		tenant_id: str,
		pattern_id: str,
		name: str,
		regex: str,
		sensitivity_label: str,
		owner: str,
	) -> dict[str, Any]:
		"""Add a named regex pattern to the shared pattern library."""
		self._require_tenant(tenant_id)
		try:
			re.compile(regex)
		except re.error as exc:
			raise ValueError(f"invalid_regex:{exc}") from exc
		key = self._key(tenant_id, pattern_id)
		record = {
			"id": pattern_id,
			"tenant_id": tenant_id,
			"name": name,
			"regex": regex,
			"sensitivity_label": sensitivity_label,
			"owner": owner,
			"created_at": _ts(),
		}
		self._regex_patterns[key] = record
		self._record_audit(tenant_id, "regex_pattern_added", pattern_id, owner, record)
		return dict(record)

	def ml_classifier_train(
		self,
		tenant_id: str,
		model_id: str,
		classifier_id: str,
		training_samples: list[dict[str, Any]],
		trainer: str,
	) -> dict[str, Any]:
		"""Register training metadata for an ML classifier model."""
		self._require_tenant(tenant_id)
		self._require_classifier(classifier_id, tenant_id)
		if len(training_samples) < 10:
			raise ValueError("ml_classifier_requires_at_least_10_samples")
		positive = sum(1 for s in training_samples if s.get("label") == "positive")
		negative = len(training_samples) - positive
		record = {
			"id": model_id,
			"tenant_id": tenant_id,
			"classifier_id": classifier_id,
			"sample_count": len(training_samples),
			"positive_samples": positive,
			"negative_samples": negative,
			"trainer": trainer,
			"status": "trained",
			"trained_at": _ts(),
		}
		self._ml_models[self._key(tenant_id, model_id)] = record
		self._record_audit(tenant_id, "ml_classifier_trained", model_id, trainer, record)
		return dict(record)

	# -------------------------------------------------------------------------
	# Content evaluation
	# -------------------------------------------------------------------------

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

	def evaluate_content(self, text: str, policy_ids: list[str], tenant_id: str) -> dict[str, Any]:
		"""Evaluate text against a set of named policies and return combined verdict."""
		self._require_tenant(tenant_id)
		policy_results: list[dict[str, Any]] = []
		overall_action = "allow"
		for pid in policy_ids:
			policy = self._require_policy(pid, tenant_id)
			classification = self.classify_content(tenant_id, text, policy.classifiers)
			hits = classification["classifier_hits"]
			severity = classification["severity"]
			action = action_for(policy.default_action, severity, False)
			if action in {"block", "quarantine"}:
				overall_action = "block"
			elif action == "alert" and overall_action == "allow":
				overall_action = "alert"
			policy_results.append({
				"policy_id": pid,
				"policy_name": policy.name,
				"action": action,
				"severity": severity,
				"hit_count": len(hits),
			})
		return {
			"tenant_id": tenant_id,
			"content_hash": stable_digest(text),
			"overall_action": overall_action,
			"policy_results": policy_results,
			"evaluated_at": _ts(),
		}

	def scan_file(
		self,
		tenant_id: str,
		file_metadata: dict[str, Any],
		content: str,
		policy_id: str,
		actor: str,
	) -> dict[str, Any]:
		"""Scan a file's content against a DLP policy."""
		self._require_tenant(tenant_id)
		policy = self._require_policy(policy_id, tenant_id)
		classification = self.classify_content(tenant_id, content, policy.classifiers)
		severity = classification["severity"]
		action = action_for(policy.default_action, severity, False)
		result = {
			"tenant_id": tenant_id,
			"file_name": file_metadata.get("name", "unknown"),
			"file_size": file_metadata.get("size_bytes", 0),
			"mime_type": file_metadata.get("mime_type", "application/octet-stream"),
			"content_hash": stable_digest(content),
			"classification_label": classification["classification_label"],
			"severity": severity,
			"action": action,
			"classifier_hits": classification["classifier_hits"],
			"scanned_by": actor,
			"scanned_at": _ts(),
		}
		self._record_audit(tenant_id, "file_scanned", file_metadata.get("name", "unknown"), actor, result)
		return result

	def scan_email(
		self,
		tenant_id: str,
		headers: dict[str, str],
		body: str,
		attachments: list[dict[str, Any]],
		policy_id: str,
		actor: str,
	) -> dict[str, Any]:
		"""Scan an email (headers + body + attachments) against a DLP policy."""
		self._require_tenant(tenant_id)
		policy = self._require_policy(policy_id, tenant_id)
		full_content = f"{json.dumps(headers)}\n{body}"
		for att in attachments:
			full_content += f"\n{att.get('content', '')}"
		classification = self.classify_content(tenant_id, full_content, policy.classifiers)
		severity = classification["severity"]
		action = action_for(policy.default_action, severity, False)
		result = {
			"tenant_id": tenant_id,
			"from": headers.get("from", ""),
			"to": headers.get("to", ""),
			"subject": headers.get("subject", ""),
			"attachment_count": len(attachments),
			"content_hash": stable_digest(full_content),
			"classification_label": classification["classification_label"],
			"severity": severity,
			"action": action,
			"classifier_hits": classification["classifier_hits"],
			"scanned_by": actor,
			"scanned_at": _ts(),
		}
		self._record_audit(tenant_id, "email_scanned", headers.get("message_id", "unknown"), actor, result)
		return result

	def endpoint_event(
		self,
		tenant_id: str,
		device_id: str,
		event_type: str,
		data: dict[str, Any],
		actor: str,
	) -> dict[str, Any]:
		"""Record and evaluate a DLP endpoint event (file copy, print, USB, etc.)."""
		self._require_tenant(tenant_id)
		supported_events = {"file_copy", "file_print", "usb_write", "email_send", "screen_capture", "clipboard_copy"}
		if event_type not in supported_events:
			raise ValueError(f"unsupported_endpoint_event:{event_type}")
		content = str(data.get("content", ""))
		severity = "low"
		if content:
			hits = detect_classifier_hits(content, list(DEFAULT_CONFIGURATION["data_patterns"]["enabled_classifiers"]))
			severity = highest_severity(hits) if hits else "low"
		record = {
			"tenant_id": tenant_id,
			"device_id": device_id,
			"event_type": event_type,
			"severity": severity,
			"data_hash": stable_digest(data),
			"actor": actor,
			"recorded_at": _ts(),
		}
		self._record_audit(tenant_id, "endpoint_event_recorded", device_id, actor, record)
		return record

	# -------------------------------------------------------------------------
	# Quarantine management
	# -------------------------------------------------------------------------

	def quarantine_item(
		self,
		tenant_id: str,
		item_id: str,
		reason: str,
		content_hash: str,
		actor: str,
		encrypted: bool = True,
	) -> dict[str, Any]:
		"""Manually quarantine an item."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, item_id)
		if key in self._quarantine:
			raise ValueError(f"item_already_quarantined:{item_id}")
		item = QuarantineItem(
			id=item_id,
			tenant_id=tenant_id,
			inspection_id="manual",
			content_hash=content_hash,
			reason=reason,
			encrypted=encrypted,
			legal_hold=DEFAULT_CONFIGURATION["governance"]["legal_hold_supported"],
		)
		self._quarantine[key] = item
		self._record_audit(tenant_id, "item_quarantined_manual", item_id, actor, {"reason": reason, "encrypted": encrypted})
		return item.to_dict()

	def release_quarantine(
		self,
		tenant_id: str,
		item_id: str,
		approver: str,
		justification: str,
	) -> dict[str, Any]:
		"""Release an item from quarantine with approver sign-off."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, item_id)
		item = self._quarantine.get(key)
		if item is None:
			raise KeyError(f"unknown_quarantine_item:{item_id}")
		if getattr(item, "legal_hold", False):
			raise PermissionError("cannot_release_legal_hold_item")
		release_record = {
			"id": f"rel_{item_id}",
			"tenant_id": tenant_id,
			"quarantine_item_id": item_id,
			"approver": approver,
			"justification": justification,
			"released_at": _ts(),
		}
		self._quarantine_releases[self._key(tenant_id, release_record["id"])] = release_record
		del self._quarantine[key]
		self._record_audit(tenant_id, "quarantine_released", item_id, approver, release_record)
		return release_record

	# -------------------------------------------------------------------------
	# Egress inspection
	# -------------------------------------------------------------------------

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

	def bulk_scan(
		self,
		tenant_id: str,
		job_id: str,
		policy_id: str,
		items: list[dict[str, Any]],
		actor: str,
	) -> dict[str, Any]:
		"""Bulk scan multiple content items against a policy. Returns summary."""
		self._require_tenant(tenant_id)
		policy = self._require_policy(policy_id, tenant_id)
		results: list[dict[str, Any]] = []
		action_counts: Counter[str] = Counter()
		for item in items:
			content = str(item.get("content", ""))
			classification = self.classify_content(tenant_id, content, policy.classifiers)
			severity = classification["severity"]
			action = action_for(policy.default_action, severity, False)
			action_counts[action] += 1
			results.append({
				"item_id": item.get("id", stable_digest(content)[:16]),
				"action": action,
				"severity": severity,
				"hit_count": len(classification["classifier_hits"]),
			})
		job = {
			"id": job_id,
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"item_count": len(items),
			"action_summary": dict(action_counts),
			"results": results,
			"actor": actor,
			"completed_at": _ts(),
		}
		self._bulk_scan_jobs[self._key(tenant_id, job_id)] = job
		self._record_audit(tenant_id, "bulk_scan_completed", job_id, actor, {
			"item_count": job["item_count"],
			"action_summary": job["action_summary"],
		})
		return dict(job)

	# -------------------------------------------------------------------------
	# Incidents
	# -------------------------------------------------------------------------

	def incident_create(
		self,
		tenant_id: str,
		incident_id: str,
		severity: str,
		title: str,
		description: str,
		owner: str,
		source_ref: str | None = None,
	) -> dict[str, Any]:
		"""Manually open a DLP incident."""
		self._require_tenant(tenant_id)
		if severity not in {"low", "medium", "high", "critical"}:
			raise ValueError(f"invalid_severity:{severity}")
		incident = DlpIncident(
			id=incident_id,
			tenant_id=tenant_id,
			inspection_id=source_ref or "manual",
			severity=severity,
			owner=owner,
			required_action=description,
			notifications_sent=DEFAULT_CONFIGURATION["response"]["notification_required"],
		)
		self._incidents[self._key(tenant_id, incident_id)] = incident
		self._record_audit(tenant_id, "incident_opened_manual", incident_id, owner, {
			"title": title,
			"severity": severity,
			"description": description,
		})
		return incident.to_dict()

	def incident_investigate(
		self,
		tenant_id: str,
		incident_id: str,
		investigator: str,
		findings: str,
		evidence_refs: list[str],
	) -> dict[str, Any]:
		"""Record investigation findings on an incident."""
		self._require_tenant(tenant_id)
		incident = self._require_incident(incident_id, tenant_id)
		if incident.status not in {"open", "investigating"}:
			raise ValueError(f"incident_not_open:{incident.status}")
		incident.status = "investigating"
		self._record_audit(tenant_id, "incident_investigated", incident_id, investigator, {
			"findings": findings,
			"evidence_count": len(evidence_refs),
		})
		return incident.to_dict()

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

	# -------------------------------------------------------------------------
	# Feedback & analytics
	# -------------------------------------------------------------------------

	def false_positive_feedback(
		self,
		tenant_id: str,
		feedback_id: str,
		inspection_id: str,
		reporter: str,
		justification: str,
		policy_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a false-positive feedback for an inspection result."""
		self._require_tenant(tenant_id)
		self._require_inspection(inspection_id, tenant_id)
		record = {
			"id": feedback_id,
			"tenant_id": tenant_id,
			"inspection_id": inspection_id,
			"policy_id": policy_id,
			"reporter": reporter,
			"justification": justification,
			"submitted_at": _ts(),
		}
		self._false_positives[self._key(tenant_id, feedback_id)] = record
		self._record_audit(tenant_id, "false_positive_submitted", feedback_id, reporter, record)
		return dict(record)

	def shadow_it_detection(
		self,
		tenant_id: str,
		detection_id: str,
		application_name: str,
		user_id: str,
		activity_type: str,
		data_size_bytes: int,
		risk_level: str,
	) -> dict[str, Any]:
		"""Record a Shadow IT detection event."""
		self._require_tenant(tenant_id)
		if risk_level not in {"low", "medium", "high", "critical"}:
			raise ValueError(f"invalid_risk_level:{risk_level}")
		record = {
			"id": detection_id,
			"tenant_id": tenant_id,
			"application_name": application_name,
			"user_id": user_id,
			"activity_type": activity_type,
			"data_size_bytes": data_size_bytes,
			"risk_level": risk_level,
			"detected_at": _ts(),
		}
		self._shadow_it[self._key(tenant_id, detection_id)] = record
		self._record_audit(tenant_id, "shadow_it_detected", detection_id, user_id, record)
		return dict(record)

	def cloud_activity_monitoring(
		self,
		tenant_id: str,
		event_id: str,
		provider: str,
		service: str,
		user_id: str,
		action: str,
		resource: str,
		risk_score: float,
	) -> dict[str, Any]:
		"""Record and evaluate a cloud activity event for DLP risk."""
		self._require_tenant(tenant_id)
		if not (0.0 <= risk_score <= 1.0):
			raise ValueError("risk_score_must_be_between_0_and_1")
		severity = (
			"critical" if risk_score >= 0.8
			else "high" if risk_score >= 0.6
			else "medium" if risk_score >= 0.3
			else "low"
		)
		record = {
			"id": event_id,
			"tenant_id": tenant_id,
			"provider": provider,
			"service": service,
			"user_id": user_id,
			"action": action,
			"resource": resource,
			"risk_score": risk_score,
			"severity": severity,
			"recorded_at": _ts(),
		}
		self._cloud_activities[self._key(tenant_id, event_id)] = record
		self._record_audit(tenant_id, "cloud_activity_monitored", event_id, user_id, record)
		return dict(record)

	def dlp_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate DLP analytics across all channels."""
		self._require_tenant(tenant_id)
		inspections = self.list_inspections(tenant_id)
		incidents = self.list_incidents(tenant_id)
		shadow = [v for v in self._shadow_it.values() if v["tenant_id"] == tenant_id]
		cloud = [v for v in self._cloud_activities.values() if v["tenant_id"] == tenant_id]
		fp_count = sum(1 for v in self._false_positives.values() if v["tenant_id"] == tenant_id)
		channel_counts: Counter[str] = Counter(i["channel"] for i in inspections)
		severity_counts: Counter[str] = Counter(i["severity"] for i in inspections)
		return {
			"tenant_id": tenant_id,
			"total_inspections": len(inspections),
			"blocked": sum(1 for i in inspections if i["blocked"]),
			"quarantined": sum(1 for i in inspections if i["quarantined"]),
			"false_positives": fp_count,
			"open_incidents": sum(1 for i in incidents if i["status"] == "open"),
			"shadow_it_detections": len(shadow),
			"cloud_activity_events": len(cloud),
			"by_channel": dict(channel_counts),
			"by_severity": dict(severity_counts),
		}

	def reporting_export(
		self,
		tenant_id: str,
		format: str = "json",
		include: list[str] | None = None,
	) -> dict[str, Any]:
		"""Export DLP report data as JSON or CSV."""
		self._require_tenant(tenant_id)
		include_sets = set(include or ["inspections", "incidents", "policies"])
		data: dict[str, Any] = {}
		if "inspections" in include_sets:
			data["inspections"] = self.list_inspections(tenant_id)
		if "incidents" in include_sets:
			data["incidents"] = self.list_incidents(tenant_id)
		if "policies" in include_sets:
			data["policies"] = self.list_policies(tenant_id)
		if format == "csv":
			buf = io.StringIO()
			all_rows: list[dict[str, Any]] = []
			for section, rows in data.items():
				for row in rows:
					all_rows.append({"_section": section, **{k: str(v) for k, v in row.items()}})
			if all_rows:
				writer = csv.DictWriter(buf, fieldnames=list(all_rows[0].keys()))
				writer.writeheader()
				writer.writerows(all_rows)
			payload = buf.getvalue()
		else:
			payload = json.dumps(data, indent=2)
		self._record_audit(tenant_id, "dlp_report_exported", "report", "system", {"format": format, "sections": list(include_sets)})
		return {
			"tenant_id": tenant_id,
			"format": format,
			"sections": list(include_sets),
			"payload": payload,
			"exported_at": _ts(),
		}

	# -------------------------------------------------------------------------
	# Agents & lifecycle batches
	# -------------------------------------------------------------------------

	def register_dlp_agent(
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
		self._ensure_new(self._dlp_agents, tenant_id, agent_id)
		self._require_tenant(tenant_id)
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_dlp_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		self._raise_if_denied(result)
		if not str(name or "").strip():
			raise ValueError("dlp_agent_name_required")
		agent = DlpAgentRecord(
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
		self._dlp_agents[self._key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, "dlp_agent_registered", agent.id, owner, {**agent.to_dict(), "rule_decision": result["decision"]})
		return agent.to_dict()

	def validate_dlpd_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "dlp_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("dlpd_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_dlpd_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_dlpd_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
		})
		accepted = result["decision"] == "allow"
		record = DlpdLifecycleBatchRecord(
			id=batch_id or f"dlpdbatch:{len(self._lifecycle_batches) + 1:06d}",
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
		self._record_audit(tenant_id, f"dlpd_lifecycle_batch_{record.status}", record.id, "bytewax", record.to_dict())
		self._raise_if_denied(result)
		return record.to_dict()

	# -------------------------------------------------------------------------
	# Bulk CRUD helpers
	# -------------------------------------------------------------------------

	def bulk_register_classifiers(
		self,
		tenant_id: str,
		classifiers: list[dict[str, Any]],
		reviewed_by: str,
	) -> list[dict[str, Any]]:
		"""Register multiple classifiers in one call."""
		self._require_tenant(tenant_id)
		results: list[dict[str, Any]] = []
		for spec in classifiers:
			results.append(self.register_classifier(
				classifier_id=spec["id"],
				tenant_id=tenant_id,
				name=spec["name"],
				classifier_type=spec.get("classifier_type", "regex"),
				sensitivity_label=spec.get("sensitivity_label", "internal"),
				pattern_keys=spec.get("pattern_keys", []),
				reviewed_by=reviewed_by,
			))
		return results

	# -------------------------------------------------------------------------
	# Dashboard / health
	# -------------------------------------------------------------------------

	def health_check(self) -> dict[str, Any]:
		return {
			"service": "dlpd",
			"status": "healthy",
			"policy_count": len(self._policies),
			"classifier_count": len(self._classifiers),
			"checked_at": _ts(),
		}

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
			"false_positive_count": sum(1 for v in self._false_positives.values() if v["tenant_id"] == tenant_id),
			"shadow_it_detection_count": sum(1 for v in self._shadow_it.values() if v["tenant_id"] == tenant_id),
			"cloud_activity_count": sum(1 for v in self._cloud_activities.values() if v["tenant_id"] == tenant_id),
			"dlp_agent_count": len(self.list_dlp_agents(tenant_id)),
			"pending_agent_review_count": sum(1 for item in self.list_dlp_agents(tenant_id) if item["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"),
		}

	# -------------------------------------------------------------------------
	# List helpers
	# -------------------------------------------------------------------------

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

	def list_dlp_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._dlp_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	# -------------------------------------------------------------------------
	# Private helpers
	# -------------------------------------------------------------------------

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


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
