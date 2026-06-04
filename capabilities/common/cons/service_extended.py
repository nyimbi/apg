"""
Consent & Privacy management service — extended methods for APG CONS capability.

Adds 21 new async methods to reach 42+ total on ConsServiceExtended:
	record_consent (override with full audit), revoke_consent,
	consent_history, preference_portal, dsar_request, erasure_execute,
	portability_export, consent_audit, cookie_consent,
	legitimate_interest_balance, breach_notify, privacy_impact_assess,
	data_map, third_party_disclosure, consent_analytics,
	health_check, bulk_capture_consent, bulk_revoke_consent,
	export_consent_data, list_dsar_requests, list_breach_records

© 2025 Datacraft · www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import utc_now
from .service import ConsService


def _sha8(value: Any) -> str:
	raw = json.dumps(value, sort_keys=True, default=str)
	return hashlib.sha256(raw.encode()).hexdigest()[:8]


class ConsServiceExtended(ConsService):
	"""ConsService + 21 domain-specific async methods (42+ total)."""

	def __init__(self) -> None:
		super().__init__()
		self._cookie_consents: dict[str, dict[str, Any]] = {}
		self._legitimate_interests: dict[str, dict[str, Any]] = {}
		self._data_maps: dict[str, dict[str, Any]] = {}
		self._third_party_disclosures: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------ 1
	async def record_consent(
		self,
		tenant_id: str,
		consent_id: str,
		subject_id: str,
		purpose: str,
		legal_basis: str,
		channel: str,
		captured_by: str = "system",
		expiry_days: int | None = None,
	) -> dict[str, Any]:
		"""High-level consent capture from any channel with full audit trail."""
		notice_id = f"notice:{purpose}:default"
		purpose_id = f"purpose:{purpose}:default"
		if self._key(tenant_id, notice_id) not in self._notices:
			self.publish_notice(
				notice_id=notice_id,
				tenant_id=tenant_id,
				version="v1",
				url=f"https://privacy.example.com/{purpose}",
				language="en",
				purposes=[purpose],
				published_by="system",
			)
		if self._key(tenant_id, purpose_id) not in self._purposes:
			self.create_purpose(
				purpose_id=purpose_id,
				tenant_id=tenant_id,
				name=purpose,
				owner="system",
				legal_basis=legal_basis,
				retention_policy="standard",
				notice_id=notice_id,
				data_categories=["general"],
			)
		result = self.capture_consent(
			consent_id=consent_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			purpose_id=purpose_id,
			notice_id=notice_id,
			source=channel,
			captured_by=captured_by,
		)
		if expiry_days is not None:
			result["expiry_date"] = (utc_now() + timedelta(days=expiry_days)).isoformat()
		return result

	# ------------------------------------------------------------------ 2
	async def revoke_consent(
		self,
		tenant_id: str,
		consent_id: str,
		reason: str,
		revoked_by: str = "subject",
	) -> dict[str, Any]:
		"""Revoke a consent record with a documented reason."""
		assert bool(reason), "revocation reason required"
		consent = self._require_consent(consent_id, tenant_id)
		consent.status = "withdrawn"
		consent.withdrawn_at = utc_now()
		revocation = consent.to_dict() | {"revocation_reason": reason, "revoked_by": revoked_by}
		self._record_audit(tenant_id, "consent_revoked", consent_id, revoked_by, revocation)
		return revocation

	# ------------------------------------------------------------------ 3
	async def consent_history(
		self,
		tenant_id: str,
		subject_id: str,
	) -> list[dict[str, Any]]:
		"""Return full consent history for a data subject."""
		consents = [c.to_dict() for c in self._consents.values() if c.tenant_id == tenant_id and c.subject_id == subject_id]
		return sorted(consents, key=lambda x: x.get("captured_at", ""))

	# ------------------------------------------------------------------ 4
	async def preference_portal(
		self,
		tenant_id: str,
		subject_id: str,
	) -> dict[str, Any]:
		"""Return the full preference + consent view for a subject's self-service portal."""
		history = await self.consent_history(tenant_id, subject_id)
		active = [c for c in history if c.get("status") == "active"]
		withdrawn = [c for c in history if c.get("status") == "withdrawn"]
		prefs = [p.to_dict() for p in self._preferences.values() if p.tenant_id == tenant_id and p.subject_id == subject_id]
		return {
			"subject_id": subject_id,
			"tenant_id": tenant_id,
			"active_consents": active,
			"withdrawn_consents": withdrawn,
			"preferences": prefs,
			"total_consent_events": len(history),
			"generated_at": utc_now().isoformat(),
		}

	# ------------------------------------------------------------------ 5
	async def dsar_request(
		self,
		tenant_id: str,
		request_id: str,
		subject_id: str,
		request_type: str = "access",
		identity_verified: bool = True,
		evidence_reference: str = "id_verified",
	) -> dict[str, Any]:
		"""Submit a GDPR/POPIA Data Subject Access Request."""
		return self.submit_privacy_request(
			request_id=request_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			request_type=request_type,
			submitted_by=subject_id,
			identity_verified=identity_verified,
			evidence_reference=evidence_reference,
		)

	# ------------------------------------------------------------------ 6
	async def erasure_execute(
		self,
		tenant_id: str,
		erasure_id: str,
		subject_id: str,
		executed_by: str,
		reason: str,
	) -> dict[str, Any]:
		"""Execute a right-to-erasure: withdraw all consents and record erasure."""
		assert erasure_id and subject_id and executed_by and reason
		consents_withdrawn: list[str] = []
		for consent in list(self._consents.values()):
			if consent.tenant_id == tenant_id and consent.subject_id == subject_id and consent.status == "active":
				consent.status = "withdrawn"
				consent.withdrawn_at = utc_now()
				consents_withdrawn.append(consent.id)
		record: dict[str, Any] = {
			"id": erasure_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"executed_by": executed_by,
			"reason": reason,
			"consents_withdrawn": consents_withdrawn,
			"status": "completed",
			"executed_at": utc_now().isoformat(),
		}
		self._erasure_records[self._key(tenant_id, erasure_id)] = record
		self._record_audit(tenant_id, "erasure_executed", erasure_id, executed_by, record)
		return record

	# ------------------------------------------------------------------ 7
	async def portability_export(
		self,
		tenant_id: str,
		export_id: str,
		subject_id: str,
		fmt: str = "json",
		exported_by: str = "system",
	) -> dict[str, Any]:
		"""Export all data held for a subject (data portability right)."""
		assert fmt in {"json", "csv"}, "fmt must be json or csv"
		history = await self.consent_history(tenant_id, subject_id)
		prefs = [p.to_dict() for p in self._preferences.values() if p.tenant_id == tenant_id and p.subject_id == subject_id]
		payload = {
			"subject_id": subject_id,
			"tenant_id": tenant_id,
			"exported_at": utc_now().isoformat(),
			"consents": history,
			"preferences": prefs,
		}
		serialized = json.dumps(payload, indent=2) if fmt == "json" else _to_csv(history)
		record: dict[str, Any] = {
			"id": export_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"fmt": fmt,
			"exported_by": exported_by,
			"record_count": len(history) + len(prefs),
			"payload_hash": _sha8(payload),
			"exported_at": utc_now().isoformat(),
		}
		self._portability_exports[self._key(tenant_id, export_id)] = record
		self._record_audit(tenant_id, "portability_exported", export_id, exported_by, record)
		record["data"] = serialized
		return record

	# ------------------------------------------------------------------ 8
	async def consent_audit(
		self,
		tenant_id: str,
		subject_id: str | None = None,
		purpose_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return audit trail for consent events, optionally filtered."""
		events = [e.to_dict() for e in self._audit_events if e.tenant_id == tenant_id]
		if subject_id:
			events = [e for e in events if e.get("subject_id") == subject_id or subject_id in str(e.get("payload", ""))]
		if purpose_id:
			events = [e for e in events if purpose_id in str(e.get("payload", ""))]
		return sorted(events, key=lambda e: e.get("id", ""))

	# ------------------------------------------------------------------ 9
	async def cookie_consent(
		self,
		tenant_id: str,
		cookie_consent_id: str,
		subject_id: str,
		categories: dict[str, bool],
		source: str = "cookie_banner",
		ip_hash: str = "",
	) -> dict[str, Any]:
		"""Record cookie consent categories from a consent banner."""
		assert cookie_consent_id and tenant_id and subject_id and categories
		key = self._key(tenant_id, cookie_consent_id)
		if key in self._cookie_consents:
			raise ValueError(f"cookie_consent_already_exists:{cookie_consent_id}")
		record: dict[str, Any] = {
			"id": cookie_consent_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"categories": categories,
			"source": source,
			"ip_hash": ip_hash,
			"strictly_necessary": True,
			"analytics": categories.get("analytics", False),
			"marketing": categories.get("marketing", False),
			"status": "active",
			"captured_at": utc_now().isoformat(),
		}
		self._cookie_consents[key] = record
		self._record_audit(tenant_id, "cookie_consent_captured", cookie_consent_id, subject_id, record)
		return record

	# ------------------------------------------------------------------ 10
	async def legitimate_interest_balance(
		self,
		assessment_id: str,
		tenant_id: str,
		purpose: str,
		controller: str,
		benefit: str,
		data_subject_impact: str,
		necessity_justification: str,
		balancing_test_passed: bool,
	) -> dict[str, Any]:
		"""Record a Legitimate Interest Assessment (LIA) balancing test."""
		assert assessment_id and tenant_id and purpose and controller
		key = self._key(tenant_id, assessment_id)
		if key in self._legitimate_interests:
			raise ValueError(f"lia_already_exists:{assessment_id}")
		record: dict[str, Any] = {
			"id": assessment_id,
			"tenant_id": tenant_id,
			"purpose": purpose,
			"controller": controller,
			"benefit": benefit,
			"data_subject_impact": data_subject_impact,
			"necessity_justification": necessity_justification,
			"balancing_test_passed": balancing_test_passed,
			"legal_basis": "legitimate_interest" if balancing_test_passed else "requires_consent",
			"status": "approved" if balancing_test_passed else "rejected",
			"assessed_at": utc_now().isoformat(),
		}
		self._legitimate_interests[key] = record
		self._record_audit(tenant_id, "lia_completed", assessment_id, controller, record)
		return record

	# ------------------------------------------------------------------ 11
	async def breach_notify(
		self,
		tenant_id: str,
		breach_id: str,
		description: str,
		affected_subjects: int,
		data_categories: list[str],
		severity: str,
		notified_by: str,
		discovery_date: datetime | None = None,
		notification_due_hours: int = 72,
	) -> dict[str, Any]:
		"""Record a data breach and compute notification deadline (GDPR Art. 33)."""
		assert breach_id and tenant_id and description and notified_by
		key = self._key(tenant_id, breach_id)
		if key in self._breach_records:
			raise ValueError(f"breach_already_exists:{breach_id}")
		discovered = discovery_date or utc_now()
		notify_by = discovered + timedelta(hours=notification_due_hours)
		record: dict[str, Any] = {
			"id": breach_id,
			"tenant_id": tenant_id,
			"description": description,
			"affected_subjects": affected_subjects,
			"data_categories": list(data_categories),
			"severity": severity,
			"notified_by": notified_by,
			"discovery_date": discovered.isoformat(),
			"notification_deadline": notify_by.isoformat(),
			"authority_notified": False,
			"subjects_notified": False,
			"status": "open",
			"created_at": utc_now().isoformat(),
		}
		self._breach_records[key] = record
		self._record_audit(tenant_id, "breach_notified", breach_id, notified_by, record)
		return record

	# ------------------------------------------------------------------ 12
	async def privacy_impact_assess(
		self,
		tenant_id: str,
		pia_id: str,
		processing_activity: str,
		data_categories: list[str],
		risk_level: str,
		assessor: str,
		mitigations: list[str] | None = None,
	) -> dict[str, Any]:
		"""Record a Privacy Impact Assessment (PIA/DPIA)."""
		assert pia_id and tenant_id and processing_activity and assessor
		key = self._key(tenant_id, pia_id)
		if key in self._pia_records:
			raise ValueError(f"pia_already_exists:{pia_id}")
		record: dict[str, Any] = {
			"id": pia_id,
			"tenant_id": tenant_id,
			"processing_activity": processing_activity,
			"data_categories": list(data_categories),
			"risk_level": risk_level,
			"assessor": assessor,
			"mitigations": list(mitigations or []),
			"dpa_consultation_required": risk_level in {"high", "critical"},
			"status": "draft",
			"assessed_at": utc_now().isoformat(),
		}
		self._pia_records[key] = record
		self._record_audit(tenant_id, "pia_completed", pia_id, assessor, record)
		return record

	# ------------------------------------------------------------------ 13
	async def data_map(
		self,
		tenant_id: str,
		map_id: str,
		system_name: str,
		data_categories: list[str],
		purposes: list[str],
		retention_days: int,
		controller: str,
		third_parties: list[str] | None = None,
	) -> dict[str, Any]:
		"""Register a data map entry (records of processing activities, GDPR Art. 30)."""
		assert map_id and tenant_id and system_name and data_categories and controller
		key = self._key(tenant_id, map_id)
		if key in self._data_maps:
			raise ValueError(f"data_map_already_exists:{map_id}")
		record: dict[str, Any] = {
			"id": map_id,
			"tenant_id": tenant_id,
			"system_name": system_name,
			"data_categories": list(data_categories),
			"purposes": list(purposes),
			"retention_days": retention_days,
			"controller": controller,
			"third_parties": list(third_parties or []),
			"status": "active",
			"created_at": utc_now().isoformat(),
		}
		self._data_maps[key] = record
		self._record_audit(tenant_id, "data_map_created", map_id, controller, record)
		return record

	# ------------------------------------------------------------------ 14
	async def third_party_disclosure(
		self,
		tenant_id: str,
		disclosure_id: str,
		subject_id: str,
		third_party_name: str,
		purpose: str,
		data_shared: list[str],
		disclosed_by: str,
		legal_basis: str = "consent",
		transfer_mechanism: str = "scc",
	) -> dict[str, Any]:
		"""Record a third-party data disclosure event."""
		assert disclosure_id and tenant_id and subject_id and third_party_name and disclosed_by
		key = self._key(tenant_id, disclosure_id)
		if key in self._third_party_disclosures:
			raise ValueError(f"disclosure_already_exists:{disclosure_id}")
		record: dict[str, Any] = {
			"id": disclosure_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"third_party_name": third_party_name,
			"purpose": purpose,
			"data_shared": list(data_shared),
			"disclosed_by": disclosed_by,
			"legal_basis": legal_basis,
			"transfer_mechanism": transfer_mechanism,
			"status": "disclosed",
			"disclosed_at": utc_now().isoformat(),
		}
		self._third_party_disclosures[key] = record
		self._record_audit(tenant_id, "third_party_disclosed", disclosure_id, disclosed_by, record)
		return record

	# ------------------------------------------------------------------ 15
	async def consent_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Compute consent KPI metrics for a tenant."""
		consents = [c for c in self._consents.values() if c.tenant_id == tenant_id]
		active = [c for c in consents if c.status == "active"]
		withdrawn = [c for c in consents if c.status == "withdrawn"]
		by_purpose: dict[str, int] = {}
		for c in consents:
			by_purpose[c.purpose_id] = by_purpose.get(c.purpose_id, 0) + 1
		breaches = [b for b in self._breach_records.values() if b["tenant_id"] == tenant_id]
		pias = [p for p in self._pia_records.values() if p["tenant_id"] == tenant_id]
		withdrawal_rate = round(len(withdrawn) / max(1, len(consents)) * 100, 2)
		return {
			"tenant_id": tenant_id,
			"total_consents": len(consents),
			"active_consents": len(active),
			"withdrawn_consents": len(withdrawn),
			"withdrawal_rate_pct": withdrawal_rate,
			"consents_by_purpose": by_purpose,
			"purposes_registered": len([p for p in self._purposes.values() if p.tenant_id == tenant_id]),
			"notices_published": len([n for n in self._notices.values() if n.tenant_id == tenant_id]),
			"pending_requests": len([r for r in self._requests.values() if r.tenant_id == tenant_id and r.status == "open"]),
			"open_breaches": len([b for b in breaches if b["status"] == "open"]),
			"pia_count": len(pias),
			"data_map_entries": len([m for m in self._data_maps.values() if m["tenant_id"] == tenant_id]),
			"generated_at": utc_now().isoformat(),
		}

	# ------------------------------------------------------------------ 16
	async def health_check(self) -> dict[str, Any]:
		"""Return service health and store cardinalities."""
		return {
			"status": "healthy",
			"checked_at": utc_now().isoformat(),
			"stores": {
				"purposes": len(self._purposes),
				"notices": len(self._notices),
				"consents": len(self._consents),
				"preferences": len(self._preferences),
				"requests": len(self._requests),
				"processing_decisions": len(self._processing_decisions),
				"agents": len(self._agents),
				"breach_records": len(self._breach_records),
				"pia_records": len(self._pia_records),
				"portability_exports": len(self._portability_exports),
				"erasure_records": len(self._erasure_records),
				"cookie_consents": len(self._cookie_consents),
				"legitimate_interests": len(self._legitimate_interests),
				"data_maps": len(self._data_maps),
				"third_party_disclosures": len(self._third_party_disclosures),
				"audit_events": len(self._audit_events),
			},
		}

	# ------------------------------------------------------------------ 17
	async def bulk_capture_consent(
		self,
		tenant_id: str,
		records: list[dict[str, Any]],
		channel: str = "bulk_import",
		captured_by: str = "system",
	) -> list[dict[str, Any]]:
		"""Capture multiple consent records in one call; skips duplicates."""
		assert tenant_id and records
		results: list[dict[str, Any]] = []
		for rec in records:
			cid = rec.get("consent_id", f"cons:{_sha8(rec)}")
			if self._key(tenant_id, cid) in self._consents:
				continue
			results.append(await self.record_consent(
				tenant_id=tenant_id,
				consent_id=cid,
				subject_id=rec["subject_id"],
				purpose=rec["purpose"],
				legal_basis=rec.get("legal_basis", "consent"),
				channel=rec.get("channel", channel),
				captured_by=rec.get("captured_by", captured_by),
				expiry_days=rec.get("expiry_days"),
			))
		return results

	# ------------------------------------------------------------------ 18
	async def bulk_revoke_consent(
		self,
		tenant_id: str,
		consent_ids: list[str],
		reason: str,
		revoked_by: str,
	) -> list[dict[str, Any]]:
		"""Revoke multiple consent records at once."""
		assert consent_ids and reason and revoked_by
		results: list[dict[str, Any]] = []
		for cid in consent_ids:
			try:
				results.append(await self.revoke_consent(tenant_id, cid, reason, revoked_by))
			except (KeyError, ValueError):
				pass
		return results

	# ------------------------------------------------------------------ 19
	async def export_consent_data(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export all consent and privacy data for a tenant."""
		assert fmt in {"json", "csv"}
		consents = [c.to_dict() for c in self._consents.values() if c.tenant_id == tenant_id]
		requests = [r.to_dict() for r in self._requests.values() if r.tenant_id == tenant_id]
		breaches = [b for b in self._breach_records.values() if b["tenant_id"] == tenant_id]
		data = {
			"tenant_id": tenant_id,
			"exported_at": utc_now().isoformat(),
			"consents": consents,
			"requests": requests,
			"breaches": breaches,
		}
		if fmt == "json":
			return json.dumps(data, indent=2, default=str)
		buf = io.StringIO()
		for section, rows in data.items():
			if not isinstance(rows, list) or not rows:
				continue
			writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
			buf.write(f"# {section}\n")
			writer.writeheader()
			writer.writerows(rows)
			buf.write("\n")
		return buf.getvalue()

	# ------------------------------------------------------------------ 20
	async def list_dsar_requests(
		self,
		tenant_id: str,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List DSAR/privacy requests optionally filtered by status."""
		requests = [r.to_dict() for r in self._requests.values() if r.tenant_id == tenant_id]
		if status:
			requests = [r for r in requests if r.get("status") == status]
		return sorted(requests, key=lambda x: x.get("submitted_at", ""))

	# ------------------------------------------------------------------ 21
	async def list_breach_records(
		self,
		tenant_id: str,
		severity: str | None = None,
	) -> list[dict[str, Any]]:
		"""List breach records optionally filtered by severity."""
		records = [b for b in self._breach_records.values() if b["tenant_id"] == tenant_id]
		if severity:
			records = [b for b in records if b.get("severity") == severity]
		return sorted(records, key=lambda x: x.get("created_at", ""))


def _to_csv(rows: list[dict[str, Any]]) -> str:
	if not rows:
		return ""
	buf = io.StringIO()
	writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
	writer.writeheader()
	writer.writerows(rows)
	return buf.getvalue()
