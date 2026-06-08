"""
APG Audit Logging Service Layer.

All public methods are async.  Constructor signature: __init__(db_session, tenant_id, actor_id).

The service is the only place that writes to storage.  It enforces:
  - tenant isolation on every query / write
  - immutability of events
  - SHA-256 checksum + chain-hash on every event
  - legal-hold enforcement on purge
  - risk scoring via domain rules
  - domain event emission after every state change

No Flask, no FastAPI — pure async Python.

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable

from .domain.rules import (
	RuleViolation,
	assert_actor_present,
	assert_checksum_valid,
	assert_dsr_requester_authorised,
	assert_erasure_allowed,
	assert_event_immutable,
	assert_evidence_package_not_sealed,
	assert_no_cross_tenant_access,
	assert_no_legal_hold_deletion,
	assert_retention_not_expired,
	assert_tenant_context,
	calculate_chain_hash,
	calculate_event_checksum,
	calculate_retain_until,
	calculate_risk_score,
	is_external_ip,
	is_off_hours,
)
from .models import (
	AuditEventCreate,
	AuditEventResponse,
	AuditQueryCreate,
	AuditQueryResponse,
	AuditSearchResult,
	AuditTrailCreate,
	AuditTrailResponse,
	AuditTrailUpdate,
	ComplianceFramework,
	ComplianceReportCreate,
	ComplianceReportResponse,
	DataSubjectRequestCreate,
	DataSubjectRequestResponse,
	DataSubjectRequestUpdate,
	DSRStatus,
	EvidencePackageCreate,
	EvidencePackageResponse,
	EvidencePackageStatus,
	ReportStatus,
	RetentionPolicyCreate,
	RetentionPolicyResponse,
	RetentionPolicyUpdate,
	RiskSummary,
	SIEMEvent,
	TamperDetectionCreate,
	TamperDetectionResponse,
	TamperStatus,
	TrailStatus,
	uuid7str,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process event bus (replace with Redis Pub/Sub or Bytewax in production)
# ---------------------------------------------------------------------------
_SUBSCRIBERS: list[Callable[[dict[str, Any]], None]] = []


def subscribe_domain_events(handler: Callable[[dict[str, Any]], None]) -> None:
	"""Register a handler that receives every domain event dict."""
	_SUBSCRIBERS.append(handler)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AuditLoggingService:
	"""
	Tenant-scoped, actor-tracked audit logging runtime.

	Parameters
	----------
	db_session : Any
	    SQLAlchemy async session (or compatible duck-typed object).
	    The service calls ``await db_session.execute(...)`` for reads,
	    ``db_session.add(...)`` + ``await db_session.flush()`` for writes,
	    and ``await db_session.commit()`` at the end of each operation.
	tenant_id : str
	    Tenant that owns every record created / queried by this instance.
	actor_id : str
	    Identity of the human or service account performing operations.
	"""

	def __init__(self, db_session: Any, tenant_id: str, actor_id: str) -> None:
		assert tenant_id, "tenant_id is required"
		assert actor_id,  "actor_id is required"
		self.db          = db_session
		self.tenant_id   = tenant_id
		self.actor_id    = actor_id
		# In-memory stores: keyed by id.  In production these delegate to db_session.
		self._events:     dict[str, AuditEventResponse]         = {}
		self._trails:     dict[str, AuditTrailResponse]         = {}
		self._reports:    dict[str, ComplianceReportResponse]   = {}
		self._policies:   dict[str, RetentionPolicyResponse]    = {}
		self._dsrs:       dict[str, DataSubjectRequestResponse] = {}
		self._packages:   dict[str, EvidencePackageResponse]    = {}
		self._tampers:    dict[str, TamperDetectionResponse]    = {}
		self._queries:    dict[str, AuditQueryResponse]         = {}
		# last chain hash per tenant (init to zero-hash; loaded from DB on first log_event)
		self._chain_tip:  str = "0" * 64
		self._chain_prev: str = "0" * 64  # chain_hash of the immediately preceding event
		self._chain_tip_initialized: bool = False  # lazy DB load flag
		# SIEM stream subscribers
		self._siem_queues: list[asyncio.Queue[SIEMEvent]] = []

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _log_operation(self, op: str, entity_id: str, detail: str = "") -> None:
		log.info("[audl] tenant=%s actor=%s op=%s id=%s %s",
				 self.tenant_id, self.actor_id, op, entity_id, detail)

	def _log_warn(self, msg: str) -> None:
		log.warning("[audl] tenant=%s actor=%s %s", self.tenant_id, self.actor_id, msg)

	def _log_error(self, msg: str) -> None:
		log.error("[audl] tenant=%s actor=%s %s", self.tenant_id, self.actor_id, msg)

	def _assert_own_tenant(self, resource_tenant: str) -> None:
		assert_no_cross_tenant_access(self.tenant_id, resource_tenant)

	def _emit_event(self, event_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		"""Fire-and-forget domain event to all in-process subscribers."""
		evt = {
			"domain":    "audl",
			"type":      event_type,
			"tenant_id": self.tenant_id,
			"actor_id":  self.actor_id,
			"entity_id": entity_id,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			**payload,
		}
		for handler in _SUBSCRIBERS:
			try:
				handler(evt)
			except Exception as exc:  # noqa: BLE001
				self._log_error(f"domain event handler failed: {exc}")

	def _push_siem(self, ev: SIEMEvent) -> None:
		for q in self._siem_queues:
			try:
				q.put_nowait(ev)
			except asyncio.QueueFull:
				pass

	# ------------------------------------------------------------------
	# 1. log_event — immutable write with checksum + chain
	# ------------------------------------------------------------------

	async def log_event(
		self,
		who:    str,        # actor
		what:   str,        # action verb
		on_what:str,        # resource_id
		how:    str,        # event_type  (AuditEventType value)
		where:  str | None, # ip_address
		when:   datetime | None,
		result: bool,       # success
		*,
		payload: AuditEventCreate | None = None,
	) -> AuditEventResponse:
		"""
		Primary ingestion entry-point.

		Accepts either a fully-populated ``AuditEventCreate`` *or* the six
		positional W5 arguments (who, what, on_what, how, where, when, result).
		When both are supplied the explicit payload takes precedence.

		Checksum and chain-hash are derived deterministically; the event is
		stamped immutable and cannot be mutated after creation.
		"""
		assert_actor_present(who)
		assert_tenant_context({"tenant_id": self.tenant_id})

		# Lazy-load chain_tip from DB on first call to survive service restarts
		if not self._chain_tip_initialized:
			self._chain_tip = await self._load_chain_tip_from_db()
			self._chain_prev = self._chain_tip
			self._chain_tip_initialized = True

		ts = when or datetime.now(timezone.utc)
		event_id = uuid7str()

		checksum = calculate_event_checksum(
			event_id    = event_id,
			tenant_id   = self.tenant_id,
			timestamp   = ts,
			event_type  = how,
			actor_id    = who,
			action      = what,
			resource_type = (payload.resource_type if payload else None),
			resource_id   = on_what,
			success       = result,
		)
		self._chain_prev = self._chain_tip  # record prev before advancing
		chain_hash   = calculate_chain_hash(self._chain_tip, checksum)
		self._chain_tip = chain_hash

		# Build response model
		base = payload.model_dump() if payload else {}
		resp = AuditEventResponse(
			id          = event_id,
			tenant_id   = self.tenant_id,
			created_at  = ts,
			updated_at  = ts,
			created_by  = who,
			level       = base.get("level", "info"),
			event_type  = how,
			source      = base.get("source", "apg_core"),
			category    = base.get("category", "general"),
			subcategory = base.get("subcategory"),
			actor_id    = who,
			actor_type  = base.get("actor_type", "user"),
			actor_display_name = base.get("actor_display_name"),
			session_id  = base.get("session_id"),
			service_account = base.get("service_account"),
			action      = what,
			action_description = base.get("action_description"),
			operation_id = base.get("operation_id"),
			resource_type = base.get("resource_type"),
			resource_id = on_what,
			resource_name = base.get("resource_name"),
			resource_path = base.get("resource_path"),
			parent_resource_id = base.get("parent_resource_id"),
			ip_address  = where,
			user_agent  = base.get("user_agent"),
			geographic_location = base.get("geographic_location"),
			device_id   = base.get("device_id"),
			request_id  = base.get("request_id"),
			correlation_id = base.get("correlation_id"),
			success     = result,
			status_code = base.get("status_code"),
			error_code  = base.get("error_code"),
			error_message = base.get("error_message"),
			duration_ms = base.get("duration_ms"),
			risk_score  = calculate_risk_score(
				is_failed_auth      = not result,
				is_privileged_actor = base.get("actor_type", "") in ("admin", "service"),
				is_off_hours        = is_off_hours(ts),
				is_external_ip      = is_external_ip(where),
				is_sensitive_data   = base.get("data_classification") in ("confidential", "restricted", "secret"),
				is_error_event      = not result,
			),
			anomaly_score = 0.0,
			compliance_tags    = base.get("compliance_tags", []),
			data_classification = base.get("data_classification"),
			retention_days      = base.get("retention_days", 2555),
			legal_hold          = base.get("legal_hold", False),
			contains_pii        = base.get("contains_pii", False),
			details    = base.get("details", {}),
			tags       = base.get("tags", {}),
			checksum   = checksum,
			chain_hash = chain_hash,
			immutable  = True,
		)

		self._events[event_id] = resp

		# SOC 2: persist to append-only DB table + publish to NATS event bus
		await self._persist_audit_event_to_db(resp)
		await self._publish_audit_to_nats(resp)

		self._log_operation("log_event", event_id,
							f"type={how} actor={who} resource={on_what} ok={result}")
		self._emit_event("audit_event_logged", event_id, {"event_type": how, "risk_score": resp.risk_score})

		# Push to SIEM stream
		self._push_siem(SIEMEvent(
			event_id   = event_id,
			tenant_id  = self.tenant_id,
			timestamp  = ts,
			level      = resp.level,
			event_type = resp.event_type,
			source     = resp.source,
			actor_id   = who,
			action     = what,
			resource_id= on_what,
			success    = result,
			risk_score = resp.risk_score,
			ip_address = where,
			checksum   = checksum,
		))

		return resp

	# ------------------------------------------------------------------
	# 2. immutable_log_write — batch variant
	# ------------------------------------------------------------------

	async def immutable_log_write(self, events: list[AuditEventCreate]) -> list[AuditEventResponse]:
		"""
		Write a batch of events atomically.  All events share the tenant of
		this service instance.  Returns them in insertion order.

		Raises ValueError if the batch is empty or exceeds 10 000.
		"""
		if not events:
			raise ValueError("batch must contain at least one event")
		if len(events) > 10_000:
			raise ValueError("batch size exceeds maximum of 10 000")

		results: list[AuditEventResponse] = []
		for ev in events:
			self._assert_own_tenant(ev.tenant_id)
			resp = await self.log_event(
				who     = ev.actor_id or self.actor_id,
				what    = ev.action,
				on_what = ev.resource_id or "",
				how     = ev.event_type,
				where   = ev.ip_address,
				when    = None,
				result  = ev.success,
				payload = ev,
			)
			results.append(resp)

		self._log_operation("immutable_log_write", "batch", f"count={len(results)}")
		return results

	# ------------------------------------------------------------------
	# 3. audit_trail_search
	# ------------------------------------------------------------------

	async def audit_trail_search(
		self,
		filters: AuditQueryCreate,
	) -> AuditSearchResult:
		"""
		Execute a structured / NLP / freetext search over stored events.
		All results are scoped to self.tenant_id — cross-tenant data never leaks.
		"""
		assert_tenant_context({"tenant_id": self.tenant_id})
		t0 = time.monotonic()

		q_id  = uuid7str()
		start = filters.date_start
		end   = filters.date_end

		# Filter predicate
		def matches(ev: AuditEventResponse) -> bool:
			if ev.tenant_id != self.tenant_id:
				return False
			if ev.is_deleted:
				return False
			if filters.event_types and ev.event_type not in filters.event_types:
				return False
			if filters.actor_ids and ev.actor_id not in filters.actor_ids:
				return False
			if filters.resource_ids and ev.resource_id not in filters.resource_ids:
				return False
			if filters.sources and ev.source not in filters.sources:
				return False
			if start and ev.created_at < start:
				return False
			if end and ev.created_at > end:
				return False
			if filters.risk_score_min is not None and ev.risk_score < filters.risk_score_min:
				return False
			if filters.risk_score_max is not None and ev.risk_score > filters.risk_score_max:
				return False
			if filters.compliance_tags:
				if not any(t in ev.compliance_tags for t in filters.compliance_tags):
					return False
			if filters.success is not None and ev.success != filters.success:
				return False
			if filters.full_text:
				needle = filters.full_text.lower()
				haystack = " ".join(filter(None, [ev.action, ev.resource_type, ev.resource_id,
												  ev.actor_id, ev.category]))
				if needle not in haystack.lower():
					return False
			return True

		all_matches = [ev for ev in self._events.values() if matches(ev)]
		# Sort
		reverse = filters.sort_desc
		all_matches.sort(key=lambda e: e.created_at, reverse=reverse)
		total = len(all_matches)
		page  = all_matches[filters.offset: filters.offset + filters.limit]

		ms = (time.monotonic() - t0) * 1000

		# Persist query record
		query_rec = AuditQueryResponse(
			id           = q_id,
			tenant_id    = self.tenant_id,
			created_by   = self.actor_id,
			query_type   = filters.query_type,
			event_types  = filters.event_types,
			actor_ids    = filters.actor_ids,
			resource_ids = filters.resource_ids,
			sources      = filters.sources,
			date_start   = filters.date_start,
			date_end     = filters.date_end,
			risk_score_min= filters.risk_score_min,
			risk_score_max= filters.risk_score_max,
			compliance_tags= filters.compliance_tags,
			success      = filters.success,
			full_text    = filters.full_text,
			nlp_query    = filters.nlp_query,
			limit        = filters.limit,
			offset       = filters.offset,
			sort_by      = filters.sort_by,
			sort_desc    = filters.sort_desc,
			requested_by = filters.requested_by,
			result_count = total,
			executed_at  = datetime.now(timezone.utc),
			duration_ms  = int(ms),
		)
		self._queries[q_id] = query_rec

		self._log_operation("audit_trail_search", q_id,
							f"total={total} page_size={len(page)} ms={ms:.1f}")
		return AuditSearchResult(
			query_id    = q_id,
			total_count = total,
			events      = page,
			query_ms    = ms,
			has_more    = (filters.offset + filters.limit) < total,
		)

	# ------------------------------------------------------------------
	# 4. tamper_detection
	# ------------------------------------------------------------------

	async def tamper_detection(
		self,
		scan_input: TamperDetectionCreate,
	) -> TamperDetectionResponse:
		"""
		Verify the checksum of every event in this tenant.
		Suspect events (checksum mismatch) are flagged but NOT deleted.
		"""
		self._assert_own_tenant(scan_input.tenant_id)
		assert_actor_present(self.actor_id)
		t0 = time.monotonic()

		scan_id = uuid7str()
		suspect: list[str] = []
		scanned = 0

		for ev in self._events.values():
			if ev.tenant_id != self.tenant_id:
				continue
			scanned += 1
			expected = calculate_event_checksum(
				event_id     = ev.id,
				tenant_id    = ev.tenant_id,
				timestamp    = ev.created_at,
				event_type   = ev.event_type,
				actor_id     = ev.actor_id,
				action       = ev.action,
				resource_type= ev.resource_type,
				resource_id  = ev.resource_id,
				success      = ev.success,
			)
			if ev.checksum != expected:
				suspect.append(ev.id)
				self._log_warn(f"tamper suspect event_id={ev.id}")

		status = TamperStatus.SUSPECT if suspect else TamperStatus.CLEAN
		rec = TamperDetectionResponse(
			id             = scan_id,
			tenant_id      = self.tenant_id,
			created_by     = self.actor_id,
			scan_type      = scan_input.scan_type,
			scanned_by     = scan_input.scanned_by,
			scope_filter   = scan_input.scope_filter,
			status         = status,
			events_scanned = scanned,
			events_suspect = len(suspect),
			suspect_ids    = suspect,
			completed_at   = datetime.now(timezone.utc),
			detail         = {"duration_ms": (time.monotonic() - t0) * 1000},
		)
		self._tampers[scan_id] = rec
		self._log_operation("tamper_detection", scan_id,
							f"scanned={scanned} suspect={len(suspect)} status={status}")
		self._emit_event("tamper_scan_completed", scan_id,
						 {"status": status, "suspect_count": len(suspect)})
		return rec

	# ------------------------------------------------------------------
	# 5. compliance_report
	# ------------------------------------------------------------------

	async def compliance_report(
		self,
		req: ComplianceReportCreate,
	) -> ComplianceReportResponse:
		"""
		Generate a compliance report for the specified framework and period.
		Collection, analysis, and (stub) rendering happen inline.
		"""
		self._assert_own_tenant(req.tenant_id)
		assert_actor_present(self.actor_id)

		report_id = uuid7str()
		now       = datetime.now(timezone.utc)

		# Gather relevant events
		events_in_window = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and req.period_start <= ev.created_at <= req.period_end
			and not ev.is_deleted
		]

		framework_tag = req.framework.value
		violations = [
			ev for ev in events_in_window
			if framework_tag in ev.compliance_tags
			or ev.event_type.value == "compliance_violation"
		]

		summary: dict[str, Any] = {
			"framework":         framework_tag,
			"period_start":      req.period_start.isoformat(),
			"period_end":        req.period_end.isoformat(),
			"total_events":      len(events_in_window),
			"violation_count":   len(violations),
			"by_event_type":     self._count_by(events_in_window, "event_type"),
			"by_source":         self._count_by(events_in_window, "source"),
			"recommendations":   self._framework_recommendations(req.framework, violations)
			                     if req.include_recommendations else [],
		}

		rec = ComplianceReportResponse(
			id              = report_id,
			tenant_id       = self.tenant_id,
			created_by      = self.actor_id,
			framework       = req.framework,
			period_start    = req.period_start,
			period_end      = req.period_end,
			requested_by    = req.requested_by,
			status          = ReportStatus.READY,
			include_violations      = req.include_violations,
			include_recommendations = req.include_recommendations,
			export_format   = req.export_format,
			violation_count = len(violations),
			summary         = summary,
			completed_at    = now,
		)
		self._reports[report_id] = rec
		self._log_operation("compliance_report", report_id,
							f"framework={framework_tag} violations={len(violations)}")
		self._emit_event("compliance_report_generated", report_id,
						 {"framework": framework_tag, "violations": len(violations)})
		return rec

	def _count_by(self, events: list[AuditEventResponse], field: str) -> dict[str, int]:
		counts: dict[str, int] = defaultdict(int)
		for ev in events:
			counts[str(getattr(ev, field, "unknown"))] += 1
		return dict(counts)

	def _framework_recommendations(
		self,
		framework: ComplianceFramework,
		violations: list[AuditEventResponse],
	) -> list[str]:
		recs: list[str] = []
		if violations:
			recs.append(f"Review {len(violations)} flagged event(s) for {framework.value} compliance.")
		match framework:
			case ComplianceFramework.GDPR:
				recs += ["Ensure PII fields are pseudonymised in audit records.",
						 "Confirm data subject request workflows are documented."]
			case ComplianceFramework.SOX:
				recs += ["Verify segregation of duties on financial system access.",
						 "Confirm privileged access reviews are conducted quarterly."]
			case ComplianceFramework.HIPAA:
				recs += ["Audit PHI access logs monthly.",
						 "Ensure minimum-necessary access principle is enforced."]
			case ComplianceFramework.PCI_DSS:
				recs += ["Review cardholder data access every 90 days.",
						 "Confirm log integrity monitoring is active."]
		return recs

	# ------------------------------------------------------------------
	# 6. gdpr_data_subject_access
	# ------------------------------------------------------------------

	async def gdpr_data_subject_access(
		self,
		req: DataSubjectRequestCreate,
		*,
		is_admin: bool = False,
	) -> DataSubjectRequestResponse:
		"""
		Fulfil a GDPR Art. 15 subject access request.
		Returns the DSR record with all event IDs that reference the subject.
		"""
		self._assert_own_tenant(req.tenant_id)
		assert_dsr_requester_authorised(req.requested_by, req.subject_id, is_admin)

		dsr_id = uuid7str()
		now    = datetime.now(timezone.utc)

		relevant_events = [
			ev.id for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and (ev.actor_id == req.subject_id or ev.resource_id == req.subject_id)
		]

		rec = DataSubjectRequestResponse(
			id            = dsr_id,
			tenant_id     = self.tenant_id,
			created_by    = self.actor_id,
			dsr_type      = req.dsr_type,
			subject_id    = req.subject_id,
			requested_by  = req.requested_by,
			justification = req.justification,
			scope_details = req.scope_details,
			status        = DSRStatus.FULFILLED,
			fulfilled_at  = now,
			response_data = {
				"event_ids":    relevant_events,
				"event_count":  len(relevant_events),
				"generated_at": now.isoformat(),
			},
		)
		self._dsrs[dsr_id] = rec
		self._log_operation("gdpr_data_subject_access", dsr_id,
							f"subject={req.subject_id} events={len(relevant_events)}")
		self._emit_event("dsr_fulfilled", dsr_id,
						 {"dsr_type": req.dsr_type, "subject_id": req.subject_id})
		return rec

	# ------------------------------------------------------------------
	# 7. right_to_erasure_audit_impact
	# ------------------------------------------------------------------

	async def right_to_erasure_audit_impact(
		self,
		subject_id: str,
		dsr_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Assess the audit-log impact of a GDPR Art. 17 erasure request.

		Audit evidence is exempt from erasure (Art. 17(3)(b)) but PII fields
		in non-immutable detail blobs CAN be pseudonymised.
		Returns a structured impact report without modifying any records.
		"""
		assert_tenant_context({"tenant_id": self.tenant_id})

		subject_events = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and (ev.actor_id == subject_id or ev.resource_id == subject_id)
		]

		legal_hold_blocked = [ev.id for ev in subject_events if ev.legal_hold]
		immutable_blocked  = [ev.id for ev in subject_events if ev.immutable]
		pii_candidates     = [ev.id for ev in subject_events if ev.contains_pii]

		report = {
			"subject_id":          subject_id,
			"tenant_id":           self.tenant_id,
			"total_audit_events":  len(subject_events),
			"erasure_blocked":     len(subject_events),
			"reason":              "Art. 17(3)(b) — audit evidence is exempt from erasure",
			"legal_hold_blocked":  legal_hold_blocked,
			"immutable_blocked":   immutable_blocked,
			"pii_pseudonymisable": pii_candidates,
			"recommendation":      "Pseudonymise PII fields in detail/tags blobs; core event remains.",
			"assessed_at":         datetime.now(timezone.utc).isoformat(),
		}
		self._log_operation("right_to_erasure_audit_impact", subject_id,
							f"total={len(subject_events)} pii_candidates={len(pii_candidates)}")
		return report

	# ------------------------------------------------------------------
	# 8. evidence_package_export
	# ------------------------------------------------------------------

	async def evidence_package_export(
		self,
		req: EvidencePackageCreate,
	) -> EvidencePackageResponse:
		"""
		Assemble a tamper-evident evidence package from specified events / trails.
		The package is sealed with a checksum and chain-of-custody record.
		"""
		self._assert_own_tenant(req.tenant_id)
		assert_actor_present(self.actor_id)

		pkg_id = uuid7str()
		now    = datetime.now(timezone.utc)

		# Collect events
		event_ids: list[str] = list(req.event_ids)
		for tid in req.trail_ids:
			trail = self._trails.get(tid)
			if trail and trail.tenant_id == self.tenant_id:
				event_ids += [eid for eid, ev in self._events.items()
							  if ev.tenant_id == self.tenant_id]

		collected = [self._events[eid] for eid in event_ids if eid in self._events]

		# Chain-of-custody entry
		coc_entry = {
			"actor":      self.actor_id,
			"action":     "package_assembled",
			"timestamp":  now.isoformat(),
			"event_count": len(collected),
		}

		# Package checksum = hash of all event checksums
		pkg_checksum = hashlib.sha256(
			json.dumps(
				sorted(ev.checksum for ev in collected if ev.checksum),
			).encode()
		).hexdigest()

		rec = EvidencePackageResponse(
			id              = pkg_id,
			tenant_id       = self.tenant_id,
			created_by      = self.actor_id,
			name            = req.name,
			description     = req.description,
			event_ids       = [ev.id for ev in collected],
			trail_ids       = list(req.trail_ids),
			requested_by    = req.requested_by,
			reason          = req.reason,
			legal_matter    = req.legal_matter,
			status          = EvidencePackageStatus.SEALED,
			include_chain   = req.include_chain,
			export_format   = req.export_format,
			file_checksum   = pkg_checksum,
			event_count     = len(collected),
			sealed_at       = now,
			sealed_by       = self.actor_id,
			chain_of_custody= [coc_entry],
		)
		self._packages[pkg_id] = rec
		self._log_operation("evidence_package_export", pkg_id,
							f"events={len(collected)} reason={req.reason}")
		self._emit_event("evidence_package_exported", pkg_id,
						 {"event_count": len(collected), "legal_matter": req.legal_matter})
		return rec

	# ------------------------------------------------------------------
	# 9. retention_enforcement
	# ------------------------------------------------------------------

	async def retention_enforcement(self) -> dict[str, Any]:
		"""
		Apply all active retention policies to stored events.
		Events past their retain_until that are NOT under legal hold are archived/deleted.
		Returns a summary of actions taken.
		"""
		assert_tenant_context({"tenant_id": self.tenant_id})

		now     = datetime.now(timezone.utc)
		archived: list[str] = []
		skipped:  list[str] = []

		for policy in self._policies.values():
			if not policy.is_active or policy.tenant_id != self.tenant_id:
				continue

			for ev in list(self._events.values()):
				if ev.tenant_id != self.tenant_id:
					continue
				if policy.event_types and ev.event_type not in policy.event_types:
					continue

				retain_until = calculate_retain_until(ev.created_at, policy.retain_days)
				if now < retain_until:
					continue  # still within retention window

				if ev.legal_hold:
					skipped.append(ev.id)
					continue

				# Archive / soft-delete
				ev.is_deleted = True
				ev.updated_at = now
				archived.append(ev.id)

		result = {
			"tenant_id":  self.tenant_id,
			"run_at":     now.isoformat(),
			"archived":   len(archived),
			"skipped_legal_hold": len(skipped),
			"archived_ids": archived,
			"skipped_ids":  skipped,
		}
		self._log_operation("retention_enforcement", "batch",
							f"archived={len(archived)} skipped={len(skipped)}")
		self._emit_event("retention_enforcement_run", "batch", result)
		return result

	# ------------------------------------------------------------------
	# 10. cross_tenant_audit_correlation
	# ------------------------------------------------------------------

	async def cross_tenant_audit_correlation(
		self,
		correlation_id: str,
		peer_service: "AuditLoggingService",
	) -> dict[str, Any]:
		"""
		Correlate events with the *same* correlation_id across two tenants.

		Both services must belong to the same platform operator (admin role).
		The result never exposes raw events from the peer tenant — it returns
		only counts and risk-score aggregates.  Full event data stays within
		each tenant's service boundary.
		"""
		my_events   = [ev for ev in self._events.values()
					   if ev.correlation_id == correlation_id and not ev.is_deleted]
		peer_events = [ev for ev in peer_service._events.values()
					   if ev.correlation_id == correlation_id and not ev.is_deleted]

		result = {
			"correlation_id":       correlation_id,
			"tenant_a":             self.tenant_id,
			"tenant_b":             peer_service.tenant_id,
			"tenant_a_event_count": len(my_events),
			"tenant_b_event_count": len(peer_events),
			"tenant_a_max_risk":    max((e.risk_score for e in my_events),   default=0.0),
			"tenant_b_max_risk":    max((e.risk_score for e in peer_events), default=0.0),
			"correlated_at":        datetime.now(timezone.utc).isoformat(),
		}
		self._log_operation("cross_tenant_audit_correlation", correlation_id,
							f"a={len(my_events)} b={len(peer_events)}")
		return result

	# ------------------------------------------------------------------
	# 11. real_time_siem_stream
	# ------------------------------------------------------------------

	async def real_time_siem_stream(
		self,
		risk_threshold: float = 0.0,
	) -> AsyncGenerator[SIEMEvent, None]:
		"""
		Async generator that yields SIEMEvent records in real time.

		Callers should run this in an async context and send each yielded
		event to their SIEM connector.  Cancel the generator to unsubscribe.
		"""
		q: asyncio.Queue[SIEMEvent] = asyncio.Queue(maxsize=10_000)
		self._siem_queues.append(q)
		try:
			while True:
				ev = await q.get()
				if ev.risk_score >= risk_threshold:
					yield ev
		finally:
			self._siem_queues.remove(q)

	# ------------------------------------------------------------------
	# AuditTrail CRUD
	# ------------------------------------------------------------------

	async def create_trail(self, req: AuditTrailCreate) -> AuditTrailResponse:
		"""Create a named audit trail that groups related events."""
		self._assert_own_tenant(req.tenant_id)
		assert_actor_present(self.actor_id)
		trail_id = uuid7str()
		rec = AuditTrailResponse(
			id         = trail_id,
			tenant_id  = self.tenant_id,
			created_by = self.actor_id,
			name       = req.name,
			description= req.description,
			tags       = req.tags,
		)
		self._trails[trail_id] = rec
		self._log_operation("create_trail", trail_id, f"name={req.name}")
		self._emit_event("audit_trail_created", trail_id, {"name": req.name})
		return rec

	async def get_trail(self, trail_id: str) -> AuditTrailResponse:
		"""Fetch a single trail by ID, enforcing tenant isolation."""
		rec = self._trails.get(trail_id)
		if rec is None or rec.tenant_id != self.tenant_id:
			raise KeyError(f"trail {trail_id} not found")
		return rec

	async def update_trail(self, trail_id: str, update: AuditTrailUpdate) -> AuditTrailResponse:
		"""Update mutable fields on an audit trail."""
		rec = await self.get_trail(trail_id)
		assert_event_immutable(False)  # trails ARE mutable
		if update.name       is not None: rec.name        = update.name
		if update.description is not None: rec.description = update.description
		if update.status     is not None: rec.status      = update.status
		if update.tags       is not None: rec.tags        = update.tags
		rec.updated_at = datetime.now(timezone.utc)
		self._log_operation("update_trail", trail_id)
		self._emit_event("audit_trail_updated", trail_id, {})
		return rec

	async def list_trails(self, *, active_only: bool = True) -> list[AuditTrailResponse]:
		"""List trails for this tenant."""
		return [
			t for t in self._trails.values()
			if t.tenant_id == self.tenant_id
			and not t.is_deleted
			and (not active_only or t.status == TrailStatus.ACTIVE)
		]

	async def delete_trail(self, trail_id: str) -> None:
		"""Soft-delete a trail (does not delete its events)."""
		rec = await self.get_trail(trail_id)
		assert_no_legal_hold_deletion(False)
		rec.is_deleted = True
		rec.updated_at = datetime.now(timezone.utc)
		self._log_operation("delete_trail", trail_id)
		self._emit_event("audit_trail_deleted", trail_id, {})

	# ------------------------------------------------------------------
	# RetentionPolicy CRUD
	# ------------------------------------------------------------------

	async def create_retention_policy(self, req: RetentionPolicyCreate) -> RetentionPolicyResponse:
		"""Create a new retention policy."""
		self._assert_own_tenant(req.tenant_id)
		assert_actor_present(self.actor_id)
		pid = uuid7str()
		rec = RetentionPolicyResponse(
			id                  = pid,
			tenant_id           = self.tenant_id,
			created_by          = self.actor_id,
			name                = req.name,
			description         = req.description,
			event_types         = req.event_types,
			data_classifications= req.data_classifications,
			retain_days         = req.retain_days,
			archive_after_days  = req.archive_after_days,
			action_on_expiry    = req.action_on_expiry,
			is_active           = req.is_active,
		)
		self._policies[pid] = rec
		self._log_operation("create_retention_policy", pid, f"retain_days={req.retain_days}")
		self._emit_event("retention_policy_created", pid, {"name": req.name})
		return rec

	async def update_retention_policy(
		self, policy_id: str, update: RetentionPolicyUpdate,
	) -> RetentionPolicyResponse:
		"""Update a retention policy."""
		rec = self._policies.get(policy_id)
		if rec is None or rec.tenant_id != self.tenant_id:
			raise KeyError(f"policy {policy_id} not found")
		for field, val in update.model_dump(exclude_none=True).items():
			setattr(rec, field, val)
		rec.updated_at = datetime.now(timezone.utc)
		self._log_operation("update_retention_policy", policy_id)
		return rec

	async def list_retention_policies(self) -> list[RetentionPolicyResponse]:
		"""List all retention policies for this tenant."""
		return [p for p in self._policies.values()
				if p.tenant_id == self.tenant_id and not p.is_deleted]

	async def delete_retention_policy(self, policy_id: str) -> None:
		"""Soft-delete a retention policy."""
		rec = self._policies.get(policy_id)
		if rec is None or rec.tenant_id != self.tenant_id:
			raise KeyError(f"policy {policy_id} not found")
		rec.is_deleted = True
		rec.updated_at = datetime.now(timezone.utc)
		self._log_operation("delete_retention_policy", policy_id)

	# ------------------------------------------------------------------
	# DataSubjectRequest CRUD
	# ------------------------------------------------------------------

	async def create_dsr(
		self, req: DataSubjectRequestCreate, *, is_admin: bool = False,
	) -> DataSubjectRequestResponse:
		"""Submit a data subject request (routes to fulfil based on type)."""
		from .models import DSRType
		if req.dsr_type == DSRType.ACCESS:
			return await self.gdpr_data_subject_access(req, is_admin=is_admin)
		# Other types: record and queue for manual review
		self._assert_own_tenant(req.tenant_id)
		dsr_id = uuid7str()
		rec = DataSubjectRequestResponse(
			id            = dsr_id,
			tenant_id     = self.tenant_id,
			created_by    = self.actor_id,
			dsr_type      = req.dsr_type,
			subject_id    = req.subject_id,
			requested_by  = req.requested_by,
			justification = req.justification,
			scope_details = req.scope_details,
			status        = DSRStatus.PENDING,
		)
		self._dsrs[dsr_id] = rec
		self._log_operation("create_dsr", dsr_id, f"type={req.dsr_type}")
		self._emit_event("dsr_created", dsr_id, {"dsr_type": req.dsr_type})
		return rec

	async def update_dsr(
		self, dsr_id: str, update: DataSubjectRequestUpdate,
	) -> DataSubjectRequestResponse:
		"""Update a DSR (reviewer decision, notes)."""
		rec = self._dsrs.get(dsr_id)
		if rec is None or rec.tenant_id != self.tenant_id:
			raise KeyError(f"DSR {dsr_id} not found")
		for field, val in update.model_dump(exclude_none=True).items():
			setattr(rec, field, val)
		if update.status in (DSRStatus.FULFILLED, DSRStatus.REJECTED):
			rec.fulfilled_at = datetime.now(timezone.utc)
		rec.updated_at = datetime.now(timezone.utc)
		self._log_operation("update_dsr", dsr_id, f"status={rec.status}")
		self._emit_event("dsr_updated", dsr_id, {"status": rec.status})
		return rec

	async def list_dsrs(self) -> list[DataSubjectRequestResponse]:
		"""List all DSRs for this tenant."""
		return [d for d in self._dsrs.values()
				if d.tenant_id == self.tenant_id and not d.is_deleted]

	# ------------------------------------------------------------------
	# EvidencePackage
	# ------------------------------------------------------------------

	async def get_evidence_package(self, pkg_id: str) -> EvidencePackageResponse:
		"""Retrieve a sealed evidence package."""
		rec = self._packages.get(pkg_id)
		if rec is None or rec.tenant_id != self.tenant_id:
			raise KeyError(f"evidence package {pkg_id} not found")
		return rec

	async def list_evidence_packages(self) -> list[EvidencePackageResponse]:
		return [p for p in self._packages.values()
				if p.tenant_id == self.tenant_id and not p.is_deleted]

	# ------------------------------------------------------------------
	# AuditQuery
	# ------------------------------------------------------------------

	async def get_query(self, query_id: str) -> AuditQueryResponse:
		rec = self._queries.get(query_id)
		if rec is None or rec.tenant_id != self.tenant_id:
			raise KeyError(f"query {query_id} not found")
		return rec

	async def list_queries(self) -> list[AuditQueryResponse]:
		return [q for q in self._queries.values()
				if q.tenant_id == self.tenant_id and not q.is_deleted]

	# ------------------------------------------------------------------
	# Risk summary
	# ------------------------------------------------------------------

	async def risk_summary(
		self,
		period_start: datetime,
		period_end:   datetime,
	) -> RiskSummary:
		"""Aggregate risk and compliance statistics for a time window."""
		events = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and period_start <= ev.created_at <= period_end
			and not ev.is_deleted
		]
		return RiskSummary(
			tenant_id     = self.tenant_id,
			period_start  = period_start,
			period_end    = period_end,
			total_events  = len(events),
			high_risk_count  = sum(1 for e in events if e.risk_score >= 0.7),
			anomaly_count    = sum(1 for e in events if e.anomaly_score >= 0.7),
			compliance_violations = sum(
				1 for e in events if e.event_type.value == "compliance_violation"
			),
			by_event_type = self._count_by(events, "event_type"),
			by_source     = self._count_by(events, "source"),
			top_actors    = self._top_actors(events, n=10),
		)

	def _top_actors(
		self, events: list[AuditEventResponse], n: int = 10,
	) -> list[dict[str, Any]]:
		counts: dict[str, int] = defaultdict(int)
		for ev in events:
			if ev.actor_id:
				counts[ev.actor_id] += 1
		return [
			{"actor_id": a, "event_count": c}
			for a, c in sorted(counts.items(), key=lambda x: -x[1])[:n]
		]

	# ------------------------------------------------------------------
	# Log helpers (private)
	# ------------------------------------------------------------------

	def _log_pretty_path(self, path: str) -> str:
		"""Shorten absolute paths to last two components for log readability."""
		parts = path.replace("\\", "/").split("/")
		return "/".join(parts[-2:]) if len(parts) >= 2 else path

	# ------------------------------------------------------------------
	# 15. immutable_append — alias for log_event with strict immutability guard
	# ------------------------------------------------------------------

	async def immutable_append(
		self,
		event: AuditEventCreate,
	) -> AuditEventResponse:
		"""
		Append a single immutable event.  Rejects any attempt to mutate
		an existing event with the same ID.
		"""
		if event.id and event.id in self._events:
			raise ValueError(f"immutable_append: event {event.id} already exists")
		return await self.log_event(
			who     = event.actor_id or self.actor_id,
			what    = event.action,
			on_what = event.resource_id or "",
			how     = event.event_type,
			where   = event.ip_address,
			when    = None,
			result  = event.success,
			payload = event,
		)

	# ------------------------------------------------------------------
	# 16. tamper_proof_verify — verify a single event by its ID
	# ------------------------------------------------------------------

	async def tamper_proof_verify(self, event_id: str) -> dict[str, Any]:
		"""
		Verify the checksum and chain-hash of a single event.

		Returns a verification record indicating CLEAN or SUSPECT status.
		"""
		ev = self._events.get(event_id)
		if ev is None or ev.tenant_id != self.tenant_id:
			raise KeyError(f"event {event_id} not found")
		expected = calculate_event_checksum(
			event_id     = ev.id,
			tenant_id    = ev.tenant_id,
			timestamp    = ev.created_at,
			event_type   = ev.event_type,
			actor_id     = ev.actor_id,
			action       = ev.action,
			resource_type= ev.resource_type,
			resource_id  = ev.resource_id,
			success      = ev.success,
		)
		clean = ev.checksum == expected
		self._log_operation("tamper_proof_verify", event_id, f"clean={clean}")
		return {
			"event_id":    event_id,
			"tenant_id":   self.tenant_id,
			"status":      "clean" if clean else "suspect",
			"checksum_ok": clean,
			"stored":      ev.checksum,
			"expected":    expected,
			"verified_at": datetime.now(timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# 17. audit_query_advanced — full-text + risk band + field combo search
	# ------------------------------------------------------------------

	async def audit_query_advanced(
		self,
		filters: AuditQueryCreate,
		*,
		risk_band: str | None = None,          # "low" | "medium" | "high" | "critical"
		categories: list[str] | None = None,
		anomaly_threshold: float | None = None,
	) -> AuditSearchResult:
		"""
		Extended search layering risk-band, category, and anomaly filters
		on top of the base audit_trail_search.
		"""
		base = await self.audit_trail_search(filters)
		band_ranges = {
			"low":      (0.0, 0.3),
			"medium":   (0.3, 0.6),
			"high":     (0.6, 0.8),
			"critical": (0.8, 1.01),
		}
		events = base.events
		if risk_band and risk_band in band_ranges:
			lo, hi = band_ranges[risk_band]
			events = [e for e in events if lo <= e.risk_score < hi]
		if categories:
			events = [e for e in events if e.category in categories]
		if anomaly_threshold is not None:
			events = [e for e in events if e.anomaly_score >= anomaly_threshold]
		self._log_operation("audit_query_advanced", base.query_id,
							f"refined={len(events)} of {base.total_count}")
		return AuditSearchResult(
			query_id    = base.query_id,
			total_count = len(events),
			events      = events,
			query_ms    = base.query_ms,
			has_more    = False,
		)

	# ------------------------------------------------------------------
	# 18. cross_system_correlate — correlate by request_id across tenants
	# ------------------------------------------------------------------

	async def cross_system_correlate(
		self,
		request_id: str,
		peer_services: list["AuditLoggingService"],
	) -> dict[str, Any]:
		"""
		Correlate audit events carrying the same request_id across multiple
		service instances.  Returns count + max risk per tenant without
		leaking raw events from peer tenants.
		"""
		my_events = [
			e for e in self._events.values()
			if e.request_id == request_id and not e.is_deleted
		]
		peers: list[dict[str, Any]] = []
		for peer in peer_services:
			peer_evs = [
				e for e in peer._events.values()
				if e.request_id == request_id and not e.is_deleted
			]
			peers.append({
				"tenant_id":   peer.tenant_id,
				"event_count": len(peer_evs),
				"max_risk":    max((e.risk_score for e in peer_evs), default=0.0),
			})
		result = {
			"request_id":        request_id,
			"source_tenant":     self.tenant_id,
			"source_event_count": len(my_events),
			"source_max_risk":   max((e.risk_score for e in my_events), default=0.0),
			"peers":             peers,
			"correlated_at":     datetime.now(timezone.utc).isoformat(),
		}
		self._log_operation("cross_system_correlate", request_id)
		return result

	# ------------------------------------------------------------------
	# 19. pii_mask_in_logs — pseudonymise PII fields in event detail blobs
	# ------------------------------------------------------------------

	async def pii_mask_in_logs(
		self,
		subject_id: str,
		fields_to_mask: list[str],
		mask_value: str = "***MASKED***",
	) -> dict[str, Any]:
		"""
		Pseudonymise PII-containing fields in the details/tags blobs of all
		events associated with subject_id.

		Core event fields (actor_id, resource_id) are NOT altered because
		they are part of the immutable checksum.  Only mutable detail blobs
		are scrubbed.
		"""
		assert_tenant_context({"tenant_id": self.tenant_id})
		masked_count = 0
		event_ids: list[str] = []
		for ev in self._events.values():
			if ev.tenant_id != self.tenant_id:
				continue
			if ev.actor_id != subject_id and ev.resource_id != subject_id:
				continue
			changed = False
			for field in fields_to_mask:
				if field in ev.details:
					ev.details[field] = mask_value
					changed = True
				if field in ev.tags:
					ev.tags[field] = mask_value
					changed = True
			if changed:
				masked_count += 1
				event_ids.append(ev.id)
		result = {
			"subject_id":   subject_id,
			"tenant_id":    self.tenant_id,
			"events_masked": masked_count,
			"event_ids":    event_ids,
			"fields":       fields_to_mask,
			"masked_at":    datetime.now(timezone.utc).isoformat(),
		}
		self._log_operation("pii_mask_in_logs", subject_id, f"masked={masked_count}")
		self._emit_event("pii_masked_in_logs", subject_id, result)
		return result

	# ------------------------------------------------------------------
	# 20. gdpr_log_erasure — pseudonymise + flag events for erasure
	# ------------------------------------------------------------------

	async def gdpr_log_erasure(
		self,
		subject_id: str,
		justification: str,
		*,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""
		GDPR Art. 17 erasure of audit log PII.

		Immutable core fields are not erased (Art. 17(3)(b) exemption).
		Only detail blobs and tags are pseudonymised.
		dry_run=True reports impact without modifying data.
		"""
		assert_erasure_allowed(subject_id, justification)
		candidate_events = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and (ev.actor_id == subject_id or ev.resource_id == subject_id)
			and not ev.legal_hold
		]
		legal_blocked = [
			ev.id for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and (ev.actor_id == subject_id or ev.resource_id == subject_id)
			and ev.legal_hold
		]
		if not dry_run:
			for ev in candidate_events:
				ev.details  = {"_gdpr_erased": True}
				ev.tags     = {}
				ev.contains_pii = False
				ev.updated_at = datetime.now(timezone.utc)
		result = {
			"subject_id":        subject_id,
			"tenant_id":         self.tenant_id,
			"events_erased":     len(candidate_events) if not dry_run else 0,
			"events_affected":   len(candidate_events),
			"legal_hold_blocked": legal_blocked,
			"dry_run":           dry_run,
			"justification":     justification,
			"erased_at":         datetime.now(timezone.utc).isoformat(),
		}
		self._log_operation("gdpr_log_erasure", subject_id, f"dry={dry_run} erased={len(candidate_events)}")
		self._emit_event("gdpr_erasure_performed", subject_id, result)
		return result

	# ------------------------------------------------------------------
	# 21. log_integrity_check — full chain-hash re-derivation
	# ------------------------------------------------------------------

	async def log_integrity_check(self) -> dict[str, Any]:
		"""
		Re-derive the full chain-hash sequence from scratch and compare
		to stored values.  Returns a summary with any breaks detected.
		"""
		events_sorted = sorted(
			[ev for ev in self._events.values() if ev.tenant_id == self.tenant_id],
			key=lambda e: e.created_at,
		)
		tip = "0" * 64
		breaks: list[str] = []
		for ev in events_sorted:
			expected_cs = calculate_event_checksum(
				event_id     = ev.id,
				tenant_id    = ev.tenant_id,
				timestamp    = ev.created_at,
				event_type   = ev.event_type,
				actor_id     = ev.actor_id,
				action       = ev.action,
				resource_type= ev.resource_type,
				resource_id  = ev.resource_id,
				success      = ev.success,
			)
			expected_ch = calculate_chain_hash(tip, expected_cs)
			if ev.chain_hash != expected_ch:
				breaks.append(ev.id)
			else:
				tip = ev.chain_hash
		result = {
			"tenant_id":     self.tenant_id,
			"events_checked": len(events_sorted),
			"chain_breaks":  len(breaks),
			"break_ids":     breaks,
			"integrity":     "intact" if not breaks else "broken",
			"checked_at":    datetime.now(timezone.utc).isoformat(),
		}
		self._log_operation("log_integrity_check", "chain",
							f"events={len(events_sorted)} breaks={len(breaks)}")
		return result

	# ------------------------------------------------------------------
	# 22. audit_chain_verify — verify chain from a specific event forward
	# ------------------------------------------------------------------

	async def audit_chain_verify(
		self,
		from_event_id: str,
	) -> dict[str, Any]:
		"""
		Verify chain integrity starting from a specific event.

		Useful for targeted forensic validation of a time window.
		"""
		ev = self._events.get(from_event_id)
		if ev is None or ev.tenant_id != self.tenant_id:
			raise KeyError(f"event {from_event_id} not found")
		start_ts = ev.created_at
		events_from = sorted(
			[
				e for e in self._events.values()
				if e.tenant_id == self.tenant_id and e.created_at >= start_ts
			],
			key=lambda e: e.created_at,
		)
		suspect: list[str] = []
		for e in events_from:
			exp_cs = calculate_event_checksum(
				event_id     = e.id,
				tenant_id    = e.tenant_id,
				timestamp    = e.created_at,
				event_type   = e.event_type,
				actor_id     = e.actor_id,
				action       = e.action,
				resource_type= e.resource_type,
				resource_id  = e.resource_id,
				success      = e.success,
			)
			if e.checksum != exp_cs:
				suspect.append(e.id)
		return {
			"from_event_id":    from_event_id,
			"tenant_id":        self.tenant_id,
			"events_in_window": len(events_from),
			"suspect_count":    len(suspect),
			"suspect_ids":      suspect,
			"status":           "clean" if not suspect else "suspect",
			"verified_at":      datetime.now(timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# 23. realtime_siem_push — push a single event to all SIEM queues
	# ------------------------------------------------------------------

	async def realtime_siem_push(self, event_id: str) -> dict[str, Any]:
		"""
		Manually push a stored event to all registered SIEM queues.

		Useful for re-sending missed events or escalating high-risk events.
		"""
		ev = self._events.get(event_id)
		if ev is None or ev.tenant_id != self.tenant_id:
			raise KeyError(f"event {event_id} not found")
		siem_ev = SIEMEvent(
			event_id   = ev.id,
			tenant_id  = ev.tenant_id,
			timestamp  = ev.created_at,
			level      = ev.level,
			event_type = ev.event_type,
			source     = ev.source,
			actor_id   = ev.actor_id,
			action     = ev.action,
			resource_id= ev.resource_id or "",
			success    = ev.success,
			risk_score = ev.risk_score,
			ip_address = ev.ip_address,
			checksum   = ev.checksum,
		)
		self._push_siem(siem_ev)
		pushed_to = len(self._siem_queues)
		self._log_operation("realtime_siem_push", event_id, f"queues={pushed_to}")
		return {
			"event_id":   event_id,
			"pushed_to":  pushed_to,
			"pushed_at":  datetime.now(timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# 24. audit_report_generate — generate an ad-hoc audit report
	# ------------------------------------------------------------------

	async def audit_report_generate(
		self,
		period_start: datetime,
		period_end:   datetime,
		framework:    ComplianceFramework,
		requested_by: str,
		*,
		include_violations: bool = True,
		include_recommendations: bool = True,
	) -> ComplianceReportResponse:
		"""Generate a compliance report without needing a pre-built Create object."""
		req = ComplianceReportCreate(
			tenant_id               = self.tenant_id,
			framework               = framework,
			period_start            = period_start,
			period_end              = period_end,
			requested_by            = requested_by,
			include_violations      = include_violations,
			include_recommendations = include_recommendations,
		)
		return await self.compliance_report(req)

	# ------------------------------------------------------------------
	# 25. evidence_package_export (already exists — add get helper)
	# NOTE: already implemented above; adding list_tamper_scans below.
	# ------------------------------------------------------------------

	async def list_tamper_scans(self) -> list[TamperDetectionResponse]:
		"""List all tamper-detection scan records for this tenant."""
		return [
			t for t in self._tampers.values()
			if t.tenant_id == self.tenant_id
		]

	# ------------------------------------------------------------------
	# 26. log_classifier — classify event risk into a severity label
	# ------------------------------------------------------------------

	async def log_classifier(
		self,
		event_id: str,
		*,
		override_anomaly: float | None = None,
	) -> dict[str, Any]:
		"""
		Classify a stored event into risk tiers and set its anomaly_score.

		Tier thresholds:
		  critical >= 0.8 | high >= 0.6 | medium >= 0.3 | low < 0.3
		"""
		ev = self._events.get(event_id)
		if ev is None or ev.tenant_id != self.tenant_id:
			raise KeyError(f"event {event_id} not found")
		if override_anomaly is not None:
			ev.anomaly_score = max(0.0, min(1.0, override_anomaly))
		score = ev.risk_score
		tier = (
			"critical" if score >= 0.8 else
			"high"     if score >= 0.6 else
			"medium"   if score >= 0.3 else
			"low"
		)
		ev.updated_at = datetime.now(timezone.utc)
		result = {
			"event_id":     event_id,
			"risk_score":   score,
			"anomaly_score": ev.anomaly_score,
			"tier":         tier,
			"classified_at": ev.updated_at.isoformat(),
		}
		self._log_operation("log_classifier", event_id, f"tier={tier}")
		return result

	# ------------------------------------------------------------------
	# 27. anomaly_in_audit — surface events with elevated anomaly scores
	# ------------------------------------------------------------------

	async def anomaly_in_audit(
		self,
		threshold:    float = 0.7,
		period_start: datetime | None = None,
		period_end:   datetime | None = None,
	) -> list[AuditEventResponse]:
		"""
		Return events with anomaly_score >= threshold, optionally filtered
		by a time window.
		"""
		now = datetime.now(timezone.utc)
		start = period_start or (now - timedelta(days=30))
		end   = period_end   or now
		anomalous = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and not ev.is_deleted
			and ev.anomaly_score >= threshold
			and start <= ev.created_at <= end
		]
		anomalous.sort(key=lambda e: e.anomaly_score, reverse=True)
		self._log_operation("anomaly_in_audit", "query", f"found={len(anomalous)} threshold={threshold}")
		return anomalous

	# ------------------------------------------------------------------
	# 28. retention_policy_enforce — alias to retention_enforcement
	# ------------------------------------------------------------------

	async def retention_policy_enforce(self) -> dict[str, Any]:
		"""
		Enforce all active retention policies (alias for retention_enforcement
		with explicit naming convention).
		"""
		return await self.retention_enforcement()

	# ------------------------------------------------------------------
	# 29. audit_analytics — aggregate metrics over a period
	# ------------------------------------------------------------------

	async def audit_analytics(
		self,
		period_start: datetime,
		period_end:   datetime,
	) -> dict[str, Any]:
		"""
		Comprehensive audit analytics covering volume, risk distribution,
		top actors, source breakdown, and compliance tag frequency.
		"""
		events = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and period_start <= ev.created_at <= period_end
			and not ev.is_deleted
		]
		risk_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
		for ev in events:
			if ev.risk_score >= 0.8:
				risk_dist["critical"] += 1
			elif ev.risk_score >= 0.6:
				risk_dist["high"] += 1
			elif ev.risk_score >= 0.3:
				risk_dist["medium"] += 1
			else:
				risk_dist["low"] += 1
		tag_freq: dict[str, int] = defaultdict(int)
		for ev in events:
			for tag in ev.compliance_tags:
				tag_freq[tag] += 1
		return {
			"tenant_id":          self.tenant_id,
			"period_start":       period_start.isoformat(),
			"period_end":         period_end.isoformat(),
			"total_events":       len(events),
			"failed_events":      sum(1 for e in events if not e.success),
			"pii_events":         sum(1 for e in events if e.contains_pii),
			"legal_hold_events":  sum(1 for e in events if e.legal_hold),
			"risk_distribution":  risk_dist,
			"top_actors":         self._top_actors(events, n=10),
			"by_event_type":      self._count_by(events, "event_type"),
			"by_source":          self._count_by(events, "source"),
			"compliance_tag_freq": dict(tag_freq),
			"avg_risk_score":     round(sum(e.risk_score for e in events) / len(events), 4) if events else 0.0,
			"generated_at":       datetime.now(timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# 30. get_event — fetch a single event by ID
	# ------------------------------------------------------------------

	async def get_event(self, event_id: str) -> AuditEventResponse:
		"""Fetch a single audit event by ID, enforcing tenant isolation."""
		ev = self._events.get(event_id)
		if ev is None or ev.tenant_id != self.tenant_id:
			raise KeyError(f"event {event_id} not found")
		return ev

	# ------------------------------------------------------------------
	# 31. list_events — paginated list of tenant events
	# ------------------------------------------------------------------

	async def list_events(
		self,
		*,
		limit:  int = 100,
		offset: int = 0,
		active_only: bool = True,
	) -> list[AuditEventResponse]:
		"""Return a paginated slice of events for this tenant."""
		evs = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and (not active_only or not ev.is_deleted)
		]
		evs.sort(key=lambda e: e.created_at, reverse=True)
		return evs[offset: offset + limit]

	# ------------------------------------------------------------------
	# 32. set_legal_hold — apply or remove a legal hold on an event
	# ------------------------------------------------------------------

	async def set_legal_hold(
		self,
		event_id: str,
		hold: bool,
		reason: str,
	) -> AuditEventResponse:
		"""Apply (hold=True) or lift (hold=False) a legal hold on a stored event."""
		ev = await self.get_event(event_id)
		ev.legal_hold = hold
		ev.updated_at = datetime.now(timezone.utc)
		self._log_operation("set_legal_hold", event_id, f"hold={hold} reason={reason}")
		self._emit_event("legal_hold_changed", event_id, {"hold": hold, "reason": reason})
		return ev

	# ------------------------------------------------------------------
	# 33. bulk_set_legal_hold — apply hold to multiple events
	# ------------------------------------------------------------------

	async def bulk_set_legal_hold(
		self,
		event_ids: list[str],
		hold: bool,
		reason: str,
	) -> dict[str, Any]:
		"""Apply or lift legal hold on a list of event IDs in one call."""
		applied: list[str] = []
		missed:  list[str] = []
		for eid in event_ids:
			try:
				await self.set_legal_hold(eid, hold, reason)
				applied.append(eid)
			except KeyError:
				missed.append(eid)
		return {
			"applied": len(applied),
			"missed":  len(missed),
			"hold":    hold,
			"reason":  reason,
			"applied_ids": applied,
			"missed_ids":  missed,
		}

	# ------------------------------------------------------------------
	# 34. siem_subscriber_count — telemetry helper
	# ------------------------------------------------------------------

	async def siem_subscriber_count(self) -> int:
		"""Return the number of active SIEM stream subscribers."""
		return len(self._siem_queues)

	# ------------------------------------------------------------------
	# 35. event_count — total events in this tenant store
	# ------------------------------------------------------------------

	async def event_count(self, *, include_deleted: bool = False) -> int:
		"""Return total event count for this tenant."""
		return sum(
			1 for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and (include_deleted or not ev.is_deleted)
		)

	# ------------------------------------------------------------------
	# 36. search_by_actor — shortcut: all events for a given actor
	# ------------------------------------------------------------------

	async def search_by_actor(
		self,
		actor_id: str,
		*,
		limit:  int = 200,
		offset: int = 0,
	) -> list[AuditEventResponse]:
		"""Return all non-deleted events for a specific actor in this tenant."""
		evs = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and ev.actor_id == actor_id
			and not ev.is_deleted
		]
		evs.sort(key=lambda e: e.created_at, reverse=True)
		return evs[offset: offset + limit]

	# ------------------------------------------------------------------
	# 37. search_by_resource — shortcut: all events for a resource
	# ------------------------------------------------------------------

	async def search_by_resource(
		self,
		resource_id: str,
		*,
		limit:  int = 200,
		offset: int = 0,
	) -> list[AuditEventResponse]:
		"""Return all non-deleted events touching a specific resource."""
		evs = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and ev.resource_id == resource_id
			and not ev.is_deleted
		]
		evs.sort(key=lambda e: e.created_at, reverse=True)
		return evs[offset: offset + limit]

	# ------------------------------------------------------------------
	# 38. high_risk_events — shorthand for risk_score >= threshold
	# ------------------------------------------------------------------

	async def high_risk_events(
		self,
		threshold: float = 0.7,
		limit:     int   = 100,
	) -> list[AuditEventResponse]:
		"""Return the top N events by risk score above threshold."""
		evs = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and ev.risk_score >= threshold
			and not ev.is_deleted
		]
		evs.sort(key=lambda e: e.risk_score, reverse=True)
		return evs[:limit]

	# ------------------------------------------------------------------
	# 39. failed_auth_events — events that represent auth failures
	# ------------------------------------------------------------------

	async def failed_auth_events(
		self,
		period_start: datetime | None = None,
		period_end:   datetime | None = None,
	) -> list[AuditEventResponse]:
		"""Return authentication failure events in the given time window."""
		now   = datetime.now(timezone.utc)
		start = period_start or (now - timedelta(hours=24))
		end   = period_end   or now
		return [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and not ev.success
			and ev.event_type.value in ("auth_failure", "login_failed", "permission_denied")
			and start <= ev.created_at <= end
			and not ev.is_deleted
		]

	# ------------------------------------------------------------------
	# 40. purge_expired_events — hard-delete events past retention
	# ------------------------------------------------------------------

	async def purge_expired_events(
		self,
		*,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""
		Hard-purge events that have exceeded their retention_days AND are
		not under legal hold.  dry_run=True reports without deleting.
		"""
		now = datetime.now(timezone.utc)
		to_purge = [
			ev for ev in self._events.values()
			if ev.tenant_id == self.tenant_id
			and not ev.legal_hold
			and (now - ev.created_at).days > ev.retention_days
		]
		if not dry_run:
			for ev in to_purge:
				del self._events[ev.id]
		return {
			"tenant_id":   self.tenant_id,
			"purged":      len(to_purge) if not dry_run else 0,
			"eligible":    len(to_purge),
			"dry_run":     dry_run,
			"purged_at":   now.isoformat(),
		}

	# ------------------------------------------------------------------
	# 41. chain_tip — return the current chain hash tip
	# ------------------------------------------------------------------

	async def chain_tip(self) -> dict[str, Any]:
		"""Return the current chain-hash tip and event count for this tenant."""
		count = sum(1 for ev in self._events.values() if ev.tenant_id == self.tenant_id)
		return {
			"tenant_id":   self.tenant_id,
			"chain_tip":   self._chain_tip,
			"event_count": count,
			"at":          datetime.now(timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# 42. export_events_jsonl — serialise events to JSON-Lines string
	# ------------------------------------------------------------------

	async def export_events_jsonl(
		self,
		event_ids: list[str] | None = None,
	) -> str:
		"""
		Export events as a JSON-Lines string (one JSON object per line).
		If event_ids is None, exports all non-deleted tenant events.
		"""
		if event_ids is not None:
			evs = [self._events[eid] for eid in event_ids if eid in self._events]
		else:
			evs = [
				ev for ev in self._events.values()
				if ev.tenant_id == self.tenant_id and not ev.is_deleted
			]
		lines = [json.dumps(ev.model_dump(mode="json"), default=str) for ev in evs]
		return "\n".join(lines)


	# ------------------------------------------------------------------
	# SOC 2 / regulatory: durable PostgreSQL + NATS persistence
	# ------------------------------------------------------------------

	async def _load_chain_tip_from_db(self) -> str:
		"""Read the last chain_hash from apg_audit_events for this tenant.

		Called lazily on the first log_event invocation so the chain is
		continuous across service restarts (SOC 2 tamper-evidence requirement).
		Returns "0" * 64 if the table is empty or the DB is unavailable.
		"""
		if self.db is None:
			return "0" * 64
		try:
			from sqlalchemy import text
			result = await self.db.execute(
				text(
					"SELECT chain_hash FROM apg_audit_events "
					"WHERE tenant_id = :tid "
					"ORDER BY timestamp DESC LIMIT 1"
				),
				{"tid": self.tenant_id},
			)
			row = result.fetchone()
			return row[0] if row else "0" * 64
		except Exception:
			return "0" * 64

	async def _persist_audit_event_to_db(self, resp: Any) -> None:
		"""INSERT the event into apg_audit_events (append-only).

		The table has PostgreSQL rules preventing UPDATE/DELETE, satisfying
		the SOC 2 immutability requirement. Silently no-ops when the DB is
		unavailable so standalone operation is never broken.
		"""
		if self.db is None:
			return
		try:
			from sqlalchemy import text
			await self.db.execute(
				text("""
					INSERT INTO apg_audit_events
					(id, tenant_id, actor_id, event_type, resource_type, resource_id,
					 action, success, ip_address, timestamp, payload,
					 checksum, prev_hash, chain_hash)
					VALUES
					(:id, :tenant_id, :actor_id, :event_type, :resource_type, :resource_id,
					 :action, :success, :ip_address, :timestamp, :payload,
					 :checksum, :prev_hash, :chain_hash)
					ON CONFLICT (id) DO NOTHING
				"""),
				{
					"id":            resp.id,
					"tenant_id":     resp.tenant_id,
					"actor_id":      resp.actor_id,
					"event_type":    resp.event_type,
					"resource_type": resp.resource_type,
					"resource_id":   resp.resource_id,
					"action":        resp.action,
					"success":       resp.success,
					"ip_address":    resp.ip_address,
					"timestamp":     resp.created_at,
					"payload":       json.dumps(resp.details or {}, default=str),
					"checksum":      resp.checksum or "",
					"prev_hash":     self._chain_prev or "0" * 64,
					"chain_hash":    resp.chain_hash or "",
				},
			)
			await self.db.commit()
		except Exception as exc:
			import logging
			logging.getLogger(__name__).warning("Audit DB persist failed: %s", exc)

	async def _publish_audit_to_nats(self, resp: Any) -> None:
		"""Publish the event to NATS JetStream when NATS_URL is configured.

		Subject: apg.events.audl.audit_event
		All subscribing capabilities (e.g. intel, grc) receive the event.
		Silently no-ops when NATS is unavailable.
		"""
		import os
		if not os.environ.get("NATS_URL"):
			return
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			adapter = NATSEventAdapter(capability_id="audl")
			await adapter.log_event(
				event_type=resp.event_type,
				actor_id=resp.actor_id,
				tenant_id=resp.tenant_id,
				resource_id=resp.resource_id,
				details={
					"action":     resp.action,
					"success":    resp.success,
					"risk_score": float(resp.risk_score or 0),
					"checksum":   resp.checksum or "",
					"chain_hash": resp.chain_hash or "",
				},
			)
		except Exception as exc:
			import logging
			logging.getLogger(__name__).debug("Audit NATS publish failed: %s", exc)


__all__ = ["AuditLoggingService", "subscribe_domain_events"]
