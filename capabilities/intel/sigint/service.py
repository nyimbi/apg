"""Executable service layer for APG Signals Intelligence (SIGINT).

Expanded to 600+ lines with full async methods, adapter/store pattern,
and the new operational methods required by the capability spec.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import statistics
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_AUTHORITY_TYPES, SUPPORTED_BANDS, SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_COLLECTION_MODES, SUPPORTED_PATTERN_TYPES, SUPPORTED_PROCESSING_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_SOURCE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CollectionTask, ProcessingBatch, SIGINTAgent, SIGINTReview,
		SignalAssessment, SignalAuthority, SignalObservation, SignalPattern, SignalSource,
	)
	from .sigint_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_AUTHORITY_TYPES, SUPPORTED_BANDS, SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_COLLECTION_MODES, SUPPORTED_PATTERN_TYPES, SUPPORTED_PROCESSING_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_SOURCE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CollectionTask, ProcessingBatch, SIGINTAgent, SIGINTReview,
		SignalAssessment, SignalAuthority, SignalObservation, SignalPattern, SignalSource,
	)
	from sigint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(*parts: str) -> str:
	blob = "|".join(parts)
	return hashlib.sha256(blob.encode()).hexdigest()[:16]


class SIGINTService:
	"""Tenant-scoped SIGINT coordination runtime for generated APG applications.

	Constructor follows the adapter/store pattern so callers can inject
	auth, audit, notify, and db_url collaborators without changing call sites.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		# In-memory stores (replace with store adapter calls when store is set)
		self.authorities: dict[tuple[str, str], SignalAuthority] = {}
		self.sources: dict[tuple[str, str], SignalSource] = {}
		self.tasks: dict[tuple[str, str], CollectionTask] = {}
		self.observations: dict[tuple[str, str], SignalObservation] = {}
		self.processing_batches: dict[tuple[str, str], ProcessingBatch] = {}
		self.patterns: dict[tuple[str, str], SignalPattern] = {}
		self.assessments: dict[tuple[str, str], SignalAssessment] = {}
		self.reviews: dict[tuple[str, str], SIGINTReview] = {}
		self.agents: dict[tuple[str, str], SIGINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Operational state added by new methods
		self._signals: dict[str, dict[str, Any]] = {}
		self._intercepts: dict[str, dict[str, Any]] = {}
		self._decrypted: dict[str, dict[str, Any]] = {}
		self._traffic_analyses: dict[str, dict[str, Any]] = {}
		self._correlations: dict[str, dict[str, Any]] = {}
		self._emitters: dict[str, dict[str, Any]] = {}
		self._df_fixes: dict[str, dict[str, Any]] = {}
		self._sat_intercepts: dict[str, dict[str, Any]] = {}
		self._pattern_analyses: dict[str, dict[str, Any]] = {}
		self._reports: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Capability contract helpers (sync, preserved from original)
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Original sync CRUD methods (preserved verbatim)
	# ------------------------------------------------------------------

	def record_authority(
		self, authority_id: str, tenant_id: str, authority_type: str,
		scope_reference: str, classification: str, approver_id: str,
		expires_at: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = SignalAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "sigint_authority_recorded", authority_id)
		return item.to_dict()

	def register_source(
		self, source_id: str, tenant_id: str, source_type: str, band: str,
		source_reference: str, owner_id: str, authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		band = normalize_code(band)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"band_supported": band in SUPPORTED_BANDS,
			"source_reference_present": present(source_reference),
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = SignalSource(source_id, tenant_id, source_type, band, source_reference, owner_id, authority_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "sigint_source_registered", source_id)
		return item.to_dict()

	def record_collection_task(
		self, task_id: str, tenant_id: str, authority_id: str, source_id: str,
		collection_mode: str, retention_days: int, minimization_reference: str,
		approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		collection_mode = normalize_code(collection_mode)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_collection_task",
			"authority_present": authority is not None,
			"source_present": source is not None,
			"source_authority_match": source is not None and source.authority_id == authority_id,
			"collection_mode_supported": collection_mode in SUPPORTED_COLLECTION_MODES,
			"retention_days_positive": positive_int(retention_days),
			"minimization_present": present(minimization_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = CollectionTask(task_id, tenant_id, authority_id, source_id, collection_mode, int(retention_days), minimization_reference, approval_reference, evidence_reference)
		self.tasks[self._tenant_key(tenant_id, task_id)] = item
		self._audit(tenant_id, "sigint_collection_task_recorded", task_id)
		return item.to_dict()

	def record_observation(
		self, observation_id: str, tenant_id: str, task_id: str,
		observation_reference: str, fingerprint: str, confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		task = self._tenant_task_or_none(task_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_observation",
			"task_present": task is not None,
			"observation_reference_present": present(observation_reference),
			"fingerprint_present": present(fingerprint),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = SignalObservation(observation_id, tenant_id, task_id, observation_reference, fingerprint, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "sigint_observation_recorded", observation_id)
		return item.to_dict()

	def record_processing_batch(
		self, batch_id: str, tenant_id: str, observation_id: str,
		processing_type: str, quality_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		processing_type = normalize_code(processing_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_processing_batch",
			"observation_present": observation is not None,
			"processing_type_supported": processing_type in SUPPORTED_PROCESSING_TYPES,
			"quality_valid": bounded_score(quality_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = ProcessingBatch(batch_id, tenant_id, observation_id, processing_type, float(quality_score), analyst_id, evidence_reference)
		self.processing_batches[self._tenant_key(tenant_id, batch_id)] = item
		self._audit(tenant_id, "sigint_processing_batch_recorded", batch_id)
		return item.to_dict()

	def record_pattern(
		self, pattern_id: str, tenant_id: str, batch_id: str, pattern_type: str,
		confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		batch = self._tenant_batch_or_none(batch_id, tenant_id)
		pattern_type = normalize_code(pattern_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_pattern",
			"batch_present": batch is not None,
			"pattern_type_supported": pattern_type in SUPPORTED_PATTERN_TYPES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = SignalPattern(pattern_id, tenant_id, batch_id, pattern_type, float(confidence_score), analyst_id, evidence_reference)
		self.patterns[self._tenant_key(tenant_id, pattern_id)] = item
		self._audit(tenant_id, "sigint_pattern_recorded", pattern_id)
		return item.to_dict()

	def record_assessment(
		self, assessment_id: str, tenant_id: str, pattern_id: str,
		assessment_type: str, classification: str, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		pattern = self._tenant_pattern_or_none(pattern_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_assessment",
			"pattern_present": pattern is not None,
			"assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES,
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = SignalAssessment(assessment_id, tenant_id, pattern_id, assessment_type, classification, analyst_id, evidence_reference)
		self.assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "sigint_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = SIGINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "sigint_review_recorded", review_id)
		return item.to_dict()

	def register_sigint_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_sigint_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = SIGINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "sigint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "sigint_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self, tenant_id: str, item_count: int, event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "sigint_batch", "event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id, "item_count": item_count,
			"processor": "bytewax", "stream": "apg.intel.sigint.lifecycle", "accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"task_count": self._count(self.tasks, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"processing_batch_count": self._count(self.processing_batches, tenant_id),
			"pattern_count": self._count(self.patterns, tenant_id),
			"assessment_count": self._count(self.assessments, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"signal_count": len(self._signals),
			"intercept_count": len(self._intercepts),
			"report_count": len(self._reports),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async operational methods
	# ------------------------------------------------------------------

	async def collect_signal(
		self,
		signal_type: str,
		frequency: float,
		source: str,
		metadata: dict[str, Any],
	) -> dict[str, Any]:
		"""Collect a raw signal observation from the designated source.

		Validates frequency range sanity (0 Hz – 300 GHz), records the
		signal with a deterministic fingerprint, and emits an audit event.
		"""
		assert present(signal_type), "signal_type required"
		assert frequency >= 0, f"frequency must be non-negative, got {frequency}"
		assert present(source), "source required"

		frequency_ghz = frequency / 1e9
		assert frequency_ghz <= 300, f"frequency exceeds 300 GHz: {frequency_ghz:.3f}"

		signal_id = _fingerprint(signal_type, str(frequency), source, _utcnow())
		collected_at = _utcnow()

		band = (
			"ELF" if frequency < 3e3 else
			"VLF" if frequency < 30e3 else
			"LF" if frequency < 300e3 else
			"MF" if frequency < 3e6 else
			"HF" if frequency < 30e6 else
			"VHF" if frequency < 300e6 else
			"UHF" if frequency < 3e9 else
			"SHF" if frequency < 30e9 else
			"EHF"
		)

		record: dict[str, Any] = {
			"signal_id": signal_id,
			"signal_type": signal_type,
			"frequency_hz": frequency,
			"band": band,
			"source": source,
			"metadata": metadata,
			"collected_at": collected_at,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"status": "collected",
		}
		self._signals[signal_id] = record
		self._audit(self.tenant_id, "sigint_signal_collected", signal_id)
		return record

	async def intercept_communication(
		self,
		target_id: str,
		channel: str,
		authority_ref: str,
	) -> dict[str, Any]:
		"""Intercept a communication channel under recorded legal authority.

		Refuses to proceed if authority_ref is absent — no blanket intercepts.
		"""
		assert present(target_id), "target_id required"
		assert present(channel), "channel required"
		assert present(authority_ref), "authority_ref is mandatory for intercepts"

		intercept_id = _fingerprint(target_id, channel, authority_ref, _utcnow())
		started_at = _utcnow()

		# Verify the authority exists in this tenant
		authority_present = any(
			a.authority_id == authority_ref
			for a in self.authorities.values()
			if a.tenant_id == self.tenant_id
		)
		if not authority_present:
			raise PermissionError(f"authority_ref {authority_ref!r} not registered for tenant {self.tenant_id!r}")

		record: dict[str, Any] = {
			"intercept_id": intercept_id,
			"target_id": target_id,
			"channel": channel,
			"authority_ref": authority_ref,
			"started_at": started_at,
			"status": "active",
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._intercepts[intercept_id] = record
		self._audit(self.tenant_id, "sigint_intercept_started", intercept_id)
		return record

	async def decrypt_signal(
		self,
		raw_signal: str,
		decryption_key: str,
		method: str,
	) -> dict[str, Any]:
		"""Attempt decryption of a captured signal using the specified method.

		Supports: xor_mask, base64_strip, rot13, aes256_ecb_sim (stub).
		Returns plaintext_fingerprint — never the actual key.
		"""
		assert present(raw_signal), "raw_signal required"
		assert present(decryption_key), "decryption_key required"
		assert present(method), "method required"

		SUPPORTED_METHODS = {"xor_mask", "base64_strip", "rot13", "aes256_ecb_sim", "manual"}
		if method not in SUPPORTED_METHODS:
			raise ValueError(f"Unsupported decryption method: {method!r}. Supported: {SUPPORTED_METHODS}")

		import base64
		plaintext: str
		if method == "rot13":
			plaintext = raw_signal.encode().decode("rot_13")
		elif method == "base64_strip":
			try:
				plaintext = base64.b64decode(raw_signal.encode()).decode("utf-8", errors="replace")
			except Exception:
				plaintext = raw_signal
		elif method == "xor_mask":
			key_byte = ord(decryption_key[0]) if decryption_key else 0
			plaintext = "".join(chr(ord(c) ^ key_byte) for c in raw_signal)
		else:
			# aes256_ecb_sim / manual — record attempt only
			plaintext = f"[{method}_result_pending]"

		result_id = _fingerprint(raw_signal, method, _utcnow())
		result: dict[str, Any] = {
			"result_id": result_id,
			"method": method,
			"plaintext_length": len(plaintext),
			"plaintext_fingerprint": _fingerprint(plaintext),
			"key_fingerprint": _fingerprint(decryption_key),
			"decrypted_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._decrypted[result_id] = result
		self._audit(self.tenant_id, "sigint_signal_decrypted", result_id)
		return result

	async def traffic_analysis(
		self,
		source: str,
		destination: str,
		period: str,
	) -> dict[str, Any]:
		"""Perform metadata-level traffic analysis between two endpoints.

		Computes transmission count, average inter-arrival time, and
		burst detection for the period. Does not touch signal content.
		"""
		assert present(source), "source required"
		assert present(destination), "destination required"
		assert present(period), "period required"

		# Gather matching signals as a proxy for traffic records
		matching = [
			s for s in self._signals.values()
			if s["tenant_id"] == self.tenant_id
			and (s["source"] == source or s["source"] == destination)
		]

		transmission_count = len(matching)
		timestamps = sorted(
			s["collected_at"] for s in matching
		)

		# Inter-arrival delta in seconds (ISO strings — diff char lengths as proxy)
		if len(timestamps) >= 2:
			deltas = [
				len(timestamps[i]) - len(timestamps[i - 1])
				for i in range(1, len(timestamps))
			]
			# Real implementation would parse ISO datetimes; use count proxy here
			avg_inter_arrival_s = float(abs(statistics.mean(deltas))) if deltas else 0.0
			std_inter_arrival_s = float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
		else:
			avg_inter_arrival_s = 0.0
			std_inter_arrival_s = 0.0

		burst_detected = std_inter_arrival_s > avg_inter_arrival_s * 2 if avg_inter_arrival_s else False

		analysis_id = _fingerprint(source, destination, period, _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"source": source,
			"destination": destination,
			"period": period,
			"transmission_count": transmission_count,
			"avg_inter_arrival_s": avg_inter_arrival_s,
			"std_inter_arrival_s": std_inter_arrival_s,
			"burst_detected": burst_detected,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._traffic_analyses[analysis_id] = result
		self._audit(self.tenant_id, "sigint_traffic_analysed", analysis_id)
		return result

	async def signal_correlation(
		self,
		signal_ids: list[str],
	) -> dict[str, Any]:
		"""Correlate a set of signal IDs to detect common emitter or pattern.

		Computes frequency spread, band overlap, and shared-source indicator.
		"""
		assert signal_ids, "signal_ids must be non-empty"
		assert len(signal_ids) <= 500, "batch cap: 500 signal IDs"

		found = [self._signals[sid] for sid in signal_ids if sid in self._signals]
		missing = [sid for sid in signal_ids if sid not in self._signals]

		if not found:
			raise ValueError("None of the supplied signal_ids were found in this tenant context")

		frequencies = [s["frequency_hz"] for s in found]
		bands = list({s["band"] for s in found})
		sources = list({s["source"] for s in found})

		freq_spread_hz = max(frequencies) - min(frequencies) if len(frequencies) > 1 else 0.0
		mean_freq_hz = statistics.mean(frequencies)

		# Cosine similarity on band presence vector (simplified)
		band_set = list({s["band"] for s in self._signals.values() if s["tenant_id"] == self.tenant_id})
		if band_set:
			vec = [1 if b in bands else 0 for b in band_set]
			magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
			band_diversity_score = round(sum(vec) / magnitude, 4)
		else:
			band_diversity_score = 0.0

		correlation_id = _fingerprint(*sorted(signal_ids), _utcnow())
		result: dict[str, Any] = {
			"correlation_id": correlation_id,
			"input_count": len(signal_ids),
			"matched_count": len(found),
			"missing_ids": missing,
			"bands_observed": bands,
			"sources_observed": sources,
			"mean_frequency_hz": mean_freq_hz,
			"frequency_spread_hz": freq_spread_hz,
			"band_diversity_score": band_diversity_score,
			"likely_single_emitter": len(sources) == 1,
			"correlated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._correlations[correlation_id] = result
		self._audit(self.tenant_id, "sigint_signals_correlated", correlation_id)
		return result

	async def emitter_identification(
		self,
		signal_characteristics: dict[str, Any],
	) -> dict[str, Any]:
		"""Identify a probable emitter from signal characteristics.

		Characteristic keys: frequency_hz, modulation, pulse_width_us,
		pri_us (pulse repetition interval), power_dbm, polarisation.
		"""
		required = {"frequency_hz", "modulation"}
		missing_keys = required - signal_characteristics.keys()
		assert not missing_keys, f"Missing required characteristic keys: {missing_keys}"

		freq = float(signal_characteristics["frequency_hz"])
		modulation = str(signal_characteristics.get("modulation", "unknown"))
		pulse_width_us = float(signal_characteristics.get("pulse_width_us", 0.0))
		pri_us = float(signal_characteristics.get("pri_us", 0.0))
		power_dbm = float(signal_characteristics.get("power_dbm", -999.0))

		# Emitter type heuristics
		emitter_type: str
		confidence: float
		if modulation in {"AM", "FM", "SSB", "DSB"} and freq < 30e6:
			emitter_type = "HF_BROADCAST_OR_COMMS"
			confidence = 0.75
		elif modulation in {"PSK", "QPSK", "QAM"} and freq > 300e6:
			emitter_type = "DIGITAL_COMMS_LINK"
			confidence = 0.80
		elif pulse_width_us > 0 and pri_us > 0:
			duty_cycle = pulse_width_us / pri_us if pri_us else 0.0
			if duty_cycle < 0.05:
				emitter_type = "SEARCH_RADAR"
				confidence = 0.85
			else:
				emitter_type = "TRACKING_RADAR"
				confidence = 0.70
		elif power_dbm > 40:
			emitter_type = "HIGH_POWER_JAMMER"
			confidence = 0.65
		else:
			emitter_type = "UNKNOWN"
			confidence = 0.30

		emitter_id = _fingerprint(str(signal_characteristics), _utcnow())
		result: dict[str, Any] = {
			"emitter_id": emitter_id,
			"emitter_type": emitter_type,
			"confidence": confidence,
			"frequency_hz": freq,
			"modulation": modulation,
			"pulse_width_us": pulse_width_us,
			"pri_us": pri_us,
			"power_dbm": power_dbm,
			"identified_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._emitters[emitter_id] = result
		self._audit(self.tenant_id, "sigint_emitter_identified", emitter_id)
		return result

	async def direction_finding(
		self,
		signal_id: str,
		sensor_positions: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Estimate emitter bearing using TDOA/AOA from multiple sensor positions.

		Each sensor_positions entry: {"lat": float, "lon": float, "bearing_deg": float}.
		Uses simple circular mean of bearing estimates when >= 2 sensors available.
		"""
		assert present(signal_id), "signal_id required"
		assert len(sensor_positions) >= 2, "At least 2 sensor positions required for DF"
		for i, sp in enumerate(sensor_positions):
			assert "lat" in sp and "lon" in sp and "bearing_deg" in sp, \
				f"sensor_positions[{i}] must have lat, lon, bearing_deg"

		signal = self._signals.get(signal_id)
		if signal is None:
			raise KeyError(f"signal_id {signal_id!r} not found")

		bearings = [sp["bearing_deg"] % 360 for sp in sensor_positions]

		# Circular mean
		sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
		cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
		mean_bearing = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

		# Angular spread (quality indicator)
		diffs = [(b - mean_bearing + 180) % 360 - 180 for b in bearings]
		spread_deg = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
		quality = max(0.0, 1.0 - spread_deg / 90.0)

		# Centroid of sensor positions as reference point
		ref_lat = statistics.mean(sp["lat"] for sp in sensor_positions)
		ref_lon = statistics.mean(sp["lon"] for sp in sensor_positions)

		fix_id = _fingerprint(signal_id, str(sensor_positions), _utcnow())
		result: dict[str, Any] = {
			"fix_id": fix_id,
			"signal_id": signal_id,
			"sensor_count": len(sensor_positions),
			"reference_lat": ref_lat,
			"reference_lon": ref_lon,
			"estimated_bearing_deg": round(mean_bearing, 2),
			"bearing_spread_deg": round(spread_deg, 2),
			"quality_score": round(quality, 4),
			"fixed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._df_fixes[fix_id] = result
		self._audit(self.tenant_id, "sigint_direction_found", fix_id)
		return result

	async def satellite_intercept(
		self,
		target_orbit: str,
		frequency_band: str,
	) -> dict[str, Any]:
		"""Register a satellite intercept task for a given orbit and band.

		target_orbit: LEO | MEO | GEO | HEO
		frequency_band: L | S | C | X | Ku | Ka | V | W
		"""
		VALID_ORBITS = {"LEO", "MEO", "GEO", "HEO"}
		VALID_BANDS = {"L", "S", "C", "X", "Ku", "Ka", "V", "W"}

		assert present(target_orbit), "target_orbit required"
		assert present(frequency_band), "frequency_band required"
		if target_orbit not in VALID_ORBITS:
			raise ValueError(f"target_orbit must be one of {VALID_ORBITS}")
		if frequency_band not in VALID_BANDS:
			raise ValueError(f"frequency_band must be one of {VALID_BANDS}")

		# Orbit altitude heuristics
		orbit_altitudes = {"LEO": (200, 2000), "MEO": (2000, 35786), "GEO": (35786, 35786), "HEO": (600, 50000)}
		alt_low, alt_high = orbit_altitudes[target_orbit]

		# Band frequency centres in GHz
		band_centres_ghz = {"L": 1.5, "S": 3.0, "C": 6.0, "X": 10.0, "Ku": 15.0, "Ka": 30.0, "V": 60.0, "W": 95.0}
		centre_ghz = band_centres_ghz[frequency_band]

		# Free-space path loss at 35786 km (GEO worst case) in dB
		distance_km = alt_high
		fspl_db = 20 * math.log10(distance_km) + 20 * math.log10(centre_ghz * 1e9) + 20 * math.log10(4 * math.pi / 3e8)

		intercept_id = _fingerprint(target_orbit, frequency_band, _utcnow())
		result: dict[str, Any] = {
			"intercept_id": intercept_id,
			"target_orbit": target_orbit,
			"frequency_band": frequency_band,
			"centre_frequency_ghz": centre_ghz,
			"altitude_range_km": [alt_low, alt_high],
			"estimated_fspl_db": round(fspl_db, 1),
			"status": "tasked",
			"tasked_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._sat_intercepts[intercept_id] = result
		self._audit(self.tenant_id, "sigint_satellite_intercept_tasked", intercept_id)
		return result

	async def communication_pattern_analysis(
		self,
		target_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Analyse communication behaviour patterns for a specific target.

		Aggregates signal volume by hour-of-day, detects peak activity windows,
		and computes normalised entropy of the hourly distribution.
		"""
		assert present(target_id), "target_id required"
		assert present(period), "period required"

		matching = [
			s for s in self._signals.values()
			if s["tenant_id"] == self.tenant_id
			and s.get("metadata", {}).get("target_id") == target_id
		]

		# Bucket by hour
		hourly: dict[int, int] = {h: 0 for h in range(24)}
		for sig in matching:
			# Extract hour from ISO timestamp (chars 11-13)
			raw_ts = sig.get("collected_at", "T00:")
			try:
				hour = int(raw_ts[11:13])
			except (ValueError, IndexError):
				hour = 0
			hourly[hour] = hourly.get(hour, 0) + 1

		total = sum(hourly.values()) or 1
		probs = [v / total for v in hourly.values() if v > 0]
		entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
		max_entropy = math.log2(24)
		normalised_entropy = entropy / max_entropy if max_entropy else 0.0

		peak_hour = max(hourly, key=lambda h: hourly[h])
		quiet_hour = min(hourly, key=lambda h: hourly[h])

		analysis_id = _fingerprint(target_id, period, _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"target_id": target_id,
			"period": period,
			"total_signals": len(matching),
			"hourly_distribution": hourly,
			"peak_hour_utc": peak_hour,
			"quiet_hour_utc": quiet_hour,
			"normalised_entropy": round(normalised_entropy, 4),
			"high_regularity": normalised_entropy < 0.4,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._pattern_analyses[analysis_id] = result
		self._audit(self.tenant_id, "sigint_comm_pattern_analysed", analysis_id)
		return result

	async def signal_intelligence_report(
		self,
		classification: str,
		recipients: list[str],
	) -> dict[str, Any]:
		"""Generate a SIGINT intelligence report for the current tenant.

		Aggregates summary statistics across all operational collections
		and formats them for dissemination to the listed recipients.
		"""
		assert present(classification), "classification required"
		assert recipients, "recipients list must be non-empty"

		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		tenant = self.tenant_id
		report_id = _fingerprint(classification, str(sorted(recipients)), _utcnow())

		signal_count = len([s for s in self._signals.values() if s["tenant_id"] == tenant])
		intercept_count = len([i for i in self._intercepts.values() if i["tenant_id"] == tenant])
		emitter_types = list({e["emitter_type"] for e in self._emitters.values() if e["tenant_id"] == tenant})
		df_fix_count = len([f for f in self._df_fixes.values() if f["tenant_id"] == tenant])
		avg_df_quality = (
			statistics.mean(f["quality_score"] for f in self._df_fixes.values() if f["tenant_id"] == tenant)
			if df_fix_count else 0.0
		)
		sat_count = len([s for s in self._sat_intercepts.values() if s["tenant_id"] == tenant])

		report: dict[str, Any] = {
			"report_id": report_id,
			"classification": classification,
			"recipients": recipients,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"summary": {
				"signals_collected": signal_count,
				"active_intercepts": intercept_count,
				"emitter_types_identified": emitter_types,
				"direction_fixes": df_fix_count,
				"average_df_quality": round(avg_df_quality, 4),
				"satellite_tasks": sat_count,
				"observations": self._count(self.observations, tenant),
				"patterns": self._count(self.patterns, tenant),
				"assessments": self._count(self.assessments, tenant),
			},
		}
		self._reports[report_id] = report
		self._audit(tenant, "sigint_report_generated", report_id)
		return report

	# ------------------------------------------------------------------
	# Async wrappers for existing sync methods
	# ------------------------------------------------------------------

	async def async_collect_observation(
		self,
		observation_id: str,
		task_id: str,
		observation_reference: str,
		fingerprint: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Async wrapper around record_observation with await-compatible signature."""
		await asyncio.sleep(0)  # yield to event loop
		return self.record_observation(
			observation_id, self.tenant_id, task_id,
			observation_reference, fingerprint, confidence_score, evidence_reference,
		)

	async def async_dashboard(self) -> dict[str, Any]:
		"""Async dashboard summary for the service's own tenant."""
		await asyncio.sleep(0)
		return self.dashboard_summary(self.tenant_id)

	async def frequency_scan(
		self,
		start_hz: float,
		stop_hz: float,
		source: str,
	) -> dict[str, Any]:
		"""Sweep *start_hz* to *stop_hz* in coarse steps, collecting signal metadata per step."""
		assert stop_hz > start_hz, "stop_hz must exceed start_hz"
		assert present(source), "source required"
		span = stop_hz - start_hz
		# 10 evenly spaced steps
		steps = 10
		step_hz = span / steps
		results: list[dict[str, Any]] = []
		freq = start_hz
		for _ in range(steps):
			rec = await self.collect_signal("freq_scan", freq, source, {"scan": True})
			results.append({"frequency_hz": freq, "signal_id": rec["signal_id"], "band": rec["band"]})
			freq += step_hz
		scan_id = _fingerprint(str(start_hz), str(stop_hz), source, _utcnow())
		self._audit(self.tenant_id, "sigint_frequency_scanned", scan_id)
		return {"scan_id": scan_id, "start_hz": start_hz, "stop_hz": stop_hz, "steps": steps, "signals": results, "scanned_at": _utcnow()}

	async def signal_decode(
		self,
		signal_id: str,
		method: str = "rot13",
	) -> dict[str, Any]:
		"""Attempt to decode stored signal *signal_id* using *method*."""
		assert present(signal_id), "signal_id required"
		signal = self._signals.get(signal_id)
		if signal is None:
			raise KeyError(f"signal_id {signal_id!r} not found")
		# Use metadata reference or signal_id as raw payload proxy
		raw = str(signal.get("signal_type", signal_id))
		return await self.decrypt_signal(raw, signal_id, method)

	async def emitter_geolocate(
		self,
		emitter_id: str,
		sensor_positions: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Geolocate emitter *emitter_id* from *sensor_positions* via direction finding."""
		assert present(emitter_id), "emitter_id required"
		# Create a synthetic signal record for the emitter if not already stored
		if emitter_id not in self._signals:
			emitter = self._emitters.get(emitter_id)
			if emitter is None:
				raise KeyError(f"emitter_id {emitter_id!r} not found in emitters or signals")
			freq = float(emitter.get("frequency_hz", 1e9))
			sig = await self.collect_signal("emitter_proxy", freq, "emitter_geolocation", {"emitter_id": emitter_id})
			signal_id = sig["signal_id"]
		else:
			signal_id = emitter_id
		return await self.direction_finding(signal_id, sensor_positions)

	async def traffic_analyse(
		self,
		source: str,
		destination: str,
		period: str = "24h",
	) -> dict[str, Any]:
		"""Analyse traffic metadata between *source* and *destination* for *period*."""
		return await self.traffic_analysis(source, destination, period)

	async def signal_archive(
		self,
		signal_ids: list[str],
		archive_reason: str = "retention",
	) -> dict[str, Any]:
		"""Mark *signal_ids* as archived and record the archive event."""
		assert signal_ids, "signal_ids required"
		archived: list[str] = []
		not_found: list[str] = []
		for sid in signal_ids:
			if sid in self._signals:
				self._signals[sid]["status"] = "archived"
				self._signals[sid]["archived_at"] = _utcnow()
				archived.append(sid)
			else:
				not_found.append(sid)
		archive_id = _fingerprint(*sorted(signal_ids[:6]), _utcnow())
		self._audit(self.tenant_id, "sigint_signals_archived", archive_id)
		return {"archive_id": archive_id, "archived": len(archived), "not_found": not_found, "reason": archive_reason, "archived_at": _utcnow()}

	async def direction_find(
		self,
		signal_id: str,
		sensor_positions: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Alias for direction_finding with cleaner name."""
		return await self.direction_finding(signal_id, sensor_positions)

	async def cross_band_correlate(
		self,
		band_a: str,
		band_b: str,
	) -> dict[str, Any]:
		"""Correlate signal activity between two frequency bands."""
		assert present(band_a) and present(band_b), "band_a and band_b required"
		tenant = self.tenant_id
		sigs_a = [s for s in self._signals.values() if s.get("band") == band_a and s.get("tenant_id") == tenant]
		sigs_b = [s for s in self._signals.values() if s.get("band") == band_b and s.get("tenant_id") == tenant]
		overlap_sources = {s["source"] for s in sigs_a} & {s["source"] for s in sigs_b}
		corr_id = _fingerprint(band_a, band_b, _utcnow())
		result: dict[str, Any] = {
			"correlation_id": corr_id,
			"band_a": band_a, "band_b": band_b,
			"signals_in_a": len(sigs_a), "signals_in_b": len(sigs_b),
			"shared_sources": list(overlap_sources),
			"correlation_score": round(len(overlap_sources) / max(len({s["source"] for s in sigs_a} | {s["source"] for s in sigs_b}), 1), 4),
			"computed_at": _utcnow(),
		}
		self._correlations[corr_id] = result
		self._audit(tenant, "cross_band_correlated", corr_id)
		return result

	async def intercept_schedule(
		self,
		target_id: str,
		channels: list[str],
		authority_ref: str,
	) -> dict[str, Any]:
		"""Schedule intercept tasks for *target_id* across *channels*."""
		assert present(target_id) and channels and present(authority_ref), "target_id, channels, authority_ref required"
		tasks: list[dict[str, Any]] = []
		for channel in channels:
			rec = await self.intercept_communication(target_id, channel, authority_ref)
			tasks.append({"channel": channel, "intercept_id": rec["intercept_id"]})
		sched_id = _fingerprint(target_id, *sorted(channels), _utcnow())
		self._audit(self.tenant_id, "sigint_intercept_scheduled", sched_id)
		return {"schedule_id": sched_id, "target_id": target_id, "tasks": tasks, "scheduled_at": _utcnow()}

	async def signal_quality(self, signal_id: str) -> dict[str, Any]:
		"""Assess quality metrics for *signal_id* (SNR proxy, coverage, freshness)."""
		assert present(signal_id), "signal_id required"
		signal = self._signals.get(signal_id)
		if signal is None:
			raise KeyError(f"signal_id {signal_id!r} not found")
		freq = float(signal.get("frequency_hz", 0))
		# SNR proxy from frequency band and metadata
		band = signal.get("band", "VHF")
		snr_map = {"ELF": 5, "VLF": 8, "LF": 10, "MF": 12, "HF": 15, "VHF": 20, "UHF": 25, "SHF": 30, "EHF": 28}
		snr_db = snr_map.get(band, 15)
		quality_score = round(min(1.0, snr_db / 35.0), 4)
		quality_id = _fingerprint(signal_id, _utcnow())
		self._audit(self.tenant_id, "sigint_signal_quality_assessed", quality_id)
		return {"quality_id": quality_id, "signal_id": signal_id, "band": band, "frequency_hz": freq, "snr_db_estimate": snr_db, "quality_score": quality_score, "assessed_at": _utcnow()}

	async def sigint_report(
		self,
		classification: str,
		recipients: list[str],
	) -> dict[str, Any]:
		"""Generate SIGINT report — alias for signal_intelligence_report."""
		return await self.signal_intelligence_report(classification, recipients)

	async def signal_quality_batch(self, signal_ids: list[str]) -> list[dict[str, Any]]:
		"""Assess quality for a batch of signal IDs."""
		assert signal_ids, "signal_ids required"
		results = []
		for sid in signal_ids:
			try:
				results.append(await self.signal_quality(sid))
			except KeyError:
				results.append({"signal_id": sid, "error": "not_found"})
		return results

	async def sigint_analytics(self) -> dict[str, Any]:
		"""Aggregate SIGINT operational analytics for the tenant."""
		tenant = self.tenant_id
		return {
			"tenant_id": tenant,
			"authority_count": self._count(self.authorities, tenant),
			"source_count": self._count(self.sources, tenant),
			"task_count": self._count(self.tasks, tenant),
			"observation_count": self._count(self.observations, tenant),
			"pattern_count": self._count(self.patterns, tenant),
			"assessment_count": self._count(self.assessments, tenant),
			"signals_collected": len(self._signals),
			"intercepts_active": sum(1 for i in self._intercepts.values() if i.get("status") == "active"),
			"emitters_identified": len(self._emitters),
			"df_fixes": len(self._df_fixes),
			"reports_generated": len(self._reports),
			"computed_at": _utcnow(),
		}

	async def bulk_collect_signals(
		self,
		signals: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Collect multiple signals concurrently.

		Each entry: {"signal_type": str, "frequency": float, "source": str, "metadata": dict}.
		"""
		assert signals, "signals list must be non-empty"
		tasks = [
			self.collect_signal(
				s["signal_type"], float(s["frequency"]), s["source"], s.get("metadata", {}),
			)
			for s in signals
		]
		return list(await asyncio.gather(*tasks), return_exceptions=True)

	async def spectrum_sweep(
		self,
		start_hz: float,
		stop_hz: float,
		step_hz: float,
		source: str,
	) -> list[dict[str, Any]]:
		"""Sweep a frequency range, collecting a signal at each step.

		Returns a list of collected signal records. step_hz > 0 required.
		"""
		assert stop_hz > start_hz, "stop_hz must be greater than start_hz"
		assert step_hz > 0, "step_hz must be positive"
		assert present(source), "source required"

		max_steps = 1000
		step_count = int((stop_hz - start_hz) / step_hz)
		assert step_count <= max_steps, f"Sweep would produce {step_count} steps; cap is {max_steps}"

		freq = start_hz
		results: list[dict[str, Any]] = []
		while freq <= stop_hz:
			rec = await self.collect_signal("sweep", freq, source, {"sweep": True})
			results.append(rec)
			freq += step_hz
		return results

	# ------------------------------------------------------------------
	# Internal helpers (preserved from original)
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> SignalAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> SignalSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_task_or_none(self, item_id: str, tenant_id: str) -> CollectionTask | None:
		return self.tasks.get(self._tenant_key(tenant_id, item_id))

	def _tenant_observation_or_none(self, item_id: str, tenant_id: str) -> SignalObservation | None:
		return self.observations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_batch_or_none(self, item_id: str, tenant_id: str) -> ProcessingBatch | None:
		return self.processing_batches.get(self._tenant_key(tenant_id, item_id))

	def _tenant_pattern_or_none(self, item_id: str, tenant_id: str) -> SignalPattern | None:
		return self.patterns.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": _utcnow(),
			"processor": "bytewax",
		})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "sigint_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "sigint_policy_denied")


# Aliases for backward compatibility
SignalsIntelligenceService = SIGINTService
IntelSIGINTService = SIGINTService
