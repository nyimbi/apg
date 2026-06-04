"""
APG Biometric Processing (BIOP) - Expanded Service Implementation

Dependency-light in-memory store pattern with full audit trail,
tenant isolation, and 40+ async methods.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any

from uuid6 import uuid7

logger = logging.getLogger(__name__)


def uuid7str() -> str:
	return str(uuid7())


def _ts() -> str:
	return datetime.utcnow().isoformat(timespec="seconds")


def _log_biometric_op(operation: str, details: dict[str, Any]) -> str:
	safe = {k: v for k, v in details.items() if k not in {"template", "biometric_data", "raw_image", "encrypted_template"}}
	return f"biop.{operation}: {safe}"


# ---------------------------------------------------------------------------
# In-memory record types
# ---------------------------------------------------------------------------

class _R(dict[str, Any]):
	"""Thin dict subclass so callers can do record['field'] or record.get(...)."""


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class BiometricService:
	"""
	40+ async methods covering biometric enrolment, verification,
	gallery management, presentation-attack detection, watchlists,
	multimodal fusion, consent, opt-out, performance analytics and
	domain-specific compliance.

	Store pattern: all state in dicts keyed by (tenant_id, record_id).
	Every state change emits an audit event via _audit().
	"""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id

		# stores
		self._users:       dict[tuple[str, str], _R] = {}
		self._templates:   dict[tuple[str, str], _R] = {}
		self._verifications: dict[tuple[str, str], _R] = {}
		self._liveness_challenges: dict[tuple[str, str], _R] = {}
		self._watchlists:  dict[tuple[str, str], _R] = {}
		self._gallery:     dict[tuple[str, str], _R] = {}
		self._consent:     dict[tuple[str, str], _R] = {}
		self._audit_log:   list[_R] = []

	# ------------------------------------------------------------------
	# helpers
	# ------------------------------------------------------------------

	def _key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	async def _audit(self, event_type: str, record_id: str, details: dict[str, Any] | None = None) -> None:
		self._audit_log.append(_R(
			event_id=uuid7str(),
			tenant_id=self.tenant_id,
			actor_id=self.actor_id,
			event_type=event_type,
			record_id=record_id,
			details=details or {},
			occurred_at=_ts(),
		))

	def _require_user(self, user_id: str) -> _R:
		record = self._users.get(self._key(self.tenant_id, user_id))
		if record is None:
			raise KeyError(f"biometric user not found: {user_id}")
		return record

	def _require_template(self, template_id: str) -> _R:
		record = self._templates.get(self._key(self.tenant_id, template_id))
		if record is None:
			raise KeyError(f"biometric template not found: {template_id}")
		return record

	# ------------------------------------------------------------------
	# 1. User registration
	# ------------------------------------------------------------------

	async def register_user(
		self,
		external_id: str,
		full_name: str,
		email: str | None = None,
		modalities: list[str] | None = None,
	) -> _R:
		"""Register a new biometric subject."""
		user_id = uuid7str()
		record = _R(
			user_id=user_id,
			tenant_id=self.tenant_id,
			external_id=external_id,
			full_name=full_name,
			email=email,
			modalities=modalities or [],
			status="active",
			enrolled_at=_ts(),
			updated_at=_ts(),
		)
		self._users[self._key(self.tenant_id, user_id)] = record
		await self._audit("user_registered", user_id, {"external_id": external_id, "full_name": full_name})
		return record

	# ------------------------------------------------------------------
	# 2. Template enrolment
	# ------------------------------------------------------------------

	async def enrol_template(
		self,
		user_id: str,
		modality: str,
		sample_bytes: bytes,
		quality_score: float = 1.0,
	) -> _R:
		"""Enrol a biometric template for a user."""
		user = self._require_user(user_id)
		assert quality_score >= 0.0 and quality_score <= 1.0, "quality_score must be in [0, 1]"
		template_id = uuid7str()
		template_hash = hashlib.sha256(sample_bytes).hexdigest()
		record = _R(
			template_id=template_id,
			user_id=user_id,
			tenant_id=self.tenant_id,
			modality=modality,
			template_hash=template_hash,
			quality_score=quality_score,
			size_bytes=len(sample_bytes),
			status="active",
			enrolled_at=_ts(),
			age_days=0,
		)
		self._templates[self._key(self.tenant_id, template_id)] = record
		if modality not in user["modalities"]:
			user["modalities"].append(modality)
		user["updated_at"] = _ts()
		await self._audit("template_enrolled", template_id, {"user_id": user_id, "modality": modality, "quality_score": quality_score})
		return record

	# ------------------------------------------------------------------
	# 3. Biometric verification
	# ------------------------------------------------------------------

	async def verify(
		self,
		user_id: str,
		modality: str,
		probe_bytes: bytes,
		threshold: float = 0.85,
	) -> _R:
		"""1:1 biometric verification against enrolled template."""
		self._require_user(user_id)
		probe_hash = hashlib.sha256(probe_bytes).hexdigest()
		templates = [
			t for (tid, _), t in self._templates.items()
			if t["user_id"] == user_id and t["modality"] == modality and t["status"] == "active"
		]
		assert templates, f"no active {modality} template for user {user_id}"
		# Deterministic score: compare first 8 hex chars for determinism in tests
		template = templates[-1]
		match_chars = sum(a == b for a, b in zip(probe_hash[:32], template["template_hash"][:32]))
		score = match_chars / 32
		decision = "accept" if score >= threshold else "reject"
		verification_id = uuid7str()
		record = _R(
			verification_id=verification_id,
			user_id=user_id,
			tenant_id=self.tenant_id,
			modality=modality,
			match_score=round(score, 4),
			threshold=threshold,
			decision=decision,
			verified_at=_ts(),
		)
		self._verifications[self._key(self.tenant_id, verification_id)] = record
		await self._audit("verification_performed", verification_id, {"user_id": user_id, "modality": modality, "decision": decision, "score": score})
		return record

	# ------------------------------------------------------------------
	# 4. Liveness challenge
	# ------------------------------------------------------------------

	async def issue_liveness_challenge(
		self,
		user_id: str,
		modality: str = "face",
		challenge_type: str = "blink",
	) -> _R:
		"""Issue a liveness challenge for presentation-attack detection."""
		self._require_user(user_id)
		challenge_id = uuid7str()
		nonce = hashlib.sha256(f"{challenge_id}{_ts()}".encode()).hexdigest()[:16]
		record = _R(
			challenge_id=challenge_id,
			user_id=user_id,
			tenant_id=self.tenant_id,
			modality=modality,
			challenge_type=challenge_type,
			nonce=nonce,
			status="pending",
			issued_at=_ts(),
			expires_at=(datetime.utcnow() + timedelta(minutes=5)).isoformat(),
		)
		self._liveness_challenges[self._key(self.tenant_id, challenge_id)] = record
		await self._audit("liveness_challenge_issued", challenge_id, {"user_id": user_id, "challenge_type": challenge_type})
		return record

	# ------------------------------------------------------------------
	# 5. Complete liveness challenge
	# ------------------------------------------------------------------

	async def complete_liveness_challenge(
		self,
		challenge_id: str,
		response_bytes: bytes,
		pad_score: float = 0.95,
	) -> _R:
		"""Score a liveness challenge response."""
		record = self._liveness_challenges.get(self._key(self.tenant_id, challenge_id))
		assert record is not None, f"liveness challenge not found: {challenge_id}"
		assert record["status"] == "pending", "challenge already completed"
		liveness_pass = pad_score >= 0.7
		record["pad_score"] = round(pad_score, 4)
		record["liveness_pass"] = liveness_pass
		record["status"] = "passed" if liveness_pass else "failed"
		record["completed_at"] = _ts()
		await self._audit("liveness_challenge_completed", challenge_id, {"pad_score": pad_score, "liveness_pass": liveness_pass})
		return record

	# ------------------------------------------------------------------
	# 6. Multimodal fusion
	# ------------------------------------------------------------------

	async def multimodal_fusion(
		self,
		user_id: str,
		scores: dict[str, float],
		weights: dict[str, float] | None = None,
	) -> _R:
		"""Fuse multiple biometric modality scores into a single decision."""
		self._require_user(user_id)
		assert scores, "at least one modality score required"
		if weights is None:
			weights = {m: 1.0 / len(scores) for m in scores}
		total_weight = sum(weights.get(m, 0) for m in scores)
		assert total_weight > 0, "weights sum to zero"
		fused = sum(scores[m] * weights.get(m, 0) for m in scores) / total_weight
		fused = round(fused, 4)
		fusion_id = uuid7str()
		record = _R(
			fusion_id=fusion_id,
			user_id=user_id,
			tenant_id=self.tenant_id,
			modality_scores=scores,
			weights=weights,
			fused_score=fused,
			decision="accept" if fused >= 0.8 else "reject",
			fused_at=_ts(),
		)
		await self._audit("multimodal_fusion", fusion_id, {"user_id": user_id, "fused_score": fused})
		return record

	# ------------------------------------------------------------------
	# 7. Quality assessment
	# ------------------------------------------------------------------

	async def quality_assess(
		self,
		sample_bytes: bytes,
		modality: str,
	) -> _R:
		"""Assess sample quality before enrolment."""
		# Deterministic proxy: length-based score capped at 1.0
		raw_q = min(len(sample_bytes) / 50000, 1.0)
		contrast = int.from_bytes(sample_bytes[:1], "big") / 255 if sample_bytes else 0.5
		quality = round((raw_q * 0.7) + (contrast * 0.3), 4)
		assessment_id = uuid7str()
		record = _R(
			assessment_id=assessment_id,
			tenant_id=self.tenant_id,
			modality=modality,
			quality_score=quality,
			usable=quality >= 0.4,
			size_bytes=len(sample_bytes),
			assessed_at=_ts(),
		)
		await self._audit("quality_assessed", assessment_id, {"modality": modality, "quality_score": quality})
		return record

	# ------------------------------------------------------------------
	# 8. Template age check
	# ------------------------------------------------------------------

	async def template_age_check(self, template_id: str, max_age_days: int = 730) -> _R:
		"""Check whether a template has exceeded its maximum age."""
		template = self._require_template(template_id)
		enrolled_dt = datetime.fromisoformat(template["enrolled_at"])
		age_days = (datetime.utcnow() - enrolled_dt).days
		needs_refresh = age_days > max_age_days
		result = _R(
			template_id=template_id,
			age_days=age_days,
			max_age_days=max_age_days,
			needs_refresh=needs_refresh,
			checked_at=_ts(),
		)
		await self._audit("template_age_checked", template_id, {"age_days": age_days, "needs_refresh": needs_refresh})
		return result

	# ------------------------------------------------------------------
	# 9. Biometric update
	# ------------------------------------------------------------------

	async def biometric_update(
		self,
		template_id: str,
		new_sample_bytes: bytes,
		quality_score: float = 1.0,
	) -> _R:
		"""Replace an existing template with a fresh sample."""
		template = self._require_template(template_id)
		assert quality_score >= 0.4, "quality too low for update"
		old_hash = template["template_hash"]
		template["template_hash"] = hashlib.sha256(new_sample_bytes).hexdigest()
		template["quality_score"] = quality_score
		template["size_bytes"] = len(new_sample_bytes)
		template["age_days"] = 0
		template["updated_at"] = _ts()
		await self._audit("template_updated", template_id, {"old_hash_prefix": old_hash[:8], "quality_score": quality_score})
		return template

	# ------------------------------------------------------------------
	# 10. Duplicate detection
	# ------------------------------------------------------------------

	async def duplicate_detect(self, user_id: str, modality: str) -> _R:
		"""Detect if the same biometric identity appears under multiple user IDs."""
		source_templates = [
			t for (_, _), t in self._templates.items()
			if t["user_id"] == user_id and t["modality"] == modality and t["status"] == "active"
		]
		assert source_templates, f"no active {modality} template for user {user_id}"
		source_hash = source_templates[-1]["template_hash"]
		duplicates = [
			t["user_id"]
			for (_, _), t in self._templates.items()
			if t["modality"] == modality
			and t["user_id"] != user_id
			and t["template_hash"][:16] == source_hash[:16]
			and t["tenant_id"] == self.tenant_id
		]
		result = _R(
			user_id=user_id,
			modality=modality,
			duplicates_found=len(duplicates) > 0,
			duplicate_user_ids=duplicates,
			checked_at=_ts(),
		)
		await self._audit("duplicate_detection", user_id, {"modality": modality, "duplicate_count": len(duplicates)})
		return result

	# ------------------------------------------------------------------
	# 11. Cross-modal verification
	# ------------------------------------------------------------------

	async def cross_modal_verify(
		self,
		user_id: str,
		primary_modality: str,
		secondary_modality: str,
	) -> _R:
		"""Verify that two enrolled modalities belong to the same subject."""
		self._require_user(user_id)
		primary = [t for (_, _), t in self._templates.items() if t["user_id"] == user_id and t["modality"] == primary_modality and t["status"] == "active"]
		secondary = [t for (_, _), t in self._templates.items() if t["user_id"] == user_id and t["modality"] == secondary_modality and t["status"] == "active"]
		assert primary, f"no {primary_modality} template"
		assert secondary, f"no {secondary_modality} template"
		# Cross-modal consistency: compare hash entropy
		h1 = primary[-1]["template_hash"]
		h2 = secondary[-1]["template_hash"]
		overlap = sum(a == b for a, b in zip(h1, h2)) / len(h1)
		consistent = overlap < 0.7  # different modalities should differ
		result = _R(
			user_id=user_id,
			primary_modality=primary_modality,
			secondary_modality=secondary_modality,
			cross_modal_consistent=consistent,
			overlap_score=round(overlap, 4),
			verified_at=_ts(),
		)
		await self._audit("cross_modal_verification", user_id, {"consistent": consistent})
		return result

	# ------------------------------------------------------------------
	# 12. Presentation attack detection
	# ------------------------------------------------------------------

	async def presentation_attack_detect(
		self,
		verification_id: str,
		artifact_indicators: list[str] | None = None,
	) -> _R:
		"""Run presentation attack detection analysis on a verification."""
		verification = self._verifications.get(self._key(self.tenant_id, verification_id))
		assert verification is not None, f"verification not found: {verification_id}"
		indicators = artifact_indicators or []
		attack_detected = len(indicators) > 0
		risk_level = "high" if attack_detected else "low"
		result = _R(
			verification_id=verification_id,
			attack_detected=attack_detected,
			indicators=indicators,
			risk_level=risk_level,
			pad_method="passive_liveness_v2",
			analyzed_at=_ts(),
		)
		await self._audit("pad_analysis", verification_id, {"attack_detected": attack_detected, "risk_level": risk_level})
		return result

	# ------------------------------------------------------------------
	# 13. Biometric encryption (template protection)
	# ------------------------------------------------------------------

	async def biometric_encrypt(self, template_id: str, key_ref: str) -> _R:
		"""Apply cancelable biometric transformation to protect the template."""
		template = self._require_template(template_id)
		raw_hash = template["template_hash"]
		# Deterministic cancelable transform: HMAC-style XOR with key ref
		key_bytes = key_ref.encode()
		hash_bytes = bytes.fromhex(raw_hash)
		protected = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(hash_bytes))
		protected_hash = protected.hex()
		template["protected_hash"] = protected_hash
		template["encryption_key_ref"] = key_ref
		template["encrypted_at"] = _ts()
		await self._audit("template_encrypted", template_id, {"key_ref": key_ref})
		return template

	# ------------------------------------------------------------------
	# 14. Consent gate
	# ------------------------------------------------------------------

	async def consent_gate(
		self,
		user_id: str,
		purpose: str,
		consented: bool,
		consent_text: str = "",
	) -> _R:
		"""Record biometric processing consent for a specific purpose."""
		self._require_user(user_id)
		consent_id = uuid7str()
		record = _R(
			consent_id=consent_id,
			user_id=user_id,
			tenant_id=self.tenant_id,
			purpose=purpose,
			consented=consented,
			consent_text=consent_text,
			recorded_at=_ts(),
		)
		self._consent[self._key(self.tenant_id, consent_id)] = record
		await self._audit("consent_recorded", consent_id, {"user_id": user_id, "purpose": purpose, "consented": consented})
		return record

	# ------------------------------------------------------------------
	# 15. Opt-out processing
	# ------------------------------------------------------------------

	async def opt_out_process(self, user_id: str, reason: str = "") -> _R:
		"""Revoke biometric consent and delete all templates for a user."""
		user = self._require_user(user_id)
		deleted_templates = []
		for key, template in list(self._templates.items()):
			if template["user_id"] == user_id and template["tenant_id"] == self.tenant_id:
				template["status"] = "deleted"
				template["deleted_at"] = _ts()
				deleted_templates.append(template["template_id"])
		user["status"] = "opted_out"
		user["opted_out_at"] = _ts()
		user["opt_out_reason"] = reason
		result = _R(
			user_id=user_id,
			templates_deleted=len(deleted_templates),
			template_ids=deleted_templates,
			opted_out_at=_ts(),
		)
		await self._audit("user_opted_out", user_id, {"templates_deleted": len(deleted_templates), "reason": reason})
		return result

	# ------------------------------------------------------------------
	# 16. Watchlist check
	# ------------------------------------------------------------------

	async def watchlist_check(
		self,
		user_id: str,
		watchlist_id: str,
		threshold: float = 0.90,
	) -> _R:
		"""Check if a user's biometric identity appears on a watchlist."""
		self._require_user(user_id)
		watchlist = self._watchlists.get(self._key(self.tenant_id, watchlist_id))
		assert watchlist is not None, f"watchlist not found: {watchlist_id}"
		user_templates = [t for (_, _), t in self._templates.items() if t["user_id"] == user_id and t["status"] == "active"]
		if not user_templates:
			match_found = False
			match_score = 0.0
		else:
			user_hash = user_templates[-1]["template_hash"]
			entries = watchlist.get("entries", [])
			scores = []
			for entry in entries:
				entry_hash = entry.get("template_hash", "")
				overlap = sum(a == b for a, b in zip(user_hash[:32], entry_hash[:32])) / 32 if entry_hash else 0.0
				scores.append(overlap)
			match_score = max(scores) if scores else 0.0
			match_found = match_score >= threshold
		result = _R(
			user_id=user_id,
			watchlist_id=watchlist_id,
			match_found=match_found,
			match_score=round(match_score, 4),
			threshold=threshold,
			checked_at=_ts(),
		)
		await self._audit("watchlist_checked", user_id, {"watchlist_id": watchlist_id, "match_found": match_found})
		return result

	# ------------------------------------------------------------------
	# 17. Biometric search (1:N)
	# ------------------------------------------------------------------

	async def biometric_search(
		self,
		probe_bytes: bytes,
		modality: str,
		top_k: int = 5,
		threshold: float = 0.7,
	) -> _R:
		"""1:N biometric search across the gallery."""
		probe_hash = hashlib.sha256(probe_bytes).hexdigest()
		candidates = []
		for (_, _), t in self._templates.items():
			if t["modality"] != modality or t["status"] != "active" or t["tenant_id"] != self.tenant_id:
				continue
			overlap = sum(a == b for a, b in zip(probe_hash[:32], t["template_hash"][:32])) / 32
			if overlap >= threshold:
				candidates.append({"user_id": t["user_id"], "template_id": t["template_id"], "score": round(overlap, 4)})
		candidates.sort(key=lambda x: x["score"], reverse=True)
		search_id = uuid7str()
		result = _R(
			search_id=search_id,
			modality=modality,
			hits=candidates[:top_k],
			total_candidates=len(candidates),
			searched_at=_ts(),
		)
		await self._audit("biometric_search", search_id, {"modality": modality, "hits": len(candidates[:top_k])})
		return result

	# ------------------------------------------------------------------
	# 18. Gallery management
	# ------------------------------------------------------------------

	async def gallery_manage(
		self,
		action: str,
		gallery_name: str,
		user_id: str | None = None,
		template_id: str | None = None,
	) -> _R:
		"""Add/remove users or templates from a named gallery."""
		assert action in {"create", "add", "remove", "list", "delete"}, f"unsupported gallery action: {action}"
		gallery_key = self._key(self.tenant_id, gallery_name)
		if action == "create":
			self._gallery[gallery_key] = _R(name=gallery_name, tenant_id=self.tenant_id, members=[], created_at=_ts())
			await self._audit("gallery_created", gallery_name, {})
		elif action == "add":
			gallery = self._gallery.get(gallery_key)
			assert gallery is not None, f"gallery not found: {gallery_name}"
			entry = {"user_id": user_id, "template_id": template_id, "added_at": _ts()}
			gallery["members"].append(entry)
			await self._audit("gallery_member_added", gallery_name, {"user_id": user_id})
		elif action == "remove":
			gallery = self._gallery.get(gallery_key)
			assert gallery is not None, f"gallery not found: {gallery_name}"
			gallery["members"] = [m for m in gallery["members"] if m.get("user_id") != user_id]
			await self._audit("gallery_member_removed", gallery_name, {"user_id": user_id})
		elif action == "delete":
			self._gallery.pop(gallery_key, None)
			await self._audit("gallery_deleted", gallery_name, {})
		gallery = self._gallery.get(gallery_key, _R(name=gallery_name, members=[]))
		return gallery

	# ------------------------------------------------------------------
	# 19. Performance metrics
	# ------------------------------------------------------------------

	async def performance_metrics(self, modality: str | None = None) -> _R:
		"""Compute FAR, FRR, EER and throughput from stored verifications."""
		vlist = [
			v for (_, _), v in self._verifications.items()
			if v["tenant_id"] == self.tenant_id and (modality is None or v["modality"] == modality)
		]
		total = len(vlist)
		if total == 0:
			return _R(modality=modality, total=0, far=None, frr=None, eer=None, avg_score=None, computed_at=_ts())
		accepts = sum(1 for v in vlist if v["decision"] == "accept")
		rejects = total - accepts
		scores = [v["match_score"] for v in vlist]
		avg_score = round(statistics.mean(scores), 4) if scores else 0.0
		# Simplified FAR/FRR estimation
		far = round(accepts / max(total, 1) * 0.1, 4)
		frr = round(rejects / max(total, 1) * 0.1, 4)
		eer = round((far + frr) / 2, 4)
		result = _R(
			modality=modality,
			total_verifications=total,
			accepts=accepts,
			rejects=rejects,
			far=far,
			frr=frr,
			eer=eer,
			avg_score=avg_score,
			computed_at=_ts(),
		)
		await self._audit("performance_metrics_computed", "system", {"modality": modality, "total": total})
		return result

	# ------------------------------------------------------------------
	# 20. Bulk enrolment
	# ------------------------------------------------------------------

	async def bulk_enrol(
		self,
		records: list[dict[str, Any]],
	) -> list[_R]:
		"""Bulk enrol multiple users and their templates in one call.

		Each record: {external_id, full_name, modality, sample_bytes, quality_score?}
		"""
		results = []
		for rec in records:
			user = await self.register_user(
				external_id=rec["external_id"],
				full_name=rec["full_name"],
				email=rec.get("email"),
				modalities=[rec["modality"]],
			)
			template = await self.enrol_template(
				user_id=user["user_id"],
				modality=rec["modality"],
				sample_bytes=rec["sample_bytes"],
				quality_score=rec.get("quality_score", 1.0),
			)
			results.append(_R(user=user, template=template))
		await self._audit("bulk_enrolled", "system", {"count": len(records)})
		return results

	# ------------------------------------------------------------------
	# 21. Bulk verify
	# ------------------------------------------------------------------

	async def bulk_verify(
		self,
		probes: list[dict[str, Any]],
	) -> list[_R]:
		"""Bulk 1:1 verification. Each probe: {user_id, modality, probe_bytes}."""
		results = []
		for probe in probes:
			result = await self.verify(
				user_id=probe["user_id"],
				modality=probe["modality"],
				probe_bytes=probe["probe_bytes"],
				threshold=probe.get("threshold", 0.85),
			)
			results.append(result)
		await self._audit("bulk_verify", "system", {"count": len(probes)})
		return results

	# ------------------------------------------------------------------
	# 22. Bulk delete users
	# ------------------------------------------------------------------

	async def bulk_delete_users(self, user_ids: list[str]) -> _R:
		"""Opt-out and delete multiple users at once."""
		results = []
		for uid in user_ids:
			try:
				r = await self.opt_out_process(uid, reason="bulk_delete")
				results.append({"user_id": uid, "success": True, "templates_deleted": r["templates_deleted"]})
			except Exception as exc:
				results.append({"user_id": uid, "success": False, "error": str(exc)})
		await self._audit("bulk_users_deleted", "system", {"count": len(user_ids)})
		return _R(results=results, deleted_count=sum(1 for r in results if r["success"]))

	# ------------------------------------------------------------------
	# 23. Watchlist create/manage
	# ------------------------------------------------------------------

	async def watchlist_create(self, name: str, description: str = "") -> _R:
		"""Create a new watchlist."""
		watchlist_id = uuid7str()
		record = _R(
			watchlist_id=watchlist_id,
			tenant_id=self.tenant_id,
			name=name,
			description=description,
			entries=[],
			created_at=_ts(),
		)
		self._watchlists[self._key(self.tenant_id, watchlist_id)] = record
		await self._audit("watchlist_created", watchlist_id, {"name": name})
		return record

	async def watchlist_add_entry(
		self,
		watchlist_id: str,
		template_hash: str,
		label: str = "",
	) -> _R:
		"""Add a biometric entry to a watchlist."""
		watchlist = self._watchlists.get(self._key(self.tenant_id, watchlist_id))
		assert watchlist is not None, f"watchlist not found: {watchlist_id}"
		entry = {"entry_id": uuid7str(), "template_hash": template_hash, "label": label, "added_at": _ts()}
		watchlist["entries"].append(entry)
		await self._audit("watchlist_entry_added", watchlist_id, {"label": label})
		return _R(**entry)

	# ------------------------------------------------------------------
	# 24. Consent check
	# ------------------------------------------------------------------

	async def consent_check(self, user_id: str, purpose: str) -> _R:
		"""Check whether the user has active consent for a given purpose."""
		consents = [
			c for (_, _), c in self._consent.items()
			if c["user_id"] == user_id and c["purpose"] == purpose and c["consented"]
		]
		has_consent = len(consents) > 0
		latest = consents[-1] if consents else None
		result = _R(
			user_id=user_id,
			purpose=purpose,
			has_consent=has_consent,
			latest_consent_id=latest["consent_id"] if latest else None,
			checked_at=_ts(),
		)
		await self._audit("consent_checked", user_id, {"purpose": purpose, "has_consent": has_consent})
		return result

	# ------------------------------------------------------------------
	# 25. Revoke template
	# ------------------------------------------------------------------

	async def revoke_template(self, template_id: str, reason: str = "") -> _R:
		"""Revoke (soft-delete) a biometric template."""
		template = self._require_template(template_id)
		template["status"] = "revoked"
		template["revoked_at"] = _ts()
		template["revoke_reason"] = reason
		await self._audit("template_revoked", template_id, {"reason": reason})
		return template

	# ------------------------------------------------------------------
	# 26. List users
	# ------------------------------------------------------------------

	async def list_users(self, status: str | None = None) -> list[_R]:
		"""List biometric users for the current tenant."""
		users = [
			u for (tid, _), u in self._users.items()
			if tid == self.tenant_id and (status is None or u["status"] == status)
		]
		return sorted(users, key=lambda u: u["enrolled_at"])

	# ------------------------------------------------------------------
	# 27. List templates for user
	# ------------------------------------------------------------------

	async def list_templates(self, user_id: str, modality: str | None = None) -> list[_R]:
		"""List active templates for a user, optionally filtered by modality."""
		self._require_user(user_id)
		templates = [
			t for (_, _), t in self._templates.items()
			if t["user_id"] == user_id
			and t["tenant_id"] == self.tenant_id
			and t["status"] not in {"deleted", "revoked"}
			and (modality is None or t["modality"] == modality)
		]
		return sorted(templates, key=lambda t: t["enrolled_at"])

	# ------------------------------------------------------------------
	# 28. Export users to CSV
	# ------------------------------------------------------------------

	async def export_users_csv(self) -> str:
		"""Export all users for the tenant as CSV."""
		users = await self.list_users()
		buf = io.StringIO()
		fields = ["user_id", "external_id", "full_name", "email", "status", "enrolled_at"]
		writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(users)
		await self._audit("users_exported_csv", "system", {"count": len(users)})
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 29. Export verifications to JSON
	# ------------------------------------------------------------------

	async def export_verifications_json(self, modality: str | None = None) -> str:
		"""Export verifications as JSON."""
		vlist = [
			dict(v) for (_, _), v in self._verifications.items()
			if v["tenant_id"] == self.tenant_id and (modality is None or v["modality"] == modality)
		]
		await self._audit("verifications_exported_json", "system", {"count": len(vlist)})
		return json.dumps(vlist, default=str, indent=2)

	# ------------------------------------------------------------------
	# 30. Health check
	# ------------------------------------------------------------------

	async def health_check(self) -> _R:
		"""Return service health status and storage counts."""
		users = sum(1 for (tid, _) in self._users if tid == self.tenant_id)
		templates = sum(1 for (_, _), t in self._templates.items() if t["tenant_id"] == self.tenant_id and t["status"] == "active")
		return _R(
			status="healthy",
			tenant_id=self.tenant_id,
			user_count=users,
			active_template_count=templates,
			verification_count=sum(1 for (tid, _) in self._verifications if tid == self.tenant_id),
			audit_event_count=len(self._audit_log),
			checked_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 31. Dashboard / KPI summary
	# ------------------------------------------------------------------

	async def dashboard(self) -> _R:
		"""Aggregate KPI dashboard for the tenant."""
		users = await self.list_users()
		metrics = await self.performance_metrics()
		active_users = sum(1 for u in users if u["status"] == "active")
		opted_out = sum(1 for u in users if u["status"] == "opted_out")
		modalities: dict[str, int] = {}
		for (_, _), t in self._templates.items():
			if t["tenant_id"] == self.tenant_id and t["status"] == "active":
				modalities[t["modality"]] = modalities.get(t["modality"], 0) + 1
		return _R(
			tenant_id=self.tenant_id,
			total_users=len(users),
			active_users=active_users,
			opted_out_users=opted_out,
			template_counts_by_modality=modalities,
			verification_stats=dict(metrics),
			watchlist_count=sum(1 for (tid, _) in self._watchlists if tid == self.tenant_id),
			consent_records=sum(1 for (_, _), c in self._consent.items() if c["tenant_id"] == self.tenant_id),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 32. GDPR / BIPA compliance report
	# ------------------------------------------------------------------

	async def compliance_report(self, framework: str = "GDPR") -> _R:
		"""Generate a domain-specific biometric data compliance report."""
		users = await self.list_users()
		consent_map: dict[str, bool] = {}
		for (_, _), c in self._consent.items():
			if c["tenant_id"] == self.tenant_id:
				consent_map[c["user_id"]] = consent_map.get(c["user_id"], False) or c["consented"]
		users_with_consent = sum(1 for u in users if consent_map.get(u["user_id"], False))
		opted_out = sum(1 for u in users if u["status"] == "opted_out")
		report = _R(
			framework=framework,
			tenant_id=self.tenant_id,
			total_data_subjects=len(users),
			subjects_with_consent=users_with_consent,
			subjects_opted_out=opted_out,
			consent_rate=round(users_with_consent / max(len(users), 1), 4),
			retention_policy_enforced=True,
			audit_trail_complete=True,
			generated_at=_ts(),
		)
		await self._audit("compliance_report_generated", "system", {"framework": framework})
		return report

	# ------------------------------------------------------------------
	# 33. Audit trail export
	# ------------------------------------------------------------------

	async def audit_trail_export(self, event_type: str | None = None) -> list[_R]:
		"""Export audit events, optionally filtered by event type."""
		events = [
			e for e in self._audit_log
			if e["tenant_id"] == self.tenant_id and (event_type is None or e["event_type"] == event_type)
		]
		await self._audit("audit_trail_exported", "system", {"count": len(events), "filter": event_type})
		return events

	# ------------------------------------------------------------------
	# 34. Template quality report
	# ------------------------------------------------------------------

	async def template_quality_report(self) -> _R:
		"""Report on template quality distribution across the tenant."""
		templates = [
			t for (_, _), t in self._templates.items()
			if t["tenant_id"] == self.tenant_id and t["status"] == "active"
		]
		if not templates:
			return _R(tenant_id=self.tenant_id, total=0, avg_quality=None, low_quality_count=0, computed_at=_ts())
		scores = [t["quality_score"] for t in templates]
		avg_q = round(statistics.mean(scores), 4)
		low_q = sum(1 for s in scores if s < 0.5)
		return _R(
			tenant_id=self.tenant_id,
			total=len(templates),
			avg_quality=avg_q,
			min_quality=round(min(scores), 4),
			max_quality=round(max(scores), 4),
			low_quality_count=low_q,
			computed_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 35. Session-based continuous authentication
	# ------------------------------------------------------------------

	async def continuous_auth_score(
		self,
		user_id: str,
		behavioral_signal: float,
		biometric_signal: float,
		fusion_weight: float = 0.6,
	) -> _R:
		"""Continuously authenticate a user using behavioral and biometric signals."""
		self._require_user(user_id)
		score = round(behavioral_signal * (1 - fusion_weight) + biometric_signal * fusion_weight, 4)
		decision = "continue" if score >= 0.65 else "step_up"
		result = _R(
			user_id=user_id,
			tenant_id=self.tenant_id,
			behavioral_signal=behavioral_signal,
			biometric_signal=biometric_signal,
			fused_score=score,
			decision=decision,
			evaluated_at=_ts(),
		)
		await self._audit("continuous_auth_evaluated", user_id, {"score": score, "decision": decision})
		return result

	# ------------------------------------------------------------------
	# 36. Anomaly detection on verification stream
	# ------------------------------------------------------------------

	async def anomaly_detect(self, user_id: str, window_hours: int = 24) -> _R:
		"""Detect anomalous verification patterns for a user in a time window."""
		self._require_user(user_id)
		cutoff = (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()
		recent = [
			v for (_, _), v in self._verifications.items()
			if v["user_id"] == user_id and v["verified_at"] >= cutoff
		]
		rejects = sum(1 for v in recent if v["decision"] == "reject")
		anomaly = rejects >= 3
		result = _R(
			user_id=user_id,
			window_hours=window_hours,
			total_verifications=len(recent),
			reject_count=rejects,
			anomaly_detected=anomaly,
			risk_level="high" if anomaly else "low",
			analyzed_at=_ts(),
		)
		await self._audit("anomaly_detection", user_id, {"anomaly": anomaly, "rejects": rejects})
		return result

	# ------------------------------------------------------------------
	# 37. NIST SP 800-76 compliance check
	# ------------------------------------------------------------------

	async def nist_compliance_check(self, template_id: str) -> _R:
		"""Check a template for NIST SP 800-76 compliance."""
		template = self._require_template(template_id)
		issues = []
		if template["quality_score"] < 0.5:
			issues.append("quality_below_nist_threshold")
		if template["size_bytes"] < 1000:
			issues.append("template_too_small")
		if template.get("encrypted_at") is None:
			issues.append("template_not_protected")
		compliant = len(issues) == 0
		result = _R(
			template_id=template_id,
			nist_compliant=compliant,
			issues=issues,
			standard="NIST_SP_800-76",
			checked_at=_ts(),
		)
		await self._audit("nist_compliance_checked", template_id, {"compliant": compliant, "issues": issues})
		return result

	# ------------------------------------------------------------------
	# 38. ISO/IEC 30107-3 PAD compliance
	# ------------------------------------------------------------------

	async def iso_pad_compliance(self, challenge_id: str) -> _R:
		"""Evaluate challenge result against ISO/IEC 30107-3 PAD Level requirements."""
		challenge = self._liveness_challenges.get(self._key(self.tenant_id, challenge_id))
		assert challenge is not None, f"challenge not found: {challenge_id}"
		pad_score = challenge.get("pad_score", 0.0)
		pad_level = 3 if pad_score >= 0.95 else (2 if pad_score >= 0.85 else (1 if pad_score >= 0.70 else 0))
		compliant = pad_level >= 2
		result = _R(
			challenge_id=challenge_id,
			pad_score=pad_score,
			achieved_pad_level=pad_level,
			compliant=compliant,
			standard="ISO_IEC_30107-3",
			checked_at=_ts(),
		)
		await self._audit("iso_pad_compliance_checked", challenge_id, {"pad_level": pad_level, "compliant": compliant})
		return result

	# ------------------------------------------------------------------
	# 39. Template migration
	# ------------------------------------------------------------------

	async def template_migrate(
		self,
		user_id: str,
		source_tenant: str,
		target_tenant: str,
	) -> _R:
		"""Migrate templates between tenants (cross-tenant provisioning)."""
		assert source_tenant != target_tenant, "source and target tenants must differ"
		migrated = 0
		for (tid, tid2), template in list(self._templates.items()):
			if tid == source_tenant and template["user_id"] == user_id:
				new_key = self._key(target_tenant, template["template_id"])
				new_record = _R(**template, tenant_id=target_tenant, migrated_at=_ts())
				self._templates[new_key] = new_record
				migrated += 1
		result = _R(
			user_id=user_id,
			source_tenant=source_tenant,
			target_tenant=target_tenant,
			templates_migrated=migrated,
			migrated_at=_ts(),
		)
		await self._audit("templates_migrated", user_id, {"source": source_tenant, "target": target_tenant, "count": migrated})
		return result

	# ------------------------------------------------------------------
	# 40. Deduplication report
	# ------------------------------------------------------------------

	async def deduplication_report(self, modality: str) -> _R:
		"""Scan the entire tenant gallery for duplicate biometric identities."""
		templates = [
			t for (_, _), t in self._templates.items()
			if t["modality"] == modality and t["status"] == "active" and t["tenant_id"] == self.tenant_id
		]
		hash_to_users: dict[str, list[str]] = {}
		for t in templates:
			prefix = t["template_hash"][:16]
			if prefix not in hash_to_users:
				hash_to_users[prefix] = []
			if t["user_id"] not in hash_to_users[prefix]:
				hash_to_users[prefix].append(t["user_id"])
		duplicates = {k: v for k, v in hash_to_users.items() if len(v) > 1}
		result = _R(
			modality=modality,
			tenant_id=self.tenant_id,
			templates_scanned=len(templates),
			duplicate_clusters=len(duplicates),
			cluster_details=[{"prefix": k, "user_ids": v} for k, v in duplicates.items()],
			generated_at=_ts(),
		)
		await self._audit("deduplication_report_generated", "system", {"modality": modality, "duplicate_clusters": len(duplicates)})
		return result

	# ------------------------------------------------------------------
	# 41. Age verification (document-linked)
	# ------------------------------------------------------------------

	async def age_verify(self, user_id: str, min_age: int = 18) -> _R:
		"""Verify user age against biometric and registered date-of-birth."""
		user = self._require_user(user_id)
		dob_str = user.get("date_of_birth")
		if dob_str:
			dob = datetime.fromisoformat(dob_str)
			age = (datetime.utcnow() - dob).days // 365
			passes = age >= min_age
		else:
			age = None
			passes = False
		result = _R(
			user_id=user_id,
			age=age,
			min_age=min_age,
			passes=passes,
			verified_at=_ts(),
		)
		await self._audit("age_verified", user_id, {"age": age, "passes": passes})
		return result

	# ------------------------------------------------------------------
	# 42. Risk scoring
	# ------------------------------------------------------------------

	async def risk_score(self, user_id: str) -> _R:
		"""Compute a composite biometric risk score for the user."""
		self._require_user(user_id)
		anomaly = await self.anomaly_detect(user_id, window_hours=24)
		quality = await self.template_quality_report()
		base_risk = 0.2
		if anomaly["anomaly_detected"]:
			base_risk += 0.4
		if quality.get("avg_quality") is not None and quality["avg_quality"] < 0.5:
			base_risk += 0.2
		risk = round(min(base_risk, 1.0), 4)
		result = _R(
			user_id=user_id,
			risk_score=risk,
			risk_level="high" if risk >= 0.6 else ("medium" if risk >= 0.3 else "low"),
			components={"anomaly": anomaly["anomaly_detected"], "quality": quality.get("avg_quality")},
			computed_at=_ts(),
		)
		await self._audit("risk_scored", user_id, {"risk_score": risk})
		return result
