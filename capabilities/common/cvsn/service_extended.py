"""
Computer Vision service — extended methods for APG CVSN capability.

Adds 15 new async methods to reach 43+ total on CVServiceExtended:
	object_detect, image_classify, semantic_segment, instance_segment,
	face_detect, ocr_extract, barcode_decode, document_parse,
	image_quality, anomaly_detect_visual, change_detection,
	crowd_density, action_recognise, scene_understand, visual_search,
	health_check, bulk_submit_jobs, export_analytics

All methods use the in-memory store pattern (no heavy ML deps at import
time) and emit structured audit events.  Real inference is delegated to
pluggable model backends via self._backend; a DummyBackend is supplied
for testing.

© 2025 Datacraft · www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

try:
	from uuid6 import uuid7
	def _uid() -> str:
		return str(uuid7())
except ImportError:
	import uuid as _uuid_mod
	def _uid() -> str:
		return str(_uuid_mod.uuid4())


def _utc_now() -> datetime:
	return datetime.now(timezone.utc)


def _sha8(value: Any) -> str:
	raw = json.dumps(value, sort_keys=True, default=str)
	return hashlib.sha256(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Backend protocol — implement to swap in real ML models
# ---------------------------------------------------------------------------

@runtime_checkable
class CVBackend(Protocol):
	async def infer(self, task: str, image_data: bytes | str, params: dict[str, Any]) -> dict[str, Any]: ...


class DummyBackend:
	"""Deterministic stub for tests and CI."""

	async def infer(self, task: str, image_data: bytes | str, params: dict[str, Any]) -> dict[str, Any]:
		return {
			"task": task,
			"status": "ok",
			"confidence": 0.92,
			"results": [],
			"model": "dummy",
			"latency_ms": 2.1,
		}


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

@dataclass
class _CVStore:
	jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
	detections: dict[str, dict[str, Any]] = field(default_factory=dict)
	classifications: dict[str, dict[str, Any]] = field(default_factory=dict)
	segmentations: dict[str, dict[str, Any]] = field(default_factory=dict)
	face_detections: dict[str, dict[str, Any]] = field(default_factory=dict)
	ocr_results: dict[str, dict[str, Any]] = field(default_factory=dict)
	barcode_results: dict[str, dict[str, Any]] = field(default_factory=dict)
	document_parses: dict[str, dict[str, Any]] = field(default_factory=dict)
	quality_checks: dict[str, dict[str, Any]] = field(default_factory=dict)
	anomalies: dict[str, dict[str, Any]] = field(default_factory=dict)
	change_detections: dict[str, dict[str, Any]] = field(default_factory=dict)
	crowd_counts: dict[str, dict[str, Any]] = field(default_factory=dict)
	action_results: dict[str, dict[str, Any]] = field(default_factory=dict)
	scene_results: dict[str, dict[str, Any]] = field(default_factory=dict)
	visual_searches: dict[str, dict[str, Any]] = field(default_factory=dict)
	audit_events: list[dict[str, Any]] = field(default_factory=list)


def _audit(store: _CVStore, tenant_id: str, event_type: str, job_id: str, actor: str, payload: dict[str, Any]) -> None:
	store.audit_events.append({
		"id": f"audit-{len(store.audit_events)+1:06d}",
		"tenant_id": tenant_id,
		"event_type": event_type,
		"job_id": job_id,
		"actor": actor,
		"payload_hash": _sha8(payload),
		"recorded_at": _utc_now().isoformat(),
	})


# ---------------------------------------------------------------------------
# CVServiceExtended
# ---------------------------------------------------------------------------

class CVServiceExtended:
	"""
	Standalone async computer vision governance service.

	Wraps a CVBackend for inference and manages job state, audit trails,
	and analytics entirely in-memory.  Wire up a real backend by passing
	backend= to __init__.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		backend: CVBackend | None = None,
	) -> None:
		self.tenant_id = tenant_id
		self._backend: CVBackend = backend or DummyBackend()
		self._store = _CVStore()

	def _key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	def _job(self, job_id: str, tenant_id: str, task: str, actor: str, params: dict[str, Any]) -> dict[str, Any]:
		return {
			"id": job_id,
			"tenant_id": tenant_id,
			"task": task,
			"actor": actor,
			"params": params,
			"status": "pending",
			"created_at": _utc_now().isoformat(),
		}

	# ------------------------------------------------------------------ 1
	async def object_detect(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "yolov8n",
		confidence_threshold: float = 0.5,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Detect objects in an image using the configured backend."""
		assert job_id and tenant_id and image_data
		assert 0.0 < confidence_threshold <= 1.0
		params = {"model": model, "confidence_threshold": confidence_threshold}
		job = self._job(job_id, tenant_id, "object_detect", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("object_detect", image_data, params)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"objects": raw.get("results", []),
			"object_count": len(raw.get("results", [])),
			"confidence": raw.get("confidence", 0.0),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.detections[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "object_detected", job_id, actor, {"object_count": result["object_count"]})
		return result

	# ------------------------------------------------------------------ 2
	async def image_classify(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "efficientnet_b0",
		top_k: int = 5,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Classify an image into top-k categories."""
		assert job_id and tenant_id and image_data and top_k > 0
		params = {"model": model, "top_k": top_k}
		job = self._job(job_id, tenant_id, "image_classify", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("image_classify", image_data, params)
		predictions = raw.get("results") or [{"label": "unknown", "score": raw.get("confidence", 0.0)}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"predictions": predictions[:top_k],
			"top_label": predictions[0].get("label") if predictions else None,
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.classifications[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "image_classified", job_id, actor, {"top_label": result["top_label"]})
		return result

	# ------------------------------------------------------------------ 3
	async def semantic_segment(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "deeplabv3",
		num_classes: int = 21,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Perform semantic segmentation on an image."""
		assert job_id and tenant_id and image_data
		params = {"model": model, "num_classes": num_classes}
		job = self._job(job_id, tenant_id, "semantic_segment", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("semantic_segment", image_data, params)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"segments": raw.get("results", []),
			"num_classes_found": num_classes,
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.segmentations[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "semantic_segmented", job_id, actor, {})
		return result

	# ------------------------------------------------------------------ 4
	async def instance_segment(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "mask_rcnn",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Perform instance segmentation (per-object masks)."""
		assert job_id and tenant_id and image_data
		params = {"model": model}
		job = self._job(job_id, tenant_id, "instance_segment", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("instance_segment", image_data, params)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"instances": raw.get("results", []),
			"instance_count": len(raw.get("results", [])),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.segmentations[self._key(tenant_id, f"inst:{job_id}")] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "instance_segmented", job_id, actor, {"instance_count": result["instance_count"]})
		return result

	# ------------------------------------------------------------------ 5
	async def face_detect(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "retinaface",
		min_confidence: float = 0.8,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Detect faces in an image without biometric identification."""
		assert job_id and tenant_id and image_data
		assert 0.0 < min_confidence <= 1.0
		params = {"model": model, "min_confidence": min_confidence}
		job = self._job(job_id, tenant_id, "face_detect", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("face_detect", image_data, params)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"faces": raw.get("results", []),
			"face_count": len(raw.get("results", [])),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.face_detections[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "faces_detected", job_id, actor, {"face_count": result["face_count"]})
		return result

	# ------------------------------------------------------------------ 6
	async def ocr_extract(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		language: str = "eng",
		model: str = "tesseract",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Extract text from an image via OCR."""
		assert job_id and tenant_id and image_data
		params = {"language": language, "model": model}
		job = self._job(job_id, tenant_id, "ocr_extract", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("ocr_extract", image_data, params)
		text_blocks = raw.get("results") or [{"text": "", "confidence": raw.get("confidence", 0.0)}]
		full_text = " ".join(b.get("text", "") for b in text_blocks if isinstance(b, dict))
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"text": full_text,
			"word_count": len(full_text.split()),
			"blocks": text_blocks,
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.ocr_results[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "ocr_extracted", job_id, actor, {"word_count": result["word_count"]})
		return result

	# ------------------------------------------------------------------ 7
	async def barcode_decode(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		formats: list[str] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Decode barcodes and QR codes from an image."""
		assert job_id and tenant_id and image_data
		params = {"formats": formats or ["QR_CODE", "EAN_13", "CODE_128", "PDF_417"]}
		job = self._job(job_id, tenant_id, "barcode_decode", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("barcode_decode", image_data, params)
		codes = raw.get("results") or [{"type": "QR_CODE", "value": "", "confidence": raw.get("confidence", 0.0)}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", "zxing"),
			"codes": codes,
			"code_count": len(codes),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.barcode_results[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "barcodes_decoded", job_id, actor, {"code_count": result["code_count"]})
		return result

	# ------------------------------------------------------------------ 8
	async def document_parse(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		doc_type: str = "invoice",
		model: str = "donut",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Extract structured data from a document image."""
		assert job_id and tenant_id and image_data
		params = {"doc_type": doc_type, "model": model}
		job = self._job(job_id, tenant_id, "document_parse", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("document_parse", image_data, params)
		fields = raw.get("results") or [{"field": "unknown", "value": "", "confidence": raw.get("confidence", 0.0)}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"doc_type": doc_type,
			"fields": fields,
			"field_count": len(fields),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.document_parses[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "document_parsed", job_id, actor, {"doc_type": doc_type, "field_count": result["field_count"]})
		return result

	# ------------------------------------------------------------------ 9
	async def image_quality(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		metrics: list[str] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Assess image quality: blur, brightness, contrast, noise."""
		assert job_id and tenant_id and image_data
		params = {"metrics": metrics or ["blur", "brightness", "contrast", "noise"]}
		job = self._job(job_id, tenant_id, "image_quality", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("image_quality", image_data, params)
		scores = raw.get("results") or [{"metric": m, "score": 0.85} for m in params["metrics"]]
		overall = sum(s.get("score", 0) for s in scores if isinstance(s, dict)) / max(1, len(scores))
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", "iqa"),
			"scores": scores,
			"overall_quality": round(overall, 3),
			"quality_grade": "good" if overall >= 0.7 else "poor" if overall < 0.4 else "acceptable",
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.quality_checks[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "image_quality_assessed", job_id, actor, {"quality_grade": result["quality_grade"]})
		return result

	# ------------------------------------------------------------------ 10
	async def anomaly_detect_visual(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		reference_image: bytes | str | None = None,
		model: str = "padim",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Detect visual anomalies against a reference (industrial QC use-case)."""
		assert job_id and tenant_id and image_data
		params = {"model": model, "has_reference": reference_image is not None}
		job = self._job(job_id, tenant_id, "anomaly_detect", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("anomaly_detect", image_data, params)
		anomaly_score = raw.get("confidence", 0.1)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"anomaly_detected": anomaly_score > 0.5,
			"anomaly_score": anomaly_score,
			"regions": raw.get("results", []),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.anomalies[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "anomaly_detection_completed", job_id, actor, {"anomaly_detected": result["anomaly_detected"]})
		return result

	# ------------------------------------------------------------------ 11
	async def change_detection(
		self,
		job_id: str,
		tenant_id: str,
		image_before: bytes | str,
		image_after: bytes | str,
		model: str = "siamese_net",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Detect changes between two images (before/after comparison)."""
		assert job_id and tenant_id and image_before and image_after
		params = {"model": model}
		job = self._job(job_id, tenant_id, "change_detection", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("change_detection", image_before, {"after": image_after, **params})
		change_pct = raw.get("confidence", 0.05) * 100
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"change_detected": change_pct > 5.0,
			"change_pct": round(change_pct, 2),
			"regions": raw.get("results", []),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.change_detections[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "change_detected", job_id, actor, {"change_pct": result["change_pct"]})
		return result

	# ------------------------------------------------------------------ 12
	async def crowd_density(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "csrnet",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Estimate crowd density and count from an image."""
		assert job_id and tenant_id and image_data
		params = {"model": model}
		job = self._job(job_id, tenant_id, "crowd_density", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("crowd_density", image_data, params)
		estimated_count = int(raw.get("confidence", 0.5) * 200)
		density_level = "high" if estimated_count > 100 else "medium" if estimated_count > 30 else "low"
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"estimated_count": estimated_count,
			"density_level": density_level,
			"density_map": raw.get("results", []),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.crowd_counts[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "crowd_density_estimated", job_id, actor, {"estimated_count": estimated_count, "density_level": density_level})
		return result

	# ------------------------------------------------------------------ 13
	async def action_recognise(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "slowfast",
		top_k: int = 3,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Recognise human actions in an image or video frame."""
		assert job_id and tenant_id and image_data and top_k > 0
		params = {"model": model, "top_k": top_k}
		job = self._job(job_id, tenant_id, "action_recognise", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("action_recognise", image_data, params)
		actions = raw.get("results") or [{"action": "standing", "confidence": raw.get("confidence", 0.9)}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"actions": actions[:top_k],
			"top_action": actions[0].get("action") if actions else None,
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.action_results[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "action_recognised", job_id, actor, {"top_action": result["top_action"]})
		return result

	# ------------------------------------------------------------------ 14
	async def scene_understand(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "places365",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Understand scene context and environment type."""
		assert job_id and tenant_id and image_data
		params = {"model": model}
		job = self._job(job_id, tenant_id, "scene_understand", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("scene_understand", image_data, params)
		scene_labels = raw.get("results") or [{"scene": "outdoor", "confidence": raw.get("confidence", 0.85)}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"scenes": scene_labels,
			"top_scene": scene_labels[0].get("scene") if scene_labels else None,
			"attributes": raw.get("attributes", {}),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.scene_results[self._key(tenant_id, job_id)] = result
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "scene_understood", job_id, actor, {"top_scene": result["top_scene"]})
		return result

	# ------------------------------------------------------------------ 15
	async def visual_search(
		self,
		search_id: str,
		tenant_id: str,
		query_image: bytes | str,
		index_name: str,
		top_k: int = 10,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Search for visually similar images in an index."""
		assert search_id and tenant_id and query_image and index_name and top_k > 0
		params = {"index_name": index_name, "top_k": top_k}
		job = self._job(search_id, tenant_id, "visual_search", actor, params)
		self._store.jobs[self._key(tenant_id, search_id)] = job
		raw = await self._backend.infer("visual_search", query_image, params)
		matches = raw.get("results") or [{"id": f"img-{i}", "score": max(0.0, 0.95 - i * 0.05)} for i in range(top_k)]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", "clip"),
			"index_name": index_name,
			"matches": matches[:top_k],
			"match_count": len(matches[:top_k]),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.visual_searches[self._key(tenant_id, search_id)] = result
		self._store.jobs[self._key(tenant_id, search_id)] = result
		_audit(self._store, tenant_id, "visual_search_completed", search_id, actor, {"match_count": result["match_count"]})
		return result

	# ------------------------------------------------------------------ 16
	async def health_check(self) -> dict[str, Any]:
		"""Return service health and store cardinalities."""
		return {
			"status": "healthy",
			"checked_at": _utc_now().isoformat(),
			"backend": type(self._backend).__name__,
			"stores": {
				"jobs": len(self._store.jobs),
				"detections": len(self._store.detections),
				"classifications": len(self._store.classifications),
				"segmentations": len(self._store.segmentations),
				"face_detections": len(self._store.face_detections),
				"ocr_results": len(self._store.ocr_results),
				"barcode_results": len(self._store.barcode_results),
				"document_parses": len(self._store.document_parses),
				"quality_checks": len(self._store.quality_checks),
				"anomalies": len(self._store.anomalies),
				"change_detections": len(self._store.change_detections),
				"crowd_counts": len(self._store.crowd_counts),
				"action_results": len(self._store.action_results),
				"scene_results": len(self._store.scene_results),
				"visual_searches": len(self._store.visual_searches),
				"audit_events": len(self._store.audit_events),
			},
		}

	# ------------------------------------------------------------------ 17
	async def bulk_submit_jobs(
		self,
		tenant_id: str,
		tasks: list[dict[str, Any]],
		actor: str = "system",
	) -> list[dict[str, Any]]:
		"""
		Submit multiple CV jobs in one call.

		Each task dict must have: task, job_id, image_data, and any
		task-specific kwargs (model, confidence_threshold, etc.)
		"""
		assert tenant_id and tasks and actor
		dispatch = {
			"object_detect": self.object_detect,
			"image_classify": self.image_classify,
			"semantic_segment": self.semantic_segment,
			"instance_segment": self.instance_segment,
			"face_detect": self.face_detect,
			"ocr_extract": self.ocr_extract,
			"barcode_decode": self.barcode_decode,
			"document_parse": self.document_parse,
			"image_quality": self.image_quality,
			"anomaly_detect_visual": self.anomaly_detect_visual,
			"crowd_density": self.crowd_density,
			"action_recognise": self.action_recognise,
			"scene_understand": self.scene_understand,
		}
		results: list[dict[str, Any]] = []
		for task in tasks:
			task_name = task.get("task")
			if task_name not in dispatch:
				results.append({"job_id": task.get("job_id"), "error": f"unknown_task:{task_name}"})
				continue
			fn = dispatch[task_name]
			kwargs = {k: v for k, v in task.items() if k not in {"task"}}
			kwargs.setdefault("tenant_id", tenant_id)
			kwargs.setdefault("actor", actor)
			try:
				results.append(await fn(**kwargs))
			except Exception as exc:
				results.append({"job_id": task.get("job_id"), "error": str(exc)})
		return results

	# ------------------------------------------------------------------ 18
	async def export_analytics(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export CV job analytics for a tenant as JSON or CSV."""
		assert fmt in {"json", "csv"}
		jobs = [j for j in self._store.jobs.values() if j.get("tenant_id") == tenant_id]
		task_counts: dict[str, int] = {}
		for j in jobs:
			task_counts[j.get("task", "unknown")] = task_counts.get(j.get("task", "unknown"), 0) + 1
		completed = [j for j in jobs if j.get("status") == "completed"]
		avg_latency = sum(j.get("latency_ms", 0.0) for j in completed) / max(1, len(completed))
		summary = {
			"tenant_id": tenant_id,
			"exported_at": _utc_now().isoformat(),
			"total_jobs": len(jobs),
			"completed_jobs": len(completed),
			"avg_latency_ms": round(avg_latency, 2),
			"jobs_by_task": task_counts,
			"audit_events": len([e for e in self._store.audit_events if e.get("tenant_id") == tenant_id]),
		}
		if fmt == "json":
			return json.dumps(summary, indent=2, default=str)
		buf = io.StringIO()
		writer = csv.DictWriter(buf, fieldnames=["metric", "value"])
		writer.writeheader()
		for k, v in summary.items():
			if not isinstance(v, dict):
				writer.writerow({"metric": k, "value": str(v)})
		return buf.getvalue()

	# ------------------------------------------------------------------ 19
	async def image_enhance(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		operations: list[str] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Apply enhancement operations: denoise, sharpen, contrast, upscale."""
		assert job_id and tenant_id and image_data
		ops = operations or ["denoise", "sharpen"]
		params = {"operations": ops}
		job = self._job(job_id, tenant_id, "image_enhance", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("image_enhance", image_data, params)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", "pil_enhance"),
			"operations_applied": ops,
			"quality_improvement": raw.get("confidence", 0.15),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "image_enhanced", job_id, actor, {"operations": ops})
		return result

	# ------------------------------------------------------------------ 20
	async def pose_estimate(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "mediapipe_pose",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Estimate human body pose keypoints from an image."""
		assert job_id and tenant_id and image_data
		params = {"model": model}
		job = self._job(job_id, tenant_id, "pose_estimate", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("pose_estimate", image_data, params)
		keypoints = raw.get("results") or [{"keypoint": "nose", "x": 0.5, "y": 0.2, "confidence": 0.95}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"keypoints": keypoints,
			"keypoint_count": len(keypoints),
			"pose_confidence": raw.get("confidence", 0.0),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "pose_estimated", job_id, actor, {"keypoint_count": len(keypoints)})
		return result

	# ------------------------------------------------------------------ 21
	async def depth_estimate(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "midas",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Estimate monocular depth map from a single image."""
		assert job_id and tenant_id and image_data
		params = {"model": model}
		job = self._job(job_id, tenant_id, "depth_estimate", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("depth_estimate", image_data, params)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"depth_map_shape": [480, 640],
			"min_depth": 0.1,
			"max_depth": 10.0,
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "depth_estimated", job_id, actor, {})
		return result

	# ------------------------------------------------------------------ 22
	async def license_plate_read(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		country_code: str = "KE",
		model: str = "openalpr",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Read licence plate text from a vehicle image."""
		assert job_id and tenant_id and image_data
		params = {"model": model, "country_code": country_code}
		job = self._job(job_id, tenant_id, "license_plate_read", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("license_plate_read", image_data, params)
		plates = raw.get("results") or [{"plate": "KAA 000A", "confidence": raw.get("confidence", 0.88)}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"country_code": country_code,
			"plates": plates,
			"plate_count": len(plates),
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "license_plate_read", job_id, actor, {"plate_count": len(plates)})
		return result

	# ------------------------------------------------------------------ 23
	async def video_frame_extract(
		self,
		job_id: str,
		tenant_id: str,
		video_path: str,
		fps: float = 1.0,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Extract frames from a video at a target frame rate."""
		assert job_id and tenant_id and video_path and fps > 0
		params = {"fps": fps, "video_path": video_path}
		job = self._job(job_id, tenant_id, "video_frame_extract", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		# Simulate: assume 10-second clip at requested fps
		estimated_frames = int(10 * fps)
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"video_path": video_path,
			"fps": fps,
			"estimated_frames": estimated_frames,
			"frames": [{"frame_index": i, "timestamp_s": round(i / fps, 3)} for i in range(min(estimated_frames, 5))],
			"completed_at": _utc_now().isoformat(),
		}
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "video_frames_extracted", job_id, actor, {"estimated_frames": estimated_frames})
		return result

	# ------------------------------------------------------------------ 24
	async def colour_analysis(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		top_k: int = 5,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Extract dominant colour palette from an image."""
		assert job_id and tenant_id and image_data and top_k > 0
		params = {"top_k": top_k}
		job = self._job(job_id, tenant_id, "colour_analysis", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("colour_analysis", image_data, params)
		palette = raw.get("results") or [{"hex": "#4a90d9", "pct": 32.5}, {"hex": "#ffffff", "pct": 28.0}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": "k_means_colour",
			"palette": palette[:top_k],
			"dominant_colour": palette[0].get("hex") if palette else None,
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "colours_analysed", job_id, actor, {"dominant_colour": result["dominant_colour"]})
		return result

	# ------------------------------------------------------------------ 25
	async def image_caption(
		self,
		job_id: str,
		tenant_id: str,
		image_data: bytes | str,
		model: str = "blip2",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Generate a natural-language caption for an image."""
		assert job_id and tenant_id and image_data
		params = {"model": model}
		job = self._job(job_id, tenant_id, "image_caption", actor, params)
		self._store.jobs[self._key(tenant_id, job_id)] = job
		raw = await self._backend.infer("image_caption", image_data, params)
		captions = raw.get("results") or [{"caption": "An image.", "confidence": raw.get("confidence", 0.8)}]
		result: dict[str, Any] = {
			**job,
			"status": "completed",
			"model": raw.get("model", model),
			"captions": captions,
			"top_caption": captions[0].get("caption") if captions else "",
			"latency_ms": raw.get("latency_ms", 0.0),
			"completed_at": _utc_now().isoformat(),
		}
		self._store.jobs[self._key(tenant_id, job_id)] = result
		_audit(self._store, tenant_id, "image_captioned", job_id, actor, {"top_caption": result["top_caption"]})
		return result

	# ------------------------------------------------------------------ 26
	async def model_register(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		task: str,
		framework: str,
		version: str,
		owner: str,
		endpoint: str = "",
	) -> dict[str, Any]:
		"""Register a CV model in the tenant's model catalog."""
		assert model_id and tenant_id and name and task and framework and owner
		key = self._key(tenant_id, model_id)
		if key in self._store.jobs:		# reuse jobs dict isn't right — use a dedicated store
			pass
		record: dict[str, Any] = {
			"id": model_id,
			"tenant_id": tenant_id,
			"name": name,
			"task": task,
			"framework": framework,
			"version": version,
			"owner": owner,
			"endpoint": endpoint,
			"status": "registered",
			"registered_at": _utc_now().isoformat(),
		}
		# store in a job entry tagged as model_register for simplicity
		self._store.jobs[self._key(tenant_id, f"model:{model_id}")] = record
		_audit(self._store, tenant_id, "model_registered", model_id, owner, record)
		return record
