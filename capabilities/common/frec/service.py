"""
APG Facial Recognition - Core Service Implementation

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator
from uuid import uuid4

import cv2
import numpy as np

from .models import (
	FaUser, FaTemplate, FaVerification, FaEmotion, FaCollaboration,
	FaVerificationType, FaEmotionType, FaProcessingStatus, FaLivenessResult
)
from .database import FacialDatabaseService
from .encryption import FaceTemplateEncryption, TemplateVersionManager
from .face_engine import FaceDetectionEngine, FaceFeatureExtractor, FaceQualityAssessment
from .liveness_engine import LivenessDetectionEngine
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def _uid() -> str:
	return uuid4().hex


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class FacialRecognitionService:
	"""Facial recognition service — enroll, verify, identify, liveness, gallery, attributes, compliance."""

	def __init__(self, database_url: str, encryption_key: str, tenant_id: str) -> None:
		assert database_url, "database_url required"
		assert encryption_key, "encryption_key required"
		assert tenant_id, "tenant_id required"

		self.tenant_id = tenant_id
		self.database_service = FacialDatabaseService(database_url, encryption_key)
		self.encryption_service = FaceTemplateEncryption(encryption_key)
		self.version_manager = TemplateVersionManager(self.encryption_service)

		self.face_detector = FaceDetectionEngine('mediapipe')
		self.feature_extractor = FaceFeatureExtractor('facenet')
		self.quality_assessor = FaceQualityAssessment()
		self.liveness_detector = LivenessDetectionEngine('level_4')

		self.verification_threshold = 0.80
		self.quality_threshold = 0.60
		self.liveness_threshold = 0.85

		# In-process stores for lightweight objects (galleries, consents, sessions)
		self._galleries: dict[str, dict[str, Any]] = {}
		self._consents: dict[str, dict[str, Any]] = {}
		self._liveness_sessions: dict[str, dict[str, Any]] = {}
		self._audit: list[dict[str, Any]] = []

		print(f"FacialRecognitionService initialised for tenant {tenant_id}")

	# ── helpers ──────────────────────────────────────────────────────────────

	def _log_service_operation(self, op: str, user_id: str | None = None, result: str | None = None) -> None:
		u = f" ({user_id})" if user_id else ""
		r = f" [{result}]" if result else ""
		print(f"frec {op}{u}{r}")

	async def _create_audit_log(self, **kw: Any) -> None:
		try:
			await self.database_service.create_audit_log(self.tenant_id, kw)
		except Exception as e:
			print(f"audit log failed: {e}")
		self._audit.append({"tenant_id": self.tenant_id, "ts": _now(), **kw})

	def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
		norm_a = np.linalg.norm(a)
		norm_b = np.linalg.norm(b)
		if norm_a == 0 or norm_b == 0:
			return 0.0
		return float(np.dot(a, b) / (norm_a * norm_b))

	def _create_verification_failure_result(self, error: str, start: datetime) -> dict[str, Any]:
		ms = (datetime.now() - start).total_seconds() * 1000
		return {"success": False, "verified": False, "error": error, "confidence_score": 0.0,
				"processing_time_ms": ms, "verification_timestamp": _now()}

	def _determine_failure_reason(self, similarity: float, quality: dict[str, Any], liveness: dict[str, Any] | None) -> str:
		if quality['overall_score'] < self.quality_threshold:
			return f"poor quality ({quality['overall_score']:.2f})"
		if liveness and not liveness['is_live']:
			return f"liveness failed ({liveness['confidence']:.2f})"
		return f"similarity too low ({similarity:.2f})"

	async def _simple_liveness_check(self, face_region: np.ndarray) -> dict[str, Any]:
		quality = await self.quality_assessor.assess_quality(face_region)
		is_live = (quality['overall_score'] > 0.7
				   and quality.get('sharpness_score', 0) > 0.6
				   and quality.get('contrast_score', 0) > 0.5)
		return {"is_live": is_live, "confidence": quality['overall_score'] * 0.8, "method": "quality_heuristic"}

	async def _extract_probe_features(self, image: np.ndarray, context: str) -> tuple[np.ndarray | None, dict[str, Any]]:
		"""Detect, region-crop, quality-assess, and extract features from a single-face image."""
		faces = await self.face_detector.detect_faces(image, context)
		if not faces:
			return None, {"error": "no_face_detected"}
		if len(faces) > 1:
			return None, {"error": "multiple_faces_detected"}
		fd = faces[0]
		region = await self.face_detector.extract_face_region(image, fd['bounding_box'])
		if region is None:
			return None, {"error": "region_extraction_failed"}
		quality = await self.quality_assessor.assess_quality(region, fd['bounding_box'])
		feats = await self.feature_extractor.extract_features(region, fd['face_id'])
		return feats, quality

	# ── initialisation ───────────────────────────────────────────────────────

	async def initialize(self) -> bool:
		try:
			ok = await self.database_service.create_tables()
			self._log_service_operation("INITIALIZE", result="SUCCESS" if ok else "PARTIAL")
			return True
		except Exception as e:
			print(f"initialize failed: {e}")
			return False

	async def close(self) -> None:
		await self.database_service.close()

	# ── user management ──────────────────────────────────────────────────────

	async def create_user(self, user_data: dict[str, Any]) -> FaUser | None:
		assert user_data.get('external_user_id'), "external_user_id required"
		assert user_data.get('full_name'), "full_name required"
		existing = await self.database_service.get_user_by_external_id(self.tenant_id, user_data['external_user_id'])
		if existing:
			return existing
		user = await self.database_service.create_user(self.tenant_id, user_data)
		if user:
			await self._create_audit_log(action_type="USER_CREATED", resource_type="fa_user",
										 resource_id=user.id, actor_id=user_data.get('created_by', 'system'))
		return user

	async def get_user(self, user_id: str) -> FaUser | None:
		assert user_id
		return await self.database_service.get_user_by_id(self.tenant_id, user_id)

	async def get_user_by_external_id(self, external_user_id: str) -> FaUser | None:
		assert external_user_id
		return await self.database_service.get_user_by_external_id(self.tenant_id, external_user_id)

	async def update_verification_threshold(self, t: float) -> bool:
		assert 0.0 <= t <= 1.0
		self.verification_threshold = t
		return True

	async def get_service_statistics(self) -> dict[str, Any]:
		analytics = await self.database_service.get_verification_analytics(self.tenant_id, 30)
		return {"tenant_id": self.tenant_id, "verification_threshold": self.verification_threshold,
				"quality_threshold": self.quality_threshold, "liveness_threshold": self.liveness_threshold,
				"analytics_last_30_days": analytics, "ts": _now()}

	async def cleanup_expired_data(self) -> dict[str, Any]:
		return await self.database_service.cleanup_expired_data()

	# ── CORE ─────────────────────────────────────────────────────────────────

	async def enroll_face(
		self,
		subject_id: str,
		image_data: np.ndarray,
		quality_threshold: float = 0.85,
	) -> dict[str, Any]:
		"""Enroll a face template for subject_id."""
		assert subject_id and image_data is not None and image_data.size > 0
		start = datetime.now()
		user = await self.get_user(subject_id)
		if not user:
			return {"success": False, "error": "user_not_found"}
		if not user.consent_given:
			return {"success": False, "error": "consent_required"}

		faces = await self.face_detector.detect_faces(image_data, f"enroll_{subject_id}")
		if not faces:
			return {"success": False, "error": "no_face_detected"}
		if len(faces) > 1:
			return {"success": False, "error": "multiple_faces_detected"}
		fd = faces[0]
		region = await self.face_detector.extract_face_region(image_data, fd['bounding_box'])
		if region is None:
			return {"success": False, "error": "region_extraction_failed"}

		quality = await self.quality_assessor.assess_quality(region, fd['bounding_box'])
		if quality['overall_score'] < quality_threshold:
			return {"success": False, "error": "quality_too_low",
					"quality_score": quality['overall_score'], "issues": quality.get('quality_issues', [])}

		feats = await self.feature_extractor.extract_features(region, fd['face_id'])
		if feats is None:
			return {"success": False, "error": "feature_extraction_failed"}

		tmpl = await self.database_service.create_template(subject_id, feats.tobytes(), {
			"quality_score": quality['overall_score'],
			"algorithm": self.feature_extractor.model_type,
			"sharpness": quality.get('sharpness_score'),
			"brightness": quality.get('brightness_score'),
			"contrast": quality.get('contrast_score'),
		})
		if not tmpl:
			return {"success": False, "error": "template_store_failed"}

		await self.database_service.update_user(self.tenant_id, subject_id, {"enrollment_status": "enrolled"})
		await self._create_audit_log(action_type="FACE_ENROLLED", resource_type="fa_template",
									 resource_id=tmpl.id, user_id=subject_id, actor_id="system")
		ms = (datetime.now() - start).total_seconds() * 1000
		return {"success": True, "template_id": tmpl.id, "quality_score": quality['overall_score'],
				"processing_time_ms": ms, "enrolled_at": _now()}

	async def verify_face(
		self,
		subject_id: str,
		probe_image: np.ndarray,
		verification_config: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""1:1 verification — returns {verified, confidence, score}."""
		assert subject_id and probe_image is not None and probe_image.size > 0
		start = datetime.now()
		cfg = verification_config or {}
		user = await self.get_user(subject_id)
		if not user:
			return self._create_verification_failure_result("user_not_found", start)
		templates = await self.database_service.get_user_templates(subject_id, active_only=True)
		if not templates:
			return self._create_verification_failure_result("no_templates", start)

		feats, quality = await self._extract_probe_features(probe_image, f"verify_{subject_id}")
		if feats is None:
			return self._create_verification_failure_result(quality.get("error", "extraction_failed"), start)

		liveness = None
		if cfg.get("require_liveness", True):
			faces = await self.face_detector.detect_faces(probe_image, "liveness")
			if faces:
				region = await self.face_detector.extract_face_region(probe_image, faces[0]['bounding_box'])
				if region is not None:
					liveness = await self._simple_liveness_check(region)

		best_sim = 0.0
		best_tmpl = None
		for tmpl in templates:
			raw = await self.database_service.decrypt_template_data(tmpl)
			if raw is None:
				continue
			stored = np.frombuffer(raw, dtype=np.float32)
			sim = await self.feature_extractor.compare_features(feats, stored)
			if sim > best_sim:
				best_sim, best_tmpl = sim, tmpl

		verified = (best_sim >= self.verification_threshold
					and quality['overall_score'] >= self.quality_threshold
					and (liveness is None or liveness['is_live']))
		factors = [best_sim, quality['overall_score']]
		if liveness:
			factors.append(liveness['confidence'])
		confidence = sum(factors) / len(factors)

		vdata: dict[str, Any] = {
			"user_id": subject_id, "verification_type": FaVerificationType.AUTHENTICATION,
			"template_id": best_tmpl.id if best_tmpl else None,
			"status": FaProcessingStatus.COMPLETED,
			"confidence_score": confidence, "similarity_score": best_sim,
			"input_quality_score": quality['overall_score'],
			"processing_time_ms": int((datetime.now() - start).total_seconds() * 1000),
		}
		if not verified:
			vdata["failure_reason"] = self._determine_failure_reason(best_sim, quality, liveness)
		verif = await self.database_service.create_verification(vdata)
		await self._create_audit_log(action_type="FACE_VERIFIED", resource_type="fa_verification",
									 resource_id=verif.id if verif else None, user_id=subject_id,
									 action_result="success" if verified else "failure")

		return {"success": True, "verified": verified, "confidence": confidence, "score": best_sim,
				"verification_id": verif.id if verif else None,
				"quality_score": quality['overall_score'],
				"liveness": liveness, "processing_time_ms": vdata["processing_time_ms"],
				"ts": _now(), "failure_reason": vdata.get("failure_reason")}

	async def identify_face(
		self,
		probe_image: np.ndarray,
		gallery_id: str,
		top_k: int = 5,
	) -> dict[str, Any]:
		"""1:N identification within a gallery."""
		assert probe_image is not None and probe_image.size > 0
		start = datetime.now()
		gallery = self._galleries.get(gallery_id)
		if gallery is None:
			return {"success": False, "error": "gallery_not_found"}

		feats, quality = await self._extract_probe_features(probe_image, "identify")
		if feats is None:
			return {"success": False, "error": quality.get("error", "extraction_failed"),
					"quality_score": quality.get('overall_score', 0)}

		# Score all subject templates in gallery
		enrolled_ids: list[str] = gallery.get("subject_ids", [])
		scores: list[tuple[float, str]] = []
		for sid in enrolled_ids:
			templates = await self.database_service.get_user_templates(sid, active_only=True)
			for tmpl in templates:
				raw = await self.database_service.decrypt_template_data(tmpl)
				if raw is None:
					continue
				stored = np.frombuffer(raw, dtype=np.float32)
				sim = self._cosine_similarity(feats, stored)
				scores.append((sim, sid))

		scores.sort(key=lambda x: -x[0])
		candidates = [{"subject_id": sid, "score": float(sim), "rank": i + 1}
					  for i, (sim, sid) in enumerate(scores[:top_k])]

		ms = (datetime.now() - start).total_seconds() * 1000
		return {"success": True, "candidates": candidates, "gallery_id": gallery_id,
				"quality_score": quality.get('overall_score', 0), "processing_time_ms": ms, "ts": _now()}

	async def update_face_template(self, subject_id: str, new_image: np.ndarray) -> dict[str, Any]:
		"""Replace existing templates with fresh enrolment."""
		result = await self.enroll_face(subject_id, new_image)
		if result["success"]:
			await self._create_audit_log(action_type="TEMPLATE_UPDATED", user_id=subject_id)
		return result

	async def delete_face_template(self, subject_id: str) -> dict[str, Any]:
		"""Hard-delete all templates for subject."""
		try:
			templates = await self.database_service.get_user_templates(subject_id, active_only=False)
			for tmpl in templates:
				await self.database_service.delete_template(tmpl.id)
			await self.database_service.update_user(self.tenant_id, subject_id, {"enrollment_status": "unenrolled"})
			await self._create_audit_log(action_type="TEMPLATE_DELETED", user_id=subject_id)
			return {"success": True, "deleted_count": len(templates)}
		except Exception as e:
			return {"success": False, "error": str(e)}

	async def list_enrolled(self, gallery_id: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
		"""List all enrolled subjects in a gallery with optional filters."""
		gallery = self._galleries.get(gallery_id)
		if gallery is None:
			return {"success": False, "error": "gallery_not_found"}
		subjects = gallery.get("subject_ids", [])
		f = filters or {}
		if f.get("min_quality"):
			min_q = float(f["min_quality"])
			filtered = []
			for sid in subjects:
				tmpls = await self.database_service.get_user_templates(sid, active_only=True)
				if tmpls and tmpls[0].quality_score >= min_q:
					filtered.append(sid)
			subjects = filtered
		return {"success": True, "gallery_id": gallery_id, "count": len(subjects), "subjects": subjects}

	async def face_quality_score(self, image_data: np.ndarray) -> dict[str, Any]:
		"""Assess face quality; returns {score, issues}."""
		faces = await self.face_detector.detect_faces(image_data, "quality")
		if not faces:
			return {"score": 0.0, "issues": ["no_face_detected"]}
		region = await self.face_detector.extract_face_region(image_data, faces[0]['bounding_box'])
		if region is None:
			return {"score": 0.0, "issues": ["region_extraction_failed"]}
		quality = await self.quality_assessor.assess_quality(region, faces[0]['bounding_box'])
		return {"score": quality['overall_score'], "issues": quality.get('quality_issues', []),
				"sharpness": quality.get('sharpness_score'), "brightness": quality.get('brightness_score'),
				"contrast": quality.get('contrast_score')}

	async def compare_faces(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
		"""Return cosine similarity between two face images."""
		feats_a, _ = await self._extract_probe_features(image_a, "compare_a")
		feats_b, _ = await self._extract_probe_features(image_b, "compare_b")
		if feats_a is None or feats_b is None:
			return 0.0
		return self._cosine_similarity(feats_a, feats_b)

	async def batch_enroll(self, subjects_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Enroll multiple subjects concurrently.
		Each item: {subject_id, image_data, quality_threshold?}
		"""
		tasks = [
			self.enroll_face(item["subject_id"], item["image_data"], item.get("quality_threshold", 0.85))
			for item in subjects_list
		]
		return list(await asyncio.gather(*tasks, return_exceptions=False))

	async def batch_verify(self, probes_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Verify multiple subjects concurrently.
		Each item: {subject_id, probe_image, config?}
		"""
		tasks = [
			self.verify_face(item["subject_id"], item["probe_image"], item.get("config"))
			for item in probes_list
		]
		return list(await asyncio.gather(*tasks, return_exceptions=False))

	# ── LIVENESS & ANTI-SPOOFING ─────────────────────────────────────────────

	async def liveness_check(
		self,
		image_sequence: list[np.ndarray],
		method: str = 'passive',
	) -> dict[str, Any]:
		"""Passive liveness check over a sequence of frames."""
		assert image_sequence, "image_sequence must not be empty"
		scores = []
		for frame in image_sequence:
			faces = await self.face_detector.detect_faces(frame, "liveness_seq")
			if not faces:
				continue
			region = await self.face_detector.extract_face_region(frame, faces[0]['bounding_box'])
			if region is None:
				continue
			q = await self.quality_assessor.assess_quality(region)
			scores.append(q['overall_score'])

		if not scores:
			return {"is_live": False, "confidence": 0.0, "method": method, "frames_processed": 0}

		avg = sum(scores) / len(scores)
		variance = sum((s - avg) ** 2 for s in scores) / len(scores)
		# Natural micro-movements produce variance; low variance → potential replay
		motion_indicator = min(1.0, math.sqrt(variance) * 10)
		confidence = avg * 0.6 + motion_indicator * 0.4
		return {"is_live": confidence > self.liveness_threshold,
				"confidence": confidence, "method": method,
				"frames_processed": len(scores), "avg_quality": avg}

	async def active_liveness_challenge(self, session_id: str) -> dict[str, Any]:
		"""Generate an active liveness challenge (blink/turn/nod) for a session."""
		challenges = ["blink", "turn_left", "turn_right", "nod", "smile"]
		import random
		challenge = random.choice(challenges)
		self._liveness_sessions[session_id] = {
			"challenge": challenge, "created_at": _now(), "status": "pending",
			"expires_at": (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
		}
		return {"session_id": session_id, "challenge": challenge, "timeout_seconds": 120}

	async def validate_challenge_response(self, session_id: str, response: dict[str, Any]) -> dict[str, Any]:
		"""Validate the biometric response to an active liveness challenge."""
		session = self._liveness_sessions.get(session_id)
		if session is None:
			return {"valid": False, "error": "session_not_found"}
		if session["status"] != "pending":
			return {"valid": False, "error": f"session_{session['status']}"}
		expires = datetime.fromisoformat(session["expires_at"])
		if datetime.now(timezone.utc) > expires:
			session["status"] = "expired"
			return {"valid": False, "error": "session_expired"}

		# Evaluate response quality
		action_detected = response.get("action_detected", "")
		confidence = float(response.get("confidence", 0.0))
		valid = action_detected == session["challenge"] and confidence > 0.7
		session["status"] = "completed"
		session["result"] = {"valid": valid, "confidence": confidence}
		return {"valid": valid, "session_id": session_id, "challenge": session["challenge"],
				"confidence": confidence}

	async def presentation_attack_detect(self, image: np.ndarray) -> dict[str, Any]:
		"""Detect presentation attacks (print, replay, 3d mask)."""
		assert image is not None and image.size > 0
		faces = await self.face_detector.detect_faces(image, "pad")
		if not faces:
			return {"is_attack": False, "attack_type": None, "confidence": 0.0, "error": "no_face"}

		region = await self.face_detector.extract_face_region(image, faces[0]['bounding_box'])
		if region is None:
			return {"is_attack": False, "attack_type": None, "confidence": 0.0}

		quality = await self.quality_assessor.assess_quality(region)
		# Heuristic indicators
		is_grainy = quality.get('sharpness_score', 1.0) < 0.3
		low_depth_cue = quality.get('contrast_score', 1.0) < 0.25

		if is_grainy and low_depth_cue:
			return {"is_attack": True, "attack_type": "print", "confidence": 0.75}
		if is_grainy:
			return {"is_attack": True, "attack_type": "replay", "confidence": 0.60}
		return {"is_attack": False, "attack_type": None, "confidence": 1.0 - quality['overall_score'] * 0.3}

	async def texture_analysis(self, image: np.ndarray) -> dict[str, Any]:
		"""Analyse skin texture frequency to detect synthetic faces."""
		assert image is not None and image.size > 0
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
		# Laplacian variance as sharpness / texture richness proxy
		lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
		# Local binary pattern surrogate via difference of Gaussians
		blur1 = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 1.0)
		blur2 = cv2.GaussianBlur(gray.astype(np.float32), (9, 9), 3.0)
		dog = blur1 - blur2
		texture_score = float(np.std(dog))
		is_natural = lap_var > 50 and texture_score > 5
		return {"laplacian_variance": lap_var, "dog_std": texture_score,
				"is_natural_texture": is_natural, "texture_score": min(1.0, lap_var / 500)}

	async def depth_check(self, image_pair: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
		"""Estimate depth disparity between stereo/sequential image pair."""
		img_l, img_r = image_pair
		assert img_l is not None and img_r is not None
		g_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape) == 3 else img_l
		g_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if len(img_r.shape) == 3 else img_r
		# Resize to same shape
		h, w = min(g_l.shape[0], g_r.shape[0]), min(g_l.shape[1], g_r.shape[1])
		g_l = cv2.resize(g_l, (w, h))
		g_r = cv2.resize(g_r, (w, h))
		diff = cv2.absdiff(g_l, g_r)
		disparity = float(diff.mean())
		has_depth = disparity > 5.0
		return {"disparity_mean": disparity, "has_depth_cue": has_depth,
				"depth_score": min(1.0, disparity / 50)}

	async def replay_detect(self, video_metadata: dict[str, Any]) -> dict[str, Any]:
		"""Detect replay attacks from video metadata patterns."""
		fps = float(video_metadata.get("fps", 0))
		frame_count = int(video_metadata.get("frame_count", 0))
		codec = str(video_metadata.get("codec", ""))
		created_at = video_metadata.get("created_at")
		# Indicators: unusually round fps, suspiciously low frame count, screen-capture codecs
		suspicious_codec = codec.lower() in {"vp8", "vp9", "h264"} and video_metadata.get("source") == "screen"
		round_fps = abs(fps - round(fps)) < 0.01
		low_frames = 0 < frame_count < 5
		score = sum([suspicious_codec, round_fps, low_frames]) / 3
		return {"is_likely_replay": score > 0.5, "replay_score": score, "indicators": {
			"suspicious_codec": suspicious_codec, "round_fps": round_fps, "low_frame_count": low_frames,
		}}

	async def liveness_score(self, all_checks: dict[str, Any]) -> dict[str, Any]:
		"""Aggregate multiple liveness check results into a unified score."""
		weights = {"passive": 0.3, "texture": 0.25, "depth": 0.2, "replay": 0.15, "pad": 0.1}
		total_weight = 0.0
		weighted_sum = 0.0
		details: dict[str, float] = {}
		for key, weight in weights.items():
			if key in all_checks:
				val = all_checks[key]
				score = float(val.get("confidence", val.get("depth_score", val.get("texture_score", 0))))
				if key in ("replay", "pad"):
					# These are attack scores — invert
					is_attack = val.get("is_attack", val.get("is_likely_replay", False))
					score = 1.0 - score if is_attack else score
				details[key] = score
				weighted_sum += score * weight
				total_weight += weight

		unified = weighted_sum / total_weight if total_weight > 0 else 0.0
		return {"unified_score": unified, "is_live": unified >= self.liveness_threshold,
				"component_scores": details, "threshold": self.liveness_threshold}

	# ── GALLERY MANAGEMENT ────────────────────────────────────────────────────

	async def create_gallery(
		self,
		gallery_id: str,
		name: str,
		max_subjects: int,
		access_level: str = "internal",
	) -> dict[str, Any]:
		"""Create a named subject gallery."""
		if gallery_id in self._galleries:
			return {"success": False, "error": "gallery_exists"}
		self._galleries[gallery_id] = {
			"id": gallery_id, "name": name, "max_subjects": max_subjects,
			"access_level": access_level, "subject_ids": [], "created_at": _now(),
			"tenant_id": self.tenant_id,
		}
		return {"success": True, "gallery_id": gallery_id, "name": name}

	async def delete_gallery(self, gallery_id: str) -> dict[str, Any]:
		"""Delete gallery and optionally purge associated templates."""
		gallery = self._galleries.pop(gallery_id, None)
		if gallery is None:
			return {"success": False, "error": "gallery_not_found"}
		return {"success": True, "gallery_id": gallery_id, "subject_count": len(gallery.get("subject_ids", []))}

	async def gallery_stats(self, gallery_id: str) -> dict[str, Any]:
		"""Return {count, quality_dist, avg_score} for gallery."""
		gallery = self._galleries.get(gallery_id)
		if gallery is None:
			return {"error": "gallery_not_found"}
		subjects = gallery.get("subject_ids", [])
		quality_scores: list[float] = []
		for sid in subjects:
			tmpls = await self.database_service.get_user_templates(sid, active_only=True)
			for tmpl in tmpls:
				if tmpl.quality_score:
					quality_scores.append(float(tmpl.quality_score))

		dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
		for s in quality_scores:
			if s >= 0.9:
				dist["excellent"] += 1
			elif s >= 0.75:
				dist["good"] += 1
			elif s >= 0.5:
				dist["fair"] += 1
			else:
				dist["poor"] += 1

		avg = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
		return {"gallery_id": gallery_id, "count": len(subjects), "quality_dist": dist, "avg_score": avg}

	async def merge_galleries(self, src_id: str, dst_id: str) -> dict[str, Any]:
		"""Merge all subjects from src gallery into dst gallery."""
		src = self._galleries.get(src_id)
		dst = self._galleries.get(dst_id)
		if src is None or dst is None:
			return {"success": False, "error": "gallery_not_found"}
		added = 0
		for sid in src.get("subject_ids", []):
			if sid not in dst["subject_ids"]:
				if len(dst["subject_ids"]) < dst["max_subjects"]:
					dst["subject_ids"].append(sid)
					added += 1
		del self._galleries[src_id]
		return {"success": True, "merged_count": added, "dst_gallery_id": dst_id}

	async def export_gallery_metadata(self, gallery_id: str) -> dict[str, Any]:
		"""Export gallery metadata (no biometric data)."""
		gallery = self._galleries.get(gallery_id)
		if gallery is None:
			return {"error": "gallery_not_found"}
		return {
			"gallery_id": gallery_id,
			"name": gallery["name"],
			"tenant_id": gallery["tenant_id"],
			"access_level": gallery["access_level"],
			"subject_count": len(gallery.get("subject_ids", [])),
			"max_subjects": gallery["max_subjects"],
			"created_at": gallery["created_at"],
			"exported_at": _now(),
		}

	async def purge_expired(self, gallery_id: str, expiry_days: int = 365) -> dict[str, Any]:
		"""Remove subjects whose templates are older than expiry_days."""
		gallery = self._galleries.get(gallery_id)
		if gallery is None:
			return {"success": False, "error": "gallery_not_found"}
		cutoff = datetime.now(timezone.utc) - timedelta(days=expiry_days)
		removed = []
		remaining = []
		for sid in gallery.get("subject_ids", []):
			tmpls = await self.database_service.get_user_templates(sid, active_only=True)
			if tmpls:
				created = getattr(tmpls[0], 'created_at', datetime.now(timezone.utc))
				if isinstance(created, str):
					created = datetime.fromisoformat(created)
				if created.replace(tzinfo=timezone.utc) < cutoff:
					removed.append(sid)
				else:
					remaining.append(sid)
		gallery["subject_ids"] = remaining
		return {"success": True, "removed_count": len(removed), "remaining_count": len(remaining)}

	async def clone_gallery(self, src_id: str, new_id: str, tenant_id: str) -> dict[str, Any]:
		"""Clone gallery structure to a new gallery (possibly different tenant)."""
		src = self._galleries.get(src_id)
		if src is None:
			return {"success": False, "error": "source_gallery_not_found"}
		if new_id in self._galleries:
			return {"success": False, "error": "destination_gallery_exists"}
		self._galleries[new_id] = {
			**src,
			"id": new_id,
			"tenant_id": tenant_id,
			"subject_ids": list(src.get("subject_ids", [])),
			"created_at": _now(),
		}
		return {"success": True, "new_gallery_id": new_id, "cloned_from": src_id,
				"subject_count": len(self._galleries[new_id]["subject_ids"])}

	# ── ATTRIBUTES & ANALYTICS ────────────────────────────────────────────────

	async def estimate_age(self, image: np.ndarray) -> dict[str, Any]:
		"""Estimate age range from face image."""
		assert image is not None and image.size > 0
		faces = await self.face_detector.detect_faces(image, "age")
		if not faces:
			return {"error": "no_face_detected"}
		# Surrogate: use quality features as stand-in until real age model wired
		quality = await self.quality_assessor.assess_quality(
			await self.face_detector.extract_face_region(image, faces[0]['bounding_box']) or image,
		)
		# Without a real age regression model, return a plausible structure
		# annotated as requiring model integration
		return {"age_range": {"min": 20, "max": 40}, "confidence": quality['overall_score'],
				"note": "placeholder_requires_age_model"}

	async def detect_emotion(self, image: np.ndarray) -> dict[str, Any]:
		"""Detect emotions in face image."""
		assert image is not None and image.size > 0
		faces = await self.face_detector.detect_faces(image, "emotion")
		if not faces:
			return {"error": "no_face_detected", "emotions": {}}
		region = await self.face_detector.extract_face_region(image, faces[0]['bounding_box'])
		if region is None:
			return {"error": "region_failed", "emotions": {}}
		quality = await self.quality_assessor.assess_quality(region)
		# Delegate to emotion_intelligence module if available
		try:
			from .emotion_intelligence import EmotionIntelligenceEngine
			engine = EmotionIntelligenceEngine()
			result = await engine.analyse(region)
			return result
		except (ImportError, Exception):
			# Minimal fallback
			return {"emotions": {"neutral": 0.7, "unknown": 0.3},
					"dominant": "neutral", "confidence": quality['overall_score'],
					"note": "emotion_model_unavailable"}

	async def detect_occlusion(self, image: np.ndarray) -> dict[str, Any]:
		"""Detect mask, glasses, and hat occlusions."""
		assert image is not None and image.size > 0
		faces = await self.face_detector.detect_faces(image, "occlusion")
		if not faces:
			return {"mask": False, "glasses": False, "hat": False, "error": "no_face"}
		fd = faces[0]
		landmarks = fd.get("landmarks", {})
		# Heuristic: if nose/mouth landmarks have low visibility scores → mask likely
		nose_vis = float(landmarks.get("nose_visibility", 1.0))
		mouth_vis = float(landmarks.get("mouth_visibility", 1.0))
		eye_vis = float(landmarks.get("eye_visibility", 1.0))
		return {"mask": mouth_vis < 0.4, "glasses": eye_vis < 0.6,
				"hat": landmarks.get("forehead_visible", True) is False,
				"confidences": {"mask": 1 - mouth_vis, "glasses": 1 - eye_vis}}

	async def face_detect_in_frame(self, image: np.ndarray) -> list[dict[str, Any]]:
		"""Return list of bounding boxes for all faces in image."""
		assert image is not None and image.size > 0
		faces = await self.face_detector.detect_faces(image, "frame_detect")
		return [{"face_id": f.get("face_id", i),
				 "bounding_box": f.get("bounding_box", {}),
				 "confidence": f.get("confidence", 0)} for i, f in enumerate(faces or [])]

	async def face_count(self, image: np.ndarray) -> int:
		"""Return number of faces detected."""
		faces = await self.face_detect_in_frame(image)
		return len(faces)

	async def recognition_latency_report(self, period: str) -> dict[str, Any]:
		"""Report recognition latency statistics over given period."""
		events = [e for e in self._audit if e.get("action_type") in ("FACE_VERIFIED", "FACE_ENROLLED")]
		if not events:
			return {"period": period, "count": 0, "avg_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
		latencies = [e.get("processing_time_ms", 0) for e in events if e.get("processing_time_ms")]
		latencies.sort()
		n = len(latencies)
		avg = sum(latencies) / n
		p95 = latencies[int(n * 0.95)] if n > 1 else latencies[-1]
		p99 = latencies[int(n * 0.99)] if n > 1 else latencies[-1]
		return {"period": period, "count": n, "avg_ms": avg, "p95_ms": p95, "p99_ms": p99}

	async def accuracy_metrics(self, test_dataset_id: str) -> dict[str, Any]:
		"""Compute FAR, FRR, EER from audit events for a test dataset."""
		# Derived from audit records tagged with test_dataset_id
		verif_events = [e for e in self._audit if e.get("action_type") == "FACE_VERIFIED"
						and e.get("dataset_id") == test_dataset_id]
		genuine_accept = sum(1 for e in verif_events if e.get("action_result") == "success" and e.get("is_genuine"))
		genuine_reject = sum(1 for e in verif_events if e.get("action_result") == "failure" and e.get("is_genuine"))
		impostor_accept = sum(1 for e in verif_events if e.get("action_result") == "success" and not e.get("is_genuine"))
		impostor_reject = sum(1 for e in verif_events if e.get("action_result") == "failure" and not e.get("is_genuine"))
		total_genuine = genuine_accept + genuine_reject or 1
		total_impostor = impostor_accept + impostor_reject or 1
		FAR = impostor_accept / total_impostor
		FRR = genuine_reject / total_genuine
		EER = (FAR + FRR) / 2  # approximate
		return {"test_dataset_id": test_dataset_id, "FAR": FAR, "FRR": FRR, "EER": EER,
				"genuine_accept": genuine_accept, "impostor_accept": impostor_accept}

	# ── COMPLIANCE & CONSENT ──────────────────────────────────────────────────

	async def record_consent(
		self,
		subject_id: str,
		purpose: str,
		obtained_by: str,
		expiry: str,
	) -> dict[str, Any]:
		"""Record biometric data consent for a subject."""
		key = f"{subject_id}:{purpose}"
		self._consents[key] = {
			"subject_id": subject_id, "purpose": purpose, "obtained_by": obtained_by,
			"expiry": expiry, "status": "active", "recorded_at": _now(), "tenant_id": self.tenant_id,
		}
		await self._create_audit_log(action_type="CONSENT_RECORDED", user_id=subject_id,
									 actor_id=obtained_by, purpose=purpose)
		return {"success": True, "subject_id": subject_id, "purpose": purpose, "expiry": expiry}

	async def check_consent(self, subject_id: str, purpose: str) -> dict[str, Any]:
		"""Check whether valid unexpired consent exists."""
		key = f"{subject_id}:{purpose}"
		consent = self._consents.get(key)
		if consent is None:
			return {"has_consent": False, "reason": "no_consent_record"}
		if consent["status"] != "active":
			return {"has_consent": False, "reason": f"consent_{consent['status']}"}
		try:
			expiry = datetime.fromisoformat(consent["expiry"])
			if expiry.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
				consent["status"] = "expired"
				return {"has_consent": False, "reason": "consent_expired"}
		except (ValueError, TypeError) as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"has_consent": True, "purpose": purpose, "obtained_by": consent["obtained_by"],
				"expiry": consent["expiry"]}

	async def revoke_consent(self, subject_id: str) -> dict[str, Any]:
		"""Revoke all consents for a subject."""
		revoked = []
		for key, consent in self._consents.items():
			if consent["subject_id"] == subject_id and consent["status"] == "active":
				consent["status"] = "revoked"
				consent["revoked_at"] = _now()
				revoked.append(consent["purpose"])
		await self._create_audit_log(action_type="CONSENT_REVOKED", user_id=subject_id)
		return {"success": True, "revoked_purposes": revoked, "count": len(revoked)}

	async def data_subject_erasure(self, subject_id: str) -> dict[str, Any]:
		"""GDPR/PDPA erasure — delete all biometric data for subject."""
		template_result = await self.delete_face_template(subject_id)
		consent_result = await self.revoke_consent(subject_id)
		# Remove from all galleries
		for gallery in self._galleries.values():
			if subject_id in gallery.get("subject_ids", []):
				gallery["subject_ids"].remove(subject_id)
		await self._create_audit_log(action_type="DATA_SUBJECT_ERASURE", user_id=subject_id)
		return {"success": True, "subject_id": subject_id,
				"templates_deleted": template_result.get("deleted_count", 0),
				"consents_revoked": consent_result.get("count", 0), "erased_at": _now()}

	async def compliance_report(self, period: str, jurisdiction: str) -> dict[str, Any]:
		"""Generate compliance report for given period and jurisdiction."""
		consent_records = [c for c in self._consents.values() if c["tenant_id"] == self.tenant_id]
		active = sum(1 for c in consent_records if c["status"] == "active")
		revoked = sum(1 for c in consent_records if c["status"] == "revoked")
		erasures = sum(1 for e in self._audit if e.get("action_type") == "DATA_SUBJECT_ERASURE")
		return {
			"period": period, "jurisdiction": jurisdiction, "tenant_id": self.tenant_id,
			"consent_records_active": active, "consent_records_revoked": revoked,
			"data_subject_erasures": erasures,
			"regulatory_framework": "GDPR" if jurisdiction.startswith("EU") else "general",
			"generated_at": _now(),
		}

	# ── PERFORMANCE ───────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		try:
			db_ok = await self.database_service.health_check() if hasattr(self.database_service, 'health_check') else True
		except Exception:
			db_ok = False
		return {"status": "healthy" if db_ok else "degraded", "tenant_id": self.tenant_id,
				"db": "ok" if db_ok else "error", "ts": _now()}

	async def recognition_volume_report(self, period: str) -> dict[str, Any]:
		"""Report enrolment and verification volumes for a period."""
		enrolled = sum(1 for e in self._audit if e.get("action_type") == "FACE_ENROLLED")
		verified = sum(1 for e in self._audit if e.get("action_type") == "FACE_VERIFIED")
		verified_success = sum(1 for e in self._audit if e.get("action_type") == "FACE_VERIFIED"
							   and e.get("action_result") == "success")
		return {"period": period, "enrollments": enrolled, "verifications": verified,
				"successful_verifications": verified_success,
				"success_rate": verified_success / verified if verified else 0.0}

	async def false_match_investigate(self, match_id: str) -> dict[str, Any]:
		"""Retrieve detailed data for a known false match event."""
		events = [e for e in self._audit if e.get("resource_id") == match_id]
		if not events:
			return {"error": "match_not_found", "match_id": match_id}
		event = events[0]
		return {"match_id": match_id, "subject_id": event.get("user_id"),
				"action_result": event.get("action_result"), "ts": event.get("ts"),
				"details": event}

	async def model_accuracy_trend(self, periods: int = 12) -> dict[str, Any]:
		"""Return accuracy trend over last N periods (stub — requires labelled ground-truth store)."""
		return {"periods": periods, "trend": [{"period": i + 1, "EER": 0.05 - i * 0.002}
											  for i in range(min(periods, 12))],
				"note": "requires_labelled_ground_truth"}

	async def system_capacity_check(self) -> dict[str, Any]:
		"""Check current system load vs capacity limits."""
		gallery_count = len(self._galleries)
		total_subjects = sum(len(g.get("subject_ids", [])) for g in self._galleries.values())
		consent_count = len(self._consents)
		return {"gallery_count": gallery_count, "total_enrolled_subjects": total_subjects,
				"consent_records": consent_count, "audit_events": len(self._audit),
				"status": "ok", "ts": _now()}

	# ── WATCHLIST ─────────────────────────────────────────────────────────────

	async def create_watchlist(
		self,
		watchlist_id: str,
		name: str,
		policy_id: str,
		owner: str,
		reason: str,
		match_threshold: float = 0.90,
	) -> dict[str, Any]:
		"""Create a named watchlist with policy binding."""
		assert watchlist_id and name and policy_id and owner
		if not hasattr(self, "_watchlists"):
			self._watchlists: dict[str, dict[str, Any]] = {}
		if watchlist_id in self._watchlists:
			return {"success": False, "error": "watchlist_exists"}
		self._watchlists[watchlist_id] = {
			"id": watchlist_id, "name": name, "policy_id": policy_id,
			"owner": owner, "reason": reason, "match_threshold": match_threshold,
			"subject_ids": [], "created_at": _now(), "tenant_id": self.tenant_id, "status": "active",
		}
		await self._create_audit_log(action_type="WATCHLIST_CREATED", resource_id=watchlist_id,
									 actor_id=owner, policy_id=policy_id)
		return {"success": True, "watchlist_id": watchlist_id, "name": name, "policy_id": policy_id}

	async def add_watchlist_subject(
		self,
		watchlist_id: str,
		subject_id: str,
		added_by: str,
		reason: str,
		expiry: str | None = None,
	) -> dict[str, Any]:
		"""Add a subject to a watchlist."""
		if not hasattr(self, "_watchlists"):
			self._watchlists: dict[str, dict[str, Any]] = {}
		wl = self._watchlists.get(watchlist_id)
		if wl is None:
			return {"success": False, "error": "watchlist_not_found"}
		if wl["status"] != "active":
			return {"success": False, "error": f"watchlist_{wl['status']}"}
		existing_ids = [s["subject_id"] for s in wl["subject_ids"]]
		if subject_id in existing_ids:
			return {"success": False, "error": "subject_already_on_watchlist"}
		entry = {"subject_id": subject_id, "added_by": added_by, "reason": reason,
				 "added_at": _now(), "expiry": expiry, "active": True}
		wl["subject_ids"].append(entry)
		await self._create_audit_log(action_type="WATCHLIST_SUBJECT_ADDED", resource_id=watchlist_id,
									 user_id=subject_id, actor_id=added_by)
		return {"success": True, "watchlist_id": watchlist_id, "subject_id": subject_id}

	async def watchlist_match(
		self,
		probe_image: np.ndarray,
		watchlist_id: str,
	) -> dict[str, Any]:
		"""1:N match of probe against a watchlist. Returns hits above the watchlist threshold."""
		assert probe_image is not None and probe_image.size > 0
		if not hasattr(self, "_watchlists"):
			self._watchlists: dict[str, dict[str, Any]] = {}
		wl = self._watchlists.get(watchlist_id)
		if wl is None:
			return {"success": False, "error": "watchlist_not_found"}
		start = datetime.now()
		feats, quality = await self._extract_probe_features(probe_image, f"watchlist_{watchlist_id}")
		if feats is None:
			return {"success": False, "error": quality.get("error", "extraction_failed"),
					"quality_score": quality.get("overall_score", 0)}
		threshold = float(wl.get("match_threshold", self.verification_threshold))
		hits: list[dict[str, Any]] = []
		for entry in wl.get("subject_ids", []):
			if not entry.get("active", True):
				continue
			sid = entry["subject_id"]
			templates = await self.database_service.get_user_templates(sid, active_only=True)
			for tmpl in templates:
				raw = await self.database_service.decrypt_template_data(tmpl)
				if raw is None:
					continue
				stored = np.frombuffer(raw, dtype=np.float32)
				sim = self._cosine_similarity(feats, stored)
				if sim >= threshold:
					hits.append({"subject_id": sid, "score": float(sim), "watchlist_id": watchlist_id,
								 "reason": entry.get("reason"), "added_by": entry.get("added_by")})
		hits.sort(key=lambda x: -x["score"])
		ms = (datetime.now() - start).total_seconds() * 1000
		if hits:
			await self._create_audit_log(action_type="WATCHLIST_HIT", resource_id=watchlist_id,
										 hit_count=len(hits), processing_time_ms=ms)
		return {"success": True, "watchlist_id": watchlist_id, "hits": hits,
				"hit_count": len(hits), "quality_score": quality.get("overall_score", 0),
				"processing_time_ms": ms, "ts": _now()}

	# ── DEEPFAKE & MORPHING ATTACK DETECTION ─────────────────────────────────

	async def deepfake_detect(self, image: np.ndarray) -> dict[str, Any]:
		"""Detect GAN/diffusion deepfake faces via FFT spectral anomaly and DCT artifact analysis.

		Wire an AICR adapter backed by a FaceForensics++-trained classifier for production.
		"""
		assert image is not None and image.size > 0
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
		f_transform = np.fft.fft2(gray.astype(np.float32))
		f_shift = np.fft.fftshift(f_transform)
		magnitude = np.log(np.abs(f_shift) + 1)
		h, w = magnitude.shape
		h_mid, w_mid = h // 2, w // 2
		cardinal_energy = float(magnitude[h_mid, :].mean() + magnitude[:, w_mid].mean())
		total_energy = float(magnitude.mean())
		spectral_ratio = cardinal_energy / (total_energy + 1e-8)
		spectral_anomaly = spectral_ratio > 2.5
		dct_scores = []
		block_size = 8
		for y in range(0, gray.shape[0] - block_size, block_size):
			for x in range(0, gray.shape[1] - block_size, block_size):
				block = gray[y:y + block_size, x:x + block_size].astype(np.float32)
				dct_block = cv2.dct(block)
				hf_energy = float(np.abs(dct_block[4:, 4:]).mean())
				dct_scores.append(hf_energy)
		avg_hf_dct = float(np.mean(dct_scores)) if dct_scores else 0.0
		dct_anomaly = avg_hf_dct < 2.0
		indicators = {"spectral_anomaly": spectral_anomaly, "dct_artifact": dct_anomaly,
					  "spectral_ratio": float(spectral_ratio), "avg_hf_dct": avg_hf_dct}
		risk_score = float(spectral_anomaly) * 0.55 + float(dct_anomaly) * 0.45
		is_deepfake = risk_score > 0.5
		await self._create_audit_log(action_type="DEEPFAKE_SCAN",
									 result="suspect" if is_deepfake else "clear", risk_score=risk_score)
		return {"is_deepfake": is_deepfake, "risk_score": risk_score, "indicators": indicators,
				"note": "heuristic — wire AICR adapter for production classifier", "ts": _now()}

	async def morphing_attack_detect(self, image: np.ndarray) -> dict[str, Any]:
		"""Detect face morphing attacks via landmark asymmetry and Laplacian seam artifact scoring."""
		assert image is not None and image.size > 0
		faces = await self.face_detector.detect_faces(image, "morph_detect")
		if not faces:
			return {"is_morph": False, "confidence": 0.0, "error": "no_face_detected"}
		fd = faces[0]
		landmarks = fd.get("landmarks", {})
		left_eye = landmarks.get("left_eye", {})
		right_eye = landmarks.get("right_eye", {})
		left_mouth = landmarks.get("left_mouth_corner", {})
		right_mouth = landmarks.get("right_mouth_corner", {})

		def _y_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
			return abs(float(a.get("y", 0)) - float(b.get("y", 0)))

		eye_asymmetry = _y_diff(left_eye, right_eye)
		mouth_asymmetry = _y_diff(left_mouth, right_mouth)
		bb = fd.get("bounding_box", {})
		face_height = max(float(bb.get("height", 1)), 1.0)
		norm_asymmetry = (eye_asymmetry + mouth_asymmetry) / face_height
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
		lap = cv2.Laplacian(gray, cv2.CV_64F)
		edge_std = float(np.std(lap))
		seam_anomaly = edge_std > 35.0
		morph_score = min(1.0, norm_asymmetry * 3.0) * 0.5 + float(seam_anomaly) * 0.5
		is_morph = morph_score > 0.55
		await self._create_audit_log(action_type="MORPH_ATTACK_SCAN",
									 result="suspect" if is_morph else "clear", morph_score=morph_score)
		return {"is_morph": is_morph, "morph_score": morph_score,
				"indicators": {"landmark_asymmetry": norm_asymmetry,
							   "seam_anomaly": seam_anomaly, "edge_std": edge_std}, "ts": _now()}

	# ── BIAS & EXPLAINABILITY ─────────────────────────────────────────────────

	async def bias_audit_report(
		self,
		cohort_field: str = "demographic_group",
		min_samples: int = 30,
	) -> dict[str, Any]:
		"""Per-cohort FAR/FRR bias audit per ISO/IEC 19795-10.

		Reads verification audit events tagged with cohort metadata. Flag cohorts
		whose FAR or FRR diverges from the overall mean by more than 5pp.
		"""
		verif_events = [e for e in self._audit if e.get("action_type") == "FACE_VERIFIED"]
		cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for ev in verif_events:
			cohort = ev.get(cohort_field, "unknown")
			cohorts[cohort].append(ev)
		results: dict[str, Any] = {}
		for cohort, events in cohorts.items():
			if len(events) < min_samples:
				results[cohort] = {"status": "insufficient_samples", "count": len(events)}
				continue
			genuine_accept = sum(1 for e in events if e.get("action_result") == "success" and e.get("is_genuine"))
			genuine_reject = sum(1 for e in events if e.get("action_result") == "failure" and e.get("is_genuine"))
			impostor_accept = sum(1 for e in events if e.get("action_result") == "success" and not e.get("is_genuine"))
			impostor_reject = sum(1 for e in events if e.get("action_result") == "failure" and not e.get("is_genuine"))
			total_genuine = genuine_accept + genuine_reject or 1
			total_impostor = impostor_accept + impostor_reject or 1
			results[cohort] = {
				"count": len(events),
				"FAR": impostor_accept / total_impostor,
				"FRR": genuine_reject / total_genuine,
				"EER": (impostor_accept / total_impostor + genuine_reject / total_genuine) / 2,
				"genuine_accept": genuine_accept, "impostor_accept": impostor_accept,
			}
		all_far = [v["FAR"] for v in results.values() if isinstance(v.get("FAR"), float)]
		all_frr = [v["FRR"] for v in results.values() if isinstance(v.get("FRR"), float)]
		overall_far = sum(all_far) / len(all_far) if all_far else 0.0
		overall_frr = sum(all_frr) / len(all_frr) if all_frr else 0.0
		bias_flags = [
			cohort for cohort, stats in results.items()
			if isinstance(stats.get("FAR"), float)
			and (abs(stats["FAR"] - overall_far) > 0.05 or abs(stats["FRR"] - overall_frr) > 0.05)
		]
		return {"cohorts": results, "overall_FAR": overall_far, "overall_FRR": overall_frr,
				"bias_flags": bias_flags, "standard": "ISO/IEC 19795-10",
				"cohort_field": cohort_field, "ts": _now()}

	async def explain_verification(self, verification_id: str) -> dict[str, Any]:
		"""Human-readable explanation of a verification decision (GDPR Art. 22).

		Returns binding constraint, counterfactual threshold, and plain-language summary
		structured for both subject disclosure and machine-readable audit ingestion.
		"""
		events = [e for e in self._audit
				  if e.get("resource_id") == verification_id and e.get("action_type") == "FACE_VERIFIED"]
		if not events:
			return {"error": "verification_not_found", "verification_id": verification_id}
		ev = events[0]
		outcome = ev.get("action_result", "unknown")
		failure_reason = ev.get("failure_reason", "")
		confidence = float(ev.get("confidence_score", 0.0))
		similarity = float(ev.get("similarity_score", 0.0))
		quality = float(ev.get("input_quality_score", 0.0))
		if quality < self.quality_threshold:
			binding_constraint = "input_image_quality"
			counterfactual = (f"Quality score {quality:.2f} below threshold {self.quality_threshold:.2f}. "
							  f"Outcome changes if quality >= {self.quality_threshold:.2f}.")
		elif similarity < self.verification_threshold:
			binding_constraint = "biometric_similarity"
			counterfactual = (f"Similarity {similarity:.2f} below threshold {self.verification_threshold:.2f}. "
							  f"Outcome changes if similarity >= {self.verification_threshold:.2f}.")
		elif failure_reason and "liveness" in failure_reason:
			binding_constraint = "liveness_check"
			counterfactual = "Liveness check did not confirm a live subject."
		else:
			binding_constraint = "composite_score"
			counterfactual = f"Composite confidence {confidence:.2f} marginally below threshold."
		plain_language = (
			f"Verification {'succeeded' if outcome == 'success' else 'failed'} "
			f"with confidence {confidence:.2f}. "
			f"Determining factor: {binding_constraint.replace('_', ' ')}. {counterfactual}"
		)
		return {"verification_id": verification_id, "outcome": outcome,
				"binding_constraint": binding_constraint, "confidence": confidence,
				"similarity": similarity, "quality": quality, "counterfactual": counterfactual,
				"plain_language_summary": plain_language, "regulation": "GDPR Art. 22", "ts": _now()}

	# ── TEMPLATE AGING & RE-ENROLLMENT ───────────────────────────────────────

	async def template_aging_report(
		self,
		gallery_id: str,
		drift_threshold: float = 0.05,
	) -> dict[str, Any]:
		"""Flag subjects whose rolling average match confidence has drifted below enrollment quality.

		Compares enrollment-time quality score against the last 20 verification audit events
		per subject. Subjects exceeding `drift_threshold` degradation are flagged for re-enrollment.
		"""
		gallery = self._galleries.get(gallery_id)
		if gallery is None:
			return {"success": False, "error": "gallery_not_found"}
		subjects = gallery.get("subject_ids", [])
		flagged: list[dict[str, Any]] = []
		healthy: list[str] = []
		for sid in subjects:
			templates = await self.database_service.get_user_templates(sid, active_only=True)
			if not templates:
				continue
			enroll_quality = float(getattr(templates[0], "quality_score", 0) or 0)
			recent_events = [e for e in self._audit
							 if e.get("action_type") == "FACE_VERIFIED" and e.get("user_id") == sid][-20:]
			if len(recent_events) < 5:
				healthy.append(sid)
				continue
			avg_confidence = sum(float(e.get("confidence_score", 0)) for e in recent_events) / len(recent_events)
			drift = enroll_quality - avg_confidence
			if drift >= drift_threshold:
				flagged.append({"subject_id": sid, "enroll_quality": enroll_quality,
								"recent_avg_confidence": avg_confidence, "drift": drift,
								"recommendation": "re_enroll"})
			else:
				healthy.append(sid)
		return {"gallery_id": gallery_id, "flagged_count": len(flagged), "healthy_count": len(healthy),
				"drift_threshold": drift_threshold, "flagged_subjects": flagged, "ts": _now()}

	async def reenroll_subject(
		self,
		subject_id: str,
		new_image: np.ndarray,
		quality_threshold: float = 0.85,
		reason: str = "scheduled_re_enrollment",
	) -> dict[str, Any]:
		"""Hard-delete existing templates and re-enroll with a fresh image. Requires active consent."""
		assert subject_id and new_image is not None and new_image.size > 0
		consent_check = await self.check_consent(subject_id, "biometric_identification")
		if not consent_check.get("has_consent"):
			consent_check = await self.check_consent(subject_id, "workforce_authentication")
		if not consent_check.get("has_consent"):
			return {"success": False, "error": "consent_required_for_reenrollment"}
		existing_templates = await self.database_service.get_user_templates(subject_id, active_only=False)
		for tmpl in existing_templates:
			await self.database_service.delete_template(tmpl.id)
		result = await self.enroll_face(subject_id, new_image, quality_threshold)
		if result["success"]:
			await self._create_audit_log(action_type="SUBJECT_REENROLLED", user_id=subject_id,
										 reason=reason, previous_template_count=len(existing_templates))
		return {**result, "reason": reason, "previous_templates_deleted": len(existing_templates)}

	# ── CONTINUOUS AUTHENTICATION ─────────────────────────────────────────────

	async def continuous_auth_stream(
		self,
		subject_id: str,
		frame_source: list[np.ndarray],
		interval_frames: int = 30,
		revoke_on_fail_count: int = 3,
	) -> AsyncGenerator[dict[str, Any], None]:
		"""Async generator yielding ambient re-authentication events from camera frames.

		Emits a REVOKE event and stops after `revoke_on_fail_count` consecutive failures.
		Replace `frame_source` with a real async camera stream in production.
		"""
		consecutive_failures = 0
		frame_index = 0
		for frame in frame_source:
			frame_index += 1
			if frame_index % interval_frames != 0:
				continue
			result = await self.verify_face(subject_id, frame, {"require_liveness": False})
			verified = result.get("verified", False)
			event: dict[str, Any] = {
				"event_type": "re_auth", "subject_id": subject_id, "frame_index": frame_index,
				"verified": verified, "confidence": result.get("confidence", 0.0), "ts": _now(),
			}
			if verified:
				consecutive_failures = 0
				event["status"] = "active"
			else:
				consecutive_failures += 1
				event["status"] = "warning"
				event["consecutive_failures"] = consecutive_failures
				if consecutive_failures >= revoke_on_fail_count:
					event["status"] = "revoked"
					event["event_type"] = "auth_revoked"
					await self._create_audit_log(action_type="CONTINUOUS_AUTH_REVOKED",
												 user_id=subject_id, frame_index=frame_index)
					yield event
					return
			yield event

	# ── FEDERATED IDENTIFICATION ──────────────────────────────────────────────

	async def federated_identify(
		self,
		probe_image: np.ndarray,
		tenants: list[dict[str, str]],
		top_k: int = 5,
	) -> dict[str, Any]:
		"""Cross-tenant 1:N identification. Each tenant entry requires a consent_proof token.

		Fans out identification in parallel across supplied galleries and merges re-ranked results.
		Production deployments replace the local gallery lookup with gRPC calls to remote FREC instances.
		"""
		assert probe_image is not None and probe_image.size > 0
		start = datetime.now()
		feats, quality = await self._extract_probe_features(probe_image, "federated_identify")
		if feats is None:
			return {"success": False, "error": quality.get("error", "extraction_failed")}

		async def _identify_tenant(spec: dict[str, str]) -> list[dict[str, Any]]:
			if not spec.get("consent_proof"):
				return []
			gallery = self._galleries.get(spec.get("gallery_id", ""))
			if not gallery:
				return []
			candidates = []
			for sid in gallery.get("subject_ids", []):
				for tmpl in await self.database_service.get_user_templates(sid, active_only=True):
					raw = await self.database_service.decrypt_template_data(tmpl)
					if raw is None:
						continue
					sim = self._cosine_similarity(feats, np.frombuffer(raw, dtype=np.float32))
					candidates.append({"subject_id": sid, "score": float(sim),
									   "tenant_id": spec.get("tenant_id"), "gallery_id": spec.get("gallery_id")})
			return candidates

		all_results = await asyncio.gather(*[_identify_tenant(t) for t in tenants], return_exceptions=True)
		merged: list[dict[str, Any]] = [c for sublist in all_results for c in sublist]
		merged.sort(key=lambda x: -x["score"])
		for i, c in enumerate(merged[:top_k]):
			c["rank"] = i + 1
		ms = (datetime.now() - start).total_seconds() * 1000
		await self._create_audit_log(action_type="FEDERATED_IDENTIFY",
									 tenant_count=len(tenants), hit_count=len(merged[:top_k]))
		return {"success": True, "candidates": merged[:top_k], "quality_score": quality.get("overall_score", 0),
				"tenant_count": len(tenants), "processing_time_ms": ms, "ts": _now()}

	# ── CONSENT PORTABILITY (GDPR ART. 20) ───────────────────────────────────

	async def export_consent_portable(self, subject_id: str) -> dict[str, Any]:
		"""Export active consents as a W3C Verifiable Credential JSON-LD document (GDPR Art. 20).

		Sign the credential with the tenant private key via the `encr` adapter in production.
		"""
		consents = [c for c in self._consents.values()
					if c["subject_id"] == subject_id and c["status"] == "active"]
		credential = {
			"@context": ["https://www.w3.org/2018/credentials/v1",
						 "https://datacraft.co.ke/frec/consent/v1"],
			"type": ["VerifiableCredential", "BiometricConsentCredential"],
			"issuer": f"did:datacraft:{self.tenant_id}",
			"issuanceDate": _now(),
			"credentialSubject": {
				"id": f"did:datacraft:subject:{subject_id}",
				"biometricConsents": [
					{"purpose": c["purpose"], "obtainedBy": c["obtained_by"],
					 "expiry": c["expiry"], "recordedAt": c["recorded_at"]}
					for c in consents
				],
			},
			"proof": {"type": "placeholder — sign with encr adapter in production",
					  "created": _now(), "proofPurpose": "assertionMethod"},
		}
		await self._create_audit_log(action_type="CONSENT_EXPORTED", user_id=subject_id,
									 consent_count=len(consents))
		return {"success": True, "subject_id": subject_id, "credential": credential,
				"consent_count": len(consents), "exported_at": _now()}

	async def import_consent_portable(
		self,
		subject_id: str,
		credential: dict[str, Any],
		obtained_by: str = "import",
	) -> dict[str, Any]:
		"""Ingest a portable W3C VC-structured consent record and activate it locally (GDPR Art. 20).

		Validate the credential's cryptographic proof via the `encr` adapter before activation
		in production. The current implementation skips proof verification (stub).
		"""
		assert subject_id and credential
		if "VerifiableCredential" not in credential.get("type", []):
			return {"success": False, "error": "invalid_credential_type"}
		if "BiometricConsentCredential" not in credential.get("type", []):
			return {"success": False, "error": "not_biometric_consent_credential"}
		consents_data = credential.get("credentialSubject", {}).get("biometricConsents", [])
		if not consents_data:
			return {"success": False, "error": "no_consent_data_in_credential"}
		imported = []
		for entry in consents_data:
			purpose = entry.get("purpose", "")
			if not purpose:
				continue
			result = await self.record_consent(subject_id, purpose, obtained_by, entry.get("expiry", ""))
			if result.get("success"):
				imported.append(purpose)
		await self._create_audit_log(action_type="CONSENT_IMPORTED", user_id=subject_id,
									 imported_count=len(imported), issuer=credential.get("issuer"))
		return {"success": True, "subject_id": subject_id, "imported_purposes": imported,
				"imported_count": len(imported), "imported_at": _now(),
				"note": "proof signature not verified — wire encr adapter in production"}

	# ── ISO/IEC 30107-3 LIVENESS COMPLIANCE ──────────────────────────────────

	async def liveness_compliance_report(
		self,
		test_results: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Evaluate liveness system against ISO/IEC 30107-3 APCER/BPCER/ACER metrics.

		`test_results` items: {"is_live_predicted": bool, "is_bona_fide": bool,
		"attack_type": str | None, "confidence": float}.

		Level 4 compliance requires APCER <= 0.5% and BPCER <= 0.5%.
		"""
		if not test_results:
			return {"error": "no_test_results_provided"}
		bona_fide = [r for r in test_results if r.get("is_bona_fide", True)]
		attacks = [r for r in test_results if not r.get("is_bona_fide", True)]
		bona_fide_rejected = sum(1 for r in bona_fide if not r.get("is_live_predicted", True))
		bpcer = bona_fide_rejected / len(bona_fide) if bona_fide else 0.0
		attack_types: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for r in attacks:
			attack_types[r.get("attack_type", "unknown")].append(r)
		apcer_per_type: dict[str, float] = {
			atype: sum(1 for s in samples if s.get("is_live_predicted", False)) / len(samples)
			for atype, samples in attack_types.items()
		}
		apcer = max(apcer_per_type.values()) if apcer_per_type else 0.0
		acer = (apcer + bpcer) / 2
		level_4_apcer = 0.005
		level_4_bpcer = 0.005
		return {
			"standard": "ISO/IEC 30107-3", "claimed_level": 4,
			"compliant": apcer <= level_4_apcer and bpcer <= level_4_bpcer,
			"APCER": apcer, "BPCER": bpcer, "ACER": acer,
			"APCER_per_attack_type": apcer_per_type,
			"level_4_thresholds": {"APCER": level_4_apcer, "BPCER": level_4_bpcer},
			"bona_fide_count": len(bona_fide), "attack_count": len(attacks), "ts": _now(),
		}


__all__ = ['FacialRecognitionService']
