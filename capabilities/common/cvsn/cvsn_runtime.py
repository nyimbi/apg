"""Dependency-light generated-app runtime for APG Computer Vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


@dataclass
class VisionAsset:
	id: str
	tenant_id: str
	asset_kind: str
	mime_type: str
	file_size_mb: float
	source_ref: str
	content_hash: str
	status: str = "ingested"
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"asset_kind": self.asset_kind,
			"mime_type": self.mime_type,
			"file_size_mb": self.file_size_mb,
			"source_ref": self.source_ref,
			"content_hash": self.content_hash,
			"status": self.status,
			"metadata": dict(self.metadata),
		}


@dataclass
class VisionJob:
	id: str
	tenant_id: str
	asset_id: str
	processing_type: str
	operator: str
	confidence_score: float
	results: dict[str, Any]
	status: str = "completed"
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"asset_id": self.asset_id,
			"processing_type": self.processing_type,
			"operator": self.operator,
			"confidence_score": self.confidence_score,
			"results": dict(self.results),
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
		}


@dataclass
class VisionModelRegistration:
	id: str
	tenant_id: str
	name: str
	model_type: str
	mlcm_model_ref: str
	owner: str
	version: str
	model_card_ref: str
	evaluated: bool = False
	approved: bool = False
	status: str = "registered"
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"model_type": self.model_type,
			"mlcm_model_ref": self.mlcm_model_ref,
			"owner": self.owner,
			"version": self.version,
			"model_card_ref": self.model_card_ref,
			"evaluated": self.evaluated,
			"approved": self.approved,
			"status": self.status,
			"metadata": dict(self.metadata),
		}


@dataclass
class VisionPipeline:
	id: str
	tenant_id: str
	name: str
	owner: str
	model_ref: str
	version: str
	tasks: list[str]
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"model_ref": self.model_ref,
			"version": self.version,
			"tasks": list(self.tasks),
			"status": self.status,
		}


@dataclass
class VisionAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"agent_id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"owner": self.owner,
			"purpose": self.purpose,
			"contribution_disclosed": self.contribution_disclosed,
			"human_approval_required": self.human_approval_required,
			"status": self.status,
		}


@dataclass
class CvsnLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	required_processor: str = "bytewax"
	status: str = "accepted"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"batch_id": self.id,
			"tenant_id": self.tenant_id,
			"event_stream": self.event_stream,
			"mutation_count": self.mutation_count,
			"operation": self.operation,
			"accepted": self.accepted,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"required_processor": self.required_processor,
			"status": self.status,
		}


@dataclass
class VisionAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	decision: str
	reasons: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"decision": self.decision,
			"reasons": list(self.reasons),
		}


class CvsnService:
	"""In-process computer-vision lifecycle service for generated applications."""

	def __init__(self) -> None:
		contract = get_capability_contract()
		self._assets: dict[str, VisionAsset] = {}
		self._jobs: dict[str, VisionJob] = {}
		self._models: dict[str, VisionModelRegistration] = {}
		self._pipelines: dict[str, VisionPipeline] = {}
		self._agents: dict[str, VisionAgentRecord] = {}
		self._lifecycle_batches: dict[str, CvsnLifecycleBatchRecord] = {}
		self._audit_events: dict[str, VisionAuditEvent] = {}
		self._config = contract["configuration"]
		self._enabled_tasks = set(self._config["vision_tasks"]["enabled"])
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def ingest_asset(
		self,
		asset_id: str,
		tenant_id: str,
		asset_kind: str,
		mime_type: str,
		file_size_mb: float,
		source_ref: str,
		content_hash: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		content_hash = content_hash or self._digest(f"{tenant_id}:{asset_id}:{source_ref}:{mime_type}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "ingest_asset",
			"source_ref_present": bool(source_ref),
			"mime_type_supported": self._mime_type_supported(asset_kind, mime_type),
			"file_size_mb": file_size_mb,
			"asset_hash_present": bool(content_hash),
		})
		self._raise_if_blocked(result)
		asset = VisionAsset(
			id=asset_id,
			tenant_id=tenant_id,
			asset_kind=asset_kind,
			mime_type=mime_type,
			file_size_mb=file_size_mb,
			source_ref=source_ref,
			content_hash=content_hash,
			metadata=dict(metadata or {}),
		)
		self._assets[asset_id] = asset
		self._audit(tenant_id, "asset_ingested", asset_id, result)
		return asset.to_dict()

	def run_job(
		self,
		job_id: str,
		tenant_id: str,
		asset_id: str,
		processing_type: str,
		operator: str,
		batch_size: int = 1,
		async_queue_enabled: bool = True,
		consent_recorded: bool = False,
		anonymization_enabled: bool = True,
		retention_days: int = 30,
		inspection_plan_attached: bool = True,
		defect_taxonomy_attached: bool = True,
		alerting_enabled: bool = True,
		incident_acknowledged: bool = True,
		sampling_policy_attached: bool = True,
		moderation_policy_attached: bool = True,
		human_review_recorded: bool = True,
		clip_seconds: int = 0,
	) -> dict[str, Any]:
		asset = self._require_asset(asset_id, tenant_id)
		preflight = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "run_job",
			"task_enabled": processing_type in self._enabled_tasks,
			"operator_present": bool(operator),
			"processing_type": processing_type,
			"asset_kind": asset.asset_kind,
			"inspection_plan_attached": inspection_plan_attached,
			"defect_taxonomy_attached": defect_taxonomy_attached,
			"alerting_enabled": alerting_enabled,
			"incident_acknowledged": incident_acknowledged,
			"consent_recorded": consent_recorded,
			"anonymization_enabled": anonymization_enabled,
			"retention_days": retention_days,
			"batch_size": batch_size,
			"async_queue_enabled": async_queue_enabled,
			"clip_seconds": clip_seconds,
			"sampling_policy_attached": sampling_policy_attached,
			"moderation_policy_attached": moderation_policy_attached,
		})
		self._raise_if_denied(preflight)
		confidence_score, results = self._run_processing(asset, processing_type)
		postflight_context = {
			"tenant_context_present": bool(tenant_id),
			"processing_type": processing_type,
			"alerting_enabled": alerting_enabled,
			"incident_acknowledged": incident_acknowledged,
			"confidence_score": confidence_score,
			"human_review_recorded": human_review_recorded,
		}
		for key in ("critical_defect_detected", "severity"):
			if key in results:
				postflight_context[key] = results[key]
		postflight = self.evaluate(postflight_context)
		combined = self._combine(preflight, postflight)
		self._raise_if_denied(combined)
		job = VisionJob(
			job_id,
			tenant_id,
			asset_id,
			processing_type,
			operator,
			confidence_score,
			results,
			status="pending_review" if combined["decision"] == "require_review" else "completed",
			decision=combined["decision"],
			matched_rules=list(combined["matched_rules"]),
			review_reasons=self._review_reasons(combined),
		)
		self._jobs[job_id] = job
		asset.status = "processed"
		self._audit(tenant_id, "job_completed", job_id, combined)
		return job.to_dict()

	def register_pipeline(
		self,
		pipeline_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		model_ref: str,
		version: str,
		tasks: list[str],
	) -> dict[str, Any]:
		if not tenant_id:
			raise PermissionError("tenant_context_required")
		if not owner:
			raise PermissionError("pipeline_owner_required")
		if not model_ref:
			raise PermissionError("registered_model_required")
		if not version:
			raise PermissionError("pipeline_version_required")
		if not tasks:
			raise PermissionError("vision_task_required")
		for task in tasks:
			if task not in self._enabled_tasks:
				raise PermissionError("vision_task_not_enabled")
		pipeline = VisionPipeline(pipeline_id, tenant_id, name, owner, model_ref, version, list(tasks))
		self._pipelines[pipeline_id] = pipeline
		self._audit(tenant_id, "pipeline_registered", pipeline_id, {"decision": "allow", "actions": []})
		return pipeline.to_dict()

	def register_model(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		model_type: str,
		mlcm_model_ref: str,
		owner: str,
		version: str,
		model_card_ref: str,
		evaluated: bool = False,
		approved: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_model",
			"mlcm_model_ref_present": bool(mlcm_model_ref),
			"model_card_present": bool(model_card_ref),
		})
		self._raise_if_blocked(result)
		if not owner:
			raise PermissionError("model_owner_required")
		if not version:
			raise PermissionError("model_version_required")
		model = VisionModelRegistration(
			id=model_id,
			tenant_id=tenant_id,
			name=name,
			model_type=model_type,
			mlcm_model_ref=mlcm_model_ref,
			owner=owner,
			version=version,
			model_card_ref=model_card_ref,
			evaluated=evaluated,
			approved=approved,
			metadata=dict(metadata or {}),
		)
		self._models[model_id] = model
		self._audit(tenant_id, "model_registered", model_id, result)
		return model.to_dict()

	def release_model(
		self,
		model_id: str,
		tenant_id: str,
		evaluation_recorded: bool = True,
		approval_recorded: bool = True,
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "release_model",
			"evaluation_recorded": evaluation_recorded or model.evaluated,
			"approval_recorded": approval_recorded or model.approved,
		})
		self._raise_if_blocked(result)
		model.evaluated = True
		model.approved = True
		model.status = "released"
		self._audit(tenant_id, "model_released", model_id, result)
		return model.to_dict()

	def register_vision_agent(
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
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_vision_agent",
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
		if not name:
			raise ValueError("vision_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		agent = VisionAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status=status,
		)
		self._agents[self._tenant_record_key(tenant_id, agent.id)] = agent
		self._audit(tenant_id, "vision_agent_registered", agent.id, result)
		return agent.to_dict()

	def validate_cvsn_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "vision_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("cvsn_lifecycle_batch_empty")
		stream_value = self._normalize_token(event_stream)
		operation_value = self._normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_cvsn_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_cvsn_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		batch = CvsnLifecycleBatchRecord(
			id=batch_id or f"cvsnbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_record_key(tenant_id, batch.id)] = batch
		self._audit(tenant_id, f"cvsn_lifecycle_batch_{batch.status}", batch.id, result)
		if not accepted:
			self._raise_if_denied(result)
		return batch.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		asset = self.ingest_asset(
			record_id,
			tenant_id,
			str(metadata.get("asset_kind") or "image"),
			str(metadata.get("mime_type") or "image/png"),
			float(metadata.get("file_size_mb") or 1),
			str(metadata.get("source_ref") or f"generated://{record_id}"),
			str(metadata.get("content_hash") or ""),
			metadata,
		)
		self._assets[record_id].status = status
		asset["status"] = status
		return asset

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_assets(tenant_id)

	def list_assets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._assets, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._jobs, tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._models, tenant_id)

	def list_pipelines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._pipelines, tenant_id)

	def list_vision_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		jobs = self.list_jobs(tenant_id)
		return {
			"tenant_id": tenant_id,
			"asset_count": len(self.list_assets(tenant_id)),
			"job_count": len(jobs),
			"document_job_count": len([job for job in jobs if job["processing_type"] == "ocr"]),
			"image_job_count": len([job for job in jobs if job["processing_type"] in {"object_detection", "image_classification"}]),
			"video_job_count": len([job for job in jobs if job["processing_type"] == "video_analytics"]),
			"quality_job_count": len([job for job in jobs if job["processing_type"] == "quality_inspection"]),
			"safety_job_count": len([job for job in jobs if job["processing_type"] == "factory_safety"]),
			"pending_job_review_count": len([job for job in jobs if job["status"] == "pending_review"]),
			"model_count": len(self.list_models(tenant_id)),
			"released_model_count": len([model for model in self.list_models(tenant_id) if model["status"] == "released"]),
			"pipeline_count": len(self.list_pipelines(tenant_id)),
			"vision_agent_count": len(self.list_vision_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_vision_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _run_processing(self, asset: VisionAsset, processing_type: str) -> tuple[float, dict[str, Any]]:
		if processing_type == "ocr":
			return 0.86, {"text": f"Extracted text from {asset.source_ref}", "word_count": 5, "language": "eng"}
		if processing_type == "object_detection":
			return 0.84, {"objects": [{"label": "asset", "confidence": 0.84}], "object_count": 1}
		if processing_type == "image_classification":
			return 0.82, {"label": "industrial_asset", "confidence": 0.82}
		if processing_type == "quality_inspection":
			return 0.88, {"inspection_result": "pass", "defect_count": 0, "critical_defect_detected": False}
		if processing_type == "factory_safety":
			return 0.90, {"severity": "normal", "alerts": []}
		if processing_type == "video_analytics":
			return 0.80, {"sampled_frames": 12, "events": []}
		if processing_type == "visual_similarity":
			return 0.79, {"nearest_assets": [asset.id], "similarity": 1.0}
		if processing_type == "barcode_qr":
			return 0.87, {"codes": [{"type": "qr", "value": asset.content_hash[:12]}]}
		if processing_type == "facial_analysis":
			return 0.81, {"faces": [], "anonymized": True}
		if processing_type == "content_moderation":
			return 0.83, {"policy": "default", "decision": "allow"}
		return 0.75, {"asset_id": asset.id}

	def _mime_type_supported(self, asset_kind: str, mime_type: str) -> bool:
		processing = self._config["processing"]
		if asset_kind == "document":
			return mime_type in processing["allowed_document_types"]
		if asset_kind == "image":
			return mime_type in processing["allowed_image_types"]
		if asset_kind == "video":
			return mime_type in processing["allowed_video_types"]
		return False

	def _require_asset(self, asset_id: str, tenant_id: str) -> VisionAsset:
		asset = self._assets.get(asset_id)
		if asset is None or asset.tenant_id != tenant_id:
			raise LookupError("asset_not_found")
		return asset

	def _require_model(self, model_id: str, tenant_id: str) -> VisionModelRegistration:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise LookupError("model_not_found")
		return model

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _audit(self, tenant_id: str, event_type: str, subject_id: str, result: dict[str, Any]) -> None:
		event_id = f"cvsnaudit:{self._digest(f'{tenant_id}:{event_type}:{subject_id}')[:12]}"
		self._audit_events[event_id] = VisionAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			decision=result["decision"],
			reasons=[action.get("reason", "") for action in result.get("actions", []) if action.get("reason")],
		)

	def _raise_if_blocked(self, result: dict[str, Any]) -> None:
		if result["decision"] in {"deny", "require_review"}:
			raise PermissionError(self._reasons(result))

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(self._reasons(result))

	def _combine(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
		actions = list(left.get("actions", [])) + list(right.get("actions", []))
		matched = list(dict.fromkeys(list(left.get("matched_rules", [])) + list(right.get("matched_rules", []))))
		decision = "allow"
		if any(action.get("decision") == "deny" for action in actions):
			decision = "deny"
		elif any(action.get("decision") == "require_review" for action in actions):
			decision = "require_review"
		return {"decision": decision, "matched_rules": matched, "actions": actions}

	def _reasons(self, result: dict[str, Any]) -> str:
		return ", ".join(action.get("reason", "cvsn_policy_blocked") for action in result.get("actions", [])) or "cvsn_policy_blocked"

	def _review_reasons(self, result: dict[str, Any]) -> list[str]:
		return [
			action.get("reason", "cvsn_review_required")
			for action in result.get("actions", [])
			if action.get("decision") == "require_review"
		]

	def _digest(self, value: str) -> str:
		return sha256(value.encode("utf-8")).hexdigest()

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"
