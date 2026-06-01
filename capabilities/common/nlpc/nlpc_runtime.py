"""Dependency-light generated-app runtime for APG NLP Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any

from .capability_contract import (
	SUPPORTED_LANGUAGES,
	evaluate_capability_rules,
	get_capability_contract,
)


BASELINE_LANGUAGE_CODES = {"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"}
AFRICAN_LANGUAGE_CODES = set(SUPPORTED_LANGUAGES) - BASELINE_LANGUAGE_CODES


@dataclass
class NlpcDocument:
	id: str
	tenant_id: str
	content: str
	language: str
	source_ref: str = ""
	status: str = "ingested"
	metadata: dict[str, Any] = field(default_factory=dict)
	content_hash: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"content": self.content,
			"language": self.language,
			"source_ref": self.source_ref,
			"status": self.status,
			"metadata": dict(self.metadata),
			"content_hash": self.content_hash,
			"char_count": len(self.content),
		}


@dataclass
class NlpcProcessingRun:
	id: str
	tenant_id: str
	document_id: str
	tasks: list[str]
	language: str
	confidence_score: float
	results: dict[str, dict[str, Any]]
	status: str = "completed"
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"document_id": self.document_id,
			"tasks": list(self.tasks),
			"language": self.language,
			"confidence_score": self.confidence_score,
			"results": {key: dict(value) for key, value in self.results.items()},
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
		}


@dataclass
class NlpcPipeline:
	id: str
	tenant_id: str
	name: str
	owner: str
	model_ref: str
	version: str
	tasks: list[str]
	status: str = "active"
	metadata: dict[str, Any] = field(default_factory=dict)

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
			"metadata": dict(self.metadata),
		}


@dataclass
class NlpcModelRegistration:
	id: str
	tenant_id: str
	name: str
	owner: str
	mlcm_model_ref: str
	policy_ref: str = ""
	evaluated: bool = False
	approved: bool = False
	status: str = "registered"
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"mlcm_model_ref": self.mlcm_model_ref,
			"policy_ref": self.policy_ref,
			"evaluated": self.evaluated,
			"approved": self.approved,
			"status": self.status,
			"metadata": dict(self.metadata),
		}


@dataclass
class NlpcAnnotationProject:
	id: str
	tenant_id: str
	name: str
	task: str
	guidelines: str
	consensus_threshold: float
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"task": self.task,
			"guidelines": self.guidelines,
			"consensus_threshold": self.consensus_threshold,
			"status": self.status,
		}


@dataclass
class NlpcAnnotation:
	id: str
	tenant_id: str
	project_id: str
	document_id: str
	annotator: str
	labels: list[str]
	consensus_score: float
	status: str = "accepted"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"project_id": self.project_id,
			"document_id": self.document_id,
			"annotator": self.annotator,
			"labels": list(self.labels),
			"consensus_score": self.consensus_score,
			"status": self.status,
		}


@dataclass
class NlpcLexicon:
	id: str
	tenant_id: str
	name: str
	language: str
	terms: list[str]
	owner: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"language": self.language,
			"terms": list(self.terms),
			"owner": self.owner,
			"term_count": len(self.terms),
		}


@dataclass
class NlpAgentRecord:
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
class NlpcLifecycleBatchRecord:
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
class NlpcAuditEvent:
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


class NlpcService:
	"""In-process NLP lifecycle service for generated applications."""

	def __init__(self) -> None:
		contract = get_capability_contract()
		self._documents: dict[str, NlpcDocument] = {}
		self._processing_runs: dict[str, NlpcProcessingRun] = {}
		self._pipelines: dict[str, NlpcPipeline] = {}
		self._models: dict[str, NlpcModelRegistration] = {}
		self._annotation_projects: dict[str, NlpcAnnotationProject] = {}
		self._annotations: dict[str, NlpcAnnotation] = {}
		self._lexicons: dict[str, NlpcLexicon] = {}
		self._agents: dict[str, NlpAgentRecord] = {}
		self._lifecycle_batches: dict[str, NlpcLifecycleBatchRecord] = {}
		self._audit_events: dict[str, NlpcAuditEvent] = {}
		self._supported_languages = set(SUPPORTED_LANGUAGES)
		self._enabled_tasks = set(contract["configuration"]["tasks"]["enabled"])
		self._max_document_chars = contract["configuration"]["processing"]["max_document_chars"]
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def ingest_document(
		self,
		document_id: str,
		tenant_id: str,
		content: str,
		language: str = "auto",
		source_ref: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		detected_language, confidence = self.detect_language(content) if language in {"", "auto"} else (language, 1.0)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "ingest_document",
			"content_present": bool(content and content.strip()),
			"document_chars": len(content or ""),
			"language_known": bool(detected_language),
			"language_detection_enabled": language in {"", "auto"},
		})
		if len(content or "") > self._max_document_chars:
			result = self._combine(result, self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "ingest_document",
				"document_chars": len(content or ""),
			}))
		if language in {"", "auto"}:
			result = self._combine(result, self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "detect_language",
				"confidence_score": confidence,
				"human_review_recorded": True,
			}))
		self._raise_if_blocked(result)
		document = NlpcDocument(
			id=document_id,
			tenant_id=tenant_id,
			content=content,
			language=detected_language,
			source_ref=source_ref,
			metadata=dict(metadata or {}),
			content_hash=self._digest(content),
		)
		self._documents[document_id] = document
		self._audit(tenant_id, "document_ingested", document_id, result)
		return document.to_dict()

	def process_document(
		self,
		run_id: str,
		tenant_id: str,
		document_id: str,
		tasks: str | list[str],
		redaction_policy_attached: bool = False,
		safety_policy_attached: bool = False,
		model_policy_attached: bool = False,
		search_index_attached: bool = False,
		translation_pair_present: bool = True,
		length_budget_present: bool = True,
		human_review_recorded: bool = True,
	) -> dict[str, Any]:
		document = self._require_document(document_id, tenant_id)
		task_list = [tasks] if isinstance(tasks, str) else list(tasks)
		if not task_list:
			raise PermissionError("nlp_task_required")
		result_map: dict[str, dict[str, Any]] = {}
		confidences: list[float] = []
		combined = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "process_document",
			"language_known": bool(document.language),
			"language_supported": document.language in self._supported_languages,
		})
		self._raise_if_denied(combined)
		for task in task_list:
			preflight = self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "process_document",
				"language_known": bool(document.language),
				"language_supported": document.language in self._supported_languages,
				"task": task,
				"task_enabled": task in self._enabled_tasks,
				"redaction_policy_attached": redaction_policy_attached,
				"safety_policy_attached": safety_policy_attached,
				"model_policy_attached": model_policy_attached,
				"search_index_attached": search_index_attached,
				"translation_pair_present": translation_pair_present,
				"length_budget_present": length_budget_present,
			})
			combined = self._combine(combined, preflight)
			self._raise_if_denied(combined)
			confidence, result_data = self._run_task(document.content, task, document.language)
			confidences.append(confidence)
			postflight = self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"confidence_score": confidence,
				"human_review_recorded": human_review_recorded,
			})
			combined = self._combine(combined, postflight)
			result_map[task] = result_data
		self._raise_if_denied(combined)
		run = NlpcProcessingRun(
			id=run_id,
			tenant_id=tenant_id,
			document_id=document_id,
			tasks=task_list,
			language=document.language,
			confidence_score=min(confidences),
			results=result_map,
			status="pending_review" if combined["decision"] == "require_review" else "completed",
			decision=combined["decision"],
			matched_rules=list(combined["matched_rules"]),
			review_reasons=[
				action.get("reason", "")
				for action in combined.get("actions", [])
				if action.get("decision") == "require_review" and action.get("reason")
			],
		)
		self._processing_runs[run_id] = run
		document.status = "processed"
		self._audit(tenant_id, "document_processed", run_id, combined)
		return run.to_dict()

	def register_pipeline(
		self,
		pipeline_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		model_ref: str,
		version: str,
		tasks: list[str],
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_pipeline",
			"owner_assigned": bool(owner),
			"registered_model_attached": bool(model_ref),
			"pipeline_version_present": bool(version),
		})
		for task in tasks:
			result = self._combine(result, self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "process_document",
				"task": task,
				"task_enabled": task in self._enabled_tasks,
			}))
		self._raise_if_blocked(result)
		pipeline = NlpcPipeline(
			id=pipeline_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			model_ref=model_ref,
			version=version,
			tasks=list(tasks),
			metadata=dict(metadata or {}),
		)
		self._pipelines[pipeline_id] = pipeline
		self._audit(tenant_id, "pipeline_registered", pipeline_id, result)
		return pipeline.to_dict()

	def register_model(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		mlcm_model_ref: str,
		owner: str,
		policy_ref: str = "",
		evaluated: bool = False,
		approved: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_model",
			"mlcm_model_ref_present": bool(mlcm_model_ref),
		})
		self._raise_if_blocked(result)
		if not owner:
			raise PermissionError("model_owner_required")
		model = NlpcModelRegistration(
			id=model_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			mlcm_model_ref=mlcm_model_ref,
			policy_ref=policy_ref,
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

	def create_annotation_project(
		self,
		project_id: str,
		tenant_id: str,
		name: str,
		guidelines: str,
		task: str,
		consensus_threshold: float = 0.80,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_annotation_project",
			"guidelines_present": bool(guidelines),
		})
		self._raise_if_blocked(result)
		project = NlpcAnnotationProject(
			id=project_id,
			tenant_id=tenant_id,
			name=name,
			task=task,
			guidelines=guidelines,
			consensus_threshold=consensus_threshold,
		)
		self._annotation_projects[project_id] = project
		self._audit(tenant_id, "annotation_project_created", project_id, result)
		return project.to_dict()

	def submit_annotation(
		self,
		annotation_id: str,
		tenant_id: str,
		project_id: str,
		document_id: str,
		annotator: str,
		labels: list[str],
		consensus_score: float = 1.0,
		adjudication_recorded: bool = True,
	) -> dict[str, Any]:
		project = self._require_annotation_project(project_id, tenant_id)
		self._require_document(document_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_annotation",
			"consensus_score": consensus_score,
			"adjudication_recorded": adjudication_recorded,
		})
		self._raise_if_blocked(result)
		annotation = NlpcAnnotation(
			id=annotation_id,
			tenant_id=tenant_id,
			project_id=project.id,
			document_id=document_id,
			annotator=annotator,
			labels=list(labels),
			consensus_score=consensus_score,
			status="accepted",
		)
		self._annotations[annotation_id] = annotation
		self._audit(tenant_id, "annotation_submitted", annotation_id, result)
		return annotation.to_dict()

	def register_lexicon(
		self,
		lexicon_id: str,
		tenant_id: str,
		name: str,
		language: str,
		terms: list[str],
		owner: str = "",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_lexicon",
			"language_known": bool(language),
		})
		self._raise_if_blocked(result)
		if language not in self._supported_languages:
			raise PermissionError("unsupported_language")
		lexicon = NlpcLexicon(lexicon_id, tenant_id, name, language, list(terms), owner)
		self._lexicons[lexicon_id] = lexicon
		self._audit(tenant_id, "lexicon_registered", lexicon_id, result)
		return lexicon.to_dict()

	def register_nlp_agent(
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
			"operation": "register_nlp_agent",
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
			raise ValueError("nlp_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		agent = NlpAgentRecord(
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
		self._audit(tenant_id, "nlp_agent_registered", agent.id, result)
		return agent.to_dict()

	def validate_nlpc_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "nlp_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("nlpc_lifecycle_batch_empty")
		stream_value = self._normalize_token(event_stream)
		operation_value = self._normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_nlpc_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_nlpc_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		batch = NlpcLifecycleBatchRecord(
			id=batch_id or f"nlpcbatch:{len(self._lifecycle_batches) + 1:06d}",
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
		self._audit(tenant_id, f"nlpc_lifecycle_batch_{batch.status}", batch.id, result)
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
		document = self.ingest_document(
			record_id,
			tenant_id,
			str(metadata.get("content") or metadata.get("text") or record_id),
			str(metadata.get("language") or "auto"),
			str(metadata.get("source_ref") or ""),
			metadata,
		)
		self._documents[record_id].status = status
		document["status"] = status
		return document

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_documents(tenant_id)

	def list_documents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._documents, tenant_id)

	def list_processing_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._processing_runs, tenant_id)

	def list_results(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_processing_runs(tenant_id)

	def list_pipelines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._pipelines, tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._models, tenant_id)

	def list_annotation_projects(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._annotation_projects, tenant_id)

	def list_annotations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._annotations, tenant_id)

	def list_lexicons(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lexicons, tenant_id)

	def list_nlp_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"document_count": len(self.list_documents(tenant_id)),
			"processing_run_count": len(self.list_processing_runs(tenant_id)),
			"pending_processing_review_count": len([item for item in self.list_processing_runs(tenant_id) if item["status"] == "pending_review"]),
			"pipeline_count": len(self.list_pipelines(tenant_id)),
			"model_count": len(self.list_models(tenant_id)),
			"released_model_count": len([item for item in self.list_models(tenant_id) if item["status"] == "released"]),
			"annotation_project_count": len(self.list_annotation_projects(tenant_id)),
			"annotation_count": len(self.list_annotations(tenant_id)),
			"lexicon_count": len(self.list_lexicons(tenant_id)),
			"nlp_agent_count": len(self.list_nlp_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_nlp_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"supported_language_count": len(self._supported_languages),
			"african_language_count": len(AFRICAN_LANGUAGE_CODES),
		}

	def detect_language(self, content: str) -> tuple[str, float]:
		lower = content.lower()
		if any(token in lower for token in ["habari", "asante", "karibu"]):
			return "sw", 0.92
		if any(token in lower for token in ["bonjour", "merci"]):
			return "fr", 0.90
		if any(token in lower for token in ["hola", "gracias"]):
			return "es", 0.90
		return "en", 0.80

	def _run_task(self, content: str, task: str, language: str) -> tuple[float, dict[str, Any]]:
		words = re.findall(r"[A-Za-z][A-Za-z'-]*", content)
		lower_words = [word.lower() for word in words]
		if task == "sentiment_analysis":
			positive = {"good", "great", "excellent", "happy", "asante", "improved"}
			negative = {"bad", "poor", "angry", "sad", "failed"}
			score = sum(word in positive for word in lower_words) - sum(word in negative for word in lower_words)
			label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
			return 0.86, {"label": label, "score": score}
		if task == "entity_recognition":
			entities = sorted({word for word in words if word[:1].isupper()})
			return 0.82, {"entities": entities}
		if task == "summarization":
			return 0.80, {"summary": content[:160]}
		if task == "pii_detection":
			emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", content)
			digits = re.findall(r"\b\d{6,}\b", content)
			return 0.88, {"pii_detected": bool(emails or digits), "redacted_types": ["email"] if emails else []}
		if task == "translation":
			return 0.78, {"source_language": language, "target_language": "en", "translation": content}
		if task == "semantic_search":
			return 0.79, {"query_terms": lower_words[:8], "content_hash": self._digest(content)}
		if task == "text_generation":
			return 0.77, {"generated_text": f"Governed draft response for: {content[:80]}"}
		if task == "text_classification":
			return 0.83, {"label": "general_text", "features": lower_words[:6]}
		if task == "topic_modeling":
			return 0.81, {"topics": sorted(set(lower_words[:5]))}
		if task == "keyword_extraction":
			keywords = sorted({word for word in lower_words if len(word) > 4})[:10]
			return 0.84, {"keywords": keywords}
		return 0.76, {"tokens": words, "token_count": len(words)}

	def _require_document(self, document_id: str, tenant_id: str) -> NlpcDocument:
		document = self._documents.get(document_id)
		if document is None or document.tenant_id != tenant_id:
			raise LookupError("document_not_found")
		return document

	def _require_model(self, model_id: str, tenant_id: str) -> NlpcModelRegistration:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise LookupError("model_not_found")
		return model

	def _require_annotation_project(self, project_id: str, tenant_id: str) -> NlpcAnnotationProject:
		project = self._annotation_projects.get(project_id)
		if project is None or project.tenant_id != tenant_id:
			raise LookupError("annotation_project_not_found")
		return project

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _audit(self, tenant_id: str, event_type: str, subject_id: str, result: dict[str, Any]) -> None:
		event_id = f"nlpcaudit:{self._digest(f'{tenant_id}:{event_type}:{subject_id}')[:12]}"
		self._audit_events[event_id] = NlpcAuditEvent(
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
		return {"decision": decision, "matched_rules": matched, "actions": actions, "context": right.get("context", {})}

	def _reasons(self, result: dict[str, Any]) -> str:
		return ", ".join(action.get("reason", "nlpc_policy_blocked") for action in result.get("actions", [])) or "nlpc_policy_blocked"

	def _digest(self, value: str) -> str:
		return sha256(value.encode("utf-8")).hexdigest()

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"
