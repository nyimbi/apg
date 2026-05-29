"""Digital Twin Framework service for the APG DTWN capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .models import (
	DigitalTwin,
	SimulationModel,
	SimulationRun,
	TelemetrySample,
	TopologyLink,
	TwinAuditEvent,
	TwinPrediction,
	utc_now,
)
from .twin_engine import fuse_state, simulation_outputs, stable_digest, state_version_for


class DtwnService:
	"""Tenant-scoped twin registry, telemetry fusion, simulation, and prediction service."""

	def __init__(self) -> None:
		self._twins: dict[str, DigitalTwin] = {}
		self._models: dict[str, SimulationModel] = {}
		self._telemetry: dict[str, TelemetrySample] = {}
		self._topology: dict[str, TopologyLink] = {}
		self._simulations: dict[str, SimulationRun] = {}
		self._predictions: dict[str, TwinPrediction] = {}
		self._audit_events: list[TwinAuditEvent] = []
		self._state_sequences: dict[str, int] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_twin(
		self,
		twin_id: str,
		tenant_id: str,
		asset_id: str,
		name: str,
		owner: str,
		twin_type: str,
		location: dict[str, Any] | None = None,
		initial_state: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_twin",
			"twin_owner_assigned": bool(owner),
		})
		self._raise_if_denied(result)
		if DEFAULT_CONFIGURATION["twins"]["asset_identity_required"] and not asset_id:
			raise PermissionError("asset_identity_required")
		state = dict(initial_state or {})
		self._state_sequences[twin_id] = 1
		twin = DigitalTwin(
			id=twin_id,
			tenant_id=tenant_id,
			asset_id=asset_id,
			name=name,
			owner=owner,
			twin_type=twin_type,
			location=dict(location or {}),
			state=state,
			state_version=state_version_for(twin_id, state, 1),
		)
		self._twins[twin_id] = twin
		self._record_audit(tenant_id, "twin_created", twin_id, owner, twin.to_dict())
		return twin.to_dict()

	def register_simulation_model(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		version: str,
		owner: str,
		model_type: str,
		calibration_evidence: str,
		approved_by: str | None,
		confidence: float = 0.75,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if DEFAULT_CONFIGURATION["simulation"]["calibration_evidence_required"] and not calibration_evidence:
			raise PermissionError("calibration_evidence_required")
		if confidence < DEFAULT_CONFIGURATION["simulation"]["prediction_confidence_threshold"]:
			raise PermissionError("prediction_confidence_threshold")
		model = SimulationModel(
			id=model_id,
			tenant_id=tenant_id,
			name=name,
			version=version,
			owner=owner,
			model_type=model_type,
			calibration_evidence=calibration_evidence,
			approved_by=approved_by,
			confidence=confidence,
			status="approved" if approved_by else "draft",
		)
		self._models[model_id] = model
		self._record_audit(tenant_id, "model_registered", model_id, owner, model.to_dict())
		return model.to_dict()

	def ingest_telemetry(
		self,
		sample_id: str,
		tenant_id: str,
		twin_id: str,
		source_id: str,
		source_type: str,
		authenticated: bool,
		measurements: dict[str, Any],
		geospatial_context: dict[str, Any] | None = None,
		vision_signals: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"telemetry_source_authenticated": authenticated,
		})
		self._raise_if_denied(result)
		sequence = self._state_sequences.get(twin_id, 1) + 1
		state = fuse_state(twin.state, measurements)
		version = state_version_for(twin_id, state, sequence)
		twin.state = state
		twin.state_version = version
		twin.updated_at = utc_now()
		self._state_sequences[twin_id] = sequence
		sample = TelemetrySample(
			id=sample_id,
			tenant_id=tenant_id,
			twin_id=twin_id,
			source_id=source_id,
			source_type=source_type,
			authenticated=authenticated,
			measurements=dict(measurements),
			geospatial_context=dict(geospatial_context or {}),
			vision_signals=dict(vision_signals or {}),
			state_version=version,
		)
		self._telemetry[sample_id] = sample
		self._record_audit(tenant_id, "telemetry_ingested", sample_id, source_id, sample.to_dict())
		return sample.to_dict()

	def link_topology(
		self,
		link_id: str,
		tenant_id: str,
		source_twin_id: str,
		target_twin_id: str,
		relationship: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		source = self._require_twin(source_twin_id, tenant_id)
		target = self._require_twin(target_twin_id, tenant_id)
		link = TopologyLink(
			id=link_id,
			tenant_id=tenant_id,
			source_twin_id=source_twin_id,
			target_twin_id=target_twin_id,
			relationship=relationship,
			metadata=dict(metadata or {}),
		)
		self._topology[link_id] = link
		for twin in (source, target):
			if link_id not in twin.topology_refs:
				twin.topology_refs.append(link_id)
				twin.updated_at = utc_now()
		self._record_audit(tenant_id, "topology_linked", link_id, source_twin_id, link.to_dict())
		return link.to_dict()

	def run_simulation(
		self,
		run_id: str,
		tenant_id: str,
		twin_id: str,
		model_id: str,
		scenario: str,
		environment: str = "sandbox",
		approved_by: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		model = self._require_model(model_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "run_production_simulation" if environment == "production" else "run_simulation",
			"model_present": model.status == "approved",
			"approval_recorded": bool(approved_by),
		})
		self._raise_if_denied(result)
		outputs = simulation_outputs(twin.state, model.confidence, scenario)
		run = SimulationRun(
			id=run_id,
			tenant_id=tenant_id,
			twin_id=twin_id,
			model_id=model_id,
			scenario=scenario,
			environment=environment,
			approved_by=approved_by,
			status="completed",
			outputs=outputs,
			completed_at=utc_now(),
		)
		self._simulations[run_id] = run
		self._record_audit(tenant_id, "simulation_completed", run_id, approved_by or model.owner, run.to_dict())
		return run.to_dict()

	def record_prediction(
		self,
		prediction_id: str,
		tenant_id: str,
		twin_id: str,
		model_id: str,
		risk_score: float,
		confidence: float,
		horizon: str,
		recommendation: str,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_twin(twin_id, tenant_id)
		self._require_model(model_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"prediction_risk_score": risk_score,
			"prediction_review_recorded": bool(reviewed_by),
		})
		self._raise_if_denied(result)
		prediction = TwinPrediction(
			id=prediction_id,
			tenant_id=tenant_id,
			twin_id=twin_id,
			model_id=model_id,
			risk_score=risk_score,
			confidence=confidence,
			horizon=horizon,
			recommendation=recommendation,
			review_required=result["decision"] == "require_review",
			reviewed_by=reviewed_by,
			status="review_required" if result["decision"] == "require_review" else "active",
		)
		self._predictions[prediction_id] = prediction
		self._record_audit(tenant_id, "prediction_recorded", prediction_id, reviewed_by or "system", prediction.to_dict())
		return prediction.to_dict()

	def review_prediction(self, prediction_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
		prediction = self._require_prediction(prediction_id, tenant_id)
		prediction.review_required = False
		prediction.reviewed_by = reviewer
		prediction.status = "reviewed"
		self._record_audit(tenant_id, "prediction_reviewed", prediction_id, reviewer, prediction.to_dict())
		return prediction.to_dict()

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		return {
			"tenant_id": tenant_id,
			"twin_count": len(self.list_twins(tenant_id)),
			"model_count": len(self.list_models(tenant_id)),
			"telemetry_sample_count": len(self.list_telemetry(tenant_id)),
			"topology_link_count": len(self.list_topology(tenant_id)),
			"simulation_count": len(self.list_simulations(tenant_id)),
			"review_required_prediction_count": sum(1 for item in self.list_predictions(tenant_id) if item["review_required"]),
		}

	def list_twins(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._twins, tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._models, tenant_id)

	def list_telemetry(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._telemetry, tenant_id)

	def list_topology(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._topology, tenant_id)

	def list_simulations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._simulations, tenant_id)

	def list_predictions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._predictions, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_twin(self, twin_id: str, tenant_id: str) -> DigitalTwin:
		twin = self._twins.get(twin_id)
		if twin is None or twin.tenant_id != tenant_id:
			raise KeyError(f"unknown_twin:{twin_id}")
		return twin

	def _require_model(self, model_id: str, tenant_id: str) -> SimulationModel:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise KeyError(f"unknown_model:{model_id}")
		return model

	def _require_prediction(self, prediction_id: str, tenant_id: str) -> TwinPrediction:
		prediction = self._predictions.get(prediction_id)
		if prediction is None or prediction.tenant_id != tenant_id:
			raise KeyError(f"unknown_prediction:{prediction_id}")
		return prediction

	def _record_audit(self, tenant_id: str, action: str, resource_id: str, actor: str, metadata: dict[str, Any]) -> None:
		payload = {
			"tenant_id": tenant_id,
			"action": action,
			"resource_id": resource_id,
			"actor": actor,
			"metadata": metadata,
		}
		self._audit_events.append(TwinAuditEvent(
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
			raise PermissionError(", ".join(action.get("reason", "digital_twin_policy_blocked") for action in result["actions"]))

	def _list_for_tenant(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]
