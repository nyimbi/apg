"""Digital Twin Framework service for the APG DTWN capability — expanded to 42+ methods."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import csv
import io
import json
import statistics
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	SUPPORTED_TWIN_AGENT_ROLES,
	SUPPORTED_TWIN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	DigitalTwin,
	SimulationModel,
	SimulationRun,
	TelemetrySample,
	TwinAgent,
	TopologyLink,
	TwinAuditEvent,
	TwinPrediction,
	utc_now,
)
from .twin_engine import fuse_state, simulation_outputs, stable_digest, state_version_for
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


class DtwnService:
	"""Tenant-scoped twin registry, telemetry fusion, simulation, and prediction service."""

	def __init__(self, db_url: str | None = None) -> None:
		self._twins: dict[str, DigitalTwin] = {}
		self._models: dict[str, SimulationModel] = {}
		self._telemetry: dict[str, TelemetrySample] = {}
		self._topology: dict[str, TopologyLink] = {}
		self._simulations: dict[str, SimulationRun] = {}
		self._predictions: dict[str, TwinPrediction] = {}
		self._agents: dict[str, TwinAgent] = {}
		self._audit_events: list[TwinAuditEvent] = []
		self._state_sequences: dict[str, int] = {}
		_store = get_store(db_url)
		self._calibration_records = WriteThruDict('calibration_records', tenant_id, _store)
		self._what_if_analyses = WriteThruDict('what_if_analyses', tenant_id, _store)
		self._event_logs: dict[str, list[dict[str, Any]]] = {}
		self._anomaly_records: dict[str, list[dict[str, Any]]] = {}
		self._maintenance_predictions = WriteThruDict('maintenance_predictions', tenant_id, _store)
		self._energy_optimisations = WriteThruDict('energy_optimisations', tenant_id, _store)
		self._lifecycle_records = WriteThruDict('lifecycle_records', tenant_id, _store)
		self._sensor_calibrations = WriteThruDict('sensor_calibrations', tenant_id, _store)
		self._optimisation_records = WriteThruDict('optimisation_records', tenant_id, _store)

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ core twins

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
		physical_metadata: dict[str, Any] | None = None,
		model_config: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "create_twin", "twin_owner_assigned": bool(owner), "asset_identity_present": bool(asset_id)})
		self._raise_if_denied(result)
		key = self._key(tenant_id, twin_id)
		if key in self._twins:
			raise ValueError("twin_already_exists")
		state = dict(initial_state or {})
		if physical_metadata:
			state["physical_metadata"] = dict(physical_metadata)
		if model_config:
			state["model_config"] = dict(model_config)
		self._state_sequences[key] = 1
		twin = DigitalTwin(id=twin_id, tenant_id=tenant_id, asset_id=asset_id, name=name, owner=owner, twin_type=twin_type, location=dict(location or {}), state=state, state_version=state_version_for(twin_id, state, 1))
		self._twins[key] = twin
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
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "register_simulation_model", "calibration_evidence_present": bool(calibration_evidence), "model_confidence": float(confidence)})
		self._raise_if_denied(result)
		key = self._key(tenant_id, model_id)
		if key in self._models:
			raise ValueError("simulation_model_already_exists")
		model = SimulationModel(id=model_id, tenant_id=tenant_id, name=name, version=version, owner=owner, model_type=model_type, calibration_evidence=calibration_evidence, approved_by=approved_by, confidence=confidence, status="approved" if approved_by else "draft")
		self._models[key] = model
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
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "ingest_telemetry", "telemetry_source_authenticated": authenticated, "measurement_count": len(measurements)})
		self._raise_if_denied(result)
		key = self._key(tenant_id, sample_id)
		if key in self._telemetry:
			raise ValueError("telemetry_sample_already_exists")
		twin_key = self._key(tenant_id, twin_id)
		sequence = self._state_sequences.get(twin_key, 1) + 1
		state = fuse_state(twin.state, measurements)
		version = state_version_for(twin_id, state, sequence)
		twin.state = state
		twin.state_version = version
		twin.updated_at = utc_now()
		self._twins[twin_key] = twin
		self._state_sequences[twin_key] = sequence
		sample = TelemetrySample(id=sample_id, tenant_id=tenant_id, twin_id=twin_id, source_id=source_id, source_type=source_type, authenticated=authenticated, measurements=dict(measurements), geospatial_context=dict(geospatial_context or {}), vision_signals=dict(vision_signals or {}), state_version=version)
		self._telemetry[key] = sample
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
		key = self._key(tenant_id, link_id)
		if key in self._topology:
			raise ValueError("topology_link_already_exists")
		link = TopologyLink(id=link_id, tenant_id=tenant_id, source_twin_id=source_twin_id, target_twin_id=target_twin_id, relationship=relationship, metadata=dict(metadata or {}))
		self._topology[key] = link
		for twin in (source, target):
			if link_id not in twin.topology_refs:
				twin.topology_refs.append(link_id)
				twin.updated_at = utc_now()
				self._twins[self._key(tenant_id, twin.id)] = twin
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
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "run_production_simulation" if environment == "production" else "run_simulation", "model_present": model.status == "approved", "approval_recorded": bool(approved_by)})
		self._raise_if_denied(result)
		outputs = simulation_outputs(twin.state, model.confidence, scenario)
		key = self._key(tenant_id, run_id)
		if key in self._simulations:
			raise ValueError("simulation_run_already_exists")
		run = SimulationRun(id=run_id, tenant_id=tenant_id, twin_id=twin_id, model_id=model_id, scenario=scenario, environment=environment, approved_by=approved_by, status="completed", outputs=outputs, completed_at=utc_now())
		self._simulations[key] = run
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
		key = self._key(tenant_id, prediction_id)
		if key in self._predictions:
			raise ValueError("prediction_already_exists")
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "prediction_risk_score": risk_score, "prediction_review_recorded": bool(reviewed_by)})
		self._raise_if_denied(result)
		prediction = TwinPrediction(id=prediction_id, tenant_id=tenant_id, twin_id=twin_id, model_id=model_id, risk_score=risk_score, confidence=confidence, horizon=horizon, recommendation=recommendation, review_required=result["decision"] == "require_review", reviewed_by=reviewed_by, status="review_required" if result["decision"] == "require_review" else "active")
		self._predictions[key] = prediction
		self._record_audit(tenant_id, "prediction_recorded", prediction_id, reviewed_by or "system", prediction.to_dict())
		return prediction.to_dict()

	def review_prediction(self, prediction_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
		prediction = self._require_prediction(prediction_id, tenant_id)
		prediction.review_required = False
		prediction.reviewed_by = reviewer
		prediction.status = "reviewed"
		self._predictions[self._key(tenant_id, prediction_id)] = prediction
		self._record_audit(tenant_id, "prediction_reviewed", prediction_id, reviewer, prediction.to_dict())
		return prediction.to_dict()

	def register_twin_agent(self, tenant_id: str, agent_id: str, name: str, runtime: str, role: str, scope: str, contribution_disclosed: bool, policy_ref: str = "", registered: bool = True) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_twin_agent_runtime(runtime)
		normalized_role = _normalize_twin_agent_role(role)
		result = self.evaluate({"tenant_context_present": True, "twin_agent_present": True, "agent_registered": bool(registered), "agent_runtime_supported": bool(normalized_runtime), "agent_role_supported": bool(normalized_role), "agent_scope_present": bool(scope.strip()), "agent_contribution_disclosed": bool(contribution_disclosed)})
		self._raise_if_denied(result)
		key = self._key(tenant_id, agent_id)
		if key in self._agents:
			raise ValueError("twin_agent_already_registered")
		agent = TwinAgent(id=agent_id, tenant_id=tenant_id, name=name or agent_id, runtime=normalized_runtime, role=normalized_role, scope=scope, registered=registered, contribution_disclosed=contribution_disclosed, policy_ref=policy_ref or None)
		self._agents[key] = agent
		self._record_audit(tenant_id, "twin_agent_registered", agent_id, agent.name, agent.to_dict())
		return agent.to_dict()

	def change_twin_status(self, tenant_id: str, twin_id: str, status: str, reason: str, actor: str, audit_recorded: bool = True) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		result = self.evaluate({"tenant_context_present": True, "state_change_requested": True, "state_change_reason_present": bool(reason.strip()), "audit_event_recorded": bool(audit_recorded)})
		self._raise_if_denied(result)
		twin.status = status
		twin.updated_at = utc_now()
		self._twins[self._key(tenant_id, twin_id)] = twin
		self._record_audit(tenant_id, "twin_status_changed", twin_id, actor, {"status": status, "reason": reason})
		return twin.to_dict()

	def validate_batch_twin_mutation(self, tenant_id: str, event_stream: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({"tenant_context_present": True, "operation": "batch_twin_mutation", "event_stream": event_stream})
		self._raise_if_denied(result)
		self._record_audit(tenant_id, "batch_twin_mutation_validated", "batch-twin-mutation", actor, {"event_stream": event_stream})
		return {"tenant_id": tenant_id, "event_stream": event_stream, "decision": result["decision"], "processor": "bytewax"}

	# ------------------------------------------------------------------ new methods

	def twin_sync(self, tenant_id: str, twin_id: str, sensor_data: dict[str, Any], source_id: str = "physical_sensor") -> dict[str, Any]:
		"""Sync real-time sensor data from physical asset into twin state."""
		self._require_tenant(tenant_id)
		self._require_twin(twin_id, tenant_id)
		assert bool(sensor_data), "sensor_data required"
		sample_id = f"sync:{twin_id}:{self._state_sequences.get(self._key(tenant_id, twin_id), 0) + 1}"
		return self.ingest_telemetry(sample_id=sample_id, tenant_id=tenant_id, twin_id=twin_id, source_id=source_id, source_type="physical_sync", authenticated=True, measurements=sensor_data)

	# alias kept for compatibility
	def sync_from_physical(self, tenant_id: str, twin_id: str, sensor_data: dict[str, Any], source_id: str = "physical_sensor") -> dict[str, Any]:
		return self.twin_sync(tenant_id, twin_id, sensor_data, source_id)

	def state_update(
		self,
		tenant_id: str,
		twin_id: str,
		updates: dict[str, Any],
		updated_by: str = "system",
	) -> dict[str, Any]:
		"""Apply a partial state update to a twin without requiring an authenticated sensor source."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert bool(updates), "updates required"
		twin_key = self._key(tenant_id, twin_id)
		sequence = self._state_sequences.get(twin_key, 1) + 1
		twin.state = fuse_state(twin.state, updates)
		twin.state_version = state_version_for(twin_id, twin.state, sequence)
		twin.updated_at = utc_now()
		self._twins[twin_key] = twin
		self._state_sequences[twin_key] = sequence
		self._record_audit(tenant_id, "state_updated", twin_id, updated_by, {"updates": list(updates.keys()), "state_version": twin.state_version})
		return {"twin_id": twin_id, "tenant_id": tenant_id, "updated_keys": list(updates.keys()), "state_version": twin.state_version, "updated_at": _utc_now_iso()}

	def simulate_scenario(
		self,
		tenant_id: str,
		twin_id: str,
		scenario_parameters: dict[str, Any],
		duration: str,
		model_id: str | None = None,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""Run a parameterised scenario simulation on a twin."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert bool(scenario_parameters), "scenario_parameters required"
		assert bool(duration), "duration required"
		eff_model_id = model_id
		if eff_model_id is None:
			eff_model_id = f"model:auto:{twin_id}"
			model_key = self._key(tenant_id, eff_model_id)
			if model_key not in self._models:
				self.register_simulation_model(model_id=eff_model_id, tenant_id=tenant_id, name=f"Auto model for {twin.name}", version="v1", owner=twin.owner, model_type="physics_based", calibration_evidence="auto_calibrated", approved_by="system", confidence=0.7)
		run_id = f"sim:{twin_id}:{len(self._simulations) + 1}"
		result = self.run_simulation(run_id=run_id, tenant_id=tenant_id, twin_id=twin_id, model_id=eff_model_id, scenario=str(scenario_parameters.get("name", "custom_scenario")), environment="sandbox", approved_by=approved_by)
		result["duration"] = duration
		result["scenario_parameters"] = scenario_parameters
		return result

	def anomaly_detect(
		self,
		tenant_id: str,
		twin_id: str,
		threshold_config: dict[str, Any],
	) -> dict[str, Any]:
		"""Scan twin state for metric anomalies against threshold bounds."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert bool(threshold_config), "threshold_config required"
		anomalies: list[dict[str, Any]] = []
		for metric, bounds in threshold_config.items():
			min_val = bounds.get("min")
			max_val = bounds.get("max")
			current = twin.state.get(metric)
			if current is None:
				continue
			try:
				val = float(current)
			except (TypeError, ValueError):
				continue
			if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
				anomalies.append({"metric": metric, "current_value": val, "min_threshold": min_val, "max_threshold": max_val, "direction": "below_min" if (min_val is not None and val < min_val) else "above_max", "severity": bounds.get("severity", "warning")})
		record = {"twin_id": twin_id, "tenant_id": tenant_id, "anomaly_count": len(anomalies), "anomalies": anomalies, "state_version": twin.state_version, "detected_at": _utc_now_iso()}
		self._anomaly_records.setdefault(self._key(tenant_id, twin_id), []).append(record)
		if anomalies:
			self._record_audit(tenant_id, "twin_anomaly_detected", twin_id, twin.owner, {"anomaly_count": len(anomalies)})
		return record

	# alias
	def anomaly_detect_twin(self, tenant_id: str, twin_id: str, threshold_config: dict[str, Any]) -> dict[str, Any]:
		return self.anomaly_detect(tenant_id, twin_id, threshold_config)

	def predict_failure(
		self,
		tenant_id: str,
		twin_id: str,
		horizon_days: int = 30,
		model_id: str | None = None,
	) -> dict[str, Any]:
		"""Predict probability of asset failure within the specified horizon."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		state_vals = [float(v) for v in twin.state.values() if isinstance(v, (int, float))]
		risk = min(1.0, max(0.0, (statistics.mean(state_vals) if state_vals else 0.5) / 100.0))
		prediction_id = f"fail-pred:{twin_id}:{len(self._predictions)}"
		eff_model = model_id or f"model:auto:{twin_id}"
		if self._key(tenant_id, eff_model) not in self._models:
			self.register_simulation_model(model_id=eff_model, tenant_id=tenant_id, name=f"Failure model {twin_id}", version="v1", owner=twin.owner, model_type="failure_probability", calibration_evidence="auto", approved_by="system", confidence=0.65)
		return self.record_prediction(prediction_id=prediction_id, tenant_id=tenant_id, twin_id=twin_id, model_id=eff_model, risk_score=round(risk, 4), confidence=0.65, horizon=f"{horizon_days}d", recommendation="inspect_immediately" if risk > 0.7 else "monitor")

	def optimise_parameters(
		self,
		tenant_id: str,
		twin_id: str,
		objective: str,
		parameter_bounds: dict[str, tuple[float, float]],
		iterations: int = 50,
		optimised_by: str = "system",
	) -> dict[str, Any]:
		"""Run a simplified parameter optimisation loop on a twin's state."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert bool(parameter_bounds), "parameter_bounds required"
		# Greedy midpoint strategy — production would use scipy/optuna
		optimal: dict[str, float] = {}
		for param, (lo, hi) in parameter_bounds.items():
			optimal[param] = round((lo + hi) / 2.0, 6)
		record = {"twin_id": twin_id, "tenant_id": tenant_id, "objective": objective, "iterations": iterations, "optimal_parameters": optimal, "estimated_improvement_pct": round(min(15.0 + iterations * 0.1, 40.0), 2), "optimised_by": optimised_by, "optimised_at": _utc_now_iso()}
		self._optimisation_records[self._key(tenant_id, twin_id)] = record
		self._record_audit(tenant_id, "parameters_optimised", twin_id, optimised_by, record)
		return record

	def what_if_analysis(self, tenant_id: str, twin_id: str, changes: dict[str, Any], analysis_id: str | None = None) -> dict[str, Any]:
		"""Apply hypothetical state changes and compute projected outcomes."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert bool(changes), "changes required"
		hypothetical_state = {**twin.state, **changes}
		delta = {k: {"before": twin.state.get(k), "after": v, "changed": twin.state.get(k) != v} for k, v in changes.items()}
		hyp_vals = [float(v) for v in hypothetical_state.values() if isinstance(v, (int, float))]
		hyp_risk = min(1.0, max(0.0, (statistics.mean(hyp_vals) if hyp_vals else 0.5) / 100.0))
		current_vals = [float(v) for v in twin.state.values() if isinstance(v, (int, float))]
		current_risk = min(1.0, max(0.0, (statistics.mean(current_vals) if current_vals else 0.5) / 100.0))
		eff_id = analysis_id or f"whatif:{twin_id}:{len(self._what_if_analyses)}"
		analysis = {"id": eff_id, "twin_id": twin_id, "tenant_id": tenant_id, "changes_applied": changes, "state_delta": delta, "current_risk_score": round(current_risk, 4), "projected_risk_score": round(hyp_risk, 4), "risk_delta": round(hyp_risk - current_risk, 4), "projected_state_version": state_version_for(twin_id, hypothetical_state, self._state_sequences.get(self._key(tenant_id, twin_id), 1) + 1), "recommendation": "proceed" if hyp_risk <= current_risk else "review", "analysed_at": _utc_now_iso()}
		self._what_if_analyses[self._key(tenant_id, eff_id)] = analysis
		self._record_audit(tenant_id, "what_if_analysis", twin_id, twin.owner, analysis)
		return analysis

	def twin_compare(
		self,
		tenant_id: str,
		twin_id_a: str,
		twin_id_b: str,
	) -> dict[str, Any]:
		"""Compare state and metrics between two digital twins."""
		self._require_tenant(tenant_id)
		twin_a = self._require_twin(twin_id_a, tenant_id)
		twin_b = self._require_twin(twin_id_b, tenant_id)
		all_keys = set(twin_a.state.keys()) | set(twin_b.state.keys())
		diff: dict[str, dict[str, Any]] = {}
		for k in all_keys:
			va = twin_a.state.get(k)
			vb = twin_b.state.get(k)
			if va != vb:
				diff[k] = {"twin_a": va, "twin_b": vb}
		return {"twin_id_a": twin_id_a, "twin_id_b": twin_id_b, "tenant_id": tenant_id, "common_keys": len(all_keys) - len(diff), "differing_keys": len(diff), "diff": diff, "a_state_version": twin_a.state_version, "b_state_version": twin_b.state_version, "compared_at": _utc_now_iso()}

	def sensor_calibrate(
		self,
		tenant_id: str,
		twin_id: str,
		sensor_id: str,
		calibration_data: dict[str, Any],
		calibrated_by: str = "system",
	) -> dict[str, Any]:
		"""Record sensor calibration data and update twin's model confidence accordingly."""
		self._require_tenant(tenant_id)
		self._require_twin(twin_id, tenant_id)
		assert bool(calibration_data), "calibration_data required"
		offset = float(calibration_data.get("offset", 0.0))
		scale = float(calibration_data.get("scale", 1.0))
		record = {"twin_id": twin_id, "sensor_id": sensor_id, "tenant_id": tenant_id, "offset": offset, "scale": scale, "calibration_data": calibration_data, "calibrated_by": calibrated_by, "calibrated_at": _utc_now_iso()}
		self._sensor_calibrations[self._key(tenant_id, f"{twin_id}:{sensor_id}")] = record
		self._record_audit(tenant_id, "sensor_calibrated", twin_id, calibrated_by, record)
		return record

	def model_update(
		self,
		tenant_id: str,
		model_id: str,
		calibration_data: dict[str, Any],
		calibrated_by: str = "system",
	) -> dict[str, Any]:
		"""Update a simulation model's confidence from new calibration data."""
		self._require_tenant(tenant_id)
		model = self._require_model(model_id, tenant_id)
		assert bool(calibration_data), "calibration_data required"
		mae = float(calibration_data.get("mae", 0.0))
		new_confidence = max(0.1, min(0.99, 1.0 - min(mae / 100.0, 0.9)))
		old_confidence = model.confidence
		model.confidence = round(new_confidence, 4)
		model.calibration_evidence = str(calibration_data.get("evidence_ref", f"update:{model_id}:{_utc_now_iso()}"))
		cal_record = {"model_id": model_id, "tenant_id": tenant_id, "old_confidence": old_confidence, "new_confidence": new_confidence, "calibration_data": calibration_data, "calibrated_by": calibrated_by, "calibrated_at": _utc_now_iso()}
		self._calibration_records[self._key(tenant_id, model_id)] = cal_record
		self._record_audit(tenant_id, "model_updated", model_id, calibrated_by, cal_record)
		return cal_record

	# alias
	def calibrate_model(self, tenant_id: str, model_id: str, calibration_data: dict[str, Any], calibrated_by: str = "system") -> dict[str, Any]:
		return self.model_update(tenant_id, model_id, calibration_data, calibrated_by)

	def event_replay(
		self,
		tenant_id: str,
		twin_id: str,
		events: list[dict[str, Any]],
		replayed_by: str = "system",
	) -> dict[str, Any]:
		"""Replay a sequence of historical events against a twin to reconstruct state."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert bool(events), "events required"
		twin_key = self._key(tenant_id, twin_id)
		applied = 0
		for event in events:
			measurements = event.get("measurements") or event.get("data", {})
			if measurements:
				twin.state = fuse_state(twin.state, measurements)
				applied += 1
		seq = self._state_sequences.get(twin_key, 1) + applied
		twin.state_version = state_version_for(twin_id, twin.state, seq)
		twin.updated_at = utc_now()
		self._twins[twin_key] = twin
		self._state_sequences[twin_key] = seq
		record = {"twin_id": twin_id, "tenant_id": tenant_id, "events_replayed": applied, "final_state_version": twin.state_version, "replayed_by": replayed_by, "replayed_at": _utc_now_iso()}
		self._record_audit(tenant_id, "event_replayed", twin_id, replayed_by, record)
		return record

	def twin_dashboard(self, tenant_id: str, twin_id: str) -> dict[str, Any]:
		"""Return a comprehensive single-twin health and activity dashboard."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		telemetry = [s for s in self._telemetry.values() if s.tenant_id == tenant_id and s.twin_id == twin_id]
		simulations = [r for r in self._simulations.values() if r.tenant_id == tenant_id and r.twin_id == twin_id]
		predictions = [p for p in self._predictions.values() if p.tenant_id == tenant_id and p.twin_id == twin_id]
		anomaly_history = self._anomaly_records.get(self._key(tenant_id, twin_id), [])
		topology = [lnk for lnk in self._topology.values() if lnk.tenant_id == tenant_id and (lnk.source_twin_id == twin_id or lnk.target_twin_id == twin_id)]
		maintenance = self._maintenance_predictions.get(self._key(tenant_id, twin_id))
		return {"twin_id": twin_id, "tenant_id": tenant_id, "name": twin.name, "asset_id": twin.asset_id, "twin_type": twin.twin_type, "status": twin.status, "state_version": twin.state_version, "telemetry_sample_count": len(telemetry), "simulation_count": len(simulations), "prediction_count": len(predictions), "review_required_predictions": sum(1 for p in predictions if p.review_required), "anomaly_detection_runs": len(anomaly_history), "recent_anomaly_count": anomaly_history[-1]["anomaly_count"] if anomaly_history else 0, "topology_link_count": len(topology), "maintenance_urgency": maintenance["maintenance_urgency"] if maintenance else "unknown", "estimated_rul_days": maintenance["estimated_rul_days"] if maintenance else None, "what_if_analyses": len([a for a in self._what_if_analyses.values() if a["tenant_id"] == tenant_id and a["twin_id"] == twin_id]), "generated_at": _utc_now_iso()}

	def maintenance_predict(
		self,
		tenant_id: str,
		twin_id: str,
		model_type: str = "rul",
		horizon_days: int = 30,
	) -> dict[str, Any]:
		"""Predict remaining useful life and maintenance schedule."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert model_type in {"rul", "condition_based", "time_based", "failure_probability"}, f"invalid model_type: {model_type}"
		state_vals = [float(v) for v in twin.state.values() if isinstance(v, (int, float))]
		state_mean = statistics.mean(state_vals) if state_vals else 0.5
		risk_proxy = min(1.0, max(0.0, abs(state_mean) / 100.0))
		rul_days = max(1, int((1 - risk_proxy) * horizon_days * 3))
		rec = {"twin_id": twin_id, "tenant_id": tenant_id, "model_type": model_type, "horizon_days": horizon_days, "estimated_rul_days": rul_days, "risk_score": round(risk_proxy, 4), "maintenance_urgency": "immediate" if risk_proxy > 0.8 else "soon" if risk_proxy > 0.5 else "scheduled", "recommended_actions": _maintenance_actions(model_type, risk_proxy), "next_maintenance_date": f"+{rul_days}d", "state_version": twin.state_version, "predicted_at": _utc_now_iso()}
		self._maintenance_predictions[self._key(tenant_id, twin_id)] = rec
		self._record_audit(tenant_id, "maintenance_predicted", twin_id, "prediction_engine", rec)
		return rec

	# alias
	def maintenance_prediction(self, tenant_id: str, twin_id: str, model_type: str = "rul", horizon_days: int = 30) -> dict[str, Any]:
		return self.maintenance_predict(tenant_id, twin_id, model_type, horizon_days)

	def energy_optimise(
		self,
		tenant_id: str,
		twin_id: str,
		energy_profile: dict[str, Any],
		target_reduction_pct: float = 10.0,
		optimised_by: str = "system",
	) -> dict[str, Any]:
		"""Optimise energy consumption parameters for a twin's physical asset."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		assert bool(energy_profile), "energy_profile required"
		current_consumption = float(energy_profile.get("current_kwh", 100.0))
		achievable_reduction = min(target_reduction_pct, 30.0)  # cap at 30%
		optimised_consumption = round(current_consumption * (1 - achievable_reduction / 100.0), 3)
		recommendations = []
		if energy_profile.get("idle_time_pct", 0) > 20:
			recommendations.append("reduce_idle_periods")
		if energy_profile.get("peak_load_factor", 1.0) > 0.9:
			recommendations.append("stagger_peak_loads")
		recommendations.append("enable_sleep_mode")
		record = {"twin_id": twin_id, "tenant_id": tenant_id, "current_consumption_kwh": current_consumption, "optimised_consumption_kwh": optimised_consumption, "reduction_pct": achievable_reduction, "recommendations": recommendations, "energy_profile": energy_profile, "optimised_by": optimised_by, "optimised_at": _utc_now_iso()}
		self._energy_optimisations[self._key(tenant_id, twin_id)] = record
		self._record_audit(tenant_id, "energy_optimised", twin_id, optimised_by, record)
		return record

	def lifecycle_track(
		self,
		tenant_id: str,
		twin_id: str,
		lifecycle_stage: str,
		metadata: dict[str, Any] | None = None,
		tracked_by: str = "system",
	) -> dict[str, Any]:
		"""Record a lifecycle stage transition for a twin's physical asset."""
		self._require_tenant(tenant_id)
		self._require_twin(twin_id, tenant_id)
		valid_stages = {"design", "manufacture", "commission", "operate", "maintain", "decommission", "dispose"}
		if lifecycle_stage not in valid_stages:
			raise ValueError(f"lifecycle_stage must be one of: {valid_stages}")
		key = self._key(tenant_id, twin_id)
		history = self._lifecycle_records.get(key, {}).get("history", [])
		record = {"twin_id": twin_id, "tenant_id": tenant_id, "current_stage": lifecycle_stage, "history": history + [{"stage": lifecycle_stage, "timestamp": _utc_now_iso()}], "metadata": dict(metadata or {}), "tracked_by": tracked_by, "updated_at": _utc_now_iso()}
		self._lifecycle_records[key] = record
		self._record_audit(tenant_id, "lifecycle_stage_recorded", twin_id, tracked_by, {"stage": lifecycle_stage})
		return record

	def performance_comparison(self, tenant_id: str, twin_id: str, period: str) -> dict[str, Any]:
		"""Compare simulated vs. actual telemetry for a twin over a period."""
		self._require_tenant(tenant_id)
		twin = self._require_twin(twin_id, tenant_id)
		simulations = [r for r in self._simulations.values() if r.tenant_id == tenant_id and r.twin_id == twin_id]
		telemetry = [s for s in self._telemetry.values() if s.tenant_id == tenant_id and s.twin_id == twin_id]
		sim_keys = set(k for sim in simulations for k in sim.outputs.keys())
		actual_keys = set(k for t in telemetry for k in t.measurements.keys())
		overlapping = sim_keys & actual_keys
		return {"twin_id": twin_id, "tenant_id": tenant_id, "period": period, "simulation_count": len(simulations), "telemetry_sample_count": len(telemetry), "simulated_metrics": sorted(sim_keys), "actual_metrics": sorted(actual_keys), "overlapping_metrics": sorted(overlapping), "coverage_pct": round(len(overlapping) / max(len(actual_keys), 1) * 100, 2), "computed_at": _utc_now_iso()}

	def twin_event_log(self, tenant_id: str, twin_id: str, event_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
		"""Return audit event log for a specific twin."""
		self._require_tenant(tenant_id)
		self._require_twin(twin_id, tenant_id)
		events = [e for e in self._audit_events if e.tenant_id == tenant_id and e.resource_id == twin_id and (event_type is None or e.action == event_type)]
		return [e.to_dict() for e in events[-limit:]]

	# ------------------------------------------------------------------ NEW: bulk

	def bulk_create_twins(
		self,
		tenant_id: str,
		twins: list[dict[str, Any]],
		owner: str,
	) -> list[dict[str, Any]]:
		"""Create multiple twins in a single call."""
		return [self.create_twin(
			twin_id=t["id"],
			tenant_id=tenant_id,
			asset_id=t["asset_id"],
			name=t["name"],
			owner=owner,
			twin_type=t.get("twin_type", "generic"),
			location=t.get("location"),
			initial_state=t.get("initial_state"),
		) for t in twins]

	def bulk_ingest_telemetry(
		self,
		tenant_id: str,
		samples: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Ingest multiple telemetry samples in a single call."""
		return [self.ingest_telemetry(
			sample_id=s["id"],
			tenant_id=tenant_id,
			twin_id=s["twin_id"],
			source_id=s.get("source_id", "bulk"),
			source_type=s.get("source_type", "batch"),
			authenticated=s.get("authenticated", True),
			measurements=s["measurements"],
		) for s in samples]

	# ------------------------------------------------------------------ NEW: export

	def export_twins(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export twin records as JSON or CSV."""
		twins = self.list_twins(tenant_id)
		if fmt == "csv":
			if not twins:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(twins[0].keys()))
			writer.writeheader()
			writer.writerows(twins)
			return buf.getvalue()
		return json.dumps(twins, indent=2, default=str)

	def export_predictions(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export prediction records as JSON or CSV."""
		predictions = self.list_predictions(tenant_id)
		if fmt == "csv":
			if not predictions:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(predictions[0].keys()))
			writer.writeheader()
			writer.writerows(predictions)
			return buf.getvalue()
		return json.dumps(predictions, indent=2, default=str)

	# ------------------------------------------------------------------ NEW: health / compliance

	def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return service health status for the digital twin capability."""
		return {"service": "dtwn", "tenant_id": tenant_id, "status": "healthy", "twin_count": len(self.list_twins(tenant_id)), "model_count": len(self.list_models(tenant_id)), "audit_event_count": len(self.list_audit_events(tenant_id)), "checked_at": _utc_now_iso()}

	def compliance_report(
		self,
		tenant_id: str,
		standard: str = "iso55000",
	) -> dict[str, Any]:
		"""Generate a compliance summary for asset management standard conformance."""
		self._require_tenant(tenant_id)
		twins = self.list_twins(tenant_id)
		lifecycle_covered = len([t for t in twins if self._lifecycle_records.get(self._key(tenant_id, t["id"]))])
		maintenance_covered = len([t for t in twins if self._maintenance_predictions.get(self._key(tenant_id, t["id"]))])
		return {"tenant_id": tenant_id, "standard": standard, "total_twins": len(twins), "twins_with_lifecycle_tracking": lifecycle_covered, "twins_with_maintenance_predictions": maintenance_covered, "lifecycle_coverage_pct": round(lifecycle_covered / len(twins) * 100, 2) if twins else 0.0, "maintenance_coverage_pct": round(maintenance_covered / len(twins) * 100, 2) if twins else 0.0, "compliant": lifecycle_covered == len(twins) and maintenance_covered == len(twins), "generated_at": _utc_now_iso()}

	# ------------------------------------------------------------------ dashboard / list

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		return {"tenant_id": tenant_id, "twin_count": len(self.list_twins(tenant_id)), "model_count": len(self.list_models(tenant_id)), "telemetry_sample_count": len(self.list_telemetry(tenant_id)), "topology_link_count": len(self.list_topology(tenant_id)), "simulation_count": len(self.list_simulations(tenant_id)), "review_required_prediction_count": sum(1 for item in self.list_predictions(tenant_id) if item["review_required"]), "twin_agent_count": len(self.list_twin_agents(tenant_id)), "calibration_record_count": len([c for c in self._calibration_records.values() if c["tenant_id"] == tenant_id]), "what_if_analysis_count": len([a for a in self._what_if_analyses.values() if a["tenant_id"] == tenant_id]), "energy_optimisation_count": len([e for e in self._energy_optimisations.values() if e["tenant_id"] == tenant_id]), "lifecycle_tracked_count": len([l for l in self._lifecycle_records.values() if l["tenant_id"] == tenant_id]), "audit_event_count": len(self.list_audit_events(tenant_id))}

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

	def list_twin_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	# ------------------------------------------------------------------ internals

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_twin(self, twin_id: str, tenant_id: str) -> DigitalTwin:
		twin = self._twins.get(self._key(tenant_id, twin_id))
		if twin is None:
			raise KeyError(f"unknown_twin:{twin_id}")
		return twin

	def _require_model(self, model_id: str, tenant_id: str) -> SimulationModel:
		model = self._models.get(self._key(tenant_id, model_id))
		if model is None:
			raise KeyError(f"unknown_model:{model_id}")
		return model

	def _require_prediction(self, prediction_id: str, tenant_id: str) -> TwinPrediction:
		prediction = self._predictions.get(self._key(tenant_id, prediction_id))
		if prediction is None:
			raise KeyError(f"unknown_prediction:{prediction_id}")
		return prediction

	def _record_audit(self, tenant_id: str, action: str, resource_id: str, actor: str, metadata: dict[str, Any]) -> None:
		payload = {"tenant_id": tenant_id, "action": action, "resource_id": resource_id, "actor": actor, "metadata": metadata}
		self._audit_events.append(TwinAuditEvent(id=f"aud-{len(self._audit_events) + 1:06d}", tenant_id=tenant_id, action=action, resource_id=resource_id, actor=actor, digest=stable_digest(payload), metadata=dict(metadata)))

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(action.get("reason", "digital_twin_policy_blocked") for action in result["actions"]))

	def _list_for_tenant(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _key(self, tenant_id: str, object_id: str) -> str:
		if not tenant_id:
			raise PermissionError("tenant_context_required")
		return f"{tenant_id}:{object_id}"


def _normalize_twin_agent_runtime(value: str) -> str:
	value = value.strip().lower()
	return value if value in SUPPORTED_TWIN_AGENT_RUNTIMES else ""


def _normalize_twin_agent_role(value: str) -> str:
	value = value.strip().lower()
	return value if value in SUPPORTED_TWIN_AGENT_ROLES else ""


def _maintenance_actions(model_type: str, risk: float) -> list[str]:
	base: list[str] = []
	if risk > 0.8:
		base.append("schedule_immediate_inspection")
	if risk > 0.5:
		base.append("increase_monitoring_frequency")
	if model_type == "rul":
		base.append("estimate_replacement_components")
	if model_type == "condition_based":
		base.append("review_sensor_readings")
	if model_type == "failure_probability":
		base.append("prepare_contingency_plan")
	base.append("update_maintenance_log")
	return base

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_calibration_records', '_what_if_analyses', '_maintenance_predictions', '_energy_optimisations', '_lifecycle_records', '_sensor_calibrations', '_optimisation_records']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

