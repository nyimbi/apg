"""APG MLOps Pipeline service — experiment tracking, feature store, model registry, drift detection."""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

from .models import (
	MlrExperiment, MlrRun, MlrFeatureView, MlrFeatureVector,
	MlrRegisteredModel, MlrModelVersion, MlrAbTest, MlrDriftReport,
	RunStatus, ModelStage, DriftStatus,
)

_log = logging.getLogger(__name__)


class MlrService:
	"""MLOps pipeline: experiment tracking, feature store, model registry, drift detection."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._experiments: dict[str, MlrExperiment] = {}
		self._runs: dict[str, MlrRun] = {}
		self._feature_views: dict[str, MlrFeatureView] = {}
		self._feature_vectors: list[MlrFeatureVector] = []
		self._registered_models: dict[str, MlrRegisteredModel] = {}
		self._model_versions: dict[str, MlrModelVersion] = {}
		self._ab_tests: dict[str, MlrAbTest] = {}
		self._drift_reports: list[MlrDriftReport] = []
		self._feature_cache = BoundedCache(max_size=10000)

	# ── Experiment Tracking ──────────────────────────────────────────────────

	async def create_experiment(
		self,
		name: str,
		description: str = "",
		tags: list[str] | None = None,
		tenant_id: str | None = None,
	) -> MlrExperiment:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		guard_non_empty_string(name, "name")
		exp = MlrExperiment(tenant_id=tid, name=name, description=description, tags=tags or [])
		self._experiments[exp.id] = exp
		_log.info("Created experiment '%s' (%s)", name, exp.id)
		return exp

	async def start_run(
		self,
		experiment_id: str,
		run_name: str = "",
		params: dict[str, str] | None = None,
		tags: dict[str, str] | None = None,
		tenant_id: str | None = None,
	) -> MlrRun:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		run = MlrRun(
			tenant_id=tid,
			experiment_id=experiment_id,
			run_name=run_name,
			params=params or {},
			tags=tags or {},
		)
		self._runs[run.id] = run
		return run

	async def log_metrics(
		self,
		run_id: str,
		metrics: dict[str, float],
		tenant_id: str | None = None,
	) -> MlrRun:
		run = self._runs.get(run_id)
		assert run is not None, f"Run {run_id} not found"
		run.metrics.update(metrics)
		return run

	async def log_params(self, run_id: str, params: dict[str, str], tenant_id: str | None = None) -> MlrRun:
		run = self._runs.get(run_id)
		assert run is not None, f"Run {run_id} not found"
		run.params.update(params)
		return run

	async def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED, tenant_id: str | None = None) -> MlrRun:
		run = self._runs.get(run_id)
		assert run is not None, f"Run {run_id} not found"
		run.status = status
		run.end_time = datetime.now(timezone.utc)
		return run

	async def list_runs(
		self,
		experiment_id: str,
		tenant_id: str | None = None,
	) -> list[MlrRun]:
		tid = tenant_id or self._tenant_id
		return [r for r in self._runs.values() if r.experiment_id == experiment_id and r.tenant_id == tid]

	async def compare_runs(
		self,
		run_ids: list[str],
		metric: str,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return sorted comparison of runs by the given metric."""
		results = []
		for run_id in run_ids:
			run = self._runs.get(run_id)
			if run and metric in run.metrics:
				results.append({"run_id": run_id, "run_name": run.run_name, metric: run.metrics[metric], "params": run.params})
		return sorted(results, key=lambda x: x.get(metric, 0), reverse=True)

	# ── Feature Store ────────────────────────────────────────────────────────

	async def create_feature_view(
		self,
		name: str,
		entities: list[str],
		features: list[dict[str, Any]],
		source_table: str = "",
		ttl_minutes: int = 60,
		tenant_id: str | None = None,
	) -> MlrFeatureView:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		guard_non_empty_string(name, "name")
		fv = MlrFeatureView(
			tenant_id=tid, name=name, entities=entities,
			features=features, source_table=source_table, ttl_minutes=ttl_minutes,
		)
		self._feature_views[f"{tid}:{name}"] = fv
		return fv

	async def get_online_features(
		self,
		feature_view: str,
		entity_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Serve features from online store (with cache). O(1) cache hit."""
		cache_key = f"{tenant_id}:{feature_view}:{entity_id}"
		cached = self._feature_cache.get(cache_key)
		if cached is not None:
			return cached
		# In production: query Redis/Postgres online store
		# Here: return empty dict (no online store connected)
		return {}

	async def get_historical_features(
		self,
		feature_view: str,
		entity_ids: list[str],
		as_of: datetime,
		tenant_id: str | None = None,
	) -> list[MlrFeatureVector]:
		"""Point-in-time correct feature retrieval for training datasets."""
		tid = tenant_id or self._tenant_id
		return [
			v for v in self._feature_vectors
			if v.feature_view == feature_view
			and v.entity_id in entity_ids
			and v.event_timestamp <= as_of
			and v.feature_view.split(":")[0] == tid
		]

	async def materialize_features(
		self,
		feature_view: str,
		start: datetime,
		end: datetime,
		tenant_id: str | None = None,
	) -> int:
		"""Trigger materialization of features for a time range. Returns rows materialized."""
		_log.info("Materializing %s from %s to %s", feature_view, start, end)
		return 0  # Production: trigger batch materialization job

	# ── Model Registry ───────────────────────────────────────────────────────

	async def register_model(
		self,
		name: str,
		description: str = "",
		tenant_id: str | None = None,
	) -> MlrRegisteredModel:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		model = MlrRegisteredModel(tenant_id=tid, name=name, description=description)
		self._registered_models[f"{tid}:{name}"] = model
		return model

	async def create_model_version(
		self,
		model_name: str,
		source_run_id: str,
		artifact_path: str = "",
		description: str = "",
		tenant_id: str | None = None,
	) -> MlrModelVersion:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		existing = [v for v in self._model_versions.values() if v.model_name == model_name and v.tenant_id == tid]
		version_number = len(existing) + 1
		mv = MlrModelVersion(
			tenant_id=tid, model_name=model_name, version=version_number,
			source_run_id=source_run_id, artifact_path=artifact_path, description=description,
		)
		self._model_versions[mv.id] = mv
		return mv

	async def promote_model(
		self,
		version_id: str,
		stage: ModelStage,
		approved_by: str,
		approval_notes: str = "",
		tenant_id: str | None = None,
	) -> MlrModelVersion:
		"""Promote a model version to a new stage. Requires human approval for production."""
		mv = self._model_versions.get(version_id)
		assert mv is not None, f"Model version {version_id} not found"
		assert mv.stage != ModelStage.PRODUCTION or mv.approved_by is not None, "Production promotion requires prior approval"
		mv.stage = stage
		mv.approved_by = approved_by
		mv.approval_notes = approval_notes
		mv.promoted_at = datetime.now(timezone.utc)
		_log.info("Model %s v%d promoted to %s by %s", mv.model_name, mv.version, stage, approved_by)
		return mv

	async def get_production_model(self, model_name: str, tenant_id: str | None = None) -> MlrModelVersion | None:
		tid = tenant_id or self._tenant_id
		prod_versions = [
			v for v in self._model_versions.values()
			if v.model_name == model_name and v.tenant_id == tid and v.stage == ModelStage.PRODUCTION
		]
		return max(prod_versions, key=lambda v: v.version) if prod_versions else None

	# ── A/B Testing ──────────────────────────────────────────────────────────

	async def create_ab_test(
		self,
		name: str,
		control_version_id: str,
		treatment_version_id: str,
		traffic_split_pct: float = 20.0,
		metrics_to_compare: list[str] | None = None,
		tenant_id: str | None = None,
	) -> MlrAbTest:
		tid = tenant_id or self._tenant_id
		test = MlrAbTest(
			tenant_id=tid, name=name,
			control_model_version_id=control_version_id,
			treatment_model_version_id=treatment_version_id,
			traffic_split_pct=traffic_split_pct,
			metrics_to_compare=metrics_to_compare or ["accuracy", "latency_p99"],
		)
		self._ab_tests[test.id] = test
		return test

	async def declare_winner(
		self,
		test_id: str,
		winner: str,
		tenant_id: str | None = None,
	) -> MlrAbTest:
		test = self._ab_tests.get(test_id)
		assert test is not None, f"A/B test {test_id} not found"
		test.winner = winner
		test.status = "completed"
		test.ended_at = datetime.now(timezone.utc)
		return test

	# ── Drift Detection ──────────────────────────────────────────────────────

	async def check_drift(
		self,
		model_version_id: str,
		feature_view_name: str,
		reference_data: list[dict[str, float]],
		current_data: list[dict[str, float]],
		psi_threshold: float = 0.2,
		tenant_id: str | None = None,
	) -> MlrDriftReport:
		"""Compute Population Stability Index (PSI) for each feature."""
		tid = tenant_id or self._tenant_id
		psi_scores: dict[str, float] = {}
		drifted: list[str] = []

		for feature in (reference_data[0].keys() if reference_data else []):
			ref_vals = [r.get(feature, 0) for r in reference_data]
			cur_vals = [r.get(feature, 0) for r in current_data]
			psi = self._compute_psi(ref_vals, cur_vals)
			psi_scores[feature] = psi
			if psi > psi_threshold:
				drifted.append(feature)

		status = (
			DriftStatus.CRITICAL if any(v > psi_threshold * 2 for v in psi_scores.values())
			else DriftStatus.WARNING if drifted
			else DriftStatus.OK
		)
		report = MlrDriftReport(
			tenant_id=tid,
			model_version_id=model_version_id,
			feature_view_name=feature_view_name,
			status=status,
			psi_scores=psi_scores,
			drifted_features=drifted,
			retraining_recommended=len(drifted) > 0,
			samples_checked=len(current_data),
		)
		self._drift_reports.append(report)
		if report.retraining_recommended:
			_log.warning("Drift detected in %s — retraining recommended (drifted: %s)", feature_view_name, drifted)
		return report

	async def on_inference_for_drift_monitoring(self, event: dict[str, Any]) -> None:
		"""NATS handler: receive inference completed events to monitor for drift."""
		_log.debug("Received inference event for drift monitoring: %s", event.get("resource_id"))

	@staticmethod
	def _compute_psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
		"""Population Stability Index — PSI > 0.2 = significant drift."""
		if not expected or not actual:
			return 0.0
		all_vals = expected + actual
		min_v, max_v = min(all_vals), max(all_vals)
		if min_v == max_v:
			return 0.0
		edges = [min_v + i * (max_v - min_v) / bins for i in range(bins + 1)]

		def bucket_pct(vals: list[float]) -> list[float]:
			counts = [0] * bins
			for v in vals:
				idx = min(int((v - min_v) / (max_v - min_v) * bins), bins - 1)
				counts[idx] += 1
			total = len(vals)
			return [max(c / total, 1e-6) for c in counts]

		exp_pct = bucket_pct(expected)
		act_pct = bucket_pct(actual)
		return sum((a - e) * math.log(a / e) for a, e in zip(act_pct, exp_pct))

	async def list_drift_reports(self, model_version_id: str | None = None, tenant_id: str | None = None) -> list[MlrDriftReport]:
		tid = tenant_id or self._tenant_id
		return [r for r in self._drift_reports if r.tenant_id == tid and (model_version_id is None or r.model_version_id == model_version_id)]
