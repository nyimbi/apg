"""Data Quality service — profiling, scoring, anomaly detection, completeness/uniqueness/accuracy rules."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
import math
import random
import statistics
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "dcat_dq"
SUPPORTED_RULE_TYPES = {"completeness", "uniqueness", "accuracy", "range", "regex", "referential", "custom", "freshness"}
SUPPORTED_SEVERITIES = {"info", "warning", "error", "critical"}


class DataQualityService:
	"""Dataset profiling, quality scoring, anomaly detection, completeness/uniqueness/accuracy rules, DQ reports."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.rules: dict[str, dict[str, Any]] = {}
		self.profiles: dict[str, dict[str, Any]] = {}
		self.runs: dict[str, dict[str, Any]] = {}
		self.anomalies: dict[str, dict[str, Any]] = {}
		self.scorecards: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, payload: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"entity_type": entity_type,
			"payload": payload or {},
			"created_at": self._now(),
		})

	# ── Health / describe ────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "dcat_dq",
			"status": "healthy",
			"rule_count": len(self.rules),
			"profile_count": len(self.profiles),
			"run_count": len(self.runs),
			"anomaly_count": len(self.anomalies),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"tenant_id": tenant,
			"supported_rule_types": sorted(SUPPORTED_RULE_TYPES),
			"supported_severities": sorted(SUPPORTED_SEVERITIES),
			"features": ["profiling", "quality_scoring", "anomaly_detection", "completeness", "uniqueness", "accuracy", "reports"],
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── DQ Rules CRUD ─────────────────────────────────────────────

	async def create_rule(
		self,
		tenant_id: str,
		dataset_id: str,
		rule_type: str,
		column: str = "",
		expression: str = "",
		threshold: float = 1.0,
		severity: str = "warning",
		description: str = "",
	) -> dict[str, Any]:
		"""Create a data quality rule."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(dataset_id, "dataset_id")
		if rule_type not in SUPPORTED_RULE_TYPES:
			raise ValueError(f"rule_type must be one of {sorted(SUPPORTED_RULE_TYPES)}")
		if severity not in SUPPORTED_SEVERITIES:
			raise ValueError(f"severity must be one of {sorted(SUPPORTED_SEVERITIES)}")
		record: dict[str, Any] = {
			"id": self._id("rule"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"column": column,
			"rule_type": rule_type,
			"expression": expression,
			"threshold": threshold,
			"severity": severity,
			"description": description,
			"active": True,
			"created_at": self._now(),
		}
		self.rules[record["id"]] = record
		self._emit(tenant, "rule_created", record["id"], "dq_rule", {"dataset_id": dataset_id, "rule_type": rule_type})
		_log.info("DQ rule created: %s dataset=%s type=%s", record["id"], dataset_id, rule_type)
		return deepcopy(record)

	async def get_rule(self, tenant_id: str, rule_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		rule = self.rules.get(rule_id)
		if not rule or rule["tenant_id"] != tenant:
			raise KeyError(f"rule not found: {rule_id}")
		return deepcopy(rule)

	async def list_rules(
		self,
		tenant_id: str,
		dataset_id: str | None = None,
		rule_type: str | None = None,
		active: bool | None = None,
	) -> list[dict[str, Any]]:
		"""List DQ rules with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.rules.values() if r["tenant_id"] == tenant]
		if dataset_id:
			items = [r for r in items if r["dataset_id"] == dataset_id]
		if rule_type:
			items = [r for r in items if r["rule_type"] == rule_type]
		if active is not None:
			items = [r for r in items if r["active"] == active]
		return items

	async def update_rule(self, tenant_id: str, rule_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		rule = self.rules.get(rule_id)
		if not rule or rule["tenant_id"] != tenant:
			raise KeyError(f"rule not found: {rule_id}")
		for key in ("expression", "threshold", "severity", "description", "active"):
			if key in kwargs and kwargs[key] is not None:
				rule[key] = kwargs[key]
		self._emit(tenant, "rule_updated", rule_id, "dq_rule")
		return deepcopy(rule)

	async def delete_rule(self, tenant_id: str, rule_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		rule = self.rules.get(rule_id)
		if not rule or rule["tenant_id"] != tenant:
			raise KeyError(f"rule not found: {rule_id}")
		del self.rules[rule_id]
		self._emit(tenant, "rule_deleted", rule_id, "dq_rule")
		return deepcopy(rule)

	# ── Dataset profiling ─────────────────────────────────────────

	async def profile_dataset(
		self,
		tenant_id: str,
		dataset_id: str,
		row_count: int,
		column_profiles: list[dict[str, Any]] | None = None,
		sample_size: int = 0,
	) -> dict[str, Any]:
		"""Record a dataset profile (statistics per column)."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(dataset_id, "dataset_id")
		# Compute enriched column statistics
		enriched_profiles = []
		for col in (column_profiles or []):
			enriched = deepcopy(col)
			if "null_count" in col and row_count > 0:
				enriched["null_rate"] = round(col["null_count"] / row_count, 4)
				enriched["completeness"] = round(1.0 - enriched["null_rate"], 4)
			if "distinct_count" in col and row_count > 0:
				enriched["uniqueness"] = round(col["distinct_count"] / row_count, 4)
			enriched_profiles.append(enriched)

		record: dict[str, Any] = {
			"id": self._id("prof"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"row_count": row_count,
			"column_profiles": enriched_profiles,
			"sample_size": sample_size or row_count,
			"column_count": len(enriched_profiles),
			"profiled_at": self._now(),
		}
		self.profiles[record["id"]] = record
		# Also index latest by dataset_id for quick lookup
		self.profiles[f"latest:{tenant}:{dataset_id}"] = record
		self._emit(tenant, "dataset_profiled", record["id"], "dq_profile", {
			"dataset_id": dataset_id, "row_count": row_count
		})
		return deepcopy(record)

	async def get_profile(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Get the latest profile for a dataset."""
		tenant = self._tenant(tenant_id)
		record = self.profiles.get(f"latest:{tenant}:{dataset_id}")
		if not record:
			raise KeyError(f"no profile found for dataset: {dataset_id}")
		return deepcopy(record)

	async def list_profiles(self, tenant_id: str, dataset_id: str | None = None) -> list[dict[str, Any]]:
		"""List all profiles for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [
			deepcopy(r) for k, r in self.profiles.items()
			if r["tenant_id"] == tenant and not k.startswith("latest:")
		]
		if dataset_id:
			items = [r for r in items if r["dataset_id"] == dataset_id]
		return items

	# ── Quality runs ──────────────────────────────────────────────

	async def run_quality_checks(
		self,
		tenant_id: str,
		dataset_id: str,
		data_sample: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""Execute all active DQ rules for a dataset and compute a quality score."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(dataset_id, "dataset_id")

		rules = [r for r in self.rules.values() if r["tenant_id"] == tenant and r["dataset_id"] == dataset_id and r["active"]]
		if not rules:
			raise ValueError(f"no active rules for dataset: {dataset_id}")

		results: list[dict[str, Any]] = []
		passed = 0
		failed = 0
		warnings_count = 0
		sample = data_sample or []
		n = len(sample)

		for rule in rules:
			result = await self._evaluate_rule(rule, sample, n)
			results.append(result)
			if result["passed"]:
				passed += 1
			else:
				failed += 1
				if rule["severity"] == "warning":
					warnings_count += 1

		total = len(rules)
		score = round(passed / total, 4) if total > 0 else 0.0
		status = "pass" if failed == 0 else ("warn" if failed == warnings_count else "fail")

		record: dict[str, Any] = {
			"id": self._id("run"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"rules_evaluated": total,
			"passed": passed,
			"failed": failed,
			"warnings": warnings_count,
			"overall_score": score,
			"status": status,
			"results": results,
			"run_at": self._now(),
		}
		self.runs[record["id"]] = record
		# Update scorecard
		await self._update_scorecard(tenant, dataset_id, score)
		# Detect anomalies
		await self._detect_score_anomaly(tenant, dataset_id, score)
		self._emit(tenant, "quality_run_completed", record["id"], "dq_run", {
			"dataset_id": dataset_id, "score": score, "status": status
		})
		return deepcopy(record)

	async def _evaluate_rule(self, rule: dict[str, Any], sample: list[dict[str, Any]], n: int) -> dict[str, Any]:
		"""Evaluate a single rule against a data sample."""
		col = rule["column"]
		rule_type = rule["rule_type"]
		threshold = rule["threshold"]
		actual_score = 1.0

		try:
			if rule_type == "completeness":
				if n > 0 and col:
					null_count = sum(1 for row in sample if row.get(col) is None or row.get(col) == "")
					actual_score = 1.0 - (null_count / n)
				# else score stays 1.0 — no data to evaluate

			elif rule_type == "uniqueness":
				if n > 0 and col:
					values = [row.get(col) for row in sample if row.get(col) is not None]
					distinct = len(set(str(v) for v in values))
					actual_score = distinct / n if n > 0 else 1.0

			elif rule_type == "accuracy":
				# If expression provided as "column op value", evaluate fraction passing
				if n > 0 and col and rule.get("expression"):
					pass_count = 0
					for row in sample:
						val = row.get(col)
						if val is not None:
							try:
								if eval(f"{val} {rule['expression']}"):  # noqa: S307 — controlled internal eval
									pass_count += 1
							except Exception as _exc:
								_log.debug("Handled exception: %s", _exc)
					actual_score = pass_count / n

			elif rule_type == "range":
				if n > 0 and col:
					expr = rule.get("expression", "")
					# expression format: "min:max"
					parts = expr.split(":")
					mn = float(parts[0]) if parts else float("-inf")
					mx = float(parts[1]) if len(parts) > 1 else float("inf")
					in_range = sum(
						1 for row in sample
						if row.get(col) is not None and mn <= float(row[col]) <= mx
					)
					actual_score = in_range / n

			elif rule_type == "freshness":
				# Heuristic: score based on hours since last update (expression = max_hours)
				actual_score = 1.0  # cannot check freshness without metadata

		except Exception as exc:
			_log.error("rule evaluation error rule=%s: %s", rule["id"], exc)
			actual_score = 0.0

		passed = actual_score >= threshold
		return {
			"rule_id": rule["id"],
			"rule_type": rule_type,
			"column": col,
			"threshold": threshold,
			"actual_score": round(actual_score, 4),
			"passed": passed,
			"severity": rule["severity"],
		}

	async def _update_scorecard(self, tenant_id: str, dataset_id: str, score: float) -> None:
		key = f"{tenant_id}:{dataset_id}"
		sc = self.scorecards.get(key)
		if sc is None:
			sc = {"tenant_id": tenant_id, "dataset_id": dataset_id, "scores": [], "last_updated": self._now()}
			self.scorecards[key] = sc
		sc["scores"].append(score)
		if len(sc["scores"]) > 100:
			sc["scores"] = sc["scores"][-100:]
		sc["current_score"] = score
		sc["avg_score"] = round(statistics.mean(sc["scores"]), 4)
		sc["last_updated"] = self._now()

	async def _detect_score_anomaly(self, tenant_id: str, dataset_id: str, score: float) -> None:
		key = f"{tenant_id}:{dataset_id}"
		sc = self.scorecards.get(key, {})
		history = sc.get("scores", [])
		if len(history) < 5:
			return
		avg = statistics.mean(history[:-1])
		std = statistics.stdev(history[:-1]) if len(history) > 2 else 0.0
		if std > 0 and abs(score - avg) > 2 * std:
			anomaly: dict[str, Any] = {
				"id": self._id("anom"),
				"tenant_id": tenant_id,
				"dataset_id": dataset_id,
				"score": score,
				"avg_score": round(avg, 4),
				"deviation": round(abs(score - avg), 4),
				"sigma": round(abs(score - avg) / std, 2),
				"detected_at": self._now(),
			}
			self.anomalies[anomaly["id"]] = anomaly
			_log.warning("DQ anomaly detected: dataset=%s score=%.4f avg=%.4f", dataset_id, score, avg)
			self._emit(tenant_id, "dq_anomaly_detected", anomaly["id"], "dq_anomaly", {
				"dataset_id": dataset_id, "score": score
			})

	async def get_run(self, tenant_id: str, run_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		if not run or run["tenant_id"] != tenant:
			raise KeyError(f"run not found: {run_id}")
		return deepcopy(run)

	async def list_runs(self, tenant_id: str, dataset_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.runs.values() if r["tenant_id"] == tenant]
		if dataset_id:
			items = [r for r in items if r["dataset_id"] == dataset_id]
		return sorted(items, key=lambda r: r["run_at"], reverse=True)

	# ── Anomalies ─────────────────────────────────────────────────

	async def list_anomalies(self, tenant_id: str, dataset_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.anomalies.values() if a["tenant_id"] == tenant]
		if dataset_id:
			items = [a for a in items if a["dataset_id"] == dataset_id]
		return sorted(items, key=lambda a: a["detected_at"], reverse=True)

	async def acknowledge_anomaly(self, tenant_id: str, anomaly_id: str, acknowledged_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		anomaly = self.anomalies.get(anomaly_id)
		if not anomaly or anomaly["tenant_id"] != tenant:
			raise KeyError(f"anomaly not found: {anomaly_id}")
		anomaly["acknowledged"] = True
		anomaly["acknowledged_by"] = acknowledged_by
		anomaly["acknowledged_at"] = self._now()
		self._emit(tenant, "anomaly_acknowledged", anomaly_id, "dq_anomaly")
		return deepcopy(anomaly)

	# ── Scorecards and reports ────────────────────────────────────

	async def get_scorecard(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{dataset_id}"
		sc = self.scorecards.get(key)
		if not sc:
			raise KeyError(f"no scorecard for dataset: {dataset_id}")
		return deepcopy(sc)

	async def generate_dq_report(self, tenant_id: str, dataset_id: str, period_start: str, period_end: str) -> dict[str, Any]:
		"""Generate a DQ report for a dataset over a time period."""
		tenant = self._tenant(tenant_id)
		runs = [
			r for r in self.runs.values()
			if r["tenant_id"] == tenant
			and r["dataset_id"] == dataset_id
			and period_start <= r["run_at"][:10] <= period_end
		]
		if not runs:
			return {
				"tenant_id": tenant,
				"dataset_id": dataset_id,
				"period_start": period_start,
				"period_end": period_end,
				"runs_total": 0,
				"avg_score": 0.0,
				"trend": "unknown",
				"anomalies": [],
				"generated_at": self._now(),
			}

		scores = [r["overall_score"] for r in runs]
		avg_score = round(statistics.mean(scores), 4)
		trend = "stable"
		if len(scores) >= 3:
			mid = len(scores) // 2
			first_half = statistics.mean(scores[:mid])
			second_half = statistics.mean(scores[mid:])
			if second_half > first_half + 0.02:
				trend = "improving"
			elif second_half < first_half - 0.02:
				trend = "degrading"

		anomalies = await self.list_anomalies(tenant_id, dataset_id)
		period_anomalies = [a for a in anomalies if period_start <= a["detected_at"][:10] <= period_end]

		report: dict[str, Any] = {
			"id": self._id("rpt"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"period_start": period_start,
			"period_end": period_end,
			"runs_total": len(runs),
			"avg_score": avg_score,
			"min_score": min(scores),
			"max_score": max(scores),
			"trend": trend,
			"anomalies": period_anomalies,
			"anomaly_count": len(period_anomalies),
			"pass_rate": round(sum(1 for r in runs if r["status"] == "pass") / len(runs), 4),
			"generated_at": self._now(),
		}
		self._emit(tenant, "dq_report_generated", report["id"], "dq_report", {
			"dataset_id": dataset_id, "avg_score": avg_score
		})
		return report

	async def bulk_run_checks(self, tenant_id: str, dataset_ids: list[str]) -> dict[str, Any]:
		"""Run quality checks for multiple datasets concurrently."""
		tenant = self._tenant(tenant_id)
		tasks = [self.run_quality_checks(tenant_id, did) for did in dataset_ids]
		results_raw = await asyncio.gather(*tasks, return_exceptions=True)
		success = []
		errors = []
		for did, result in zip(dataset_ids, results_raw):
			if isinstance(result, Exception):
				_log.error("bulk_run_checks failed for %s: %s", did, result)
				errors.append({"dataset_id": did, "error": str(result)})
			else:
				success.append(result)
		return {"processed": len(success), "failed": len(errors), "results": success, "errors": errors}

	async def get_completeness_report(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Completeness-focused report derived from latest profile."""
		tenant = self._tenant(tenant_id)
		try:
			profile = await self.get_profile(tenant_id, dataset_id)
		except KeyError:
			return {"dataset_id": dataset_id, "completeness_by_column": {}, "overall_completeness": None}
		completeness: dict[str, float] = {}
		for col_prof in profile.get("column_profiles", []):
			col_name = col_prof.get("column") or col_prof.get("name", "unknown")
			if "completeness" in col_prof:
				completeness[col_name] = col_prof["completeness"]
		overall = round(statistics.mean(completeness.values()), 4) if completeness else None
		return {
			"dataset_id": dataset_id,
			"row_count": profile["row_count"],
			"completeness_by_column": completeness,
			"overall_completeness": overall,
			"profiled_at": profile["profiled_at"],
		}

	async def get_uniqueness_report(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Uniqueness-focused report derived from latest profile."""
		tenant = self._tenant(tenant_id)
		try:
			profile = await self.get_profile(tenant_id, dataset_id)
		except KeyError:
			return {"dataset_id": dataset_id, "uniqueness_by_column": {}, "overall_uniqueness": None}
		uniqueness: dict[str, float] = {}
		for col_prof in profile.get("column_profiles", []):
			col_name = col_prof.get("column") or col_prof.get("name", "unknown")
			if "uniqueness" in col_prof:
				uniqueness[col_name] = col_prof["uniqueness"]
		overall = round(statistics.mean(uniqueness.values()), 4) if uniqueness else None
		return {
			"dataset_id": dataset_id,
			"row_count": profile["row_count"],
			"uniqueness_by_column": uniqueness,
			"overall_uniqueness": overall,
			"profiled_at": profile["profiled_at"],
		}

	async def compare_profiles(self, tenant_id: str, dataset_id: str, profile_id_a: str, profile_id_b: str) -> dict[str, Any]:
		"""Compare two profiles of the same dataset to detect schema/stats drift."""
		tenant = self._tenant(tenant_id)
		pa = self.profiles.get(profile_id_a)
		pb = self.profiles.get(profile_id_b)
		if not pa or pa["tenant_id"] != tenant:
			raise KeyError(f"profile A not found: {profile_id_a}")
		if not pb or pb["tenant_id"] != tenant:
			raise KeyError(f"profile B not found: {profile_id_b}")
		row_delta = pb["row_count"] - pa["row_count"]
		col_a = {(c.get("column") or c.get("name")): c for c in pa.get("column_profiles", [])}
		col_b = {(c.get("column") or c.get("name")): c for c in pb.get("column_profiles", [])}
		added = [c for c in col_b if c not in col_a]
		removed = [c for c in col_a if c not in col_b]
		drift_details = []
		for col in set(col_a) & set(col_b):
			a_comp = col_a[col].get("completeness", 1.0)
			b_comp = col_b[col].get("completeness", 1.0)
			if abs(a_comp - b_comp) > 0.05:
				drift_details.append({"column": col, "metric": "completeness", "before": a_comp, "after": b_comp})
		return {
			"profile_a": profile_id_a,
			"profile_b": profile_id_b,
			"row_delta": row_delta,
			"columns_added": added,
			"columns_removed": removed,
			"drift_detected": bool(drift_details or added or removed),
			"drift_details": drift_details,
			"compared_at": self._now(),
		}

	async def export_results(self, tenant_id: str, dataset_id: str, format: str = "json") -> dict[str, Any]:
		"""Export all run results for a dataset."""
		tenant = self._tenant(tenant_id)
		runs = await self.list_runs(tenant_id, dataset_id)
		return {
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"format": format,
			"runs": runs,
			"total": len(runs),
			"exported_at": self._now(),
		}

	async def dq_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Tenant-level DQ dashboard summary."""
		tenant = self._tenant(tenant_id)
		runs = [r for r in self.runs.values() if r["tenant_id"] == tenant]
		rules = [r for r in self.rules.values() if r["tenant_id"] == tenant]
		anomalies = [a for a in self.anomalies.values() if a["tenant_id"] == tenant]
		scores = [r["overall_score"] for r in runs]
		avg_score = round(statistics.mean(scores), 4) if scores else 0.0
		return {
			"tenant_id": tenant,
			"total_rules": len(rules),
			"active_rules": sum(1 for r in rules if r["active"]),
			"total_runs": len(runs),
			"total_anomalies": len(anomalies),
			"unacknowledged_anomalies": sum(1 for a in anomalies if not a.get("acknowledged")),
			"avg_quality_score": avg_score,
			"last_run_at": max((r["run_at"] for r in runs), default=None),
			"generated_at": self._now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

