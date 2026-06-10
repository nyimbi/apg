"""Process Mining service — BPMN inference from NATS events, conformance checking, bottleneck analysis, variant discovery."""
from __future__ import annotations

import asyncio
import collections
import logging
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "pmin"
SUPPORTED_ALGORITHMS = {"alpha_miner", "heuristics_miner", "inductive_miner", "directly_follows"}


class ProcessMiningService:
	"""Infer BPMN process models from NATS event streams, conformance checking, bottleneck analysis, variant discovery."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.event_logs: dict[str, dict[str, Any]] = {}
		self.raw_events: dict[str, list[dict[str, Any]]] = {}  # event_log_id -> events
		self.bpmn_models: dict[str, dict[str, Any]] = {}
		self.conformance_results: dict[str, dict[str, Any]] = {}
		self.bottleneck_reports: dict[str, dict[str, Any]] = {}
		self.variant_analyses: dict[str, dict[str, Any]] = {}
		self.simulations: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, payload: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"payload": payload or {},
			"created_at": self._now(),
		})

	# ── Health / describe ────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "pmin",
			"status": "healthy",
			"event_log_count": len(self.event_logs),
			"bpmn_model_count": len(self.bpmn_models),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"tenant_id": tenant,
			"supported_algorithms": sorted(SUPPORTED_ALGORITHMS),
			"features": [
				"bpmn_discovery", "conformance_checking", "bottleneck_analysis",
				"variant_discovery", "nats_event_stream_ingestion", "dfg_analysis",
				"process_simulation", "case_filtering", "performance_analysis"
			],
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Event log management ──────────────────────────────────────

	async def create_event_log(
		self,
		tenant_id: str,
		name: str,
		description: str = "",
		subject_filter: str = "",
		case_id_field: str = "case_id",
		activity_field: str = "activity",
		timestamp_field: str = "timestamp",
		resource_field: str = "resource",
	) -> dict[str, Any]:
		"""Create an event log definition (backed by NATS subject filter)."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		record: dict[str, Any] = {
			"id": self._id("elog"),
			"tenant_id": tenant,
			"name": name,
			"description": description,
			"subject_filter": subject_filter,
			"case_id_field": case_id_field,
			"activity_field": activity_field,
			"timestamp_field": timestamp_field,
			"resource_field": resource_field,
			"event_count": 0,
			"case_count": 0,
			"status": "active",
			"created_at": self._now(),
		}
		self.event_logs[record["id"]] = record
		self.raw_events[record["id"]] = []
		self._emit(tenant, "event_log_created", record["id"], {"name": name, "subject_filter": subject_filter})
		_log.info("event log created: %s tenant=%s", record["id"], tenant)
		return deepcopy(record)

	async def get_event_log(self, tenant_id: str, log_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")
		return deepcopy(log)

	async def list_event_logs(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(l) for l in self.event_logs.values() if l["tenant_id"] == tenant]

	async def delete_event_log(self, tenant_id: str, log_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")
		del self.event_logs[log_id]
		self.raw_events.pop(log_id, None)
		self._emit(tenant, "event_log_deleted", log_id)
		return deepcopy(log)

	async def update_event_log(self, tenant_id: str, log_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")
		for key in ("name", "description", "subject_filter", "case_id_field", "activity_field"):
			if key in kwargs and kwargs[key] is not None:
				log[key] = kwargs[key]
		self._emit(tenant, "event_log_updated", log_id)
		return deepcopy(log)

	# ── Event ingestion ───────────────────────────────────────────

	async def ingest_events(
		self,
		tenant_id: str,
		log_id: str,
		events: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Ingest a batch of process events into an event log."""
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")

		case_id_field = log["case_id_field"]
		activity_field = log["activity_field"]
		timestamp_field = log["timestamp_field"]
		resource_field = log["resource_field"]

		ingested = []
		skipped = 0
		for raw_event in events:
			case_id = raw_event.get(case_id_field) or raw_event.get("case_id")
			activity = raw_event.get(activity_field) or raw_event.get("activity")
			if not case_id or not activity:
				skipped += 1
				continue
			event: dict[str, Any] = {
				"id": self._id("ev"),
				"log_id": log_id,
				"case_id": str(case_id),
				"activity": str(activity),
				"timestamp": raw_event.get(timestamp_field) or raw_event.get("timestamp") or self._now(),
				"resource": raw_event.get(resource_field) or raw_event.get("resource", ""),
				"attributes": {k: v for k, v in raw_event.items() if k not in (case_id_field, activity_field, timestamp_field, resource_field)},
			}
			self.raw_events[log_id].append(event)
			ingested.append(event)

		# Update log stats
		all_events = self.raw_events[log_id]
		log["event_count"] = len(all_events)
		log["case_count"] = len({e["case_id"] for e in all_events})

		self._emit(tenant, "events_ingested", log_id, {
			"ingested": len(ingested), "skipped": skipped
		})
		return {
			"log_id": log_id,
			"ingested": len(ingested),
			"skipped": skipped,
			"total_events": log["event_count"],
			"total_cases": log["case_count"],
		}

	async def ingest_nats_events(
		self,
		tenant_id: str,
		log_id: str,
		nats_messages: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Ingest events from NATS message payloads."""
		tenant = self._tenant(tenant_id)
		# Extract payload from NATS message wrapper
		events = []
		for msg in nats_messages:
			payload = msg.get("data") or msg.get("payload") or msg
			if isinstance(payload, dict):
				events.append(payload)
		return await self.ingest_events(tenant_id, log_id, events)

	async def get_events(
		self,
		tenant_id: str,
		log_id: str,
		case_id: str | None = None,
		activity: str | None = None,
		limit: int = 1000,
	) -> list[dict[str, Any]]:
		"""Query events from an event log."""
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")
		events = deepcopy(self.raw_events.get(log_id, []))
		if case_id:
			events = [e for e in events if e["case_id"] == case_id]
		if activity:
			events = [e for e in events if e["activity"] == activity]
		return sorted(events, key=lambda e: e["timestamp"])[:limit]

	# ── Process discovery ─────────────────────────────────────────

	async def discover_process_model(
		self,
		tenant_id: str,
		log_id: str,
		algorithm: str = "alpha_miner",
		noise_threshold: float = 0.2,
	) -> dict[str, Any]:
		"""Discover a BPMN process model from an event log."""
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")
		if algorithm not in SUPPORTED_ALGORITHMS:
			raise ValueError(f"algorithm must be one of {sorted(SUPPORTED_ALGORITHMS)}")

		events = self.raw_events.get(log_id, [])
		if not events:
			raise ValueError(f"event log has no events: {log_id}")

		# Build directly-follows graph
		dfg = self._build_dfg(events)

		# Apply noise threshold: remove edges with frequency < threshold * max_freq
		if dfg["edges"]:
			max_freq = max(e["frequency"] for e in dfg["edges"])
			min_freq = max_freq * noise_threshold
			dfg["edges"] = [e for e in dfg["edges"] if e["frequency"] >= min_freq]

		# Extract BPMN nodes and edges from DFG
		nodes = []
		for activity in dfg["activities"]:
			node_type = "task"
			if activity in dfg["start_activities"]:
				node_type = "start_event"
			elif activity in dfg["end_activities"]:
				node_type = "end_event"
			nodes.append({
				"id": f"node_{activity.replace(' ', '_').lower()}",
				"name": activity,
				"type": node_type,
				"frequency": dfg["activity_frequencies"].get(activity, 0),
			})

		edges = [
			{
				"source": f"node_{e['source'].replace(' ', '_').lower()}",
				"target": f"node_{e['target'].replace(' ', '_').lower()}",
				"label": e["source"] + " → " + e["target"],
				"frequency": e["frequency"],
				"avg_duration_s": e.get("avg_duration_s", 0.0),
			}
			for e in dfg["edges"]
		]

		model: dict[str, Any] = {
			"id": self._id("bpmn"),
			"tenant_id": tenant,
			"event_log_id": log_id,
			"algorithm": algorithm,
			"noise_threshold": noise_threshold,
			"nodes": nodes,
			"edges": edges,
			"start_activities": dfg["start_activities"],
			"end_activities": dfg["end_activities"],
			"activity_count": len(dfg["activities"]),
			"edge_count": len(edges),
			"discovered_at": self._now(),
		}
		self.bpmn_models[model["id"]] = model
		self._emit(tenant, "bpmn_model_discovered", model["id"], {
			"log_id": log_id, "algorithm": algorithm, "nodes": len(nodes)
		})
		_log.info("BPMN model discovered: %s log=%s algo=%s nodes=%d", model["id"], log_id, algorithm, len(nodes))
		return deepcopy(model)

	def _build_dfg(self, events: list[dict[str, Any]]) -> dict[str, Any]:
		"""Build a Directly-Follows Graph from event log."""
		# Group events by case
		cases: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
		for event in events:
			cases[event["case_id"]].append(event)

		# Sort each case by timestamp
		for case_id in cases:
			cases[case_id].sort(key=lambda e: e["timestamp"])

		activity_frequencies: dict[str, int] = collections.Counter()
		edge_frequencies: dict[tuple[str, str], int] = collections.Counter()
		edge_durations: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
		start_activities: set[str] = set()
		end_activities: set[str] = set()

		for case_id, case_events in cases.items():
			if not case_events:
				continue
			start_activities.add(case_events[0]["activity"])
			end_activities.add(case_events[-1]["activity"])
			for event in case_events:
				activity_frequencies[event["activity"]] += 1
			for i in range(len(case_events) - 1):
				src = case_events[i]["activity"]
				tgt = case_events[i + 1]["activity"]
				edge_frequencies[(src, tgt)] += 1
				# Compute duration between consecutive events
				try:
					t1 = datetime.fromisoformat(case_events[i]["timestamp"].replace("Z", "+00:00"))
					t2 = datetime.fromisoformat(case_events[i + 1]["timestamp"].replace("Z", "+00:00"))
					duration_s = abs((t2 - t1).total_seconds())
					edge_durations[(src, tgt)].append(duration_s)
				except Exception:
					pass

		edges = [
			{
				"source": src,
				"target": tgt,
				"frequency": freq,
				"avg_duration_s": round(statistics.mean(edge_durations[(src, tgt)]), 2) if edge_durations[(src, tgt)] else 0.0,
			}
			for (src, tgt), freq in edge_frequencies.items()
		]

		return {
			"activities": set(activity_frequencies.keys()),
			"activity_frequencies": dict(activity_frequencies),
			"edges": edges,
			"start_activities": sorted(start_activities),
			"end_activities": sorted(end_activities),
			"case_count": len(cases),
		}

	async def get_bpmn_model(self, tenant_id: str, model_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		model = self.bpmn_models.get(model_id)
		if not model or model["tenant_id"] != tenant:
			raise KeyError(f"model not found: {model_id}")
		return deepcopy(model)

	async def list_bpmn_models(self, tenant_id: str, log_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.bpmn_models.values() if m["tenant_id"] == tenant]
		if log_id:
			items = [m for m in items if m["event_log_id"] == log_id]
		return items

	async def delete_bpmn_model(self, tenant_id: str, model_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		model = self.bpmn_models.get(model_id)
		if not model or model["tenant_id"] != tenant:
			raise KeyError(f"model not found: {model_id}")
		del self.bpmn_models[model_id]
		self._emit(tenant, "bpmn_model_deleted", model_id)
		return deepcopy(model)

	async def export_bpmn_xml(self, tenant_id: str, model_id: str) -> dict[str, Any]:
		"""Export BPMN model as BPMN 2.0 XML (simplified)."""
		tenant = self._tenant(tenant_id)
		model = await self.get_bpmn_model(tenant_id, model_id)
		xml_parts = [
			'<?xml version="1.0" encoding="UTF-8"?>',
			'<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" id="apg_bpmn">',
			'  <process id="discovered_process" isExecutable="false">',
		]
		for node in model["nodes"]:
			tag = "startEvent" if node["type"] == "start_event" else ("endEvent" if node["type"] == "end_event" else "task")
			xml_parts.append(f'    <{tag} id="{node["id"]}" name="{node["name"]}"/>')
		for i, edge in enumerate(model["edges"]):
			xml_parts.append(f'    <sequenceFlow id="sf{i}" sourceRef="{edge["source"]}" targetRef="{edge["target"]}"/>')
		xml_parts.extend(['  </process>', '</definitions>'])
		return {
			"model_id": model_id,
			"format": "bpmn2_xml",
			"xml": "\n".join(xml_parts),
			"exported_at": self._now(),
		}

	# ── Conformance checking ──────────────────────────────────────

	async def check_conformance(
		self,
		tenant_id: str,
		log_id: str,
		model_id: str,
	) -> dict[str, Any]:
		"""Check how well the event log conforms to the discovered BPMN model."""
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")
		model = self.bpmn_models.get(model_id)
		if not model or model["tenant_id"] != tenant:
			raise KeyError(f"model not found: {model_id}")

		events = self.raw_events.get(log_id, [])
		model_edges: set[tuple[str, str]] = {(e["source"], e["target"]) for e in model["edges"]}
		model_activities: set[str] = {n["name"] for n in model["nodes"]}

		# Group by case
		cases: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
		for event in events:
			cases[event["case_id"]].append(event)

		conforming = 0
		deviating_cases = []
		for case_id, case_events in cases.items():
			case_events.sort(key=lambda e: e["timestamp"])
			is_conforming = True
			for event in case_events:
				if event["activity"] not in model_activities:
					is_conforming = False
					break
			for i in range(len(case_events) - 1):
				src = case_events[i]["activity"]
				tgt = case_events[i + 1]["activity"]
				# Simplified: check that edge (src, tgt) exists in model
				src_id = f"node_{src.replace(' ', '_').lower()}"
				tgt_id = f"node_{tgt.replace(' ', '_').lower()}"
				if (src_id, tgt_id) not in model_edges:
					is_conforming = False
					break
			if is_conforming:
				conforming += 1
			else:
				deviating_cases.append(case_id)

		total_cases = len(cases)
		fitness = conforming / total_cases if total_cases > 0 else 1.0
		# Simplified precision/generalization
		model_edge_count = len(model_edges)
		log_edge_count = len({
			(f"node_{e['source'].replace(' ', '_').lower()}", f"node_{e['target'].replace(' ', '_').lower()}")
			for case_events in cases.values()
			for e in [{"source": case_events[i]["activity"], "target": case_events[i+1]["activity"]}
					  for i in range(len(case_events) - 1)]
		}) if cases else 0
		precision = min(1.0, log_edge_count / model_edge_count) if model_edge_count > 0 else 1.0
		generalization = fitness  # simplified
		simplicity = max(0.0, 1.0 - (len(model["edges"]) / max(len(model["nodes"]) * 2, 1)))

		result: dict[str, Any] = {
			"id": self._id("conf"),
			"tenant_id": tenant,
			"event_log_id": log_id,
			"model_id": model_id,
			"fitness": round(fitness, 4),
			"precision": round(precision, 4),
			"generalization": round(generalization, 4),
			"simplicity": round(simplicity, 4),
			"total_cases": total_cases,
			"conforming_cases": conforming,
			"deviating_cases": deviating_cases[:50],
			"deviation_rate": round(len(deviating_cases) / total_cases, 4) if total_cases > 0 else 0.0,
			"checked_at": self._now(),
		}
		self.conformance_results[result["id"]] = result
		self._emit(tenant, "conformance_checked", result["id"], {
			"fitness": fitness, "deviating_cases": len(deviating_cases)
		})
		return deepcopy(result)

	async def list_conformance_results(self, tenant_id: str, log_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.conformance_results.values() if r["tenant_id"] == tenant]
		if log_id:
			items = [r for r in items if r["event_log_id"] == log_id]
		return sorted(items, key=lambda r: r["checked_at"], reverse=True)

	# ── Bottleneck analysis ───────────────────────────────────────

	async def analyze_bottlenecks(
		self,
		tenant_id: str,
		log_id: str,
		top_n: int = 10,
	) -> dict[str, Any]:
		"""Identify process bottlenecks by analysing waiting times between activities."""
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")

		events = self.raw_events.get(log_id, [])
		if not events:
			raise ValueError(f"no events in log: {log_id}")

		# Group by case and sort
		cases: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
		for event in events:
			cases[event["case_id"]].append(event)

		edge_durations: dict[str, list[float]] = collections.defaultdict(list)
		for case_events in cases.values():
			case_events.sort(key=lambda e: e["timestamp"])
			for i in range(len(case_events) - 1):
				src = case_events[i]["activity"]
				tgt = case_events[i + 1]["activity"]
				edge_key = f"{src} → {tgt}"
				try:
					t1 = datetime.fromisoformat(case_events[i]["timestamp"].replace("Z", "+00:00"))
					t2 = datetime.fromisoformat(case_events[i + 1]["timestamp"].replace("Z", "+00:00"))
					edge_durations[edge_key].append(abs((t2 - t1).total_seconds()))
				except Exception:
					edge_durations[edge_key].append(0.0)

		bottlenecks = []
		for edge_key, durations in edge_durations.items():
			if not durations:
				continue
			bottlenecks.append({
				"transition": edge_key,
				"frequency": len(durations),
				"avg_wait_s": round(statistics.mean(durations), 2),
				"median_wait_s": round(statistics.median(durations), 2),
				"max_wait_s": round(max(durations), 2),
				"p95_wait_s": round(sorted(durations)[int(len(durations) * 0.95)], 2) if len(durations) >= 20 else round(max(durations), 2),
			})

		bottlenecks.sort(key=lambda b: b["avg_wait_s"], reverse=True)
		top_bottlenecks = bottlenecks[:top_n]

		report: dict[str, Any] = {
			"id": self._id("bnk"),
			"tenant_id": tenant,
			"event_log_id": log_id,
			"total_transitions_analysed": len(edge_durations),
			"bottlenecks": top_bottlenecks,
			"worst_transition": top_bottlenecks[0]["transition"] if top_bottlenecks else None,
			"worst_avg_wait_s": top_bottlenecks[0]["avg_wait_s"] if top_bottlenecks else 0.0,
			"generated_at": self._now(),
		}
		self.bottleneck_reports[report["id"]] = report
		self._emit(tenant, "bottleneck_analysis_completed", report["id"], {
			"bottleneck_count": len(top_bottlenecks)
		})
		return deepcopy(report)

	async def get_bottleneck_report(self, tenant_id: str, report_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		report = self.bottleneck_reports.get(report_id)
		if not report or report["tenant_id"] != tenant:
			raise KeyError(f"report not found: {report_id}")
		return deepcopy(report)

	async def list_bottleneck_reports(self, tenant_id: str, log_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.bottleneck_reports.values() if r["tenant_id"] == tenant]
		if log_id:
			items = [r for r in items if r["event_log_id"] == log_id]
		return items

	# ── Variant discovery ─────────────────────────────────────────

	async def discover_variants(
		self,
		tenant_id: str,
		log_id: str,
		top_n: int = 20,
	) -> dict[str, Any]:
		"""Discover process variants (unique activity sequences) and their frequencies."""
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")

		events = self.raw_events.get(log_id, [])
		if not events:
			raise ValueError(f"no events in log: {log_id}")

		# Group by case
		cases: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
		for event in events:
			cases[event["case_id"]].append(event)

		variant_map: dict[str, list[str]] = collections.defaultdict(list)
		for case_id, case_events in cases.items():
			case_events.sort(key=lambda e: e["timestamp"])
			variant_key = " → ".join(e["activity"] for e in case_events)
			variant_map[variant_key].append(case_id)

		total_cases = len(cases)
		variants = [
			{
				"variant": variant,
				"case_count": len(case_ids),
				"frequency": round(len(case_ids) / total_cases, 4),
				"sample_cases": case_ids[:5],
				"step_count": len(variant.split(" → ")),
			}
			for variant, case_ids in variant_map.items()
		]
		variants.sort(key=lambda v: v["case_count"], reverse=True)

		analysis: dict[str, Any] = {
			"id": self._id("var"),
			"tenant_id": tenant,
			"event_log_id": log_id,
			"total_cases": total_cases,
			"total_variants": len(variants),
			"top_variants": variants[:top_n],
			"dominant_variant_frequency": variants[0]["frequency"] if variants else 0.0,
			"happy_path": variants[0]["variant"] if variants else None,
			"discovered_at": self._now(),
		}
		self.variant_analyses[analysis["id"]] = analysis
		self._emit(tenant, "variants_discovered", analysis["id"], {
			"variant_count": len(variants), "case_count": total_cases
		})
		return deepcopy(analysis)

	async def get_variant_analysis(self, tenant_id: str, analysis_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		analysis = self.variant_analyses.get(analysis_id)
		if not analysis or analysis["tenant_id"] != tenant:
			raise KeyError(f"variant analysis not found: {analysis_id}")
		return deepcopy(analysis)

	async def list_variant_analyses(self, tenant_id: str, log_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.variant_analyses.values() if a["tenant_id"] == tenant]
		if log_id:
			items = [a for a in items if a["event_log_id"] == log_id]
		return items

	# ── Case analysis ─────────────────────────────────────────────

	async def get_case_trace(self, tenant_id: str, log_id: str, case_id: str) -> dict[str, Any]:
		"""Return the complete trace for a specific case."""
		tenant = self._tenant(tenant_id)
		events = await self.get_events(tenant_id, log_id, case_id=case_id)
		if not events:
			raise KeyError(f"case not found: {case_id}")
		total_duration_s = 0.0
		if len(events) >= 2:
			try:
				t_start = datetime.fromisoformat(events[0]["timestamp"].replace("Z", "+00:00"))
				t_end = datetime.fromisoformat(events[-1]["timestamp"].replace("Z", "+00:00"))
				total_duration_s = abs((t_end - t_start).total_seconds())
			except Exception:
				pass
		return {
			"case_id": case_id,
			"log_id": log_id,
			"event_count": len(events),
			"activities": [e["activity"] for e in events],
			"trace": " → ".join(e["activity"] for e in events),
			"start_time": events[0]["timestamp"] if events else None,
			"end_time": events[-1]["timestamp"] if events else None,
			"total_duration_s": round(total_duration_s, 2),
			"events": events,
		}

	async def filter_deviating_cases(self, tenant_id: str, log_id: str, model_id: str) -> list[dict[str, Any]]:
		"""Return list of deviating case traces from last conformance check."""
		tenant = self._tenant(tenant_id)
		# Find latest conformance result for this log+model
		results = [
			r for r in self.conformance_results.values()
			if r["tenant_id"] == tenant and r["event_log_id"] == log_id and r["model_id"] == model_id
		]
		if not results:
			raise KeyError("no conformance results found for this log+model combination")
		latest = max(results, key=lambda r: r["checked_at"])
		traces = []
		for case_id in latest["deviating_cases"]:
			try:
				trace = await self.get_case_trace(tenant_id, log_id, case_id)
				traces.append(trace)
			except Exception as exc:
				_log.error("get_case_trace failed for %s: %s", case_id, exc)
		return traces

	# ── Process simulation ────────────────────────────────────────

	async def simulate_process(
		self,
		tenant_id: str,
		model_id: str,
		simulation_cases: int = 100,
	) -> dict[str, Any]:
		"""Run a Monte Carlo simulation on the discovered process model."""
		tenant = self._tenant(tenant_id)
		model = self.bpmn_models.get(model_id)
		if not model or model["tenant_id"] != tenant:
			raise KeyError(f"model not found: {model_id}")
		if simulation_cases > 10000:
			raise ValueError("simulation_cases max 10000")

		# Simulate random walks through the model graph
		import random
		rng = random.Random(42)  # deterministic seed
		adjacency: dict[str, list[str]] = collections.defaultdict(list)
		for edge in model["edges"]:
			adjacency[edge["source"]].append(edge["target"])

		start_nodes = [
			n["id"] for n in model["nodes"]
			if n["name"] in model["start_activities"] or n["type"] == "start_event"
		]
		end_nodes = {
			n["id"] for n in model["nodes"]
			if n["name"] in model["end_activities"] or n["type"] == "end_event"
		}

		if not start_nodes:
			start_nodes = [model["nodes"][0]["id"]] if model["nodes"] else []

		completed = 0
		incomplete = 0
		trace_lengths = []
		for _ in range(simulation_cases):
			if not start_nodes:
				incomplete += 1
				continue
			current = rng.choice(start_nodes)
			trace_length = 1
			max_steps = 50
			for _ in range(max_steps):
				if current in end_nodes:
					completed += 1
					trace_lengths.append(trace_length)
					break
				next_nodes = adjacency.get(current, [])
				if not next_nodes:
					incomplete += 1
					break
				current = rng.choice(next_nodes)
				trace_length += 1
			else:
				incomplete += 1

		simulation: dict[str, Any] = {
			"id": self._id("sim"),
			"tenant_id": tenant,
			"model_id": model_id,
			"simulation_cases": simulation_cases,
			"completed": completed,
			"incomplete": incomplete,
			"completion_rate": round(completed / simulation_cases, 4),
			"avg_trace_length": round(statistics.mean(trace_lengths), 2) if trace_lengths else 0.0,
			"median_trace_length": round(statistics.median(trace_lengths), 2) if trace_lengths else 0.0,
			"simulated_at": self._now(),
		}
		self.simulations[simulation["id"]] = simulation
		self._emit(tenant, "simulation_completed", simulation["id"], {
			"completion_rate": simulation["completion_rate"]
		})
		return deepcopy(simulation)

	# ── Statistics ────────────────────────────────────────────────

	async def process_mining_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Return tenant-level process mining dashboard."""
		tenant = self._tenant(tenant_id)
		logs = [l for l in self.event_logs.values() if l["tenant_id"] == tenant]
		models = [m for m in self.bpmn_models.values() if m["tenant_id"] == tenant]
		conf_results = [r for r in self.conformance_results.values() if r["tenant_id"] == tenant]
		avg_fitness = round(statistics.mean(r["fitness"] for r in conf_results), 4) if conf_results else None
		return {
			"tenant_id": tenant,
			"event_logs": len(logs),
			"total_events": sum(l["event_count"] for l in logs),
			"total_cases": sum(l["case_count"] for l in logs),
			"bpmn_models": len(models),
			"conformance_checks": len(conf_results),
			"avg_fitness": avg_fitness,
			"bottleneck_reports": len([r for r in self.bottleneck_reports.values() if r["tenant_id"] == tenant]),
			"variant_analyses": len([a for a in self.variant_analyses.values() if a["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def get_performance_metrics(self, tenant_id: str, log_id: str) -> dict[str, Any]:
		"""Compute per-activity performance statistics (throughput times, waiting times)."""
		tenant = self._tenant(tenant_id)
		log = self.event_logs.get(log_id)
		if not log or log["tenant_id"] != tenant:
			raise KeyError(f"event log not found: {log_id}")

		events = self.raw_events.get(log_id, [])
		activity_times: dict[str, list[float]] = collections.defaultdict(list)

		cases: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
		for event in events:
			cases[event["case_id"]].append(event)

		for case_events in cases.values():
			case_events.sort(key=lambda e: e["timestamp"])
			for i in range(len(case_events) - 1):
				src_activity = case_events[i]["activity"]
				try:
					t1 = datetime.fromisoformat(case_events[i]["timestamp"].replace("Z", "+00:00"))
					t2 = datetime.fromisoformat(case_events[i + 1]["timestamp"].replace("Z", "+00:00"))
					activity_times[src_activity].append(abs((t2 - t1).total_seconds()))
				except Exception:
					pass

		metrics = {}
		for activity, times in activity_times.items():
			if times:
				metrics[activity] = {
					"count": len(times),
					"avg_s": round(statistics.mean(times), 2),
					"median_s": round(statistics.median(times), 2),
					"min_s": round(min(times), 2),
					"max_s": round(max(times), 2),
					"stdev_s": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
				}

		return {
			"log_id": log_id,
			"activity_performance": metrics,
			"activity_count": len(metrics),
			"generated_at": self._now(),
		}
