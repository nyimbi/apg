"""Executable service layer for APG ITSM CMDB."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_CI_STATUSES, SUPPORTED_CI_TYPES, SUPPORTED_DISCOVERY_METHODS,
		SUPPORTED_ENVIRONMENTS, SUPPORTED_HEALTH_STATUSES, SUPPORTED_RELATIONSHIP_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import ItCmdbCI, ItCmdbChangeRecord, ItCmdbRelationship, ItDiscoveryJob
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_CI_STATUSES, SUPPORTED_CI_TYPES, SUPPORTED_DISCOVERY_METHODS,
		SUPPORTED_ENVIRONMENTS, SUPPORTED_HEALTH_STATUSES, SUPPORTED_RELATIONSHIP_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import ItCmdbCI, ItCmdbChangeRecord, ItCmdbRelationship, ItDiscoveryJob  # type: ignore

try:
	from uuid6 import uuid7
	def _uuid7() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid
	def _uuid7() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Health scoring weights
# ---------------------------------------------------------------------------
_HEALTH_FIELD_WEIGHTS: dict[str, float] = {
	"hostname": 5.0,
	"ip_addresses": 10.0,
	"os_name": 5.0,
	"owner_id": 15.0,
	"environment": 10.0,
	"last_seen_at": 20.0,
	"manufacturer": 3.0,
	"serial_number": 3.0,
	"location": 4.0,
}
_BASE_HEALTH = 100.0 - sum(_HEALTH_FIELD_WEIGHTS.values())  # ~25


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
	return datetime.now(timezone.utc)


def _parse_iso(ts: str | None) -> datetime | None:
	if not ts:
		return None
	try:
		dt = datetime.fromisoformat(ts)
		return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
	except (ValueError, TypeError):
		return None


def _present(v: Any) -> bool:
	if v is None:
		return False
	if isinstance(v, str):
		return bool(v.strip())
	if isinstance(v, (list, dict)):
		return bool(v)
	return True


class CmdbService:
	"""Tenant-scoped CMDB runtime for APG ITSM."""

	def __init__(self) -> None:
		# Primary stores: (tenant_id, id) -> model
		self._cis: dict[tuple[str, str], ItCmdbCI] = {}
		self._relationships: dict[tuple[str, str], ItCmdbRelationship] = {}
		self._change_records: dict[tuple[str, str], ItCmdbChangeRecord] = {}
		self._discovery_jobs: dict[tuple[str, str], ItDiscoveryJob] = {}
		# Derived indexes
		self._ci_tags: dict[tuple[str, str], dict[str, str]] = {}			# (tenant, ci_id) -> tags
		self._relationship_index: dict[tuple[str, str], list[str]] = {}	# (tenant, ci_id) -> [rel_id]
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / evaluation
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# CI Registration & Lifecycle
	# ------------------------------------------------------------------

	def register_ci(
		self,
		tenant_id: str,
		name: str,
		ci_type: str,
		environment: str,
		owner_id: str,
		*,
		ci_id: str | None = None,
		hostname: str | None = None,
		ip_addresses: list[str] | None = None,
		mac_addresses: list[str] | None = None,
		serial_number: str | None = None,
		asset_tag: str | None = None,
		manufacturer: str | None = None,
		model: str | None = None,
		os_name: str | None = None,
		os_version: str | None = None,
		cpu_cores: int | None = None,
		ram_gb: float | None = None,
		disk_gb: float | None = None,
		location: str | None = None,
		datacenter: str | None = None,
		cloud_provider: str | None = None,
		cloud_region: str | None = None,
		cloud_instance_id: str | None = None,
		tags: dict[str, str] | None = None,
		custom_attributes: dict[str, Any] | None = None,
		discovery_method: str | None = None,
		discovery_job_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a new CI in the CMDB. Returns the created CI record."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "register_ci",
			"ci_type_supported": ci_type in SUPPORTED_CI_TYPES,
			"owner_present": _present(owner_id),
			"environment_present": environment in SUPPORTED_ENVIRONMENTS,
		})
		ci = ItCmdbCI(
			id=ci_id or _uuid7(),
			tenant_id=tenant_id,
			name=name,
			ci_type=ci_type,
			environment=environment,
			owner_id=owner_id,
			hostname=hostname,
			ip_addresses=ip_addresses or [],
			mac_addresses=mac_addresses or [],
			serial_number=serial_number,
			asset_tag=asset_tag,
			manufacturer=manufacturer,
			model=model,
			os_name=os_name,
			os_version=os_version,
			cpu_cores=cpu_cores,
			ram_gb=ram_gb,
			disk_gb=disk_gb,
			location=location,
			datacenter=datacenter,
			cloud_provider=cloud_provider,
			cloud_region=cloud_region,
			cloud_instance_id=cloud_instance_id,
			tags=tags or {},
			custom_attributes=custom_attributes or {},
			discovery_method=discovery_method,
			discovery_job_id=discovery_job_id,
		)
		ci.health_score = self._compute_health_score(ci)
		ci.health_status = self._health_status_from_score(ci.health_score)
		key = (tenant_id, ci.id)
		self._cis[key] = ci
		self._relationship_index.setdefault(key, [])
		self._audit(tenant_id, "ci_registered", ci.id)
		return ci.model_dump()

	def update_ci(
		self,
		tenant_id: str,
		ci_id: str,
		updated_by: str,
		fields: dict[str, Any],
	) -> dict[str, Any]:
		"""Patch mutable fields on a CI, recording a change record for each."""
		ci = self._get_ci_or_raise(tenant_id, ci_id)
		old_data = ci.model_dump()
		for field, value in fields.items():
			if hasattr(ci, field):
				setattr(ci, field, value)
		ci.updated_at = _now()
		ci.version += 1
		ci.health_score = self._compute_health_score(ci)
		ci.health_status = self._health_status_from_score(ci.health_score)
		# Record a compound change record
		change = ItCmdbChangeRecord(
			tenant_id=tenant_id,
			ci_id=ci_id,
			changed_by=updated_by,
			change_type="modify",
			diff_payload={k: {"old": old_data.get(k), "new": v} for k, v in fields.items()},
			status="applied",
			applied_at=_now(),
		)
		self._change_records[(tenant_id, change.id)] = change
		self._audit(tenant_id, "ci_updated", ci_id)
		return ci.model_dump()

	def decommission_ci(
		self,
		tenant_id: str,
		ci_id: str,
		decommissioned_by: str,
		approver_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Mark a CI as decommissioned. Requires approval."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "decommission_ci",
			"approval_present": _present(approver_id),
		})
		ci = self._get_ci_or_raise(tenant_id, ci_id)
		ci.status = "decommissioned"
		ci.decommissioned_at = _now()
		ci.updated_at = _now()
		ci.version += 1
		change = ItCmdbChangeRecord(
			tenant_id=tenant_id,
			ci_id=ci_id,
			changed_by=decommissioned_by,
			change_type="decommission",
			status="applied",
			applied_at=_now(),
			approver_id=approver_id,
			notes=reason,
		)
		self._change_records[(tenant_id, change.id)] = change
		self._audit(tenant_id, "ci_decommissioned", ci_id)
		return {"ci_id": ci_id, "status": "decommissioned", "decommissioned_at": ci.decommissioned_at}

	def get_ci(self, tenant_id: str, ci_id: str) -> dict[str, Any]:
		return self._get_ci_or_raise(tenant_id, ci_id).model_dump()

	def list_cis(
		self,
		tenant_id: str,
		*,
		ci_type: str | None = None,
		environment: str | None = None,
		status: str | None = None,
		owner_id: str | None = None,
		health_status: str | None = None,
		tags: dict[str, str] | None = None,
	) -> list[dict[str, Any]]:
		"""Filter CIs by any combination of indexed fields."""
		results: list[dict[str, Any]] = []
		for (tid, _), ci in self._cis.items():
			if tid != tenant_id:
				continue
			if ci_type and ci.ci_type != ci_type:
				continue
			if environment and ci.environment != environment:
				continue
			if status and ci.status != status:
				continue
			if owner_id and ci.owner_id != owner_id:
				continue
			if health_status and ci.health_status != health_status:
				continue
			if tags:
				if not all(ci.tags.get(k) == v for k, v in tags.items()):
					continue
			results.append(ci.model_dump())
		return results

	def search_cis(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
		"""Naive full-text search across name, hostname, ip_addresses, asset_tag."""
		q = query.lower()
		results: list[dict[str, Any]] = []
		for (tid, _), ci in self._cis.items():
			if tid != tenant_id:
				continue
			haystack = " ".join(filter(None, [
				ci.name, ci.hostname, ci.asset_tag, ci.serial_number,
				" ".join(ci.ip_addresses),
			])).lower()
			if q in haystack:
				results.append(ci.model_dump())
		return results

	# ------------------------------------------------------------------
	# Relationships
	# ------------------------------------------------------------------

	def create_relationship(
		self,
		tenant_id: str,
		source_ci_id: str,
		target_ci_id: str,
		relationship_type: str,
		*,
		description: str | None = None,
		strength: float = 1.0,
		bidirectional: bool = False,
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Create a directed (optionally bidirectional) CI relationship."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "create_relationship",
			"relationship_type_supported": relationship_type in SUPPORTED_RELATIONSHIP_TYPES,
			"source_ci_present": self._ci_exists(tenant_id, source_ci_id),
		})
		if not self._ci_exists(tenant_id, target_ci_id):
			raise KeyError(f"target CI {target_ci_id!r} not found for tenant {tenant_id!r}")
		rel = ItCmdbRelationship(
			tenant_id=tenant_id,
			source_ci_id=source_ci_id,
			target_ci_id=target_ci_id,
			relationship_type=relationship_type,
			description=description,
			strength=min(1.0, max(0.0, strength)),
			bidirectional=bidirectional,
			created_by=created_by,
			metadata=metadata or {},
		)
		self._relationships[(tenant_id, rel.id)] = rel
		# Update adjacency index
		src_key = (tenant_id, source_ci_id)
		self._relationship_index.setdefault(src_key, []).append(rel.id)
		self._audit(tenant_id, "relationship_created", rel.id)
		return rel.model_dump()

	def remove_relationship(self, tenant_id: str, rel_id: str) -> dict[str, Any]:
		key = (tenant_id, rel_id)
		rel = self._relationships.get(key)
		if rel is None:
			raise KeyError(f"relationship {rel_id!r} not found")
		del self._relationships[key]
		src_key = (tenant_id, rel.source_ci_id)
		idx = self._relationship_index.get(src_key, [])
		if rel_id in idx:
			idx.remove(rel_id)
		self._audit(tenant_id, "relationship_removed", rel_id)
		return {"rel_id": rel_id, "removed": True}

	def get_relationships(
		self,
		tenant_id: str,
		ci_id: str,
		direction: str = "outbound",
	) -> list[dict[str, Any]]:
		"""Return relationships for a CI. direction: 'outbound', 'inbound', 'both'."""
		results: list[dict[str, Any]] = []
		for (tid, _), rel in self._relationships.items():
			if tid != tenant_id:
				continue
			is_source = rel.source_ci_id == ci_id
			is_target = rel.target_ci_id == ci_id
			if direction == "outbound" and is_source:
				results.append(rel.model_dump())
			elif direction == "inbound" and is_target:
				results.append(rel.model_dump())
			elif direction == "both" and (is_source or is_target):
				results.append(rel.model_dump())
		return results

	def dependency_graph(self, tenant_id: str) -> dict[str, Any]:
		"""Return full CI dependency graph as adjacency list."""
		nodes: list[dict[str, Any]] = []
		edges: list[dict[str, Any]] = []
		for (tid, _), ci in self._cis.items():
			if tid != tenant_id:
				continue
			nodes.append({"id": ci.id, "name": ci.name, "type": ci.ci_type, "status": ci.status, "health": ci.health_score})
		for (tid, _), rel in self._relationships.items():
			if tid != tenant_id:
				continue
			edges.append({"id": rel.id, "source": rel.source_ci_id, "target": rel.target_ci_id, "type": rel.relationship_type, "strength": rel.strength})
		return {"tenant_id": tenant_id, "node_count": len(nodes), "edge_count": len(edges), "nodes": nodes, "edges": edges, "as_of": _now()}

	def impact_analysis(self, tenant_id: str, ci_id: str, depth: int = 3) -> dict[str, Any]:
		"""BFS upstream/downstream impact analysis up to `depth` hops."""
		self._get_ci_or_raise(tenant_id, ci_id)
		visited: set[str] = {ci_id}
		frontier: list[str] = [ci_id]
		impact_layers: list[list[str]] = []
		for _ in range(depth):
			next_frontier: list[str] = []
			for node in frontier:
				for (tid, _), rel in self._relationships.items():
					if tid != tenant_id:
						continue
					downstream = None
					if rel.source_ci_id == node and rel.target_ci_id not in visited:
						downstream = rel.target_ci_id
					elif rel.target_ci_id == node and rel.source_ci_id not in visited:
						downstream = rel.source_ci_id
					if downstream:
						next_frontier.append(downstream)
						visited.add(downstream)
			if not next_frontier:
				break
			impact_layers.append(list(set(next_frontier)))
			frontier = next_frontier
		impacted_cis: list[dict[str, Any]] = []
		for cid in visited - {ci_id}:
			ci = self._cis.get((tenant_id, cid))
			if ci:
				impacted_cis.append({"id": cid, "name": ci.name, "type": ci.ci_type, "status": ci.status})
		return {
			"tenant_id": tenant_id,
			"source_ci_id": ci_id,
			"depth": depth,
			"impacted_count": len(impacted_cis),
			"impacted_cis": impacted_cis,
			"impact_layers": impact_layers,
			"as_of": _now(),
		}

	# ------------------------------------------------------------------
	# Discovery Jobs
	# ------------------------------------------------------------------

	def create_discovery_job(
		self,
		tenant_id: str,
		name: str,
		discovery_method: str,
		target: str,
		environment: str,
		*,
		schedule_cron: str | None = None,
		credentials_ref: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "create_discovery_job",
			"discovery_method_supported": discovery_method in SUPPORTED_DISCOVERY_METHODS,
		})
		job = ItDiscoveryJob(
			tenant_id=tenant_id,
			name=name,
			discovery_method=discovery_method,
			target=target,
			environment=environment,
			schedule_cron=schedule_cron,
			credentials_ref=credentials_ref,
			created_by=created_by,
		)
		self._discovery_jobs[(tenant_id, job.id)] = job
		self._audit(tenant_id, "discovery_job_created", job.id)
		return job.model_dump()

	def start_discovery_job(self, tenant_id: str, job_id: str) -> dict[str, Any]:
		job = self._get_job_or_raise(tenant_id, job_id)
		job.status = "running"
		job.started_at = _now()
		job.last_run_at = _now()
		job.run_count += 1
		self._audit(tenant_id, "discovery_job_started", job_id)
		return job.model_dump()

	def complete_discovery_job(
		self,
		tenant_id: str,
		job_id: str,
		ci_discovered: int,
		ci_updated: int,
		ci_decommissioned: int,
		result_summary: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		job = self._get_job_or_raise(tenant_id, job_id)
		job.status = "completed"
		job.completed_at = _now()
		job.ci_discovered = ci_discovered
		job.ci_updated = ci_updated
		job.ci_decommissioned = ci_decommissioned
		job.result_summary = result_summary or {}
		self._audit(tenant_id, "discovery_job_completed", job_id)
		return job.model_dump()

	def fail_discovery_job(self, tenant_id: str, job_id: str, error_message: str) -> dict[str, Any]:
		job = self._get_job_or_raise(tenant_id, job_id)
		job.status = "failed"
		job.completed_at = _now()
		job.error_message = error_message
		self._audit(tenant_id, "discovery_job_failed", job_id)
		return job.model_dump()

	def list_discovery_jobs(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		results: list[dict[str, Any]] = []
		for (tid, _), job in self._discovery_jobs.items():
			if tid != tenant_id:
				continue
			if status and job.status != status:
				continue
			results.append(job.model_dump())
		return results

	# ------------------------------------------------------------------
	# Change Tracking
	# ------------------------------------------------------------------

	def record_ci_change(
		self,
		tenant_id: str,
		ci_id: str,
		changed_by: str,
		change_type: str,
		*,
		change_ticket_id: str | None = None,
		field_name: str | None = None,
		old_value: str | None = None,
		new_value: str | None = None,
		diff_payload: dict[str, Any] | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "record_ci_change",
			"ci_present": self._ci_exists(tenant_id, ci_id),
		})
		change = ItCmdbChangeRecord(
			tenant_id=tenant_id,
			ci_id=ci_id,
			change_ticket_id=change_ticket_id,
			changed_by=changed_by,
			change_type=change_type,
			field_name=field_name,
			old_value=old_value,
			new_value=new_value,
			diff_payload=diff_payload or {},
			notes=notes,
		)
		self._change_records[(tenant_id, change.id)] = change
		self._audit(tenant_id, "change_record_created", change.id)
		return change.model_dump()

	def approve_ci_change(self, tenant_id: str, change_id: str, approver_id: str) -> dict[str, Any]:
		change = self._change_records.get((tenant_id, change_id))
		if change is None:
			raise KeyError(f"change record {change_id!r} not found")
		change.status = "approved"
		change.approver_id = approver_id
		change.approved_at = _now()
		self._audit(tenant_id, "change_record_approved", change_id)
		return change.model_dump()

	def apply_ci_change(self, tenant_id: str, change_id: str) -> dict[str, Any]:
		change = self._change_records.get((tenant_id, change_id))
		if change is None:
			raise KeyError(f"change record {change_id!r} not found")
		if change.status not in ("approved", "pending"):
			raise ValueError(f"cannot apply change in status {change.status!r}")
		change.status = "applied"
		change.applied_at = _now()
		self._audit(tenant_id, "change_record_applied", change_id)
		return change.model_dump()

	def rollback_ci_change(self, tenant_id: str, change_id: str, reason: str) -> dict[str, Any]:
		change = self._change_records.get((tenant_id, change_id))
		if change is None:
			raise KeyError(f"change record {change_id!r} not found")
		change.status = "rolled_back"
		change.rollback_reason = reason
		self._audit(tenant_id, "change_record_rolled_back", change_id)
		return change.model_dump()

	def get_ci_change_history(self, tenant_id: str, ci_id: str) -> list[dict[str, Any]]:
		return [
			rec.model_dump() for (tid, _), rec in self._change_records.items()
			if tid == tenant_id and rec.ci_id == ci_id
		]

	# ------------------------------------------------------------------
	# Health Scoring
	# ------------------------------------------------------------------

	def compute_health_score(self, tenant_id: str, ci_id: str) -> dict[str, Any]:
		"""Re-compute and persist health score for a CI."""
		ci = self._get_ci_or_raise(tenant_id, ci_id)
		score = self._compute_health_score(ci)
		status = self._health_status_from_score(score)
		ci.health_score = score
		ci.health_status = status
		self._audit(tenant_id, "health_score_updated", ci_id)
		return {"ci_id": ci_id, "health_score": score, "health_status": status, "as_of": _now()}

	def health_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Portfolio health overview: counts by status and average score."""
		status_counts: dict[str, int] = {s: 0 for s in SUPPORTED_HEALTH_STATUSES}
		score_sum = 0.0
		total = 0
		for (tid, _), ci in self._cis.items():
			if tid != tenant_id:
				continue
			if ci.status == "decommissioned":
				continue
			status_counts[ci.health_status] = status_counts.get(ci.health_status, 0) + 1
			score_sum += ci.health_score
			total += 1
		avg_score = round(score_sum / total, 2) if total else 0.0
		return {
			"tenant_id": tenant_id,
			"total_active_cis": total,
			"average_health_score": avg_score,
			"by_status": status_counts,
			"as_of": _now(),
		}

	def _compute_health_score(self, ci: ItCmdbCI) -> float:
		score = _BASE_HEALTH
		for field, weight in _HEALTH_FIELD_WEIGHTS.items():
			val = getattr(ci, field, None)
			if _present(val):
				score += weight
		# Penalty: last_seen stale (>24h for production)
		if ci.environment == "production" and ci.last_seen_at:
			last_dt = _parse_iso(ci.last_seen_at)
			if last_dt:
				age_h = (_now_dt() - last_dt).total_seconds() / 3600.0
				if age_h > 24:
					score -= min(30.0, age_h / 24.0 * 10.0)
		return round(max(0.0, min(100.0, score)), 2)

	def _health_status_from_score(self, score: float) -> str:
		if score >= 80:
			return "healthy"
		if score >= 50:
			return "degraded"
		if score > 0:
			return "critical"
		return "unknown"

	# ------------------------------------------------------------------
	# Analytics & Reporting
	# ------------------------------------------------------------------

	def cmdb_summary(self, tenant_id: str) -> dict[str, Any]:
		ci_by_type: dict[str, int] = {}
		ci_by_env: dict[str, int] = {}
		ci_by_status: dict[str, int] = {}
		total = 0
		for (tid, _), ci in self._cis.items():
			if tid != tenant_id:
				continue
			total += 1
			ci_by_type[ci.ci_type] = ci_by_type.get(ci.ci_type, 0) + 1
			ci_by_env[ci.environment] = ci_by_env.get(ci.environment, 0) + 1
			ci_by_status[ci.status] = ci_by_status.get(ci.status, 0) + 1
		return {
			"tenant_id": tenant_id,
			"total_cis": total,
			"by_type": ci_by_type,
			"by_environment": ci_by_env,
			"by_status": ci_by_status,
			"total_relationships": sum(1 for (t, _) in self._relationships if t == tenant_id),
			"total_discovery_jobs": sum(1 for (t, _) in self._discovery_jobs if t == tenant_id),
			"total_change_records": sum(1 for (t, _) in self._change_records if t == tenant_id),
			"as_of": _now(),
		}

	def orphan_ci_report(self, tenant_id: str) -> dict[str, Any]:
		"""CIs with no relationships — potential orphans."""
		orphans: list[dict[str, Any]] = []
		for (tid, cid), ci in self._cis.items():
			if tid != tenant_id:
				continue
			if ci.status == "decommissioned":
				continue
			rels = self._relationship_index.get((tid, cid), [])
			# Also check inbound
			inbound = any(
				r.target_ci_id == cid
				for (t, _), r in self._relationships.items()
				if t == tenant_id
			)
			if not rels and not inbound:
				orphans.append({"id": cid, "name": ci.name, "type": ci.ci_type, "owner_id": ci.owner_id})
		return {"tenant_id": tenant_id, "orphan_count": len(orphans), "orphans": orphans, "as_of": _now()}

	def stale_ci_report(self, tenant_id: str, stale_hours: int = 72) -> dict[str, Any]:
		"""CIs not seen by discovery in stale_hours for production."""
		stale: list[dict[str, Any]] = []
		for (tid, _), ci in self._cis.items():
			if tid != tenant_id or ci.status != "active":
				continue
			if not ci.last_seen_at:
				stale.append({"id": ci.id, "name": ci.name, "last_seen_at": None, "age_hours": None})
				continue
			dt = _parse_iso(ci.last_seen_at)
			if dt:
				age_h = (_now_dt() - dt).total_seconds() / 3600.0
				if age_h > stale_hours:
					stale.append({"id": ci.id, "name": ci.name, "last_seen_at": ci.last_seen_at, "age_hours": round(age_h, 1)})
		return {"tenant_id": tenant_id, "stale_threshold_hours": stale_hours, "stale_count": len(stale), "stale_cis": stale, "as_of": _now()}

	def export_cmdb(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export all active CIs and relationships as JSON or CSV."""
		cis = [ci.model_dump() for (tid, _), ci in self._cis.items() if tid == tenant_id]
		rels = [r.model_dump() for (tid, _), r in self._relationships.items() if tid == tenant_id]
		if format == "json":
			data = json.dumps({"cis": cis, "relationships": rels}, indent=2, default=str)
		else:
			raise ValueError(f"unsupported format {format!r}")
		return {"tenant_id": tenant_id, "format": format, "ci_count": len(cis), "relationship_count": len(rels), "data": data}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _get_ci_or_raise(self, tenant_id: str, ci_id: str) -> ItCmdbCI:
		ci = self._cis.get((tenant_id, ci_id))
		if ci is None:
			raise KeyError(f"CI {ci_id!r} not found for tenant {tenant_id!r}")
		return ci

	def _ci_exists(self, tenant_id: str, ci_id: str) -> bool:
		return (tenant_id, ci_id) in self._cis

	def _get_job_or_raise(self, tenant_id: str, job_id: str) -> ItDiscoveryJob:
		job = self._discovery_jobs.get((tenant_id, job_id))
		if job is None:
			raise KeyError(f"discovery job {job_id!r} not found for tenant {tenant_id!r}")
		return job

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"ts": _now(),
			"processor": "bytewax",
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "cmdb_policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "cmdb_policy_denied")


ItsmCmdbService = CmdbService
