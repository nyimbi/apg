"""Domain service for the APG capability registry."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_REGISTRY_AGENT_ROLES,
		SUPPORTED_REGISTRY_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (
		SUPPORTED_REGISTRY_AGENT_ROLES,
		SUPPORTED_REGISTRY_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


class CompositionRegistryService:
	"""Tenant-scoped capability catalog, dependency, and publication coordinator."""

	def __init__(self) -> None:
		self._capabilities: dict[str, dict[str, Any]] = {}
		self._dependencies: dict[str, dict[str, Any]] = {}
		self._compositions: dict[str, dict[str, Any]] = {}
		self._versions: dict[str, dict[str, Any]] = {}
		self._publications: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def register_capability(
		self,
		capability_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		category: str,
		version: str,
		provides: list[str],
		contract_ref: str,
		requires: list[str] | None = None,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_capability",
			"capability_owner_assigned": bool(owner),
			"capability_category_present": bool(category),
			"capability_version_present": bool(version),
			"capability_provides_present": bool(provides),
			"capability_contract_present": bool(contract_ref),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("registered_capability", capability_id),
			"capability_id": capability_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"category": category,
			"version": version,
			"provides": list(provides),
			"requires": list(requires or []),
			"contract_ref": contract_ref,
			"status": "registered",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._capabilities[record["id"]] = record
		self._emit("capability_registered", tenant_id, record["id"], {"capability_id": capability_id})
		return deepcopy(record)

	def add_dependency(
		self,
		dependency_id: str,
		tenant_id: str,
		source_capability_id: str,
		target_capability_id: str,
		dependency_type: str,
		version_constraint: str,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "add_dependency",
			"dependency_target_present": bool(target_capability_id),
			"dependency_type_present": bool(dependency_type),
			"version_constraint_present": bool(version_constraint),
		}
		self._enforce(context)
		self._require_capability(source_capability_id, tenant_id)
		self._require_capability(target_capability_id, tenant_id)
		record = {
			"id": self._record_id("capability_dependency", dependency_id),
			"dependency_id": dependency_id,
			"tenant_id": tenant_id,
			"source_capability_id": source_capability_id,
			"target_capability_id": target_capability_id,
			"dependency_type": dependency_type,
			"version_constraint": version_constraint,
			"status": "validated",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._dependencies[record["id"]] = record
		try:
			self._assert_no_dependency_cycle(tenant_id)
		except ValueError:
			self._dependencies.pop(record["id"], None)
			raise
		self._emit("dependency_added", tenant_id, record["id"], {"source": source_capability_id, "target": target_capability_id})
		return deepcopy(record)

	def create_composition(
		self,
		composition_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		capability_ids: list[str],
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_composition",
			"composition_owner_assigned": bool(owner),
			"composition_capabilities_present": bool(capability_ids),
		}
		self._enforce(context)
		for capability_id in capability_ids:
			self._require_capability(capability_id, tenant_id)
		validation = self.validate_composition(tenant_id, capability_ids)
		record = {
			"id": self._record_id("composition_blueprint", composition_id),
			"composition_id": composition_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"capability_ids": list(capability_ids),
			"validation": validation,
			"status": "validated" if validation["valid"] else "draft",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._compositions[record["id"]] = record
		self._emit("composition_created", tenant_id, record["id"], {"capability_count": len(capability_ids)})
		self._emit("composition_validated", tenant_id, record["id"], {"valid": validation["valid"]})
		return deepcopy(record)

	def publish_composition(
		self,
		tenant_id: str,
		composition_record_id: str,
		validation_evidence: str,
	) -> dict[str, Any]:
		record = self._require_composition(composition_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_composition",
			"validation_evidence_present": bool(validation_evidence),
		}
		self._enforce(context)
		record["status"] = "published"
		record["validation_evidence"] = validation_evidence
		record["updated_at"] = self._now()
		self._emit("composition_validated", tenant_id, composition_record_id, {"status": "published"})
		return deepcopy(record)

	def validate_composition(self, tenant_id: str, capability_ids: list[str]) -> dict[str, Any]:
		registered = {record["capability_id"] for record in self.list_capabilities(tenant_id)}
		missing = [capability_id for capability_id in capability_ids if capability_id not in registered]
		edges = [
			(dependency["source_capability_id"], dependency["target_capability_id"])
			for dependency in self.list_dependencies(tenant_id)
			if dependency["source_capability_id"] in capability_ids
		]
		unmet = [target for _, target in edges if target not in capability_ids]
		return {
			"valid": not missing and not unmet,
			"missing_capabilities": missing,
			"unmet_dependencies": sorted(set(unmet)),
			"capability_count": len(capability_ids),
			"dependency_edge_count": len(edges),
		}

	def release_version(
		self,
		release_id: str,
		tenant_id: str,
		capability_id: str,
		version: str,
		compatibility_evidence: str,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		self._require_capability(capability_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_version",
			"compatibility_evidence_present": bool(compatibility_evidence),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("capability_version", release_id),
			"release_id": release_id,
			"tenant_id": tenant_id,
			"capability_id": capability_id,
			"version": version,
			"compatibility_evidence": compatibility_evidence,
			"reviewed_by": reviewed_by,
			"status": "released",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._versions[record["id"]] = record
		self._emit("version_released", tenant_id, record["id"], {"capability_id": capability_id, "version": version})
		return deepcopy(record)

	def deprecate_capability(
		self,
		tenant_id: str,
		capability_id: str,
		migration_plan: str,
	) -> dict[str, Any]:
		record = self._require_capability(capability_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "deprecate_capability",
			"migration_plan_present": bool(migration_plan),
		}
		self._enforce(context)
		record["status"] = "deprecated"
		record["migration_plan"] = migration_plan
		record["updated_at"] = self._now()
		self._emit("capability_deprecated", tenant_id, record["id"], {"capability_id": capability_id})
		return deepcopy(record)

	def publish_to_marketplace(
		self,
		publication_id: str,
		tenant_id: str,
		capability_id: str,
		documentation_ref: str,
		reviewed_by: str,
	) -> dict[str, Any]:
		self._require_capability(capability_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_marketplace",
			"review_recorded": bool(reviewed_by),
			"documentation_present": bool(documentation_ref),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("marketplace_publication", publication_id),
			"publication_id": publication_id,
			"tenant_id": tenant_id,
			"capability_id": capability_id,
			"documentation_ref": documentation_ref,
			"reviewed_by": reviewed_by,
			"status": "prepared",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._publications[record["id"]] = record
		self._emit("marketplace_publication_prepared", tenant_id, record["id"], {"capability_id": capability_id})
		return deepcopy(record)

	def register_registry_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_registry_agent",
			"agent_runtime_supported": runtime in SUPPORTED_REGISTRY_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_REGISTRY_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("registry_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._agents[record["id"]] = record
		self._emit("registry_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_registry_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown registry agent: {agent_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_registry_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		}
		return evaluate_capability_rules(context)

	def validate_import_batch(self, tenant_id: str, record_count: int) -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "registry_import", "event_stream": "bytewax"}
		result = evaluate_capability_rules(context)
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"capability_count": len(self.list_capabilities(tenant_id)),
			"dependency_count": len(self.list_dependencies(tenant_id)),
			"composition_count": len(self.list_compositions(tenant_id)),
			"version_release_count": len(self.list_versions(tenant_id)),
			"marketplace_publication_count": len(self.list_publications(tenant_id)),
			"registry_agent_count": len(self.list_registry_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def list_capabilities(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._capabilities, tenant_id)

	def list_dependencies(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._dependencies, tenant_id)

	def list_compositions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._compositions, tenant_id)

	def list_versions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._versions, tenant_id)

	def list_publications(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._publications, tenant_id)

	def list_registry_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.register_capability(
			data.get("capability_id", data.get("id", "capability")),
			data.get("tenant_id", "default"),
			data.get("name", "Capability"),
			data.get("owner", "owner"),
			data.get("category", "composition"),
			data.get("version", "1.0.0"),
			data.get("provides", ["capability_surface"]),
			data.get("contract_ref", "capability_contract.py"),
			data.get("requires", []),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_capabilities(tenant_id)

	def _require_capability(self, capability_id: str, tenant_id: str) -> dict[str, Any]:
		for record in self._capabilities.values():
			if record["tenant_id"] == tenant_id and record["capability_id"] == capability_id:
				return record
		raise KeyError(f"Unknown capability: {capability_id}")

	def _require_composition(self, composition_record_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._compositions.get(composition_record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown composition: {composition_record_id}")
		return record

	def _assert_no_dependency_cycle(self, tenant_id: str) -> None:
		edges: dict[str, set[str]] = {}
		for dependency in self.list_dependencies(tenant_id):
			edges.setdefault(dependency["source_capability_id"], set()).add(dependency["target_capability_id"])
		visiting: set[str] = set()
		visited: set[str] = set()

		def visit(capability_id: str) -> None:
			if capability_id in visiting:
				raise ValueError(f"dependency cycle detected at {capability_id}")
			if capability_id in visited:
				return
			visiting.add(capability_id)
			for target in edges.get(capability_id, set()):
				visit(target)
			visiting.remove(capability_id)
			visited.add(capability_id)

		for capability_id in list(edges):
			visit(capability_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review":
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(record) for record in records.values() if record["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"event": event_name,
			"tenant_id": tenant_id,
			"record_id": record_id,
			"payload": deepcopy(payload),
			"processor": "bytewax",
			"stream": streaming_manifest()["stream"],
			"created_at": self._now(),
		})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


CRService = CompositionRegistryService


async def get_registry_service(*args: Any, **kwargs: Any) -> CompositionRegistryService:
	"""Return a dependency-light registry service for compatibility imports."""
	_ = args, kwargs
	return CompositionRegistryService()
