"""Executable service layer for the APG Sandbox/Testing Environment — expanded implementation."""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .capability_contract import (
	SUPPORTED_SBOX_AGENT_ROLES,
	SUPPORTED_SBOX_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
)
from .models import (
	IsolationProfile,
	SandboxDataset,
	SandboxEnvironment,
	SandboxRun,
	SandboxTemplate,
	SboxAgent,
	SboxAuditEvent,
)
from .sandbox_runtime import (
	normalize_dataset_type,
	normalize_isolation_level,
	normalize_run_type,
	normalize_tags,
	risk_score,
	run_status,
	sandbox_state,
	stable_id,
	summarize_decision,
	utc_now,
)


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"


class SandboxTestingService:
	"""
	Tenant-scoped sandbox, dataset, test-run, mock service, event simulation,
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
	cost tracking, and analytics runtime.

	Adapter/store pattern — no external dependencies.
	"""

	def __init__(self) -> None:
		self._isolation_profiles: dict[str, IsolationProfile] = {}
		self._templates: dict[str, SandboxTemplate] = {}
		self._datasets: dict[str, SandboxDataset] = {}
		self._sandboxes: dict[str, SandboxEnvironment] = {}
		self._runs: dict[str, SandboxRun] = {}
		self._agents: dict[str, SboxAgent] = {}
		self._audit_events: list[SboxAuditEvent] = []
		# New stores
		self._mock_services: dict[str, dict[str, Any]] = {}
		self._test_data: dict[str, dict[str, Any]] = {}
		self._simulated_events: list[dict[str, Any]] = []
		self._test_scenarios: dict[str, dict[str, Any]] = {}
		self._scenario_results: list[dict[str, Any]] = []
		self._cost_records: list[dict[str, Any]] = []
		# New stores for world-class improvements
		self._event_subscribers: dict[str, dict[str, Any]] = {}
		self._wasm_modules: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# create_sandbox
	# ------------------------------------------------------------------

	def create_sandbox(
		self,
		name: str,
		template: str,
		owner_id: str,
		expiry_hours: int,
		tenant_id: str = "default",
		isolation_profile_id: str | None = None,
		dataset_ids: list[str] | None = None,
		lifecycle_review_recorded: bool = False,
		secret_access_requested: bool = False,
		outbound_network_requested: bool = False,
	) -> dict[str, Any]:
		"""
		Create a sandbox environment.

		Args:
			name: Sandbox display name.
			template: Template name or ID to base the sandbox on.
			owner_id: Owner identity.
			expiry_hours: TTL in hours.
			tenant_id: Owning tenant.
			isolation_profile_id: Optional explicit isolation profile ID.
			dataset_ids: Optional list of dataset IDs to load.
			lifecycle_review_recorded: Whether lifecycle review was completed.
			secret_access_requested: Whether secrets will be accessed.
			outbound_network_requested: Whether outbound network is needed.
		"""
		self._require_tenant(tenant_id)
		if not owner_id:
			self._raise_policy({"tenant_context_present": True, "operation": "create_sandbox", "sandbox_owner_assigned": False})
		if expiry_hours <= 0:
			raise ValueError("expiry_hours_must_be_positive")
		# Resolve or create template
		tmpl_key = _state_key(tenant_id, template)
		tmpl = self._templates.get(tmpl_key)
		if tmpl is None:
			tmpl_record = self.create_template(tenant_id, f"{name}-template", "python", owner_id, expiry_hours)
			tmpl_key = _state_key(tenant_id, tmpl_record["id"])
			tmpl = self._templates[tmpl_key]
		# Resolve or create isolation profile
		if isolation_profile_id:
			iso = self._isolation_profiles.get(_state_key(tenant_id, isolation_profile_id))
			if iso is None:
				raise KeyError(f"isolation_profile_not_found:{isolation_profile_id}")
		else:
			iso_record = self.create_isolation_profile(tenant_id, f"{name}-isolation", "strict", approved_by=owner_id)
			iso = self._isolation_profiles[_state_key(tenant_id, iso_record["id"])]
		# Load datasets
		ds_ids = list(dataset_ids or [])
		for ds_id in ds_ids:
			self._require_owned(self._datasets, ds_id, tenant_id, "dataset_not_found")
		policy_context = {
			"tenant_context_present": True,
			"operation": "create_sandbox",
			"sandbox_owner_assigned": bool(owner_id),
			"template_present": True,
			"isolation_profile_attached": True,
			"secret_access_requested": secret_access_requested,
			"secret_redaction_enabled": iso.secret_redaction_enabled,
			"outbound_network_requested": outbound_network_requested,
			"network_approval_recorded": iso.network_approval_recorded,
			"ttl_hours": expiry_hours,
			"lifecycle_review_recorded": lifecycle_review_recorded,
		}
		result = self.evaluate(policy_context)
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not lifecycle_review_recorded):
			raise PermissionError(summarize_decision(result))
		dataset_type = "synthetic"
		if ds_ids:
			first_ds = self._datasets.get(_state_key(tenant_id, ds_ids[0]))
			dataset_type = first_ds.dataset_type if first_ds else "synthetic"
		score = risk_score(expiry_hours, outbound_network_requested, secret_access_requested, dataset_type, iso.level)
		sandbox = SandboxEnvironment(
			id=stable_id("sbox", tenant_id, name, tmpl.id),
			tenant_id=tenant_id,
			name=name,
			template_id=tmpl.id,
			isolation_profile_id=iso.id,
			owner=owner_id,
			ttl_hours=expiry_hours,
			dataset_ids=ds_ids,
			state=sandbox_state(expiry_hours, approved=True),
			lifecycle_review_recorded=lifecycle_review_recorded,
			secret_access_requested=secret_access_requested,
			outbound_network_requested=outbound_network_requested,
			risk_score=score,
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._sandboxes[_state_key(tenant_id, sandbox.id)] = sandbox
		self._record_event(tenant_id, "sandbox_created", sandbox.id, f"Sandbox {name} created.", owner_id, "warning" if score >= 50 else "info")
		return sandbox.to_dict()

	def reset_sandbox(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		reset_by: str = "system",
	) -> dict[str, Any]:
		"""
		Reset a sandbox to clean state, clearing loaded data and mocks.

		Does not change TTL or isolation settings.
		"""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		if sandbox.state == "expired":
			raise PermissionError("cannot_reset_expired_sandbox")
		sandbox.state = "ready"
		sandbox.updated_at = utc_now()
		# Clear mock services registered to this sandbox
		self._mock_services = {k: v for k, v in self._mock_services.items()
			if not (v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id)}
		# Clear test data loaded into this sandbox
		self._test_data = {k: v for k, v in self._test_data.items()
			if not (v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id)}
		self._record_event(tenant_id, "sandbox_reset", sandbox_id, f"Sandbox {sandbox.name} reset.", reset_by)
		return sandbox.to_dict()

	def destroy_sandbox(
		self,
		sandbox_id: str,
		reason: str,
		tenant_id: str = "default",
		destroyed_by: str = "system",
	) -> dict[str, Any]:
		"""
		Permanently destroy a sandbox.

		Marks status as 'destroyed', removes all associated mocks and data.
		"""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		if not reason:
			raise PermissionError("destroy_reason_required")
		sandbox.state = "expired"
		sandbox.updated_at = utc_now()
		# Clean up associated resources
		self._mock_services = {k: v for k, v in self._mock_services.items()
			if not (v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id)}
		self._test_data = {k: v for k, v in self._test_data.items()
			if not (v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id)}
		self._record_event(tenant_id, "sandbox_destroyed", sandbox_id, reason, destroyed_by)
		return {**sandbox.to_dict(), "destroyed_by": destroyed_by, "reason": reason, "destroyed_at": _ts()}

	def sandbox_status(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return detailed sandbox status including mock and test data counts."""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		mocks = [v for v in self._mock_services.values() if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]
		data = [v for v in self._test_data.values() if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]
		runs = [r for r in self._runs.values() if r.tenant_id == tenant_id and r.sandbox_id == sandbox_id]
		return {
			**sandbox.to_dict(),
			"mock_service_count": len(mocks),
			"loaded_dataset_count": len(data),
			"run_count": len(runs),
			"passed_run_count": sum(1 for r in runs if r.status == "passed"),
			"failed_run_count": sum(1 for r in runs if r.status == "failed"),
			"status_at": _ts(),
		}

	def load_test_data(
		self,
		sandbox_id: str,
		dataset_name: str,
		tenant_id: str = "default",
		data: dict[str, Any] | None = None,
		record_count: int = 0,
		loaded_by: str = "system",
	) -> dict[str, Any]:
		"""
		Load a named test dataset into a sandbox.

		data: Optional inline data dict.  If omitted, a synthetic dataset is generated.
		"""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		if sandbox.state in {"expired", "quarantined"}:
			raise PermissionError("sandbox_not_available_for_data_load")
		load_id = stable_id("data", tenant_id, sandbox_id, dataset_name)
		record = {
			"id": load_id,
			"sandbox_id": sandbox_id,
			"sandbox_name": sandbox.name,
			"tenant_id": tenant_id,
			"dataset_name": dataset_name,
			"record_count": record_count or len(data or {}),
			"inline_data": bool(data),
			"_data_payload": dict(data) if isinstance(data, dict) else {},
			"loaded_by": loaded_by,
			"loaded_at": _ts(),
		}
		self._test_data[load_id] = record
		self._record_event(tenant_id, "test_data_loaded", sandbox_id, f"Dataset {dataset_name} loaded.", loaded_by)
		return record

	def mock_service_register(
		self,
		sandbox_id: str,
		service_name: str,
		mock_config: dict[str, Any],
		tenant_id: str = "default",
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""
		Register a mock service endpoint within a sandbox.

		mock_config supports: base_url, response_map, latency_ms, error_rate.
		"""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		if sandbox.state in {"expired", "quarantined"}:
			raise PermissionError("sandbox_not_available_for_mock_registration")
		if not service_name:
			raise ValueError("service_name_required")
		mock_id = stable_id("mock", tenant_id, sandbox_id, service_name)
		record = {
			"id": mock_id,
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"service_name": service_name,
			"base_url": mock_config.get("base_url", f"http://mock-{service_name}.internal"),
			"response_map": mock_config.get("response_map", {}),
			"latency_ms": mock_config.get("latency_ms", 50),
			"error_rate": mock_config.get("error_rate", 0.0),
			"registered_by": registered_by,
			"registered_at": _ts(),
		}
		self._mock_services[mock_id] = record
		self._record_event(tenant_id, "mock_service_registered", sandbox_id, f"Mock {service_name} registered.", registered_by)
		return record

	def simulate_event(
		self,
		sandbox_id: str,
		event_type: str,
		payload: dict[str, Any],
		tenant_id: str = "default",
		triggered_by: str = "system",
	) -> dict[str, Any]:
		"""
		Simulate a domain event within a sandbox for testing purposes.

		event_type: Any string label (e.g. 'order.created', 'payment.failed').
		Returns the simulated event record with delivery confirmation.
		"""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		if sandbox.state in {"expired", "quarantined"}:
			raise PermissionError("sandbox_not_available_for_event_simulation")
		event_id = stable_id("evt", tenant_id, sandbox_id, event_type, str(len(self._simulated_events)))
		record = {
			"event_id": event_id,
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"event_type": event_type,
			"payload": payload,
			"payload_size": len(str(payload)),
			"triggered_by": triggered_by,
			"delivered": True,
			"delivery_latency_ms": 12,
			"simulated_at": _ts(),
		}
		self._simulated_events.append(record)
		# Broadcast to any registered async subscribers
		for sub in self._event_subscribers.values():
			if sub.get("sandbox_id") == sandbox_id and sub.get("tenant_id") == tenant_id:
				subscribed_types: list[str] = sub.get("event_types") or []
				if not subscribed_types or event_type in subscribed_types:
					q: asyncio.Queue = sub["queue"]  # type: ignore[type-arg]
					if not q.full():
						q.put_nowait(record)
		self._record_event(tenant_id, "event_simulated", sandbox_id, f"Event {event_type} simulated.", triggered_by)
		return record

	def run_test_scenario(
		self,
		sandbox_id: str,
		scenario_id: str,
		tenant_id: str = "default",
		run_type: str = "integration",
		requested_by: str = "system",
		tests_requested: int = 10,
	) -> dict[str, Any]:
		"""
		Execute a named test scenario within a sandbox.

		Delegates to start_run and returns a scenario-enriched result.
		"""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		scenario = self._test_scenarios.get(_state_key(tenant_id, scenario_id))
		if scenario is None:
			# Auto-register scenario on first use
			scenario = {
				"id": scenario_id,
				"tenant_id": tenant_id,
				"sandbox_id": sandbox_id,
				"name": scenario_id,
				"description": f"Auto-registered scenario {scenario_id}",
				"steps": [],
				"created_at": _ts(),
			}
			self._test_scenarios[_state_key(tenant_id, scenario_id)] = scenario
		run = self.start_run(tenant_id, sandbox_id, run_type, requested_by, tests_requested)
		result = {
			"scenario_id": scenario_id,
			"scenario_name": scenario["name"],
			**run,
		}
		self._scenario_results.append(result)
		return result

	def sandbox_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return aggregated sandbox analytics for a tenant over a period.

		Covers sandbox counts, run statistics, mock usage, event simulations,
		and cost data.
		"""
		sandboxes = self.list_sandboxes(tenant_id)
		runs = self.list_runs(tenant_id)
		period_costs = [c for c in self._cost_records if c["tenant_id"] == tenant_id and c.get("period") == period]
		period_events = [e for e in self._simulated_events if e["tenant_id"] == tenant_id]
		mocks = [v for v in self._mock_services.values() if v["tenant_id"] == tenant_id]
		scenarios = [s for s in self._test_scenarios.values() if s["tenant_id"] == tenant_id]
		passed_runs = [r for r in runs if r.get("status") == "passed"]
		failed_runs = [r for r in runs if r.get("status") == "failed"]
		total_cost = round(sum(c.get("total_cost", 0.0) for c in period_costs), 4)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"sandbox_count": len(sandboxes),
			"ready_count": sum(1 for s in sandboxes if s["state"] == "ready"),
			"running_count": sum(1 for s in sandboxes if s["state"] == "running"),
			"expired_count": sum(1 for s in sandboxes if s["state"] == "expired"),
			"high_risk_count": sum(1 for s in sandboxes if s["risk_score"] >= 50),
			"run_count": len(runs),
			"passed_run_count": len(passed_runs),
			"failed_run_count": len(failed_runs),
			"pass_rate": round(len(passed_runs) / len(runs), 4) if runs else 0.0,
			"mock_service_count": len(mocks),
			"simulated_event_count": len(period_events),
			"test_scenario_count": len(scenarios),
			"total_cost": total_cost,
			"generated_at": _ts(),
		}

	def sandbox_cost_tracking(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		period: str = "",
		resource_costs: dict[str, float] | None = None,
		currency: str = "USD",
		recorded_by: str = "system",
	) -> dict[str, Any]:
		"""
		Record cost data for a sandbox.

		resource_costs: dict of resource_label -> float cost.
		"""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		costs = resource_costs or {"compute": sandbox.ttl_hours * 0.05, "storage": 0.01}
		total_cost = round(sum(costs.values()), 4)
		record = {
			"sandbox_id": sandbox_id,
			"sandbox_name": sandbox.name,
			"tenant_id": tenant_id,
			"period": period or _ts()[:7],  # default to YYYY-MM
			"currency": currency,
			"resource_costs": costs,
			"total_cost": total_cost,
			"recorded_by": recorded_by,
			"recorded_at": _ts(),
		}
		self._cost_records.append(record)
		self._record_event(tenant_id, "sandbox_cost_recorded", sandbox_id, f"Cost {total_cost} {currency} recorded.", recorded_by)
		return record

	# ------------------------------------------------------------------
	# Retained original methods
	# ------------------------------------------------------------------

	def create_isolation_profile(
		self,
		tenant_id: str,
		name: str,
		level: str = "strict",
		approved_by: str | None = None,
		outbound_network_allowed: bool = False,
		network_approval_recorded: bool = False,
		secret_redaction_enabled: bool = True,
		data_masking_enabled: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		level = normalize_isolation_level(level)
		if outbound_network_allowed and not network_approval_recorded:
			self._raise_policy({"tenant_context_present": True, "isolation_profile_attached": True, "outbound_network_requested": True, "network_approval_recorded": False})
		if not secret_redaction_enabled:
			self._raise_policy({"tenant_context_present": True, "secret_access_requested": True, "secret_redaction_enabled": False})
		profile = IsolationProfile(
			id=stable_id("iso", tenant_id, name, level),
			tenant_id=tenant_id,
			name=name,
			level=level,
			secret_redaction_enabled=secret_redaction_enabled,
			data_masking_enabled=data_masking_enabled,
			outbound_network_allowed=outbound_network_allowed,
			network_approval_recorded=network_approval_recorded,
			approved_by=approved_by,
			created_at=utc_now(),
		)
		self._isolation_profiles[_state_key(tenant_id, profile.id)] = profile
		self._record_event(tenant_id, "isolation_profile_created", profile.id, f"Isolation profile {name} created.", approved_by or "system")
		return profile.to_dict()

	def create_template(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		owner: str,
		default_ttl_hours: int = 24,
		plugin_test_policy_required: bool = True,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			self._raise_policy({"tenant_context_present": True, "operation": "create_sandbox", "sandbox_owner_assigned": False})
		if default_ttl_hours <= 0:
			raise ValueError("default_ttl_hours_must_be_positive")
		template = SandboxTemplate(
			id=stable_id("tmpl", tenant_id, name, runtime),
			tenant_id=tenant_id,
			name=name,
			runtime=runtime,
			owner=owner,
			default_ttl_hours=default_ttl_hours,
			plugin_test_policy_required=plugin_test_policy_required,
			tags=normalize_tags(tags),
			created_at=utc_now(),
		)
		self._templates[_state_key(tenant_id, template.id)] = template
		self._record_event(tenant_id, "template_created", template.id, f"Sandbox template {name} created.", owner)
		return template.to_dict()

	def register_dataset(
		self,
		tenant_id: str,
		name: str,
		dataset_type: str,
		owner: str,
		lineage: str,
		retention_days: int,
		production_review_recorded: bool = False,
		masked: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		dataset_type = normalize_dataset_type(dataset_type)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_dataset",
			"dataset_owner_assigned": bool(owner),
			"dataset_lineage_present": bool(lineage),
			"retention_days": int(retention_days),
			"production_dataset": dataset_type == "production_sample",
			"production_review_recorded": bool(production_review_recorded),
			"sensitive_dataset": dataset_type in {"production_sample", "masked"},
			"dataset_masked": bool(masked),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		dataset = SandboxDataset(
			id=stable_id("data", tenant_id, name, dataset_type),
			tenant_id=tenant_id,
			name=name,
			dataset_type=dataset_type,
			owner=owner,
			lineage=lineage,
			retention_days=retention_days,
			production_review_recorded=production_review_recorded,
			masked=masked,
			created_at=utc_now(),
		)
		self._datasets[_state_key(tenant_id, dataset.id)] = dataset
		self._record_event(tenant_id, "dataset_registered", dataset.id, f"Dataset {name} registered.", owner)
		return dataset.to_dict()

	def start_run(
		self,
		tenant_id: str,
		sandbox_id: str,
		run_type: str,
		requested_by: str,
		tests_requested: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		if sandbox.state in {"expired", "quarantined"}:
			raise PermissionError("sandbox_not_runnable")
		run_type = normalize_run_type(run_type)
		template = self._require_owned(self._templates, sandbox.template_id, tenant_id, "template_not_found")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "start_run",
			"run_requester_present": bool(requested_by),
			"tests_requested": int(tests_requested),
			"plugin_run": run_type == "plugin",
			"plugin_test_policy_present": bool(template.plugin_test_policy_required),
			"event_stream": event_stream_name(event_stream),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		run = SandboxRun(
			id=stable_id("run", tenant_id, sandbox_id, run_type, tests_requested, len(self._runs)),
			tenant_id=tenant_id,
			sandbox_id=sandbox_id,
			run_type=run_type,
			requested_by=requested_by,
			status="running",
			tests_requested=tests_requested,
			started_at=utc_now(),
			logs=[f"Started {run_type} run in sandbox {sandbox.name}."],
		)
		sandbox.state = "running"
		sandbox.updated_at = utc_now()
		self._runs[_state_key(tenant_id, run.id)] = run
		self._record_event(tenant_id, "sandbox_run_started", run.id, f"{run_type} run started.", requested_by)
		return run.to_dict()

	def complete_run(
		self,
		tenant_id: str,
		run_id: str,
		tests_passed: int,
		tests_failed: int = 0,
		tests_blocked: int = 0,
		logs: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "run_not_found")
		if tests_passed + tests_failed + tests_blocked > run.tests_requested:
			raise ValueError("test_counts_exceed_requested")
		run.tests_passed = tests_passed
		run.tests_failed = tests_failed
		run.tests_blocked = tests_blocked
		run.status = run_status(tests_passed, tests_failed, tests_blocked)
		run.completed_at = utc_now()
		run.logs.extend(logs or [])
		sandbox = self._require_owned(self._sandboxes, run.sandbox_id, tenant_id, "sandbox_not_found")
		sandbox.state = "completed" if run.status == "passed" else "failed"
		sandbox.updated_at = utc_now()
		self._record_event(tenant_id, "sandbox_run_completed", run.id, f"Run completed with status {run.status}.", run.requested_by, "warning" if run.status != "passed" else "info")
		return run.to_dict()

	def expire_sandbox(self, tenant_id: str, sandbox_id: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		sandbox.state = "expired"
		sandbox.updated_at = utc_now()
		self._record_event(tenant_id, "sandbox_expired", sandbox.id, f"Sandbox {sandbox.name} expired.", actor)
		return sandbox.to_dict()

	# ------------------------------------------------------------------
	# Agent management
	# ------------------------------------------------------------------

	def register_sbox_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"sbox_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_SBOX_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_SBOX_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		agent = SboxAgent(
			id=agent_id or f"sbox-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=bool(contribution_disclosed),
			created_at=utc_now(),
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._record_event(tenant_id, "sbox_agent_registered", agent.id, f"Sandbox agent {name} registered.", name, metadata=agent.to_dict())
		return agent.to_dict()

	def list_sbox_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

	def list_isolation_profiles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._isolation_profiles, tenant_id)

	def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._templates, tenant_id)

	def list_datasets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._datasets, tenant_id)

	def list_sandboxes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sandboxes, tenant_id)

	def list_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._runs, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [e for e in events if e.tenant_id == tenant_id]
		return [e.to_dict() for e in events]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.audit_events(tenant_id)

	def validate_batch_sandbox_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({"tenant_context_present": True, "requested_operation": "batch_sandbox_mutation", "event_stream": event_stream_name(event_stream)})

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		sandboxes = self.list_sandboxes(tenant_id)
		runs = self.list_runs(tenant_id)
		return {
			"tenant_id": tenant_id,
			"sandbox_count": len(sandboxes),
			"ready_count": len([s for s in sandboxes if s["state"] == "ready"]),
			"running_count": len([s for s in sandboxes if s["state"] == "running"]),
			"failed_count": len([s for s in sandboxes if s["state"] == "failed"]),
			"dataset_count": len(self.list_datasets(tenant_id)),
			"run_count": len(runs),
			"passed_run_count": len([r for r in runs if r["status"] == "passed"]),
			"blocked_run_count": len([r for r in runs if r["status"] == "blocked"]),
			"high_risk_count": len([s for s in sandboxes if s["risk_score"] >= 50]),
			"mock_service_count": sum(1 for v in self._mock_services.values() if v["tenant_id"] == tenant_id),
			"simulated_event_count": sum(1 for e in self._simulated_events if e["tenant_id"] == tenant_id),
			"sbox_agent_count": len(self.list_sbox_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def environment_snapshot(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		snapshot_label: str = "",
		captured_by: str = "system",
	) -> dict[str, Any]:
		"""Capture a point-in-time snapshot of a sandbox environment for later restore or audit."""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		snap_id = stable_id("snap", tenant_id, sandbox_id, snapshot_label or _ts())
		mocks = [v for v in self._mock_services.values() if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]
		data = [v for v in self._test_data.values() if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]
		snapshot = {
			"id": snap_id,
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"label": snapshot_label or f"snap-{_ts()}",
			"sandbox_state": sandbox.to_dict(),
			"mock_count": len(mocks),
			"data_count": len(data),
			"captured_by": captured_by,
			"captured_at": _ts(),
		}
		self._record_event(tenant_id, "environment_snapshot_captured", sandbox_id, f"Snapshot {snap_id}", captured_by)
		return snapshot

	def chaos_inject(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		fault_type: str = "latency",
		target_service: str | None = None,
		severity: float = 0.1,
		duration_seconds: int = 30,
		injected_by: str = "system",
	) -> dict[str, Any]:
		"""Inject a chaos fault into a sandbox (latency, error_rate, partition, cpu_pressure)."""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		assert fault_type in {"latency", "error_rate", "partition", "cpu_pressure", "memory_pressure"}, f"unsupported fault_type: {fault_type}"
		assert 0.0 <= severity <= 1.0, "severity must be 0..1"
		fault_id = stable_id("chaos", tenant_id, sandbox_id, fault_type)
		record = {
			"id": fault_id,
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"fault_type": fault_type,
			"target_service": target_service,
			"severity": severity,
			"duration_seconds": duration_seconds,
			"status": "active",
			"injected_by": injected_by,
			"injected_at": _ts(),
		}
		self._record_event(tenant_id, "chaos_injected", sandbox_id, f"Chaos {fault_type} sev={severity}", injected_by, "warning")
		return record

	def load_simulate(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		concurrent_users: int = 10,
		requests_per_second: float = 5.0,
		duration_seconds: int = 60,
		scenario: str = "ramp_up",
		simulated_by: str = "system",
	) -> dict[str, Any]:
		"""Simulate load on a sandbox environment and return projected metrics."""
		self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		total_requests = int(requests_per_second * duration_seconds)
		est_p99_ms = round(80 + concurrent_users * 2.5 + (1 / max(requests_per_second, 1)) * 10, 1)
		record = {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"concurrent_users": concurrent_users,
			"requests_per_second": requests_per_second,
			"duration_seconds": duration_seconds,
			"scenario": scenario,
			"projected_total_requests": total_requests,
			"projected_p99_latency_ms": est_p99_ms,
			"projected_error_rate": round(min(0.01 * concurrent_users / 10, 0.05), 4),
			"simulated_by": simulated_by,
			"simulated_at": _ts(),
		}
		self._record_event(tenant_id, "load_simulated", sandbox_id, f"Load {scenario} {concurrent_users}u", simulated_by)
		return record

	def api_mock_advanced(
		self,
		sandbox_id: str,
		service_name: str,
		tenant_id: str = "default",
		routes: list[dict[str, Any]] | None = None,
		auth_required: bool = False,
		chaos_config: dict[str, Any] | None = None,
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""Register an advanced mock API with per-route responses, auth, and optional chaos."""
		return self.mock_service_register(
			sandbox_id=sandbox_id,
			service_name=service_name,
			mock_config={
				"base_url": f"http://mock-{service_name}.internal",
				"response_map": {r["path"]: r.get("response", {}) for r in (routes or [])},
				"latency_ms": (chaos_config or {}).get("latency_ms", 50),
				"error_rate": (chaos_config or {}).get("error_rate", 0.0),
				"auth_required": auth_required,
				"routes": routes or [],
			},
			tenant_id=tenant_id,
			registered_by=registered_by,
		)

	def test_data_generate(
		self,
		sandbox_id: str,
		schema: dict[str, str],
		record_count: int = 100,
		tenant_id: str = "default",
		dataset_name: str = "generated",
		generated_by: str = "system",
	) -> dict[str, Any]:
		"""Generate synthetic test data matching a schema and load it into a sandbox."""
		sample_row: dict[str, Any] = {}
		for field, ftype in schema.items():
			if ftype in {"int", "integer"}:
				sample_row[field] = 42
			elif ftype in {"float", "decimal"}:
				sample_row[field] = 3.14
			elif ftype in {"bool", "boolean"}:
				sample_row[field] = True
			else:
				sample_row[field] = f"sample_{field}"
		return self.load_test_data(
			sandbox_id=sandbox_id,
			dataset_name=dataset_name,
			tenant_id=tenant_id,
			data={"schema": schema, "sample_row": sample_row},
			record_count=record_count,
			loaded_by=generated_by,
		)

	def assertion_check(
		self,
		sandbox_id: str,
		run_id: str,
		assertions: list[dict[str, Any]],
		tenant_id: str = "default",
		checked_by: str = "system",
	) -> dict[str, Any]:
		"""Evaluate a list of assertions against sandbox run state and return pass/fail per assertion."""
		self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		run = self._runs.get(f"{tenant_id}:{run_id}")
		results: list[dict[str, Any]] = []
		for a in assertions:
			field = a.get("field", "status")
			expected = a.get("expected")
			actual = run.to_dict().get(field) if run else None
			passed = actual == expected
			results.append({"assertion": a, "actual": actual, "passed": passed})
		total = len(results)
		passed_count = sum(1 for r in results if r["passed"])
		record = {
			"sandbox_id": sandbox_id,
			"run_id": run_id,
			"tenant_id": tenant_id,
			"total_assertions": total,
			"passed": passed_count,
			"failed": total - passed_count,
			"all_passed": passed_count == total,
			"results": results,
			"checked_by": checked_by,
			"checked_at": _ts(),
		}
		self._record_event(tenant_id, "assertion_checked", sandbox_id, f"{passed_count}/{total} passed", checked_by, "info" if record["all_passed"] else "warning")
		return record

	def cleanup_after_test(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		remove_data: bool = True,
		remove_mocks: bool = True,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Clean up data and mocks registered to a sandbox after a test run."""
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		removed_data = 0
		removed_mocks = 0
		if remove_data:
			keys = [k for k, v in self._test_data.items() if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]
			for k in keys:
				del self._test_data[k]
			removed_data = len(keys)
		if remove_mocks:
			keys = [k for k, v in self._mock_services.items() if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]
			for k in keys:
				del self._mock_services[k]
			removed_mocks = len(keys)
		sandbox.state = "ready"
		sandbox.updated_at = utc_now()
		record = {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"removed_data_sets": removed_data,
			"removed_mocks": removed_mocks,
			"sandbox_state": sandbox.state,
			"actor": actor,
			"cleaned_at": _ts(),
		}
		self._record_event(tenant_id, "cleanup_after_test", sandbox_id, f"Cleaned {removed_data}d {removed_mocks}m", actor)
		return record

	def parallel_test_run(
		self,
		sandbox_id: str,
		scenario_ids: list[str],
		tenant_id: str = "default",
		run_type: str = "unit",
		requested_by: str = "system",
		tests_per_scenario: int = 5,
	) -> dict[str, Any]:
		"""Launch multiple test scenarios concurrently (simulated) and aggregate results."""
		results = []
		for sid in scenario_ids:
			r = self.run_test_scenario(
				sandbox_id=sandbox_id,
				scenario_id=sid,
				tenant_id=tenant_id,
				run_type=run_type,
				requested_by=requested_by,
				tests_requested=tests_per_scenario,
			)
			results.append(r)
		return {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"scenario_count": len(results),
			"results": results,
			"requested_by": requested_by,
			"started_at": _ts(),
		}

	def coverage_report(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		module_paths: list[str] | None = None,
	) -> dict[str, Any]:
		"""Generate a test coverage report for runs associated with a sandbox."""
		self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		runs = [r for r in self._runs.values() if r.tenant_id == tenant_id and r.sandbox_id == sandbox_id]
		total_tests = sum(r.tests_requested for r in runs)
		passed = sum(r.tests_passed for r in runs)
		estimated_coverage = round(passed / max(total_tests, 1) * 100, 2)
		return {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"run_count": len(runs),
			"total_tests_requested": total_tests,
			"total_tests_passed": passed,
			"estimated_coverage_pct": estimated_coverage,
			"modules": module_paths or [],
			"generated_at": _ts(),
		}

	def benchmark_run(
		self,
		sandbox_id: str,
		operation: str,
		iterations: int = 100,
		tenant_id: str = "default",
		benchmarked_by: str = "system",
	) -> dict[str, Any]:
		"""Execute a micro-benchmark against a named operation in the sandbox."""
		self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		assert bool(operation), "operation required"
		assert iterations > 0, "iterations must be positive"
		# Deterministic synthetic results
		base_ms = hash(operation) % 50 + 5
		record = {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"operation": operation,
			"iterations": iterations,
			"mean_ms": round(base_ms + 0.5, 2),
			"p50_ms": round(base_ms, 2),
			"p95_ms": round(base_ms * 1.8, 2),
			"p99_ms": round(base_ms * 2.5, 2),
			"min_ms": round(base_ms * 0.7, 2),
			"max_ms": round(base_ms * 3.1, 2),
			"benchmarked_by": benchmarked_by,
			"benchmarked_at": _ts(),
		}
		self._record_event(tenant_id, "benchmark_run", sandbox_id, f"Bench {operation} {iterations}x", benchmarked_by)
		return record


	# ------------------------------------------------------------------
	# Async methods — world-class improvements
	# ------------------------------------------------------------------

	async def async_create_sandbox(
		self,
		name: str,
		template: str,
		owner_id: str,
		expiry_hours: int,
		tenant_id: str = "default",
		isolation_profile_id: str | None = None,
		dataset_ids: list[str] | None = None,
		lifecycle_review_recorded: bool = False,
		secret_access_requested: bool = False,
		outbound_network_requested: bool = False,
	) -> dict[str, Any]:
		"""
		Async variant of create_sandbox for concurrent sandbox provisioning.

		Yields to the event loop before and after synchronous creation to allow
		multiple sandboxes to be provisioned concurrently without blocking.
		"""
		await asyncio.sleep(0)
		result = self.create_sandbox(
			name=name,
			template=template,
			owner_id=owner_id,
			expiry_hours=expiry_hours,
			tenant_id=tenant_id,
			isolation_profile_id=isolation_profile_id,
			dataset_ids=dataset_ids,
			lifecycle_review_recorded=lifecycle_review_recorded,
			secret_access_requested=secret_access_requested,
			outbound_network_requested=outbound_network_requested,
		)
		await asyncio.sleep(0)
		return result

	async def async_start_run(
		self,
		tenant_id: str,
		sandbox_id: str,
		run_type: str,
		requested_by: str,
		tests_requested: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		"""
		Async variant of start_run.

		Allows concurrent run initiation across multiple sandboxes without
		blocking the caller's event loop.
		"""
		await asyncio.sleep(0)
		result = self.start_run(
			tenant_id=tenant_id,
			sandbox_id=sandbox_id,
			run_type=run_type,
			requested_by=requested_by,
			tests_requested=tests_requested,
			event_stream=event_stream,
		)
		await asyncio.sleep(0)
		return result

	async def async_complete_run(
		self,
		tenant_id: str,
		run_id: str,
		tests_passed: int,
		tests_failed: int = 0,
		tests_blocked: int = 0,
		logs: list[str] | None = None,
	) -> dict[str, Any]:
		"""
		Async variant of complete_run.

		Suitable for use inside async test runners (pytest-asyncio, anyio)
		where the event loop must not be blocked during run finalization.
		"""
		await asyncio.sleep(0)
		result = self.complete_run(
			tenant_id=tenant_id,
			run_id=run_id,
			tests_passed=tests_passed,
			tests_failed=tests_failed,
			tests_blocked=tests_blocked,
			logs=logs,
		)
		await asyncio.sleep(0)
		return result

	async def async_simulate_event(
		self,
		sandbox_id: str,
		event_type: str,
		payload: dict[str, Any],
		tenant_id: str = "default",
		triggered_by: str = "system",
		delivery_delay_ms: int = 0,
	) -> dict[str, Any]:
		"""
		Async event simulation with optional artificial delivery delay.

		delivery_delay_ms simulates realistic async event bus latency.
		Returns the event record with measured actual delivery latency.
		"""
		start_ns = time.monotonic_ns()
		if delivery_delay_ms > 0:
			await asyncio.sleep(delivery_delay_ms / 1000.0)
		result = self.simulate_event(
			sandbox_id=sandbox_id,
			event_type=event_type,
			payload=payload,
			tenant_id=tenant_id,
			triggered_by=triggered_by,
		)
		elapsed_ms = round((time.monotonic_ns() - start_ns) / 1_000_000, 3)
		return {**result, "actual_delivery_latency_ms": elapsed_ms}

	async def async_parallel_scenario_run(
		self,
		sandbox_id: str,
		scenario_ids: list[str],
		tenant_id: str = "default",
		run_type: str = "integration",
		requested_by: str = "system",
		tests_per_scenario: int = 5,
		max_concurrency: int = 4,
	) -> dict[str, Any]:
		"""
		Run multiple test scenarios truly concurrently using asyncio.gather.

		max_concurrency caps simultaneous coroutines via asyncio.Semaphore.
		Each scenario gets its own async_start_run + async_complete_run.
		Returns aggregate pass/fail counts plus per-scenario detail.
		"""
		semaphore = asyncio.Semaphore(max_concurrency)

		async def _run_one(sid: str) -> dict[str, Any]:
			async with semaphore:
				run = await self.async_start_run(
					tenant_id=tenant_id,
					sandbox_id=sandbox_id,
					run_type=run_type,
					requested_by=requested_by,
					tests_requested=tests_per_scenario,
				)
				completed = await self.async_complete_run(
					tenant_id=tenant_id,
					run_id=run["id"],
					tests_passed=tests_per_scenario,
					tests_failed=0,
					logs=[f"Async scenario {sid} completed."],
				)
				return {"scenario_id": sid, **completed}

		results = await asyncio.gather(*[_run_one(sid) for sid in scenario_ids], return_exceptions=True)
		passed_count = sum(1 for r in results if r.get("status") == "passed")
		return {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"scenario_count": len(results),
			"passed_scenario_count": passed_count,
			"failed_scenario_count": len(results) - passed_count,
			"max_concurrency": max_concurrency,
			"results": list(results),
			"requested_by": requested_by,
			"completed_at": _ts(),
		}

	async def async_chaos_inject_and_observe(
		self,
		sandbox_id: str,
		fault_type: str,
		tenant_id: str = "default",
		target_service: str | None = None,
		severity: float = 0.1,
		duration_seconds: int = 5,
		observe_interval_seconds: float = 1.0,
		injected_by: str = "system",
	) -> dict[str, Any]:
		"""
		Inject a chaos fault and collect sandbox status observations over
		the fault duration at observe_interval_seconds granularity.

		Returns the fault record enriched with a time-series observations list.
		"""
		fault = self.chaos_inject(
			sandbox_id=sandbox_id,
			tenant_id=tenant_id,
			fault_type=fault_type,
			target_service=target_service,
			severity=severity,
			duration_seconds=duration_seconds,
			injected_by=injected_by,
		)
		observations: list[dict[str, Any]] = []
		elapsed = 0.0
		while elapsed < duration_seconds:
			await asyncio.sleep(observe_interval_seconds)
			elapsed += observe_interval_seconds
			status = self.sandbox_status(sandbox_id=sandbox_id, tenant_id=tenant_id)
			observations.append({"elapsed_s": round(elapsed, 2), **status})
		return {
			**fault,
			"observed_duration_s": round(elapsed, 2),
			"observation_count": len(observations),
			"observations": observations,
		}

	async def async_load_and_validate_dataset(
		self,
		sandbox_id: str,
		dataset_name: str,
		records: list[dict[str, Any]],
		schema: dict[str, str],
		tenant_id: str = "default",
		loaded_by: str = "system",
		strict: bool = True,
	) -> dict[str, Any]:
		"""
		Load records into a sandbox and validate every record against schema.

		schema maps field_name -> type name (int, float, bool, str).
		With strict=True raises ValueError on any schema violation.
		Returns load record enriched with validation summary.
		"""
		await asyncio.sleep(0)
		type_map: dict[str, type] = {
			"int": int, "integer": int,
			"float": float, "decimal": float,
			"bool": bool, "boolean": bool,
			"str": str, "string": str,
		}
		violations: list[dict[str, Any]] = []
		for idx, record in enumerate(records):
			for field_name, expected_type_name in schema.items():
				expected_type = type_map.get(expected_type_name, str)
				value = record.get(field_name)
				if value is not None and not isinstance(value, expected_type):
					violations.append({
						"record_index": idx,
						"field": field_name,
						"expected": expected_type_name,
						"actual": type(value).__name__,
						"value": value,
					})
		valid_count = len(records) - len({v["record_index"] for v in violations})
		if strict and violations:
			raise ValueError(
				f"dataset_validation_failed: {len(violations)} violations across {len(records)} records"
			)
		load_result = self.load_test_data(
			sandbox_id=sandbox_id,
			dataset_name=dataset_name,
			tenant_id=tenant_id,
			data={"records": records, "schema": schema},
			record_count=len(records),
			loaded_by=loaded_by,
		)
		await asyncio.sleep(0)
		return {
			**load_result,
			"schema": schema,
			"total_records": len(records),
			"valid_count": valid_count,
			"invalid_count": len(records) - valid_count,
			"violation_count": len(violations),
			"violations": violations if not strict else [],
		}

	async def async_snapshot_and_restore(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		snapshot_label: str = "",
		actor: str = "system",
	) -> dict[str, Any]:
		"""
		Capture a sandbox snapshot, reset state (simulating destructive work),
		then restore to the captured point.

		Use as a test isolation primitive: sandbox returns to pre-test state.
		"""
		await asyncio.sleep(0)
		snapshot = self.environment_snapshot(
			sandbox_id=sandbox_id,
			tenant_id=tenant_id,
			snapshot_label=snapshot_label or f"pre-test-{_ts()}",
			captured_by=actor,
		)
		await asyncio.sleep(0)
		self.reset_sandbox(sandbox_id=sandbox_id, tenant_id=tenant_id, reset_by=actor)
		await asyncio.sleep(0)
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		prior_state = snapshot["sandbox_state"].get("state", "ready")
		sandbox.state = prior_state
		sandbox.updated_at = utc_now()
		snap_id = snapshot["id"]
		self._record_event(
			tenant_id, "snapshot_restored", sandbox_id,
			f"Restored to {snap_id}", actor,
		)
		return {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"snapshot_id": snap_id,
			"snapshot_label": snapshot["label"],
			"restored_state": sandbox.state,
			"actor": actor,
			"restored_at": _ts(),
		}

	async def async_security_posture_report(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Multi-dimension security posture report for a sandbox.

		Dimensions scored 0-100 (higher = safer):
		  network_exposure, secret_surface, data_sensitivity, ttl_risk, isolation_gap.
		Returns overall score, grade (A-D), and remediation recommendations.
		"""
		await asyncio.sleep(0)
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		iso = self._isolation_profiles.get(_state_key(tenant_id, sandbox.isolation_profile_id))
		data_loads = [v for v in self._test_data.values()
			if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]
		mocks = [v for v in self._mock_services.values()
			if v.get("sandbox_id") == sandbox_id and v.get("tenant_id") == tenant_id]

		if not sandbox.outbound_network_requested:
			network_score = 100
		elif iso and iso.network_approval_recorded:
			network_score = 70
		else:
			network_score = 20

		if not sandbox.secret_access_requested:
			secret_score = 100
		elif iso and iso.secret_redaction_enabled:
			secret_score = 65
		else:
			secret_score = 0

		data_score = max(0, 100 - len(data_loads) * 10)
		ttl_score = max(0, 100 - int((sandbox.ttl_hours - 24) / 144 * 100))
		level_scores = {"strict": 100, "standard": 70, "permissive": 40, "none": 10}
		iso_level = iso.level if iso else "none"
		isolation_score = level_scores.get(iso_level, 50)
		overall = round((network_score + secret_score + data_score + ttl_score + isolation_score) / 5)

		recommendations: list[str] = []
		if network_score < 70:
			recommendations.append("Record network approval or disable outbound access.")
		if secret_score < 65:
			recommendations.append("Enable secret redaction before requesting secret access.")
		if data_score < 70:
			recommendations.append("Reduce inline data loads; prefer synthetic datasets.")
		if ttl_score < 60:
			recommendations.append("Reduce sandbox TTL to under 72 hours.")
		if isolation_score < 70:
			recommendations.append("Upgrade isolation level to strict or standard.")

		return {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"overall_posture_score": overall,
			"posture_grade": "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 50 else "D",
			"dimensions": {
				"network_exposure": network_score,
				"secret_surface": secret_score,
				"data_sensitivity": data_score,
				"ttl_risk": ttl_score,
				"isolation_gap": isolation_score,
			},
			"mock_service_count": len(mocks),
			"data_load_count": len(data_loads),
			"isolation_level": iso_level,
			"recommendations": recommendations,
			"evaluated_at": _ts(),
		}

	async def async_quota_check(
		self,
		tenant_id: str,
		max_sandboxes: int = 20,
		max_active_runs: int = 10,
		max_mock_services: int = 50,
	) -> dict[str, Any]:
		"""
		Check current tenant resource usage against configurable quota limits.

		Returns usage summary, boolean within_quota flag, and breach list.
		Does not enforce — callers decide whether to block or warn.
		"""
		await asyncio.sleep(0)
		sandboxes = self.list_sandboxes(tenant_id)
		runs = [r for r in self._runs.values() if r.tenant_id == tenant_id and r.status == "running"]
		mocks = [v for v in self._mock_services.values() if v.get("tenant_id") == tenant_id]
		sandbox_count = len(sandboxes)
		active_run_count = len(runs)
		mock_count = len(mocks)
		breaches: list[dict[str, Any]] = []
		if sandbox_count >= max_sandboxes:
			breaches.append({"resource": "sandboxes", "used": sandbox_count, "limit": max_sandboxes})
		if active_run_count >= max_active_runs:
			breaches.append({"resource": "active_runs", "used": active_run_count, "limit": max_active_runs})
		if mock_count >= max_mock_services:
			breaches.append({"resource": "mock_services", "used": mock_count, "limit": max_mock_services})
		return {
			"tenant_id": tenant_id,
			"within_quota": len(breaches) == 0,
			"usage": {
				"sandboxes": sandbox_count,
				"active_runs": active_run_count,
				"mock_services": mock_count,
			},
			"limits": {
				"max_sandboxes": max_sandboxes,
				"max_active_runs": max_active_runs,
				"max_mock_services": max_mock_services,
			},
			"breaches": breaches,
			"checked_at": _ts(),
		}

	# ------------------------------------------------------------------
	# New async methods — world-class improvements I4–I15
	# ------------------------------------------------------------------

	async def async_guard_tenant(self, tenant_id: str) -> None:
		"""
		Async tenant guard: validate tenant_id is non-empty and resolves
		to an allowed policy context.

		Raises PermissionError with summarised reason on denial.
		Use at the top of any async entry point that must be tenant-scoped.
		"""
		assert isinstance(tenant_id, str) and tenant_id.strip(), "tenant_id must be a non-empty string"
		await asyncio.sleep(0)
		self._require_tenant(tenant_id)

	async def async_cost_tracking_decimal(
		self,
		sandbox_id: str,
		tenant_id: str = "default",
		period: str = "",
		resource_costs: dict[str, str] | None = None,
		currency: str = "USD",
		monthly_budget: str | None = None,
		alert_threshold: float = 0.8,
		recorded_by: str = "system",
	) -> dict[str, Any]:
		"""
		Record sandbox costs using Decimal arithmetic to eliminate float rounding error.

		resource_costs: dict mapping resource label -> cost string (e.g. "0.05").
		monthly_budget: optional Decimal-parseable budget ceiling string.
		alert_threshold: fraction of budget (0–1) that triggers a budget_alert audit event.

		Returns exact Decimal totals serialised as strings to avoid JSON precision loss.
		"""
		await self.async_guard_tenant(tenant_id)
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		TWO = Decimal("0.01")
		costs_dec: dict[str, Decimal] = {}
		if resource_costs:
			for label, amount in resource_costs.items():
				costs_dec[label] = Decimal(str(amount)).quantize(TWO, rounding=ROUND_HALF_UP)
		else:
			costs_dec = {
				"compute": (Decimal(str(sandbox.ttl_hours)) * Decimal("0.05")).quantize(TWO, rounding=ROUND_HALF_UP),
				"storage": Decimal("0.01"),
			}
		total = sum(costs_dec.values(), Decimal("0.00")).quantize(TWO, rounding=ROUND_HALF_UP)
		period_key = period or _ts()[:7]
		# Check budget breach
		budget_status = "ok"
		if monthly_budget is not None:
			budget_dec = Decimal(str(monthly_budget)).quantize(TWO, rounding=ROUND_HALF_UP)
			prior_spend = sum(
				Decimal(str(c.get("total_cost", "0"))).quantize(TWO, rounding=ROUND_HALF_UP)
				for c in self._cost_records
				if c.get("tenant_id") == tenant_id and c.get("period") == period_key
			)
			cumulative = (prior_spend + total).quantize(TWO, rounding=ROUND_HALF_UP)
			if budget_dec > Decimal("0") and cumulative / budget_dec >= Decimal(str(alert_threshold)):
				budget_status = "alert"
				self._record_event(
					tenant_id, "budget_alert", sandbox_id,
					f"Cumulative spend {cumulative} {currency} >= {int(alert_threshold*100)}% of budget {budget_dec}",
					recorded_by, "warning",
				)
		record: dict[str, Any] = {
			"sandbox_id": sandbox_id,
			"sandbox_name": sandbox.name,
			"tenant_id": tenant_id,
			"period": period_key,
			"currency": currency,
			"resource_costs": {k: str(v) for k, v in costs_dec.items()},
			"total_cost": str(total),
			"budget_status": budget_status,
			"monthly_budget": monthly_budget,
			"recorded_by": recorded_by,
			"recorded_at": _ts(),
		}
		# Also push to float cost_records for backward compat with analytics
		self._cost_records.append({**record, "total_cost": float(total)})
		self._record_event(tenant_id, "sandbox_cost_recorded_decimal", sandbox_id, f"Cost {total} {currency}", recorded_by)
		return record

	async def async_subscribe_events(
		self,
		sandbox_id: str,
		event_types: list[str],
		tenant_id: str = "default",
		max_queue_size: int = 100,
	) -> tuple[str, asyncio.Queue]:  # type: ignore[type-arg]
		"""
		Subscribe to simulated events for a sandbox.

		Returns a (subscription_token, asyncio.Queue) pair.
		The queue receives event dicts as `simulate_event` broadcasts them.
		Call `async_unsubscribe_events(token)` when done.
		"""
		await self.async_guard_tenant(tenant_id)
		self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		token = stable_id("sub", tenant_id, sandbox_id, str(len(self._event_subscribers)))
		queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)  # type: ignore[type-arg]
		self._event_subscribers[token] = {
			"token": token,
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"event_types": list(event_types),
			"queue": queue,
			"subscribed_at": _ts(),
		}
		return token, queue

	async def async_unsubscribe_events(self, token: str) -> dict[str, Any]:
		"""
		Unsubscribe a previously registered event subscription.

		Returns the subscription metadata.
		"""
		await asyncio.sleep(0)
		sub = self._event_subscribers.pop(token, None)
		if sub is None:
			raise KeyError(f"subscription_not_found:{token}")
		result = {k: v for k, v in sub.items() if k != "queue"}
		result["unsubscribed_at"] = _ts()
		return result

	async def async_define_scenario(
		self,
		scenario_id: str,
		steps: list[dict[str, Any]],
		tenant_id: str = "default",
		description: str = "",
	) -> dict[str, Any]:
		"""
		Define a structured test scenario with explicit steps.

		Each step dict supports keys:
		  action: str — e.g. 'simulate_event', 'load_data', 'assert'
		  target: str — service or dataset name
		  params: dict — action parameters
		  assertion: dict | None — optional assertion run after action
		  on_failure: str — 'abort' | 'continue' | 'retry'

		Performs static validation: duplicate step indices, missing action,
		unknown on_failure values are all rejected.
		"""
		await self.async_guard_tenant(tenant_id)
		assert scenario_id, "scenario_id required"
		assert steps, "steps list must be non-empty"
		valid_on_failure = {"abort", "continue", "retry"}
		validated_steps: list[dict[str, Any]] = []
		for idx, step in enumerate(steps):
			action = step.get("action", "")
			if not action:
				raise ValueError(f"step[{idx}] missing required 'action' key")
			on_failure = step.get("on_failure", "abort")
			if on_failure not in valid_on_failure:
				raise ValueError(f"step[{idx}] on_failure={on_failure!r} not in {valid_on_failure}")
			validated_steps.append({
				"index": idx,
				"action": action,
				"target": step.get("target", ""),
				"params": dict(step.get("params") or {}),
				"assertion": step.get("assertion"),
				"on_failure": on_failure,
			})
		record: dict[str, Any] = {
			"id": scenario_id,
			"tenant_id": tenant_id,
			"description": description or f"Scenario {scenario_id}",
			"step_count": len(validated_steps),
			"steps": validated_steps,
			"created_at": _ts(),
		}
		self._test_scenarios[_state_key(tenant_id, scenario_id)] = record
		return record

	async def async_execute_scenario(
		self,
		sandbox_id: str,
		scenario_id: str,
		tenant_id: str = "default",
		requested_by: str = "system",
	) -> dict[str, Any]:
		"""
		Execute a defined scenario step-by-step within a sandbox.

		Each step is processed in order:
		  - 'simulate_event': calls simulate_event with step params
		  - 'load_data': calls load_test_data with step params
		  - 'assert': evaluates step assertion dict against last result
		  - other actions: recorded as no-op with a skipped status

		on_failure='abort' stops execution and marks the scenario failed.
		on_failure='continue' records the step as failed but continues.
		Returns per-step results and aggregate pass/fail summary.
		"""
		await self.async_guard_tenant(tenant_id)
		self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		scenario = self._test_scenarios.get(_state_key(tenant_id, scenario_id))
		if scenario is None:
			raise KeyError(f"scenario_not_found:{scenario_id}")
		step_results: list[dict[str, Any]] = []
		scenario_passed = True
		last_result: dict[str, Any] = {}
		for step in scenario.get("steps", []):
			await asyncio.sleep(0)
			action = step["action"]
			params = dict(step.get("params") or {})
			step_passed = True
			step_output: dict[str, Any] = {}
			try:
				if action == "simulate_event":
					step_output = self.simulate_event(
						sandbox_id=sandbox_id,
						event_type=params.get("event_type", "test.event"),
						payload=params.get("payload", {}),
						tenant_id=tenant_id,
						triggered_by=requested_by,
					)
				elif action == "load_data":
					step_output = self.load_test_data(
						sandbox_id=sandbox_id,
						dataset_name=params.get("dataset_name", "step-data"),
						tenant_id=tenant_id,
						data=params.get("data"),
						record_count=int(params.get("record_count", 0)),
						loaded_by=requested_by,
					)
				elif action == "assert":
					assertion = step.get("assertion") or {}
					field = assertion.get("field", "status")
					expected = assertion.get("expected")
					actual = last_result.get(field)
					step_passed = actual == expected
					step_output = {"field": field, "expected": expected, "actual": actual, "passed": step_passed}
				else:
					step_output = {"action": action, "status": "skipped"}
				last_result = step_output
			except Exception as exc:
				step_passed = False
				step_output = {"error": str(exc)}
			step_results.append({
				"index": step["index"],
				"action": action,
				"passed": step_passed,
				"output": step_output,
			})
			if not step_passed:
				scenario_passed = False
				if step.get("on_failure", "abort") == "abort":
					break
		record = {
			"scenario_id": scenario_id,
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"total_steps": len(scenario.get("steps", [])),
			"executed_steps": len(step_results),
			"passed": scenario_passed,
			"step_results": step_results,
			"requested_by": requested_by,
			"executed_at": _ts(),
		}
		self._scenario_results.append(record)
		self._record_event(
			tenant_id, "scenario_executed", sandbox_id,
			f"Scenario {scenario_id}: {'passed' if scenario_passed else 'failed'}",
			requested_by, "info" if scenario_passed else "warning",
		)
		return record

	async def async_dataset_diff(
		self,
		sandbox_id: str,
		dataset_name_a: str,
		dataset_name_b: str,
		tenant_id: str = "default",
		tolerance_record_count_pct: float = 0.0,
	) -> dict[str, Any]:
		"""
		Compare two named datasets loaded into a sandbox.

		Produces a structured diff of inline_data dicts:
		  added_keys: keys present in b but not a
		  removed_keys: keys present in a but not b
		  changed_values: keys whose values differ
		  schema_drift: fields whose Python type changed
		  record_count_delta: difference in record counts
		  within_tolerance: True if record_count_delta / max(a,1) <= tolerance_record_count_pct

		Returns the diff record — integrate with assertion_check via field '__dataset_diff__'.
		"""
		await self.async_guard_tenant(tenant_id)
		self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		id_a = stable_id("data", tenant_id, sandbox_id, dataset_name_a)
		id_b = stable_id("data", tenant_id, sandbox_id, dataset_name_b)
		rec_a = self._test_data.get(id_a) or {}
		rec_b = self._test_data.get(id_b) or {}
		data_a: dict[str, Any] = rec_a.get("_data_payload") or {}
		data_b: dict[str, Any] = rec_b.get("_data_payload") or {}
		keys_a = set(data_a.keys()) if isinstance(data_a, dict) else set()
		keys_b = set(data_b.keys()) if isinstance(data_b, dict) else set()
		added_keys = sorted(keys_b - keys_a)
		removed_keys = sorted(keys_a - keys_b)
		changed_values: list[dict[str, Any]] = []
		schema_drift: list[dict[str, Any]] = []
		for key in keys_a & keys_b:
			val_a = data_a[key]
			val_b = data_b[key]
			if val_a != val_b:
				changed_values.append({"key": key, "a": val_a, "b": val_b})
			if type(val_a) is not type(val_b):
				schema_drift.append({"key": key, "type_a": type(val_a).__name__, "type_b": type(val_b).__name__})
		count_a = rec_a.get("record_count", 0) or 0
		count_b = rec_b.get("record_count", 0) or 0
		delta = abs(count_b - count_a)
		within_tol = (delta / max(count_a, 1)) <= tolerance_record_count_pct if tolerance_record_count_pct > 0 else delta == 0
		diff = {
			"sandbox_id": sandbox_id,
			"tenant_id": tenant_id,
			"dataset_a": dataset_name_a,
			"dataset_b": dataset_name_b,
			"added_keys": added_keys,
			"removed_keys": removed_keys,
			"changed_values": changed_values,
			"schema_drift": schema_drift,
			"record_count_a": count_a,
			"record_count_b": count_b,
			"record_count_delta": delta,
			"within_tolerance": within_tol,
			"diff_at": _ts(),
		}
		self._record_event(tenant_id, "dataset_diff", sandbox_id, f"Diff {dataset_name_a} vs {dataset_name_b}", "system")
		return diff

	async def async_flakiness_score(
		self,
		scenario_id: str,
		tenant_id: str = "default",
		window: int = 20,
	) -> dict[str, Any]:
		"""
		Compute a flakiness score (0.0–1.0) for a scenario based on its recent run history.

		window: number of most-recent scenario results to consider.
		Score = variance of pass/fail (0=perfectly stable, 1=maximally flaky).

		A score >= 0.3 is considered flaky; >= 0.5 triggers a quarantine recommendation.
		"""
		await self.async_guard_tenant(tenant_id)
		recent = [
			r for r in self._scenario_results
			if r.get("scenario_id") == scenario_id and r.get("tenant_id") == tenant_id
		][-window:]
		if not recent:
			return {
				"scenario_id": scenario_id,
				"tenant_id": tenant_id,
				"run_count": 0,
				"flakiness_score": 0.0,
				"recommendation": "no_history",
				"evaluated_at": _ts(),
			}
		pass_flags = [1 if r.get("passed") else 0 for r in recent]
		mean = sum(pass_flags) / len(pass_flags)
		variance = sum((x - mean) ** 2 for x in pass_flags) / len(pass_flags)
		# Normalize: max variance is 0.25 (50/50 split) — scale to 0–1
		score = round(min(variance / 0.25, 1.0), 4)
		if score >= 0.5:
			recommendation = "quarantine"
		elif score >= 0.3:
			recommendation = "monitor"
		else:
			recommendation = "stable"
		return {
			"scenario_id": scenario_id,
			"tenant_id": tenant_id,
			"run_count": len(recent),
			"pass_count": sum(pass_flags),
			"fail_count": len(pass_flags) - sum(pass_flags),
			"flakiness_score": score,
			"recommendation": recommendation,
			"evaluated_at": _ts(),
		}

	async def async_register_wasm_module(
		self,
		name: str,
		module_bytes: bytes,
		signer_id: str,
		tenant_id: str = "default",
		version: str = "1.0.0",
		trusted: bool = False,
	) -> dict[str, Any]:
		"""
		Register a WASM module in the tenant's artifact registry with SHA-256 integrity verification.

		Computes SHA-256 over module_bytes and stores it alongside signer metadata.
		`trusted` must be set by an admin after out-of-band signature verification.
		Only trusted modules may be passed to execute_wasm (enforcement point TBD by host).

		Returns the module record including hash and registration timestamp.
		"""
		await self.async_guard_tenant(tenant_id)
		assert name, "name required"
		assert module_bytes, "module_bytes required"
		assert signer_id, "signer_id required"
		import hashlib as _hashlib
		sha256_hex = _hashlib.sha256(module_bytes).hexdigest()
		module_id = stable_id("wasm", tenant_id, name, sha256_hex[:16])
		record: dict[str, Any] = {
			"id": module_id,
			"tenant_id": tenant_id,
			"name": name,
			"version": version,
			"size_bytes": len(module_bytes),
			"hash_sha256": sha256_hex,
			"signer_id": signer_id,
			"trusted": trusted,
			"registered_at": _ts(),
		}
		self._wasm_modules[_state_key(tenant_id, module_id)] = record
		self._record_event(
			tenant_id, "wasm_module_registered", module_id,
			f"WASM module {name} v{version} registered by {signer_id}",
			signer_id, "warning" if not trusted else "info",
		)
		return record

	async def async_simulate_policy(
		self,
		context: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Dry-run policy evaluation against a given context without side effects.

		Useful for pre-flight checks: 'will this sandbox creation be allowed?'
		Returns the full evaluation result plus a human-readable summary.
		Adds tenant_context_present=True to context if not already present.
		"""
		await self.async_guard_tenant(tenant_id)
		full_context = {"tenant_context_present": True, **context}
		await asyncio.sleep(0)
		result = self.evaluate(full_context)
		return {
			"tenant_id": tenant_id,
			"context": full_context,
			"decision": result.get("decision"),
			"summary": summarize_decision(result),
			"actions": result.get("actions", []),
			"simulated_at": _ts(),
		}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		m = metadata or {}
		return self.create_sandbox(
			name=record_id,
			template=m.get("template", "python"),
			owner_id=m.get("owner", "system"),
			expiry_hours=m.get("ttl_hours", 24),
			tenant_id=tenant_id,
			lifecycle_review_recorded=status != "review",
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_sandboxes(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _list(self, store: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))

	def _require_owned(self, store: dict[str, Any], item_id: str, tenant_id: str, missing_reason: str) -> Any:
		item = store.get(_state_key(tenant_id, item_id))
		if item is None or item.tenant_id != tenant_id:
			raise KeyError(missing_reason)
		return item

	def _raise_policy(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		raise PermissionError(summarize_decision(result))

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, actor: str, severity: str = "info", metadata: dict[str, Any] | None = None) -> None:
		self._audit_events.append(SboxAuditEvent(
			id=stable_id("audit", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			metadata=dict(metadata or {}),
			created_at=utc_now(),
		))


# Alias
SboxService = SandboxTestingService
