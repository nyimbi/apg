"""Executable service layer for the APG Sandbox/Testing Environment."""

from __future__ import annotations

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


class SboxService:
	"""Tenant-scoped sandbox, dataset, and test-run runtime."""

	def __init__(self) -> None:
		self._isolation_profiles: dict[str, IsolationProfile] = {}
		self._templates: dict[str, SandboxTemplate] = {}
		self._datasets: dict[str, SandboxDataset] = {}
		self._sandboxes: dict[str, SandboxEnvironment] = {}
		self._runs: dict[str, SandboxRun] = {}
		self._agents: dict[str, SboxAgent] = {}
		self._audit_events: list[SboxAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

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
			self._raise_policy({
				"tenant_context_present": True,
				"isolation_profile_attached": True,
				"outbound_network_requested": True,
				"network_approval_recorded": False,
			})
		if not secret_redaction_enabled:
			self._raise_policy({
				"tenant_context_present": True,
				"secret_access_requested": True,
				"secret_redaction_enabled": False,
			})
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

	def create_sandbox(
		self,
		tenant_id: str,
		name: str,
		template_id: str,
		isolation_profile_id: str,
		owner: str,
		ttl_hours: int | None = None,
		dataset_ids: list[str] | None = None,
		lifecycle_review_recorded: bool = False,
		secret_access_requested: bool = False,
		outbound_network_requested: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			self._raise_policy({"tenant_context_present": True, "operation": "create_sandbox", "sandbox_owner_assigned": False})
		template = self._require_owned(self._templates, template_id, tenant_id, "template_not_found")
		isolation = self._require_owned(self._isolation_profiles, isolation_profile_id, tenant_id, "isolation_profile_not_found")
		dataset_ids = list(dataset_ids or [])
		datasets = [self._require_owned(self._datasets, dataset_id, tenant_id, "dataset_not_found") for dataset_id in dataset_ids]
		ttl_hours = int(ttl_hours if ttl_hours is not None else template.default_ttl_hours)
		if ttl_hours <= 0:
			raise ValueError("ttl_hours_must_be_positive")
		policy_context = {
			"tenant_context_present": True,
			"operation": "create_sandbox",
			"sandbox_owner_assigned": bool(owner),
			"template_present": bool(template_id),
			"isolation_profile_attached": bool(isolation_profile_id),
			"secret_access_requested": secret_access_requested,
			"secret_redaction_enabled": isolation.secret_redaction_enabled,
			"outbound_network_requested": outbound_network_requested,
			"network_approval_recorded": isolation.network_approval_recorded,
			"ttl_hours": ttl_hours,
			"lifecycle_review_recorded": lifecycle_review_recorded,
		}
		result = self.evaluate(policy_context)
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not lifecycle_review_recorded):
			raise PermissionError(summarize_decision(result))
		dataset_type = datasets[0].dataset_type if datasets else "synthetic"
		score = risk_score(ttl_hours, outbound_network_requested, secret_access_requested, dataset_type, isolation.level)
		sandbox = SandboxEnvironment(
			id=stable_id("sbox", tenant_id, name, template_id),
			tenant_id=tenant_id,
			name=name,
			template_id=template_id,
			isolation_profile_id=isolation_profile_id,
			owner=owner,
			ttl_hours=ttl_hours,
			dataset_ids=dataset_ids,
			state=sandbox_state(ttl_hours, approved=True),
			lifecycle_review_recorded=lifecycle_review_recorded,
			secret_access_requested=secret_access_requested,
			outbound_network_requested=outbound_network_requested,
			risk_score=score,
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._sandboxes[_state_key(tenant_id, sandbox.id)] = sandbox
		self._record_event(tenant_id, "sandbox_created", sandbox.id, f"Sandbox {name} created.", owner, "warning" if score >= 50 else "info")
		return sandbox.to_dict()

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
		if run_type == "plugin":
			if not template.plugin_test_policy_required:
				raise PermissionError("plugin_test_policy_required")
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

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility shim for package tooling that expects create_record."""
		self._require_tenant(tenant_id)
		template = self.create_template(tenant_id, f"{record_id}-template", "python", metadata.get("owner", "system") if metadata else "system")
		isolation = self.create_isolation_profile(tenant_id, f"{record_id}-isolation", "strict", approved_by="system")
		return self.create_sandbox(
			tenant_id=tenant_id,
			name=record_id,
			template_id=template["id"],
			isolation_profile_id=isolation["id"],
			owner=metadata.get("owner", "system") if metadata else "system",
			ttl_hours=metadata.get("ttl_hours", 24) if metadata else 24,
			lifecycle_review_recorded=status != "review",
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_sandboxes(tenant_id)

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
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.audit_events(tenant_id)

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

	def validate_batch_sandbox_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_sandbox_mutation",
			"event_stream": event_stream_name(event_stream),
		})

	def list_sbox_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		sandboxes = self.list_sandboxes(tenant_id)
		runs = self.list_runs(tenant_id)
		return {
			"tenant_id": tenant_id,
			"sandbox_count": len(sandboxes),
			"ready_count": len([sandbox for sandbox in sandboxes if sandbox["state"] == "ready"]),
			"running_count": len([sandbox for sandbox in sandboxes if sandbox["state"] == "running"]),
			"failed_count": len([sandbox for sandbox in sandboxes if sandbox["state"] == "failed"]),
			"dataset_count": len(self.list_datasets(tenant_id)),
			"run_count": len(runs),
			"passed_run_count": len([run for run in runs if run["status"] == "passed"]),
			"blocked_run_count": len([run for run in runs if run["status"] == "blocked"]),
			"high_risk_count": len([sandbox for sandbox in sandboxes if sandbox["risk_score"] >= 50]),
			"sbox_agent_count": len(self.list_sbox_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

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


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"
