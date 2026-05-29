"""Executable service layer for APG Custom Scripting Engine."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	ScptAuditEvent,
	ScriptApproval,
	ScriptDefinition,
	ScriptExecution,
	ScriptPackagePolicy,
	ScriptSandbox,
)
from .script_runtime import (
	detect_dangerous_permissions,
	execution_status,
	normalize_language,
	normalize_permissions,
	normalize_tags,
	stable_id,
	summarize_decision,
	utc_now,
	validate_python_source,
)


class ScptService:
	"""Tenant-scoped script registry, sandbox, package policy, and execution runtime."""

	def __init__(self) -> None:
		self._package_policies: dict[str, ScriptPackagePolicy] = {}
		self._sandboxes: dict[str, ScriptSandbox] = {}
		self._scripts: dict[str, ScriptDefinition] = {}
		self._approvals: dict[str, ScriptApproval] = {}
		self._executions: dict[str, ScriptExecution] = {}
		self._audit_events: list[ScptAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_package_policy(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		allowed_packages: list[str] | None = None,
		blocked_imports: list[str] | None = None,
		secret_access_allowed: bool = False,
		filesystem_access_allowed: bool = False,
		network_policy_attached: bool = False,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("package_policy_owner_required")
		policy = ScriptPackagePolicy(
			id=stable_id("pkg", tenant_id, name, owner),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			allowed_packages=normalize_tags(allowed_packages),
			blocked_imports=normalize_tags(blocked_imports),
			secret_access_allowed=secret_access_allowed,
			filesystem_access_allowed=filesystem_access_allowed,
			network_policy_attached=network_policy_attached,
			approved_by=approved_by,
			created_at=utc_now(),
		)
		self._package_policies[policy.id] = policy
		self._record_event(tenant_id, "package_policy_created", policy.id, f"Package policy {name} created.", owner)
		return policy.to_dict()

	def create_sandbox(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		max_runtime_seconds: int = 300,
		max_memory_mb: int = 512,
		network_enabled: bool = False,
		network_policy_attached: bool = False,
		resource_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("sandbox_owner_required")
		if max_runtime_seconds <= 0 or max_memory_mb <= 0:
			raise ValueError("sandbox_resource_limits_must_be_positive")
		policy_context = {
			"tenant_context_present": True,
			"network_access_requested": network_enabled,
			"network_policy_attached": network_policy_attached,
			"requested_memory_mb": max_memory_mb,
			"resource_review_recorded": resource_review_recorded,
		}
		result = self.evaluate(policy_context)
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not resource_review_recorded):
			raise PermissionError(summarize_decision(result))
		sandbox = ScriptSandbox(
			id=stable_id("sbox", tenant_id, name, owner),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			max_runtime_seconds=max_runtime_seconds,
			max_memory_mb=max_memory_mb,
			network_enabled=network_enabled,
			network_policy_attached=network_policy_attached,
			resource_review_recorded=resource_review_recorded,
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._sandboxes[sandbox.id] = sandbox
		self._record_event(tenant_id, "sandbox_created", sandbox.id, f"Sandbox {name} created.", owner)
		return sandbox.to_dict()

	def create_script(
		self,
		tenant_id: str,
		name: str,
		language: str,
		source: str,
		owner: str,
		requested_permissions: list[str] | None = None,
		package_policy_id: str | None = None,
		sandbox_id: str | None = None,
		approval_recorded: bool = False,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		language = normalize_language(language)
		if not owner:
			self._raise_policy({"tenant_context_present": True, "operation": "create_script", "script_owner_assigned": False})
		if not source.strip():
			raise ValueError("script_source_required")
		if language == "python":
			errors = validate_python_source(source)
			if errors:
				raise ValueError(",".join(errors))
		requested = normalize_permissions(requested_permissions)
		dangerous = detect_dangerous_permissions(language, source, requested)
		policy = self._package_policies.get(package_policy_id or "") if package_policy_id else None
		if package_policy_id and (policy is None or policy.tenant_id != tenant_id):
			raise KeyError("package_policy_not_found")
		if sandbox_id:
			self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		policy_context = {
			"tenant_context_present": True,
			"operation": "create_script",
			"script_owner_assigned": bool(owner),
			"dangerous_permission_requested": bool(dangerous),
			"approval_recorded": approval_recorded,
			"network_access_requested": "network" in dangerous,
			"network_policy_attached": bool(policy and policy.network_policy_attached),
			"requested_memory_mb": 0,
			"resource_review_recorded": True,
		}
		result = self.evaluate(policy_context)
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		script = ScriptDefinition(
			id=stable_id("scr", tenant_id, name, language, 1),
			tenant_id=tenant_id,
			name=name,
			language=language,
			source=source,
			owner=owner,
			requested_permissions=requested,
			dangerous_permissions=dangerous,
			approval_recorded=approval_recorded,
			package_policy_id=package_policy_id,
			sandbox_id=sandbox_id,
			tags=normalize_tags(tags),
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._scripts[script.id] = script
		self._record_event(tenant_id, "script_created", script.id, f"Script {name} created.", owner)
		return script.to_dict()

	def approve_script(self, tenant_id: str, script_id: str, approver: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		if not approver:
			raise PermissionError("script_approver_required")
		approval = ScriptApproval(
			id=stable_id("apr", tenant_id, script_id, approver, reason),
			tenant_id=tenant_id,
			script_id=script_id,
			reason=reason,
			approver=approver,
			created_at=utc_now(),
			decided_at=utc_now(),
		)
		script.approval_recorded = True
		script.updated_at = utc_now()
		self._approvals[approval.id] = approval
		self._record_event(tenant_id, "script_approved", script_id, f"Script {script.name} approved.", approver)
		return approval.to_dict()

	def publish_script(self, tenant_id: str, script_id: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		if script.dangerous_permissions and not script.approval_recorded:
			self._raise_policy({
				"tenant_context_present": True,
				"dangerous_permission_requested": True,
				"approval_recorded": False,
			})
		if not script.package_policy_id:
			raise PermissionError("package_allowlist_required")
		if not script.sandbox_id:
			raise PermissionError("sandbox_required")
		script.state = "published"
		script.published_at = utc_now()
		script.updated_at = utc_now()
		self._record_event(tenant_id, "script_published", script.id, f"Script {script.name} published.", actor)
		return script.to_dict()

	def bind_workflow(self, tenant_id: str, script_id: str, workflow_binding: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		if script.state != "published":
			raise PermissionError("published_script_required")
		if not workflow_binding:
			raise ValueError("workflow_binding_required")
		if workflow_binding not in script.workflow_bindings:
			script.workflow_bindings.append(workflow_binding)
			script.updated_at = utc_now()
			self._record_event(tenant_id, "workflow_bound", script.id, f"Script bound to {workflow_binding}.", actor)
		return script.to_dict()

	def execute_script(
		self,
		tenant_id: str,
		script_id: str,
		sandbox_id: str | None,
		requested_by: str,
		input_payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		if script.state != "published":
			raise PermissionError("published_script_required")
		sandbox_id = sandbox_id or script.sandbox_id
		if not sandbox_id:
			self._raise_policy({"tenant_context_present": True, "operation": "execute_script", "sandbox_attached": False})
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		if sandbox.state != "ready":
			raise PermissionError("sandbox_not_ready")
		if "network" in script.dangerous_permissions and not sandbox.network_policy_attached:
			self._raise_policy({"tenant_context_present": True, "network_access_requested": True, "network_policy_attached": False})
		execution = ScriptExecution(
			id=stable_id("exec", tenant_id, script_id, sandbox_id, len(self._executions) + 1),
			tenant_id=tenant_id,
			script_id=script_id,
			sandbox_id=sandbox_id,
			requested_by=requested_by,
			status="running",
			input_payload=dict(input_payload or {}),
			started_at=utc_now(),
			logs=[f"Started script {script.name} in sandbox {sandbox.name}."],
		)
		sandbox.state = "running"
		sandbox.updated_at = utc_now()
		self._executions[execution.id] = execution
		self._record_event(tenant_id, "script_execution_started", execution.id, f"Execution started for script {script.name}.", requested_by)
		return execution.to_dict()

	def complete_execution(
		self,
		tenant_id: str,
		execution_id: str,
		exit_code: int = 0,
		output: dict[str, Any] | None = None,
		error: str | None = None,
		runtime_seconds: float = 0.0,
		memory_mb: int = 0,
		timed_out: bool = False,
		logs: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._require_owned(self._executions, execution_id, tenant_id, "execution_not_found")
		sandbox = self._require_owned(self._sandboxes, execution.sandbox_id, tenant_id, "sandbox_not_found")
		if runtime_seconds > sandbox.max_runtime_seconds:
			timed_out = True
			error = error or "max_runtime_seconds_exceeded"
		if memory_mb > sandbox.max_memory_mb:
			exit_code = exit_code or 1
			error = error or "max_memory_mb_exceeded"
		execution.status = execution_status(exit_code, timed_out)
		execution.output = dict(output or {})
		execution.error = error
		execution.runtime_seconds = runtime_seconds
		execution.memory_mb = memory_mb
		execution.logs.extend(logs or [])
		execution.completed_at = utc_now()
		sandbox.state = "ready"
		sandbox.updated_at = utc_now()
		self._record_event(tenant_id, "script_execution_completed", execution.id, f"Execution completed with status {execution.status}.", execution.requested_by, "warning" if execution.status != "succeeded" else "info")
		return execution.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility shim for package tooling that expects create_record."""
		self._require_tenant(tenant_id)
		metadata = metadata or {}
		owner = metadata.get("owner", "system")
		policy = self.create_package_policy(tenant_id, f"{record_id}-policy", owner, allowed_packages=metadata.get("allowed_packages", ["stdlib"]))
		sandbox = self.create_sandbox(tenant_id, f"{record_id}-sandbox", owner)
		script = self.create_script(
			tenant_id,
			record_id,
			metadata.get("language", "python"),
			metadata.get("source", "result = input_payload"),
			owner,
			package_policy_id=policy["id"],
			sandbox_id=sandbox["id"],
			tags=metadata.get("tags"),
		)
		if status == "active":
			return self.publish_script(tenant_id, script["id"], owner)
		return script

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_scripts(tenant_id)

	def list_package_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._package_policies, tenant_id)

	def list_sandboxes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sandboxes, tenant_id)

	def list_scripts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._scripts, tenant_id)

	def list_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._approvals, tenant_id)

	def list_executions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._executions, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		scripts = self.list_scripts(tenant_id)
		executions = self.list_executions(tenant_id)
		return {
			"script_count": len(scripts),
			"published_script_count": sum(1 for item in scripts if item["state"] == "published"),
			"sandbox_count": len(self.list_sandboxes(tenant_id)),
			"package_policy_count": len(self.list_package_policies(tenant_id)),
			"approval_count": len(self.list_approvals(tenant_id)),
			"execution_count": len(executions),
			"succeeded_execution_count": sum(1 for item in executions if item["status"] == "succeeded"),
			"failed_execution_count": sum(1 for item in executions if item["status"] == "failed"),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			self._raise_policy({"tenant_context_present": False})

	def _require_owned(self, store: dict[str, Any], object_id: str, tenant_id: str, missing_reason: str) -> Any:
		item = store.get(object_id)
		if item is None or item.tenant_id != tenant_id:
			raise KeyError(missing_reason)
		return item

	def _raise_policy(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		raise PermissionError(summarize_decision(result))

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, actor: str, severity: str = "info") -> None:
		event = ScptAuditEvent(
			id=stable_id("evt", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			created_at=utc_now(),
		)
		self._audit_events.append(event)

	def _list(self, store: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = store.values()
		if tenant_id is not None:
			values = [value for value in values if value.tenant_id == tenant_id]
		return [value.to_dict() for value in values]
