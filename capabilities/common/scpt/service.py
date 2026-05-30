"""Executable service layer for APG Custom Scripting Engine."""

from __future__ import annotations

from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .models import (
	ScptAuditEvent,
	ScriptApproval,
	ScriptDefinition,
	ScriptExecution,
	ScriptPackagePolicy,
	ScriptSandbox,
	ScriptingAgent,
)
from .script_runtime import (
	detect_dangerous_permissions,
	execution_status,
	normalize_isolation_mode,
	normalize_language,
	normalize_permissions,
	normalize_sandbox_state,
	normalize_tags,
	python_imports,
	source_checksum,
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
		self._agents: dict[str, ScriptingAgent] = {}
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
		approval_evidence_ref: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		allowed = normalize_tags(allowed_packages)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "create_package_policy",
			"package_policy_owner_assigned": bool(str(owner or "").strip()),
			"package_allowlist_present": bool(allowed),
			"secret_access_requested": bool(secret_access_allowed),
			"filesystem_access_requested": bool(filesystem_access_allowed),
			"network_access_requested": bool(network_policy_attached),
			"network_policy_attached": bool(network_policy_attached),
			"approval_recorded": bool(approved_by and approval_evidence_ref),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		policy = ScriptPackagePolicy(
			id=stable_id("pkg", tenant_id, name, owner),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			allowed_packages=allowed,
			blocked_imports=normalize_tags(blocked_imports),
			secret_access_allowed=secret_access_allowed,
			filesystem_access_allowed=filesystem_access_allowed,
			network_policy_attached=network_policy_attached,
			approved_by=approved_by,
			approval_evidence_ref=approval_evidence_ref,
			created_at=utc_now(),
			updated_at=utc_now(),
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
		runtime_language: str = "python",
		isolation_mode: str = "process",
		network_enabled: bool = False,
		network_policy_attached: bool = False,
		health_check_ref: str = "local://sandbox-health",
		resource_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		runtime_language = normalize_language(runtime_language)
		isolation_mode = normalize_isolation_mode(isolation_mode)
		policy_context = {
			"tenant_context_present": True,
			"operation": "create_sandbox",
			"sandbox_owner_assigned": bool(str(owner or "").strip()),
			"sandbox_limits_positive": max_runtime_seconds > 0 and max_memory_mb > 0,
			"health_check_attached": bool(str(health_check_ref or "").strip()),
			"network_access_requested": network_enabled,
			"network_policy_attached": network_policy_attached,
			"requested_memory_mb": max_memory_mb,
			"resource_review_recorded": resource_review_recorded,
			"state_change_requested": True,
			"audit_event_recorded": True,
		}
		result = self.evaluate(policy_context)
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not resource_review_recorded):
			self._raise_policy_result(result)
		sandbox = ScriptSandbox(
			id=stable_id("sbox", tenant_id, name, owner),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			max_runtime_seconds=max_runtime_seconds,
			max_memory_mb=max_memory_mb,
			runtime_language=runtime_language,
			isolation_mode=isolation_mode,
			network_enabled=network_enabled,
			network_policy_attached=network_policy_attached,
			resource_review_recorded=resource_review_recorded,
			health_check_ref=health_check_ref,
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
		source_present = bool(str(source or "").strip())
		if language == "python":
			errors = validate_python_source(source)
			if errors:
				raise ValueError(",".join(errors))
		requested = normalize_permissions(requested_permissions)
		dangerous = detect_dangerous_permissions(language, source, requested)
		policy = self._package_policies.get(package_policy_id or "") if package_policy_id else None
		if package_policy_id and (policy is None or policy.tenant_id != tenant_id):
			raise KeyError("package_policy_not_found")
		sandbox = self._sandboxes.get(sandbox_id or "") if sandbox_id else None
		if sandbox_id and (sandbox is None or sandbox.tenant_id != tenant_id):
			raise KeyError("sandbox_not_found")
		imports = python_imports(source) if language == "python" else []
		blocked = sorted({imported.split(".", 1)[0] for imported in imports} & set(policy.blocked_imports if policy else []))
		policy_context = {
			"tenant_context_present": True,
			"operation": "create_script",
			"script_owner_assigned": bool(str(owner or "").strip()),
			"script_name_present": bool(str(name or "").strip()),
			"script_source_present": source_present,
			"package_policy_attached": policy is not None,
			"sandbox_attached": sandbox is not None,
			"blocked_import_detected": bool(blocked),
			"dangerous_permission_requested": bool(dangerous),
			"approval_recorded": approval_recorded,
			"network_access_requested": "network" in dangerous,
			"network_policy_attached": bool(policy and policy.network_policy_attached),
			"requested_memory_mb": 0,
			"resource_review_recorded": True,
		}
		result = self.evaluate(policy_context)
		if result["decision"] != "allow":
			self._raise_policy_result(result)
		script = ScriptDefinition(
			id=stable_id("scr", tenant_id, name, language, 1),
			tenant_id=tenant_id,
			name=name,
			language=language,
			source=source,
			owner=owner,
			source_checksum=source_checksum(source),
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

	def request_script_review(self, tenant_id: str, script_id: str, reviewer: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		if not str(reviewer or "").strip():
			raise PermissionError("script_reviewer_required")
		if not str(reason or "").strip():
			raise PermissionError("script_review_reason_required")
		script.review_status = "approved"
		script.reviewed_by = reviewer
		script.review_reason = reason
		script.state = "review"
		script.updated_at = utc_now()
		self._record_event(tenant_id, "script_reviewed", script.id, f"Script {script.name} reviewed.", reviewer)
		return script.to_dict()

	def approve_script(self, tenant_id: str, script_id: str, approver: str, reason: str, approval_type: str = "publish", evidence_ref: str = "") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		if not approver:
			raise PermissionError("script_approver_required")
		if not reason:
			raise PermissionError("script_approval_reason_required")
		approval = ScriptApproval(
			id=stable_id("apr", tenant_id, script_id, approver, reason),
			tenant_id=tenant_id,
			script_id=script_id,
			reason=reason,
			approver=approver,
			approval_type=approval_type,
			evidence_ref=evidence_ref,
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
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "publish_script",
			"script_reviewed": script.review_status == "approved",
			"dangerous_permission_requested": bool(script.dangerous_permissions),
			"approval_recorded": script.approval_recorded or not script.dangerous_permissions,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		self._require_owned(self._package_policies, script.package_policy_id or "", tenant_id, "package_policy_not_found")
		self._require_owned(self._sandboxes, script.sandbox_id or "", tenant_id, "sandbox_not_found")
		script.state = "published"
		script.published_at = utc_now()
		script.published_by = actor
		script.updated_at = utc_now()
		self._record_event(tenant_id, "script_published", script.id, f"Script {script.name} published.", actor)
		return script.to_dict()

	def bind_workflow(self, tenant_id: str, script_id: str, workflow_binding: str, actor: str, policy_ref: str = "wflo://script-binding") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		if script.state != "published":
			raise PermissionError("published_script_required")
		if not workflow_binding:
			raise ValueError("workflow_binding_required")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "bind_workflow",
			"workflow_binding_policy_attached": bool(str(policy_ref or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
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
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		sandbox_id = sandbox_id or script.sandbox_id
		sandbox = self._sandboxes.get(sandbox_id or "") if sandbox_id else None
		if sandbox_id and (sandbox is None or sandbox.tenant_id != tenant_id):
			raise KeyError("sandbox_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "execute_script",
			"sandbox_attached": sandbox is not None,
			"script_published": script.state == "published",
			"sandbox_ready": bool(sandbox and sandbox.state == "ready"),
			"requested_by_present": bool(str(requested_by or "").strip()),
			"event_stream": event_stream,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		if "network" in script.dangerous_permissions and not sandbox.network_policy_attached:
			self._raise_policy({"tenant_context_present": True, "network_access_requested": True, "network_policy_attached": False})
		execution = ScriptExecution(
			id=stable_id("exec", tenant_id, script_id, sandbox_id, len(self._executions) + 1),
			tenant_id=tenant_id,
			script_id=script_id,
			sandbox_id=sandbox_id,
			requested_by=requested_by,
			event_stream=event_stream,
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
		completion_evidence_ref: str = "execution://local-evidence",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._require_owned(self._executions, execution_id, tenant_id, "execution_not_found")
		sandbox = self._require_owned(self._sandboxes, execution.sandbox_id, tenant_id, "sandbox_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "complete_execution",
			"completion_evidence_present": bool(str(completion_evidence_ref or "").strip()),
			"execution_metrics_valid": runtime_seconds >= 0 and memory_mb >= 0,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
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
		execution.timed_out = timed_out
		execution.completion_evidence_ref = completion_evidence_ref
		execution.logs.extend(logs or [])
		execution.completed_at = utc_now()
		sandbox.state = "ready"
		sandbox.updated_at = utc_now()
		self._record_event(tenant_id, "script_execution_completed", execution.id, f"Execution completed with status {execution.status}.", execution.requested_by, "warning" if execution.status != "succeeded" else "info")
		return execution.to_dict()

	def cancel_execution(self, tenant_id: str, execution_id: str, actor: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._require_owned(self._executions, execution_id, tenant_id, "execution_not_found")
		sandbox = self._require_owned(self._sandboxes, execution.sandbox_id, tenant_id, "sandbox_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "cancel_execution",
			"cancel_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		execution.status = "cancelled"
		execution.cancel_reason = reason
		execution.completed_at = utc_now()
		sandbox.state = "ready"
		sandbox.updated_at = utc_now()
		self._record_event(tenant_id, "script_execution_cancelled", execution.id, f"Execution cancelled: {reason}", actor, "warning")
		return execution.to_dict()

	def change_sandbox_state(self, tenant_id: str, sandbox_id: str, state: str, actor: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		sandbox = self._require_owned(self._sandboxes, sandbox_id, tenant_id, "sandbox_not_found")
		target_state = normalize_sandbox_state(state)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "change_sandbox_state",
			"target_sandbox_state": target_state,
			"state_change_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		sandbox.state = target_state
		sandbox.state_reason = reason
		sandbox.updated_at = utc_now()
		self._record_event(tenant_id, "sandbox_state_changed", sandbox.id, f"Sandbox {sandbox.name} changed to {target_state}.", actor, "warning" if target_state != "ready" else "info")
		return sandbox.to_dict()

	def retire_script(self, tenant_id: str, script_id: str, actor: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		script = self._require_owned(self._scripts, script_id, tenant_id, "script_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "retire_script",
			"retirement_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		script.state = "retired"
		script.retired_at = utc_now()
		script.retired_by = actor
		script.retirement_reason = reason
		script.updated_at = utc_now()
		self._record_event(tenant_id, "script_retired", script.id, f"Script {script.name} retired: {reason}", actor, "warning")
		return script.to_dict()

	def register_scripting_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope_ref: str,
		registered_by: str,
		contribution_disclosed: bool,
	) -> dict[str, Any]:
		config = DEFAULT_CONFIGURATION["scripting_agents"]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"scripting_agent_present": True,
			"agent_registered": bool(name and registered_by),
			"agent_runtime_supported": runtime in config["supported_runtimes"],
			"agent_scope_present": bool(str(scope_ref or "").strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		if role not in config["allowed_roles"]:
			raise PermissionError("scripting_agent_role_not_supported")
		if not self._scope_exists_for_tenant(tenant_id, scope_ref):
			raise KeyError("scripting_agent_scope_not_found")
		agent = ScriptingAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime,
			role=role,
			scope_ref=scope_ref,
			registered_by=registered_by,
			contribution_disclosed=bool(contribution_disclosed),
			created_at=utc_now(),
		)
		self._agents[agent.id] = agent
		self._record_event(tenant_id, "scripting_agent_registered", agent.id, f"Scripting agent {name} registered.", registered_by)
		return agent.to_dict()

	def validate_batch_mutation(self, event_stream: str) -> dict[str, Any]:
		result = self.evaluate({"tenant_context_present": True, "operation": "batch_script_mutation", "event_stream": event_stream})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		return result

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
			script = self.request_script_review(tenant_id, script["id"], owner, "compatibility review")
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

	def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

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
			"agent_count": len(self.list_agents(tenant_id)),
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
		self._raise_policy_result(result)

	def _raise_policy_result(self, result: dict[str, Any]) -> None:
		raise PermissionError(summarize_decision(result))

	def _scope_exists_for_tenant(self, tenant_id: str, scope_ref: str) -> bool:
		for store in (self._package_policies, self._sandboxes, self._scripts, self._approvals, self._executions):
			item = store.get(scope_ref)
			if item is not None:
				return item.tenant_id == tenant_id
		return False

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
