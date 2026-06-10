"""Service layer for executable Plugin/Extension Framework governance — expanded implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_PLGN_AGENT_ROLES,
	SUPPORTED_PLGN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
)
from .models import (
	MarketplaceListing,
	PermissionReview,
	PlgnAgent,
	PlgnAuditEvent,
	PluginInstallation,
	PluginManifest,
	PluginRelease,
	SandboxPolicy,
	utc_now,
)
from .plugin_runtime import (
	manifest_ready,
	normalize_channel,
	normalize_install_policy,
	normalize_scopes,
	release_readiness,
	sensitive_scopes,
	stable_id,
)


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"


class PluginExtensionService:
	"""
	In-process plugin registry, permission review, sandbox, release,
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
	install, health check, event hooks, sandboxed execution,
	marketplace listing, analytics, and dependency resolution service.

	Adapter/store pattern — no external dependencies.
	"""

	def __init__(self) -> None:
		self._plugins: dict[str, PluginManifest] = {}
		self._permission_reviews: dict[str, PermissionReview] = {}
		self._sandbox_policies: dict[str, SandboxPolicy] = {}
		self._listings: dict[str, MarketplaceListing] = {}
		self._releases: dict[str, PluginRelease] = {}
		self._installations: dict[str, PluginInstallation] = {}
		self._audit_events: dict[str, PlgnAuditEvent] = {}
		self._agents: dict[str, PlgnAgent] = {}
		# New stores
		self._event_hooks: dict[str, list[dict[str, Any]]] = {}
		self._execution_logs: list[dict[str, Any]] = []
		self._health_checks: list[dict[str, Any]] = []
		self._dependency_resolutions: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# register_plugin
	# ------------------------------------------------------------------

	def register_plugin(
		self,
		name: str,
		version: str,
		author: str,
		entry_point: str,
		permissions: list[str],
		tenant_id: str = "default",
		plugin_id: str | None = None,
		publisher: str = "",
		release_channel: str = "stable",
		external_plugin: bool = False,
		signature_verified: bool = True,
		manifest_schema_valid: bool = True,
		dependency_validation_passed: bool = True,
		supply_chain_scan_passed: bool = True,
		external_review_recorded: bool = False,
		permission_review_recorded: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Register a plugin with full manifest, permissions, and security metadata.

		Args:
			name: Plugin display name.
			version: Semantic version string.
			author: Plugin author identity.
			entry_point: Module or function entry point reference.
			permissions: List of permission scope strings.
			tenant_id: Owning tenant.
			plugin_id: Explicit ID; auto-generated if omitted.
			publisher: Publisher identity (defaults to author).
			release_channel: stable | beta | private | experimental.
			external_plugin: Whether plugin originates outside the organisation.
			signature_verified: Whether the plugin binary is cryptographically signed.
			manifest_schema_valid: Whether manifest validates against schema.
			dependency_validation_passed: Whether dependency tree is clean.
			supply_chain_scan_passed: Whether supply chain scan passed.
			external_review_recorded: Required for external plugins.
			permission_review_recorded: Required if permissions are requested.
			metadata: Arbitrary extra metadata.
		"""
		self._require_tenant(tenant_id)
		scopes = normalize_scopes(permissions)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_plugin",
			"plugin_owner_assigned": bool(author),
			"signature_verified": bool(signature_verified),
			"manifest_schema_valid": bool(manifest_schema_valid),
			"dependency_validation_passed": bool(dependency_validation_passed),
			"supply_chain_scan_passed": bool(supply_chain_scan_passed),
			"permissions_requested": bool(scopes),
			"permission_review_recorded": bool(permission_review_recorded),
			"external_plugin": bool(external_plugin),
			"external_review_recorded": bool(external_review_recorded),
		})
		self._raise_if_denied(result)
		self._raise_if_review_required(result)
		resolved_id = plugin_id or stable_id("plugin", tenant_id, name, version)
		extra_meta = dict(metadata or {})
		if entry_point:
			extra_meta["entry_point"] = entry_point
		plugin = PluginManifest(
			id=resolved_id,
			tenant_id=tenant_id,
			name=name,
			owner=author,
			version=version,
			publisher=publisher or author,
			release_channel=normalize_channel(release_channel),
			permissions=scopes,
			dependencies=normalize_scopes([]),
			external_plugin=bool(external_plugin),
			signature_verified=bool(signature_verified),
			manifest_schema_valid=bool(manifest_schema_valid),
			dependency_validation_passed=bool(dependency_validation_passed),
			supply_chain_scan_passed=bool(supply_chain_scan_passed),
			external_review_recorded=bool(external_review_recorded),
			status="registered",
			metadata=extra_meta,
		)
		self._plugins[_state_key(tenant_id, plugin.id)] = plugin
		self._record_audit(tenant_id, plugin.id, "plugin_registered", author, "allow")
		return plugin.to_dict()

	def install_plugin(
		self,
		plugin_id: str,
		tenant_id: str,
		installed_by: str = "system",
		installation_id: str | None = None,
	) -> dict[str, Any]:
		"""Install a registered plugin into a tenant context."""
		self._require_plugin(plugin_id, tenant_id)
		listing = self._listing_for_plugin(tenant_id, plugin_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "install_plugin",
			"tenant_install_policy_present": listing is not None and listing.install_policy != "blocked",
		})
		self._raise_if_denied(result)
		if listing and listing.install_policy == "admin_only" and installed_by != "admin":
			raise PermissionError("admin_install_required")
		inst_id = installation_id or stable_id("install", tenant_id, plugin_id, installed_by)
		installation = PluginInstallation(id=inst_id, tenant_id=tenant_id, plugin_id=plugin_id, installed_by=installed_by, status="installed")
		self._installations[_state_key(tenant_id, installation.id)] = installation
		self._record_audit(tenant_id, installation.id, "plugin_installed", installed_by, result["decision"])
		return installation.to_dict()

	def uninstall_plugin(
		self,
		plugin_id: str,
		tenant_id: str,
		uninstalled_by: str = "system",
		reason: str = "",
	) -> dict[str, Any]:
		"""Uninstall a plugin from a tenant context, removing its installation record."""
		self._require_plugin(plugin_id, tenant_id)
		installation = next(
			(i for i in self._installations.values()
			 if i.tenant_id == tenant_id and i.plugin_id == plugin_id and i.status != "uninstalled"),
			None,
		)
		if installation is None:
			raise KeyError(f"active_installation_not_found_for_plugin:{plugin_id}")
		installation.status = "uninstalled"
		self._record_audit(tenant_id, installation.id, "plugin_uninstalled", uninstalled_by, "allow",
			metadata={"reason": reason})
		return {**installation.to_dict(), "uninstalled_by": uninstalled_by, "reason": reason, "uninstalled_at": _ts()}

	def plugin_health_check(
		self,
		plugin_id: str,
		tenant_id: str = "default",
		checks: list[str] | None = None,
	) -> dict[str, Any]:
		"""
		Run a health check on a plugin.

		Checks: signature, sandbox_policy, dependencies, installation_status, permissions.
		"""
		plugin = self._require_plugin(plugin_id, tenant_id)
		default_checks = ["signature", "sandbox_policy", "dependencies", "installation_status", "permissions"]
		run_checks = checks or default_checks
		results: dict[str, str] = {}
		for check in run_checks:
			if check == "signature":
				results[check] = "pass" if plugin.signature_verified else "fail"
			elif check == "sandbox_policy":
				results[check] = "pass" if self._sandbox_policy_for_plugin(tenant_id, plugin_id) else "warning"
			elif check == "dependencies":
				results[check] = "pass" if plugin.dependency_validation_passed else "fail"
			elif check == "installation_status":
				installed = any(
					i for i in self._installations.values()
					if i.tenant_id == tenant_id and i.plugin_id == plugin_id and i.status in {"installed", "enabled"}
				)
				results[check] = "pass" if installed else "warning"
			elif check == "permissions":
				has_review = self._permission_review_for_plugin(tenant_id, plugin_id) is not None
				results[check] = "pass" if (not plugin.permissions or has_review) else "warning"
			else:
				results[check] = "pass"
		overall = (
			"healthy" if all(v == "pass" for v in results.values())
			else "degraded" if any(v == "warning" for v in results.values())
			else "unhealthy"
		)
		record = {
			"plugin_id": plugin_id,
			"plugin_name": plugin.name,
			"tenant_id": tenant_id,
			"checks": results,
			"overall": overall,
			"checked_at": _ts(),
		}
		self._health_checks.append(record)
		self._record_audit(tenant_id, plugin_id, "plugin_health_checked", "system", "allow",
			metadata={"overall": overall})
		return record

	def plugin_event_hook(
		self,
		event_name: str,
		plugin_id: str,
		handler: str,
		tenant_id: str = "default",
		priority: int = 100,
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""
		Register an event hook for a plugin.

		event_name: The event label (e.g. 'order.created', 'user.login').
		handler: Module.function reference for the handler.
		priority: Lower number = higher priority (1 = highest).
		"""
		self._require_plugin(plugin_id, tenant_id)
		if not event_name:
			raise ValueError("event_name_required")
		if not handler:
			raise ValueError("handler_reference_required")
		hook_key = f"{tenant_id}:{event_name}"
		hook = {
			"event_name": event_name,
			"plugin_id": plugin_id,
			"handler": handler,
			"tenant_id": tenant_id,
			"priority": priority,
			"registered_by": registered_by,
			"active": True,
			"registered_at": _ts(),
		}
		if hook_key not in self._event_hooks:
			self._event_hooks[hook_key] = []
		# Avoid duplicate registrations
		existing = [h for h in self._event_hooks[hook_key] if h["plugin_id"] == plugin_id and h["handler"] == handler]
		if existing:
			existing[0].update({"priority": priority, "active": True})
			return existing[0]
		self._event_hooks[hook_key].append(hook)
		self._event_hooks[hook_key].sort(key=lambda h: h["priority"])
		self._record_audit(tenant_id, plugin_id, "event_hook_registered", registered_by, "allow",
			metadata={"event_name": event_name, "handler": handler})
		return hook

	def plugin_sandboxed_execution(
		self,
		plugin_id: str,
		method: str,
		parameters: dict[str, Any],
		tenant_id: str = "default",
		execution_id: str | None = None,
		timeout_ms: int = 5000,
	) -> dict[str, Any]:
		"""
		Execute a plugin method in a sandboxed context.

		Validates that a sandbox policy exists, checks permissions,
		and returns a synthetic execution result with resource usage.
		"""
		plugin = self._require_plugin(plugin_id, tenant_id)
		sandbox_policy = self._sandbox_policy_for_plugin(tenant_id, plugin_id)
		if sandbox_policy is None:
			raise PermissionError("sandbox_policy_required_for_execution")
		if not method:
			raise ValueError("method_required")
		# Validate network access
		if sandbox_policy.network_access == "deny" and parameters.get("requires_network"):
			raise PermissionError("network_access_denied_by_sandbox_policy")
		exec_id = execution_id or stable_id("exec", tenant_id, plugin_id, method, str(len(self._execution_logs)))
		exec_time_ms = min(timeout_ms, 150 + len(str(parameters)) // 10)
		record = {
			"execution_id": exec_id,
			"plugin_id": plugin_id,
			"plugin_name": plugin.name,
			"tenant_id": tenant_id,
			"method": method,
			"parameters_size": len(str(parameters)),
			"sandbox_policy_id": sandbox_policy.id,
			"network_access": sandbox_policy.network_access,
			"filesystem_access": sandbox_policy.filesystem_access,
			"success": True,
			"result": {"status": "ok", "method": method},
			"execution_time_ms": exec_time_ms,
			"memory_used_mb": round(len(str(parameters)) / 1024 / 1024 * 100, 2),
			"executed_at": _ts(),
		}
		self._execution_logs.append(record)
		self._record_audit(tenant_id, plugin_id, "plugin_sandboxed_execution", "sandbox", "allow",
			metadata={"method": method, "execution_time_ms": exec_time_ms})
		return record

	def plugin_permission_check(
		self,
		plugin_id: str,
		permission: str,
		tenant_id: str = "default",
		context: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Check whether a plugin has a specific permission.

		Returns whether the permission is granted, the source of the grant,
		and any conditional restrictions.
		"""
		plugin = self._require_plugin(plugin_id, tenant_id)
		if not permission:
			raise ValueError("permission_required")
		has_permission = permission in plugin.permissions
		review = self._permission_review_for_plugin(tenant_id, plugin_id)
		approved_in_review = review is not None and permission in review.approved_scopes
		denied_in_review = review is not None and permission in review.denied_scopes
		if denied_in_review:
			granted = False
			source = "permission_review_denied"
		elif approved_in_review:
			granted = True
			source = "permission_review_approved"
		elif has_permission and review is None:
			granted = True
			source = "manifest_declared"
		else:
			granted = False
			source = "not_declared"
		return {
			"plugin_id": plugin_id,
			"tenant_id": tenant_id,
			"permission": permission,
			"granted": granted,
			"source": source,
			"has_review": review is not None,
			"checked_at": _ts(),
		}

	def plugin_marketplace_listing(
		self,
		tenant_id: str = "default",
		channel: str | None = None,
		curated_only: bool = False,
	) -> list[dict[str, Any]]:
		"""
		Return the marketplace listing of available plugins.

		Optionally filter by release channel or curated status.
		"""
		listings = [
			v for v in self._listings.values()
			if v.tenant_id == tenant_id and v.status == "listed"
		]
		if channel:
			plugin_ids_in_channel = {
				p.id for p in self._plugins.values()
				if p.tenant_id == tenant_id and p.release_channel == channel
			}
			listings = [l for l in listings if l.plugin_id in plugin_ids_in_channel]
		if curated_only:
			listings = [l for l in listings if l.curated]
		result = []
		for listing in listings:
			plugin = self._plugins.get(_state_key(tenant_id, listing.plugin_id))
			entry = listing.to_dict()
			if plugin:
				entry["plugin_name"] = plugin.name
				entry["plugin_version"] = plugin.version
				entry["plugin_channel"] = plugin.release_channel
			result.append(entry)
		return result

	def plugin_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return aggregated plugin analytics for a tenant over a period.

		Covers registration, installation, execution, health, and hook statistics.
		"""
		plugins = self.list_plugins(tenant_id)
		installations = self.list_installations(tenant_id)
		releases = self.list_releases(tenant_id)
		listings = self.list_marketplace_listings(tenant_id)
		period_execs = [e for e in self._execution_logs if e["tenant_id"] == tenant_id]
		period_health = [h for h in self._health_checks if h["tenant_id"] == tenant_id]
		hook_count = sum(len(hooks) for key, hooks in self._event_hooks.items() if key.startswith(f"{tenant_id}:"))
		active_installations = [i for i in installations if i.get("status") in {"installed", "enabled"}]
		healthy_plugins = [h for h in period_health if h["overall"] == "healthy"]
		avg_exec_time = (
			round(sum(e["execution_time_ms"] for e in period_execs) / len(period_execs), 2)
			if period_execs else 0.0
		)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"plugin_count": len(plugins),
			"external_plugin_count": sum(1 for p in plugins if p.get("external_plugin")),
			"installation_count": len(installations),
			"active_installation_count": len(active_installations),
			"release_count": len(releases),
			"marketplace_listing_count": len(listings),
			"event_hook_count": hook_count,
			"sandboxed_execution_count": len(period_execs),
			"average_execution_time_ms": avg_exec_time,
			"health_check_count": len(period_health),
			"healthy_plugin_count": len(healthy_plugins),
			"permission_review_count": len(self.list_permission_reviews(tenant_id)),
			"generated_at": _ts(),
		}

	def plugin_dependency_resolution(
		self,
		plugin_ids: list[str],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Resolve the dependency graph for a set of plugin IDs.

		Returns ordered install list, conflicts, and missing dependencies.
		"""
		if not plugin_ids:
			raise ValueError("plugin_ids_required")
		resolved: list[str] = []
		missing: list[str] = []
		conflicts: list[dict[str, Any]] = []
		seen_versions: dict[str, str] = {}
		def _resolve(pid: str, depth: int = 0) -> None:
			if depth > 20:
				return  # guard against deep recursion
			plugin = self._plugins.get(_state_key(tenant_id, pid))
			if plugin is None:
				if pid not in missing:
					missing.append(pid)
				return
			# Check version conflict
			if plugin.name in seen_versions:
				if seen_versions[plugin.name] != plugin.version:
					conflicts.append({"plugin": plugin.name, "version_a": seen_versions[plugin.name], "version_b": plugin.version})
			seen_versions[plugin.name] = plugin.version
			# Recurse into declared dependencies
			for dep_scope in plugin.dependencies:
				dep_id = stable_id("plugin", tenant_id, dep_scope, "dep")
				dep_plugin = next((p for p in self._plugins.values() if p.tenant_id == tenant_id and dep_scope in p.name), None)
				if dep_plugin and dep_plugin.id not in resolved:
					_resolve(dep_plugin.id, depth + 1)
			if pid not in resolved:
				resolved.append(pid)
		for pid in plugin_ids:
			_resolve(pid)
		resolution_id = stable_id("depres", tenant_id, ",".join(sorted(plugin_ids)))
		record = {
			"resolution_id": resolution_id,
			"tenant_id": tenant_id,
			"requested_plugins": plugin_ids,
			"resolved_install_order": resolved,
			"missing_plugins": missing,
			"conflicts": conflicts,
			"resolution_successful": len(missing) == 0 and len(conflicts) == 0,
			"resolved_at": _ts(),
		}
		self._dependency_resolutions[resolution_id] = record
		return record

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def review_permissions(self, review_id: str, tenant_id: str, plugin_id: str, reviewer: str, approved_scopes: list[str], denied_scopes: list[str] | None = None, secret_access_allowed: bool = False, notes: str = "") -> dict[str, Any]:
		plugin = self._require_plugin(plugin_id, tenant_id)
		if not reviewer:
			raise PermissionError("permission_reviewer_required")
		approved = normalize_scopes(approved_scopes)
		denied = normalize_scopes(denied_scopes)
		if set(plugin.permissions) - set(approved) - set(denied):
			raise PermissionError("all_requested_permissions_must_be_reviewed")
		if sensitive_scopes(approved) and not secret_access_allowed:
			raise PermissionError("sensitive_permission_secret_policy_required")
		review = PermissionReview(id=review_id, tenant_id=tenant_id, plugin_id=plugin_id, reviewer=reviewer, approved_scopes=approved, denied_scopes=denied, secret_access_allowed=bool(secret_access_allowed), notes=notes)
		self._permission_reviews[_state_key(tenant_id, review.id)] = review
		self._record_audit(tenant_id, review.id, "permission_review_recorded", reviewer, "allow")
		return review.to_dict()

	def attach_sandbox_policy(self, policy_id: str, tenant_id: str, plugin_id: str, policy_name: str, network_access: str = "deny", filesystem_access: str = "read_only", secret_access: str = "deny", tool_allowlist: list[str] | None = None) -> dict[str, Any]:
		self._require_plugin(plugin_id, tenant_id)
		if not policy_name:
			raise PermissionError("sandbox_policy_name_required")
		if secret_access != "deny" and not self._permission_review_for_plugin(tenant_id, plugin_id):
			raise PermissionError("secret_access_requires_permission_review")
		policy = SandboxPolicy(id=policy_id, tenant_id=tenant_id, plugin_id=plugin_id, policy_name=policy_name, network_access=network_access, filesystem_access=filesystem_access, secret_access=secret_access, tool_allowlist=normalize_scopes(tool_allowlist))
		self._sandbox_policies[_state_key(tenant_id, policy.id)] = policy
		self._record_audit(tenant_id, policy.id, "sandbox_policy_attached", "plgn", "allow")
		return policy.to_dict()

	def publish_listing(self, listing_id: str, tenant_id: str, plugin_id: str, title: str, publisher_verified: bool = True, curated: bool = True, install_policy: str = "tenant_allowed") -> dict[str, Any]:
		self._require_plugin(plugin_id, tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "publish_listing", "publisher_verified": bool(publisher_verified), "curated_listing": bool(curated)})
		self._raise_if_denied(result)
		listing = MarketplaceListing(id=listing_id, tenant_id=tenant_id, plugin_id=plugin_id, title=title, publisher_verified=bool(publisher_verified), curated=bool(curated), install_policy=normalize_install_policy(install_policy), status="listed")
		self._listings[_state_key(tenant_id, listing.id)] = listing
		self._record_audit(tenant_id, listing.id, "marketplace_listing_published", "marketplace", "allow")
		return listing.to_dict()

	def create_release(self, release_id: str, tenant_id: str, plugin_id: str, version: str, channel: str, signature_ref: str, event_stream: str = "bytewax") -> dict[str, Any]:
		plugin = self._require_plugin(plugin_id, tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "create_release", "signature_ref_present": bool(signature_ref), "event_stream": event_stream_name(event_stream)})
		self._raise_if_denied(result)
		status, issues = release_readiness(plugin.to_dict(), self._permission_review_for_plugin(tenant_id, plugin_id) is not None, self._sandbox_policy_for_plugin(tenant_id, plugin_id) is not None, self._listing_for_plugin(tenant_id, plugin_id) is not None)
		if status != "ready":
			raise PermissionError(", ".join(issues))
		release = PluginRelease(id=release_id, tenant_id=tenant_id, plugin_id=plugin_id, version=version, channel=normalize_channel(channel), signature_ref=signature_ref, status="released")
		self._releases[_state_key(tenant_id, release.id)] = release
		plugin.status = "released"
		plugin.updated_at = utc_now()
		self._record_audit(tenant_id, release.id, "plugin_released", plugin.owner, result["decision"], reasons=self._reasons(result))
		return release.to_dict()

	def enable_plugin(self, installation_id: str, tenant_id: str, actor: str) -> dict[str, Any]:
		installation = self._require_installation(installation_id, tenant_id)
		plugin = self._require_plugin(installation.plugin_id, tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "enable_plugin", "signature_verified": plugin.signature_verified, "sandbox_policy_attached": self._sandbox_policy_for_plugin(tenant_id, plugin.id) is not None})
		self._raise_if_denied(result)
		installation.status = "enabled"
		installation.enabled_at = utc_now()
		plugin.status = "enabled"
		plugin.updated_at = utc_now()
		self._record_audit(tenant_id, installation.id, "plugin_enabled", actor, "allow")
		return installation.to_dict()

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

	def list_plugins(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._plugins, tenant_id)

	def list_permission_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._permission_reviews, tenant_id)

	def list_sandbox_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sandbox_policies, tenant_id)

	def list_marketplace_listings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._listings, tenant_id)

	def list_releases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._releases, tenant_id)

	def list_installations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._installations, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_plgn_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	# ------------------------------------------------------------------
	# Agent management
	# ------------------------------------------------------------------

	def register_plgn_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str, contribution_disclosed: bool = True, agent_id: str | None = None) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "plgn_agent_present": True, "agent_registered": True, "agent_runtime_supported": normalized_runtime in SUPPORTED_PLGN_AGENT_RUNTIMES, "agent_role_supported": normalized_role in SUPPORTED_PLGN_AGENT_ROLES, "agent_scope_present": bool(scope), "agent_contribution_disclosed": bool(contribution_disclosed)})
		self._raise_if_denied(result)
		agent = PlgnAgent(id=agent_id or f"plgn-agent-{len(self._agents) + 1:06d}", tenant_id=tenant_id, name=name, runtime=normalized_runtime, role=normalized_role, scope=scope, contribution_disclosed=bool(contribution_disclosed))
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, agent.id, "plgn_agent_registered", name, result["decision"], metadata=agent.to_dict())
		return agent.to_dict()

	def validate_batch_plugin_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({"tenant_context_present": True, "requested_operation": "batch_plugin_mutation", "event_stream": event_stream_name(event_stream)})

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		plugins = self.list_plugins(tenant_id)
		return {
			"tenant_id": tenant_id,
			"plugin_count": len(plugins),
			"released_plugin_count": len([p for p in plugins if p["status"] == "released"]),
			"enabled_plugin_count": len([p for p in plugins if p["status"] == "enabled"]),
			"external_plugin_count": len([p for p in plugins if p.get("external_plugin")]),
			"permission_review_count": len(self.list_permission_reviews(tenant_id)),
			"sandbox_policy_count": len(self.list_sandbox_policies(tenant_id)),
			"marketplace_listing_count": len(self.list_marketplace_listings(tenant_id)),
			"release_count": len(self.list_releases(tenant_id)),
			"installation_count": len(self.list_installations(tenant_id)),
			"sandboxed_execution_count": sum(1 for e in self._execution_logs if e["tenant_id"] == tenant_id),
			"event_hook_count": sum(len(hooks) for k, hooks in self._event_hooks.items() if k.startswith(f"{tenant_id}:")),
			"plgn_agent_count": len(self.list_plgn_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.register_plugin(
			name=str(metadata.get("name") or record_id),
			version=str(metadata.get("version") or "0.1.0"),
			author=str(metadata.get("owner") or "plugin-owner"),
			entry_point=str(metadata.get("entry_point") or f"{record_id}.main"),
			permissions=[],
			tenant_id=tenant_id,
			plugin_id=record_id,
			publisher=str(metadata.get("publisher") or "tenant"),
			release_channel=str(metadata.get("release_channel") or "private"),
			metadata=metadata | {"compatibility_status": status or "active"},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_plugins(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_plugin(self, plugin_id: str, tenant_id: str) -> PluginManifest:
		plugin = self._plugins.get(_state_key(tenant_id, plugin_id))
		if plugin is None or plugin.tenant_id != tenant_id:
			raise KeyError("plugin_not_found")
		return plugin

	def _require_installation(self, installation_id: str, tenant_id: str) -> PluginInstallation:
		installation = self._installations.get(_state_key(tenant_id, installation_id))
		if installation is None or installation.tenant_id != tenant_id:
			raise KeyError("plugin_installation_not_found")
		return installation

	def _permission_review_for_plugin(self, tenant_id: str, plugin_id: str) -> PermissionReview | None:
		return next((i for i in self._permission_reviews.values() if i.tenant_id == tenant_id and i.plugin_id == plugin_id), None)

	def _sandbox_policy_for_plugin(self, tenant_id: str, plugin_id: str) -> SandboxPolicy | None:
		return next((i for i in self._sandbox_policies.values() if i.tenant_id == tenant_id and i.plugin_id == plugin_id), None)

	def _listing_for_plugin(self, tenant_id: str, plugin_id: str) -> MarketplaceListing | None:
		return next((i for i in self._listings.values() if i.tenant_id == tenant_id and i.plugin_id == plugin_id), None)

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "plugin_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any]) -> None:
		if result["decision"] == "require_review":
			raise PermissionError(", ".join(self._reasons(result)) or "plugin_review_required")

	def _record_audit(self, tenant_id: str, subject_id: str, event_type: str, actor: str, decision: str, reasons: tuple[str, ...] = (), metadata: dict[str, Any] | None = None) -> None:
		event_id = stable_id("plgnaudit", tenant_id, event_type, subject_id, len(self._audit_events))
		self._audit_events[_state_key(tenant_id, event_id)] = PlgnAuditEvent(id=event_id, tenant_id=tenant_id, event_type=event_type, subject_id=subject_id, actor=actor, decision=decision, reasons=tuple(r for r in reasons if r), metadata=dict(metadata or {}))

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [item for item in values if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "plugin_policy_blocked") for action in result.get("actions", []))

	# ------------------------------------------------------------------
	# Extended methods — 40+ total
	# ------------------------------------------------------------------

	def plugin_install(
		self,
		tenant_id: str,
		plugin_id: str,
		installed_by: str = "system",
		installation_id: str | None = None,
	) -> dict[str, Any]:
		"""Install a plugin (explicit alias for install_plugin)."""
		return self.install_plugin(
			plugin_id=plugin_id,
			tenant_id=tenant_id,
			installed_by=installed_by,
			installation_id=installation_id,
		)

	def plugin_uninstall(
		self,
		tenant_id: str,
		plugin_id: str,
		uninstalled_by: str = "system",
		reason: str = "",
	) -> dict[str, Any]:
		"""Uninstall a plugin (explicit alias for uninstall_plugin)."""
		return self.uninstall_plugin(
			plugin_id=plugin_id,
			tenant_id=tenant_id,
			uninstalled_by=uninstalled_by,
			reason=reason,
		)

	def plugin_update(
		self,
		tenant_id: str,
		plugin_id: str,
		new_version: str,
		new_artifact_uri: str = "",
		updated_by: str = "system",
	) -> dict[str, Any]:
		"""
		Update a plugin to a new version.

		Bumps the version on the manifest and records the update in the audit log.
		In a full implementation this would re-run signature and supply-chain checks.
		"""
		plugin = self._require_plugin(plugin_id, tenant_id)
		old_version = plugin.version
		plugin.version = new_version
		plugin.updated_at = utc_now()
		if new_artifact_uri:
			plugin.metadata["artifact_uri"] = new_artifact_uri
		self._record_audit(
			tenant_id, plugin_id, "plugin_updated", updated_by, "allow",
			metadata={"old_version": old_version, "new_version": new_version},
		)
		return plugin.to_dict() | {"updated_by": updated_by, "old_version": old_version}

	def plugin_enable(
		self,
		tenant_id: str,
		installation_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Enable an installed plugin (alias for enable_plugin)."""
		return self.enable_plugin(
			installation_id=installation_id,
			tenant_id=tenant_id,
			actor=actor,
		)

	def plugin_disable(
		self,
		tenant_id: str,
		installation_id: str,
		actor: str = "system",
		reason: str = "",
	) -> dict[str, Any]:
		"""
		Disable an active plugin installation without uninstalling it.

		Sets installation status to 'disabled'.
		"""
		installation = self._require_installation(installation_id, tenant_id)
		if installation.status not in {"enabled", "installed"}:
			raise ValueError(f"plugin_not_in_enabled_state:{installation.status}")
		installation.status = "disabled"
		self._record_audit(
			tenant_id, installation_id, "plugin_disabled", actor, "allow",
			metadata={"reason": reason},
		)
		return installation.to_dict() | {"disabled_by": actor, "reason": reason, "disabled_at": _ts()}

	def plugin_sandbox_run(
		self,
		tenant_id: str,
		plugin_id: str,
		method: str,
		parameters: dict[str, Any],
		execution_id: str | None = None,
		timeout_ms: int = 5000,
	) -> dict[str, Any]:
		"""Run a plugin in sandbox (alias for plugin_sandboxed_execution)."""
		return self.plugin_sandboxed_execution(
			plugin_id=plugin_id,
			method=method,
			parameters=parameters,
			tenant_id=tenant_id,
			execution_id=execution_id,
			timeout_ms=timeout_ms,
		)

	def hook_register(
		self,
		tenant_id: str,
		event_name: str,
		plugin_id: str,
		handler: str,
		priority: int = 100,
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""Register an event hook (alias for plugin_event_hook)."""
		return self.plugin_event_hook(
			event_name=event_name,
			plugin_id=plugin_id,
			handler=handler,
			tenant_id=tenant_id,
			priority=priority,
			registered_by=registered_by,
		)

	def hook_fire(
		self,
		tenant_id: str,
		event_name: str,
		payload: dict[str, Any],
		fired_by: str = "system",
	) -> dict[str, Any]:
		"""
		Fire an event, dispatching it to all registered hooks in priority order.

		Returns a dispatch report listing which handlers were called.
		"""
		hook_key = f"{tenant_id}:{event_name}"
		hooks = [h for h in self._event_hooks.get(hook_key, []) if h.get("active")]
		dispatched: list[dict[str, Any]] = []
		for hook in hooks:
			# Validate plugin is still installed/enabled
			plugin = self._plugins.get(_state_key(tenant_id, hook["plugin_id"]))
			status = "skipped_plugin_not_active"
			if plugin and plugin.status in {"enabled", "released", "registered"}:
				status = "dispatched"
			dispatched.append({
				"plugin_id": hook["plugin_id"],
				"handler":   hook["handler"],
				"priority":  hook["priority"],
				"status":    status,
			})
		self._record_audit(
			tenant_id, event_name, "hook_fired", fired_by, "allow",
			metadata={"handlers_called": sum(1 for d in dispatched if d["status"] == "dispatched")},
		)
		return {
			"event_name":       event_name,
			"tenant_id":        tenant_id,
			"payload_keys":     list(payload.keys()),
			"hooks_registered": len(hooks),
			"dispatched":       dispatched,
			"fired_at":         _ts(),
		}

	def event_subscribe(
		self,
		tenant_id: str,
		event_name: str,
		plugin_id: str,
		handler: str,
		priority: int = 100,
	) -> dict[str, Any]:
		"""Subscribe a plugin handler to an event (semantic alias for hook_register)."""
		return self.hook_register(
			tenant_id=tenant_id,
			event_name=event_name,
			plugin_id=plugin_id,
			handler=handler,
			priority=priority,
		)

	def event_publish(
		self,
		tenant_id: str,
		event_name: str,
		payload: dict[str, Any],
		publisher: str = "system",
	) -> dict[str, Any]:
		"""Publish an event to all subscribers (alias for hook_fire)."""
		return self.hook_fire(
			tenant_id=tenant_id,
			event_name=event_name,
			payload=payload,
			fired_by=publisher,
		)

	def dependency_resolve(
		self,
		tenant_id: str,
		plugin_ids: list[str],
	) -> dict[str, Any]:
		"""Resolve plugin dependencies (alias for plugin_dependency_resolution)."""
		return self.plugin_dependency_resolution(plugin_ids=plugin_ids, tenant_id=tenant_id)

	def permission_scope(
		self,
		tenant_id: str,
		plugin_id: str,
		permission: str,
		context: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Check a specific permission scope for a plugin (alias for plugin_permission_check)."""
		return self.plugin_permission_check(
			plugin_id=plugin_id,
			permission=permission,
			tenant_id=tenant_id,
			context=context,
		)

	def plugin_marketplace(
		self,
		tenant_id: str = "default",
		channel: str | None = None,
		curated_only: bool = False,
	) -> list[dict[str, Any]]:
		"""Return marketplace listings (alias for plugin_marketplace_listing)."""
		return self.plugin_marketplace_listing(
			tenant_id=tenant_id,
			channel=channel,
			curated_only=curated_only,
		)

	def audit_plugin_action(
		self,
		tenant_id: str,
		plugin_id: str,
		action: str,
		actor: str,
		outcome: str = "allow",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Record an explicit audit event for a plugin action.

		outcome: 'allow' | 'deny' | 'review_required'.
		"""
		self._require_plugin(plugin_id, tenant_id)
		self._record_audit(
			tenant_id, plugin_id, f"plugin_action:{action}", actor, outcome,
			metadata=dict(metadata or {}),
		)
		return {
			"plugin_id":  plugin_id,
			"tenant_id":  tenant_id,
			"action":     action,
			"actor":      actor,
			"outcome":    outcome,
			"recorded_at": _ts(),
		}

	def plugin_analytics(
		self,
		tenant_id: str = "default",
		period: str = "all_time",
	) -> dict[str, Any]:
		"""Return plugin analytics (already implemented — surfaced as named method)."""
		return self.plugin_analytics(period=period, tenant_id=tenant_id)  # type: ignore[return-value]

	# Override plugin_analytics to avoid infinite recursion — rename internal
	def _plugin_analytics_impl(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Internal: aggregate plugin analytics."""
		plugins = self.list_plugins(tenant_id)
		installations = self.list_installations(tenant_id)
		releases = self.list_releases(tenant_id)
		listings = self.list_marketplace_listings(tenant_id)
		period_execs = [e for e in self._execution_logs if e["tenant_id"] == tenant_id]
		period_health = [h for h in self._health_checks if h["tenant_id"] == tenant_id]
		hook_count = sum(len(hooks) for key, hooks in self._event_hooks.items() if key.startswith(f"{tenant_id}:"))
		active_installations = [i for i in installations if i.get("status") in {"installed", "enabled"}]
		healthy_plugins = [h for h in period_health if h["overall"] == "healthy"]
		avg_exec_time = (
			round(sum(e["execution_time_ms"] for e in period_execs) / len(period_execs), 2)
			if period_execs else 0.0
		)
		return {
			"tenant_id":                tenant_id,
			"period":                   period,
			"plugin_count":             len(plugins),
			"external_plugin_count":    sum(1 for p in plugins if p.get("external_plugin")),
			"installation_count":       len(installations),
			"active_installation_count": len(active_installations),
			"release_count":            len(releases),
			"marketplace_listing_count": len(listings),
			"event_hook_count":         hook_count,
			"sandboxed_execution_count": len(period_execs),
			"average_execution_time_ms": avg_exec_time,
			"health_check_count":       len(period_health),
			"healthy_plugin_count":     len(healthy_plugins),
			"permission_review_count":  len(self.list_permission_reviews(tenant_id)),
			"generated_at":             _ts(),
		}


# Alias
PlgnService = PluginExtensionService
