"""Service layer for executable Plugin/Extension Framework governance."""

from __future__ import annotations

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


class PlgnService:
	"""In-process plugin registry, permission review, sandbox, release, and install service."""

	def __init__(self) -> None:
		self._plugins: dict[str, PluginManifest] = {}
		self._permission_reviews: dict[str, PermissionReview] = {}
		self._sandbox_policies: dict[str, SandboxPolicy] = {}
		self._listings: dict[str, MarketplaceListing] = {}
		self._releases: dict[str, PluginRelease] = {}
		self._installations: dict[str, PluginInstallation] = {}
		self._audit_events: dict[str, PlgnAuditEvent] = {}
		self._agents: dict[str, PlgnAgent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_plugin(
		self,
		plugin_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		version: str,
		publisher: str,
		release_channel: str = "stable",
		permissions: list[str] | None = None,
		dependencies: list[str] | None = None,
		external_plugin: bool = False,
		signature_verified: bool = True,
		manifest_schema_valid: bool = True,
		dependency_validation_passed: bool = True,
		supply_chain_scan_passed: bool = True,
		external_review_recorded: bool = False,
		permission_review_recorded: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		scopes = normalize_scopes(permissions)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_plugin",
			"plugin_owner_assigned": bool(owner),
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
		plugin = PluginManifest(
			id=plugin_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			version=version,
			publisher=publisher,
			release_channel=normalize_channel(release_channel),
			permissions=scopes,
			dependencies=normalize_scopes(dependencies),
			external_plugin=bool(external_plugin),
			signature_verified=bool(signature_verified),
			manifest_schema_valid=bool(manifest_schema_valid),
			dependency_validation_passed=bool(dependency_validation_passed),
			supply_chain_scan_passed=bool(supply_chain_scan_passed),
			external_review_recorded=bool(external_review_recorded),
			status="registered",
			metadata=dict(metadata or {}),
		)
		self._plugins[_state_key(tenant_id, plugin.id)] = plugin
		self._record_audit(tenant_id, plugin.id, "plugin_registered", owner, "allow")
		return plugin.to_dict()

	def review_permissions(
		self,
		review_id: str,
		tenant_id: str,
		plugin_id: str,
		reviewer: str,
		approved_scopes: list[str],
		denied_scopes: list[str] | None = None,
		secret_access_allowed: bool = False,
		notes: str = "",
	) -> dict[str, Any]:
		plugin = self._require_plugin(plugin_id, tenant_id)
		if not reviewer:
			raise PermissionError("permission_reviewer_required")
		approved = normalize_scopes(approved_scopes)
		denied = normalize_scopes(denied_scopes)
		if set(plugin.permissions) - set(approved) - set(denied):
			raise PermissionError("all_requested_permissions_must_be_reviewed")
		if sensitive_scopes(approved) and not secret_access_allowed:
			raise PermissionError("sensitive_permission_secret_policy_required")
		review = PermissionReview(
			id=review_id,
			tenant_id=tenant_id,
			plugin_id=plugin_id,
			reviewer=reviewer,
			approved_scopes=approved,
			denied_scopes=denied,
			secret_access_allowed=bool(secret_access_allowed),
			notes=notes,
		)
		self._permission_reviews[_state_key(tenant_id, review.id)] = review
		self._record_audit(tenant_id, review.id, "permission_review_recorded", reviewer, "allow")
		return review.to_dict()

	def attach_sandbox_policy(
		self,
		policy_id: str,
		tenant_id: str,
		plugin_id: str,
		policy_name: str,
		network_access: str = "deny",
		filesystem_access: str = "read_only",
		secret_access: str = "deny",
		tool_allowlist: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_plugin(plugin_id, tenant_id)
		if not policy_name:
			raise PermissionError("sandbox_policy_name_required")
		if secret_access != "deny" and not self._permission_review_for_plugin(tenant_id, plugin_id):
			raise PermissionError("secret_access_requires_permission_review")
		policy = SandboxPolicy(
			id=policy_id,
			tenant_id=tenant_id,
			plugin_id=plugin_id,
			policy_name=policy_name,
			network_access=network_access,
			filesystem_access=filesystem_access,
			secret_access=secret_access,
			tool_allowlist=normalize_scopes(tool_allowlist),
		)
		self._sandbox_policies[_state_key(tenant_id, policy.id)] = policy
		self._record_audit(tenant_id, policy.id, "sandbox_policy_attached", "plgn", "allow")
		return policy.to_dict()

	def publish_listing(
		self,
		listing_id: str,
		tenant_id: str,
		plugin_id: str,
		title: str,
		publisher_verified: bool = True,
		curated: bool = True,
		install_policy: str = "tenant_allowed",
	) -> dict[str, Any]:
		self._require_plugin(plugin_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_listing",
			"publisher_verified": bool(publisher_verified),
			"curated_listing": bool(curated),
		})
		self._raise_if_denied(result)
		listing = MarketplaceListing(
			id=listing_id,
			tenant_id=tenant_id,
			plugin_id=plugin_id,
			title=title,
			publisher_verified=bool(publisher_verified),
			curated=bool(curated),
			install_policy=normalize_install_policy(install_policy),
			status="listed",
		)
		self._listings[_state_key(tenant_id, listing.id)] = listing
		self._record_audit(tenant_id, listing.id, "marketplace_listing_published", "marketplace", "allow")
		return listing.to_dict()

	def create_release(
		self,
		release_id: str,
		tenant_id: str,
		plugin_id: str,
		version: str,
		channel: str,
		signature_ref: str,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		plugin = self._require_plugin(plugin_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_release",
			"signature_ref_present": bool(signature_ref),
			"event_stream": event_stream_name(event_stream),
		})
		self._raise_if_denied(result)
		status, issues = release_readiness(
			plugin.to_dict(),
			self._permission_review_for_plugin(tenant_id, plugin_id) is not None,
			self._sandbox_policy_for_plugin(tenant_id, plugin_id) is not None,
			self._listing_for_plugin(tenant_id, plugin_id) is not None,
		)
		if status != "ready":
			raise PermissionError(", ".join(issues))
		release = PluginRelease(
			id=release_id,
			tenant_id=tenant_id,
			plugin_id=plugin_id,
			version=version,
			channel=normalize_channel(channel),
			signature_ref=signature_ref,
			status="released",
		)
		self._releases[_state_key(tenant_id, release.id)] = release
		plugin.status = "released"
		plugin.updated_at = utc_now()
		self._record_audit(tenant_id, release.id, "plugin_released", plugin.owner, result["decision"], reasons=self._reasons(result))
		return release.to_dict()

	def install_plugin(self, installation_id: str, tenant_id: str, plugin_id: str, installed_by: str) -> dict[str, Any]:
		self._require_plugin(plugin_id, tenant_id)
		listing = self._listing_for_plugin(tenant_id, plugin_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "install_plugin",
			"tenant_install_policy_present": listing is not None and listing.install_policy != "blocked",
		})
		self._raise_if_denied(result)
		if listing.install_policy == "admin_only" and installed_by != "admin":
			raise PermissionError("admin_install_required")
		installation = PluginInstallation(
			id=installation_id,
			tenant_id=tenant_id,
			plugin_id=plugin_id,
			installed_by=installed_by,
			status="installed",
		)
		self._installations[_state_key(tenant_id, installation.id)] = installation
		self._record_audit(tenant_id, installation.id, "plugin_installed", installed_by, result["decision"])
		return installation.to_dict()

	def enable_plugin(self, installation_id: str, tenant_id: str, actor: str) -> dict[str, Any]:
		installation = self._require_installation(installation_id, tenant_id)
		plugin = self._require_plugin(installation.plugin_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "enable_plugin",
			"signature_verified": plugin.signature_verified,
			"sandbox_policy_attached": self._sandbox_policy_for_plugin(tenant_id, plugin.id) is not None,
		})
		self._raise_if_denied(result)
		installation.status = "enabled"
		installation.enabled_at = utc_now()
		plugin.status = "enabled"
		plugin.updated_at = utc_now()
		self._record_audit(tenant_id, installation.id, "plugin_enabled", actor, "allow")
		return installation.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		plugin = self.register_plugin(
			plugin_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "plugin-owner"),
			version=str(metadata.get("version") or "0.1.0"),
			publisher=str(metadata.get("publisher") or "tenant"),
			release_channel=str(metadata.get("release_channel") or "private"),
			permissions=[],
			signature_verified=bool(metadata.get("signature_verified", True)),
			manifest_schema_valid=bool(metadata.get("manifest_schema_valid", True)),
			dependency_validation_passed=bool(metadata.get("dependency_validation_passed", True)),
			supply_chain_scan_passed=bool(metadata.get("supply_chain_scan_passed", True)),
			metadata=metadata | {"compatibility_status": status or "active"},
		)
		return plugin

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_plugins(tenant_id)

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

	def register_plgn_agent(
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
			"plgn_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_PLGN_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_PLGN_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		self._raise_if_denied(result)
		agent = PlgnAgent(
			id=agent_id or f"plgn-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=bool(contribution_disclosed),
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, agent.id, "plgn_agent_registered", name, result["decision"], metadata=agent.to_dict())
		return agent.to_dict()

	def validate_batch_plugin_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_plugin_mutation",
			"event_stream": event_stream_name(event_stream),
		})

	def list_plgn_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		plugins = self.list_plugins(tenant_id)
		return {
			"tenant_id": tenant_id,
			"plugin_count": len(plugins),
			"released_plugin_count": len([item for item in plugins if item["status"] == "released"]),
			"enabled_plugin_count": len([item for item in plugins if item["status"] == "enabled"]),
			"external_plugin_count": len([item for item in plugins if item["external_plugin"]]),
			"permission_review_count": len(self.list_permission_reviews(tenant_id)),
			"sandbox_policy_count": len(self.list_sandbox_policies(tenant_id)),
			"marketplace_listing_count": len(self.list_marketplace_listings(tenant_id)),
			"release_count": len(self.list_releases(tenant_id)),
			"installation_count": len(self.list_installations(tenant_id)),
			"plgn_agent_count": len(self.list_plgn_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

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
		return next((item for item in self._permission_reviews.values() if item.tenant_id == tenant_id and item.plugin_id == plugin_id), None)

	def _sandbox_policy_for_plugin(self, tenant_id: str, plugin_id: str) -> SandboxPolicy | None:
		return next((item for item in self._sandbox_policies.values() if item.tenant_id == tenant_id and item.plugin_id == plugin_id), None)

	def _listing_for_plugin(self, tenant_id: str, plugin_id: str) -> MarketplaceListing | None:
		return next((item for item in self._listings.values() if item.tenant_id == tenant_id and item.plugin_id == plugin_id), None)

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "plugin_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any]) -> None:
		if result["decision"] == "require_review":
			raise PermissionError(", ".join(self._reasons(result)) or "plugin_review_required")

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		event_id = stable_id("plgnaudit", tenant_id, event_type, subject_id, len(self._audit_events))
		self._audit_events[_state_key(tenant_id, event_id)] = PlgnAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [item for item in values if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			action.get("reason", "plugin_policy_blocked")
			for action in result.get("actions", [])
		)


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"
