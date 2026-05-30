"""Service layer for the UI/UX Theming and Branding capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_THEM_AGENT_ROLES,
	SUPPORTED_THEM_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .theme_runtime import (
	BrandAssetRecord,
	ThemAgentRecord,
	ThemeAuditEventRecord,
	ThemePreviewRecord,
	ThemePublicationRecord,
	ThemeRecord,
	ThemeTokenRecord,
	normalize_token_group,
	stable_id,
	theme_required_actions,
	utc_now,
)


class ThemService:
	"""Deterministic theme and brand-governance service for APG composition."""

	def __init__(self) -> None:
		self.themes: dict[str, ThemeRecord] = {}
		self.tokens: dict[str, ThemeTokenRecord] = {}
		self.assets: dict[str, BrandAssetRecord] = {}
		self.previews: dict[str, ThemePreviewRecord] = {}
		self.publications: dict[str, ThemePublicationRecord] = {}
		self.them_agents: dict[str, ThemAgentRecord] = {}
		self.audit_events: dict[str, ThemeAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_theme(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		brand_name: str,
		guidelines_ref: str,
		fallback_theme_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("theme_name_required")
		if not str(brand_name or "").strip():
			raise ValueError("brand_name_required")
		context = {
			"tenant_context_present": True,
			"operation": "create_theme",
			"theme_owner_assigned": bool(str(owner or "").strip()),
			"brand_guidelines_present": bool(str(guidelines_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = ThemeRecord(
			id=stable_id("them_theme", tenant_id, name),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			brand_name=brand_name,
			guidelines_ref=guidelines_ref,
			fallback_theme_id=fallback_theme_id,
		)
		self.themes[record.id] = record
		self._record_event(tenant_id, "theme_created", record.id, f"Theme created: {name}", owner)
		return record.to_dict()

	def update_tokens(
		self,
		tenant_id: str,
		theme_id: str,
		group: str,
		tokens: dict[str, str],
		updated_by: str,
		contrast_validated: bool = False,
		reviewer: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		if not tokens:
			raise ValueError("theme_tokens_required")
		reviewer_value = reviewer or updated_by
		context = {
			"tenant_context_present": True,
			"operation": "update_tokens",
			"token_reviewer_present": bool(str(reviewer_value or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		version = theme.token_version + 1
		record = ThemeTokenRecord(
			id=stable_id("them_tokens", tenant_id, theme.id, group, version),
			tenant_id=tenant_id,
			theme_id=theme.id,
			group=normalize_token_group(group),
			tokens={str(key): str(value) for key, value in tokens.items()},
			version=version,
			contrast_validated=bool(contrast_validated),
			updated_by=reviewer_value,
		)
		self.tokens[record.id] = record
		theme.token_version = version
		theme.updated_at = utc_now()
		self._record_event(tenant_id, "tokens_updated", record.id, f"Theme tokens updated: {theme.name}", updated_by)
		return record.to_dict()

	def add_brand_asset(
		self,
		tenant_id: str,
		theme_id: str,
		asset_name: str,
		asset_type: str,
		license_ref: str,
		approved_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		if not str(asset_name or "").strip():
			raise ValueError("brand_asset_name_required")
		context = {
			"tenant_context_present": True,
			"operation": "add_brand_asset",
			"brand_asset_present": True,
			"license_verified": bool(str(license_ref or "").strip()),
			"asset_approval_recorded": bool(str(approved_by or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = BrandAssetRecord(
			id=stable_id("them_asset", tenant_id, theme.id, asset_name),
			tenant_id=tenant_id,
			theme_id=theme.id,
			asset_name=asset_name,
			asset_type=str(asset_type or "image"),
			license_ref=license_ref,
			approved_by=approved_by,
			status="approved",
		)
		self.assets[record.id] = record
		self._record_event(tenant_id, "brand_asset_added", record.id, f"Brand asset added: {asset_name}", approved_by)
		return record.to_dict()

	def create_preview(
		self,
		tenant_id: str,
		theme_id: str,
		surface: str,
		viewport: str,
		preview_ref: str,
		contrast_passed: bool,
		created_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		if not str(preview_ref or "").strip():
			result = self.evaluate({
				"tenant_context_present": True,
				"operation": "create_preview",
				"preview_artifact_present": False,
			})
			self._raise_policy(result)
		record = ThemePreviewRecord(
			id=stable_id("them_preview", tenant_id, theme.id, surface, viewport),
			tenant_id=tenant_id,
			theme_id=theme.id,
			surface=str(surface or "app_shell"),
			viewport=str(viewport or "desktop"),
			preview_ref=preview_ref,
			contrast_passed=bool(contrast_passed),
			created_by=created_by,
		)
		self.previews[record.id] = record
		theme.status = "preview_ready"
		theme.updated_at = utc_now()
		self._record_event(tenant_id, "theme_preview_created", record.id, f"Theme preview created: {theme.name}", created_by)
		return record.to_dict()

	def publish_theme(
		self,
		tenant_id: str,
		theme_id: str,
		published_by: str,
		approval_ref: str,
		target_tenant_count: int = 1,
		rollout_review_recorded: bool = False,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		if target_tenant_count <= 0:
			raise ValueError("target_tenant_count_must_be_positive")
		latest_preview = self._latest_preview(tenant_id, theme.id)
		if latest_preview is None:
			raise PermissionError("theme_preview_required")
		context = {
			"tenant_context_present": True,
			"operation": "publish_theme",
			"approval_recorded": bool(str(approval_ref or "").strip()),
			"accessibility_contrast_passed": bool(latest_preview.contrast_passed),
			"target_tenant_count": int(target_tenant_count),
			"rollout_review_recorded": bool(rollout_review_recorded),
			"event_stream": self._normalize_token(event_stream),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		status = "review_required" if result["decision"] == "require_review" else "published"
		record = ThemePublicationRecord(
			id=stable_id("them_publication", tenant_id, theme.id, len(self.publications)),
			tenant_id=tenant_id,
			theme_id=theme.id,
			target_tenant_count=int(target_tenant_count),
			approval_ref=approval_ref,
			status=status,
			published_by=published_by,
			required_actions=theme_required_actions(result),
			matched_rules=list(result["matched_rules"]),
		)
		self.publications[record.id] = record
		theme.status = status
		theme.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"theme_published",
			record.id,
			f"Theme publication {status}: {theme.name}",
			published_by,
			metadata={"event_stream": self._normalize_token(event_stream)},
		)
		return record.to_dict()

	def register_them_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str = "platform",
		human_approval_required: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		context = {
			"tenant_context_present": True,
			"operation": "register_them_agent",
			"agent_runtime_supported": runtime_value in SUPPORTED_THEM_AGENT_RUNTIMES,
			"agent_role_supported": role_value in SUPPORTED_THEM_AGENT_ROLES,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = ThemAgentRecord(
			id=stable_id("them_agent", tenant_id, name, runtime_value, role_value),
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=scope,
			owner=owner,
			human_approval_required=bool(human_approval_required),
		)
		self.them_agents[record.id] = record
		self._record_event(
			tenant_id,
			"them_agent_registered",
			record.id,
			f"Theme agent registered: {name}",
			owner,
			metadata={"runtime": runtime_value, "role": role_value, "event_stream": event_stream_name()},
		)
		return record.to_dict()

	def validate_agent_theme_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool = False,
		human_approval_ref: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		agent = self.them_agents.get(agent_id)
		if agent is None or agent.tenant_id != tenant_id:
			raise KeyError(f"them_agent_not_found:{agent_id}")
		context = {
			"tenant_context_present": True,
			"operation": "agent_theme_action",
			"agent_id": agent_id,
			"agent_role": agent.role,
			"action": action,
			"privileged_scope": bool(privileged_scope),
			"human_approval_recorded": bool(str(human_approval_ref or "").strip()),
		}
		return self.evaluate(context)

	def validate_batch_theme_rollout(
		self,
		tenant_id: str,
		target_tenant_count: int,
		event_stream: str = "bytewax",
		rollout_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		context = {
			"tenant_context_present": True,
			"operation": "batch_theme_rollout",
			"target_tenant_count": int(target_tenant_count),
			"event_stream": self._normalize_token(event_stream),
			"rollout_review_recorded": bool(rollout_review_recorded),
		}
		return self.evaluate(context)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_theme(
			tenant_id=tenant_id,
			name=record_id,
			owner=str(metadata.get("owner") or "compatibility-owner"),
			brand_name=str(metadata.get("brand_name") or status),
			guidelines_ref=str(metadata.get("guidelines_ref") or "guidelines://compatibility"),
			fallback_theme_id=metadata.get("fallback_theme_id"),
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_themes(tenant_id)

	def list_themes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.themes, tenant_id)

	def list_tokens(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.tokens, tenant_id)

	def list_assets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.assets, tenant_id)

	def list_previews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.previews, tenant_id)

	def list_publications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.publications, tenant_id)

	def list_them_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.them_agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		themes = self.list_themes(tenant_id)
		return {
			"tenant_id": tenant_id,
			"theme_count": len(themes),
			"published_theme_count": sum(1 for item in themes if item["status"] == "published"),
			"review_required_theme_count": sum(1 for item in themes if item["status"] == "review_required"),
			"token_version_count": len(self.list_tokens(tenant_id)),
			"approved_asset_count": len([item for item in self.list_assets(tenant_id) if item["status"] == "approved"]),
			"preview_count": len(self.list_previews(tenant_id)),
			"publication_count": len(self.list_publications(tenant_id)),
			"them_agent_count": len(self.list_them_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": streaming_manifest(),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "them_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "them_policy_blocked")

	def _get_theme(self, tenant_id: str, theme_id: str) -> ThemeRecord:
		theme = self.themes.get(theme_id)
		if theme is None:
			theme = next((item for item in self.themes.values() if item.tenant_id == tenant_id and item.name == theme_id), None)
		if theme is None or theme.tenant_id != tenant_id:
			raise KeyError(f"theme_not_found:{theme_id}")
		return theme

	def _latest_preview(self, tenant_id: str, theme_id: str) -> ThemePreviewRecord | None:
		previews = [
			preview
			for preview in self.previews.values()
			if preview.tenant_id == tenant_id and preview.theme_id == theme_id
		]
		if not previews:
			return None
		return sorted(previews, key=lambda item: item.created_at)[-1]

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		record = ThemeAuditEventRecord(
			id=stable_id("them_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
