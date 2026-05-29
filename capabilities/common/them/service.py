"""Service layer for the UI/UX Theming and Branding capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .theme_runtime import (
	BrandAssetRecord,
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
		if not str(guidelines_ref or "").strip():
			raise PermissionError("brand_guidelines_required")
		context = {
			"tenant_context_present": True,
			"operation": "create_theme",
			"theme_owner_assigned": bool(str(owner or "").strip()),
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
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		if not tokens:
			raise ValueError("theme_tokens_required")
		version = theme.token_version + 1
		record = ThemeTokenRecord(
			id=stable_id("them_tokens", tenant_id, theme.id, group, version),
			tenant_id=tenant_id,
			theme_id=theme.id,
			group=normalize_token_group(group),
			tokens={str(key): str(value) for key, value in tokens.items()},
			version=version,
			contrast_validated=bool(contrast_validated),
			updated_by=updated_by,
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
			"brand_asset_present": True,
			"license_verified": bool(str(license_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(approved_by or "").strip():
			raise PermissionError("brand_asset_approval_required")
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
			raise PermissionError("theme_preview_required")
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
		self._record_event(tenant_id, "theme_published", record.id, f"Theme publication {status}: {theme.name}", published_by)
		return record.to_dict()

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
	) -> dict[str, Any]:
		record = ThemeAuditEventRecord(
			id=stable_id("them_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
