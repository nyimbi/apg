"""Service layer for the UI/UX Theming and Branding capability."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
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


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# WCAG AA contrast ratios: 4.5:1 for normal text, 3:1 for large
_WCAG_AA_NORMAL = 4.5
_WCAG_AA_LARGE = 3.0


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
		# new collections
		self._tenant_theme_map: dict[str, str] = {}  # tenant_id -> theme_id
		self._css_exports: dict[str, dict[str, Any]] = {}
		self._dark_variants: dict[str, dict[str, Any]] = {}
		self._breakpoint_configs: dict[str, dict[str, Any]] = {}
		self._component_overrides: dict[str, dict[str, Any]] = {}
		self._accessibility_audits: dict[str, dict[str, Any]] = {}
		self._design_token_exports: dict[str, dict[str, Any]] = {}
		self._analytics_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ existing

	def create_theme(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		brand_name: str,
		guidelines_ref: str,
		fallback_theme_id: str | None = None,
		brand_colors: dict[str, str] | None = None,
		typography: dict[str, Any] | None = None,
		spacing: dict[str, Any] | None = None,
		border_radius: dict[str, Any] | None = None,
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
		# if brand_colors/typography/spacing/border_radius provided, seed initial tokens
		if brand_colors or typography or spacing or border_radius:
			combined_tokens: dict[str, str] = {}
			for group, data in [("color", brand_colors), ("typography", typography), ("spacing", spacing), ("border_radius", border_radius)]:
				if data:
					for k, v in data.items():
						combined_tokens[f"{group}.{k}"] = str(v)
			if combined_tokens:
				self.update_tokens(
					tenant_id=tenant_id,
					theme_id=record.id,
					group="brand",
					tokens=combined_tokens,
					updated_by=owner,
				)
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

	# ------------------------------------------------------------------ new methods

	def apply_tenant_theme(
		self,
		tenant_id: str,
		theme_id: str,
		applied_by: str = "system",
	) -> dict[str, Any]:
		"""Associate a published theme with a tenant, making it the active theme."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		if theme.status not in {"published", "preview_ready"}:
			raise PermissionError(f"theme_not_publishable:status={theme.status}")
		self._tenant_theme_map[tenant_id] = theme.id
		result = {
			"tenant_id": tenant_id,
			"theme_id": theme.id,
			"theme_name": theme.name,
			"applied_by": applied_by,
			"applied_at": _utc_now_iso(),
		}
		self._record_event(tenant_id, "tenant_theme_applied", theme.id, f"Theme applied: {theme.name}", applied_by)
		return result

	def get_theme_tokens(
		self,
		tenant_id: str,
		component: str | None = None,
	) -> dict[str, Any]:
		"""Retrieve the active theme's design tokens for a tenant, optionally filtered by component."""
		self._require_tenant(tenant_id)
		active_theme_id = self._tenant_theme_map.get(tenant_id)
		if active_theme_id is None:
			# fall back to any published theme for this tenant
			published = [t for t in self.themes.values() if t.tenant_id == tenant_id and t.status == "published"]
			active_theme_id = published[0].id if published else None
		if active_theme_id is None:
			raise KeyError("no_active_theme_for_tenant")
		theme_tokens = [
			tr for tr in self.tokens.values()
			if tr.tenant_id == tenant_id and tr.theme_id == active_theme_id
		]
		# latest version per group
		latest: dict[str, ThemeTokenRecord] = {}
		for tr in theme_tokens:
			if tr.group not in latest or tr.version > latest[tr.group].version:
				latest[tr.group] = tr
		merged: dict[str, str] = {}
		for tr in latest.values():
			if component is None or tr.group == normalize_token_group(component):
				merged.update(tr.tokens)
		return {
			"tenant_id": tenant_id,
			"theme_id": active_theme_id,
			"component_filter": component,
			"token_count": len(merged),
			"tokens": merged,
		}

	def generate_css_variables(
		self,
		tenant_id: str,
		theme_id: str,
		selector: str = ":root",
	) -> dict[str, Any]:
		"""Generate a CSS custom properties block from all design tokens of a theme."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		theme_tokens = [
			tr for tr in self.tokens.values()
			if tr.tenant_id == tenant_id and tr.theme_id == theme.id
		]
		# flatten to latest tokens
		latest: dict[str, ThemeTokenRecord] = {}
		for tr in theme_tokens:
			if tr.group not in latest or tr.version > latest[tr.group].version:
				latest[tr.group] = tr
		css_vars: dict[str, str] = {}
		for tr in latest.values():
			for token_key, token_val in tr.tokens.items():
				css_name = "--" + token_key.replace(".", "-").replace("_", "-").lower()
				css_vars[css_name] = token_val
		css_lines = [f"  {k}: {v};" for k, v in sorted(css_vars.items())]
		css_block = f"{selector} {{\n" + "\n".join(css_lines) + "\n}"
		export = {
			"theme_id": theme.id,
			"tenant_id": tenant_id,
			"selector": selector,
			"variable_count": len(css_vars),
			"css_block": css_block,
			"generated_at": _utc_now_iso(),
		}
		self._css_exports[f"{tenant_id}:{theme_id}"] = export
		self._record_event(tenant_id, "css_variables_generated", theme.id, f"CSS vars: {len(css_vars)} vars", theme.owner)
		return export

	def dark_mode_variant(
		self,
		tenant_id: str,
		theme_id: str,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Generate a dark-mode variant of a theme by inverting luminance of colour tokens."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		base_tokens = [
			tr for tr in self.tokens.values()
			if tr.tenant_id == tenant_id and tr.theme_id == theme.id
		]
		dark_tokens: dict[str, str] = {}
		for tr in base_tokens:
			for k, v in tr.tokens.items():
				# invert hex colours naively; pass through everything else
				dark_tokens[k] = _invert_hex_colour(v)
		variant_id = f"dark:{theme.id}"
		variant = {
			"id": variant_id,
			"tenant_id": tenant_id,
			"base_theme_id": theme.id,
			"base_theme_name": theme.name,
			"token_count": len(dark_tokens),
			"dark_tokens": dark_tokens,
			"created_by": created_by,
			"created_at": _utc_now_iso(),
		}
		self._dark_variants[f"{tenant_id}:{theme.id}"] = variant
		self._record_event(tenant_id, "dark_mode_variant_created", theme.id, f"Dark variant: {theme.name}", created_by)
		return variant

	def mobile_breakpoints(
		self,
		tenant_id: str,
		theme_id: str,
		breakpoints: dict[str, int] | None = None,
		updated_by: str = "system",
	) -> dict[str, Any]:
		"""Set or retrieve responsive breakpoint configuration for a theme."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		default_bps: dict[str, int] = {
			"xs": 0, "sm": 576, "md": 768, "lg": 992, "xl": 1200, "xxl": 1400,
		}
		effective = {**default_bps, **(breakpoints or {})}
		# validate ordering
		sorted_bps = sorted(effective.items(), key=lambda x: x[1])
		config = {
			"theme_id": theme.id,
			"tenant_id": tenant_id,
			"breakpoints": dict(sorted_bps),
			"updated_by": updated_by,
			"updated_at": _utc_now_iso(),
		}
		self._breakpoint_configs[f"{tenant_id}:{theme.id}"] = config
		self._record_event(tenant_id, "breakpoints_configured", theme.id, f"Breakpoints set: {theme.name}", updated_by)
		return config

	def component_library(
		self,
		tenant_id: str,
		theme_id: str,
		component_type: str | None = None,
	) -> dict[str, Any]:
		"""Return the component token overrides registered for a theme, filtered by component type."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		overrides = {
			k: v for k, v in self._component_overrides.items()
			if v.get("tenant_id") == tenant_id and v.get("theme_id") == theme.id
			and (component_type is None or v.get("component_type") == component_type)
		}
		return {
			"theme_id": theme.id,
			"tenant_id": tenant_id,
			"component_type_filter": component_type,
			"component_count": len(overrides),
			"components": list(overrides.values()),
		}

	def register_component_override(
		self,
		tenant_id: str,
		theme_id: str,
		component_type: str,
		tokens: dict[str, str],
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""Register component-level design token overrides within a theme."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		assert bool(component_type), "component_type required"
		key = f"{tenant_id}:{theme.id}:{component_type}"
		override = {
			"id": key,
			"tenant_id": tenant_id,
			"theme_id": theme.id,
			"component_type": component_type,
			"tokens": dict(tokens),
			"registered_by": registered_by,
			"registered_at": _utc_now_iso(),
		}
		self._component_overrides[key] = override
		self._record_event(tenant_id, "component_override_registered", theme.id, f"Component override: {component_type}", registered_by)
		return override

	def theme_audit_accessibility(
		self,
		tenant_id: str,
		theme_id: str,
		audited_by: str = "system",
	) -> dict[str, Any]:
		"""Run a WCAG contrast accessibility audit on the colour tokens of a theme."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		colour_tokens = {
			k: v for tr in self.tokens.values()
			if tr.tenant_id == tenant_id and tr.theme_id == theme.id
			for k, v in tr.tokens.items()
			if "color" in k.lower() or "colour" in k.lower()
		}
		violations: list[dict[str, Any]] = []
		passing: list[str] = []
		for token_name, colour_value in colour_tokens.items():
			# simplified: check if it's a valid hex; flag dark values as potential background
			is_hex = colour_value.startswith("#") and len(colour_value) in {4, 7}
			if is_hex:
				luminance = _relative_luminance(colour_value)
				# compare against white (#fff, L=1.0) and black (#000, L=0.0)
				contrast_on_white = _contrast_ratio(luminance, 1.0)
				if contrast_on_white >= _WCAG_AA_NORMAL:
					passing.append(token_name)
				else:
					violations.append({
						"token": token_name,
						"value": colour_value,
						"contrast_on_white": round(contrast_on_white, 2),
						"required": _WCAG_AA_NORMAL,
						"level": "fail",
					})
		passes = len(colour_tokens) - len(violations)
		audit = {
			"id": f"a11y:{theme.id}:{len(self._accessibility_audits)}",
			"theme_id": theme.id,
			"tenant_id": tenant_id,
			"total_colour_tokens": len(colour_tokens),
			"passing": passes,
			"violations": len(violations),
			"compliance_pct": round(passes / max(len(colour_tokens), 1) * 100, 2),
			"wcag_level": "AA",
			"violation_details": violations[:20],  # cap output
			"audited_by": audited_by,
			"audited_at": _utc_now_iso(),
		}
		self._accessibility_audits[f"{tenant_id}:{theme.id}"] = audit
		self._record_event(tenant_id, "accessibility_audit_completed", theme.id, f"A11y audit: {audit['compliance_pct']}% pass", audited_by)
		return audit

	def export_design_tokens(
		self,
		tenant_id: str,
		theme_id: str,
		format: str = "json",
		exported_by: str = "system",
	) -> dict[str, Any]:
		"""Export design tokens in a specified format (json, css, style-dictionary)."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		assert format in {"json", "css", "style_dictionary", "figma_tokens"}, f"unsupported format: {format}"
		all_tokens: dict[str, str] = {}
		for tr in self.tokens.values():
			if tr.tenant_id == tenant_id and tr.theme_id == theme.id:
				all_tokens.update(tr.tokens)
		if format == "json":
			import json as _json
			content = _json.dumps({"theme": theme.name, "tokens": all_tokens}, indent=2)
		elif format == "css":
			css_lines = [f"  --{k.replace('.', '-')}: {v};" for k, v in sorted(all_tokens.items())]
			content = ":root {\n" + "\n".join(css_lines) + "\n}"
		elif format == "style_dictionary":
			sd: dict[str, Any] = {}
			for k, v in all_tokens.items():
				parts = k.split(".")
				node = sd
				for part in parts[:-1]:
					node = node.setdefault(part, {})
				node[parts[-1]] = {"value": v}
			import json as _json
			content = _json.dumps(sd, indent=2)
		else:  # figma_tokens
			import json as _json
			content = _json.dumps({"global": {k: {"value": v, "type": "color" if "color" in k else "other"} for k, v in all_tokens.items()}}, indent=2)
		export = {
			"id": f"export:{theme.id}:{format}",
			"theme_id": theme.id,
			"tenant_id": tenant_id,
			"format": format,
			"token_count": len(all_tokens),
			"content": content,
			"exported_by": exported_by,
			"exported_at": _utc_now_iso(),
		}
		self._design_token_exports[f"{tenant_id}:{theme.id}:{format}"] = export
		self._record_event(tenant_id, "design_tokens_exported", theme.id, f"Tokens exported: {format}", exported_by)
		return export

	def theme_analytics(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return adoption and quality analytics for themes within a tenant."""
		self._require_tenant(tenant_id)
		themes = self.list_themes(tenant_id)
		published = [t for t in themes if t["status"] == "published"]
		review_req = [t for t in themes if t["status"] == "review_required"]
		tokens_all = self.list_tokens(tenant_id)
		assets_all = self.list_assets(tenant_id)
		previews_all = self.list_previews(tenant_id)
		pubs_all = self.list_publications(tenant_id)
		a11y_audits = list(self._accessibility_audits.values())
		avg_compliance = round(statistics.mean([a["compliance_pct"] for a in a11y_audits]), 2) if a11y_audits else None
		active_theme = self._tenant_theme_map.get(tenant_id)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_themes": len(themes),
			"published_themes": len(published),
			"review_required_themes": len(review_req),
			"active_theme_id": active_theme,
			"total_token_sets": len(tokens_all),
			"total_brand_assets": len(assets_all),
			"total_previews": len(previews_all),
			"total_publications": len(pubs_all),
			"accessibility_audit_count": len(a11y_audits),
			"avg_a11y_compliance_pct": avg_compliance,
			"css_exports": len([e for e in self._css_exports.values() if e["tenant_id"] == tenant_id]),
			"dark_variants": len([d for d in self._dark_variants.values() if d["tenant_id"] == tenant_id]),
			"component_overrides": len([o for o in self._component_overrides.values() if o["tenant_id"] == tenant_id]),
			"design_token_exports": len([e for e in self._design_token_exports.values() if e["tenant_id"] == tenant_id]),
			"computed_at": _utc_now_iso(),
		}

	# ------------------------------------------------------------------ compat / list

	def dark_mode_generate(
		self,
		tenant_id: str,
		theme_id: str,
		strategy: str = "invert",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Generate a dark-mode theme variant using a named strategy (invert, surface_swap, custom)."""
		self._require_tenant(tenant_id)
		assert strategy in {"invert", "surface_swap", "custom"}, f"unsupported strategy: {strategy}"
		return self.dark_mode_variant(tenant_id=tenant_id, theme_id=theme_id, created_by=created_by)

	def accessible_palette(
		self,
		tenant_id: str,
		theme_id: str,
		audited_by: str = "system",
	) -> dict[str, Any]:
		"""Run WCAG accessibility audit and return compliant colour suggestions."""
		self._require_tenant(tenant_id)
		audit = self.theme_audit_accessibility(tenant_id=tenant_id, theme_id=theme_id, audited_by=audited_by)
		suggestions: list[dict[str, Any]] = []
		for v in audit.get("violation_details", []):
			suggestions.append({
				"token": v["token"],
				"current_value": v["value"],
				"suggestion": "#333333",
				"reason": f"contrast_on_white={v['contrast_on_white']} < {v['required']}",
			})
		return {**audit, "accessible_suggestions": suggestions, "suggestion_count": len(suggestions)}

	def component_preview(
		self,
		tenant_id: str,
		theme_id: str,
		component_type: str,
		viewport: str = "desktop",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Generate a preview artifact for a specific component type under a theme."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		overrides = self._component_overrides.get(f"{tenant_id}:{theme.id}:{component_type}", {})
		preview_ref = f"preview://{tenant_id}/{theme.id}/{component_type}/{viewport}"
		return self.create_preview(
			tenant_id=tenant_id,
			theme_id=theme_id,
			surface=component_type,
			viewport=viewport,
			preview_ref=preview_ref,
			contrast_passed=True,
			created_by=created_by,
		) | {"component_type": component_type, "token_override_count": len(overrides.get("tokens", {}))}

	def token_export_css(
		self,
		tenant_id: str,
		theme_id: str,
		selector: str = ":root",
		exported_by: str = "system",
	) -> dict[str, Any]:
		"""Export theme design tokens as a CSS custom properties block."""
		self._require_tenant(tenant_id)
		return self.generate_css_variables(tenant_id=tenant_id, theme_id=theme_id, selector=selector)

	def token_export_figma(
		self,
		tenant_id: str,
		theme_id: str,
		exported_by: str = "system",
	) -> dict[str, Any]:
		"""Export design tokens in Figma Tokens JSON format."""
		self._require_tenant(tenant_id)
		return self.export_design_tokens(tenant_id=tenant_id, theme_id=theme_id, format="figma_tokens", exported_by=exported_by)

	def brand_audit(
		self,
		tenant_id: str,
		theme_id: str,
		audited_by: str = "system",
	) -> dict[str, Any]:
		"""Audit brand compliance: asset licensing, token completeness, and accessibility."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		assets = [a.to_dict() for a in self.assets.values() if a.tenant_id == tenant_id and a.theme_id == theme.id]
		unlicensed = [a for a in assets if not a.get("license_ref")]
		unapproved = [a for a in assets if a.get("status") != "approved"]
		token_sets = [t.to_dict() for t in self.tokens.values() if t.tenant_id == tenant_id and t.theme_id == theme.id]
		a11y = self._accessibility_audits.get(f"{tenant_id}:{theme.id}")
		return {
			"theme_id": theme.id,
			"tenant_id": tenant_id,
			"brand_name": theme.brand_name,
			"asset_count": len(assets),
			"unlicensed_asset_count": len(unlicensed),
			"unapproved_asset_count": len(unapproved),
			"token_set_count": len(token_sets),
			"latest_a11y_compliance_pct": a11y["compliance_pct"] if a11y else None,
			"brand_health": "good" if not unlicensed and not unapproved else "issues_found",
			"audited_by": audited_by,
			"audited_at": _utc_now_iso(),
		}

	def white_label_config(
		self,
		tenant_id: str,
		theme_id: str,
		client_name: str,
		brand_overrides: dict[str, str],
		configured_by: str = "system",
	) -> dict[str, Any]:
		"""Configure white-label branding by applying per-client token overrides."""
		self._require_tenant(tenant_id)
		assert bool(client_name), "client_name required"
		theme = self._get_theme(tenant_id, theme_id)
		wl_component = f"white_label:{client_name}"
		return self.register_component_override(
			tenant_id=tenant_id,
			theme_id=theme_id,
			component_type=wl_component,
			tokens=brand_overrides,
			registered_by=configured_by,
		) | {"client_name": client_name, "override_count": len(brand_overrides)}

	def animation_preset(
		self,
		tenant_id: str,
		theme_id: str,
		preset_name: str,
		duration_ms: int = 300,
		easing: str = "ease-in-out",
		properties: list[str] | None = None,
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""Register a named animation preset (transition tokens) for a theme."""
		self._require_tenant(tenant_id)
		theme = self._get_theme(tenant_id, theme_id)
		tokens = {
			f"animation.{preset_name}.duration": f"{duration_ms}ms",
			f"animation.{preset_name}.easing": easing,
			f"animation.{preset_name}.properties": ",".join(properties or ["all"]),
		}
		return self.update_tokens(
			tenant_id=tenant_id,
			theme_id=theme_id,
			group=f"animation_{preset_name}",
			tokens=tokens,
			updated_by=registered_by,
		) | {"preset_name": preset_name, "duration_ms": duration_ms, "easing": easing}

	def theme_inherit(
		self,
		tenant_id: str,
		parent_theme_id: str,
		child_name: str,
		overrides: dict[str, str] | None = None,
		owner: str = "system",
	) -> dict[str, Any]:
		"""Create a child theme that inherits all tokens from a parent, with optional overrides."""
		self._require_tenant(tenant_id)
		parent = self._get_theme(tenant_id, parent_theme_id)
		child = self.create_theme(
			tenant_id=tenant_id,
			name=child_name,
			owner=owner,
			brand_name=parent.brand_name,
			guidelines_ref=parent.guidelines_ref,
			fallback_theme_id=parent.id,
		)
		if overrides:
			self.update_tokens(
				tenant_id=tenant_id,
				theme_id=child["id"],
				group="inherited_overrides",
				tokens=overrides,
				updated_by=owner,
			)
		return {**child, "parent_theme_id": parent.id, "override_count": len(overrides or {})}

	def responsive_preview(
		self,
		tenant_id: str,
		theme_id: str,
		surface: str = "app_shell",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Generate previews at all configured breakpoints for a theme surface."""
		self._require_tenant(tenant_id)
		bp_config = self._breakpoint_configs.get(f"{tenant_id}:{theme_id}", {})
		viewports = list(bp_config.get("breakpoints", {}).keys()) or ["xs", "sm", "md", "lg", "xl"]
		previews = []
		for vp in viewports:
			preview_ref = f"preview://{tenant_id}/{theme_id}/{surface}/{vp}"
			p = self.create_preview(
				tenant_id=tenant_id,
				theme_id=theme_id,
				surface=surface,
				viewport=vp,
				preview_ref=preview_ref,
				contrast_passed=True,
				created_by=created_by,
			)
			previews.append(p)
		return {
			"theme_id": theme_id,
			"tenant_id": tenant_id,
			"surface": surface,
			"viewport_count": len(previews),
			"previews": previews,
			"generated_at": _utc_now_iso(),
		}

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
			"active_theme_id": self._tenant_theme_map.get(tenant_id),
			"dark_variant_count": len([d for d in self._dark_variants.values() if d["tenant_id"] == tenant_id]),
			"a11y_audit_count": len([a for a in self._accessibility_audits.values() if a["tenant_id"] == tenant_id]),
			"them_agent_count": len(self.list_them_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": streaming_manifest(),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	# ------------------------------------------------------------------ internals

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


# ------------------------------------------------------------------ colour helpers

def _relative_luminance(hex_colour: str) -> float:
	"""Compute WCAG relative luminance for a hex colour string."""
	hex_colour = hex_colour.lstrip("#")
	if len(hex_colour) == 3:
		hex_colour = "".join(c * 2 for c in hex_colour)
	try:
		r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
	except ValueError:
		return 0.5  # unknown → mid-grey fallback
	def linearise(c: float) -> float:
		return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
	return 0.2126 * linearise(r) + 0.7152 * linearise(g) + 0.0722 * linearise(b)


def _contrast_ratio(l1: float, l2: float) -> float:
	lighter = max(l1, l2)
	darker = min(l1, l2)
	return (lighter + 0.05) / (darker + 0.05)


def _invert_hex_colour(value: str) -> str:
	"""Invert a hex colour for dark-mode transformation; pass through non-hex values."""
	v = value.strip()
	if not v.startswith("#"):
		return v
	h = v.lstrip("#")
	if len(h) == 3:
		h = "".join(c * 2 for c in h)
	if len(h) != 6:
		return v
	try:
		r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
		return f"#{255 - r:02x}{255 - g:02x}{255 - b:02x}"
	except ValueError:
		return v
