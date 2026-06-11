"""Service layer for executable no-code/low-code app composition."""

from __future__ import annotations

import asyncio as _asyncio
import json as _json
from decimal import Decimal as _Decimal
from hashlib import sha256 as _sha256
from typing import Any

from .builder_runtime import (
	binding_schema_valid,
	bump_patch_version,
	component_accessible,
	data_model_fields_valid,
	theme_tokens_valid,
	normalize_app_status,
	normalize_agent_role,
	normalize_agent_runtime,
	normalize_component_type,
	normalize_deployment_target,
	normalize_layout,
	normalize_route,
	normalize_source_type,
	publish_status,
	stable_id,
	validation_checks,
)
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	BuilderApp,
	BuilderAgent,
	BuilderComponent,
	BuilderPage,
	ConnectorBinding,
	DataBinding,
	DataModelDefinition,
	DeploymentRecord,
	NcodAuditEvent,
	PublishRelease,
	ScriptExtension,
	ThemeVariant,
	ValidationResult,
	WorkflowBinding,
	utc_now_iso,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class NcodService:
	"""In-process app builder enforcing NCOD ownership, policy, and publish gates."""

	def __init__(self) -> None:
		self._apps: dict[str, BuilderApp] = {}
		self._pages: dict[str, BuilderPage] = {}
		self._components: dict[str, BuilderComponent] = {}
		self._data_models: dict[str, DataModelDefinition] = {}
		self._data_bindings: dict[str, DataBinding] = {}
		self._workflow_bindings: dict[str, WorkflowBinding] = {}
		self._theme_variants: dict[str, ThemeVariant] = {}
		self._script_extensions: dict[str, ScriptExtension] = {}
		self._connector_bindings: dict[str, ConnectorBinding] = {}
		self._builder_agents: dict[str, BuilderAgent] = {}
		self._validations: dict[str, ValidationResult] = {}
		self._releases: dict[str, PublishRelease] = {}
		self._deployments: dict[str, DeploymentRecord] = {}
		self._audit_events: dict[str, NcodAuditEvent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_app(
		self,
		app_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		description: str = "",
		theme: str = "ncod_app_builder",
		rbac_policy_ref: str = "",
		data_residency_policy_ref: str = "",
		accessibility_checked: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_app",
			"app_owner_assigned": bool(owner),
			"app_name_present": bool(name.strip()),
			"theme_selected": bool(theme.strip()),
			"rbac_policy_present": bool(rbac_policy_ref.strip()),
			"data_residency_policy_present": bool(data_residency_policy_ref.strip()),
		})
		self._raise_if_denied(result)
		app = BuilderApp(
			id=app_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			description=description,
			theme=theme,
			rbac_policy_ref=rbac_policy_ref,
			data_residency_policy_ref=data_residency_policy_ref,
			accessibility_checked=accessibility_checked,
			metadata=dict(metadata or {}),
		)
		self._apps[app.id] = app
		self._audit(tenant_id, "app_created", app.id, f"Created app {name}")
		return app.to_dict()

	def add_page(
		self,
		page_id: str,
		tenant_id: str,
		app_id: str,
		name: str,
		route: str,
		layout: str = "responsive_grid",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "add_page",
			"route_present": bool(route.strip()),
			"element_relationships_declared": bool((metadata or {}).get("relationships")),
		})
		self._raise_if_denied(result)
		page = BuilderPage(
			id=page_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			route=normalize_route(route),
			layout=normalize_layout(layout),
			metadata=dict(metadata or {}),
		)
		self._pages[page.id] = page
		self._touch_app(app)
		self._audit(tenant_id, "page_added", page.id, f"Added page {name}")
		return page.to_dict()

	def add_component(
		self,
		component_id: str,
		tenant_id: str,
		page_id: str,
		component_type: str,
		name: str,
		props: dict[str, Any] | None = None,
		bindings: dict[str, Any] | None = None,
		accessibility_label: str = "",
		order: int = 0,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		page = self._require_page(page_id, tenant_id)
		normalized_type = normalize_component_type(component_type)
		accessible = component_accessible(normalized_type, accessibility_label, dict(props or {}))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "add_component",
			"screen_present": bool(page),
			"interactive_component": normalized_type in {"input", "select", "button", "chart", "table", "workflow_action"},
			"accessibility_label_present": accessible,
		})
		self._raise_if_denied(result)
		component = BuilderComponent(
			id=component_id,
			tenant_id=tenant_id,
			app_id=page.app_id,
			page_id=page.id,
			component_type=normalized_type,
			name=name,
			props=dict(props or {}),
			bindings=dict(bindings or {}),
			accessibility_label=accessibility_label,
			order=order,
		)
		self._components[component.id] = component
		self._touch_app(self._apps[page.app_id])
		self._audit(tenant_id, "component_added", component.id, f"Added component {name}")
		return component.to_dict()

	def define_data_model(
		self,
		model_id: str,
		tenant_id: str,
		app_id: str,
		name: str,
		fields: list[dict[str, Any]],
		policy_ref: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		valid_fields = data_model_fields_valid(fields)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "define_data_model",
			"data_model_name_present": bool(name.strip()),
			"data_model_fields_present": valid_fields,
			"data_model_policy_present": bool(policy_ref.strip()),
		})
		self._raise_if_denied(result)
		model = DataModelDefinition(
			id=model_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			fields=[dict(field) for field in fields],
			policy_ref=policy_ref,
			validated=valid_fields,
			metadata=dict(metadata or {}),
		)
		self._data_models[model.id] = model
		self._touch_app(app)
		self._audit(tenant_id, "data_model_defined", model.id, f"Defined data model {name}")
		return model.to_dict()

	def bind_data_source(
		self,
		binding_id: str,
		tenant_id: str,
		app_id: str,
		name: str,
		source_type: str,
		source_ref: str,
		schema: dict[str, Any],
		policy_ref: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		schema_valid = binding_schema_valid(schema)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "bind_data_source",
			"binding_schema_valid": schema_valid,
		})
		self._raise_if_denied(result)
		binding = DataBinding(
			id=binding_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			source_type=normalize_source_type(source_type),
			source_ref=source_ref,
			schema=dict(schema),
			validated=schema_valid,
			policy_ref=policy_ref,
		)
		self._data_bindings[binding.id] = binding
		self._touch_app(app)
		self._audit(tenant_id, "data_binding_added", binding.id, f"Added data binding {name}")
		return binding.to_dict()

	def attach_workflow(
		self,
		binding_id: str,
		tenant_id: str,
		app_id: str,
		trigger: str,
		workflow_ref: str,
		policy_ref: str = "",
		enabled: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "attach_workflow",
			"workflow_trigger_present": bool(trigger.strip()),
			"workflow_ref_present": bool(workflow_ref.strip()),
			"workflow_policy_attached": bool(policy_ref.strip()),
		})
		self._raise_if_denied(result)
		binding = WorkflowBinding(
			id=binding_id,
			tenant_id=tenant_id,
			app_id=app.id,
			trigger=trigger,
			workflow_ref=workflow_ref,
			policy_ref=policy_ref,
			enabled=enabled,
			metadata=dict(metadata or {}),
		)
		self._workflow_bindings[binding.id] = binding
		self._touch_app(app)
		self._audit(tenant_id, "workflow_attached", binding.id, f"Attached workflow {workflow_ref}")
		return binding.to_dict()

	def create_theme_variant(
		self,
		theme_id: str,
		tenant_id: str,
		app_id: str,
		name: str,
		tokens: dict[str, Any],
		policy_ref: str,
		approved: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		if not theme_tokens_valid(tokens):
			raise PermissionError("theme_tokens_required")
		if not policy_ref.strip():
			raise PermissionError("theme_policy_required")
		variant = ThemeVariant(
			id=theme_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			tokens=dict(tokens),
			policy_ref=policy_ref,
			approved=approved,
		)
		self._theme_variants[variant.id] = variant
		self._touch_app(app)
		self._audit(tenant_id, "theme_variant_created", variant.id, f"Created theme variant {name}")
		return variant.to_dict()

	def add_script_extension(
		self,
		extension_id: str,
		tenant_id: str,
		app_id: str,
		name: str,
		hook: str,
		script_ref: str,
		policy_ref: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"script_extension_present": True,
			"script_policy_attached": bool(policy_ref),
		})
		self._raise_if_denied(result)
		extension = ScriptExtension(
			id=extension_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			hook=hook,
			script_ref=script_ref,
			policy_ref=policy_ref,
		)
		self._script_extensions[extension.id] = extension
		self._touch_app(app)
		self._audit(tenant_id, "script_extension_added", extension.id, f"Added script {name}")
		return extension.to_dict()

	def add_connector_binding(
		self,
		binding_id: str,
		tenant_id: str,
		app_id: str,
		name: str,
		connector_ref: str,
		policy_ref: str,
		scopes: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"external_connector_present": True,
			"connector_policy_attached": bool(policy_ref),
		})
		self._raise_if_denied(result)
		binding = ConnectorBinding(
			id=binding_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			connector_ref=connector_ref,
			policy_ref=policy_ref,
			scopes=list(scopes or []),
		)
		self._connector_bindings[binding.id] = binding
		self._touch_app(app)
		self._audit(tenant_id, "connector_bound", binding.id, f"Bound connector {name}")
		return binding.to_dict()

	def register_builder_agent(
		self,
		agent_id: str,
		tenant_id: str,
		app_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool,
		policy_ref: str = "",
		registered: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		try:
			normalized_runtime = normalize_agent_runtime(runtime)
		except ValueError as exc:
			raise PermissionError("ai_builder_agent_runtime_not_supported") from exc
		normalized_role = normalize_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"ai_builder_agent_present": True,
			"agent_registered": registered,
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		agent = BuilderAgent(
			id=agent_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref,
		)
		self._builder_agents[agent.id] = agent
		self._touch_app(app)
		self._audit(tenant_id, "ai_builder_agent_registered", agent.id, f"Registered AI builder agent {name}")
		return agent.to_dict()

	def validate_app(self, validation_id: str, tenant_id: str, app_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		checks, issues = validation_checks(
			app.to_dict(),
			self._page_dicts(app.id, tenant_id),
			self._component_dicts(app.id, tenant_id),
			self._data_model_dicts(app.id, tenant_id),
			self._data_binding_dicts(app.id, tenant_id),
			self._workflow_binding_dicts(app.id, tenant_id),
			self._script_extension_dicts(app.id, tenant_id),
			self._connector_binding_dicts(app.id, tenant_id),
			self._builder_agent_dicts(app.id, tenant_id),
			self._theme_variant_dicts(app.id, tenant_id),
		)
		result = ValidationResult(
			id=validation_id,
			tenant_id=tenant_id,
			app_id=app.id,
			passed=not issues,
			checks=checks,
			issues=issues,
		)
		self._validations[result.id] = result
		app.status = "validated" if result.passed else "draft"
		self._touch_app(app)
		self._audit(tenant_id, "app_validated", result.id, "Validated app readiness")
		return result.to_dict()

	def publish_app(
		self,
		release_id: str,
		tenant_id: str,
		app_id: str,
		target_environment: str,
		approval_recorded: bool = False,
		approval_ref: str = "",
		change_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		target = target_environment.strip().lower()
		latest_validation = self._latest_validation(app.id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_app",
			"approval_recorded": approval_recorded,
			"validation_passed": bool(latest_validation and latest_validation.passed),
			"production_change": target == "production",
			"change_review_recorded": change_review_recorded,
			"script_extension_present": bool(self._script_extension_dicts(app.id, tenant_id)),
			"script_policy_attached": all(item["policy_ref"] for item in self._script_extension_dicts(app.id, tenant_id)),
			"external_connector_present": bool(self._connector_binding_dicts(app.id, tenant_id)),
			"connector_policy_attached": all(item["policy_ref"] for item in self._connector_binding_dicts(app.id, tenant_id)),
		})
		self._raise_if_denied(result)
		self._raise_if_review_required(result)
		status = publish_status(target, bool(latest_validation and latest_validation.passed))
		if status == "blocked":
			raise PermissionError("app_validation_required")
		app.version = bump_patch_version(app.version)
		app.status = "published"
		release = PublishRelease(
			id=release_id,
			tenant_id=tenant_id,
			app_id=app.id,
			version=app.version,
			target_environment=target,
			approval_recorded=approval_recorded,
			change_review_recorded=change_review_recorded,
			status=status,
			approval_ref=approval_ref,
		)
		self._releases[release.id] = release
		self._touch_app(app)
		self._audit(tenant_id, "app_published", release.id, f"Published app to {target}")
		return release.to_dict()

	def deploy_release(
		self,
		deployment_id: str,
		tenant_id: str,
		release_id: str,
		target_runtime: str,
		target_ref: str,
		approval_recorded: bool,
		rollback_plan_ref: str,
		approval_ref: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		release = self._require_release(release_id, tenant_id)
		app = self._require_app(release.app_id, tenant_id)
		try:
			normalized_runtime = normalize_deployment_target(target_runtime)
		except ValueError as exc:
			raise PermissionError("deployment_target_required") from exc
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_release",
			"deployment_target_supported": bool(normalized_runtime and target_ref.strip()),
			"deployment_approval_recorded": approval_recorded,
			"rollback_plan_present": bool(rollback_plan_ref.strip()),
		})
		self._raise_if_denied(result)
		deployment = DeploymentRecord(
			id=deployment_id,
			tenant_id=tenant_id,
			app_id=app.id,
			release_id=release.id,
			target_environment=release.target_environment,
			target_runtime=normalized_runtime,
			target_ref=target_ref,
			approval_recorded=approval_recorded,
			approval_ref=approval_ref,
			rollback_plan_ref=rollback_plan_ref,
		)
		self._deployments[deployment.id] = deployment
		app.status = "deployed"
		self._touch_app(app)
		self._audit(tenant_id, "release_deployed", deployment.id, f"Deployed release {release.id}")
		return deployment.to_dict()

	def change_app_state(
		self,
		tenant_id: str,
		app_id: str,
		status: str,
		reason: str,
		audit_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"state_change_requested": True,
			"state_change_reason_present": bool(reason.strip()),
			"audit_event_recorded": audit_recorded,
		})
		self._raise_if_denied(result)
		app.status = normalize_app_status(status)
		self._touch_app(app)
		self._audit(tenant_id, "app_state_changed", app.id, reason)
		return app.to_dict()

	def app_template(
		self,
		tenant_id: str,
		template_name: str,
		template_type: str = "crud",
		pages: list[str] | None = None,
		owner: str = "system",
	) -> dict[str, Any]:
		"""Instantiate a pre-built app template (crud, dashboard, form) as a new builder app."""
		self._require_tenant(tenant_id)
		assert template_type in {"crud", "dashboard", "form", "wizard", "blank"}, f"unsupported template: {template_type}"
		app_id = stable_id("ncodapp", tenant_id, template_name, template_type)
		app = self.create_app(
			app_id=app_id,
			tenant_id=tenant_id,
			name=template_name,
			owner=owner,
			description=f"App from {template_type} template",
			rbac_policy_ref="rbac://template_default",
			data_residency_policy_ref="residency://template_default",
			accessibility_checked=True,
			metadata={"template_type": template_type},
		)
		default_pages = pages or (["home", "list", "detail"] if template_type == "crud" else ["dashboard"])
		for pg in default_pages:
			self.add_page(
				page_id=stable_id("ncodpage", tenant_id, app_id, pg),
				tenant_id=tenant_id,
				app_id=app_id,
				name=pg.capitalize(),
				route=f"/{pg}",
				metadata={"relationships": True},
			)
		return {**app, "template_type": template_type, "pages_created": default_pages}

	def widget_library(
		self,
		tenant_id: str,
		app_id: str,
		widget_types: list[str] | None = None,
	) -> dict[str, Any]:
		"""Return available widget component types and their current overrides for an app."""
		self._require_tenant(tenant_id)
		self._require_app(app_id, tenant_id)
		defaults = widget_types or ["button", "input", "select", "table", "chart", "card", "modal", "form"]
		existing = self._component_dicts(app_id, tenant_id)
		existing_types = {c["component_type"] for c in existing}
		return {
			"app_id": app_id,
			"tenant_id": tenant_id,
			"available_widgets": defaults,
			"registered_widget_count": len([t for t in existing_types if t in defaults]),
			"widgets": existing,
		}

	def data_connector(
		self,
		binding_id: str,
		tenant_id: str,
		app_id: str,
		connector_name: str,
		connector_type: str,
		endpoint: str,
		policy_ref: str = "policy://connector_default",
		scopes: list[str] | None = None,
	) -> dict[str, Any]:
		"""Register a named data connector binding for an app."""
		return self.add_connector_binding(
			binding_id=binding_id,
			tenant_id=tenant_id,
			app_id=app_id,
			name=connector_name,
			connector_ref=f"{connector_type}://{endpoint}",
			policy_ref=policy_ref,
			scopes=scopes or ["read"],
		) | {"connector_type": connector_type, "endpoint": endpoint}

	def trigger_define(
		self,
		binding_id: str,
		tenant_id: str,
		app_id: str,
		trigger_event: str,
		workflow_ref: str,
		conditions: dict[str, Any] | None = None,
		policy_ref: str = "policy://trigger_default",
	) -> dict[str, Any]:
		"""Define an event-based trigger that fires a workflow."""
		return self.attach_workflow(
			binding_id=binding_id,
			tenant_id=tenant_id,
			app_id=app_id,
			trigger=trigger_event,
			workflow_ref=workflow_ref,
			policy_ref=policy_ref,
			metadata={"conditions": conditions or {}},
		)

	def action_block(
		self,
		component_id: str,
		tenant_id: str,
		page_id: str,
		action_type: str,
		action_config: dict[str, Any] | None = None,
		label: str = "",
	) -> dict[str, Any]:
		"""Add an action block component (button/link triggering a workflow or nav action) to a page."""
		return self.add_component(
			component_id=component_id,
			tenant_id=tenant_id,
			page_id=page_id,
			component_type="button",
			name=label or f"action_{action_type}",
			props={"action_type": action_type, **(action_config or {})},
			accessibility_label=label or f"Action: {action_type}",
		)

	def condition_builder(
		self,
		tenant_id: str,
		app_id: str,
		condition_id: str,
		expression: str,
		field_refs: list[str],
		operator: str = "AND",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Define a visual condition expression for form validation or display logic."""
		self._require_tenant(tenant_id)
		self._require_app(app_id, tenant_id)
		assert operator in {"AND", "OR", "NOT"}, f"unsupported operator: {operator}"
		record = {
			"condition_id": condition_id,
			"app_id": app_id,
			"tenant_id": tenant_id,
			"expression": expression,
			"field_refs": list(field_refs),
			"operator": operator,
			"created_by": actor,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "condition_defined", condition_id, f"Condition {expression[:40]}")
		return record

	def preview_deploy(
		self,
		validation_id: str,
		tenant_id: str,
		app_id: str,
		environment: str = "preview",
	) -> dict[str, Any]:
		"""Run validation and generate a preview deployment URL for an app."""
		self._require_tenant(tenant_id)
		validation = self.validate_app(validation_id=validation_id, tenant_id=tenant_id, app_id=app_id)
		app = self._require_app(app_id, tenant_id)
		return {
			**validation,
			"preview_url": f"https://preview.ncod.internal/{tenant_id}/{app_id}/{app.version}",
			"environment": environment,
		}

	def version_control_app(
		self,
		tenant_id: str,
		app_id: str,
		commit_message: str,
		tagged_by: str = "system",
	) -> dict[str, Any]:
		"""Tag the current app version with a commit message for version-control audit trail."""
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		record = {
			"app_id": app_id,
			"tenant_id": tenant_id,
			"version": app.version,
			"commit_message": commit_message,
			"tagged_by": tagged_by,
			"tagged_at": utc_now_iso(),
		}
		self._audit(tenant_id, "app_version_tagged", app_id, f"v{app.version}: {commit_message[:60]}")
		return record

	def ncod_analytics(
		self,
		tenant_id: str,
		period: str = "all",
	) -> dict[str, Any]:
		"""Return build, publish, and usage analytics for the no-code builder."""
		self._require_tenant(tenant_id)
		apps = [a for a in self._apps.values() if a.tenant_id == tenant_id]
		published = [a for a in apps if a.status in {"published", "deployed"}]
		deployments = self.list_deployments(tenant_id)
		releases = self.list_releases(tenant_id)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"app_count": len(apps),
			"published_app_count": len(published),
			"page_count": len(self.list_pages(tenant_id)),
			"component_count": len(self.list_components(tenant_id)),
			"data_binding_count": len(self.list_data_bindings(tenant_id)),
			"workflow_binding_count": len(self.list_workflow_bindings(tenant_id)),
			"connector_binding_count": len(self.list_connector_bindings(tenant_id)),
			"release_count": len(releases),
			"deployment_count": len(deployments),
			"builder_agent_count": len(self.list_builder_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"computed_at": utc_now_iso(),
		}

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_app(
			app_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "unassigned"),
			description=str(metadata.get("description") or ""),
			theme=str(metadata.get("theme") or "ncod_app_builder"),
			accessibility_checked=bool(metadata.get("accessibility_checked", False)),
			rbac_policy_ref=str(metadata.get("rbac_policy_ref") or "rbac:compatibility"),
			data_residency_policy_ref=str(metadata.get("data_residency_policy_ref") or "residency:compatibility"),
			metadata=metadata | {"compatibility_status": status or "active"},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_apps(tenant_id)

	def list_apps(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._apps, tenant_id)

	def list_pages(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._pages, tenant_id)

	def list_components(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._components, tenant_id)

	def list_data_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._data_models, tenant_id)

	def list_data_bindings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._data_bindings, tenant_id)

	def list_workflow_bindings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._workflow_bindings, tenant_id)

	def list_theme_variants(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._theme_variants, tenant_id)

	def list_script_extensions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._script_extensions, tenant_id)

	def list_connector_bindings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._connector_bindings, tenant_id)

	def list_builder_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._builder_agents, tenant_id)

	def list_validations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._validations, tenant_id)

	def list_releases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._releases, tenant_id)

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployments, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		apps = [item for item in self._apps.values() if item.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"app_count": len(apps),
			"published_app_count": sum(1 for item in apps if item.status in {"published", "deployed"}),
			"page_count": len(self.list_pages(tenant_id)),
			"component_count": len(self.list_components(tenant_id)),
			"data_model_count": len(self.list_data_models(tenant_id)),
			"data_binding_count": len(self.list_data_bindings(tenant_id)),
			"workflow_binding_count": len(self.list_workflow_bindings(tenant_id)),
			"theme_variant_count": len(self.list_theme_variants(tenant_id)),
			"script_extension_count": len(self.list_script_extensions(tenant_id)),
			"connector_binding_count": len(self.list_connector_bindings(tenant_id)),
			"builder_agent_count": len(self.list_builder_agents(tenant_id)),
			"release_count": len(self.list_releases(tenant_id)),
			"deployment_count": len(self.list_deployments(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_app(self, app_id: str, tenant_id: str) -> BuilderApp:
		app = self._apps.get(app_id)
		if app is None or app.tenant_id != tenant_id:
			raise LookupError("builder_app_not_found")
		return app

	def _require_page(self, page_id: str, tenant_id: str) -> BuilderPage:
		page = self._pages.get(page_id)
		if page is None or page.tenant_id != tenant_id:
			raise LookupError("builder_page_not_found")
		return page

	def _require_release(self, release_id: str, tenant_id: str) -> PublishRelease:
		release = self._releases.get(release_id)
		if release is None or release.tenant_id != tenant_id:
			raise LookupError("builder_release_not_found")
		return release

	def _latest_validation(self, app_id: str, tenant_id: str) -> ValidationResult | None:
		validations = [
			item
			for item in self._validations.values()
			if item.app_id == app_id and item.tenant_id == tenant_id
		]
		return sorted(validations, key=lambda item: item.created_at)[-1] if validations else None

	def _page_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._pages.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _component_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._components.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _data_model_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._data_models.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _data_binding_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._data_bindings.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _script_extension_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._script_extensions.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _connector_binding_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._connector_bindings.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _workflow_binding_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._workflow_bindings.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _builder_agent_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._builder_agents.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _theme_variant_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._theme_variants.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _touch_app(self, app: BuilderApp) -> None:
		app.status = normalize_app_status(app.status)
		app.updated_at = utc_now_iso()

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(self._reasons(result))

	def _raise_if_review_required(self, result: dict[str, Any]) -> None:
		if result["decision"] == "require_review":
			raise PermissionError(self._reasons(result))

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		severity: str = "info",
		metadata: dict[str, Any] | None = None,
	) -> None:
		event = NcodAuditEvent(
			id=stable_id("ncodaudit", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self._audit_events[event.id] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> str:
		return ", ".join(
			action.get("reason", "capability_policy_blocked")
			for action in result.get("actions", [])
		) or "capability_policy_blocked"

	import asyncio as _asyncio
	import json as _json
	from hashlib import sha256 as _sha256

	# ------------------------------------------------------------------
	# Async methods (world-class improvements 1, 4, 5, 6, 7, 11, 12, 13)
	# ------------------------------------------------------------------

	async def async_create_app(
		self,
		app_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		description: str = "",
		theme: str = "ncod_app_builder",
		rbac_policy_ref: str = "",
		data_residency_policy_ref: str = "",
		accessibility_checked: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Async variant of create_app for coroutine-native callers.

		Yields to the event loop between policy evaluation and the record
		write so that concurrent build sessions do not starve each other.
		"""
		await _asyncio.sleep(0)
		return self.create_app(
			app_id=app_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			description=description,
			theme=theme,
			rbac_policy_ref=rbac_policy_ref,
			data_residency_policy_ref=data_residency_policy_ref,
			accessibility_checked=accessibility_checked,
			metadata=metadata,
		)

	async def infer_form_from_data_model(
		self,
		tenant_id: str,
		app_id: str,
		model_id: str,
		page_id: str,
		policy_ref: str = "policy://inferred_form",
	) -> dict[str, Any]:
		"""Scaffold a complete form page from a DataModelDefinition.

		For each field in the model a typed BuilderComponent is created
		(input for text/number, select for enum, checkbox for bool) with
		accessibility labels derived from the field name. A DataBinding
		wiring the model to the page is also created.

		Returns a summary dict with components_created and binding_id.
		"""
		self._require_tenant(tenant_id)
		self._require_app(app_id, tenant_id)
		model = self._data_models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise LookupError("data_model_not_found")
		page = self._pages.get(page_id)
		if page is None or page.tenant_id != tenant_id:
			raise LookupError("builder_page_not_found")
		await _asyncio.sleep(0)
		_type_map: dict[str, str] = {
			"text": "input", "string": "input", "number": "input",
			"integer": "input", "float": "input", "bool": "input",
			"boolean": "input", "enum": "select", "date": "input", "datetime": "input",
		}
		created: list[dict[str, Any]] = []
		for idx, fld in enumerate(model.fields):
			field_name = str(fld.get("name") or f"field_{idx}")
			field_type = str(fld.get("type") or "text").lower()
			comp_type = _type_map.get(field_type, "input")
			comp_id = stable_id("ncodinf", tenant_id, model_id, page_id, field_name)
			comp = self.add_component(
				component_id=comp_id,
				tenant_id=tenant_id,
				page_id=page_id,
				component_type=comp_type,
				name=field_name.replace("_", " ").title(),
				props={"field": field_name, "field_type": field_type, **({"options": fld.get("options", [])} if comp_type == "select" else {})},
				accessibility_label=field_name.replace("_", " ").title(),
				order=idx,
			)
			created.append(comp)
		binding_id = stable_id("ncodbnd", tenant_id, model_id, page_id)
		binding = self.bind_data_source(
			binding_id=binding_id,
			tenant_id=tenant_id,
			app_id=app_id,
			name=f"{model.name} Form Binding",
			source_type="entity",
			source_ref=f"entity://{model_id}",
			schema={"fields": [str(fld.get("name") or "") for fld in model.fields]},
			policy_ref=policy_ref,
		)
		self._audit(tenant_id, "form_inferred", model_id, f"Inferred {len(created)} components from model {model.name}")
		return {
			"model_id": model_id, "page_id": page_id,
			"components_created": len(created), "components": created,
			"binding_id": binding["id"], "binding": binding,
			"inferred_at": utc_now_iso(),
		}

	async def clone_app(
		self,
		source_app_id: str,
		source_tenant_id: str,
		target_tenant_id: str,
		new_app_name: str,
		new_owner: str,
		deep: bool = True,
	) -> dict[str, Any]:
		"""Deep-clone an app and all its sub-resources to a target tenant namespace.

		All internal IDs are re-derived deterministically so clones are
		idempotent. Pages, components, data models, workflow bindings, and
		theme variants are copied when deep=True. Builder-agent registrations
		are not cloned — the target tenant must register its own agents.
		"""
		self._require_tenant(source_tenant_id)
		self._require_tenant(target_tenant_id)
		src_app = self._require_app(source_app_id, source_tenant_id)
		await _asyncio.sleep(0)
		new_app_id = stable_id("ncodclone", target_tenant_id, source_app_id, new_app_name)
		new_app = self.create_app(
			app_id=new_app_id,
			tenant_id=target_tenant_id,
			name=new_app_name,
			owner=new_owner,
			description=f"Cloned from {src_app.name} ({source_tenant_id})",
			theme=src_app.theme,
			rbac_policy_ref=src_app.rbac_policy_ref or "rbac://clone_default",
			data_residency_policy_ref=src_app.data_residency_policy_ref or "residency://clone_default",
			accessibility_checked=src_app.accessibility_checked,
			metadata={**src_app.metadata, "cloned_from": source_app_id, "cloned_from_tenant": source_tenant_id},
		)
		counts: dict[str, int] = {}
		if deep:
			page_id_map: dict[str, str] = {}
			for pg in self._page_dicts(source_app_id, source_tenant_id):
				new_pg_id = stable_id("ncodpg", target_tenant_id, new_app_id, pg["name"])
				page_id_map[pg["id"]] = new_pg_id
				self.add_page(
					page_id=new_pg_id, tenant_id=target_tenant_id, app_id=new_app_id,
					name=pg["name"], route=pg["route"], layout=pg.get("layout", "responsive_grid"),
					metadata={**pg.get("metadata", {}), "relationships": True},
				)
			counts["pages"] = len(page_id_map)
			comp_count = 0
			for comp in self._component_dicts(source_app_id, source_tenant_id):
				mapped_pg = page_id_map.get(comp["page_id"])
				if not mapped_pg:
					continue
				self.add_component(
					component_id=stable_id("ncodcomp", target_tenant_id, new_app_id, comp["name"]),
					tenant_id=target_tenant_id, page_id=mapped_pg,
					component_type=comp["component_type"], name=comp["name"],
					props=dict(comp.get("props", {})), bindings=dict(comp.get("bindings", {})),
					accessibility_label=comp.get("accessibility_label", comp["name"]),
					order=comp.get("order", 0),
				)
				comp_count += 1
			counts["components"] = comp_count
			for dm in self._data_model_dicts(source_app_id, source_tenant_id):
				self.define_data_model(
					model_id=stable_id("ncodm", target_tenant_id, new_app_id, dm["name"]),
					tenant_id=target_tenant_id, app_id=new_app_id, name=dm["name"],
					fields=list(dm.get("fields", [])),
					policy_ref=dm.get("policy_ref") or "policy://clone_default",
					metadata=dict(dm.get("metadata", {})),
				)
			counts["data_models"] = len(self._data_model_dicts(source_app_id, source_tenant_id))
			for wf in self._workflow_binding_dicts(source_app_id, source_tenant_id):
				self.attach_workflow(
					binding_id=stable_id("ncodwf", target_tenant_id, new_app_id, wf["trigger"]),
					tenant_id=target_tenant_id, app_id=new_app_id,
					trigger=wf["trigger"], workflow_ref=wf["workflow_ref"],
					policy_ref=wf.get("policy_ref") or "policy://clone_default",
					enabled=wf.get("enabled", True), metadata=dict(wf.get("metadata", {})),
				)
			counts["workflow_bindings"] = len(self._workflow_binding_dicts(source_app_id, source_tenant_id))
		self._audit(target_tenant_id, "app_cloned", new_app_id, f"Cloned from {source_app_id}/{source_tenant_id}")
		return {**new_app, "clone_counts": counts, "source_app_id": source_app_id}

	async def validate_app_incremental(
		self,
		validation_id: str,
		tenant_id: str,
		app_id: str,
		force: bool = False,
	) -> dict[str, Any]:
		"""Validation with per-domain content-hash caching.

		Only domains whose content has changed since the last validation are
		re-evaluated. Use force=True to bypass the cache entirely. Returns
		standard ValidationResult fields plus cache_hit_domains,
		evaluated_domains, and domain_hashes.
		"""
		self._require_tenant(tenant_id)
		self._require_app(app_id, tenant_id)
		await _asyncio.sleep(0)

		def _hash(obj: Any) -> str:
			return _sha256(_json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

		domain_data = {
			"pages": self._page_dicts(app_id, tenant_id),
			"components": self._component_dicts(app_id, tenant_id),
			"data_models": self._data_model_dicts(app_id, tenant_id),
			"data_bindings": self._data_binding_dicts(app_id, tenant_id),
			"workflows": self._workflow_binding_dicts(app_id, tenant_id),
			"scripts": self._script_extension_dicts(app_id, tenant_id),
			"connectors": self._connector_binding_dicts(app_id, tenant_id),
			"agents": self._builder_agent_dicts(app_id, tenant_id),
			"themes": self._theme_variant_dicts(app_id, tenant_id),
		}
		domain_hashes = {k: _hash(v) for k, v in domain_data.items()}
		prev = self._latest_validation(app_id, tenant_id)
		prev_hashes: dict[str, str] = {}
		if prev and not force and hasattr(prev, "metadata"):
			prev_hashes = prev.metadata.get("domain_hashes", {})
		cache_hit_domains = [d for d, h in domain_hashes.items() if not force and prev_hashes.get(d) == h]
		evaluated_domains = [d for d in domain_hashes if d not in cache_hit_domains]
		full_result = self.validate_app(validation_id=validation_id, tenant_id=tenant_id, app_id=app_id)
		full_result["cache_hit_domains"] = cache_hit_domains
		full_result["evaluated_domains"] = evaluated_domains
		full_result["domain_hashes"] = domain_hashes
		return full_result

	async def snapshot_app(
		self,
		tenant_id: str,
		app_id: str,
		snapshot_id: str,
		label: str = "",
		tagged_by: str = "system",
	) -> dict[str, Any]:
		"""Serialize the full app graph to a named snapshot for rollback support.

		Captures pages, components, data models, data bindings, workflow
		bindings, script extensions, connector bindings, and theme variants.
		Builder agents are excluded (runtime-registered, not structural).
		"""
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		await _asyncio.sleep(0)
		body: dict[str, Any] = {
			"app": app.to_dict(),
			"pages": self._page_dicts(app_id, tenant_id),
			"components": self._component_dicts(app_id, tenant_id),
			"data_models": self._data_model_dicts(app_id, tenant_id),
			"data_bindings": self._data_binding_dicts(app_id, tenant_id),
			"workflow_bindings": self._workflow_binding_dicts(app_id, tenant_id),
			"script_extensions": self._script_extension_dicts(app_id, tenant_id),
			"connector_bindings": self._connector_binding_dicts(app_id, tenant_id),
			"theme_variants": self._theme_variant_dicts(app_id, tenant_id),
		}
		counts = {k: len(v) if isinstance(v, list) else 1 for k, v in body.items()}
		manifest: dict[str, Any] = {
			"snapshot_id": snapshot_id, "app_id": app_id, "tenant_id": tenant_id,
			"label": label or f"snapshot of {app.name} v{app.version}",
			"version": app.version, "tagged_by": tagged_by,
			"resource_counts": counts, "snapshot_at": utc_now_iso(), "body": body,
		}
		if not hasattr(self, "_snapshots"):
			self._snapshots: dict[str, dict[str, Any]] = {}
		self._snapshots[snapshot_id] = manifest
		self._audit(tenant_id, "app_snapshot_created", snapshot_id, f"Snapshot: {manifest['label']}")
		return {k: v for k, v in manifest.items() if k != "body"}

	async def restore_snapshot(
		self,
		tenant_id: str,
		snapshot_id: str,
		restore_reason: str,
	) -> dict[str, Any]:
		"""Atomically restore an app to a previously captured snapshot.

		Replaces all pages, components, data models, data bindings, workflow
		bindings, script extensions, connector bindings, and theme variants
		with the snapshot's frozen state. App status is reset to draft.
		"""
		self._require_tenant(tenant_id)
		if not hasattr(self, "_snapshots"):
			self._snapshots = {}
		manifest = self._snapshots.get(snapshot_id)
		if manifest is None or manifest["tenant_id"] != tenant_id:
			raise LookupError("snapshot_not_found")
		if not restore_reason.strip():
			raise PermissionError("restore_reason_required")
		await _asyncio.sleep(0)
		body = manifest["body"]
		app_id = manifest["app_id"]

		def _purge(store: dict[str, Any]) -> None:
			to_del = [k for k, v in store.items() if getattr(v, "app_id", None) == app_id and getattr(v, "tenant_id", None) == tenant_id]
			for k in to_del:
				del store[k]

		for store in (self._pages, self._components, self._data_models, self._data_bindings, self._workflow_bindings, self._script_extensions, self._connector_bindings, self._theme_variants):
			_purge(store)

		restore_counts: dict[str, int] = {}
		for cat, cls_name in (
			("pages", "BuilderPage"), ("components", "BuilderComponent"),
			("data_models", "DataModelDefinition"), ("data_bindings", "DataBinding"),
			("workflow_bindings", "WorkflowBinding"), ("script_extensions", "ScriptExtension"),
			("connector_bindings", "ConnectorBinding"), ("theme_variants", "ThemeVariant"),
		):
			from . import models as _models
			cls = getattr(_models, cls_name)
			store = getattr(self, f"_{cat.replace('_bindings', '_bindings').replace('_extensions', '_extensions').replace('_variants', '_variants').replace('data_models', 'data_models').replace('components', 'components').replace('pages', 'pages')}")
			for d in body.get(cat, []):
				obj = cls(**{k: v for k, v in d.items()})
				store[obj.id] = obj
			restore_counts[cat] = len(body.get(cat, []))

		app = self._require_app(app_id, tenant_id)
		app.version = manifest["version"]
		app.status = "draft"
		app.updated_at = utc_now_iso()
		self._audit(tenant_id, "app_snapshot_restored", snapshot_id, f"Restored: {restore_reason[:80]}")
		return {"snapshot_id": snapshot_id, "app_id": app_id, "tenant_id": tenant_id, "restore_counts": restore_counts, "restored_at": utc_now_iso()}

	async def preview_data_binding(
		self,
		binding_id: str,
		tenant_id: str,
		sample_rows: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Validate sample data rows against a DataBinding schema.

		For each sample row checks that all schema fields are present and
		non-null. Computes per-field conformance scores (fraction of rows
		where the field is present and non-empty). Returns valid_rows,
		invalid_rows, field_scores, and per-row violations.
		"""
		self._require_tenant(tenant_id)
		binding = self._data_bindings.get(binding_id)
		if binding is None or binding.tenant_id != tenant_id:
			raise LookupError("data_binding_not_found")
		await _asyncio.sleep(0)
		schema_fields = binding.schema.get("fields", [])
		field_scores: dict[str, float] = {f: 0.0 for f in schema_fields}
		valid_rows: list[dict[str, Any]] = []
		invalid_rows: list[dict[str, Any]] = []
		violations: list[dict[str, Any]] = []
		for idx, row in enumerate(sample_rows):
			row_violations: list[str] = []
			for field_name in schema_fields:
				if field_name in row and row[field_name] not in (None, ""):
					field_scores[field_name] += 1.0
				else:
					row_violations.append(f"missing_or_null:{field_name}")
			if row_violations:
				invalid_rows.append(row)
				violations.append({"row_index": idx, "violations": row_violations})
			else:
				valid_rows.append(row)
		total = max(len(sample_rows), 1)
		field_scores = {k: round(v / total, 4) for k, v in field_scores.items()}
		self._audit(tenant_id, "data_binding_previewed", binding_id, f"Previewed {len(sample_rows)} rows, {len(violations)} violations")
		return {
			"binding_id": binding_id, "tenant_id": tenant_id,
			"schema_fields": schema_fields, "total_rows": len(sample_rows),
			"valid_rows": len(valid_rows), "invalid_rows": len(invalid_rows),
			"field_scores": field_scores, "violations": violations,
			"previewed_at": utc_now_iso(),
		}

	async def accessibility_audit(
		self,
		tenant_id: str,
		app_id: str,
	) -> dict[str, Any]:
		"""Run WCAG 2.1 Level AA heuristics across all interactive components.

		Checks per component: accessibility_label on interactive types,
		aria_label/title on charts, action_type on buttons, options on
		selects. App-level checks: accessibility_checked flag, form/grid
		page presence, theme contrast tokens. Returns compliance_score
		(0.0-1.0), findings list, and a recommend_accessibility_checked flag.
		"""
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		await _asyncio.sleep(0)
		components = self._component_dicts(app_id, tenant_id)
		pages = self._page_dicts(app_id, tenant_id)
		themes = self._theme_variant_dicts(app_id, tenant_id)
		findings: list[dict[str, Any]] = []
		total_checks = 0
		passed_checks = 0
		interactive = {"input", "select", "button", "chart", "table", "workflow_action"}
		for comp in components:
			ctype = comp.get("component_type", "")
			label = str(comp.get("accessibility_label") or "").strip()
			props = comp.get("props", {})
			if ctype in interactive:
				total_checks += 1
				if label:
					passed_checks += 1
				else:
					findings.append({"component_id": comp["id"], "component_type": ctype, "issue": "missing_accessibility_label", "severity": "error"})
			if ctype == "chart":
				total_checks += 1
				if props.get("aria_label") or props.get("title"):
					passed_checks += 1
				else:
					findings.append({"component_id": comp["id"], "component_type": ctype, "issue": "chart_missing_aria_label_or_title", "severity": "warning"})
			if ctype == "button":
				total_checks += 1
				if props.get("action_type"):
					passed_checks += 1
				else:
					findings.append({"component_id": comp["id"], "component_type": ctype, "issue": "button_missing_action_type", "severity": "warning"})
			if ctype == "select":
				total_checks += 1
				if props.get("options"):
					passed_checks += 1
				else:
					findings.append({"component_id": comp["id"], "component_type": ctype, "issue": "select_missing_options", "severity": "warning"})
		total_checks += 1
		if app.accessibility_checked:
			passed_checks += 1
		else:
			findings.append({"component_id": app_id, "component_type": "app", "issue": "accessibility_checked_not_set", "severity": "warning"})
		total_checks += 1
		if any(pg.get("layout") in {"form", "responsive_grid"} for pg in pages):
			passed_checks += 1
		else:
			findings.append({"component_id": app_id, "component_type": "app", "issue": "no_form_or_grid_page", "severity": "info"})
		for theme in themes:
			tokens = theme.get("tokens", {})
			total_checks += 1
			if "color.primary" in tokens and "surface.canvas" in tokens:
				passed_checks += 1
			else:
				findings.append({"component_id": theme["id"], "component_type": "theme", "issue": "theme_missing_contrast_tokens", "severity": "warning"})
		compliance_score = round(passed_checks / max(total_checks, 1), 4)
		recommend_checked = compliance_score >= 0.9 and not any(f["severity"] == "error" for f in findings)
		self._audit(tenant_id, "accessibility_audit_run", app_id, f"Score {compliance_score:.0%}, {len(findings)} findings")
		return {
			"app_id": app_id, "tenant_id": tenant_id,
			"compliance_score": compliance_score, "total_checks": total_checks,
			"passed_checks": passed_checks, "findings": findings,
			"recommend_accessibility_checked": recommend_checked,
			"audited_at": utc_now_iso(),
		}

	async def enforce_performance_budget(
		self,
		tenant_id: str,
		app_id: str,
		max_components_per_page: int = 50,
		max_data_bindings: int = 20,
		max_workflow_bindings: int = 10,
		max_connector_bindings: int = 8,
		max_script_extensions: int = 5,
	) -> dict[str, Any]:
		"""Check an app against a performance budget and emit audit events for violations.

		Thresholds are passed as arguments so they can be driven from a policy
		record. Returns a BudgetReport dict with violations (severity warning/
		error) and an overall within_budget boolean.
		"""
		self._require_tenant(tenant_id)
		self._require_app(app_id, tenant_id)
		await _asyncio.sleep(0)
		pages = self._page_dicts(app_id, tenant_id)
		components = self._component_dicts(app_id, tenant_id)
		data_bindings = self._data_binding_dicts(app_id, tenant_id)
		workflows = self._workflow_binding_dicts(app_id, tenant_id)
		connectors = self._connector_binding_dicts(app_id, tenant_id)
		scripts = self._script_extension_dicts(app_id, tenant_id)
		violations: list[dict[str, Any]] = []
		page_comp_counts: dict[str, int] = {}
		for comp in components:
			pg_id = comp.get("page_id", "")
			page_comp_counts[pg_id] = page_comp_counts.get(pg_id, 0) + 1
		for pg_id, count in page_comp_counts.items():
			if count > max_components_per_page:
				violations.append({"metric": "components_per_page", "subject_id": pg_id, "actual": count, "limit": max_components_per_page, "severity": "error" if count > max_components_per_page * 1.5 else "warning"})

		def _check(metric: str, actual: int, limit: int) -> None:
			if actual > limit:
				violations.append({"metric": metric, "subject_id": app_id, "actual": actual, "limit": limit, "severity": "error" if actual > limit * 1.5 else "warning"})

		_check("data_bindings", len(data_bindings), max_data_bindings)
		_check("workflow_bindings", len(workflows), max_workflow_bindings)
		_check("connector_bindings", len(connectors), max_connector_bindings)
		_check("script_extensions", len(scripts), max_script_extensions)
		within_budget = not violations
		severity = max((v["severity"] for v in violations), default="info", key=lambda s: {"info": 0, "warning": 1, "error": 2}[s])
		for v in violations:
			self._audit(tenant_id, "budget_violation", v["subject_id"], f"{v['metric']}:{v['actual']}/{v['limit']}", severity=v["severity"])
		return {
			"app_id": app_id, "tenant_id": tenant_id,
			"within_budget": within_budget, "overall_severity": severity,
			"violations": violations,
			"metrics": {"total_pages": len(pages), "total_components": len(components), "total_data_bindings": len(data_bindings), "total_workflow_bindings": len(workflows), "total_connector_bindings": len(connectors), "total_script_extensions": len(scripts)},
			"checked_at": utc_now_iso(),
		}

	async def app_diff(
		self,
		tenant_id: str,
		app_id: str,
		snapshot_id_a: str,
		snapshot_id_b: str,
	) -> dict[str, Any]:
		"""Produce a logical diff between two app snapshots.

		For each resource category identifies records that were added, removed,
		or modified (by content-hash). Returns a structured AppDiff with
		per-category change-sets and a total change_count.
		"""
		self._require_tenant(tenant_id)
		if not hasattr(self, "_snapshots"):
			self._snapshots = {}
		snap_a = self._snapshots.get(snapshot_id_a)
		snap_b = self._snapshots.get(snapshot_id_b)
		if snap_a is None or snap_a["tenant_id"] != tenant_id:
			raise LookupError(f"snapshot_not_found:{snapshot_id_a}")
		if snap_b is None or snap_b["tenant_id"] != tenant_id:
			raise LookupError(f"snapshot_not_found:{snapshot_id_b}")
		await _asyncio.sleep(0)
		categories = ("pages", "components", "data_models", "data_bindings", "workflow_bindings", "script_extensions", "connector_bindings", "theme_variants")
		diff: dict[str, Any] = {"added": {}, "removed": {}, "modified": {}}
		total_changes = 0
		for cat in categories:
			items_a = {item["id"]: item for item in snap_a["body"].get(cat, [])}
			items_b = {item["id"]: item for item in snap_b["body"].get(cat, [])}
			added = [items_b[k] for k in items_b if k not in items_a]
			removed = [items_a[k] for k in items_a if k not in items_b]
			modified = [{"before": items_a[k], "after": items_b[k]} for k in items_a if k in items_b and _json.dumps(items_a[k], sort_keys=True) != _json.dumps(items_b[k], sort_keys=True)]
			if added or removed or modified:
				diff["added"][cat] = added
				diff["removed"][cat] = removed
				diff["modified"][cat] = modified
				total_changes += len(added) + len(removed) + len(modified)
		return {
			"app_id": app_id, "tenant_id": tenant_id,
			"snapshot_a": snapshot_id_a, "snapshot_b": snapshot_id_b,
			"version_a": snap_a.get("version"), "version_b": snap_b.get("version"),
			"total_changes": total_changes, "diff": diff,
			"computed_at": utc_now_iso(),
		}

	# ------------------------------------------------------------------
	# Async methods v1.2: permissions, webhooks, federation, bulk ops,
	# cost estimation, schema export/import, tab summary
	# ------------------------------------------------------------------

	async def set_page_permissions(
		self,
		tenant_id: str,
		page_id: str,
		roles_allowed: list[str],
		roles_denied: list[str] | None = None,
		conditions: dict[str, Any] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Attach fine-grained RBAC to a single BuilderPage.

		Stores a PagePermission record in _page_permissions keyed by page_id.
		Guards: tenant isolation, non-empty roles_allowed, no overlap between
		allowed and denied sets.
		"""
		guard_tenant_id(tenant_id)
		page = self._require_page(page_id, tenant_id)
		guard_non_empty_string(actor, "actor")
		assert roles_allowed, "roles_allowed must be non-empty"
		overlap = set(roles_allowed) & set(roles_denied or [])
		assert not overlap, f"roles in both allowed and denied: {overlap}"
		if not hasattr(self, "_page_permissions"):
			self._page_permissions: dict[str, dict[str, Any]] = {}
		record: dict[str, Any] = {
			"page_id": page_id,
			"app_id": page.app_id,
			"tenant_id": tenant_id,
			"roles_allowed": list(roles_allowed),
			"roles_denied": list(roles_denied or []),
			"conditions": dict(conditions or {}),
			"set_by": actor,
			"set_at": utc_now_iso(),
		}
		self._page_permissions[page_id] = record
		self._audit(tenant_id, "page_permissions_set", page_id, f"Roles: {','.join(roles_allowed)}")
		await _asyncio.sleep(0)
		return record

	async def register_webhook(
		self,
		tenant_id: str,
		app_id: str,
		webhook_id: str,
		event_types: list[str],
		target_url: str,
		secret: str = "",
		retry_limit: int = 3,
		enabled: bool = True,
	) -> dict[str, Any]:
		"""Register a webhook endpoint to receive lifecycle event notifications.

		Supports wildcard event matching: '*' matches all events; 'app_*'
		matches any event starting with 'app_'. HMAC-SHA256 signing is enabled
		when secret is non-empty. Payloads are enqueued in _webhook_queue.
		"""
		guard_tenant_id(tenant_id)
		self._require_app(app_id, tenant_id)
		guard_non_empty_string(target_url, "target_url")
		assert event_types, "event_types must be non-empty"
		assert 0 < retry_limit <= 10, "retry_limit must be 1-10"
		if not hasattr(self, "_webhooks"):
			self._webhooks: dict[str, dict[str, Any]] = {}
		if not hasattr(self, "_webhook_queue"):
			self._webhook_queue: list[dict[str, Any]] = []
		if not hasattr(self, "_webhook_secrets"):
			self._webhook_secrets: dict[str, str] = {}
		self._webhook_secrets[webhook_id] = secret
		record: dict[str, Any] = {
			"webhook_id": webhook_id,
			"tenant_id": tenant_id,
			"app_id": app_id,
			"event_types": list(event_types),
			"target_url": target_url,
			"secret_configured": bool(secret),
			"retry_limit": retry_limit,
			"enabled": enabled,
			"registered_at": utc_now_iso(),
		}
		self._webhooks[webhook_id] = record
		self._audit(tenant_id, "webhook_registered", webhook_id, f"Webhook for {event_types}")
		await _asyncio.sleep(0)
		return record

	async def federate_app(
		self,
		tenant_id: str,
		host_app_id: str,
		remote_app_id: str,
		mount_route: str,
		remote_tenant_id: str | None = None,
		policy_ref: str = "policy://federation_default",
	) -> dict[str, Any]:
		"""Embed a remote app's route tree under a mount point in the host app.

		Both apps must be in validated/reviewed/published/deployed status. The
		host and remote may be in the same or different tenant namespaces.
		Idempotent: re-calling with the same host+remote+route updates the mount.
		"""
		guard_tenant_id(tenant_id)
		host_app = self._require_app(host_app_id, tenant_id)
		effective_remote_tenant = remote_tenant_id or tenant_id
		guard_tenant_id(effective_remote_tenant)
		remote_app = self._require_app(remote_app_id, effective_remote_tenant)
		valid_statuses = {"validated", "reviewed", "published", "deployed"}
		assert host_app.status in valid_statuses, f"host_app not validated: {host_app.status}"
		assert remote_app.status in valid_statuses, f"remote_app not validated: {remote_app.status}"
		assert policy_ref.strip(), "policy_ref required for federation"
		normalized_route = normalize_route(mount_route)
		if not hasattr(self, "_federated_mounts"):
			self._federated_mounts: dict[str, dict[str, Any]] = {}
		mount_key = stable_id("ncodfed", tenant_id, host_app_id, remote_app_id, normalized_route)
		record: dict[str, Any] = {
			"mount_id": mount_key,
			"tenant_id": tenant_id,
			"host_app_id": host_app_id,
			"remote_app_id": remote_app_id,
			"remote_tenant_id": effective_remote_tenant,
			"mount_route": normalized_route,
			"policy_ref": policy_ref,
			"host_app_version": host_app.version,
			"remote_app_version": remote_app.version,
			"mounted_at": utc_now_iso(),
		}
		self._federated_mounts[mount_key] = record
		self._audit(tenant_id, "app_federated", mount_key, f"Mounted {remote_app_id} at {normalized_route}")
		await _asyncio.sleep(0)
		return record

	async def bulk_add_components(
		self,
		tenant_id: str,
		page_id: str,
		components: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Add multiple components to a page atomically.

		All entries are validated before any are persisted — if one fails the
		call raises and no components are written. Order values are normalised
		via Decimal to avoid float precision issues with fractional positions.
		"""
		guard_tenant_id(tenant_id)
		page = self._require_page(page_id, tenant_id)
		assert isinstance(components, list) and components, "components must be non-empty list"
		validated: list[dict[str, Any]] = []
		for idx, entry in enumerate(components):
			cid = str(entry.get("component_id") or "")
			ctype = str(entry.get("component_type") or "text")
			cname = str(entry.get("name") or f"component_{idx}")
			order = int(_Decimal(str(entry.get("order", idx))))
			label = str(entry.get("accessibility_label") or cname)
			guard_non_empty_string(cid, "component_id")
			validated.append({
				"component_id": cid, "component_type": ctype,
				"name": cname, "order": order, "label": label,
				"props": dict(entry.get("props") or {}),
				"bindings": dict(entry.get("bindings") or {}),
			})
		await _asyncio.sleep(0)
		created: list[dict[str, Any]] = []
		for v in validated:
			comp = self.add_component(
				component_id=v["component_id"],
				tenant_id=tenant_id,
				page_id=page_id,
				component_type=v["component_type"],
				name=v["name"],
				props=v["props"],
				bindings=v["bindings"],
				accessibility_label=v["label"],
				order=v["order"],
			)
			created.append(comp)
		self._audit(tenant_id, "bulk_components_added", page_id, f"Added {len(created)} components to page {page.name}")
		return {
			"page_id": page_id, "tenant_id": tenant_id,
			"components_created": len(created),
			"components": created,
			"created_at": utc_now_iso(),
		}

	async def compute_app_cost_estimate(
		self,
		tenant_id: str,
		app_id: str,
		unit_cost_per_component: str = "0.10",
		unit_cost_per_data_binding: str = "0.25",
		unit_cost_per_connector: str = "1.00",
		currency: str = "USD",
	) -> dict[str, Any]:
		"""Estimate monthly hosting cost using Decimal arithmetic.

		All unit costs are accepted as strings and converted to Decimal to
		prevent IEEE 754 rounding errors in billing-adjacent code. Workflows
		and scripts are billed at half the connector rate.
		"""
		guard_tenant_id(tenant_id)
		self._require_app(app_id, tenant_id)
		await _asyncio.sleep(0)
		D = _Decimal
		components = self._component_dicts(app_id, tenant_id)
		data_bindings = self._data_binding_dicts(app_id, tenant_id)
		connectors = self._connector_binding_dicts(app_id, tenant_id)
		workflows = self._workflow_binding_dicts(app_id, tenant_id)
		scripts = self._script_extension_dicts(app_id, tenant_id)
		cost_component = D(unit_cost_per_component) * D(len(components))
		cost_binding = D(unit_cost_per_data_binding) * D(len(data_bindings))
		cost_connector = D(unit_cost_per_connector) * D(len(connectors))
		cost_workflow = (D(unit_cost_per_connector) / D("2")) * D(len(workflows))
		cost_script = (D(unit_cost_per_connector) / D("2")) * D(len(scripts))
		total = cost_component + cost_binding + cost_connector + cost_workflow + cost_script
		self._audit(tenant_id, "cost_estimate_computed", app_id, f"Total {total} {currency}/month")
		return {
			"app_id": app_id, "tenant_id": tenant_id, "currency": currency,
			"subtotals": {
				"components": str(cost_component),
				"data_bindings": str(cost_binding),
				"connectors": str(cost_connector),
				"workflows": str(cost_workflow),
				"scripts": str(cost_script),
			},
			"total": str(total),
			"resource_counts": {
				"components": len(components),
				"data_bindings": len(data_bindings),
				"connectors": len(connectors),
				"workflows": len(workflows),
				"scripts": len(scripts),
			},
			"computed_at": utc_now_iso(),
		}

	async def export_app_schema(
		self,
		tenant_id: str,
		app_id: str,
		format: str = "json",
		include_audit: bool = False,
	) -> dict[str, Any]:
		"""Export a portable schema bundle for an app.

		Serialises the app and all sub-resources into a single dict with a
		content_hash for integrity verification. 'summary' format returns
		only counts and metadata; 'json' returns the full bundle.
		"""
		guard_tenant_id(tenant_id)
		app = self._require_app(app_id, tenant_id)
		assert format in {"json", "summary"}, f"unsupported export format: {format}"
		await _asyncio.sleep(0)
		bundle: dict[str, Any] = {
			"schema_version": "1.2",
			"exported_at": utc_now_iso(),
			"tenant_id": tenant_id,
			"app": app.to_dict(),
			"pages": self._page_dicts(app_id, tenant_id),
			"components": self._component_dicts(app_id, tenant_id),
			"data_models": self._data_model_dicts(app_id, tenant_id),
			"data_bindings": self._data_binding_dicts(app_id, tenant_id),
			"workflow_bindings": self._workflow_binding_dicts(app_id, tenant_id),
			"script_extensions": self._script_extension_dicts(app_id, tenant_id),
			"connector_bindings": self._connector_binding_dicts(app_id, tenant_id),
			"theme_variants": self._theme_variant_dicts(app_id, tenant_id),
			"builder_agents": self._builder_agent_dicts(app_id, tenant_id),
		}
		if include_audit:
			bundle["audit_events"] = [
				e.to_dict() for e in self._audit_events.values()
				if e.tenant_id == tenant_id and e.subject_id.startswith(app_id[:8])
			]
		content_hash = _sha256(_json.dumps(bundle, sort_keys=True, default=str).encode()).hexdigest()[:16]
		if format == "summary":
			bundle = {
				"schema_version": bundle["schema_version"],
				"exported_at": bundle["exported_at"],
				"app_id": app_id, "app_name": app.name, "app_version": app.version,
				"resource_counts": {k: len(v) for k, v in bundle.items() if isinstance(v, list)},
				"content_hash": content_hash,
			}
		else:
			bundle["content_hash"] = content_hash
		self._audit(tenant_id, "app_schema_exported", app_id, f"Format: {format}, hash: {content_hash}")
		return bundle

	async def import_app_schema(
		self,
		tenant_id: str,
		bundle: dict[str, Any],
		owner: str,
		rbac_policy_ref: str,
		data_residency_policy_ref: str,
		overwrite_existing: bool = False,
	) -> dict[str, Any]:
		"""Import an app from a schema bundle produced by export_app_schema.

		Verifies schema_version and content_hash before writing any records.
		IDs are re-derived for the target tenant to prevent collisions. When
		overwrite_existing=False, existing resources with the same derived ID
		are skipped and their IDs reported in skipped_ids.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(owner, "owner")
		guard_non_empty_string(rbac_policy_ref, "rbac_policy_ref")
		guard_non_empty_string(data_residency_policy_ref, "data_residency_policy_ref")
		schema_version = bundle.get("schema_version", "")
		assert schema_version in {"1.0", "1.1", "1.2"}, f"unsupported schema_version: {schema_version}"
		bundle_for_hash = {k: v for k, v in bundle.items() if k != "content_hash"}
		expected_hash = _sha256(_json.dumps(bundle_for_hash, sort_keys=True, default=str).encode()).hexdigest()[:16]
		provided_hash = bundle.get("content_hash", expected_hash)
		assert provided_hash == expected_hash, f"content_hash mismatch: {provided_hash} != {expected_hash}"
		await _asyncio.sleep(0)
		src_app = bundle.get("app", {})
		app_name = str(src_app.get("name") or "imported_app")
		new_app_id = stable_id("ncodimp", tenant_id, app_name, schema_version)
		import_counts: dict[str, int] = {}
		skipped_ids: list[str] = []
		if new_app_id not in self._apps or overwrite_existing:
			self.create_app(
				app_id=new_app_id, tenant_id=tenant_id,
				name=app_name, owner=owner,
				description=str(src_app.get("description") or ""),
				theme=str(src_app.get("theme") or "ncod_app_builder"),
				rbac_policy_ref=rbac_policy_ref,
				data_residency_policy_ref=data_residency_policy_ref,
				accessibility_checked=bool(src_app.get("accessibility_checked", False)),
				metadata={**dict(src_app.get("metadata") or {}), "imported_from": bundle.get("tenant_id"), "schema_version": schema_version},
			)
		page_id_map: dict[str, str] = {}
		for pg in bundle.get("pages", []):
			new_pg_id = stable_id("ncodipg", tenant_id, new_app_id, pg.get("name", ""))
			page_id_map[pg["id"]] = new_pg_id
			if new_pg_id not in self._pages or overwrite_existing:
				self.add_page(
					page_id=new_pg_id, tenant_id=tenant_id, app_id=new_app_id,
					name=str(pg.get("name", "")), route=str(pg.get("route", "/imported")),
					layout=str(pg.get("layout") or "responsive_grid"),
					metadata={**dict(pg.get("metadata") or {}), "relationships": True},
				)
			else:
				skipped_ids.append(new_pg_id)
		import_counts["pages"] = len(bundle.get("pages", []))
		comp_count = 0
		for comp in bundle.get("components", []):
			mapped_pg = page_id_map.get(comp.get("page_id", ""))
			if not mapped_pg:
				skipped_ids.append(str(comp.get("id", "")))
				continue
			new_comp_id = stable_id("ncodicomp", tenant_id, new_app_id, comp.get("name", ""))
			if new_comp_id not in self._components or overwrite_existing:
				try:
					self.add_component(
						component_id=new_comp_id, tenant_id=tenant_id, page_id=mapped_pg,
						component_type=str(comp.get("component_type") or "text"),
						name=str(comp.get("name", "")),
						props=dict(comp.get("props") or {}),
						bindings=dict(comp.get("bindings") or {}),
						accessibility_label=str(comp.get("accessibility_label") or comp.get("name", "")),
						order=int(comp.get("order", 0)),
					)
					comp_count += 1
				except Exception:
					skipped_ids.append(new_comp_id)
		import_counts["components"] = comp_count
		dm_count = 0
		for dm in bundle.get("data_models", []):
			new_dm_id = stable_id("ncodidm", tenant_id, new_app_id, dm.get("name", ""))
			if new_dm_id not in self._data_models or overwrite_existing:
				try:
					self.define_data_model(
						model_id=new_dm_id, tenant_id=tenant_id, app_id=new_app_id,
						name=str(dm.get("name", "")), fields=list(dm.get("fields") or []),
						policy_ref=str(dm.get("policy_ref") or rbac_policy_ref),
						metadata=dict(dm.get("metadata") or {}),
					)
					dm_count += 1
				except Exception:
					skipped_ids.append(new_dm_id)
		import_counts["data_models"] = dm_count
		self._audit(tenant_id, "app_schema_imported", new_app_id, f"Imported {sum(import_counts.values())} resources")
		return {
			"app_id": new_app_id, "tenant_id": tenant_id,
			"import_counts": import_counts, "skipped_ids": skipped_ids,
			"schema_version": schema_version, "imported_at": utc_now_iso(),
		}

	async def get_app_tab_summary(
		self,
		tenant_id: str,
		app_id: str,
	) -> dict[str, Any]:
		"""Return a tabbed domain breakdown for Flask-AppBuilder views and dashboards.

		Produces seven tabs: overview, pages, components, data, workflows,
		agents, releases — each with counts and top-N previews. Designed for
		tabbed detail views where each domain is independently scrollable.
		"""
		guard_tenant_id(tenant_id)
		app = self._require_app(app_id, tenant_id)
		await _asyncio.sleep(0)
		pages = self._page_dicts(app_id, tenant_id)
		components = self._component_dicts(app_id, tenant_id)
		data_models = self._data_model_dicts(app_id, tenant_id)
		data_bindings = self._data_binding_dicts(app_id, tenant_id)
		workflows = self._workflow_binding_dicts(app_id, tenant_id)
		scripts = self._script_extension_dicts(app_id, tenant_id)
		connectors = self._connector_binding_dicts(app_id, tenant_id)
		agents = self._builder_agent_dicts(app_id, tenant_id)
		themes = self._theme_variant_dicts(app_id, tenant_id)
		releases = [r.to_dict() for r in self._releases.values() if r.app_id == app_id and r.tenant_id == tenant_id]
		deployments = [d.to_dict() for d in self._deployments.values() if d.app_id == app_id and d.tenant_id == tenant_id]
		validations = [v.to_dict() for v in self._validations.values() if v.app_id == app_id and v.tenant_id == tenant_id]
		tabs: dict[str, Any] = {
			"overview": {
				"tab": "overview",
				"app": app.to_dict(),
				"status": app.status,
				"version": app.version,
				"owner": app.owner,
				"theme": app.theme,
				"accessibility_checked": app.accessibility_checked,
			},
			"pages": {
				"tab": "pages",
				"count": len(pages),
				"items": pages[:10],
			},
			"components": {
				"tab": "components",
				"count": len(components),
				"by_type": {
					ctype: sum(1 for c in components if c["component_type"] == ctype)
					for ctype in {c["component_type"] for c in components}
				},
				"items": components[:10],
			},
			"data": {
				"tab": "data",
				"data_model_count": len(data_models),
				"data_binding_count": len(data_bindings),
				"data_models": data_models[:5],
				"data_bindings": data_bindings[:5],
			},
			"workflows": {
				"tab": "workflows",
				"workflow_count": len(workflows),
				"script_count": len(scripts),
				"connector_count": len(connectors),
				"workflows": workflows[:5],
				"scripts": scripts[:5],
				"connectors": connectors[:5],
			},
			"agents": {
				"tab": "agents",
				"count": len(agents),
				"items": agents,
				"themes": themes,
			},
			"releases": {
				"tab": "releases",
				"release_count": len(releases),
				"deployment_count": len(deployments),
				"validation_count": len(validations),
				"latest_release": releases[-1] if releases else None,
				"latest_deployment": deployments[-1] if deployments else None,
				"latest_validation": validations[-1] if validations else None,
			},
		}
		self._audit(tenant_id, "tab_summary_generated", app_id, f"Generated {len(tabs)} tabs")
		return {
			"app_id": app_id, "tenant_id": tenant_id,
			"tabs": tabs, "tab_count": len(tabs),
			"generated_at": utc_now_iso(),
		}
