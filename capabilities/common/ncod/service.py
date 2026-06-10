"""Service layer for executable no-code/low-code app composition."""

from __future__ import annotations

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
