"""Service layer for executable no-code/low-code app composition."""

from __future__ import annotations

from typing import Any

from .builder_runtime import (
	binding_schema_valid,
	bump_patch_version,
	component_accessible,
	normalize_app_status,
	normalize_component_type,
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
	BuilderComponent,
	BuilderPage,
	ConnectorBinding,
	DataBinding,
	NcodAuditEvent,
	PublishRelease,
	ScriptExtension,
	ValidationResult,
	WorkflowBinding,
	utc_now_iso,
)


class NcodService:
	"""In-process app builder enforcing NCOD ownership, policy, and publish gates."""

	def __init__(self) -> None:
		self._apps: dict[str, BuilderApp] = {}
		self._pages: dict[str, BuilderPage] = {}
		self._components: dict[str, BuilderComponent] = {}
		self._data_bindings: dict[str, DataBinding] = {}
		self._workflow_bindings: dict[str, WorkflowBinding] = {}
		self._script_extensions: dict[str, ScriptExtension] = {}
		self._connector_bindings: dict[str, ConnectorBinding] = {}
		self._validations: dict[str, ValidationResult] = {}
		self._releases: dict[str, PublishRelease] = {}
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
		if not component_accessible(normalized_type, accessibility_label, dict(props or {})):
			raise PermissionError("accessibility_label_required")
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
		binding = DataBinding(
			id=binding_id,
			tenant_id=tenant_id,
			app_id=app.id,
			name=name,
			source_type=normalize_source_type(source_type),
			source_ref=source_ref,
			schema=dict(schema),
			validated=binding_schema_valid(schema),
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
		enabled: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		binding = WorkflowBinding(
			id=binding_id,
			tenant_id=tenant_id,
			app_id=app.id,
			trigger=trigger,
			workflow_ref=workflow_ref,
			enabled=enabled,
			metadata=dict(metadata or {}),
		)
		self._workflow_bindings[binding.id] = binding
		self._touch_app(app)
		self._audit(tenant_id, "workflow_attached", binding.id, f"Attached workflow {workflow_ref}")
		return binding.to_dict()

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

	def validate_app(self, validation_id: str, tenant_id: str, app_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		app = self._require_app(app_id, tenant_id)
		checks, issues = validation_checks(
			app.to_dict(),
			self._page_dicts(app.id, tenant_id),
			self._component_dicts(app.id, tenant_id),
			self._data_binding_dicts(app.id, tenant_id),
			self._script_extension_dicts(app.id, tenant_id),
			self._connector_binding_dicts(app.id, tenant_id),
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
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_app",
			"approval_recorded": approval_recorded,
			"production_change": target == "production",
			"change_review_recorded": change_review_recorded,
			"script_extension_present": bool(self._script_extension_dicts(app.id, tenant_id)),
			"script_policy_attached": all(item["policy_ref"] for item in self._script_extension_dicts(app.id, tenant_id)),
			"external_connector_present": bool(self._connector_binding_dicts(app.id, tenant_id)),
			"connector_policy_attached": all(item["policy_ref"] for item in self._connector_binding_dicts(app.id, tenant_id)),
		})
		self._raise_if_denied(result)
		self._raise_if_review_required(result)
		latest_validation = self._latest_validation(app.id, tenant_id)
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
			rbac_policy_ref=str(metadata.get("rbac_policy_ref") or ""),
			data_residency_policy_ref=str(metadata.get("data_residency_policy_ref") or ""),
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

	def list_data_bindings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._data_bindings, tenant_id)

	def list_workflow_bindings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._workflow_bindings, tenant_id)

	def list_script_extensions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._script_extensions, tenant_id)

	def list_connector_bindings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._connector_bindings, tenant_id)

	def list_validations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._validations, tenant_id)

	def list_releases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._releases, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		apps = [item for item in self._apps.values() if item.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"app_count": len(apps),
			"published_app_count": sum(1 for item in apps if item.status == "published"),
			"page_count": len(self.list_pages(tenant_id)),
			"component_count": len(self.list_components(tenant_id)),
			"data_binding_count": len(self.list_data_bindings(tenant_id)),
			"workflow_binding_count": len(self.list_workflow_bindings(tenant_id)),
			"script_extension_count": len(self.list_script_extensions(tenant_id)),
			"connector_binding_count": len(self.list_connector_bindings(tenant_id)),
			"release_count": len(self.list_releases(tenant_id)),
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

	def _data_binding_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._data_bindings.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _script_extension_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._script_extensions.values() if item.app_id == app_id and item.tenant_id == tenant_id]

	def _connector_binding_dicts(self, app_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._connector_bindings.values() if item.app_id == app_id and item.tenant_id == tenant_id]

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
