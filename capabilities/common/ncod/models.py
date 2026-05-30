"""Domain models for the No-Code/Low-Code Builder capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
	"""Return a compact UTC timestamp string for in-process builder records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class BuilderApp:
	"""Tenant-scoped application assembled through the no-code builder."""

	id: str
	tenant_id: str
	name: str
	owner: str
	description: str = ""
	status: str = "draft"
	version: str = "0.1.0"
	theme: str = "ncod_app_builder"
	accessibility_checked: bool = False
	rbac_policy_ref: str = ""
	data_residency_policy_ref: str = ""
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"description": self.description,
			"status": self.status,
			"version": self.version,
			"theme": self.theme,
			"accessibility_checked": self.accessibility_checked,
			"rbac_policy_ref": self.rbac_policy_ref,
			"data_residency_policy_ref": self.data_residency_policy_ref,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class BuilderPage:
	"""Page or form canvas within a builder application."""

	id: str
	tenant_id: str
	app_id: str
	name: str
	route: str
	layout: str = "responsive_grid"
	status: str = "draft"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"name": self.name,
			"route": self.route,
			"layout": self.layout,
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass
class BuilderComponent:
	"""Component placed on a builder page."""

	id: str
	tenant_id: str
	app_id: str
	page_id: str
	component_type: str
	name: str
	props: dict[str, Any] = field(default_factory=dict)
	bindings: dict[str, Any] = field(default_factory=dict)
	accessibility_label: str = ""
	order: int = 0
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"page_id": self.page_id,
			"component_type": self.component_type,
			"name": self.name,
			"props": dict(self.props),
			"bindings": dict(self.bindings),
			"accessibility_label": self.accessibility_label,
			"order": self.order,
			"created_at": self.created_at,
		}


@dataclass
class DataModelDefinition:
	"""Business entity model created in the no-code builder."""

	id: str
	tenant_id: str
	app_id: str
	name: str
	fields: list[dict[str, Any]] = field(default_factory=list)
	policy_ref: str = ""
	validated: bool = False
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"name": self.name,
			"fields": [dict(field) for field in self.fields],
			"policy_ref": self.policy_ref,
			"validated": self.validated,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass
class DataBinding:
	"""Data source binding exposed to app pages and components."""

	id: str
	tenant_id: str
	app_id: str
	name: str
	source_type: str
	source_ref: str
	schema: dict[str, Any] = field(default_factory=dict)
	validated: bool = False
	policy_ref: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"name": self.name,
			"source_type": self.source_type,
			"source_ref": self.source_ref,
			"schema": dict(self.schema),
			"validated": self.validated,
			"policy_ref": self.policy_ref,
			"created_at": self.created_at,
		}


@dataclass
class WorkflowBinding:
	"""Workflow or automation binding attached to an app."""

	id: str
	tenant_id: str
	app_id: str
	trigger: str
	workflow_ref: str
	policy_ref: str = ""
	enabled: bool = True
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"trigger": self.trigger,
			"workflow_ref": self.workflow_ref,
			"policy_ref": self.policy_ref,
			"enabled": self.enabled,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass
class ThemeVariant:
	"""Tenant-approved visual theme variant for a builder app."""

	id: str
	tenant_id: str
	app_id: str
	name: str
	tokens: dict[str, Any] = field(default_factory=dict)
	policy_ref: str = ""
	approved: bool = False
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"name": self.name,
			"tokens": dict(self.tokens),
			"policy_ref": self.policy_ref,
			"approved": self.approved,
			"created_at": self.created_at,
		}


@dataclass
class ScriptExtension:
	"""Approved low-code script extension for validation or automation."""

	id: str
	tenant_id: str
	app_id: str
	name: str
	hook: str
	script_ref: str
	policy_ref: str
	status: str = "approved"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"name": self.name,
			"hook": self.hook,
			"script_ref": self.script_ref,
			"policy_ref": self.policy_ref,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class ConnectorBinding:
	"""External connector binding attached to a builder app."""

	id: str
	tenant_id: str
	app_id: str
	name: str
	connector_ref: str
	policy_ref: str
	status: str = "active"
	scopes: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"name": self.name,
			"connector_ref": self.connector_ref,
			"policy_ref": self.policy_ref,
			"status": self.status,
			"scopes": list(self.scopes),
			"created_at": self.created_at,
		}


@dataclass
class BuilderAgent:
	"""First-class AI agent configured to assist with app composition."""

	id: str
	tenant_id: str
	app_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = False
	policy_ref: str = ""
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"policy_ref": self.policy_ref,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class ValidationResult:
	"""Readiness validation result for an application version."""

	id: str
	tenant_id: str
	app_id: str
	passed: bool
	checks: dict[str, bool] = field(default_factory=dict)
	issues: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"passed": self.passed,
			"checks": dict(self.checks),
			"issues": list(self.issues),
			"created_at": self.created_at,
		}


@dataclass
class PublishRelease:
	"""Governed publication of a builder app version."""

	id: str
	tenant_id: str
	app_id: str
	version: str
	target_environment: str
	approval_recorded: bool
	change_review_recorded: bool = False
	status: str = "published"
	approval_ref: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"version": self.version,
			"target_environment": self.target_environment,
			"approval_recorded": self.approval_recorded,
			"change_review_recorded": self.change_review_recorded,
			"status": self.status,
			"approval_ref": self.approval_ref,
			"created_at": self.created_at,
		}


@dataclass
class DeploymentRecord:
	"""Governed deployment of a published application release."""

	id: str
	tenant_id: str
	app_id: str
	release_id: str
	target_environment: str
	target_runtime: str
	target_ref: str
	approval_recorded: bool
	rollback_plan_ref: str
	status: str = "deployed"
	approval_ref: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"app_id": self.app_id,
			"release_id": self.release_id,
			"target_environment": self.target_environment,
			"target_runtime": self.target_runtime,
			"target_ref": self.target_ref,
			"approval_recorded": self.approval_recorded,
			"approval_ref": self.approval_ref,
			"rollback_plan_ref": self.rollback_plan_ref,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class NcodAuditEvent:
	"""Audit event emitted by builder operations."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	severity: str = "info"
	created_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"severity": self.severity,
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
		}


# Compatibility alias for older package callers that import NcodRecord.
NcodRecord = BuilderApp
