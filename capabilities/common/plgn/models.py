"""Domain models for APG Plugin/Extension Framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class PluginManifest:
	id: str
	tenant_id: str
	name: str
	owner: str
	version: str
	publisher: str
	release_channel: str
	permissions: tuple[str, ...] = ()
	dependencies: tuple[str, ...] = ()
	external_plugin: bool = False
	signature_verified: bool = False
	manifest_schema_valid: bool = False
	dependency_validation_passed: bool = False
	supply_chain_scan_passed: bool = False
	external_review_recorded: bool = False
	status: str = "registered"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"version": self.version,
			"publisher": self.publisher,
			"release_channel": self.release_channel,
			"permissions": list(self.permissions),
			"dependencies": list(self.dependencies),
			"external_plugin": self.external_plugin,
			"signature_verified": self.signature_verified,
			"manifest_schema_valid": self.manifest_schema_valid,
			"dependency_validation_passed": self.dependency_validation_passed,
			"supply_chain_scan_passed": self.supply_chain_scan_passed,
			"external_review_recorded": self.external_review_recorded,
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class PermissionReview:
	id: str
	tenant_id: str
	plugin_id: str
	reviewer: str
	approved_scopes: tuple[str, ...]
	denied_scopes: tuple[str, ...] = ()
	secret_access_allowed: bool = False
	notes: str = ""
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plugin_id": self.plugin_id,
			"reviewer": self.reviewer,
			"approved_scopes": list(self.approved_scopes),
			"denied_scopes": list(self.denied_scopes),
			"secret_access_allowed": self.secret_access_allowed,
			"notes": self.notes,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class SandboxPolicy:
	id: str
	tenant_id: str
	plugin_id: str
	policy_name: str
	network_access: str = "deny"
	filesystem_access: str = "read_only"
	secret_access: str = "deny"
	tool_allowlist: tuple[str, ...] = ()
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plugin_id": self.plugin_id,
			"policy_name": self.policy_name,
			"network_access": self.network_access,
			"filesystem_access": self.filesystem_access,
			"secret_access": self.secret_access,
			"tool_allowlist": list(self.tool_allowlist),
			"created_at": isoformat(self.created_at),
		}


@dataclass
class MarketplaceListing:
	id: str
	tenant_id: str
	plugin_id: str
	title: str
	publisher_verified: bool
	curated: bool
	install_policy: str
	status: str = "listed"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plugin_id": self.plugin_id,
			"title": self.title,
			"publisher_verified": self.publisher_verified,
			"curated": self.curated,
			"install_policy": self.install_policy,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PluginRelease:
	id: str
	tenant_id: str
	plugin_id: str
	version: str
	channel: str
	signature_ref: str
	status: str
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plugin_id": self.plugin_id,
			"version": self.version,
			"channel": self.channel,
			"signature_ref": self.signature_ref,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PluginInstallation:
	id: str
	tenant_id: str
	plugin_id: str
	installed_by: str
	status: str = "installed"
	enabled_at: datetime | None = None
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plugin_id": self.plugin_id,
			"installed_by": self.installed_by,
			"status": self.status,
			"enabled_at": isoformat(self.enabled_at) if self.enabled_at else None,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PlgnAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
		}


PlgnRecord = PluginManifest


@dataclass
class PlgnAgent:
	"""Registered AI plugin governance agent."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}
