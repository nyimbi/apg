"""Dependency-light runtime records for central configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: str) -> str:
	seed = "::".join(str(part) for part in parts)
	return f"{prefix}_{sha256(seed.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class ConfigNamespaceRecord:
	id: str
	tenant_id: str
	name: str
	environment: str
	owner_id: str
	path_prefix: str
	capability_id: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ConfigurationRecord:
	id: str
	tenant_id: str
	namespace_id: str
	key_path: str
	value: dict[str, Any]
	version: int
	owner_id: str
	restricted: bool = False
	secret: bool = False
	schema: dict[str, Any] | None = None
	secret_reference: str | None = None
	status: str = "draft"
	validation_evidence: str | None = None
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		data = asdict(self)
		if self.secret:
			data["value"] = {"redacted": True, "secret_reference": self.secret_reference}
		return data


@dataclass
class ConfigDeploymentRecord:
	id: str
	tenant_id: str
	configuration_id: str
	environment: str
	impact_level: str
	status: str
	approved_by: str | None = None
	canary_evidence: str | None = None
	event_stream: str = "bytewax"
	created_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ConfigTemplateRecord:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	values: dict[str, Any]
	variable_schema: dict[str, Any]
	shared: bool = False
	reviewed_by: str | None = None
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ConfigDriftRecord:
	id: str
	tenant_id: str
	configuration_id: str
	expected_version: int
	observed_version: int
	severity: str
	status: str = "open"
	created_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ConfigAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	instructions: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ConfigAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	entity_id: str
	actor_id: str
	created_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
