"""Pydantic v2 models for APG ITSM CMDB."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class ItCmdbCI(BaseModel):
	"""Configuration Item — the atomic unit of the CMDB."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	ci_type: str											# server, vm, application, …
	status: str = "active"
	environment: str										# production, staging, …
	owner_id: str											# responsible person/team
	hostname: str | None = None
	ip_addresses: list[str] = Field(default_factory=list)
	mac_addresses: list[str] = Field(default_factory=list)
	serial_number: str | None = None
	asset_tag: str | None = None
	manufacturer: str | None = None
	model: str | None = None
	os_name: str | None = None
	os_version: str | None = None
	cpu_cores: int | None = None
	ram_gb: float | None = None
	disk_gb: float | None = None
	location: str | None = None							# rack, datacenter, cloud region
	datacenter: str | None = None
	cloud_provider: str | None = None					# aws, gcp, azure
	cloud_region: str | None = None
	cloud_instance_id: str | None = None
	tags: dict[str, str] = Field(default_factory=dict)
	custom_attributes: dict[str, Any] = Field(default_factory=dict)
	health_score: float = 100.0							# 0–100
	health_status: str = "healthy"
	last_seen_at: str | None = None
	discovery_method: str | None = None
	discovery_job_id: str | None = None
	created_at: str = Field(default_factory=_now_iso)
	updated_at: str = Field(default_factory=_now_iso)
	decommissioned_at: str | None = None
	version: int = 1										# optimistic lock counter

	@field_validator("health_score")
	@classmethod
	def _validate_health_score(cls, v: float) -> float:
		assert 0.0 <= v <= 100.0, "health_score must be 0–100"
		return v


class ItCmdbRelationship(BaseModel):
	"""Directed relationship between two CIs."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	source_ci_id: str
	target_ci_id: str
	relationship_type: str								# depends_on, hosts, runs_on, …
	description: str | None = None
	strength: float = 1.0									# 0–1 confidence weight
	bidirectional: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str = Field(default_factory=_now_iso)
	created_by: str = "system"
	valid_from: str | None = None
	valid_until: str | None = None


class ItCmdbChangeRecord(BaseModel):
	"""Tracks attribute-level changes to a CI, linked to itsm_chg change tickets."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	ci_id: str
	change_ticket_id: str | None = None				# FK to itsm_chg.ItChange
	changed_by: str
	change_type: str										# add, modify, remove, decommission
	field_name: str | None = None
	old_value: str | None = None
	new_value: str | None = None
	diff_payload: dict[str, Any] = Field(default_factory=dict)
	status: str = "pending"								# pending, approved, applied, failed, rolled_back
	approver_id: str | None = None
	approved_at: str | None = None
	applied_at: str | None = None
	rollback_reason: str | None = None
	created_at: str = Field(default_factory=_now_iso)
	notes: str = ""


class ItDiscoveryJob(BaseModel):
	"""Auto-discovery scan job targeting a network range or cloud account."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	discovery_method: str								# network_scan, agent_based, cloud_api, …
	target: str											# CIDR, cloud account id, hostname
	environment: str
	schedule_cron: str | None = None
	credentials_ref: str | None = None
	status: str = "pending"								# pending, running, completed, failed, cancelled
	started_at: str | None = None
	completed_at: str | None = None
	ci_discovered: int = 0
	ci_updated: int = 0
	ci_decommissioned: int = 0
	error_message: str | None = None
	created_at: str = Field(default_factory=_now_iso)
	created_by: str = "system"
	last_run_at: str | None = None
	run_count: int = 0
	result_summary: dict[str, Any] = Field(default_factory=dict)
