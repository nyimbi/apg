"""In-memory models for APG Network Inventory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class InvAsset:
	id: str
	tenant_id: str
	asset_type: str
	serial_number: str
	vendor: str
	model: str
	location: str
	status: str
	commissioned_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class InvCircuit:
	id: str
	tenant_id: str
	circuit_type: str
	a_end: str
	z_end: str
	capacity: str
	status: str
	provisioned_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class InvIpBlock:
	id: str
	tenant_id: str
	ip_version: str
	prefix: str
	prefix_length: int
	block_type: str
	vrf: str
	allocated_to: str | None
	allocated_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class InvTopology:
	id: str
	tenant_id: str
	topology_type: str
	domain: str
	name: str
	description: str
	nodes: str
	edges: str
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class InvSite:
	id: str
	tenant_id: str
	site_name: str
	site_type: str
	latitude: float
	longitude: float
	address: str
	region: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class InvReconciliation:
	id: str
	tenant_id: str
	asset_id: str
	discrepancy_description: str
	approval_reference: str
	resolved_by: str | None
	resolved_at: str | None
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class InvAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
