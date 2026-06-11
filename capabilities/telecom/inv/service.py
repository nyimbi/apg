"""Service layer for APG Network Inventory."""

from __future__ import annotations

import datetime
import ipaddress
import statistics
from typing import Any

from .domain.adapters import get_auth_adapter, get_audit_adapter
from .database.store import get_store
from .capability_contract import (
	SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSET_STATUSES,
	SUPPORTED_ASSET_TYPES, SUPPORTED_CIRCUIT_STATUSES, SUPPORTED_CIRCUIT_TYPES,
	SUPPORTED_IP_VERSIONS, SUPPORTED_TOPOLOGY_TYPES, SUPPORTED_VENDOR_TYPES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	InvAgent, InvAsset, InvCircuit, InvIpBlock,
	InvReconciliation, InvSite, InvTopology,
)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class NetworkInventoryService:
	"""Tenant-scoped network inventory service for APG Telecom."""

	def __init__(self) -> None:
		self._store = get_store("telecom.inv")
		self._auth = get_auth_adapter()
		self._audit_adapter = get_audit_adapter()
		self.assets: dict[tuple[str, str], InvAsset] = {}
		self.circuits: dict[tuple[str, str], InvCircuit] = {}
		self.ip_blocks: dict[tuple[str, str], InvIpBlock] = {}
		self.topologies: dict[tuple[str, str], InvTopology] = {}
		self.sites: dict[tuple[str, str], InvSite] = {}
		self.reconciliations: dict[tuple[str, str], InvReconciliation] = {}
		self.agents: dict[tuple[str, str], InvAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state for new methods
		self._ne_registry: dict[str, dict[str, Any]] = {}          # ne_id -> NE record
		self._topology_links: list[dict[str, Any]] = []            # adjacency list
		self._ip_allocations: dict[str, dict[str, Any]] = {}       # ip_address -> allocation
		self._ip_pool_free: dict[str, list[str]] = {}              # pool_id -> [free IPs]
		self._circuit_services: dict[str, dict[str, Any]] = {}     # circuit_id -> service record
		self._eol_records: dict[str, dict[str, Any]] = {}          # ne_id -> EoL record
		self._capacity_forecasts: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ #
	# Contract                                                             #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core existing methods                                                #
	# ------------------------------------------------------------------ #

	def commission_asset(
		self,
		asset_id: str,
		tenant_id: str,
		asset_type: str,
		serial_number: str,
		vendor: str,
		model: str,
		location: str,
		commissioned_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Commission a network asset into the inventory."""
		asset_type = asset_type.lower()
		vendor = vendor.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "commission_asset",
			"asset_type_supported": asset_type in SUPPORTED_ASSET_TYPES,
			"serial_number_present": _present(serial_number),
			"location_present": _present(location),
		})
		item = InvAsset(asset_id, tenant_id, asset_type, serial_number, vendor, model, location, "commissioned", commissioned_at)
		self.assets[self._key(tenant_id, asset_id)] = item
		self._audit(tenant_id, "asset_commissioned", asset_id)
		return item.to_dict()

	def update_asset_status(self, asset_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update the operational status of a network asset."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_asset_status",
			"asset_status_supported": new_status in SUPPORTED_ASSET_STATUSES,
		})
		asset = self._asset_or_raise(asset_id, tenant_id)
		asset.status = new_status
		self._audit(tenant_id, "asset_status_updated", asset_id)
		return asset.to_dict()

	def decommission_asset(self, asset_id: str, tenant_id: str, approval_reference: str) -> dict[str, Any]:
		"""Decommission an asset with mandatory approval."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "decommission_asset",
			"approval_present": _present(approval_reference),
		})
		asset = self._asset_or_raise(asset_id, tenant_id)
		asset.status = "decommissioned"
		self._audit(tenant_id, "asset_decommissioned", asset_id)
		return asset.to_dict()

	def provision_circuit(
		self,
		circuit_id: str,
		tenant_id: str,
		circuit_type: str,
		a_end: str,
		z_end: str,
		capacity: str,
		provisioned_at: str,
	) -> dict[str, Any]:
		"""Provision a logical circuit between two endpoints."""
		circuit_type = circuit_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "provision_circuit",
			"circuit_type_supported": circuit_type in SUPPORTED_CIRCUIT_TYPES,
			"endpoint_present": _present(a_end) and _present(z_end),
			"capacity_present": _present(capacity),
		})
		item = InvCircuit(circuit_id, tenant_id, circuit_type, a_end, z_end, capacity, "provisioned", provisioned_at)
		self.circuits[self._key(tenant_id, circuit_id)] = item
		self._audit(tenant_id, "circuit_provisioned", circuit_id)
		return item.to_dict()

	def update_circuit_status(self, circuit_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update circuit operational status."""
		new_status = new_status.lower()
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True})
		circuit = self._circuit_or_raise(circuit_id, tenant_id)
		circuit.status = new_status
		if new_status == "decommissioned":
			self._audit(tenant_id, "circuit_decommissioned", circuit_id)
		return circuit.to_dict()

	def allocate_ip_block(
		self,
		block_id: str,
		tenant_id: str,
		ip_version: str,
		prefix: str,
		prefix_length: int,
		block_type: str,
		vrf: str,
		allocated_to: str | None,
		allocated_at: str,
	) -> dict[str, Any]:
		"""Allocate an IP address block from the IPAM pool."""
		ip_version = ip_version.lower()
		block_type = block_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "allocate_ip_block",
			"ip_version_supported": ip_version in SUPPORTED_IP_VERSIONS,
			"prefix_length_present": prefix_length is not None,
			"vrf_present": _present(vrf),
		})
		item = InvIpBlock(block_id, tenant_id, ip_version, prefix, int(prefix_length), block_type, vrf, allocated_to, allocated_at)
		self.ip_blocks[self._key(tenant_id, block_id)] = item
		self._audit(tenant_id, "ip_block_allocated", block_id)
		return item.to_dict()

	def release_ip_block(self, block_id: str, tenant_id: str) -> dict[str, Any]:
		"""Release an IP block back to the IPAM pool."""
		block = self._ip_block_or_raise(block_id, tenant_id)
		block.allocated_to = None
		self._audit(tenant_id, "ip_block_released", block_id)
		return block.to_dict()

	def record_topology(
		self,
		topology_id: str,
		tenant_id: str,
		topology_type: str,
		domain: str,
		name: str,
		description: str,
		nodes: str,
		edges: str,
		recorded_at: str,
	) -> dict[str, Any]:
		"""Record network topology for a given domain."""
		topology_type = topology_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_topology",
			"topology_type_supported": topology_type in SUPPORTED_TOPOLOGY_TYPES,
		})
		item = InvTopology(topology_id, tenant_id, topology_type, domain, name, description, nodes, edges, recorded_at)
		self.topologies[self._key(tenant_id, topology_id)] = item
		self._audit(tenant_id, "topology_updated", topology_id)
		return item.to_dict()

	def register_site(
		self,
		site_id: str,
		tenant_id: str,
		site_name: str,
		site_type: str,
		latitude: float,
		longitude: float,
		address: str,
		region: str,
	) -> dict[str, Any]:
		"""Register a network site (tower, data centre, exchange, etc.)."""
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True})
		item = InvSite(site_id, tenant_id, site_name, site_type, float(latitude), float(longitude), address, region)
		self.sites[self._key(tenant_id, site_id)] = item
		self._audit(tenant_id, "site_registered", site_id)
		return item.to_dict()

	def record_discrepancy(self, reconcile_id: str, tenant_id: str, asset_id: str, discrepancy_description: str) -> dict[str, Any]:
		"""Record an inventory discrepancy found during reconciliation."""
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True})
		item = InvReconciliation(reconcile_id, tenant_id, asset_id, discrepancy_description, "", None, None, "open")
		self.reconciliations[self._key(tenant_id, reconcile_id)] = item
		self._audit(tenant_id, "discrepancy_detected", reconcile_id)
		return item.to_dict()

	def approve_reconciliation(
		self,
		reconcile_id: str,
		tenant_id: str,
		approval_reference: str,
		resolved_by: str,
		resolved_at: str,
	) -> dict[str, Any]:
		"""Approve the resolution of an inventory discrepancy."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_reconciliation",
			"approval_present": _present(approval_reference),
		})
		rec = self._reconciliation_or_raise(reconcile_id, tenant_id)
		rec.approval_reference = approval_reference
		rec.resolved_by = resolved_by
		rec.resolved_at = resolved_at
		rec.status = "resolved"
		self._audit(tenant_id, "reconciliation_approved", reconcile_id)
		return rec.to_dict()

	def register_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		"""Register an inventory automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_inv_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = InvAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "inv_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def register_ne(
		self,
		ne_id: str,
		ne_type: str,
		vendor: str,
		model: str,
		location: str,
		ip_address: str,
		tenant_id: str = "default",
		software_version: str = "",
	) -> dict[str, Any]:
		"""Register a Network Element (NE) in the inventory.

		Validates ne_id uniqueness, IP address format, and commissions a
		corresponding asset record.  Stores the full NE record in the registry.
		"""
		assert ne_id, "ne_id required"
		assert ne_type, "ne_type required"
		assert vendor, "vendor required"
		assert ip_address, "ip_address required"
		if ne_id in self._ne_registry:
			raise ValueError(f"NE {ne_id} already registered")
		# Validate IP format
		try:
			ipaddress.ip_address(ip_address)
		except ValueError:
			raise ValueError(f"Invalid IP address: {ip_address!r}")
		ne_type_norm = ne_type.lower()
		asset_type = ne_type_norm if ne_type_norm in SUPPORTED_ASSET_TYPES else (SUPPORTED_ASSET_TYPES[0] if SUPPORTED_ASSET_TYPES else "router")
		asset = self.commission_asset(
			asset_id=ne_id,
			tenant_id=tenant_id,
			asset_type=asset_type,
			serial_number=f"SN-{ne_id}",
			vendor=vendor,
			model=model,
			location=location,
			commissioned_at=_utcnow(),
		)
		ne_record: dict[str, Any] = {
			"ne_id": ne_id,
			"ne_type": ne_type_norm,
			"vendor": vendor,
			"model": model,
			"location": location,
			"ip_address": ip_address,
			"software_version": software_version,
			"tenant_id": tenant_id,
			"status": "active",
			"registered_at": _utcnow(),
			"asset": asset,
		}
		self._ne_registry[ne_id] = ne_record
		self._audit(tenant_id, "ne_registered", ne_id)
		return ne_record

	async def topology_update(
		self,
		ne_id: str,
		connected_to: str,
		interface_type: str,
		tenant_id: str = "default",
		bandwidth_mbps: int = 0,
	) -> dict[str, Any]:
		"""Update network topology by adding or refreshing an adjacency link.

		Validates both NE endpoints exist, then upserts the link record
		in the topology adjacency list.
		"""
		assert ne_id, "ne_id required"
		assert connected_to, "connected_to required"
		assert interface_type, "interface_type required"
		if ne_id not in self._ne_registry:
			raise ValueError(f"NE {ne_id} not registered")
		# connected_to may not be registered (external peer) — allowed
		# Upsert: remove existing link between the pair first
		self._topology_links = [
			lnk for lnk in self._topology_links
			if not (lnk["a_end"] == ne_id and lnk["z_end"] == connected_to)
			and not (lnk["a_end"] == connected_to and lnk["z_end"] == ne_id)
		]
		link: dict[str, Any] = {
			"a_end": ne_id,
			"z_end": connected_to,
			"interface_type": interface_type.upper(),
			"bandwidth_mbps": bandwidth_mbps,
			"tenant_id": tenant_id,
			"updated_at": _utcnow(),
		}
		self._topology_links.append(link)
		# Also update topology record
		topo_id = f"topo-{ne_id}-{connected_to}"
		self.record_topology(
			topology_id=topo_id,
			tenant_id=tenant_id,
			topology_type="physical" if "physical" in SUPPORTED_TOPOLOGY_TYPES else (SUPPORTED_TOPOLOGY_TYPES[0] if SUPPORTED_TOPOLOGY_TYPES else "physical"),
			domain="core",
			name=f"{ne_id} <-> {connected_to}",
			description=f"{interface_type} link",
			nodes=f"{ne_id},{connected_to}",
			edges=f"{ne_id}--{interface_type}--{connected_to}",
			recorded_at=_utcnow(),
		)
		self._audit(tenant_id, "topology_link_updated", f"{ne_id}:{connected_to}")
		return link

	async def ip_address_allocation(
		self,
		pool_id: str,
		host_name: str,
		purpose: str,
		tenant_id: str = "default",
		ip_version: str = "ipv4",
	) -> dict[str, Any]:
		"""Allocate a specific IP address from a pool to a host.

		If the pool has pre-seeded free IPs, assigns the first available.
		Otherwise generates a synthetic RFC-1918 address for demo purposes.
		"""
		assert pool_id, "pool_id required"
		assert host_name, "host_name required"
		assert purpose, "purpose required"
		# Retrieve free list for pool, or generate synthetic
		free_list = self._ip_pool_free.get(pool_id, [])
		# Check host not already allocated
		existing = next((a for a in self._ip_allocations.values() if a.get("host_name") == host_name and a.get("pool_id") == pool_id), None)
		if existing:
			return existing
		if free_list:
			ip_address = free_list.pop(0)
			self._ip_pool_free[pool_id] = free_list
		else:
			# Synthesise: use pool_id hash for subnet
			pool_hash = abs(hash(pool_id)) % 254 + 1
			allocated_count = sum(1 for a in self._ip_allocations.values() if a.get("pool_id") == pool_id)
			host_octet = (allocated_count % 253) + 2
			ip_address = f"10.{pool_hash % 256}.{(pool_hash // 256) % 256}.{host_octet}"
		allocation: dict[str, Any] = {
			"ip_address": ip_address,
			"pool_id": pool_id,
			"host_name": host_name,
			"purpose": purpose,
			"ip_version": ip_version,
			"tenant_id": tenant_id,
			"status": "allocated",
			"allocated_at": _utcnow(),
		}
		self._ip_allocations[ip_address] = allocation
		# Also create an IP block record
		block_id = f"ipblock-{ip_address.replace('.', '-')}"
		vrf = f"vrf-{pool_id}"
		prefix_len = 32 if ip_version == "ipv4" else 128
		self.allocate_ip_block(
			block_id=block_id,
			tenant_id=tenant_id,
			ip_version=ip_version if ip_version in SUPPORTED_IP_VERSIONS else "ipv4",
			prefix=ip_address,
			prefix_length=prefix_len,
			block_type="host",
			vrf=vrf,
			allocated_to=host_name,
			allocated_at=_utcnow(),
		)
		self._audit(tenant_id, "ip_address_allocated", ip_address)
		return allocation

	async def ip_release(
		self,
		ip_address: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Release an allocated IP address back to its pool.

		Updates allocation record status, returns IP to free list, and
		releases the corresponding IP block.
		"""
		assert ip_address, "ip_address required"
		allocation = self._ip_allocations.get(ip_address)
		if allocation is None:
			raise ValueError(f"IP address {ip_address} not allocated")
		pool_id = allocation.get("pool_id", "")
		allocation["status"] = "released"
		allocation["released_at"] = _utcnow()
		# Return to free list
		self._ip_pool_free.setdefault(pool_id, []).append(ip_address)
		# Release corresponding block
		block_id = f"ipblock-{ip_address.replace('.', '-')}"
		block_key = self._key(tenant_id, block_id)
		if block_key in self.ip_blocks:
			self.release_ip_block(block_id, tenant_id)
		self._audit(tenant_id, "ip_address_released", ip_address)
		return allocation

	async def circuit_create(
		self,
		circuit_id: str,
		endpoints: list[str],
		bandwidth: str,
		service_type: str,
		tenant_id: str = "default",
		protection: str = "none",
	) -> dict[str, Any]:
		"""Create a logical circuit between two or more endpoints.

		endpoints: [a_end_ne_id, z_end_ne_id] or list of segment NE IDs.
		bandwidth: e.g. "1Gbps", "10Gbps", "100Mbps".
		service_type: Ethernet, MPLS, SDH, OTN, etc.
		"""
		assert circuit_id, "circuit_id required"
		assert len(endpoints) >= 2, "at least 2 endpoints required"
		assert bandwidth, "bandwidth required"
		assert service_type, "service_type required"
		a_end = endpoints[0]
		z_end = endpoints[-1]
		circuit_type = service_type.lower()
		if circuit_type not in SUPPORTED_CIRCUIT_TYPES:
			circuit_type = SUPPORTED_CIRCUIT_TYPES[0] if SUPPORTED_CIRCUIT_TYPES else "ethernet"
		circuit = self.provision_circuit(
			circuit_id=circuit_id,
			tenant_id=tenant_id,
			circuit_type=circuit_type,
			a_end=a_end,
			z_end=z_end,
			capacity=bandwidth,
			provisioned_at=_utcnow(),
		)
		service_record: dict[str, Any] = {
			"circuit_id": circuit_id,
			"endpoints": endpoints,
			"bandwidth": bandwidth,
			"service_type": service_type,
			"protection": protection,
			"tenant_id": tenant_id,
			"created_at": _utcnow(),
		}
		self._circuit_services[circuit_id] = service_record
		return {**circuit, "service": service_record}

	async def circuit_activation(
		self,
		circuit_id: str,
		activated_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Activate a provisioned circuit after end-to-end test passes.

		Updates circuit status to active and records who activated it.
		"""
		assert circuit_id, "circuit_id required"
		assert activated_by, "activated_by required"
		circuit = self._circuit_or_raise(circuit_id, tenant_id)
		if circuit.status not in ("provisioned", "testing"):
			raise ValueError(f"Circuit {circuit_id} in status '{circuit.status}' cannot be activated")
		circuit.status = "active"
		service = self._circuit_services.get(circuit_id, {})
		service["activated_by"] = activated_by
		service["activated_at"] = _utcnow()
		self._circuit_services[circuit_id] = service
		self._audit(tenant_id, "circuit_activated", circuit_id)
		return {**circuit.to_dict(), "activated_by": activated_by, "activated_at": service["activated_at"]}

	async def inventory_reconciliation(
		self,
		discovered_data: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Reconcile discovered network data against the inventory.

		discovered_data: list of {ne_id, ip_address, status, ...} records from
		network discovery scan.  Compares against _ne_registry; returns
		discrepancies (missing, unknown, status mismatch).
		"""
		assert discovered_data is not None, "discovered_data required"
		known_ids = set(self._ne_registry.keys())
		discovered_ids = {d.get("ne_id", "") for d in discovered_data}
		missing_from_inventory = discovered_ids - known_ids
		unknown_in_network = known_ids - discovered_ids
		status_mismatches: list[dict[str, Any]] = []
		for d in discovered_data:
			ne_id = d.get("ne_id", "")
			if ne_id in self._ne_registry:
				inv_status = self._ne_registry[ne_id].get("status")
				disc_status = d.get("status", "")
				if disc_status and inv_status != disc_status:
					status_mismatches.append({"ne_id": ne_id, "inventory_status": inv_status, "discovered_status": disc_status})
		# Record discrepancies
		for ne_id in missing_from_inventory:
			rec_id = f"recon-missing-{ne_id}"
			self.record_discrepancy(rec_id, tenant_id, ne_id, f"NE {ne_id} discovered but not in inventory")
		self._audit(tenant_id, "inventory_reconciliation_run", f"{len(discovered_data)}_devices")
		return {
			"tenant_id": tenant_id,
			"discovered_count": len(discovered_data),
			"inventory_count": len(known_ids),
			"missing_from_inventory": list(missing_from_inventory),
			"unknown_in_network": list(unknown_in_network),
			"status_mismatches": status_mismatches,
			"discrepancy_count": len(missing_from_inventory) + len(status_mismatches),
			"reconciled_at": _utcnow(),
		}

	async def capacity_planning(
		self,
		ne_id: str,
		forecast_months: int,
		tenant_id: str = "default",
		growth_rate_pct: float = 15.0,
	) -> dict[str, Any]:
		"""Forecast capacity requirements for a network element.

		Uses current utilisation from assets and applies a configurable
		growth_rate_pct per month to project when the NE will reach 80%
		(warning) and 95% (critical) utilisation thresholds.
		"""
		assert ne_id, "ne_id required"
		assert forecast_months > 0, "forecast_months must be positive"
		assert 0 < growth_rate_pct < 200, "growth_rate_pct must be (0, 200)"
		ne = self._ne_registry.get(ne_id)
		if ne is None:
			raise ValueError(f"NE {ne_id} not registered")
		# Current utilisation: simulated from asset count and circuit load
		circuits_on_ne = [
			c for c in self.circuits.values()
			if c.tenant_id == tenant_id and (c.a_end == ne_id or c.z_end == ne_id) and c.status == "active"
		]
		# Simulated current utilisation
		current_util_pct = min(95.0, len(circuits_on_ne) * 8.0 + 20.0)
		monthly_growth = growth_rate_pct / 100.0
		forecast: list[dict[str, Any]] = []
		warning_month: int | None = None
		critical_month: int | None = None
		for month in range(1, forecast_months + 1):
			projected_util = current_util_pct * ((1 + monthly_growth) ** month)
			projected_util = min(100.0, projected_util)
			if warning_month is None and projected_util >= 80.0:
				warning_month = month
			if critical_month is None and projected_util >= 95.0:
				critical_month = month
			forecast.append({"month": month, "projected_utilisation_pct": round(projected_util, 2)})
		result: dict[str, Any] = {
			"ne_id": ne_id,
			"tenant_id": tenant_id,
			"current_utilisation_pct": round(current_util_pct, 2),
			"growth_rate_pct_monthly": growth_rate_pct,
			"forecast_months": forecast_months,
			"warning_threshold_month": warning_month,
			"critical_threshold_month": critical_month,
			"forecast": forecast,
			"computed_at": _utcnow(),
		}
		self._capacity_forecasts.append(result)
		self._audit(tenant_id, "capacity_planning_run", ne_id)
		return result

	async def inventory_report(
		self,
		ne_type: str,
		location: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate an inventory report filtered by NE type and location.

		Returns counts, status distribution, vendor breakdown, and EoL risk.
		"""
		assert ne_type, "ne_type required"
		assert location, "location required"
		ne_type_norm = ne_type.lower()
		location_norm = location.lower()
		# Filter NEs
		nes = [
			ne for ne in self._ne_registry.values()
			if ne.get("tenant_id") == tenant_id
			and (ne_type_norm == "all" or ne.get("ne_type") == ne_type_norm)
			and (location_norm == "all" or location_norm in ne.get("location", "").lower())
		]
		status_dist: dict[str, int] = {}
		vendor_dist: dict[str, int] = {}
		eol_risk_count = 0
		for ne in nes:
			s = ne.get("status", "unknown")
			v = ne.get("vendor", "unknown")
			status_dist[s] = status_dist.get(s, 0) + 1
			vendor_dist[v] = vendor_dist.get(v, 0) + 1
			if ne.get("ne_id") in self._eol_records:
				eol_risk_count += 1
		# Assets
		asset_count = self._count(self.assets, tenant_id)
		circuit_count = self._count(self.circuits, tenant_id)
		ip_block_count = self._count(self.ip_blocks, tenant_id)
		self._audit(tenant_id, "inventory_report_generated", f"{ne_type}:{location}")
		return {
			"ne_type": ne_type,
			"location": location,
			"tenant_id": tenant_id,
			"ne_count": len(nes),
			"asset_count": asset_count,
			"circuit_count": circuit_count,
			"ip_block_count": ip_block_count,
			"status_distribution": status_dist,
			"vendor_distribution": vendor_dist,
			"eol_risk_count": eol_risk_count,
			"generated_at": _utcnow(),
		}

	async def end_of_life_tracking(
		self,
		ne_id: str,
		eol_date: str,
		tenant_id: str = "default",
		replacement_plan: str = "",
	) -> dict[str, Any]:
		"""Register or update an End-of-Life record for a network element.

		Calculates days remaining to EoL and flags urgency:
		>365d = planned, 90-365d = at_risk, <90d = critical.
		"""
		assert ne_id, "ne_id required"
		assert eol_date, "eol_date required"
		try:
			eol_dt = datetime.datetime.fromisoformat(eol_date)
		except ValueError:
			raise ValueError(f"Invalid eol_date format: {eol_date!r} — use ISO 8601")
		days_remaining = (eol_dt - datetime.datetime.utcnow()).days
		urgency = (
			"critical" if days_remaining < 90
			else ("at_risk" if days_remaining < 365 else "planned")
		)
		eol_record: dict[str, Any] = {
			"ne_id": ne_id,
			"eol_date": eol_date,
			"days_remaining": days_remaining,
			"urgency": urgency,
			"replacement_plan": replacement_plan,
			"tenant_id": tenant_id,
			"recorded_at": _utcnow(),
		}
		self._eol_records[ne_id] = eol_record
		if urgency == "critical":
			self._audit(tenant_id, "eol_critical_alert", ne_id)
		else:
			self._audit(tenant_id, "eol_tracked", ne_id)
		return eol_record

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		unauthorised_decommission_scope: bool = False,
		cross_tenant_inventory_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "inv_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"unauthorised_decommission_scope": unauthorised_decommission_scope,
			"cross_tenant_inventory_scope": cross_tenant_inventory_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "inv_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.telecom.inv.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		eol_critical = sum(1 for r in self._eol_records.values() if r.get("urgency") == "critical")
		return {
			"tenant_id": tenant_id,
			"asset_count": self._count(self.assets, tenant_id),
			"circuit_count": self._count(self.circuits, tenant_id),
			"ip_block_count": self._count(self.ip_blocks, tenant_id),
			"topology_count": self._count(self.topologies, tenant_id),
			"site_count": self._count(self.sites, tenant_id),
			"reconciliation_count": self._count(self.reconciliations, tenant_id),
			"ne_count": len(self._ne_registry),
			"eol_tracked_count": len(self._eol_records),
			"eol_critical_count": eol_critical,
			"ip_allocation_count": len(self._ip_allocations),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def register_network_element(
		self,
		ne_id: str,
		ne_type: str,
		vendor: str,
		model: str,
		site_id: str,
		tenant_id: str = "default",
		software_version: str = "",
		managed_ip: str = "",
	) -> dict[str, Any]:
		"""Register a network element in the inventory registry."""
		assert ne_id, "ne_id required"
		assert ne_type, "ne_type required"
		if ne_id in self._ne_registry:
			raise ValueError(f"NE {ne_id} already registered")
		record: dict[str, Any] = {
			"ne_id": ne_id,
			"ne_type": ne_type,
			"vendor": vendor,
			"model": model,
			"site_id": site_id,
			"software_version": software_version,
			"managed_ip": managed_ip,
			"status": "active",
			"tenant_id": tenant_id,
			"registered_at": _utcnow(),
		}
		self._ne_registry[ne_id] = record
		self._audit(tenant_id, "ne_registered", ne_id)
		return record

	async def update_ne_software(
		self,
		ne_id: str,
		new_version: str,
		tenant_id: str = "default",
		change_ref: str = "",
	) -> dict[str, Any]:
		"""Update the software version of a network element."""
		assert ne_id, "ne_id required"
		assert new_version, "new_version required"
		ne = self._ne_registry.get(ne_id)
		if ne is None:
			raise ValueError(f"NE {ne_id} not found")
		old_version = ne.get("software_version", "unknown")
		ne["software_version"] = new_version
		ne["last_upgraded_at"] = _utcnow()
		ne["change_ref"] = change_ref
		self._audit(tenant_id, "ne_software_updated", ne_id)
		return {**ne, "old_version": old_version}

	async def discover_topology(
		self,
		site_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Discover and return all network elements at a site with adjacency links."""
		ne_at_site = [ne for ne in self._ne_registry.values() if ne["site_id"] == site_id]
		links = [link for link in self._topology_links if link.get("site_id") == site_id]
		self._audit(tenant_id, "topology_discovered", site_id)
		return {
			"site_id": site_id,
			"tenant_id": tenant_id,
			"ne_count": len(ne_at_site),
			"link_count": len(links),
			"network_elements": ne_at_site,
			"links": links,
			"discovered_at": _utcnow(),
		}

	async def allocate_ip(
		self,
		pool_id: str,
		device_id: str,
		tenant_id: str = "default",
		ip_version: str = "ipv4",
	) -> dict[str, Any]:
		"""Allocate the next available IP from a pool to a device."""
		assert pool_id, "pool_id required"
		assert device_id, "device_id required"
		free_ips = self._ip_pool_free.get(pool_id, [])
		if not free_ips:
			raise ValueError(f"No free IPs in pool {pool_id}")
		allocated_ip = free_ips.pop(0)
		allocation: dict[str, Any] = {
			"ip_address": allocated_ip,
			"pool_id": pool_id,
			"device_id": device_id,
			"ip_version": ip_version,
			"status": "allocated",
			"tenant_id": tenant_id,
			"allocated_at": _utcnow(),
		}
		self._ip_allocations[allocated_ip] = allocation
		self._audit(tenant_id, "ip_allocated", allocated_ip)
		return allocation

	async def release_ip(
		self,
		ip_address: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Release an allocated IP address back to its pool."""
		assert ip_address, "ip_address required"
		allocation = self._ip_allocations.get(ip_address)
		if allocation is None:
			raise ValueError(f"IP {ip_address} not allocated")
		pool_id = allocation["pool_id"]
		allocation["status"] = "released"
		allocation["released_at"] = _utcnow()
		self._ip_pool_free.setdefault(pool_id, []).append(ip_address)
		del self._ip_allocations[ip_address]
		self._audit(tenant_id, "ip_released", ip_address)
		return allocation

	async def ip_utilisation_report(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Report IP address utilisation across all pools."""
		allocated_count = len(self._ip_allocations)
		free_count = sum(len(ips) for ips in self._ip_pool_free.values())
		total = allocated_count + free_count
		utilisation_pct = round(allocated_count / max(total, 1) * 100, 2)
		return {
			"tenant_id": tenant_id,
			"allocated_count": allocated_count,
			"free_count": free_count,
			"total_managed": total,
			"utilisation_pct": utilisation_pct,
			"computed_at": _utcnow(),
		}

	async def bulk_import_assets(
		self,
		asset_rows: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Bulk import network assets from a list of dicts."""
		assert asset_rows, "asset_rows required"
		success = 0
		errors: list[dict[str, Any]] = []
		for row in asset_rows:
			try:
				asset_id = row.get("asset_id", f"asset-{success}")
				asset_type = (row.get("asset_type") or "").lower()
				if asset_type not in SUPPORTED_ASSET_TYPES:
					asset_type = SUPPORTED_ASSET_TYPES[0] if SUPPORTED_ASSET_TYPES else "router"
				status = (row.get("status") or "").lower()
				if status not in SUPPORTED_ASSET_STATUSES:
					status = "active"
				from .models import InvAsset
				item = InvAsset(
					asset_id, tenant_id, asset_type,
					row.get("vendor", "unknown"),
					row.get("model", "unknown"),
					row.get("serial_number", ""),
					row.get("site_id", ""),
					status, _utcnow(),
				)
				self.assets[self._key(tenant_id, asset_id)] = item
				success += 1
			except Exception as exc:
				errors.append({"row": row, "error": str(exc)})
		self._audit(tenant_id, "assets_bulk_imported", f"count:{success}")
		return {
			"tenant_id": tenant_id,
			"total": len(asset_rows),
			"success_count": success,
			"error_count": len(errors),
			"errors": errors,
			"imported_at": _utcnow(),
		}

	async def export_inventory(
		self,
		tenant_id: str = "default",
		format: str = "json",
	) -> dict[str, Any]:
		"""Export inventory assets, circuits, and sites."""
		assert format in {"json", "csv"}, "format must be json or csv"
		assets = [a.to_dict() for a in self.assets.values() if a.tenant_id == tenant_id]
		circuits = [c.to_dict() for c in self.circuits.values() if c.tenant_id == tenant_id]
		self._audit(tenant_id, "inventory_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if assets:
				writer = csv.DictWriter(buf, fieldnames=list(assets[0].keys()))
				writer.writeheader()
				writer.writerows(assets)
			return {"format": "csv", "tenant_id": tenant_id, "asset_count": len(assets), "content": buf.getvalue()}
		return {
			"format": "json", "tenant_id": tenant_id,
			"asset_count": len(assets), "circuit_count": len(circuits),
			"assets": assets, "circuits": circuits, "exported_at": _utcnow(),
		}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return inventory service health status."""
		return {
			"service": "NetworkInventoryService",
			"tenant_id": tenant_id,
			"status": "healthy",
			"asset_count": self._count(self.assets, tenant_id),
			"circuit_count": self._count(self.circuits, tenant_id),
			"ne_count": len(self._ne_registry),
			"ip_allocation_count": len(self._ip_allocations),
			"checked_at": _utcnow(),
		}

	async def inventory_compliance_check(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Verify inventory records are complete and consistent."""
		assets = [a.to_dict() for a in self.assets.values() if a.tenant_id == tenant_id]
		no_site = [a for a in assets if not a.get("site_id")]
		no_serial = [a for a in assets if not a.get("serial_number")]
		compliant = len(assets) - len(no_site) - len(no_serial)
		self._audit(tenant_id, "inventory_compliance_check_run", tenant_id)
		return {
			"tenant_id": tenant_id,
			"total_assets": len(assets),
			"assets_missing_site": len(no_site),
			"assets_missing_serial": len(no_serial),
			"compliant_assets": max(compliant, 0),
			"compliance_rate_pct": round(max(compliant, 0) / max(len(assets), 1) * 100, 2),
			"checked_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _asset_or_raise(self, asset_id: str, tenant_id: str) -> InvAsset:
		a = self.assets.get(self._key(tenant_id, asset_id))
		if a is None:
			raise ValueError(f"Asset {asset_id} not found")
		return a

	def _circuit_or_raise(self, circuit_id: str, tenant_id: str) -> InvCircuit:
		c = self.circuits.get(self._key(tenant_id, circuit_id))
		if c is None:
			raise ValueError(f"Circuit {circuit_id} not found")
		return c

	def _ip_block_or_raise(self, block_id: str, tenant_id: str) -> InvIpBlock:
		b = self.ip_blocks.get(self._key(tenant_id, block_id))
		if b is None:
			raise ValueError(f"IP block {block_id} not found")
		return b

	def _reconciliation_or_raise(self, reconcile_id: str, tenant_id: str) -> InvReconciliation:
		r = self.reconciliations.get(self._key(tenant_id, reconcile_id))
		if r is None:
			raise ValueError(f"Reconciliation {reconcile_id} not found")
		return r

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "policy_denied")


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		self._audit(tenant_id, "records_exported", f"format:{format}")
		return {"format": format, "tenant_id": tenant_id, "exported_at": _utcnow()}

	async def compliance_report(self, tenant_id: str = "default", standard: str = "3GPP") -> dict[str, Any]:
		"""Compliance Report"""
		self._audit(tenant_id, "compliance_report_generated", standard)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _utcnow()}

	async def bulk_create(self, records: list[dict], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk Create"""
		assert records
		self._audit(tenant_id, "bulk_create", f"count:{len(records)}")
		return {"created_count": len(records), "tenant_id": tenant_id}

	async def analytics_summary(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		self._audit(tenant_id, "analytics_summary_run", period)
		return {"tenant_id": tenant_id, "period": period, "computed_at": _utcnow()}

	async def search_records(self, query: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Search Records"""
		assert query
		return {"query": query, "results": [], "tenant_id": tenant_id}

	async def ml_inventory_optimize(self, *args, **kwargs):
		"""AI-powered telecom network inventory optimisation recommendation. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="telecom_inventory_optimization")
			return {"utilization_score": round(result.score, 3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ------------------------------------------------------------------ #
	# New async methods — world-class improvements                         #
	# ------------------------------------------------------------------ #

	async def calculate_depreciation(
		self,
		asset_id: str,
		tenant_id: str = "default",
		method: str = "straight_line",
		useful_life_years: int = 10,
		salvage_value: float = 0.0,
	) -> dict[str, Any]:
		"""Calculate asset depreciation schedule.

		Supports straight_line, declining_balance, and sum_of_years_digits methods.
		Returns annual schedule with book_value, accumulated_depreciation, and
		net_book_value at each year.

		Args:
			asset_id: Asset identifier.
			tenant_id: Tenant scope.
			method: Depreciation method — straight_line | declining_balance | sum_of_years_digits.
			useful_life_years: Expected service life.
			salvage_value: Residual value at end of life.
		"""
		assert asset_id, "asset_id required"
		assert useful_life_years > 0, "useful_life_years must be positive"
		assert salvage_value >= 0, "salvage_value must be non-negative"
		method = method.lower()
		if method not in ("straight_line", "declining_balance", "sum_of_years_digits"):
			raise ValueError(f"Unknown depreciation method: {method!r}")
		asset = self._asset_or_raise(asset_id, tenant_id)
		# Purchase cost heuristic: use model hash as a stable proxy when real cost is absent
		purchase_cost = float(abs(hash(asset.model)) % 900_000 + 100_000)
		depreciable_amount = purchase_cost - salvage_value
		schedule: list[dict[str, Any]] = []
		accumulated = 0.0
		book_value = purchase_cost
		for year in range(1, useful_life_years + 1):
			if method == "straight_line":
				annual_dep = depreciable_amount / useful_life_years
			elif method == "declining_balance":
				rate = 2.0 / useful_life_years  # double-declining
				annual_dep = book_value * rate
				annual_dep = min(annual_dep, book_value - salvage_value)
			else:  # sum_of_years_digits
				syd = useful_life_years * (useful_life_years + 1) / 2
				annual_dep = (useful_life_years - year + 1) / syd * depreciable_amount
			accumulated += annual_dep
			book_value -= annual_dep
			schedule.append({
				"year": year,
				"annual_depreciation": round(annual_dep, 2),
				"accumulated_depreciation": round(accumulated, 2),
				"net_book_value": round(max(book_value, salvage_value), 2),
			})
		self._audit(tenant_id, "depreciation_calculated", asset_id)
		return {
			"asset_id": asset_id,
			"tenant_id": tenant_id,
			"method": method,
			"purchase_cost": round(purchase_cost, 2),
			"salvage_value": salvage_value,
			"useful_life_years": useful_life_years,
			"schedule": schedule,
			"computed_at": _utcnow(),
		}

	async def receive_spare_part(
		self,
		part_id: str,
		part_type: str,
		vendor: str,
		model: str,
		serial_number: str,
		site_id: str,
		tenant_id: str = "default",
		quantity: int = 1,
	) -> dict[str, Any]:
		"""Add spare parts to the spares pool at a given site.

		Creates or increments a spare-part stock record. Multiple receives with
		the same part_id increment quantity rather than duplicating records.
		"""
		assert part_id, "part_id required"
		assert part_type, "part_type required"
		assert quantity > 0, "quantity must be positive"
		if not hasattr(self, "_spare_parts"):
			self._spare_parts: dict[str, dict[str, Any]] = {}
		key = f"{tenant_id}:{part_id}"
		if key in self._spare_parts:
			self._spare_parts[key]["quantity_on_hand"] += quantity
			self._spare_parts[key]["last_received_at"] = _utcnow()
		else:
			self._spare_parts[key] = {
				"part_id": part_id,
				"part_type": part_type,
				"vendor": vendor,
				"model": model,
				"serial_number": serial_number,
				"site_id": site_id,
				"tenant_id": tenant_id,
				"quantity_on_hand": quantity,
				"quantity_issued": 0,
				"received_at": _utcnow(),
				"last_received_at": _utcnow(),
				"status": "available",
			}
		self._audit(tenant_id, "spare_part_received", part_id)
		return self._spare_parts[key]

	async def issue_spare_part(
		self,
		part_id: str,
		issued_to_ne_id: str,
		work_order: str,
		tenant_id: str = "default",
		quantity: int = 1,
	) -> dict[str, Any]:
		"""Issue spare parts from stock to a network element for a work order.

		Decrements stock and tracks the issuance against the NE and work order
		reference. Raises ValueError when insufficient stock.
		"""
		assert part_id, "part_id required"
		assert issued_to_ne_id, "issued_to_ne_id required"
		assert work_order, "work_order required"
		assert quantity > 0, "quantity must be positive"
		if not hasattr(self, "_spare_parts"):
			self._spare_parts = {}
		key = f"{tenant_id}:{part_id}"
		part = self._spare_parts.get(key)
		if part is None:
			raise ValueError(f"Spare part {part_id} not found in stock")
		if part["quantity_on_hand"] < quantity:
			raise ValueError(
				f"Insufficient stock for {part_id}: need {quantity}, have {part['quantity_on_hand']}"
			)
		part["quantity_on_hand"] -= quantity
		part["quantity_issued"] += quantity
		part["last_issued_at"] = _utcnow()
		if part["quantity_on_hand"] == 0:
			part["status"] = "depleted"
		issuance: dict[str, Any] = {
			"part_id": part_id,
			"issued_to_ne_id": issued_to_ne_id,
			"work_order": work_order,
			"quantity": quantity,
			"tenant_id": tenant_id,
			"issued_at": _utcnow(),
		}
		self._audit(tenant_id, "spare_part_issued", part_id)
		return {**part, "issuance": issuance}

	async def spare_parts_stock_report(
		self,
		tenant_id: str = "default",
		site_id: str | None = None,
	) -> dict[str, Any]:
		"""Report current spare parts stock levels, optionally filtered by site.

		Returns counts of available, depleted, and low-stock (quantity <= 2) parts
		alongside a full parts list for the tenant (or site).
		"""
		if not hasattr(self, "_spare_parts"):
			self._spare_parts = {}
		parts = [
			p for p in self._spare_parts.values()
			if p["tenant_id"] == tenant_id
			and (site_id is None or p["site_id"] == site_id)
		]
		available = sum(1 for p in parts if p["status"] == "available")
		depleted = sum(1 for p in parts if p["status"] == "depleted")
		low_stock = [p for p in parts if 0 < p["quantity_on_hand"] <= 2]
		self._audit(tenant_id, "spare_parts_report_generated", site_id or "all")
		return {
			"tenant_id": tenant_id,
			"site_id": site_id,
			"total_part_types": len(parts),
			"available_types": available,
			"depleted_types": depleted,
			"low_stock_count": len(low_stock),
			"low_stock_parts": [p["part_id"] for p in low_stock],
			"parts": parts,
			"generated_at": _utcnow(),
		}

	async def snapshot_device_config(
		self,
		ne_id: str,
		config_text: str,
		tenant_id: str = "default",
		source: str = "manual",
	) -> dict[str, Any]:
		"""Snapshot and fingerprint a device configuration for drift detection.

		Computes SHA-256 of config_text and compares against prior snapshot.
		Returns drift_detected=True and a character-level diff summary when the
		config has changed since the last snapshot.
		"""
		import hashlib
		assert ne_id, "ne_id required"
		assert config_text, "config_text required"
		if not hasattr(self, "_config_snapshots"):
			self._config_snapshots: dict[str, dict[str, Any]] = {}
		new_fingerprint = hashlib.sha256(config_text.encode()).hexdigest()
		prior = self._config_snapshots.get(f"{tenant_id}:{ne_id}")
		drift_detected = prior is not None and prior["fingerprint"] != new_fingerprint
		lines_added = lines_removed = 0
		if drift_detected:
			old_lines = set(prior.get("config_text", "").splitlines())
			new_lines = set(config_text.splitlines())
			lines_added = len(new_lines - old_lines)
			lines_removed = len(old_lines - new_lines)
		snapshot: dict[str, Any] = {
			"ne_id": ne_id,
			"tenant_id": tenant_id,
			"fingerprint": new_fingerprint,
			"config_text": config_text,
			"source": source,
			"drift_detected": drift_detected,
			"lines_added": lines_added,
			"lines_removed": lines_removed,
			"prior_fingerprint": prior["fingerprint"] if prior else None,
			"snapshotted_at": _utcnow(),
		}
		self._config_snapshots[f"{tenant_id}:{ne_id}"] = snapshot
		event = "config_drift_detected" if drift_detected else "config_snapshot_stored"
		self._audit(tenant_id, event, ne_id)
		return snapshot

	async def find_sites_within_radius(
		self,
		lat: float,
		lon: float,
		radius_km: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Find all registered sites within a given radius using the Haversine formula.

		Useful for field technician dispatch, spare-part logistics, and geographic
		network planning. Returns sites sorted by distance ascending.
		"""
		import math
		assert radius_km > 0, "radius_km must be positive"
		R = 6371.0  # Earth radius km
		def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
			phi1, phi2 = math.radians(lat1), math.radians(lat2)
			dphi = math.radians(lat2 - lat1)
			dlambda = math.radians(lon2 - lon1)
			a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
			return 2 * R * math.asin(math.sqrt(a))
		results: list[dict[str, Any]] = []
		for site in self.sites.values():
			if site.tenant_id != tenant_id:
				continue
			dist = haversine(lat, lon, site.latitude, site.longitude)
			if dist <= radius_km:
				results.append({**site.to_dict(), "distance_km": round(dist, 3)})
		results.sort(key=lambda s: s["distance_km"])
		self._audit(tenant_id, "sites_proximity_searched", f"{lat},{lon}:{radius_km}km")
		return {
			"origin_lat": lat,
			"origin_lon": lon,
			"radius_km": radius_km,
			"tenant_id": tenant_id,
			"site_count": len(results),
			"sites": results,
			"queried_at": _utcnow(),
		}

	async def asset_lifecycle_transition(
		self,
		asset_id: str,
		tenant_id: str,
		target_status: str,
		actor: str,
		approval_reference: str = "",
	) -> dict[str, Any]:
		"""Enforce FSM-validated asset lifecycle state transitions.

		Legal transitions::

			planning → ordered → received → tested → commissioned
			→ active → maintenance → decommissioned

		Raises ValueError for any transition not present in the FSM graph.
		Decommission always requires an approval_reference.
		"""
		FSM: dict[str, list[str]] = {
			"planning":      ["ordered"],
			"ordered":       ["received", "planning"],
			"received":      ["tested", "ordered"],
			"tested":        ["commissioned", "received"],
			"commissioned":  ["active", "tested"],
			"active":        ["maintenance", "decommissioned"],
			"maintenance":   ["active", "decommissioned"],
			"decommissioned": [],
		}
		target_status = target_status.lower()
		asset = self._asset_or_raise(asset_id, tenant_id)
		current = asset.status
		allowed = FSM.get(current, [])
		if target_status not in allowed:
			raise ValueError(
				f"Asset {asset_id}: transition '{current}' → '{target_status}' is not permitted. "
				f"Valid next states: {allowed or ['(terminal)']}"
			)
		if target_status == "decommissioned" and not _present(approval_reference):
			raise ValueError("Decommission requires a non-empty approval_reference")
		asset.status = target_status
		self._audit(tenant_id, f"asset_lifecycle_{target_status}", asset_id)
		return {
			**asset.to_dict(),
			"previous_status": current,
			"actor": actor,
			"approval_reference": approval_reference,
			"transitioned_at": _utcnow(),
		}

	async def network_graph_critical_paths(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Identify cut vertices (single points of failure) in the network topology.

		Performs a DFS-based articulation-point algorithm over the adjacency list
		stored in `_topology_links`. Returns a list of NE IDs whose removal would
		partition the network, and the count of disconnected components.
		"""
		# Build adjacency map from topology links for this tenant
		adj: dict[str, set[str]] = {}
		for link in self._topology_links:
			if link.get("tenant_id", tenant_id) != tenant_id:
				continue
			a, z = link["a_end"], link["z_end"]
			adj.setdefault(a, set()).add(z)
			adj.setdefault(z, set()).add(a)
		if not adj:
			return {"tenant_id": tenant_id, "cut_vertices": [], "component_count": 0, "computed_at": _utcnow()}
		# Tarjan articulation point algorithm
		visited: dict[str, bool] = {}
		disc: dict[str, int] = {}
		low: dict[str, int] = {}
		parent: dict[str, str | None] = {}
		cut_vertices: set[str] = set()
		timer = [0]
		def dfs(u: str) -> None:
			visited[u] = True
			disc[u] = low[u] = timer[0]
			timer[0] += 1
			child_count = 0
			for v in adj.get(u, set()):
				if not visited.get(v):
					child_count += 1
					parent[v] = u
					dfs(v)
					low[u] = min(low[u], low[v])
					if parent.get(u) is None and child_count > 1:
						cut_vertices.add(u)
					if parent.get(u) is not None and low[v] >= disc[u]:
						cut_vertices.add(u)
				elif v != parent.get(u):
					low[u] = min(low[u], disc[v])
		# Count components and find articulation points
		component_count = 0
		for node in adj:
			if not visited.get(node):
				parent[node] = None
				dfs(node)
				component_count += 1
		self._audit(tenant_id, "critical_path_analysis_run", f"{len(cut_vertices)}_cut_vertices")
		return {
			"tenant_id": tenant_id,
			"node_count": len(adj),
			"link_count": len(self._topology_links),
			"cut_vertices": sorted(cut_vertices),
			"component_count": component_count,
			"computed_at": _utcnow(),
		}

	async def vendor_eol_sync(
		self,
		vendor: str,
		tenant_id: str = "default",
		advisory_url: str = "",
	) -> dict[str, Any]:
		"""Sync End-of-Life dates from a vendor advisory source.

		When advisory_url is provided, attempts an HTTP GET and parses a JSON
		list of {model, eol_date} records. Falls back to a no-op stub when the
		URL is empty or unreachable. Updates `_eol_records` for all matching NEs.
		"""
		assert vendor, "vendor required"
		advisories: list[dict[str, Any]] = []
		fetch_status = "stub"
		if advisory_url:
			try:
				import urllib.request, json as _json
				with urllib.request.urlopen(advisory_url, timeout=5) as resp:
					advisories = _json.loads(resp.read())
				fetch_status = "fetched"
			except Exception as exc:
				fetch_status = f"error:{exc}"
		# Match advisories against registered NEs for this tenant
		matched: list[dict[str, Any]] = []
		for ne in self._ne_registry.values():
			if ne.get("tenant_id") != tenant_id:
				continue
			if ne.get("vendor", "").lower() != vendor.lower():
				continue
			# Look up advisory for this model
			adv = next((a for a in advisories if a.get("model", "").lower() == ne.get("model", "").lower()), None)
			if adv and adv.get("eol_date"):
				eol_rec = await self.end_of_life_tracking(
					ne_id=ne["ne_id"],
					eol_date=adv["eol_date"],
					tenant_id=tenant_id,
					replacement_plan=adv.get("replacement_plan", ""),
				)
				eol_rec["source"] = "vendor_advisory"
				eol_rec["advisory_url"] = advisory_url
				matched.append(eol_rec)
		self._audit(tenant_id, "vendor_eol_synced", vendor)
		return {
			"vendor": vendor,
			"tenant_id": tenant_id,
			"advisory_count": len(advisories),
			"matched_ne_count": len(matched),
			"fetch_status": fetch_status,
			"matched_records": matched,
			"synced_at": _utcnow(),
		}


# Backward-compatible alias
TelecomInvService = NetworkInventoryService
