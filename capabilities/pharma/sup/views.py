"""View models for APG Pharma Supply Chain screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import PharmaceuticalSupplyChainService


def dashboard_model(service: PharmaceuticalSupplyChainService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the supply chain dashboard view model."""
	contract = get_capability_contract(tenant_id)
	license_alerts = service.check_import_license_expiry(tenant_id)
	contract_alerts = service.check_contract_expiry(tenant_id)
	return {
		"title": "Pharmaceutical Supply Chain",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"license_expiry_alerts": license_alerts,
		"contract_expiry_alerts": contract_alerts,
	}


def supplier_registry_model(service: PharmaceuticalSupplyChainService, tenant_id: str = "default",
							qualified_only: bool = False) -> dict[str, Any]:
	"""Supplier registry view."""
	suppliers = service.list_suppliers(tenant_id, qualified_only=qualified_only)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Supplier Registry",
		"tenant_id": tenant_id,
		"qualified_only": qualified_only,
		"count": len(suppliers),
		"qualified_count": sum(1 for s in suppliers if s.qualification_status == "qualified"),
		"items": [s.model_dump() for s in suppliers],
		"supported_types": contract["configuration"]["suppliers"]["supported_types"],
	}


def supplier_detail_model(service: PharmaceuticalSupplyChainService, supplier_id: str,
						tenant_id: str = "default") -> dict[str, Any]:
	"""Supplier detail with orders and contracts."""
	supplier = service.get_supplier(supplier_id, tenant_id)
	orders = service.list_orders(tenant_id, supplier_id=supplier_id)
	contracts = service.list_contracts(tenant_id, supplier_id=supplier_id)
	return {
		"title": f"Supplier: {supplier.name}",
		"tenant_id": tenant_id,
		"supplier": supplier.model_dump(),
		"order_count": len(orders),
		"contract_count": len(contracts),
		"contracts": [c.model_dump() for c in contracts],
	}


def approved_supplier_list_model(service: PharmaceuticalSupplyChainService,
								tenant_id: str = "default") -> dict[str, Any]:
	"""Approved Supplier List view."""
	asl = service.list_suppliers(tenant_id, qualified_only=True)
	return {
		"title": "Approved Supplier List",
		"tenant_id": tenant_id,
		"count": len(asl),
		"items": [s.model_dump() for s in asl],
	}


def cmo_management_model(service: PharmaceuticalSupplyChainService, tenant_id: str = "default",
						active_only: bool = True) -> dict[str, Any]:
	"""CMO management view."""
	cmos = service.list_cmos(tenant_id, active_only=active_only)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "CMO Management",
		"tenant_id": tenant_id,
		"active_only": active_only,
		"count": len(cmos),
		"items": [c.model_dump() for c in cmos],
		"supported_types": contract["configuration"]["cmo"]["supported_types"],
	}


def demand_planning_model(service: PharmaceuticalSupplyChainService, tenant_id: str = "default",
						product_id: str | None = None) -> dict[str, Any]:
	"""Demand planning view."""
	forecasts = service.list_forecasts(tenant_id, product_id=product_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Demand Planning",
		"tenant_id": tenant_id,
		"product_id": product_id,
		"count": len(forecasts),
		"sop_approved_count": sum(1 for f in forecasts if f.sop_approved),
		"items": [f.model_dump() for f in forecasts],
		"supported_methods": contract["configuration"]["demand_planning"]["supported_methods"],
	}


def import_license_registry_model(service: PharmaceuticalSupplyChainService,
									tenant_id: str = "default") -> dict[str, Any]:
	"""Import license registry view."""
	licenses = service.list_import_licenses(tenant_id)
	alerts = service.check_import_license_expiry(tenant_id)
	return {
		"title": "Import Licenses",
		"tenant_id": tenant_id,
		"count": len(licenses),
		"active_count": sum(1 for l in licenses if l.status == "active"),
		"expiry_alerts": alerts,
		"items": [l.model_dump() for l in licenses],
	}


def supply_security_monitor_model(service: PharmaceuticalSupplyChainService,
									tenant_id: str = "default", at_risk_only: bool = False) -> dict[str, Any]:
	"""Supply security monitor view."""
	records = service.list_supply_security(tenant_id, at_risk_only=at_risk_only)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Supply Security Monitor",
		"tenant_id": tenant_id,
		"at_risk_only": at_risk_only,
		"count": len(records),
		"shortage_count": sum(1 for r in records if r.supply_status == "shortage"),
		"items": [r.model_dump() for r in records],
		"supported_risk_levels": contract["configuration"]["supply_security"]["supported_risk_levels"],
	}


def order_management_model(service: PharmaceuticalSupplyChainService, tenant_id: str = "default",
							supplier_id: str | None = None) -> dict[str, Any]:
	"""Purchase order management view."""
	orders = service.list_orders(tenant_id, supplier_id=supplier_id)
	return {
		"title": "Order Management",
		"tenant_id": tenant_id,
		"supplier_id": supplier_id,
		"count": len(orders),
		"open_count": sum(1 for o in orders if o.status == "placed"),
		"items": [o.model_dump() for o in orders],
	}
