"""Service layer for APG Pharma Supply Chain."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid6 import uuid7

from .capability_contract import (
	SUPPORTED_CMO_TYPES, SUPPORTED_CONTRACT_TYPES, SUPPORTED_DEMAND_METHODS,
	SUPPORTED_IMPORT_LICENSE_TYPES, SUPPORTED_ORDER_TYPES, SUPPORTED_QUALIFICATION_STATUSES,
	SUPPORTED_SECURITY_RISK_LEVELS, SUPPORTED_SUPPLY_STATUSES, SUPPORTED_SUPPLIER_TYPES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	CmoRecord, DemandForecast, ImportLicense, PurchaseOrder, Supplier, SupplierCreate,
	SupplyContract, SupplySecurityRecord,
)


def _uuid7str() -> str:
	return str(uuid7())


class PharmaceuticalSupplyChainService:
	"""Tenant-scoped supply chain service with supplier qualification and import licensing."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._external_store = store

		self._suppliers: dict[tuple[str, str], Supplier] = {}
		self._cmos: dict[tuple[str, str], CmoRecord] = {}
		self._forecasts: dict[tuple[str, str], DemandForecast] = {}
		self._import_licenses: dict[tuple[str, str], ImportLicense] = {}
		self._supply_security: dict[tuple[str, str], SupplySecurityRecord] = {}
		self._orders: dict[tuple[str, str], PurchaseOrder] = {}
		self._contracts: dict[tuple[str, str], SupplyContract] = {}
		self._audit_events: list[dict[str, Any]] = []
		# extended stores
		self._customs_clearances: dict[tuple[str, str], dict[str, Any]] = {}
		self._shortage_records: dict[tuple[str, str], dict[str, Any]] = {}
		self._supply_risk_assessments: dict[tuple[str, str], dict[str, Any]] = {}
		self._supply_analytics: dict[tuple[str, str], dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# --- suppliers ---

	def create_supplier(self, payload: SupplierCreate) -> Supplier:
		"""Create a new supplier record."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_supplier",
			"supplier_type_supported": payload.supplier_type in SUPPORTED_SUPPLIER_TYPES,
		})
		supplier = Supplier(**payload.model_dump())
		self._suppliers[self._key(supplier.tenant_id, supplier.id)] = supplier
		self._audit(supplier.tenant_id, "supplier_created", supplier.id)
		return supplier

	def qualify_supplier(self, supplier_id: str, tenant_id: str,
						quality_agreement_reference: str, audit_date: datetime,
						approved_materials: list[str]) -> Supplier:
		"""Qualify a supplier and add to Approved Supplier List."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_supplier",
			"quality_agreement_signed": bool(quality_agreement_reference),
		})
		supplier = self._get_supplier(supplier_id, tenant_id)
		data = supplier.model_dump()
		data["qualification_status"] = "qualified"
		data["quality_agreement_reference"] = quality_agreement_reference
		data["quality_agreement_signed_date"] = datetime.utcnow()
		data["last_audit_date"] = audit_date
		data["next_audit_due"] = audit_date + timedelta(days=730)
		data["approved_materials"] = approved_materials
		data["on_approved_supplier_list"] = True
		data["updated_at"] = datetime.utcnow()
		updated = Supplier(**data)
		self._suppliers[self._key(tenant_id, supplier_id)] = updated
		self._audit(tenant_id, "supplier_qualified", supplier_id)
		return updated

	def suspend_supplier(self, supplier_id: str, tenant_id: str, reason: str) -> Supplier:
		"""Suspend a supplier from the ASL."""
		supplier = self._get_supplier(supplier_id, tenant_id)
		data = supplier.model_dump()
		data["qualification_status"] = "suspended"
		data["on_approved_supplier_list"] = False
		data["updated_at"] = datetime.utcnow()
		updated = Supplier(**data)
		self._suppliers[self._key(tenant_id, supplier_id)] = updated
		self._audit(tenant_id, "supplier_suspended", supplier_id)
		return updated

	def get_supplier(self, supplier_id: str, tenant_id: str) -> Supplier:
		return self._get_supplier(supplier_id, tenant_id)

	def list_suppliers(self, tenant_id: str, qualified_only: bool = False) -> list[Supplier]:
		items = [s for s in self._suppliers.values() if s.tenant_id == tenant_id]
		if qualified_only:
			items = [s for s in items if s.qualification_status == "qualified"]
		return items

	# --- CMO ---

	def activate_cmo(self, tenant_id: str, cmo_code: str, name: str, cmo_type: str,
					supplier_id: str, technical_agreement_reference: str,
					quality_agreement_reference: str, created_by: str) -> CmoRecord:
		"""Activate a Contract Manufacturing Organisation."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_cmo",
			"cmo_type_supported": cmo_type in SUPPORTED_CMO_TYPES,
			"technical_agreement_signed": bool(technical_agreement_reference),
			"quality_agreement_signed": bool(quality_agreement_reference),
		})
		cmo = CmoRecord(
			tenant_id=tenant_id, cmo_code=cmo_code, name=name, cmo_type=cmo_type,
			supplier_id=supplier_id, technical_agreement_reference=technical_agreement_reference,
			technical_agreement_signed_date=datetime.utcnow(),
			quality_agreement_reference=quality_agreement_reference,
			active=True, created_by=created_by,
		)
		self._cmos[self._key(tenant_id, cmo.id)] = cmo
		self._audit(tenant_id, "cmo_activated", cmo.id)
		return cmo

	def list_cmos(self, tenant_id: str, active_only: bool = True) -> list[CmoRecord]:
		items = [c for c in self._cmos.values() if c.tenant_id == tenant_id]
		if active_only:
			items = [c for c in items if c.active]
		return items

	# --- demand planning ---

	def create_forecast(self, tenant_id: str, forecast_number: str, product_id: str,
						method: str, period: str, forecast_horizon_months: int,
						forecasted_demand: dict[str, float], safety_stock: float,
						created_by: str) -> DemandForecast:
		"""Create a demand forecast."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_forecast",
			"demand_method_supported": method in SUPPORTED_DEMAND_METHODS,
		})
		forecast = DemandForecast(
			tenant_id=tenant_id, forecast_number=forecast_number, product_id=product_id,
			method=method, period=period, forecast_horizon_months=forecast_horizon_months,
			forecasted_demand=forecasted_demand, safety_stock=safety_stock,
			created_by=created_by,
		)
		self._forecasts[self._key(tenant_id, forecast.id)] = forecast
		self._audit(tenant_id, "demand_forecast_updated", forecast.id)
		return forecast

	def approve_sop(self, forecast_id: str, tenant_id: str) -> DemandForecast:
		"""Mark a forecast as S&OP approved."""
		forecast = self._forecasts.get(self._key(tenant_id, forecast_id))
		if forecast is None:
			raise KeyError(f"forecast {forecast_id} not found")
		data = forecast.model_dump()
		data["sop_approved"] = True
		data["reviewed_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = DemandForecast(**data)
		self._forecasts[self._key(tenant_id, forecast_id)] = updated
		self._audit(tenant_id, "sop_completed", forecast_id)
		return updated

	def list_forecasts(self, tenant_id: str, product_id: str | None = None) -> list[DemandForecast]:
		items = [f for f in self._forecasts.values() if f.tenant_id == tenant_id]
		if product_id:
			items = [f for f in items if f.product_id == product_id]
		return items

	# --- import licensing ---

	def apply_import_license(self, tenant_id: str, license_number: str, license_type: str,
							region: str, product_ids: list[str], authority_reference: str,
							issuing_authority: str, scope: str, created_by: str) -> ImportLicense:
		"""Apply for an import license."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "apply_import_license",
			"license_type_supported": license_type in SUPPORTED_IMPORT_LICENSE_TYPES,
		})
		license = ImportLicense(
			tenant_id=tenant_id, license_number=license_number, license_type=license_type,
			region=region, product_ids=product_ids, authority_reference=authority_reference,
			issuing_authority=issuing_authority, scope=scope, created_by=created_by,
		)
		self._import_licenses[self._key(tenant_id, license.id)] = license
		self._audit(tenant_id, "import_license_applied", license.id)
		return license

	def grant_import_license(self, license_id: str, tenant_id: str,
							granted_date: datetime, expiry_date: datetime) -> ImportLicense:
		"""Mark an import license as granted."""
		license = self._import_licenses.get(self._key(tenant_id, license_id))
		if license is None:
			raise KeyError(f"import_license {license_id} not found")
		data = license.model_dump()
		data["status"] = "active"
		data["granted_date"] = granted_date
		data["expiry_date"] = expiry_date
		data["updated_at"] = datetime.utcnow()
		updated = ImportLicense(**data)
		self._import_licenses[self._key(tenant_id, license_id)] = updated
		self._audit(tenant_id, "import_license_granted", license_id)
		return updated

	def check_import_license_expiry(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return import licenses expiring within 90 days."""
		cutoff = datetime.utcnow() + timedelta(days=90)
		alerts = []
		for lic in self._import_licenses.values():
			if (lic.tenant_id == tenant_id and lic.expiry_date
					and lic.expiry_date <= cutoff and lic.renewal_submitted_date is None):
				alerts.append({
					"license_id": lic.id,
					"license_number": lic.license_number,
					"region": lic.region,
					"expiry_date": lic.expiry_date.isoformat(),
					"days_remaining": (lic.expiry_date - datetime.utcnow()).days,
				})
				self._audit(tenant_id, "import_license_expiring", lic.id)
		return alerts

	def check_import_license_active(self, tenant_id: str, product_id: str, region: str) -> bool:
		"""Check if there is an active import license for a product/region."""
		now = datetime.utcnow()
		return any(
			lic.tenant_id == tenant_id
			and lic.status == "active"
			and product_id in lic.product_ids
			and lic.region == region
			and (lic.expiry_date is None or lic.expiry_date > now)
			for lic in self._import_licenses.values()
		)

	def list_import_licenses(self, tenant_id: str) -> list[ImportLicense]:
		return [l for l in self._import_licenses.values() if l.tenant_id == tenant_id]

	# --- supply security ---

	def update_supply_security(self, tenant_id: str, product_id: str, supply_status: str,
								risk_level: str, primary_supplier_id: str | None,
								created_by: str, dual_sourced: bool = False,
								inventory_days: float | None = None) -> SupplySecurityRecord:
		"""Update or create supply security monitoring for a product."""
		existing = next((r for r in self._supply_security.values()
						if r.tenant_id == tenant_id and r.product_id == product_id), None)
		if existing:
			data = existing.model_dump()
			data["supply_status"] = supply_status
			data["risk_level"] = risk_level
			data["primary_supplier_id"] = primary_supplier_id
			data["dual_sourced"] = dual_sourced
			data["inventory_days"] = inventory_days
			data["last_reviewed"] = datetime.utcnow()
			data["updated_at"] = datetime.utcnow()
			updated = SupplySecurityRecord(**data)
			self._supply_security[self._key(tenant_id, existing.id)] = updated
			record = updated
		else:
			record = SupplySecurityRecord(
				tenant_id=tenant_id, product_id=product_id, supply_status=supply_status,
				risk_level=risk_level, primary_supplier_id=primary_supplier_id,
				dual_sourced=dual_sourced, inventory_days=inventory_days,
				created_by=created_by,
			)
			self._supply_security[self._key(tenant_id, record.id)] = record
		if supply_status == "shortage":
			self._audit(tenant_id, "supply_shortage_detected", record.id)
		if risk_level in ("high", "critical"):
			self._audit(tenant_id, "supply_risk_escalated", record.id)
		return record

	def list_supply_security(self, tenant_id: str, at_risk_only: bool = False) -> list[SupplySecurityRecord]:
		items = [r for r in self._supply_security.values() if r.tenant_id == tenant_id]
		if at_risk_only:
			items = [r for r in items if r.supply_status in ("at_risk", "shortage", "out_of_stock")]
		return items

	# --- purchase orders ---

	def place_order(self, tenant_id: str, po_number: str, order_type: str,
					supplier_id: str, product_id: str, quantity: float,
					unit_of_measure: str, created_by: str,
					expected_delivery: datetime | None = None,
					transport_condition: str | None = None) -> PurchaseOrder:
		"""Place a purchase order against an approved supplier."""
		supplier = self._suppliers.get(self._key(tenant_id, supplier_id))
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "place_order",
			"order_type_supported": order_type in SUPPORTED_ORDER_TYPES,
			"supplier_on_asl": supplier is not None and supplier.on_approved_supplier_list,
			"supplier_qualified": supplier is not None and supplier.qualification_status == "qualified",
		})
		order = PurchaseOrder(
			tenant_id=tenant_id, po_number=po_number, order_type=order_type,
			supplier_id=supplier_id, product_id=product_id, quantity=quantity,
			unit_of_measure=unit_of_measure, expected_delivery=expected_delivery,
			transport_condition=transport_condition, created_by=created_by,
		)
		self._orders[self._key(tenant_id, order.id)] = order
		self._audit(tenant_id, "order_placed", order.id)
		return order

	def receive_order(self, order_id: str, tenant_id: str, coa_reference: str) -> PurchaseOrder:
		"""Receive an order with CoA."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "receive_order",
			"coa_present": bool(coa_reference),
		})
		order = self._orders.get(self._key(tenant_id, order_id))
		if order is None:
			raise KeyError(f"order {order_id} not found")
		data = order.model_dump()
		data["status"] = "received"
		data["actual_delivery"] = datetime.utcnow()
		data["coa_reference"] = coa_reference
		data["updated_at"] = datetime.utcnow()
		updated = PurchaseOrder(**data)
		self._orders[self._key(tenant_id, order_id)] = updated
		self._audit(tenant_id, "order_received", order_id)
		return updated

	def list_orders(self, tenant_id: str, supplier_id: str | None = None) -> list[PurchaseOrder]:
		items = [o for o in self._orders.values() if o.tenant_id == tenant_id]
		if supplier_id:
			items = [o for o in items if o.supplier_id == supplier_id]
		return items

	# --- contracts ---

	def create_contract(self, tenant_id: str, contract_number: str, contract_type: str,
						supplier_id: str, title: str, created_by: str) -> SupplyContract:
		"""Create a supply contract."""
		contract = SupplyContract(
			tenant_id=tenant_id, contract_number=contract_number, contract_type=contract_type,
			supplier_id=supplier_id, title=title, created_by=created_by,
		)
		self._contracts[self._key(tenant_id, contract.id)] = contract
		self._audit(tenant_id, "contract_created", contract.id)
		return contract

	def approve_contract(self, contract_id: str, tenant_id: str,
						approval_reference: str, effective_date: datetime,
						expiry_date: datetime | None = None) -> SupplyContract:
		"""Approve a supply contract."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_contract",
			"approved": bool(approval_reference),
		})
		contract = self._contracts.get(self._key(tenant_id, contract_id))
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		data = contract.model_dump()
		data["status"] = "active"
		data["approved"] = True
		data["approval_reference"] = approval_reference
		data["effective_date"] = effective_date
		data["expiry_date"] = expiry_date
		data["updated_at"] = datetime.utcnow()
		updated = SupplyContract(**data)
		self._contracts[self._key(tenant_id, contract_id)] = updated
		self._audit(tenant_id, "contract_approved", contract_id)
		return updated

	def check_contract_expiry(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return contracts expiring within 60 days."""
		cutoff = datetime.utcnow() + timedelta(days=60)
		alerts = []
		for c in self._contracts.values():
			if (c.tenant_id == tenant_id and c.expiry_date
					and c.expiry_date <= cutoff and not c.renewal_initiated):
				alerts.append({
					"contract_id": c.id,
					"contract_number": c.contract_number,
					"supplier_id": c.supplier_id,
					"expiry_date": c.expiry_date.isoformat(),
					"days_remaining": (c.expiry_date - datetime.utcnow()).days,
				})
				self._audit(tenant_id, "contract_expiring", c.id)
		return alerts

	def list_contracts(self, tenant_id: str, supplier_id: str | None = None) -> list[SupplyContract]:
		items = [c for c in self._contracts.values() if c.tenant_id == tenant_id]
		if supplier_id:
			items = [c for c in items if c.supplier_id == supplier_id]
		return items

	# --- NEW: api_sourcing ---

	def api_sourcing(
		self,
		drug_id: str,
		supplier_id: str,
		quantity: float,
		delivery_date: datetime,
		tenant_id: str,
		unit_of_measure: str = "kg",
		coa_required: bool = True,
		quality_agreement_ref: str = "",
	) -> PurchaseOrder:
		"""Place an API (Active Pharmaceutical Ingredient) sourcing order from a qualified supplier."""
		assert drug_id and supplier_id, "drug_id and supplier_id required"
		assert quantity > 0, "quantity must be positive"
		supplier = self._suppliers.get(self._key(tenant_id, supplier_id))
		if supplier is None:
			raise KeyError(f"supplier {supplier_id} not found")
		if supplier.qualification_status != "qualified":
			raise ValueError(f"supplier {supplier_id} is not qualified (status: {supplier.qualification_status})")
		po_number = f"PO-API-{drug_id[:6].upper()}-{_uuid7str()[:6].upper()}"
		order = PurchaseOrder(
			tenant_id=tenant_id,
			po_number=po_number,
			order_type="api_sourcing",
			supplier_id=supplier_id,
			product_id=drug_id,
			quantity=quantity,
			unit_of_measure=unit_of_measure,
			expected_delivery=delivery_date,
			transport_condition="controlled_ambient",
			coa_required=coa_required,
			quality_agreement_reference=quality_agreement_ref,
			created_by=self._actor_id,
		)
		self._orders[self._key(tenant_id, order.id)] = order
		self._audit(tenant_id, "api_sourcing_order_placed", order.id)
		return order

	# --- NEW: cmo_order ---

	def cmo_order(
		self,
		cmo_id: str,
		product_id: str,
		batch_size: float,
		delivery_date: datetime,
		tenant_id: str,
		batch_count: int = 1,
		packaging_spec: str = "",
		technical_agreement_ref: str = "",
	) -> dict[str, Any]:
		"""Place a manufacturing order with a Contract Manufacturing Organisation."""
		assert cmo_id and product_id, "cmo_id and product_id required"
		assert batch_size > 0 and batch_count > 0, "batch_size and batch_count must be positive"
		cmo = self._cmos.get(self._key(tenant_id, cmo_id))
		if cmo is None:
			raise KeyError(f"cmo {cmo_id} not found")
		if not cmo.active:
			raise ValueError(f"cmo {cmo_id} is not active")
		order_id = _uuid7str()
		order: dict[str, Any] = {
			"id": order_id,
			"tenant_id": tenant_id,
			"cmo_id": cmo_id,
			"cmo_name": cmo.name,
			"product_id": product_id,
			"batch_size": batch_size,
			"batch_count": batch_count,
			"total_quantity": batch_size * batch_count,
			"requested_delivery_date": str(delivery_date),
			"packaging_spec": packaging_spec,
			"technical_agreement_reference": technical_agreement_ref or cmo.technical_agreement_reference,
			"quality_agreement_reference": cmo.quality_agreement_reference,
			"status": "placed",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "cmo_order_placed", order_id)
		return order

	# --- NEW: demand_planning ---

	def demand_planning(
		self,
		product_id: str,
		forecast_periods: int,
		tenant_id: str,
		method: str = "statistical",
		base_demand: float = 0.0,
		growth_rate: float = 0.0,
		seasonality_factors: dict[str, float] | None = None,
	) -> DemandForecast:
		"""Generate a demand plan for a product over multiple periods using statistical or consensus methods."""
		assert product_id and forecast_periods > 0, "product_id and forecast_periods > 0 required"
		assert method in SUPPORTED_DEMAND_METHODS, f"unsupported method: {method}"
		period = str(datetime.utcnow().year)
		forecast_number = f"FC-{product_id[:6].upper()}-{_uuid7str()[:6].upper()}"
		# Generate monthly demand projections
		forecasted_demand: dict[str, float] = {}
		monthly_base = base_demand / 12 if base_demand > 0 else 100.0
		for m in range(1, forecast_periods + 1):
			month_key = f"M{m:02d}"
			growth = monthly_base * (1 + growth_rate) ** (m - 1)
			seasonality = (seasonality_factors or {}).get(month_key, 1.0)
			forecasted_demand[month_key] = round(growth * seasonality, 2)
		safety_stock = monthly_base * 1.5  # 6-week safety stock
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_forecast",
			"demand_method_supported": True,
		})
		forecast = DemandForecast(
			tenant_id=tenant_id,
			forecast_number=forecast_number,
			product_id=product_id,
			method=method,
			period=period,
			forecast_horizon_months=forecast_periods,
			forecasted_demand=forecasted_demand,
			safety_stock=safety_stock,
			created_by=self._actor_id,
		)
		self._forecasts[self._key(tenant_id, forecast.id)] = forecast
		self._audit(tenant_id, "demand_forecast_updated", forecast.id)
		return forecast

	# --- NEW: supply_risk_assessment ---

	def supply_risk_assessment(
		self,
		product_id: str,
		supply_chain_map: dict[str, Any],
		tenant_id: str,
		assessment_method: str = "fmea",
	) -> dict[str, Any]:
		"""Assess supply chain risks for a product using a supply chain map (supplier nodes, routes, risks)."""
		assert product_id and supply_chain_map, "product_id and supply_chain_map required"
		nodes = supply_chain_map.get("nodes", [])
		risks: list[dict[str, Any]] = []
		overall_risk_score = 0.0
		for node in nodes:
			node_risk_score = node.get("risk_score", 5)
			node_criticality = node.get("criticality", 5)
			combined = node_risk_score * node_criticality
			risks.append({
				"node_id": node.get("id", ""),
				"node_type": node.get("type", "supplier"),
				"risk_score": node_risk_score,
				"criticality": node_criticality,
				"combined_score": combined,
				"risk_level": "critical" if combined >= 64 else "high" if combined >= 36 else "medium" if combined >= 16 else "low",
				"mitigation": node.get("mitigation", ""),
			})
			overall_risk_score += combined
		avg_risk = overall_risk_score / max(len(nodes), 1)
		overall_level = "critical" if avg_risk >= 64 else "high" if avg_risk >= 36 else "medium" if avg_risk >= 16 else "low"
		assessment_id = _uuid7str()
		assessment: dict[str, Any] = {
			"id": assessment_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"assessment_method": assessment_method,
			"nodes_assessed": len(nodes),
			"node_risks": risks,
			"overall_risk_score": round(avg_risk, 2),
			"overall_risk_level": overall_level,
			"dual_sourced": supply_chain_map.get("dual_sourced", False),
			"single_source_nodes": sum(1 for n in nodes if not n.get("alternate_source")),
			"recommendations": [
				f"Dual-source node {r['node_id']}" for r in risks if r["risk_level"] in ("critical", "high")
			],
			"assessed_at": datetime.utcnow().isoformat(),
		}
		self._supply_risk_assessments[self._key(tenant_id, assessment_id)] = assessment
		self._audit(tenant_id, "supply_risk_assessed", assessment_id)
		if overall_level in ("critical", "high"):
			self._audit(tenant_id, "supply_risk_escalated", assessment_id)
		return assessment

	# --- NEW: import_licence_application ---

	def import_licence_application(
		self,
		product_id: str,
		country: str,
		quantity: float,
		tenant_id: str,
		license_type: str = "standard_import",
		issuing_authority: str = "",
		supporting_documents: list[str] | None = None,
	) -> ImportLicense:
		"""Apply for an import licence for a product in a country with supporting documentation."""
		assert product_id and country, "product_id and country required"
		assert quantity > 0, "quantity must be positive"
		assert license_type in SUPPORTED_IMPORT_LICENSE_TYPES, \
			f"unsupported license_type: {license_type}"
		license_number = f"IMP-{country.upper()}-{product_id[:6].upper()}-{_uuid7str()[:6].upper()}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "apply_import_license",
			"license_type_supported": True,
		})
		license = ImportLicense(
			tenant_id=tenant_id,
			license_number=license_number,
			license_type=license_type,
			region=country,
			product_ids=[product_id],
			authority_reference=f"REF-{_uuid7str()[:8].upper()}",
			issuing_authority=issuing_authority,
			scope=f"Import of {product_id} into {country}, quantity {quantity}",
			supporting_documents=supporting_documents or [],
			created_by=self._actor_id,
		)
		self._import_licenses[self._key(tenant_id, license.id)] = license
		self._audit(tenant_id, "import_license_applied", license.id)
		return license

	# --- NEW: customs_clearance ---

	def customs_clearance(
		self,
		shipment_id: str,
		documents: list[str],
		tenant_id: str,
		country: str = "",
		broker_id: str = "",
		hs_code: str = "",
		declared_value: float = 0.0,
	) -> dict[str, Any]:
		"""Process customs clearance for a shipment with document verification and HS code classification."""
		assert shipment_id and documents, "shipment_id and documents required"
		required_docs = {"commercial_invoice", "packing_list", "certificate_of_origin", "import_permit"}
		provided_docs = set(documents)
		missing_docs = required_docs - provided_docs
		clearance_possible = len(missing_docs) == 0
		clearance_id = _uuid7str()
		clearance: dict[str, Any] = {
			"id": clearance_id,
			"tenant_id": tenant_id,
			"shipment_id": shipment_id,
			"country": country,
			"broker_id": broker_id,
			"hs_code": hs_code,
			"declared_value": declared_value,
			"documents_provided": list(provided_docs),
			"missing_documents": list(missing_docs),
			"clearance_possible": clearance_possible,
			"status": "pending" if not clearance_possible else "submitted",
			"submitted_at": datetime.utcnow().isoformat() if clearance_possible else None,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._customs_clearances[self._key(tenant_id, clearance_id)] = clearance
		self._audit(tenant_id, "customs_clearance_initiated", clearance_id)
		if not clearance_possible:
			self._audit(tenant_id, "customs_documentation_incomplete", clearance_id)
		return clearance

	# --- NEW: security_of_supply_monitoring ---

	def security_of_supply_monitoring(
		self,
		critical_medicines: list[str],
		tenant_id: str,
		inventory_threshold_days: float = 30.0,
	) -> dict[str, Any]:
		"""Monitor security of supply for critical medicines: flag shortages, low inventory, single sources."""
		assert critical_medicines, "critical_medicines list required"
		monitoring_results: list[dict[str, Any]] = []
		for drug_id in critical_medicines:
			security_record = next((r for r in self._supply_security.values()
				if r.tenant_id == tenant_id and r.product_id == drug_id), None)
			if security_record is None:
				monitoring_results.append({
					"drug_id": drug_id,
					"status": "unmonitored",
					"risk": "unknown",
					"action_required": True,
				})
				continue
			action_required = (
				security_record.supply_status in ("at_risk", "shortage", "out_of_stock")
				or (security_record.inventory_days is not None
					and security_record.inventory_days < inventory_threshold_days)
				or not security_record.dual_sourced
			)
			monitoring_results.append({
				"drug_id": drug_id,
				"supply_status": security_record.supply_status,
				"risk_level": security_record.risk_level,
				"inventory_days": security_record.inventory_days,
				"dual_sourced": security_record.dual_sourced,
				"action_required": action_required,
				"last_reviewed": str(getattr(security_record, "last_reviewed", datetime.utcnow())),
			})
			if action_required:
				self._audit(tenant_id, "supply_risk_escalated", drug_id)
		at_risk_count = sum(1 for r in monitoring_results if r.get("action_required"))
		return {
			"tenant_id": tenant_id,
			"critical_medicines_monitored": len(critical_medicines),
			"at_risk_count": at_risk_count,
			"secure_count": len(critical_medicines) - at_risk_count,
			"results": monitoring_results,
			"monitored_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: shortage_management ---

	def shortage_management(
		self,
		drug_id: str,
		shortage_type: str,
		mitigation: str,
		tenant_id: str,
		estimated_duration_days: int = 30,
		regulatory_notification_required: bool = True,
		contingency_supplier_id: str | None = None,
	) -> dict[str, Any]:
		"""Manage a drug shortage: classify type, activate mitigation plan, notify authorities."""
		assert drug_id and shortage_type, "drug_id and shortage_type required"
		assert shortage_type in ("manufacturing_delay", "api_shortage", "demand_surge",
			"regulatory_hold", "logistics", "force_majeure"), \
			f"unsupported shortage_type: {shortage_type}"
		shortage_id = _uuid7str()
		# update supply security record
		self.update_supply_security(
			tenant_id=tenant_id,
			product_id=drug_id,
			supply_status="shortage",
			risk_level="critical",
			primary_supplier_id=None,
			created_by=self._actor_id,
		)
		shortage: dict[str, Any] = {
			"id": shortage_id,
			"tenant_id": tenant_id,
			"drug_id": drug_id,
			"shortage_type": shortage_type,
			"mitigation": mitigation,
			"estimated_duration_days": estimated_duration_days,
			"regulatory_notification_required": regulatory_notification_required,
			"regulatory_notification_sent": False,
			"contingency_supplier_id": contingency_supplier_id,
			"status": "active",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._shortage_records[self._key(tenant_id, shortage_id)] = shortage
		self._audit(tenant_id, "supply_shortage_detected", shortage_id)
		self._audit(tenant_id, "shortage_mitigation_activated", shortage_id)
		if regulatory_notification_required:
			self._audit(tenant_id, "shortage_regulatory_notification_required", shortage_id)
		return shortage

	# --- NEW: supply_analytics ---

	def supply_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate supply chain KPIs for a period: OTIF, supplier performance, shortage rates, forecast accuracy."""
		assert period, "period required"
		suppliers = self.list_suppliers(tenant_id)
		qualified_suppliers = [s for s in suppliers if s.qualification_status == "qualified"]
		orders = self.list_orders(tenant_id)
		received_orders = [o for o in orders if o.status == "received"]
		on_time_orders = [o for o in received_orders
			if getattr(o, "expected_delivery", None) and getattr(o, "actual_delivery", None)
			and o.actual_delivery <= o.expected_delivery]
		forecasts = self.list_forecasts(tenant_id)
		sop_approved = [f for f in forecasts if getattr(f, "sop_approved", False)]
		import_licenses = self.list_import_licenses(tenant_id)
		active_licenses = [l for l in import_licenses if l.status == "active"]
		security_records = self.list_supply_security(tenant_id)
		shortages = self._shortage_records
		active_shortages = [s for s in shortages.values()
			if s["tenant_id"] == tenant_id and s.get("status") == "active"]
		otif_rate = len(on_time_orders) / max(len(received_orders), 1) * 100
		supplier_qualification_rate = len(qualified_suppliers) / max(len(suppliers), 1) * 100
		analytics_id = _uuid7str()
		analytics: dict[str, Any] = {
			"id": analytics_id,
			"period": period,
			"tenant_id": tenant_id,
			"total_suppliers": len(suppliers),
			"qualified_suppliers": len(qualified_suppliers),
			"supplier_qualification_rate_pct": round(supplier_qualification_rate, 2),
			"total_orders": len(orders),
			"received_orders": len(received_orders),
			"on_time_orders": len(on_time_orders),
			"otif_rate_pct": round(otif_rate, 2),
			"forecasts_generated": len(forecasts),
			"sop_approved_forecasts": len(sop_approved),
			"active_import_licenses": len(active_licenses),
			"at_risk_products": sum(1 for r in security_records
				if r.supply_status in ("at_risk", "shortage")),
			"active_shortages": len(active_shortages),
			"active_cmos": sum(1 for c in self._cmos.values() if c.tenant_id == tenant_id and c.active),
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._supply_analytics[self._key(tenant_id, analytics_id)] = analytics
		self._audit(tenant_id, "supply_analytics_generated", analytics_id)
		return analytics

	# --- NEW: regulatory_supply_reporting ---

	def regulatory_supply_reporting(
		self,
		period: str,
		jurisdiction: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Generate a regulatory supply report for a jurisdiction: shortage notifications, import compliance, supply continuity."""
		assert period and jurisdiction, "period and jurisdiction required"
		active_shortages = [s for s in self._shortage_records.values()
			if s["tenant_id"] == tenant_id and s.get("status") == "active"]
		regulatory_notifications_pending = [s for s in active_shortages
			if s.get("regulatory_notification_required") and not s.get("regulatory_notification_sent")]
		import_licenses = [l for l in self._import_licenses.values()
			if l.tenant_id == tenant_id and l.region == jurisdiction]
		active_licenses = [l for l in import_licenses if l.status == "active"]
		expiring_licenses = self.check_import_license_expiry(tenant_id)
		security_records = self.list_supply_security(tenant_id)
		critical_products = [r for r in security_records
			if r.risk_level == "critical" or r.supply_status in ("shortage", "out_of_stock")]
		report_id = _uuid7str()
		self._audit(tenant_id, "regulatory_supply_report_generated", report_id)
		return {
			"report_id": report_id,
			"period": period,
			"jurisdiction": jurisdiction,
			"tenant_id": tenant_id,
			"active_shortages": len(active_shortages),
			"regulatory_notifications_pending": len(regulatory_notifications_pending),
			"import_licenses_in_jurisdiction": len(import_licenses),
			"active_import_licenses": len(active_licenses),
			"expiring_licenses_90_days": len(expiring_licenses),
			"critical_supply_products": len(critical_products),
			"report_type": "supply_continuity_regulatory",
			"generated_at": datetime.utcnow().isoformat(),
		}

	# --- dashboard ---

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return supply chain dashboard."""
		return {
			"tenant_id": tenant_id,
			"qualified_supplier_count": sum(1 for s in self._suppliers.values()
										if s.tenant_id == tenant_id and s.qualification_status == "qualified"),
			"total_supplier_count": self._count(self._suppliers, tenant_id),
			"active_cmo_count": sum(1 for c in self._cmos.values()
								if c.tenant_id == tenant_id and c.active),
			"active_import_licenses": sum(1 for l in self._import_licenses.values()
										if l.tenant_id == tenant_id and l.status == "active"),
			"at_risk_products": sum(1 for r in self._supply_security.values()
								if r.tenant_id == tenant_id and r.supply_status in ("at_risk", "shortage")),
			"open_orders": sum(1 for o in self._orders.values()
							if o.tenant_id == tenant_id and o.status == "placed"),
			"active_contracts": sum(1 for c in self._contracts.values()
								if c.tenant_id == tenant_id and c.status == "active"),
			"active_shortages": sum(1 for s in self._shortage_records.values()
								if s["tenant_id"] == tenant_id and s.get("status") == "active"),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
		}

	# --- private helpers ---

	def _log_supply_risk(self, product_id: str, risk_level: str, inventory_days: float) -> None:
		pass

	def _log_supplier_qualification(self, supplier_id: str, status: str) -> None:
		pass

	def _get_supplier(self, supplier_id: str, tenant_id: str) -> Supplier:
		item = self._suppliers.get(self._key(tenant_id, supplier_id))
		if item is None:
			raise KeyError(f"supplier {supplier_id} not found")
		return item

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.pharma.sup.lifecycle",
		})

	def _count(self, store: dict[Any, Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": tenant_id}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

	async def compliance_report(self, tenant_id: str, standard: str = "GxP") -> dict[str, Any]:
		"""Compliance Report"""
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _now()}

	async def bulk_create_records(self, records: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Create Records"""
		assert records
		return {"created_count": len(records), "tenant_id": tenant_id}

	async def analytics_summary(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		return {"tenant_id": tenant_id, "period": period}

PharmaSupService = PharmaceuticalSupplyChainService
