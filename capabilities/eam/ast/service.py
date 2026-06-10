"""Domain service for APG enterprise asset management — expanded implementation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_EAM_AGENT_ROLES,
		SUPPORTED_EAM_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_EAM_AGENT_ROLES,
		SUPPORTED_EAM_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _record_id(prefix: str, value: str) -> str:
	slug = "".join(c.lower() if c.isalnum() else "_" for c in str(value)).strip("_")
	return f"{prefix}_{slug or 'record'}"


class EnterpriseAssetManagementService:
	"""
	Tenant-scoped asset, maintenance, inspection, inventory,
	depreciation, disposal, insurance, warranty, and analytics service.

	Expanded with: register_asset, asset_transfer, asset_disposal,
	depreciation_run, condition_assessment, maintenance_record,
	asset_insurance, warranty_tracking, asset_lifecycle_report,
	asset_register.
	"""

	def __init__(self) -> None:
		self._locations: dict[str, dict[str, Any]] = {}
		self._assets: dict[str, dict[str, Any]] = {}
		self._maintenance_plans: dict[str, dict[str, Any]] = {}
		self._work_orders: dict[str, dict[str, Any]] = {}
		self._inspections: dict[str, dict[str, Any]] = {}
		self._condition_readings: dict[str, dict[str, Any]] = {}
		self._inventory_reservations: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		# New stores
		self._transfers: list[dict[str, Any]] = []
		self._disposals: dict[str, dict[str, Any]] = {}
		self._depreciation_runs: list[dict[str, Any]] = []
		self._condition_assessments: dict[str, dict[str, Any]] = {}
		self._maintenance_records: list[dict[str, Any]] = []
		self._insurance_policies: dict[str, dict[str, Any]] = {}
		self._warranties: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# register_asset
	# ------------------------------------------------------------------

	def register_asset(
		self,
		asset_id: str,
		name: str,
		asset_class: str,
		acquisition_date: str,
		cost: float,
		location: str,
		responsible_dept: str,
		tenant_id: str = "default",
		owner: str = "asset_manager",
		useful_life_years: int = 5,
		salvage_value: float = 0.0,
		serial_number: str = "",
		capitalized: bool = True,
		criticality: str = "medium",
	) -> dict[str, Any]:
		"""
		Register an enterprise asset with full lifecycle metadata.

		asset_class: e.g. 'equipment', 'vehicle', 'it_hardware', 'building', 'intangible'.
		acquisition_date: ISO date string.
		cost: Acquisition cost.
		location: Physical or logical location label.
		responsible_dept: Owning department.
		useful_life_years: Depreciable life in years.
		salvage_value: Residual value at end of useful life.
		"""
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_asset",
			"asset_owner_assigned": bool(owner),
			"asset_category_present": bool(asset_class),
			"asset_location_present": bool(location),
			"criticality_present": bool(criticality),
			"capitalized": capitalized,
			"fixed_asset_ref_present": bool(serial_number),
			"health_score": 100,
		}
		self._enforce(context)
		if not asset_id:
			raise ValueError("asset_id_required")
		if float(cost) < 0:
			raise ValueError("asset_cost_must_be_non_negative")
		record = {
			"id": _record_id("eam_asset", asset_id),
			"asset_id": asset_id,
			"tenant_id": tenant_id,
			"name": name,
			"asset_class": asset_class,
			"category": asset_class,
			"owner": owner,
			"responsible_dept": responsible_dept,
			"location": location,
			"location_id": location,
			"acquisition_date": acquisition_date,
			"cost": Decimal(str(cost)),
			"net_book_value": Decimal(str(cost)),
			"accumulated_depreciation": Decimal("0"),
			"useful_life_years": useful_life_years,
			"salvage_value": Decimal(str(salvage_value)),
			"serial_number": serial_number,
			"capitalized": capitalized,
			"criticality": criticality,
			"health_score": 100.0,
			"status": "in_service",
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		self._assets[record["id"]] = record
		self._emit("asset_registered", tenant_id, record["id"], {"asset_id": asset_id, "asset_class": asset_class, "cost": float(cost)})
		return deepcopy(record)

	def asset_transfer(
		self,
		asset_id: str,
		from_location: str,
		to_location: str,
		transfer_date: str,
		approved_by: str,
		tenant_id: str = "default",
		transfer_id: str | None = None,
		transferred_by: str = "system",
		reason: str = "",
	) -> dict[str, Any]:
		"""
		Transfer an asset from one location to another.

		from_location / to_location: Location labels or IDs.
		transfer_date: ISO date string.
		approved_by: Approver identity.
		"""
		asset = self._require_asset(asset_id, tenant_id)
		if not approved_by:
			raise PermissionError("transfer_approval_required")
		if not from_location or not to_location:
			raise ValueError("from_location_and_to_location_required")
		if from_location == to_location:
			raise ValueError("transfer_locations_must_differ")
		if asset.get("status") == "disposed":
			raise PermissionError("cannot_transfer_disposed_asset")
		resolved_id = transfer_id or _record_id("eam_transfer", f"{asset_id}_{transfer_date}")
		record = {
			"transfer_id": resolved_id,
			"asset_id": asset_id,
			"asset_name": asset["name"],
			"tenant_id": tenant_id,
			"from_location": from_location,
			"to_location": to_location,
			"transfer_date": transfer_date,
			"approved_by": approved_by,
			"transferred_by": transferred_by,
			"reason": reason,
			"transferred_at": _now(),
		}
		self._transfers.append(record)
		asset["location"] = to_location
		asset["location_id"] = to_location
		asset["updated_at"] = _now()
		self._emit("asset_transferred", tenant_id, asset["id"], {"from": from_location, "to": to_location})
		return record

	def asset_disposal(
		self,
		asset_id: str,
		disposal_method: str,
		proceeds: float,
		disposal_date: str,
		approved_by: str,
		tenant_id: str = "default",
		disposal_id: str | None = None,
		reason: str = "",
	) -> dict[str, Any]:
		"""
		Record the disposal of an asset.

		disposal_method: 'sale', 'scrap', 'donation', 'write_off', 'trade_in'.
		proceeds: Sale/recovery proceeds (0 for scrap/write-off).
		"""
		asset = self._require_asset(asset_id, tenant_id)
		if not approved_by:
			raise PermissionError("disposal_approval_required")
		if asset.get("status") == "disposed":
			raise PermissionError("asset_already_disposed")
		supported_methods = {"sale", "scrap", "donation", "write_off", "trade_in", "auction"}
		if disposal_method not in supported_methods:
			raise ValueError(f"unsupported_disposal_method:{disposal_method}")
		nbv = float(asset.get("net_book_value", asset.get("cost", 0)))
		gain_loss = float(proceeds) - nbv
		resolved_id = disposal_id or _record_id("eam_disposal", f"{asset_id}_{disposal_date}")
		record = {
			"disposal_id": resolved_id,
			"asset_id": asset_id,
			"asset_name": asset["name"],
			"tenant_id": tenant_id,
			"disposal_method": disposal_method,
			"proceeds": Decimal(str(proceeds)),
			"net_book_value_at_disposal": Decimal(str(nbv)),
			"gain_loss": Decimal(str(gain_loss)),
			"disposal_date": disposal_date,
			"reason": reason,
			"approved_by": approved_by,
			"disposed_at": _now(),
		}
		self._disposals[resolved_id] = record
		asset["status"] = "disposed"
		asset["net_book_value"] = Decimal("0")
		asset["disposed_at"] = disposal_date
		asset["updated_at"] = _now()
		self._emit("asset_disposed", tenant_id, asset["id"], {"disposal_method": disposal_method, "proceeds": float(proceeds), "gain_loss": float(gain_loss)})
		return deepcopy(record)

	def depreciation_run(
		self,
		period: str,
		method: str,
		tenant_id: str = "default",
		asset_class: str | None = None,
		run_by: str = "finance",
	) -> dict[str, Any]:
		"""
		Run periodic depreciation for all (or a class of) assets.

		period: 'YYYY-MM' period for depreciation.
		method: 'straight_line', 'declining_balance', 'units_of_production', 'sum_of_years'.
		asset_class: Optional filter; None = all classes.
		"""
		supported_methods = {"straight_line", "declining_balance", "units_of_production", "sum_of_years"}
		if method not in supported_methods:
			raise ValueError(f"unsupported_depreciation_method:{method}")
		assets = [
			a for a in self._assets.values()
			if a["tenant_id"] == tenant_id
			and a.get("capitalized", True)
			and a.get("status") not in {"disposed", "written_off"}
			and (asset_class is None or a.get("asset_class") == asset_class or a.get("category") == asset_class)
		]
		results: list[dict[str, Any]] = []
		total_depreciation = Decimal("0")
		for asset in assets:
			cost = Decimal(str(asset.get("cost", 0)))
			salvage = Decimal(str(asset.get("salvage_value", 0)))
			useful_life_years = int(asset.get("useful_life_years", 5))
			acc_dep = Decimal(str(asset.get("accumulated_depreciation", 0)))
			nbv = Decimal(str(asset.get("net_book_value", cost)))
			depreciable_base = cost - salvage
			if method == "straight_line":
				annual_dep = depreciable_base / Decimal(str(max(1, useful_life_years)))
				period_dep = annual_dep / Decimal("12")
			elif method == "declining_balance":
				rate = Decimal("2") / Decimal(str(max(1, useful_life_years)))
				period_dep = nbv * rate / Decimal("12")
			else:
				annual_dep = depreciable_base / Decimal(str(max(1, useful_life_years)))
				period_dep = annual_dep / Decimal("12")
			# Cap at remaining NBV - salvage
			max_dep = max(Decimal("0"), nbv - salvage)
			period_dep = min(period_dep, max_dep)
			period_dep = period_dep.quantize(Decimal("0.01"))
			new_acc = acc_dep + period_dep
			new_nbv = cost - new_acc
			asset["accumulated_depreciation"] = new_acc
			asset["net_book_value"] = new_nbv
			asset["updated_at"] = _now()
			total_depreciation += period_dep
			results.append({
				"asset_id": asset["asset_id"],
				"asset_name": asset["name"],
				"asset_class": asset.get("asset_class", asset.get("category")),
				"cost": str(cost),
				"period_depreciation": str(period_dep),
				"accumulated_depreciation": str(new_acc),
				"net_book_value": str(new_nbv),
			})
		run = {
			"period": period,
			"method": method,
			"tenant_id": tenant_id,
			"asset_class_filter": asset_class,
			"assets_processed": len(results),
			"total_depreciation": str(total_depreciation),
			"run_by": run_by,
			"run_at": _now(),
			"results": results,
		}
		self._depreciation_runs.append(run)
		return run

	def condition_assessment(
		self,
		asset_id: str,
		assessor_id: str,
		condition_rating: int,
		notes: str,
		tenant_id: str = "default",
		assessment_id: str | None = None,
		next_assessment_date: str = "",
	) -> dict[str, Any]:
		"""
		Record a condition assessment for an asset.

		condition_rating: Integer 1-5 (1=poor, 5=excellent).
		notes: Assessment notes.
		"""
		asset = self._require_asset(asset_id, tenant_id)
		if not 1 <= condition_rating <= 5:
			raise ValueError("condition_rating_must_be_1_to_5")
		if not assessor_id:
			raise ValueError("assessor_id_required")
		health_map = {1: 20.0, 2: 40.0, 3: 60.0, 4: 80.0, 5: 100.0}
		health_score = health_map[condition_rating]
		resolved_id = assessment_id or _record_id("eam_assess", f"{asset_id}_{condition_rating}")
		record = {
			"assessment_id": resolved_id,
			"asset_id": asset_id,
			"asset_name": asset["name"],
			"tenant_id": tenant_id,
			"assessor_id": assessor_id,
			"condition_rating": condition_rating,
			"condition_label": {1: "poor", 2: "below_average", 3: "average", 4: "good", 5: "excellent"}[condition_rating],
			"health_score": health_score,
			"notes": notes,
			"next_assessment_date": next_assessment_date,
			"assessed_at": _now(),
		}
		self._condition_assessments[resolved_id] = record
		asset["health_score"] = health_score
		if health_score < 40:
			asset["status"] = "degraded"
		asset["updated_at"] = _now()
		self._emit("condition_assessed", tenant_id, asset["id"], {"condition_rating": condition_rating, "health_score": health_score})
		return record

	def maintenance_record(
		self,
		asset_id: str,
		maintenance_type: str,
		cost: float,
		performed_by: str,
		date: str,
		tenant_id: str = "default",
		record_id: str | None = None,
		description: str = "",
		parts_used: list[str] | None = None,
	) -> dict[str, Any]:
		"""
		Record a maintenance activity on an asset.

		maintenance_type: 'preventive', 'corrective', 'predictive', 'breakdown', 'overhaul'.
		cost: Maintenance cost.
		performed_by: Technician or contractor identity.
		"""
		asset = self._require_asset(asset_id, tenant_id)
		if not performed_by:
			raise ValueError("performed_by_required")
		if float(cost) < 0:
			raise ValueError("maintenance_cost_must_be_non_negative")
		supported_types = {"preventive", "corrective", "predictive", "breakdown", "overhaul", "inspection"}
		if maintenance_type not in supported_types:
			raise ValueError(f"unsupported_maintenance_type:{maintenance_type}")
		resolved_id = record_id or _record_id("eam_maint", f"{asset_id}_{date}")
		record = {
			"record_id": resolved_id,
			"asset_id": asset_id,
			"asset_name": asset["name"],
			"tenant_id": tenant_id,
			"maintenance_type": maintenance_type,
			"cost": Decimal(str(cost)),
			"performed_by": performed_by,
			"date": date,
			"description": description,
			"parts_used": list(parts_used or []),
			"recorded_at": _now(),
		}
		self._maintenance_records.append(record)
		# Update asset status after corrective/breakdown maintenance
		if maintenance_type in {"corrective", "breakdown"} and asset.get("status") == "degraded":
			asset["status"] = "in_service"
			asset["updated_at"] = _now()
		self._emit("maintenance_recorded", tenant_id, asset["id"], {"maintenance_type": maintenance_type, "cost": float(cost)})
		return deepcopy(record)

	def asset_insurance(
		self,
		asset_id: str,
		policy_id: str,
		insured_value: float,
		tenant_id: str = "default",
		insurer: str = "",
		policy_start: str = "",
		policy_end: str = "",
		premium: float = 0.0,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""
		Record insurance policy information for an asset.

		policy_id: Insurance policy reference.
		insured_value: Insured replacement value.
		"""
		asset = self._require_asset(asset_id, tenant_id)
		if not policy_id:
			raise ValueError("policy_id_required")
		if float(insured_value) < 0:
			raise ValueError("insured_value_must_be_non_negative")
		ins_key = f"{tenant_id}:{asset_id}:{policy_id}"
		record = {
			"asset_id": asset_id,
			"asset_name": asset["name"],
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"insurer": insurer,
			"insured_value": Decimal(str(insured_value)),
			"premium": Decimal(str(premium)),
			"currency": currency,
			"policy_start": policy_start,
			"policy_end": policy_end,
			"status": "active",
			"recorded_at": _now(),
		}
		self._insurance_policies[ins_key] = record
		asset["insurance_policy_id"] = policy_id
		asset["insured_value"] = float(insured_value)
		asset["updated_at"] = _now()
		self._emit("insurance_recorded", tenant_id, asset["id"], {"policy_id": policy_id, "insured_value": float(insured_value)})
		return deepcopy(record)

	def warranty_tracking(
		self,
		asset_id: str,
		warranty_expiry: str,
		vendor: str,
		tenant_id: str = "default",
		warranty_type: str = "manufacturer",
		coverage: str = "",
		extended: bool = False,
	) -> dict[str, Any]:
		"""
		Track warranty information for an asset.

		warranty_expiry: ISO date string for warranty end date.
		vendor: Warranty provider name.
		warranty_type: 'manufacturer', 'extended', 'service_contract', 'on-site'.
		"""
		asset = self._require_asset(asset_id, tenant_id)
		if not warranty_expiry:
			raise ValueError("warranty_expiry_required")
		if not vendor:
			raise ValueError("warranty_vendor_required")
		today = datetime.now(timezone.utc).date().isoformat()
		days_remaining = None
		try:
			expiry_date = datetime.strptime(warranty_expiry, "%Y-%m-%d").date()
			today_date = datetime.now(timezone.utc).date()
			days_remaining = (expiry_date - today_date).days
		except ValueError as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		w_key = f"{tenant_id}:{asset_id}"
		record = {
			"asset_id": asset_id,
			"asset_name": asset["name"],
			"tenant_id": tenant_id,
			"warranty_type": warranty_type,
			"vendor": vendor,
			"warranty_expiry": warranty_expiry,
			"coverage": coverage,
			"extended": extended,
			"days_remaining": days_remaining,
			"status": "active" if days_remaining is None or days_remaining > 0 else "expired",
			"recorded_at": _now(),
		}
		self._warranties[w_key] = record
		asset["warranty_expiry"] = warranty_expiry
		asset["warranty_vendor"] = vendor
		asset["warranty_status"] = record["status"]
		asset["updated_at"] = _now()
		return deepcopy(record)

	def asset_lifecycle_report(
		self,
		asset_class: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Generate a lifecycle report for all assets of a given class over a period.

		period: 'YYYY' year or 'YYYY-MM' month.
		Returns acquisition, depreciation, maintenance, disposal, and NBV statistics.
		"""
		assets = [
			a for a in self._assets.values()
			if a["tenant_id"] == tenant_id
			and (a.get("asset_class") == asset_class or a.get("category") == asset_class)
		]
		active = [a for a in assets if a.get("status") != "disposed"]
		disposed = [a for a in assets if a.get("status") == "disposed"]
		total_cost = sum(float(a.get("cost", 0)) for a in assets)
		total_nbv = sum(float(a.get("net_book_value", 0)) for a in active)
		total_acc_dep = sum(float(a.get("accumulated_depreciation", 0)) for a in assets)
		# Maintenance costs for period
		period_maint = [
			m for m in self._maintenance_records
			if m["tenant_id"] == tenant_id
			and m.get("date", "")[:len(period)] == period
			and any(a["asset_id"] == m["asset_id"] for a in assets)
		]
		total_maint_cost = sum(float(m.get("cost", 0)) for m in period_maint)
		# Disposals for period
		period_disposals = [
			d for d in self._disposals.values()
			if d["tenant_id"] == tenant_id
			and d.get("disposal_date", "")[:len(period)] == period
			and any(a["asset_id"] == d["asset_id"] for a in assets)
		]
		return {
			"tenant_id": tenant_id,
			"asset_class": asset_class,
			"period": period,
			"total_asset_count": len(assets),
			"active_asset_count": len(active),
			"disposed_asset_count": len(disposed),
			"total_acquisition_cost": total_cost,
			"total_accumulated_depreciation": total_acc_dep,
			"total_net_book_value": total_nbv,
			"average_health_score": round(sum(a.get("health_score", 100) for a in active) / len(active), 2) if active else 0.0,
			"maintenance_event_count": len(period_maint),
			"total_maintenance_cost": total_maint_cost,
			"disposal_count": len(period_disposals),
			"total_disposal_proceeds": sum(float(d.get("proceeds", 0)) for d in period_disposals),
			"insurance_coverage_count": sum(1 for k in self._insurance_policies if k.startswith(f"{tenant_id}:") and any(a["asset_id"] in k for a in assets)),
			"warranty_tracked_count": sum(1 for k in self._warranties if k.startswith(f"{tenant_id}:") and any(a["asset_id"] in k for a in assets)),
			"generated_at": _now(),
		}

	def asset_register(
		self,
		tenant_id: str = "default",
		filters: dict[str, Any] | None = None,
		include_disposed: bool = False,
	) -> list[dict[str, Any]]:
		"""
		Return the full asset register for a tenant.

		filters: Dict supporting 'asset_class', 'status', 'location', 'criticality', 'department'.
		include_disposed: Whether to include disposed assets.
		"""
		f = filters or {}
		assets = [a for a in self._assets.values() if a["tenant_id"] == tenant_id]
		if not include_disposed:
			assets = [a for a in assets if a.get("status") != "disposed"]
		if "asset_class" in f:
			assets = [a for a in assets if a.get("asset_class") == f["asset_class"] or a.get("category") == f["asset_class"]]
		if "status" in f:
			assets = [a for a in assets if a.get("status") == f["status"]]
		if "location" in f:
			assets = [a for a in assets if a.get("location") == f["location"] or a.get("location_id") == f["location"]]
		if "criticality" in f:
			assets = [a for a in assets if a.get("criticality") == f["criticality"]]
		if "department" in f:
			assets = [a for a in assets if a.get("responsible_dept") == f["department"]]
		# Enrich with warranty and insurance status
		result = []
		for asset in sorted(assets, key=lambda a: a.get("asset_id", a["id"])):
			entry = deepcopy(asset)
			warranty = self._warranties.get(f"{tenant_id}:{asset['asset_id']}")
			entry["warranty_status"] = warranty["status"] if warranty else "no_warranty"
			entry["warranty_expiry"] = warranty["warranty_expiry"] if warranty else None
			# Convert Decimal to float for serialisation
			for field in ("cost", "net_book_value", "accumulated_depreciation", "salvage_value"):
				if field in entry and isinstance(entry[field], Decimal):
					entry[field] = float(entry[field])
			result.append(entry)
		return result

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def register_location(self, location_id: str, tenant_id: str, name: str, location_type: str, parent_location_id: str | None = None) -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_location", "location_type_present": bool(location_type)}
		self._enforce(context)
		if parent_location_id:
			self._require_location(parent_location_id, tenant_id)
		record = {"id": _record_id("eam_location", location_id), "location_id": location_id, "tenant_id": tenant_id, "name": name, "location_type": location_type, "parent_location_id": parent_location_id, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._locations[record["id"]] = record
		self._emit("location_registered", tenant_id, record["id"], {"location_id": location_id, "location_type": location_type})
		return deepcopy(record)

	def create_maintenance_plan(self, plan_id: str, tenant_id: str, asset_record_id: str, strategy: str, interval_days: int, condition_source: str | None = None) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id)
		context = {"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_maintenance_plan", "maintenance_strategy_present": bool(strategy), "interval_present": interval_days is not None, "interval_days": interval_days, "predictive_plan": strategy == "predictive", "condition_source_present": bool(condition_source)}
		self._enforce(context)
		record = {"id": _record_id("eam_maintenance_plan", plan_id), "plan_id": plan_id, "tenant_id": tenant_id, "asset_record_id": asset["id"], "asset_id": asset["asset_id"], "strategy": strategy, "interval_days": interval_days, "condition_source": condition_source, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._maintenance_plans[record["id"]] = record
		self._emit("maintenance_plan_created", tenant_id, record["id"], {"asset_id": asset["asset_id"], "strategy": strategy})
		return deepcopy(record)

	def open_work_order(self, work_order_id: str, tenant_id: str, asset_record_id: str, title: str, priority: str, safety_plan: str, approved_by: str | None = None) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id) if asset_record_id else None
		context = {"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_work_order", "asset_present": asset is not None, "priority_present": bool(priority), "safety_plan_present": bool(safety_plan), "critical_asset": asset["criticality"] == "critical" if asset else False, "approved": bool(approved_by)}
		self._enforce(context)
		record = {"id": _record_id("eam_work_order", work_order_id), "work_order_id": work_order_id, "tenant_id": tenant_id, "asset_record_id": asset["id"] if asset else None, "asset_id": asset["asset_id"] if asset else None, "title": title, "priority": priority, "safety_plan": safety_plan, "approved_by": approved_by, "status": "work_open", "event_stream": "bytewax", "updated_at": _now()}
		self._work_orders[record["id"]] = record
		self._emit("work_order_opened", tenant_id, record["id"], {"asset_id": asset["asset_id"] if asset else None, "priority": priority})
		return deepcopy(record)

	def create_work_order(self, work_order_id: str, tenant_id: str, asset_id: str, title: str, priority: str, safety_plan: str = "standard", approved_by: str | None = None) -> dict[str, Any]:
		return self.open_work_order(work_order_id, tenant_id, asset_id, title, priority, safety_plan, approved_by)

	def complete_work_order(self, tenant_id: str, work_order_record_id: str, outcome: str, completed_by: str) -> dict[str, Any]:
		work_order = self._require_work_order(work_order_record_id, tenant_id)
		work_order["outcome"] = outcome
		work_order["completed_by"] = completed_by
		work_order["status"] = "work_complete"
		work_order["updated_at"] = _now()
		self._emit("work_order_completed", tenant_id, work_order_record_id, {"outcome": outcome})
		return deepcopy(work_order)

	def record_inspection(self, inspection_id: str, tenant_id: str, asset_record_id: str, result: str, inspector: str, condition_score: float | None = None) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id) if asset_record_id else None
		record = {"id": _record_id("eam_inspection", inspection_id), "inspection_id": inspection_id, "tenant_id": tenant_id, "asset_record_id": asset["id"] if asset else None, "asset_id": asset["asset_id"] if asset else None, "result": result, "inspector": inspector, "condition_score": condition_score, "status": "recorded", "event_stream": "bytewax", "updated_at": _now()}
		self._inspections[record["id"]] = record
		if asset and condition_score is not None:
			asset["health_score"] = float(condition_score)
			asset["status"] = "degraded" if condition_score < 50 else "in_service"
			asset["updated_at"] = _now()
		self._emit("inspection_recorded", tenant_id, record["id"], {"asset_id": asset["asset_id"] if asset else None, "result": result})
		return deepcopy(record)

	def record_condition_reading(self, reading_id: str, tenant_id: str, asset_record_id: str, metric: str, value: float | None, unit: str, review_recorded: bool = False, alert_threshold: float | None = None) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id)
		condition_alert = alert_threshold is not None and value is not None and value > alert_threshold
		record = {"id": _record_id("eam_condition", reading_id), "reading_id": reading_id, "tenant_id": tenant_id, "asset_record_id": asset["id"], "asset_id": asset["asset_id"], "metric": metric, "value": float(value) if value is not None else None, "unit": unit, "alert_threshold": alert_threshold, "status": "degraded" if condition_alert else "normal", "event_stream": "bytewax", "updated_at": _now()}
		self._condition_readings[record["id"]] = record
		if condition_alert:
			asset["status"] = "degraded"
			asset["updated_at"] = _now()
		self._emit("condition_reading_recorded", tenant_id, record["id"], {"metric": metric, "status": record["status"]})
		return deepcopy(record)

	def record_condition(self, reading_id: str, tenant_id: str, asset_id: str, reading_type: str, value: float, threshold: float) -> dict[str, Any]:
		return self.record_condition_reading(reading_id, tenant_id, asset_id, reading_type, value, "unit", review_recorded=True, alert_threshold=threshold)

	def reserve_inventory(self, reservation_id: str, tenant_id: str, part_id: str, quantity: int, work_order_record_id: str | None = None) -> dict[str, Any]:
		if work_order_record_id:
			self._require_work_order(work_order_record_id, tenant_id)
		context = {"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "reserve_inventory", "part_present": bool(part_id), "quantity_present": quantity is not None, "quantity": quantity}
		self._enforce(context)
		record = {"id": _record_id("eam_inventory_reservation", reservation_id), "reservation_id": reservation_id, "tenant_id": tenant_id, "part_id": part_id, "quantity": quantity, "work_order_record_id": work_order_record_id, "status": "reserved", "event_stream": "bytewax", "updated_at": _now()}
		self._inventory_reservations[record["id"]] = record
		self._emit("inventory_reservation_created", tenant_id, record["id"], {"part_id": part_id, "quantity": quantity})
		return deepcopy(record)

	def register_eam_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_eam_agent", "agent_runtime_supported": runtime in SUPPORTED_EAM_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_EAM_AGENT_ROLES}
		self._enforce(context)
		record = {"id": _record_id("eam_agent", name), "tenant_id": tenant_id, "name": name, "runtime": runtime, "role": role, "instructions": instructions, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._agents[record["id"]] = record
		self._emit("eam_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_eam_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown EAM agent: {agent_id}")
		return evaluate_capability_rules({"tenant_context_present": bool(tenant_id), "operation": "agent_eam_action", "action": action, "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})

	def validate_batch_import(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		result = evaluate_capability_rules({"tenant_context_present": bool(tenant_id), "operation": "eam_batch_import", "event_stream": event_stream})
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"location_count": len(self.list_locations(tenant_id)),
			"asset_count": len(self.list_assets(tenant_id)),
			"in_service_count": sum(1 for a in self._assets.values() if a["tenant_id"] == tenant_id and a.get("status") == "in_service"),
			"disposed_count": sum(1 for a in self._assets.values() if a["tenant_id"] == tenant_id and a.get("status") == "disposed"),
			"degraded_count": sum(1 for a in self._assets.values() if a["tenant_id"] == tenant_id and a.get("status") == "degraded"),
			"maintenance_plan_count": len(self.list_maintenance_plans(tenant_id)),
			"open_work_order_count": len([o for o in self.list_work_orders(tenant_id) if o["status"] != "work_complete"]),
			"inspection_count": len(self.list_inspections(tenant_id)),
			"maintenance_record_count": sum(1 for m in self._maintenance_records if m["tenant_id"] == tenant_id),
			"disposal_count": sum(1 for d in self._disposals.values() if d["tenant_id"] == tenant_id),
			"insurance_policy_count": sum(1 for k in self._insurance_policies if k.startswith(f"{tenant_id}:")),
			"warranty_tracked_count": sum(1 for k in self._warranties if k.startswith(f"{tenant_id}:")),
			"eam_agent_count": len(self.list_eam_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def reliability_summary(self, tenant_id: str) -> dict[str, Any]:
		assets = self.list_assets(tenant_id)
		critical = [a for a in assets if a["criticality"] == "critical"]
		degraded = [a for a in assets if a["status"] == "degraded"]
		avg_health = round(sum(a["health_score"] for a in assets) / len(assets), 2) if assets else 0
		return {"tenant_id": tenant_id, "asset_count": len(assets), "critical_asset_count": len(critical), "degraded_asset_count": len(degraded), "average_health_score": avg_health, "condition_reading_count": len(self.list_condition_readings(tenant_id))}

	def list_locations(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._locations, tenant_id)

	def list_assets(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._assets, tenant_id)

	def list_maintenance_plans(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._maintenance_plans, tenant_id)

	def list_work_orders(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._work_orders, tenant_id)

	def list_inspections(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._inspections, tenant_id)

	def list_condition_readings(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._condition_readings, tenant_id)

	def list_conditions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self.list_condition_readings(tenant_id)

	def list_inventory_reservations(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._inventory_reservations, tenant_id)

	def list_eam_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		tenant_id = data.get("tenant_id", "default")
		location = data.get("location_id", data.get("location", "main-site"))
		if not self._find_location(location, tenant_id):
			self.register_location(location, tenant_id, data.get("location_name", "Main Site"), data.get("location_type", "site"))
		return self.register_asset(
			asset_id=data.get("asset_id", data.get("id", "asset")),
			name=data.get("name", "Asset"),
			asset_class=data.get("asset_class", data.get("category", "equipment")),
			acquisition_date=data.get("acquisition_date", "2025-01-01"),
			cost=float(data.get("cost", 0)),
			location=location,
			responsible_dept=data.get("responsible_dept", data.get("owner", "operations")),
			tenant_id=tenant_id,
			owner=data.get("owner", "asset_manager"),
			criticality=data.get("criticality", "medium"),
			capitalized=data.get("capitalized", True),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_assets(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _require_location(self, location_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._find_location(location_id, tenant_id)
		if record is None:
			raise KeyError(f"Unknown location: {location_id}")
		return record

	def _find_location(self, location_id: str, tenant_id: str) -> dict[str, Any] | None:
		for record in self._locations.values():
			if record["tenant_id"] == tenant_id and (record["location_id"] == location_id or record["id"] == location_id):
				return record
		return None

	def _require_asset(self, asset_id: str, tenant_id: str) -> dict[str, Any]:
		for record in self._assets.values():
			if record["tenant_id"] == tenant_id and (record["asset_id"] == asset_id or record["id"] == asset_id):
				return record
		raise KeyError(f"Unknown asset: {asset_id}")

	def _require_work_order(self, work_order_record_id: str, tenant_id: str) -> dict[str, Any]:
		for record in self._work_orders.values():
			if record["tenant_id"] == tenant_id and (record["work_order_id"] == work_order_record_id or record["id"] == work_order_record_id):
				return record
		raise KeyError(f"Unknown work order: {work_order_record_id}")

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] in {"deny", "require_review"}:
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		result = []
		for record in records.values():
			if record["tenant_id"] != tenant_id:
				continue
			entry = deepcopy(record)
			for field in ("cost", "net_book_value", "accumulated_depreciation", "salvage_value", "insured_value", "premium"):
				if field in entry and isinstance(entry[field], Decimal):
					entry[field] = float(entry[field])
			result.append(entry)
		return result

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({"event": event_name, "tenant_id": tenant_id, "record_id": record_id, "payload": deepcopy(payload), "processor": "bytewax", "stream": streaming_manifest()["stream"], "created_at": _now()})


	async def total_cost_ownership(
		self,
		tenant_id: str,
		asset_id: str,
	) -> dict[str, Any]:
		"""Compute Total Cost of Ownership for an asset.

		TCO = acquisition cost + maintenance spend + insurance premiums - salvage value.
		"""
		asset = self._require_asset(asset_id, tenant_id)
		cost = float(asset.get("cost", 0))
		salvage = float(asset.get("salvage_value", 0))
		insured = float(asset.get("insured_value", 0))
		# Approximate maintenance from work orders
		wo_cost = sum(
			float(wo.get("cost", 0))
			for wo in self._work_orders.values()
			if wo["tenant_id"] == tenant_id and wo.get("asset_id") == asset_id
		)
		tco = round(cost + wo_cost + insured * 0.02 - salvage, 2)  # 2% insurance premium proxy
		return {
			"asset_id": asset_id,
			"tenant_id": tenant_id,
			"acquisition_cost": cost,
			"maintenance_spend": round(wo_cost, 2),
			"insurance_premium_estimate": round(insured * 0.02, 2),
			"salvage_value": salvage,
			"total_cost_of_ownership": tco,
			"generated_at": _now(),
		}

	async def capital_expenditure_plan(
		self,
		tenant_id: str,
		horizon_years: int = 5,
	) -> dict[str, Any]:
		"""Build a forward-looking capital expenditure plan.

		Assets within 2 years of end-of-life are flagged as replacement candidates.
		"""
		from datetime import datetime as _dt, timedelta as _td
		horizon_date = (_dt.utcnow() + _td(days=365 * horizon_years)).isoformat()
		assets = self._tenant_records(self._assets, tenant_id)
		candidates: list[dict[str, Any]] = []
		for a in assets:
			eol = a.get("end_of_life_date") or a.get("disposal_date")
			if eol and eol <= horizon_date:
				candidates.append({
					"asset_id": a.get("asset_id") or a.get("id"),
					"name": a.get("name", ""),
					"cost": a.get("cost", 0),
					"end_of_life_date": eol,
				})
		total_capex = sum(float(c["cost"]) for c in candidates)
		return {
			"tenant_id": tenant_id,
			"horizon_years": horizon_years,
			"replacement_candidates": len(candidates),
			"total_estimated_capex": round(total_capex, 2),
			"candidates": candidates,
			"generated_at": _now(),
		}

	async def asset_kpi_summary(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise asset KPI card for dashboard consumption."""
		assets = self._tenant_records(self._assets, tenant_id)
		work_orders = self._tenant_records(self._work_orders, tenant_id)
		active = sum(1 for a in assets if a.get("status") == "active")
		maintenance = sum(1 for a in assets if a.get("status") in {"maintenance", "under_repair"})
		open_wo = sum(1 for wo in work_orders if wo.get("status") in {"open", "in_progress"})
		total_nbv = sum(float(a.get("net_book_value", 0)) for a in assets)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_assets": len(assets),
			"active_assets": active,
			"in_maintenance": maintenance,
			"open_work_orders": open_wo,
			"total_net_book_value": round(total_nbv, 2),
			"utilisation_rate_pct": round(active / max(len(assets), 1) * 100, 1),
			"generated_at": _now(),
		}

	async def asset_portfolio_value(
		self,
		tenant_id: str,
		as_of_date: str | None = None,
	) -> dict[str, Any]:
		"""Return the aggregate portfolio value: gross cost, accumulated depreciation, NBV."""
		assets = self._tenant_records(self._assets, tenant_id)
		gross = sum(float(a.get("cost", 0)) for a in assets)
		accum_dep = sum(float(a.get("accumulated_depreciation", 0)) for a in assets)
		nbv = sum(float(a.get("net_book_value", 0)) for a in assets)
		insured = sum(float(a.get("insured_value", 0)) for a in assets)
		return {
			"tenant_id": tenant_id,
			"as_of_date": as_of_date or _now()[:10],
			"asset_count": len(assets),
			"gross_cost": round(gross, 2),
			"accumulated_depreciation": round(accum_dep, 2),
			"net_book_value": round(nbv, 2),
			"insured_value": round(insured, 2),
			"generated_at": _now(),
		}



	async def ml_failure_predict(self, *args, **kwargs):
		"""AI-powered predictive asset failure risk scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="asset_failure_prediction")
			return {"failure_risk": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

EAMAssetService = EnterpriseAssetManagementService
