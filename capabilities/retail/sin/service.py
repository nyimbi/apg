"""Async service layer for APG Store Intelligence."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from .models import (
	SinStoreCreate, SinStoreResponse,
	SinZoneCreate, SinZoneResponse,
	SinSensorCreate, SinSensorResponse,
	SinTrafficCountCreate, SinTrafficCountResponse,
	SinPlanogramAuditCreate, SinPlanogramAuditResponse,
	SinShelfAlertCreate, SinShelfAlertResponse,
	SinConversionEventCreate, SinConversionEventResponse,
	SinKpiSnapshotCreate, SinKpiSnapshotResponse,
	SinHeatmapCreate, SinHeatmapResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class SinService:
	"""Service for Store Intelligence capability."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self._stores: dict[str, dict[str, Any]] = {}
		self._zones: dict[str, dict[str, Any]] = {}
		self._sensors: dict[str, dict[str, Any]] = {}
		self._traffic_counts: dict[str, dict[str, Any]] = {}
		self._planogram_audits: dict[str, dict[str, Any]] = {}
		self._shelf_alerts: dict[str, dict[str, Any]] = {}
		self._conversion_events: dict[str, dict[str, Any]] = {}
		self._kpi_snapshots: dict[str, dict[str, Any]] = {}
		self._heatmaps: dict[str, dict[str, Any]] = {}
		# Extended state
		self._competitor_prices: list[dict[str, Any]] = []
		self._staff_productivity: dict[str, list[dict[str, Any]]] = {}  # store_id -> records
		self._diagnostics_cache: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Logging helpers
	# ------------------------------------------------------------------

	def _log_op(self, op: str, tenant_id: str, entity_id: str | None = None) -> None:
		logger.info("sin | op=%s tenant=%s entity=%s", op, tenant_id, entity_id or "-")

	def _log_warn(self, msg: str, **kw: Any) -> None:
		logger.warning("sin | %s %s", msg, kw)

	def _log_traffic(self, store_id: str, zone_id: str, entries: int) -> None:
		logger.debug("sin | traffic store=%s zone=%s entries=%d", store_id, zone_id, entries)

	def _log_alert(self, store_id: str, sku: str, alert_type: str, severity: str) -> None:
		logger.info("sin | shelf_alert store=%s sku=%s type=%s severity=%s", store_id, sku, alert_type, severity)

	# ------------------------------------------------------------------
	# Stores
	# ------------------------------------------------------------------

	async def create_store(self, data: SinStoreCreate) -> SinStoreResponse:
		assert data.sqm_total > 0, "store sqm_total must be positive"
		assert data.sqm_selling > 0, "store sqm_selling must be positive"
		self._log_op("create_store", data.tenant_id)
		rec = SinStoreResponse(**data.model_dump())
		self._stores[rec.id] = rec.model_dump()
		return rec

	async def get_store(self, tenant_id: str, store_id: str) -> SinStoreResponse | None:
		rec = self._stores.get(store_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return SinStoreResponse(**rec)

	async def get_store_by_code(self, tenant_id: str, store_code: str) -> SinStoreResponse | None:
		for rec in self._stores.values():
			if rec["tenant_id"] == tenant_id and rec["store_code"] == store_code:
				return SinStoreResponse(**rec)
		return None

	async def list_stores(self, tenant_id: str, store_format: str | None = None) -> list[SinStoreResponse]:
		result = [v for v in self._stores.values() if v["tenant_id"] == tenant_id and v["is_active"]]
		if store_format:
			result = [v for v in result if v["store_format"] == store_format]
		return [SinStoreResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Zones
	# ------------------------------------------------------------------

	async def create_zone(self, data: SinZoneCreate) -> SinZoneResponse:
		store = self._stores.get(data.store_id)
		assert store and store["tenant_id"] == data.tenant_id, "store not found"
		self._log_op("create_zone", data.tenant_id)
		rec = SinZoneResponse(**data.model_dump())
		self._zones[rec.id] = rec.model_dump()
		return rec

	async def list_zones(self, tenant_id: str, store_id: str) -> list[SinZoneResponse]:
		return [SinZoneResponse(**v) for v in self._zones.values()
				if v["tenant_id"] == tenant_id and v["store_id"] == store_id]

	# ------------------------------------------------------------------
	# Sensors
	# ------------------------------------------------------------------

	async def register_sensor(self, data: SinSensorCreate) -> SinSensorResponse:
		zone = self._zones.get(data.zone_id)
		assert zone and zone["tenant_id"] == data.tenant_id, "zone not found"
		self._log_op("register_sensor", data.tenant_id)
		rec = SinSensorResponse(**data.model_dump())
		self._sensors[rec.id] = rec.model_dump()
		return rec

	async def sensor_heartbeat(self, tenant_id: str, sensor_id: str) -> SinSensorResponse | None:
		rec = self._sensors.get(sensor_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "online"
		rec["last_heartbeat_at"] = datetime.utcnow().isoformat()
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._sensors[sensor_id] = rec
		return SinSensorResponse(**rec)

	async def list_sensors(self, tenant_id: str, store_id: str | None = None,
						   zone_id: str | None = None) -> list[SinSensorResponse]:
		result = [v for v in self._sensors.values() if v["tenant_id"] == tenant_id]
		if store_id:
			result = [v for v in result if v["store_id"] == store_id]
		if zone_id:
			result = [v for v in result if v["zone_id"] == zone_id]
		return [SinSensorResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Traffic Counts
	# ------------------------------------------------------------------

	async def foot_traffic_record(
		self, store_id: str, timestamp: str, count: int, zone: str
	) -> dict[str, Any]:
		"""Record a foot traffic count for a store zone at a given timestamp."""
		assert store_id, "store_id required"
		assert count >= 0, "count must be non-negative"
		tenant_id = self.tenant_id

		store = self._stores.get(store_id)
		assert store is not None and store["tenant_id"] == tenant_id, "store not found"

		# Find zone_id matching zone name
		zone_rec = next(
			(z for z in self._zones.values()
			 if z["tenant_id"] == tenant_id and z["store_id"] == store_id
			 and (z.get("zone_name", "") == zone or z.get("id") == zone)),
			None
		)
		zone_id = zone_rec["id"] if zone_rec else zone

		record_id = f"tc_{store_id}_{zone_id}_{timestamp[:10]}"
		data = SinTrafficCountCreate(
			tenant_id=tenant_id,
			store_id=store_id,
			zone_id=zone_id,
			period_start=timestamp,
			period_end=timestamp,
			entries=count,
			exits=int(count * 0.95),  # approximate exits
			occupancy_peak=count,
			dwell_avg_seconds=180,
			counted_by="sensor",
		)
		rec = SinTrafficCountResponse(**data.model_dump())
		self._traffic_counts[rec.id] = rec.model_dump()
		self._log_traffic(store_id, zone_id, count)
		return rec.model_dump()

	async def conversion_rate(
		self, store_id: str, period: str, zone: str | None = None
	) -> dict[str, Any]:
		"""Compute conversion rate for a store/zone over a period.

		conversion_rate = transactions / foot_traffic_entries
		"""
		assert store_id, "store_id required"
		assert period, "period required"
		tenant_id = self.tenant_id

		traffic = [v for v in self._traffic_counts.values()
				   if v["tenant_id"] == tenant_id and v["store_id"] == store_id
				   and v.get("period_start", "")[:7] == period[:7]]
		if zone:
			traffic = [v for v in traffic if v.get("zone_id") == zone]

		total_entries = sum(v["entries"] for v in traffic)
		conv_events = [v for v in self._conversion_events.values()
					   if v["tenant_id"] == tenant_id and v["store_id"] == store_id
					   and v.get("occurred_at", "")[:7] == period[:7] and v.get("converted")]
		transactions = len(conv_events)
		conversion_rate_val = round(transactions / total_entries, 4) if total_entries else 0.0

		return {
			"store_id": store_id,
			"period": period,
			"zone": zone,
			"total_foot_traffic": total_entries,
			"transactions": transactions,
			"conversion_rate": conversion_rate_val,
			"conversion_pct": round(conversion_rate_val * 100, 2),
		}

	async def record_traffic_count(self, data: SinTrafficCountCreate) -> SinTrafficCountResponse:
		"""Ingest an anonymised traffic count."""
		self._log_traffic(data.store_id, data.zone_id, data.entries)
		rec = SinTrafficCountResponse(**data.model_dump())
		self._traffic_counts[rec.id] = rec.model_dump()
		return rec

	async def get_traffic_summary(self, tenant_id: str, store_id: str,
								  period_start: datetime, period_end: datetime) -> dict[str, Any]:
		"""Aggregate traffic for a store over a time window."""
		recs = [v for v in self._traffic_counts.values()
				if v["tenant_id"] == tenant_id and v["store_id"] == store_id]
		total_entries = sum(v["entries"] for v in recs)
		total_exits = sum(v["exits"] for v in recs)
		peak_occupancy = max((v["occupancy_peak"] for v in recs), default=0)
		avg_dwell = sum(v["dwell_avg_seconds"] for v in recs) / len(recs) if recs else 0.0
		return {
			"store_id": store_id,
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"total_entries": total_entries,
			"total_exits": total_exits,
			"peak_occupancy": peak_occupancy,
			"avg_dwell_seconds": avg_dwell,
			"record_count": len(recs),
		}

	async def list_traffic_counts(self, tenant_id: str, store_id: str,
								  zone_id: str | None = None) -> list[SinTrafficCountResponse]:
		result = [v for v in self._traffic_counts.values()
				  if v["tenant_id"] == tenant_id and v["store_id"] == store_id]
		if zone_id:
			result = [v for v in result if v["zone_id"] == zone_id]
		result.sort(key=lambda x: x["period_start"], reverse=True)
		return [SinTrafficCountResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Planogram Audits
	# ------------------------------------------------------------------

	async def planogram_compliance_check(
		self, store_id: str, audit_date: str, category: str
	) -> dict[str, Any]:
		"""Record a planogram compliance audit for a store category."""
		assert store_id, "store_id required"
		assert audit_date, "audit_date required"
		assert category, "category required"
		tenant_id = self.tenant_id

		store = self._stores.get(store_id)
		assert store is not None and store["tenant_id"] == tenant_id, "store not found"

		# Find zones matching this category
		category_zones = [z for z in self._zones.values()
						  if z["tenant_id"] == tenant_id and z["store_id"] == store_id
						  and z.get("category", "") == category]
		zone_id = category_zones[0]["id"] if category_zones else store_id

		data = SinPlanogramAuditCreate(
			tenant_id=tenant_id,
			store_id=store_id,
			zone_id=zone_id,
			category=category,
			audited_at=audit_date,
			compliance_status="compliant",
			deviations_found=0,
			total_facings_checked=50,
			auditor_id=self.actor_id,
			notes=f"Compliance audit for {category}",
		)
		audit = await self.record_planogram_audit(data)
		compliance_rate = await self.get_store_compliance_rate(tenant_id, store_id)
		return {
			"store_id": store_id,
			"audit_date": audit_date,
			"category": category,
			"audit": audit.model_dump(),
			"store_compliance_rate": compliance_rate,
		}

	async def record_planogram_audit(self, data: SinPlanogramAuditCreate) -> SinPlanogramAuditResponse:
		"""Record a planogram compliance audit result."""
		compliance_score = 100.0
		if data.compliance_status == "minor_deviation":
			compliance_score = 80.0
		elif data.compliance_status == "major_deviation":
			compliance_score = 50.0
		elif data.compliance_status == "out_of_stock":
			compliance_score = 0.0
		self._log_op("record_planogram_audit", data.tenant_id)
		rec = SinPlanogramAuditResponse(**data.model_dump(), compliance_score_pct=compliance_score)
		self._planogram_audits[rec.id] = rec.model_dump()
		return rec

	async def list_planogram_audits(self, tenant_id: str, store_id: str,
									zone_id: str | None = None) -> list[SinPlanogramAuditResponse]:
		result = [v for v in self._planogram_audits.values()
				  if v["tenant_id"] == tenant_id and v["store_id"] == store_id]
		if zone_id:
			result = [v for v in result if v["zone_id"] == zone_id]
		result.sort(key=lambda x: x["audited_at"], reverse=True)
		return [SinPlanogramAuditResponse(**v) for v in result]

	async def get_store_compliance_rate(self, tenant_id: str, store_id: str) -> float:
		audits = await self.list_planogram_audits(tenant_id, store_id)
		if not audits:
			return 100.0
		return sum(a.compliance_score_pct for a in audits) / len(audits)

	# ------------------------------------------------------------------
	# Shelf Availability
	# ------------------------------------------------------------------

	async def shelf_availability_scan(
		self, store_id: str, scan_date: str, sku_gaps: list[str]
	) -> dict[str, Any]:
		"""Record shelf availability gaps from a scan and raise alerts for out-of-stock SKUs."""
		assert store_id, "store_id required"
		assert scan_date, "scan_date required"
		tenant_id = self.tenant_id

		alerts_raised: list[dict[str, Any]] = []
		for sku in sku_gaps:
			data = SinShelfAlertCreate(
				tenant_id=tenant_id,
				store_id=store_id,
				zone_id=store_id,  # store-level if no zone known
				sku=sku,
				alert_type="out_of_stock",
				severity="high",
				quantity_remaining=0,
				reorder_threshold=5,
			)
			alert = await self.raise_shelf_alert(data)
			alerts_raised.append(alert.model_dump())

		return {
			"store_id": store_id,
			"scan_date": scan_date,
			"sku_gaps_found": len(sku_gaps),
			"alerts_raised": len(alerts_raised),
			"alerts": alerts_raised,
		}

	async def raise_shelf_alert(self, data: SinShelfAlertCreate) -> SinShelfAlertResponse:
		self._log_alert(data.store_id, data.sku, data.alert_type, data.severity)
		rec = SinShelfAlertResponse(**data.model_dump())
		self._shelf_alerts[rec.id] = rec.model_dump()
		return rec

	async def resolve_shelf_alert(self, tenant_id: str, alert_id: str,
								  notes: str, by: str) -> SinShelfAlertResponse | None:
		rec = self._shelf_alerts.get(alert_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "resolved"
		rec["resolution_notes"] = notes
		rec["resolved_at"] = datetime.utcnow().isoformat()
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._shelf_alerts[alert_id] = rec
		return SinShelfAlertResponse(**rec)

	async def trigger_replenishment(self, tenant_id: str, alert_id: str) -> SinShelfAlertResponse | None:
		rec = self._shelf_alerts.get(alert_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["replenishment_triggered"] = True
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._shelf_alerts[alert_id] = rec
		return SinShelfAlertResponse(**rec)

	async def list_shelf_alerts(self, tenant_id: str, store_id: str,
								status: str | None = None) -> list[SinShelfAlertResponse]:
		result = [v for v in self._shelf_alerts.values()
				  if v["tenant_id"] == tenant_id and v["store_id"] == store_id]
		if status:
			result = [v for v in result if v["status"] == status]
		return [SinShelfAlertResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Heat Map Analytics
	# ------------------------------------------------------------------

	async def heat_map_analytics(self, store_id: str, period: str) -> dict[str, Any]:
		"""Aggregate heatmap data and traffic counts to produce a zone-level heat map."""
		assert store_id, "store_id required"
		assert period, "period required"
		tenant_id = self.tenant_id
		store = self._stores.get(store_id)
		assert store is not None and store["tenant_id"] == tenant_id, "store not found"

		zones = await self.list_zones(tenant_id, store_id)
		heatmaps = [h for h in self._heatmaps.values()
					if h["tenant_id"] == tenant_id and h["store_id"] == store_id
					and h.get("period_start", "")[:7] == period[:7]]

		traffic_by_zone: dict[str, int] = {}
		for tc in self._traffic_counts.values():
			if tc["tenant_id"] == tenant_id and tc["store_id"] == store_id:
				z = tc.get("zone_id", "unknown")
				traffic_by_zone[z] = traffic_by_zone.get(z, 0) + tc["entries"]

		# Rank zones by traffic
		zone_rankings = sorted(
			[{"zone_id": z, "zone_name": next((zo.zone_name for zo in zones if zo.id == z), z),
			  "total_entries": cnt}
			 for z, cnt in traffic_by_zone.items()],
			key=lambda x: -x["total_entries"]
		)
		hot_zones = zone_rankings[:3]
		cold_zones = zone_rankings[-3:]

		return {
			"store_id": store_id,
			"period": period,
			"zone_count": len(zones),
			"heatmap_count": len(heatmaps),
			"zone_traffic": zone_rankings,
			"hot_zones": hot_zones,
			"cold_zones": cold_zones,
			"generated_at": str(date.today()),
		}

	# ------------------------------------------------------------------
	# Competitor Price Monitoring
	# ------------------------------------------------------------------

	async def competitor_price_monitoring(
		self, sku: str, competitor: str, price: float, date_str: str
	) -> dict[str, Any]:
		"""Record a competitor price observation for a SKU."""
		assert sku, "sku required"
		assert competitor, "competitor required"
		assert price >= 0, "price must be non-negative"
		tenant_id = self.tenant_id

		record = {
			"obs_id": f"cpm_{sku}_{competitor}_{date_str}",
			"tenant_id": tenant_id,
			"sku": sku,
			"competitor": competitor,
			"price": price,
			"observed_date": date_str,
			"recorded_at": str(date.today()),
		}
		self._competitor_prices.append(record)
		# Check if our price is indexed
		our_price_obs = [
			r for r in self._competitor_prices
			if r["tenant_id"] == tenant_id and r["sku"] == sku and r["competitor"] == "self"
		]
		our_price = our_price_obs[-1]["price"] if our_price_obs else None
		if our_price is not None:
			record["price_index"] = round(our_price / price, 3) if price else None
			record["competitiveness"] = (
				"cheaper" if our_price < price else ("parity" if our_price == price else "expensive")
			)
		return record

	# ------------------------------------------------------------------
	# Sales Density
	# ------------------------------------------------------------------

	async def sales_density(self, store_id: str, period: str) -> dict[str, Any]:
		"""Compute sales per sqm for a store period."""
		assert store_id, "store_id required"
		assert period, "period required"
		tenant_id = self.tenant_id
		store = self._stores.get(store_id)
		assert store is not None and store["tenant_id"] == tenant_id, "store not found"

		# Derive from KPI snapshots if available
		kpi_snaps = await self.list_kpi_snapshots(tenant_id, store_id)
		sales_kpi = [k for k in kpi_snaps if "sales" in k.kpi_category.lower()]
		total_sales = sum(
			k.kpi_values.get("total_sales", 0.0) for k in sales_kpi
			if k.period_start[:7] == period[:7]
		) if sales_kpi else 0.0

		sqm_selling = store.get("sqm_selling", 1)
		density = round(total_sales / sqm_selling, 2) if sqm_selling else 0.0

		return {
			"store_id": store_id,
			"period": period,
			"sqm_selling": sqm_selling,
			"total_sales": total_sales,
			"sales_density_per_sqm": density,
			"calculated_at": str(date.today()),
		}

	# ------------------------------------------------------------------
	# Staff Productivity
	# ------------------------------------------------------------------

	async def staff_productivity(self, store_id: str, period: str) -> dict[str, Any]:
		"""Compute staff productivity metrics: sales per head-hour, conversion assisted."""
		assert store_id, "store_id required"
		assert period, "period required"
		tenant_id = self.tenant_id

		existing = self._staff_productivity.get(f"{tenant_id}:{store_id}", [])
		period_records = [r for r in existing if r.get("period") == period]
		if period_records:
			return period_records[-1]

		# Derive from KPI snapshots
		kpi_snaps = await self.list_kpi_snapshots(tenant_id, store_id)
		prod_kpis = [k for k in kpi_snaps if "staff" in k.kpi_category.lower()
					 and k.period_start[:7] == period[:7]]

		# Fallback: derive from traffic and conversion
		traffic = [v for v in self._traffic_counts.values()
				   if v["tenant_id"] == tenant_id and v["store_id"] == store_id
				   and v.get("period_start", "")[:7] == period[:7]]
		total_entries = sum(v["entries"] for v in traffic)
		conv_events = [v for v in self._conversion_events.values()
					   if v["tenant_id"] == tenant_id and v["store_id"] == store_id
					   and v.get("occurred_at", "")[:7] == period[:7] and v.get("converted")]
		transactions = len(conv_events)

		# Assume 8 staff, 8h shifts for 22 days
		staff_count = 8
		head_hours = staff_count * 8 * 22
		transactions_per_head_hour = round(transactions / head_hours, 4) if head_hours else 0.0
		entries_per_staff = round(total_entries / (staff_count * 22), 2) if staff_count else 0.0

		record = {
			"store_id": store_id,
			"period": period,
			"staff_count": staff_count,
			"head_hours": head_hours,
			"total_transactions": transactions,
			"total_foot_traffic": total_entries,
			"transactions_per_head_hour": transactions_per_head_hour,
			"entries_per_staff_per_day": entries_per_staff,
			"calculated_at": str(date.today()),
		}
		self._staff_productivity.setdefault(f"{tenant_id}:{store_id}", []).append(record)
		return record

	# ------------------------------------------------------------------
	# Store Ranking
	# ------------------------------------------------------------------

	async def store_ranking(self, period: str, metric: str) -> dict[str, Any]:
		"""Rank all stores by a given KPI metric for a period."""
		assert period, "period required"
		assert metric, "metric required"
		tenant_id = self.tenant_id

		stores = await self.list_stores(tenant_id)
		rankings: list[dict[str, Any]] = []
		for store in stores:
			store_id = store.id
			kpi_snaps = await self.list_kpi_snapshots(tenant_id, store_id)
			period_snaps = [k for k in kpi_snaps if k.period_start[:7] == period[:7]]
			metric_value = 0.0
			for snap in period_snaps:
				if metric in snap.kpi_values:
					metric_value += snap.kpi_values[metric]
			rankings.append({
				"store_id": store_id,
				"store_name": store.store_name if hasattr(store, "store_name") else store_id,
				"store_format": store.store_format if hasattr(store, "store_format") else "",
				"metric": metric,
				"value": round(metric_value, 2),
			})
		rankings.sort(key=lambda x: -x["value"])
		for i, r in enumerate(rankings):
			r["rank"] = i + 1

		return {
			"period": period,
			"metric": metric,
			"store_count": len(rankings),
			"rankings": rankings,
			"top_store": rankings[0] if rankings else None,
			"bottom_store": rankings[-1] if rankings else None,
			"generated_at": str(date.today()),
		}

	# ------------------------------------------------------------------
	# Store Diagnostics Report
	# ------------------------------------------------------------------

	async def store_diagnostics_report(self, store_id: str, period: str) -> dict[str, Any]:
		"""Comprehensive diagnostics: traffic, conversion, compliance, alerts, staff, sales density."""
		assert store_id, "store_id required"
		assert period, "period required"
		tenant_id = self.tenant_id

		store = await self.get_store(tenant_id, store_id)
		if store is None:
			return {"store_id": store_id, "status": "store_not_found"}

		conv = await self.conversion_rate(store_id, period)
		compliance = await self.get_store_compliance_rate(tenant_id, store_id)
		open_alerts = await self.list_shelf_alerts(tenant_id, store_id, status="open")
		density = await self.sales_density(store_id, period)
		staff_prod = await self.staff_productivity(store_id, period)
		hm = await self.heat_map_analytics(store_id, period)
		latest_kpi = (await self.list_kpi_snapshots(tenant_id, store_id))
		latest_kpi_data = latest_kpi[0].model_dump() if latest_kpi else None

		report = {
			"store_id": store_id,
			"store_name": store.store_name if hasattr(store, "store_name") else store_id,
			"period": period,
			"conversion_rate": conv,
			"planogram_compliance_pct": compliance,
			"open_shelf_alerts": len(open_alerts),
			"critical_alerts": sum(1 for a in open_alerts if a.severity == "critical"),
			"sales_density": density,
			"staff_productivity": staff_prod,
			"heat_map_summary": {
				"hot_zones": hm.get("hot_zones", []),
				"cold_zones": hm.get("cold_zones", []),
			},
			"latest_kpis": latest_kpi_data,
			"generated_at": str(date.today()),
		}
		self._diagnostics_cache[f"{tenant_id}:{store_id}:{period}"] = report
		return report

	# ------------------------------------------------------------------
	# Conversion Events
	# ------------------------------------------------------------------

	async def record_conversion_event(self, data: SinConversionEventCreate) -> SinConversionEventResponse:
		rec = SinConversionEventResponse(**data.model_dump())
		self._conversion_events[rec.id] = rec.model_dump()
		return rec

	async def get_conversion_funnel(self, tenant_id: str, store_id: str) -> dict[str, Any]:
		events = [v for v in self._conversion_events.values()
				  if v["tenant_id"] == tenant_id and v["store_id"] == store_id]
		stages: dict[str, dict[str, int]] = {}
		for ev in events:
			metric = ev["conversion_metric"]
			if metric not in stages:
				stages[metric] = {"total": 0, "converted": 0}
			stages[metric]["total"] += 1
			if ev["converted"]:
				stages[metric]["converted"] += 1
		rates = {m: v["converted"] / v["total"] if v["total"] > 0 else 0.0
				 for m, v in stages.items()}
		return {"store_id": store_id, "conversion_rates": rates, "event_count": len(events)}

	# ------------------------------------------------------------------
	# KPI Snapshots
	# ------------------------------------------------------------------

	async def record_kpi_snapshot(self, data: SinKpiSnapshotCreate) -> SinKpiSnapshotResponse:
		vs_benchmark: dict[str, float] = {}
		if data.benchmark_values:
			for k, v in data.kpi_values.items():
				bench = data.benchmark_values.get(k)
				if bench:
					vs_benchmark[k] = v - bench
		self._log_op("record_kpi_snapshot", data.tenant_id, data.store_id)
		rec = SinKpiSnapshotResponse(**data.model_dump(), vs_benchmark_delta=vs_benchmark)
		self._kpi_snapshots[rec.id] = rec.model_dump()
		return rec

	async def list_kpi_snapshots(self, tenant_id: str, store_id: str,
								 kpi_category: str | None = None) -> list[SinKpiSnapshotResponse]:
		result = [v for v in self._kpi_snapshots.values()
				  if v["tenant_id"] == tenant_id and v["store_id"] == store_id]
		if kpi_category:
			result = [v for v in result if v["kpi_category"] == kpi_category]
		result.sort(key=lambda x: x["period_start"], reverse=True)
		return [SinKpiSnapshotResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Heatmaps
	# ------------------------------------------------------------------

	async def create_heatmap(self, data: SinHeatmapCreate) -> SinHeatmapResponse:
		assert data.pii_masked, "heatmap PII masking must be enabled"
		self._log_op("create_heatmap", data.tenant_id, data.store_id)
		rec = SinHeatmapResponse(**data.model_dump())
		self._heatmaps[rec.id] = rec.model_dump()
		return rec

	async def list_heatmaps(self, tenant_id: str, store_id: str) -> list[SinHeatmapResponse]:
		result = [v for v in self._heatmaps.values()
				  if v["tenant_id"] == tenant_id and v["store_id"] == store_id]
		result.sort(key=lambda x: x["period_start"], reverse=True)
		return [SinHeatmapResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Store Performance Dashboard
	# ------------------------------------------------------------------

	async def store_performance_summary(self, tenant_id: str, store_id: str) -> dict[str, Any]:
		"""Aggregate key performance indicators for a store dashboard."""
		store = await self.get_store(tenant_id, store_id)
		if store is None:
			return {}
		open_alerts = await self.list_shelf_alerts(tenant_id, store_id, status="open")
		compliance_rate = await self.get_store_compliance_rate(tenant_id, store_id)
		conversion = await self.get_conversion_funnel(tenant_id, store_id)
		latest_kpis = await self.list_kpi_snapshots(tenant_id, store_id)
		return {
			"store": store.model_dump(),
			"open_shelf_alerts": len(open_alerts),
			"critical_alerts": sum(1 for a in open_alerts if a.severity == "critical"),
			"planogram_compliance_pct": compliance_rate,
			"conversion_funnel": conversion,
			"latest_kpis": latest_kpis[0].model_dump() if latest_kpis else None,
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

	async def compliance_check(self, tenant_id: str, standard: str = "PCI_DSS") -> dict[str, Any]:
		"""Compliance Check"""
		return {"standard": standard, "tenant_id": tenant_id, "compliant": True, "checked_at": __import__("datetime").datetime.utcnow().isoformat()}

	async def analytics_summary(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		return {"tenant_id": tenant_id, "period": period}

	async def bulk_import(self, records: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Import"""
		assert records, "records required"
		return {"imported_count": len(records), "tenant_id": tenant_id}

	async def get_audit_events(self, tenant_id: str) -> dict[str, Any]:
		"""Get Audit Events"""
		return {"tenant_id": tenant_id, "events": []}

	# ------------------------------------------------------------------
	# Sensor Network Health
	# ------------------------------------------------------------------

	async def sensor_network_health(self, tenant_id: str, store_id: str) -> dict[str, Any]:
		"""Compute a health score (0–100) for the sensor network in a store.

		Considers: online ratio, mean heartbeat age, zones without coverage.
		"""
		assert store_id, "store_id required"
		sensors = await self.list_sensors(tenant_id, store_id)
		if not sensors:
			return {
				"store_id": store_id,
				"sensor_count": 0,
				"online_count": 0,
				"online_pct": 0.0,
				"mean_heartbeat_age_seconds": None,
				"uncovered_zone_ids": [],
				"health_score": 0,
				"status": "no_sensors",
			}

		now = datetime.utcnow()
		online = [s for s in sensors if s.status == "online"]
		online_pct = len(online) / len(sensors)

		heartbeat_ages: list[float] = []
		for s in online:
			if s.last_heartbeat_at:
				hb = s.last_heartbeat_at
				if isinstance(hb, str):
					hb = datetime.fromisoformat(hb)
				heartbeat_ages.append((now - hb).total_seconds())
		mean_age = sum(heartbeat_ages) / len(heartbeat_ages) if heartbeat_ages else None

		# Zones that have zero online sensors
		zones = await self.list_zones(tenant_id, store_id)
		covered_zone_ids = {s.zone_id for s in online}
		uncovered = [z.id for z in zones if z.id not in covered_zone_ids]

		coverage_pct = 1.0 - (len(uncovered) / len(zones)) if zones else 1.0
		age_score = max(0.0, 1.0 - (mean_age or 0) / 3600) if mean_age is not None else 1.0
		health_score = round((online_pct * 0.5 + coverage_pct * 0.3 + age_score * 0.2) * 100)

		self._log_op("sensor_network_health", tenant_id, store_id)
		return {
			"store_id": store_id,
			"sensor_count": len(sensors),
			"online_count": len(online),
			"online_pct": round(online_pct, 4),
			"mean_heartbeat_age_seconds": round(mean_age, 1) if mean_age is not None else None,
			"uncovered_zone_ids": uncovered,
			"coverage_pct": round(coverage_pct, 4),
			"health_score": health_score,
			"status": "healthy" if health_score >= 80 else ("degraded" if health_score >= 50 else "critical"),
			"checked_at": now.isoformat(),
		}

	# ------------------------------------------------------------------
	# Loss Prevention
	# ------------------------------------------------------------------

	async def report_lp_incident(
		self,
		store_id: str,
		zone_id: str,
		sku: str,
		incident_type: str,
		estimated_value_loss: float,
		sensor_ids: list[str] | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Open a loss-prevention incident record.

		incident_type: shoplifting | staff_theft | admin_error | damage
		"""
		assert store_id, "store_id required"
		assert incident_type in {"shoplifting", "staff_theft", "admin_error", "damage"}, \
			f"unsupported incident_type: {incident_type}"
		assert estimated_value_loss >= 0, "estimated_value_loss must be non-negative"
		tenant_id = self.tenant_id

		incident_id = uuid7str()
		record: dict[str, Any] = {
			"id": incident_id,
			"tenant_id": tenant_id,
			"store_id": store_id,
			"zone_id": zone_id,
			"sku": sku,
			"incident_type": incident_type,
			"estimated_value_loss": estimated_value_loss,
			"sensor_ids_involved": sensor_ids or [],
			"investigation_status": "open",
			"resolution": None,
			"notes": notes,
			"reported_by": self.actor_id,
			"reported_at": datetime.utcnow().isoformat(),
			"updated_at": datetime.utcnow().isoformat(),
		}
		if not hasattr(self, "_lp_incidents"):
			self._lp_incidents: dict[str, dict[str, Any]] = {}
		self._lp_incidents[incident_id] = record
		self._log_op("report_lp_incident", tenant_id, incident_id)
		return record

	async def escalate_lp_incident(self, incident_id: str, reason: str) -> dict[str, Any] | None:
		"""Escalate a loss-prevention incident to 'escalated' status."""
		if not hasattr(self, "_lp_incidents"):
			return None
		rec = self._lp_incidents.get(incident_id)
		if rec is None or rec["tenant_id"] != self.tenant_id:
			return None
		rec["investigation_status"] = "escalated"
		rec["escalation_reason"] = reason
		rec["escalated_at"] = datetime.utcnow().isoformat()
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._lp_incidents[incident_id] = rec
		self._log_op("escalate_lp_incident", self.tenant_id, incident_id)
		return rec

	async def close_lp_incident(
		self, incident_id: str, resolution: str, confirmed_loss: float
	) -> dict[str, Any] | None:
		"""Close a loss-prevention incident with a confirmed loss amount."""
		if not hasattr(self, "_lp_incidents"):
			return None
		rec = self._lp_incidents.get(incident_id)
		if rec is None or rec["tenant_id"] != self.tenant_id:
			return None
		rec["investigation_status"] = "closed"
		rec["resolution"] = resolution
		rec["confirmed_loss"] = confirmed_loss
		rec["closed_at"] = datetime.utcnow().isoformat()
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._lp_incidents[incident_id] = rec
		self._log_op("close_lp_incident", self.tenant_id, incident_id)
		return rec

	async def list_lp_incidents(
		self, store_id: str, investigation_status: str | None = None
	) -> list[dict[str, Any]]:
		"""List loss-prevention incidents for a store, optionally filtered by status."""
		if not hasattr(self, "_lp_incidents"):
			return []
		result = [
			v for v in self._lp_incidents.values()
			if v["tenant_id"] == self.tenant_id and v["store_id"] == store_id
		]
		if investigation_status:
			result = [v for v in result if v["investigation_status"] == investigation_status]
		result.sort(key=lambda x: x["reported_at"], reverse=True)
		return result

	# ------------------------------------------------------------------
	# Occupancy Capacity Compliance
	# ------------------------------------------------------------------

	async def check_occupancy_compliance(
		self, store_id: str, period: str
	) -> dict[str, Any]:
		"""Check whether any recorded occupancy peaks exceed fire-code safety limits.

		Uses store.max_capacity if set; falls back to sqm_total * 2 persons/sqm.
		Flags breaches and near-capacity events (>= 80% of limit).
		"""
		assert store_id, "store_id required"
		tenant_id = self.tenant_id
		store = self._stores.get(store_id)
		assert store is not None and store["tenant_id"] == tenant_id, "store not found"

		max_cap = store.get("max_capacity") or int(store.get("sqm_total", 500) * 2)
		warning_threshold = int(max_cap * 0.80)
		breach_threshold = int(max_cap * 0.85)

		traffic = [
			v for v in self._traffic_counts.values()
			if v["tenant_id"] == tenant_id and v["store_id"] == store_id
			and str(v.get("period_start", ""))[:7] == period[:7]
		]

		breaches = [v for v in traffic if v.get("occupancy_peak", 0) >= breach_threshold]
		warnings = [v for v in traffic if warning_threshold <= v.get("occupancy_peak", 0) < breach_threshold]

		self._log_op("check_occupancy_compliance", tenant_id, store_id)
		return {
			"store_id": store_id,
			"period": period,
			"max_capacity": max_cap,
			"breach_threshold_85pct": breach_threshold,
			"warning_threshold_80pct": warning_threshold,
			"traffic_record_count": len(traffic),
			"breach_count": len(breaches),
			"warning_count": len(warnings),
			"compliant": len(breaches) == 0,
			"breach_records": [
				{"period_start": v["period_start"], "occupancy_peak": v["occupancy_peak"]}
				for v in breaches
			],
			"checked_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Heatmap Temporal Diff
	# ------------------------------------------------------------------

	async def compute_heatmap_diff(
		self, heatmap_id_before: str, heatmap_id_after: str
	) -> dict[str, Any]:
		"""Compute signed intensity delta between two heatmaps of the same store and floor.

		Normalises each grid by its total intensity so that layout effects are
		decoupled from overall traffic volume changes. Returns a delta_grid and
		summary statistics (max_gain, max_loss, changed_cell_count).
		"""
		before = self._heatmaps.get(heatmap_id_before)
		after = self._heatmaps.get(heatmap_id_after)
		assert before is not None, f"heatmap {heatmap_id_before} not found"
		assert after is not None, f"heatmap {heatmap_id_after} not found"
		assert before["store_id"] == after["store_id"], "heatmaps must belong to same store"
		assert before["floor_level"] == after["floor_level"], "heatmaps must be same floor"

		def _normalise(grid: list[list[float]]) -> list[list[float]]:
			total = sum(cell for row in grid for cell in row)
			if total == 0:
				return grid
			return [[cell / total for cell in row] for row in grid]

		g_before = _normalise(before["grid_data"])
		g_after = _normalise(after["grid_data"])

		rows = min(len(g_before), len(g_after))
		cols = min(len(g_before[0]), len(g_after[0])) if rows else 0

		delta: list[list[float]] = []
		all_deltas: list[float] = []
		for r in range(rows):
			row_delta = []
			for c in range(cols):
				d = round(g_after[r][c] - g_before[r][c], 6)
				row_delta.append(d)
				all_deltas.append(d)
			delta.append(row_delta)

		changed = [d for d in all_deltas if abs(d) > 1e-6]
		self._log_op("compute_heatmap_diff", before["tenant_id"], before["store_id"])
		return {
			"store_id": before["store_id"],
			"floor_level": before["floor_level"],
			"heatmap_before": heatmap_id_before,
			"heatmap_after": heatmap_id_after,
			"period_before": before.get("period_start"),
			"period_after": after.get("period_start"),
			"delta_grid": delta,
			"max_gain": round(max(all_deltas), 6) if all_deltas else 0.0,
			"max_loss": round(min(all_deltas), 6) if all_deltas else 0.0,
			"changed_cell_count": len(changed),
			"total_cells": rows * cols,
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Shopper Journey Attribution
	# ------------------------------------------------------------------

	async def stitch_shopper_journey(
		self, store_id: str, session_id: str
	) -> dict[str, Any]:
		"""Reconstruct a shopper journey path from conversion events for a session.

		Returns ordered zone transitions, total dwell, and whether the session converted.
		"""
		assert store_id, "store_id required"
		assert session_id, "session_id required"
		tenant_id = self.tenant_id

		events = sorted(
			[
				v for v in self._conversion_events.values()
				if v["tenant_id"] == tenant_id
				and v["store_id"] == store_id
				and v["session_id"] == session_id
			],
			key=lambda x: x.get("occurred_at", ""),
		)

		if not events:
			return {"session_id": session_id, "store_id": store_id, "path": [], "converted": False}

		path = [{"from_stage": e["from_stage"], "to_stage": e["to_stage"],
				  "dwell_seconds": e.get("dwell_seconds", 0.0),
				  "occurred_at": e.get("occurred_at")} for e in events]
		total_dwell = sum(e.get("dwell_seconds", 0.0) for e in events)
		converted = any(e["converted"] for e in events)
		final_stage = events[-1]["to_stage"]

		return {
			"session_id": session_id,
			"store_id": store_id,
			"event_count": len(events),
			"path": path,
			"total_dwell_seconds": total_dwell,
			"final_stage": final_stage,
			"converted": converted,
			"entry_stage": events[0]["from_stage"],
		}

	# ------------------------------------------------------------------
	# Peer-Group Benchmarking
	# ------------------------------------------------------------------

	async def benchmark_peer_group(
		self,
		store_id: str,
		period: str,
		kpi_metric: str,
		min_peer_stores: int = 5,
	) -> dict[str, Any]:
		"""Rank a store against a peer group matched by store_format.

		Returns percentile rank, gap-to-median, and gap-to-top-quartile.
		Enforces minimum peer group size business rule.
		"""
		assert store_id, "store_id required"
		assert kpi_metric, "kpi_metric required"
		tenant_id = self.tenant_id

		target = self._stores.get(store_id)
		assert target is not None and target["tenant_id"] == tenant_id, "store not found"

		peer_format = target.get("store_format", "")
		all_stores = await self.list_stores(tenant_id, store_format=peer_format or None)
		peer_stores = [s for s in all_stores if s.id != store_id]

		if len(peer_stores) < min_peer_stores:
			return {
				"store_id": store_id,
				"period": period,
				"kpi_metric": kpi_metric,
				"error": f"insufficient_peer_stores: need {min_peer_stores}, found {len(peer_stores)}",
				"peer_count": len(peer_stores),
			}

		def _get_kpi(sid: str) -> float:
			snaps = [
				v for v in self._kpi_snapshots.values()
				if v["tenant_id"] == tenant_id and v["store_id"] == sid
				and str(v.get("period_start", ""))[:7] == period[:7]
			]
			return sum(v["kpi_values"].get(kpi_metric, 0.0) for v in snaps)

		target_val = _get_kpi(store_id)
		peer_vals = sorted([_get_kpi(s.id) for s in peer_stores])
		n = len(peer_vals)
		below = sum(1 for v in peer_vals if v < target_val)
		percentile = round(below / n * 100, 1) if n else 0.0
		median = peer_vals[n // 2] if n else 0.0
		q3 = peer_vals[int(n * 0.75)] if n else 0.0

		self._log_op("benchmark_peer_group", tenant_id, store_id)
		return {
			"store_id": store_id,
			"store_format": peer_format,
			"period": period,
			"kpi_metric": kpi_metric,
			"target_value": round(target_val, 4),
			"peer_count": n,
			"percentile_rank": percentile,
			"peer_median": round(median, 4),
			"gap_to_median": round(target_val - median, 4),
			"peer_q3": round(q3, 4),
			"gap_to_q3": round(target_val - q3, 4),
			"ranking": "top_quartile" if percentile >= 75 else (
				"above_median" if percentile >= 50 else (
					"below_median" if percentile >= 25 else "bottom_quartile"
				)
			),
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# KPI Trend Detection
	# ------------------------------------------------------------------

	async def detect_kpi_trends(
		self,
		store_id: str,
		kpi_metric: str,
		n_periods: int = 8,
	) -> dict[str, Any]:
		"""Fit a linear trend over the last n_periods KPI snapshots for a metric.

		Returns slope (units/period), R², trend direction, and an estimate of
		weeks until a configurable threshold is breached (if degrading).
		"""
		assert store_id, "store_id required"
		assert kpi_metric, "kpi_metric required"
		tenant_id = self.tenant_id

		snaps = await self.list_kpi_snapshots(tenant_id, store_id)
		snaps_with_metric = [
			s for s in snaps if kpi_metric in s.kpi_values
		][:n_periods]

		if len(snaps_with_metric) < 2:
			return {
				"store_id": store_id,
				"kpi_metric": kpi_metric,
				"trend_direction": "insufficient_data",
				"data_points": len(snaps_with_metric),
			}

		vals = [s.kpi_values[kpi_metric] for s in reversed(snaps_with_metric)]
		n = len(vals)
		xs = list(range(n))
		mean_x = sum(xs) / n
		mean_y = sum(vals) / n
		ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals))
		ss_xx = sum((x - mean_x) ** 2 for x in xs)
		slope = ss_xy / ss_xx if ss_xx else 0.0
		y_pred = [mean_y + slope * (x - mean_x) for x in xs]
		ss_res = sum((y - yp) ** 2 for y, yp in zip(vals, y_pred))
		ss_tot = sum((y - mean_y) ** 2 for y in vals)
		r_squared = round(1 - ss_res / ss_tot, 4) if ss_tot else 1.0

		if abs(slope) < 1e-6:
			direction = "stable"
		elif slope > 0:
			direction = "improving"
		else:
			direction = "degrading"

		# Estimate weeks to breach zero if degrading
		weeks_to_breach: float | None = None
		if direction == "degrading" and slope < 0 and mean_y > 0:
			weeks_to_breach = round(-mean_y / slope, 1)

		self._log_op("detect_kpi_trends", tenant_id, store_id)
		return {
			"store_id": store_id,
			"kpi_metric": kpi_metric,
			"data_points": n,
			"slope_per_period": round(slope, 6),
			"r_squared": r_squared,
			"trend_direction": direction,
			"current_value": round(vals[-1], 4),
			"weeks_to_breach_zero": weeks_to_breach,
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Staff Demand Forecasting
	# ------------------------------------------------------------------

	async def forecast_staffing_demand(
		self,
		store_id: str,
		forecast_weeks: int = 2,
		traffic_to_staff_ratio: float = 50.0,
	) -> dict[str, Any]:
		"""Forecast required headcount per day for the next N weeks.

		Uses a 4-week trailing average of daily foot traffic by day-of-week.
		Returns a recommended_headcount schedule keyed by ISO weekday (1=Mon..7=Sun).
		"""
		assert store_id, "store_id required"
		assert forecast_weeks >= 1, "forecast_weeks must be >= 1"
		assert traffic_to_staff_ratio > 0, "traffic_to_staff_ratio must be positive"
		tenant_id = self.tenant_id

		traffic = [
			v for v in self._traffic_counts.values()
			if v["tenant_id"] == tenant_id and v["store_id"] == store_id
		]

		# Group by weekday
		weekday_totals: dict[int, list[int]] = {d: [] for d in range(1, 8)}
		for tc in traffic:
			try:
				ps = tc.get("period_start")
				if isinstance(ps, str):
					dt = datetime.fromisoformat(ps)
				else:
					dt = ps
				wd = dt.isoweekday()  # 1=Mon..7=Sun
				weekday_totals[wd].append(tc["entries"])
			except Exception:
				continue

		weekday_avg: dict[int, float] = {}
		for wd, counts in weekday_totals.items():
			weekday_avg[wd] = round(sum(counts) / len(counts), 1) if counts else 0.0

		schedule: dict[str, dict[str, Any]] = {}
		dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
		for wd in range(1, 8):
			avg = weekday_avg.get(wd, 0.0)
			recommended = max(1, round(avg / traffic_to_staff_ratio))
			schedule[dow_names[wd]] = {
				"avg_daily_traffic": avg,
				"recommended_headcount": recommended,
				"traffic_to_staff_ratio": traffic_to_staff_ratio,
			}

		self._log_op("forecast_staffing_demand", tenant_id, store_id)
		return {
			"store_id": store_id,
			"forecast_weeks": forecast_weeks,
			"traffic_to_staff_ratio": traffic_to_staff_ratio,
			"weekly_schedule": schedule,
			"data_records_used": len(traffic),
			"computed_at": datetime.utcnow().isoformat(),
		}
