"""Standalone async API helpers for APG advanced CRM.

All functions are plain async callables — no web framework required.  They
accept a ``service`` (CRMService) and explicit ``tenant_id`` / ``user_id``
parameters so they can be called directly from tests or wired into any HTTP
framework as thin adapters.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CRMService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import CRMService  # type: ignore


# ── context resolution helpers ───────────────────────────────────────────────

Request = Any  # type alias so `request: Request` annotations are valid standalone


def _clean_text(value: Any) -> str:
	return str(value or "").strip()


async def get_current_user(request: Request, credentials: Any = None) -> dict[str, Any]:
	"""Resolve user context: state → headers → env → default, returns dict."""
	state = getattr(request, "state", None)
	if state:
		user = getattr(state, "current_user", None)
		if isinstance(user, dict) and user.get("user_id"):
			headers_inner = getattr(request, "headers", {}) or {}
			def _h(k: str) -> str:
				return _clean_text((headers_inner.get(k) if isinstance(headers_inner, dict) else getattr(headers_inner, k, "")) or "")
			return {
				"user_id": _clean_text(user["user_id"]),
				"tenant_id": _clean_text(user.get("tenant_id") or _h("X-APG-Tenant-ID") or os.environ.get("APG_DEFAULT_TENANT_ID", "default")),
				"roles": user.get("roles", ["crm_user"]),
			}
	headers = getattr(request, "headers", {}) or {}
	def _hdr(key: str) -> str:
		if isinstance(headers, dict):
			return _clean_text(headers.get(key, ""))
		return _clean_text(getattr(headers, key, "") or "")
	user_id = _hdr("X-APG-User-ID") or _hdr("X-User-ID") or os.environ.get("APG_DEFAULT_USER_ID", "anonymous")
	tenant_id = _hdr("X-APG-Tenant-ID") or _hdr("X-Tenant-ID") or os.environ.get("APG_DEFAULT_TENANT_ID", "default")
	roles_raw = _hdr("X-APG-Roles")
	roles = [r.strip() for r in roles_raw.split(",") if r.strip()] if roles_raw else ["crm_user"]
	return {"user_id": user_id, "tenant_id": tenant_id, "roles": roles}


def get_tenant_id(request: Request, credentials: Any = None) -> str:
	"""Resolve tenant_id: state → X-APG-Tenant-ID header → APG_DEFAULT_TENANT_ID env."""
	state = getattr(request, "state", None)
	if state:
		user = getattr(state, "current_user", None)
		if isinstance(user, dict) and user.get("tenant_id"):
			return _clean_text(user["tenant_id"])
	headers = getattr(request, "headers", {}) or {}
	for key in ("X-APG-Tenant-ID", "X-Tenant-ID"):
		val = (headers.get(key) if isinstance(headers, dict) else getattr(headers, key, None))
		if val:
			return _clean_text(val)
	return os.environ.get("APG_DEFAULT_TENANT_ID", "default")


# ---------------------------------------------------------------------------
# Minimal Flask-like app shim so `from capabilities.crm.adv.api import app`
# works in integration contexts that expect an WSGI/ASGI object.
# ---------------------------------------------------------------------------

class _MinimalApp:
	"""Lightweight app placeholder — real Flask-AppBuilder app lives in app.py."""
	name = "crm_adv"


app = _MinimalApp()


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

@dataclass
class APIResponse:
	"""Thin response container returned by every API function."""
	data: Any = None
	error: str | None = None
	uptime_seconds: float = 0.0

	@property
	def body(self) -> bytes:
		"""Serialise data as JSON bytes (mirrors starlette Response.body)."""
		return json.dumps(self.data, default=_json_default).encode()


def _json_default(obj: Any) -> Any:
	if isinstance(obj, Decimal):
		return str(obj)
	if isinstance(obj, datetime):
		return obj.isoformat()
	raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Request dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImportRequest:
	file_format: str = "csv"
	file_data: str = ""
	mapping_type: str | None = None
	deduplicate: bool = True


@dataclass
class ExportRequest:
	export_format: str = "json"
	include_fields: list[str] | None = None
	exclude_fields: list[str] | None = None
	contact_ids: list[str] | None = None


@dataclass
class ClockInRequest:
	location: dict[str, Any] | None = None
	device_info: dict[str, Any] | None = None
	notes: str = ""


# ---------------------------------------------------------------------------
# Service uptime tracker
# ---------------------------------------------------------------------------

_START_TIME = time.monotonic()


# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------

async def get_leads(
	search_term: str | None = None,
	lead_source: Any = None,
	lead_status: Any = None,
	owner_id: str | None = None,
	page: int = 1,
	page_size: int = 50,
	service: CRMService | None = None,
	tenant_id: str = "default",
) -> APIResponse:
	svc = service or CRMService()
	filters: dict[str, Any] = {}
	if lead_source is not None:
		filters["lead_source"] = lead_source
	if lead_status is not None:
		filters["lead_status"] = lead_status
	if owner_id is not None:
		filters["owner_id"] = owner_id
	result = await svc.db_manager.list_leads(
		tenant_id,
		filters=filters,
		search_term=search_term,
		page=page,
		page_size=page_size,
	)
	items = [_lead_to_dict(r) for r in result["items"]]
	return APIResponse(data={"items": items, "total_count": result["total_count"]})


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

async def get_accounts(
	search_term: str | None = None,
	account_type: Any = None,
	owner_id: str | None = None,
	page: int = 1,
	page_size: int = 50,
	service: CRMService | None = None,
	tenant_id: str = "default",
) -> APIResponse:
	svc = service or CRMService()
	filters: dict[str, Any] = {}
	if account_type is not None:
		filters["account_type"] = account_type
	if owner_id is not None:
		filters["account_owner_id"] = owner_id
	result = await svc.db_manager.list_accounts(
		tenant_id,
		filters=filters,
		search_term=search_term,
		page=page,
		page_size=page_size,
	)
	items = [_account_to_dict(r) for r in result["items"]]
	return APIResponse(data={"items": items, "total_count": result["total_count"]})


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------

async def get_opportunities(
	search_term: str | None = None,
	stage: Any = None,
	account_id: str | None = None,
	owner_id: str | None = None,
	page: int = 1,
	page_size: int = 50,
	service: CRMService | None = None,
	tenant_id: str = "default",
) -> APIResponse:
	svc = service or CRMService()
	filters: dict[str, Any] = {}
	if stage is not None:
		filters["stage"] = stage
	if account_id is not None:
		filters["account_id"] = account_id
	if owner_id is not None:
		filters["owner_id"] = owner_id
	result = await svc.db_manager.list_opportunities(
		tenant_id,
		filters=filters,
		search_term=search_term,
		page=page,
		page_size=page_size,
	)
	items = [_opportunity_to_dict(r) for r in result["items"]]
	return APIResponse(data={"items": items, "total_count": result["total_count"]})


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

async def get_activities(
	search_term: str | None = None,
	activity_type: Any = None,
	related_to_type: str | None = None,
	related_to_id: str | None = None,
	assigned_to_id: str | None = None,
	page: int = 1,
	page_size: int = 50,
	service: CRMService | None = None,
	tenant_id: str = "default",
) -> APIResponse:
	svc = service or CRMService()
	filters: dict[str, Any] = {}
	if activity_type is not None:
		filters["activity_type"] = activity_type
	if related_to_type is not None:
		filters["related_to_type"] = related_to_type
	if related_to_id is not None:
		filters["related_to_id"] = related_to_id
	if assigned_to_id is not None:
		filters["assigned_to_id"] = assigned_to_id
	result = await svc.db_manager.list_activities(
		tenant_id,
		filters=filters,
		search_term=search_term,
		page=page,
		page_size=page_size,
	)
	items = [_activity_to_dict(r) for r in result["items"]]
	return APIResponse(data={"items": items, "total_count": result["total_count"]})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health_check(service: CRMService | None = None) -> APIResponse:
	uptime = time.monotonic() - _START_TIME
	resp = APIResponse(data={"status": "ok"})
	resp.uptime_seconds = uptime
	return resp


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

async def get_dashboard(
	service: CRMService | None = None,
	tenant_id: str = "default",
	user_id: str = "anonymous",
) -> APIResponse:
	svc = service or CRMService()
	db = svc.db_manager

	opps_result = await db.list_opportunities(tenant_id, page=1, page_size=10_000)
	opps = opps_result["items"]

	pipeline_value = sum(o.amount for o in opps)
	weighted_value = sum(
		o.amount * Decimal(str(o.probability)) / Decimal("100") for o in opps
	)
	stage_breakdown: dict[str, int] = {}
	for o in opps:
		key = o.stage.value if hasattr(o.stage, "value") else str(o.stage)
		stage_breakdown[key] = stage_breakdown.get(key, 0) + 1

	record_counts = {
		"contacts": db.count_contacts(tenant_id),
		"accounts": db.count_accounts(tenant_id),
		"leads": db.count_leads(tenant_id),
		"opportunities": db.count_opportunities(tenant_id),
		"activities": db.count_activities(tenant_id),
	}

	return APIResponse(data={
		"record_counts": record_counts,
		"pipeline_value": str(pipeline_value),
		"weighted_pipeline_value": str(weighted_value),
		"opportunity_stage_breakdown": stage_breakdown,
	})


# ---------------------------------------------------------------------------
# Pipeline analytics
# ---------------------------------------------------------------------------

async def get_pipeline_analytics(
	service: CRMService | None = None,
	tenant_id: str = "default",
	user_id: str = "anonymous",
) -> APIResponse:
	svc = service or CRMService()
	db = svc.db_manager

	opps_result = await db.list_opportunities(tenant_id, page=1, page_size=10_000)
	opps = opps_result["items"]

	stage_breakdown: dict[str, int] = {}
	for o in opps:
		key = o.stage.value if hasattr(o.stage, "value") else str(o.stage)
		stage_breakdown[key] = stage_breakdown.get(key, 0) + 1

	weighted_value = sum(
		o.amount * Decimal(str(o.probability)) / Decimal("100") for o in opps
	)

	return APIResponse(data={
		"opportunity_count": len(opps),
		"stage_breakdown": stage_breakdown,
		"weighted_pipeline_value": str(weighted_value),
	})


# ---------------------------------------------------------------------------
# Clock-in / time entry
# ---------------------------------------------------------------------------

async def clock_in(
	clock_in_data: ClockInRequest,
	service: CRMService | None = None,
	tenant_id: str = "default",
	user_id: str = "anonymous",
) -> APIResponse:
	svc = service or CRMService()
	entry = {
		"tenant_id": tenant_id,
		"user_id": user_id,
		"status": "clocked_in",
		"location": clock_in_data.location,
		"device_info": clock_in_data.device_info,
		"notes": clock_in_data.notes,
		"clocked_in_at": datetime.now(timezone.utc).isoformat(),
	}
	bucket = svc._time_entries.setdefault(tenant_id, [])
	bucket.append(entry)
	return APIResponse(data=entry)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

async def get_metrics(
	service: CRMService | None = None,
	tenant_id: str = "default",
) -> APIResponse:
	svc = service or CRMService()
	db = svc.db_manager
	time_entries = len(svc._time_entries.get(tenant_id, []))
	return APIResponse(data={
		"record_counts": {
			"contacts": db.count_contacts(tenant_id),
			"accounts": db.count_accounts(tenant_id),
			"leads": db.count_leads(tenant_id),
			"opportunities": db.count_opportunities(tenant_id),
			"activities": db.count_activities(tenant_id),
			"time_entries": time_entries,
		}
	})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

async def get_configuration(
	service: CRMService | None = None,
	tenant_id: str = "default",
) -> APIResponse:
	svc = service or CRMService()
	return APIResponse(data=dict(svc._get_config(tenant_id)))


async def update_configuration(
	update_data: dict[str, Any],
	service: CRMService | None = None,
	tenant_id: str = "default",
	user_id: str = "anonymous",
) -> APIResponse:
	svc = service or CRMService()
	cfg = svc._get_config(tenant_id)
	cfg.update(update_data)
	return APIResponse(data=dict(cfg))


# ---------------------------------------------------------------------------
# Import / export
# ---------------------------------------------------------------------------

async def import_contacts(
	import_request: ImportRequest,
	service: CRMService | None = None,
	tenant_id: str = "default",
	user_id: str = "anonymous",
) -> APIResponse:
	svc = service or CRMService()
	from .import_export import ContactImportExportManager
	manager = ContactImportExportManager(svc.db_manager, tenant_id)
	stats = await manager.import_contacts(
		file_data=import_request.file_data,
		file_format=import_request.file_format,
		mapping_config=None,
		deduplicate=import_request.deduplicate,
		validate_data=True,
		created_by=user_id,
	)
	return APIResponse(data=stats)


async def export_contacts(
	export_request: ExportRequest,
	service: CRMService | None = None,
	tenant_id: str = "default",
) -> APIResponse:
	svc = service or CRMService()
	from .import_export import ContactImportExportManager
	manager = ContactImportExportManager(svc.db_manager, tenant_id)
	data, _filename = await manager.export_contacts(
		export_format=export_request.export_format,
		contact_ids=export_request.contact_ids,
		include_fields=export_request.include_fields,
		exclude_fields=export_request.exclude_fields,
	)
	# data is a JSON string when export_format=="json"
	if isinstance(data, str):
		raw = json.loads(data)
	else:
		raw = {"data": data.decode("utf-8") if isinstance(data, bytes) else data}
	resp = APIResponse(data=raw)
	# store raw bytes so .body works correctly
	resp._raw_body = data.encode("utf-8") if isinstance(data, str) else data
	return resp


async def get_import_template(
	file_format: str = "csv",
	mapping_type: str | None = None,
	service: CRMService | None = None,
) -> APIResponse:
	svc = service or CRMService()
	from .import_export import ContactImportExportManager
	manager = ContactImportExportManager(svc.db_manager, "default")
	data, _filename = await manager.get_import_template(
		file_format=file_format,
		mapping_type=mapping_type,
	)
	resp = APIResponse(data=data)
	resp._raw_body = data.encode("utf-8") if isinstance(data, str) else data
	return resp


# ---------------------------------------------------------------------------
# Patch APIResponse.body to use _raw_body when present
# ---------------------------------------------------------------------------

_original_body = APIResponse.body.fget  # type: ignore[union-attr]


def _body_property(self: APIResponse) -> bytes:
	raw = getattr(self, "_raw_body", None)
	if raw is not None:
		return raw if isinstance(raw, bytes) else raw.encode("utf-8")
	return json.dumps(self.data, default=_json_default).encode()


APIResponse.body = property(_body_property)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _lead_to_dict(lead: Any) -> dict[str, Any]:
	d = lead.model_dump()
	# normalise enum values to their .value strings
	for k, v in d.items():
		if hasattr(v, "value"):
			d[k] = v.value
	return d


def _account_to_dict(account: Any) -> dict[str, Any]:
	d = account.model_dump()
	for k, v in d.items():
		if hasattr(v, "value"):
			d[k] = v.value
	return d


def _opportunity_to_dict(opp: Any) -> dict[str, Any]:
	d = opp.model_dump()
	for k, v in d.items():
		if hasattr(v, "value"):
			d[k] = v.value
	# expected_revenue must be a string matching Decimal format
	if d.get("expected_revenue") is not None:
		d["expected_revenue"] = str(d["expected_revenue"])
	return d


def _activity_to_dict(activity: Any) -> dict[str, Any]:
	d = activity.model_dump()
	for k, v in d.items():
		if hasattr(v, "value"):
			d[k] = v.value
	return d


# ---------------------------------------------------------------------------
# Backward-compat thin wrappers (synchronous, module-level service)
# ---------------------------------------------------------------------------

_SERVICE = CRMService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
	}


def service() -> CRMService:
	return _SERVICE
