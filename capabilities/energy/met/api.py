"""REST API for APG Smart Metering & AMI."""

from __future__ import annotations

from typing import Any

try:
	from .service import SmartMeteringService
	from .views import (
		agent_workbench_model, dashboard_model, data_quality_model,
		demand_response_model, head_end_model, interval_data_model,
		meter_detail_model, meter_registry_model, remote_command_model,
		tamper_alert_model,
	)
except ImportError:
	from service import SmartMeteringService  # type: ignore
	from views import (  # type: ignore
		agent_workbench_model, dashboard_model, data_quality_model,
		demand_response_model, head_end_model, interval_data_model,
		meter_detail_model, meter_registry_model, remote_command_model,
		tamper_alert_model,
	)

_SERVICE = SmartMeteringService()


def _ok(data: Any) -> dict[str, Any]:
	return {"status": "ok", "data": data}


def _err(reason: str, code: int = 400) -> dict[str, Any]:
	return {"status": "error", "code": code, "reason": reason}


def _tenant(payload: dict[str, Any]) -> str:
	return payload.get("tenant_id", "default")


def get_contract(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/contract"""
	return _ok(_SERVICE.describe(_tenant(payload)))


def get_dashboard(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/dashboard"""
	return _ok(dashboard_model(_SERVICE, _tenant(payload)))


# ── meters ────────────────────────────────────────────────────────────────────

def list_meters(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/meters"""
	return _ok(meter_registry_model(_SERVICE, _tenant(payload)))


def create_meter(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/meters"""
	try:
		result = _SERVICE.register_meter(
			meter_id=payload["meter_id"],
			tenant_id=_tenant(payload),
			serial_number=payload["serial_number"],
			meter_type=payload["meter_type"],
			communication_technology=payload["communication_technology"],
			customer_id=payload["customer_id"],
			location_reference=payload.get("location_reference", ""),
			installed_at=payload["installed_at"],
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_meter(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/meters/<id>"""
	try:
		return _ok(meter_detail_model(_SERVICE, _tenant(payload), payload["meter_id"]))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


def update_meter_status(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-met/api/v1/meters/<id>/status"""
	try:
		return _ok(_SERVICE.update_meter_status(
			meter_id=payload["meter_id"],
			tenant_id=_tenant(payload),
			new_status=payload["status"],
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── readings ──────────────────────────────────────────────────────────────────

def list_readings(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/readings"""
	return _ok(interval_data_model(_SERVICE, _tenant(payload)))


def submit_reading(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/readings"""
	try:
		result = _SERVICE.submit_reading(
			reading_id=payload["reading_id"],
			tenant_id=_tenant(payload),
			meter_id=payload["meter_id"],
			reading_type=payload["reading_type"],
			interval_length=payload["interval_length"],
			interval_start=payload["interval_start"],
			interval_end=payload["interval_end"],
			value=float(payload["value"]),
			unit=payload["unit"],
			quality_flag=payload.get("quality_flag", "valid"),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── tamper ────────────────────────────────────────────────────────────────────

def list_tamper(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/tamper"""
	return _ok(tamper_alert_model(_SERVICE, _tenant(payload)))


def report_tamper(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/tamper"""
	try:
		result = _SERVICE.report_tamper(
			tamper_id=payload["tamper_id"],
			tenant_id=_tenant(payload),
			meter_id=payload["meter_id"],
			tamper_type=payload["tamper_type"],
			evidence_reference=payload.get("evidence_reference", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def resolve_tamper(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-met/api/v1/tamper/<id>/resolve"""
	try:
		return _ok(_SERVICE.resolve_tamper(
			tamper_id=payload["tamper_id"],
			tenant_id=_tenant(payload),
			investigated_by=payload.get("investigated_by", ""),
			notes=payload.get("notes", ""),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── remote commands ───────────────────────────────────────────────────────────

def list_commands(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/commands"""
	return _ok(remote_command_model(_SERVICE, _tenant(payload)))


def issue_command(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/commands"""
	try:
		result = _SERVICE.issue_command(
			command_id=payload["command_id"],
			tenant_id=_tenant(payload),
			meter_id=payload["meter_id"],
			command_type=payload["command_type"],
			issued_by=payload["issued_by"],
			approved_by=payload.get("approved_by", ""),
			parameters=payload.get("parameters", {}),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def acknowledge_command(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-met/api/v1/commands/<id>/acknowledge"""
	try:
		return _ok(_SERVICE.acknowledge_command(payload["command_id"], _tenant(payload)))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


def complete_command(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-met/api/v1/commands/<id>/complete"""
	try:
		return _ok(_SERVICE.complete_command(payload["command_id"], _tenant(payload)))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


# ── demand response ───────────────────────────────────────────────────────────

def list_dr_events(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/demand-response"""
	return _ok(demand_response_model(_SERVICE, _tenant(payload)))


def create_dr_event(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/demand-response"""
	try:
		result = _SERVICE.create_dr_event(
			dr_id=payload["dr_id"],
			tenant_id=_tenant(payload),
			event_type=payload["event_type"],
			target_reduction_kw=float(payload["target_reduction_kw"]),
			start_time=payload["start_time"],
			end_time=payload["end_time"],
			meter_ids=payload.get("meter_ids", []),
			created_by=payload.get("created_by", "system"),
			notification_sent=payload.get("notification_sent", True),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def opt_out_meter(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/demand-response/<id>/opt-out"""
	try:
		return _ok(_SERVICE.opt_out_meter(
			payload["dr_id"], _tenant(payload), payload["meter_id"],
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def complete_dr_event(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-met/api/v1/demand-response/<id>/complete"""
	try:
		return _ok(_SERVICE.complete_dr_event(
			payload["dr_id"], _tenant(payload),
			actual_reduction_kw=float(payload.get("actual_reduction_kw", 0)),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── data quality ──────────────────────────────────────────────────────────────

def list_quality_flags(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/data-quality"""
	return _ok(data_quality_model(_SERVICE, _tenant(payload)))


def set_quality_flag(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/data-quality"""
	try:
		result = _SERVICE.set_quality_flag(
			flag_id=payload["flag_id"],
			tenant_id=_tenant(payload),
			reading_id=payload["reading_id"],
			meter_id=payload["meter_id"],
			quality_flag=payload["quality_flag"],
			reason=payload.get("reason", ""),
			flagged_by=payload.get("flagged_by", "system"),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── head end ──────────────────────────────────────────────────────────────────

def get_head_end(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/head-end"""
	return _ok(head_end_model(_SERVICE, _tenant(payload)))


def update_head_end(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/head-end"""
	try:
		result = _SERVICE.update_head_end_status(
			he_id=payload["he_id"],
			tenant_id=_tenant(payload),
			head_end_name=payload["head_end_name"],
			protocol=payload["protocol"],
			connected_meters=int(payload["connected_meters"]),
			total_meters=int(payload["total_meters"]),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")


# ── agents ────────────────────────────────────────────────────────────────────

def list_agents(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-met/api/v1/agents"""
	return _ok(agent_workbench_model(_SERVICE, _tenant(payload)))


def register_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-met/api/v1/agents"""
	try:
		result = _SERVICE.register_agent(
			agent_id=payload["agent_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			runtime=payload["runtime"],
			role=payload["role"],
			scope=payload.get("scope", "smart metering operations"),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def service() -> SmartMeteringService:
	return _SERVICE
