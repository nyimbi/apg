"""REST API for APG Distribution Network."""

from __future__ import annotations

from typing import Any

try:
	from .service import DistributionNetworkService
	from .views import (
		agent_workbench_model, dashboard_model, fault_detail_model,
		fault_management_model, load_balancing_model, outage_manager_model,
		reliability_kpi_model, scada_console_model, switching_orders_model,
		topology_model,
	)
except ImportError:
	from service import DistributionNetworkService  # type: ignore
	from views import (  # type: ignore
		agent_workbench_model, dashboard_model, fault_detail_model,
		fault_management_model, load_balancing_model, outage_manager_model,
		reliability_kpi_model, scada_console_model, switching_orders_model,
		topology_model,
	)

_SERVICE = DistributionNetworkService()


def _ok(data: Any) -> dict[str, Any]:
	return {"status": "ok", "data": data}


def _err(reason: str, code: int = 400) -> dict[str, Any]:
	return {"status": "error", "code": code, "reason": reason}


def _tenant(payload: dict[str, Any]) -> str:
	return payload.get("tenant_id", "default")


def get_contract(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/contract"""
	return _ok(_SERVICE.describe(_tenant(payload)))


def get_dashboard(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/dashboard"""
	return _ok(dashboard_model(_SERVICE, _tenant(payload)))


# ── topology ──────────────────────────────────────────────────────────────────

def get_topology(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/topology"""
	return _ok(topology_model(_SERVICE, _tenant(payload)))


def create_feeder(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/feeders"""
	try:
		result = _SERVICE.register_feeder(
			feeder_id=payload["feeder_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			substation_id=payload["substation_id"],
			voltage_level=payload["voltage_level"],
			normal_capacity_mw=float(payload["normal_capacity_mw"]),
			emergency_capacity_mw=float(payload.get("emergency_capacity_mw", 0)),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def list_feeders(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/feeders"""
	return _ok({"feeders": _SERVICE.list_feeders(_tenant(payload))})


def create_element(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/elements"""
	try:
		result = _SERVICE.register_element(
			element_id=payload["element_id"],
			tenant_id=_tenant(payload),
			element_type=payload["element_type"],
			name=payload["name"],
			feeder_id=payload["feeder_id"],
			voltage_level=payload["voltage_level"],
			location_reference=payload.get("location_reference", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def list_elements(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/elements"""
	feeder_id = payload.get("feeder_id")
	return _ok({"elements": _SERVICE.list_elements(_tenant(payload), feeder_id=feeder_id)})


# ── faults ────────────────────────────────────────────────────────────────────

def list_faults(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/faults"""
	return _ok(fault_management_model(_SERVICE, _tenant(payload)))


def report_fault(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/faults"""
	try:
		result = _SERVICE.report_fault(
			fault_id=payload["fault_id"],
			tenant_id=_tenant(payload),
			element_id=payload["element_id"],
			fault_type=payload["fault_type"],
			location_reference=payload.get("location_reference", ""),
			affected_customers=int(payload.get("affected_customers", 0)),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_fault(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/faults/<id>"""
	return _ok(fault_detail_model(_SERVICE, _tenant(payload), payload["fault_id"]))


def isolate_fault(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-dis/api/v1/faults/<id>/isolate"""
	try:
		return _ok(_SERVICE.isolate_fault(payload["fault_id"], _tenant(payload)))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


def restore_fault(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-dis/api/v1/faults/<id>/restore"""
	try:
		return _ok(_SERVICE.restore_fault(
			payload["fault_id"], _tenant(payload),
			strategy=payload.get("strategy", "manual_switching"),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def dispatch_crew(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-dis/api/v1/faults/<id>/dispatch-crew"""
	try:
		return _ok(_SERVICE.dispatch_crew(
			payload["fault_id"], _tenant(payload),
			crew_id=payload["crew_id"],
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── switching orders ──────────────────────────────────────────────────────────

def list_switching_orders(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/switching"""
	return _ok(switching_orders_model(_SERVICE, _tenant(payload)))


def create_switching_order(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/switching"""
	try:
		result = _SERVICE.create_switching_order(
			order_id=payload["order_id"],
			tenant_id=_tenant(payload),
			element_id=payload["element_id"],
			operation=payload["operation"],
			requested_by=payload["requested_by"],
			purpose=payload.get("purpose", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def approve_switching_order(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-dis/api/v1/switching/<id>/approve"""
	try:
		return _ok(_SERVICE.approve_switching_order(
			payload["order_id"], _tenant(payload),
			approved_by=payload.get("approved_by", ""),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def execute_switching_order(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-dis/api/v1/switching/<id>/execute"""
	try:
		return _ok(_SERVICE.execute_switching_order(
			payload["order_id"], _tenant(payload),
			network_live=payload.get("network_live", True),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── outages ───────────────────────────────────────────────────────────────────

def list_outages(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/outages"""
	return _ok(outage_manager_model(_SERVICE, _tenant(payload)))


def record_outage(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/outages"""
	try:
		result = _SERVICE.record_outage(
			outage_id=payload["outage_id"],
			tenant_id=_tenant(payload),
			feeder_id=payload["feeder_id"],
			cause=payload["cause"],
			started_at=payload["started_at"],
			restoration_strategy=payload.get("restoration_strategy", "manual_switching"),
			affected_customers=int(payload.get("affected_customers", 0)),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def restore_outage(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-dis/api/v1/outages/<id>/restore"""
	try:
		return _ok(_SERVICE.restore_outage(
			payload["outage_id"], _tenant(payload),
			saidi_minutes=float(payload.get("saidi_minutes", 0)),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── scada ─────────────────────────────────────────────────────────────────────

def get_scada(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/scada"""
	return _ok(scada_console_model(_SERVICE, _tenant(payload)))


def post_scada_reading(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/scada/readings"""
	try:
		result = _SERVICE.process_scada_reading(
			reading_id=payload["reading_id"],
			tenant_id=_tenant(payload),
			element_id=payload["element_id"],
			protocol=payload["protocol"],
			parameter=payload["parameter"],
			value=float(payload["value"]),
			unit=payload["unit"],
			quality=payload.get("quality", "good"),
			timestamp=payload["timestamp"],
			heartbeat_valid=payload.get("heartbeat_valid", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── load balancing ────────────────────────────────────────────────────────────

def get_load_balancing(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/load-balancing"""
	return _ok(load_balancing_model(_SERVICE, _tenant(payload)))


def apply_load_balance(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/load-balancing"""
	try:
		result = _SERVICE.apply_load_balance(
			action_id=payload["action_id"],
			tenant_id=_tenant(payload),
			feeder_id=payload["feeder_id"],
			mode=payload["mode"],
			action_type=payload.get("action_type", "load_transfer"),
			load_transferred_mw=float(payload.get("load_transferred_mw", 0)),
			voltage_improvement_pu=float(payload.get("voltage_improvement_pu", 0)),
			voltage_within_limits=payload.get("voltage_within_limits", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── reliability ───────────────────────────────────────────────────────────────

def get_reliability(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/reliability"""
	return _ok(reliability_kpi_model(_SERVICE, _tenant(payload)))


# ── agents ────────────────────────────────────────────────────────────────────

def list_agents(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-dis/api/v1/agents"""
	return _ok(agent_workbench_model(_SERVICE, _tenant(payload)))


def register_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-dis/api/v1/agents"""
	try:
		result = _SERVICE.register_agent(
			agent_id=payload["agent_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			runtime=payload["runtime"],
			role=payload["role"],
			scope=payload.get("scope", "distribution network operations"),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def service() -> DistributionNetworkService:
	return _SERVICE
