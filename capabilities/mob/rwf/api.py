"""Flask Blueprint REST API for APG Remote Workforce."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .capability_contract import get_capability_contract
	from .models import (
		ComplianceCheckCreate,
		EquipmentRequisitionCreate,
		OnboardingRecordCreate,
		OnboardingStepCreate,
		PolicyAcknowledgmentCreate,
		ProductivityMetricCreate,
		RemoteIncidentCreate,
		VpnAccessCreate,
		WorkPolicyCreate,
		WorkPolicyUpdate,
	)
	from .service import RemoteWorkforceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from models import (  # type: ignore
		ComplianceCheckCreate,
		EquipmentRequisitionCreate,
		OnboardingRecordCreate,
		OnboardingStepCreate,
		PolicyAcknowledgmentCreate,
		ProductivityMetricCreate,
		RemoteIncidentCreate,
		VpnAccessCreate,
		WorkPolicyCreate,
		WorkPolicyUpdate,
	)
	from service import RemoteWorkforceService  # type: ignore


bp = Blueprint("mob_rwf", __name__, url_prefix="/api/mob/rwf")
_svc = RemoteWorkforceService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400):
	return jsonify({"status": "error", "message": msg}), status


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@bp.get("/contract")
def get_contract():
	"""Return RWF capability contract.
	---
	GET /api/mob/rwf/contract
	"""
	return _ok(get_capability_contract(_tenant()))


# ---------------------------------------------------------------------------
# Work Policies
# ---------------------------------------------------------------------------

@bp.get("/policies")
def list_policies():
	"""List remote work policies.
	---
	GET /api/mob/rwf/policies
	Permission: mob_rwf:policies:list
	"""
	try:
		policies = _run(_svc.list_work_policies(
			_tenant(),
			policy_type=request.args.get("policy_type"),
			state=request.args.get("state"),
		))
		return _ok([p.model_dump() for p in policies])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/policies")
def create_policy():
	"""Create a work policy.
	---
	POST /api/mob/rwf/policies
	Permission: mob_rwf:policies:create
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = WorkPolicyCreate(**body)
		policy = _run(_svc.create_work_policy(payload))
		return _ok(policy.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.get("/policies/<policy_id>")
def get_policy(policy_id: str):
	"""Get a work policy.
	---
	GET /api/mob/rwf/policies/<policy_id>
	Permission: mob_rwf:policies:view
	"""
	try:
		policy = _run(_svc.get_work_policy(_tenant(), policy_id))
		return _ok(policy.model_dump())
	except AssertionError as exc:
		return _err(str(exc), 404)


@bp.put("/policies/<policy_id>")
def update_policy(policy_id: str):
	"""Update a work policy.
	---
	PUT /api/mob/rwf/policies/<policy_id>
	Permission: mob_rwf:policies:edit
	"""
	body = request.get_json(force=True) or {}
	try:
		payload = WorkPolicyUpdate(**body)
		policy = _run(_svc.update_work_policy(_tenant(), policy_id, payload))
		return _ok(policy.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/policies/<policy_id>/activate")
def activate_policy(policy_id: str):
	"""Activate a work policy.
	---
	POST /api/mob/rwf/policies/<policy_id>/activate
	Permission: mob_rwf:policies:activate
	"""
	body = request.get_json(force=True) or {}
	try:
		policy = _run(_svc.activate_work_policy(
			_tenant(), policy_id,
			body.get("approval_reference", ""),
			body.get("activated_by", "system"),
		))
		return _ok(policy.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/policies/<policy_id>/acknowledge")
def acknowledge_policy(policy_id: str):
	"""Record policy acknowledgment.
	---
	POST /api/mob/rwf/policies/<policy_id>/acknowledge
	Permission: mob_rwf:policies:acknowledge
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	body.setdefault("policy_id", policy_id)
	try:
		payload = PolicyAcknowledgmentCreate(**body)
		ack = _run(_svc.acknowledge_policy(payload))
		return _ok(ack.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.get("/policies/<policy_id>/acknowledgments")
def list_acknowledgments(policy_id: str):
	"""List policy acknowledgments.
	---
	GET /api/mob/rwf/policies/<policy_id>/acknowledgments
	Permission: mob_rwf:policies:view
	"""
	try:
		acks = _run(_svc.list_acknowledgments(_tenant(), policy_id=policy_id))
		return _ok([a.model_dump() for a in acks])
	except Exception as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# VPN Access
# ---------------------------------------------------------------------------

@bp.get("/vpn")
def list_vpn():
	"""List VPN access records.
	---
	GET /api/mob/rwf/vpn
	Permission: mob_rwf:vpn:list
	"""
	try:
		records = _run(_svc.list_vpn_access(
			_tenant(),
			employee_id=request.args.get("employee_id"),
			state=request.args.get("state"),
		))
		return _ok([r.model_dump() for r in records])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/vpn")
def provision_vpn():
	"""Provision VPN access.
	---
	POST /api/mob/rwf/vpn
	Permission: mob_rwf:vpn:provision
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = VpnAccessCreate(**body)
		access = _run(_svc.provision_vpn(payload))
		return _ok(access.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.delete("/vpn/<access_id>")
def revoke_vpn(access_id: str):
	"""Revoke VPN access.
	---
	DELETE /api/mob/rwf/vpn/<access_id>
	Permission: mob_rwf:vpn:revoke
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		access = _run(_svc.revoke_vpn(
			_tenant(), access_id,
			body.get("reason", "revoked"),
			body.get("revoked_by", "system"),
		))
		return _ok(access.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/vpn/<access_id>/sessions")
def start_vpn_session(access_id: str):
	"""Start a VPN session.
	---
	POST /api/mob/rwf/vpn/<access_id>/sessions
	Permission: mob_rwf:vpn:connect
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		session = _run(_svc.start_vpn_session(_tenant(), access_id, body.get("client_ip")))
		return _ok(session.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/vpn/sessions/<session_id>/end")
def end_vpn_session(session_id: str):
	"""End a VPN session.
	---
	POST /api/mob/rwf/vpn/sessions/<session_id>/end
	Permission: mob_rwf:vpn:connect
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		session = _run(_svc.end_vpn_session(
			_tenant(), session_id,
			body.get("bytes_in", 0),
			body.get("bytes_out", 0),
		))
		return _ok(session.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Productivity
# ---------------------------------------------------------------------------

@bp.get("/productivity")
def list_productivity():
	"""List productivity metrics.
	---
	GET /api/mob/rwf/productivity
	Permission: mob_rwf:productivity:view
	"""
	try:
		metrics = _run(_svc.list_productivity_metrics(
			_tenant(),
			employee_id=request.args.get("employee_id"),
			metric_type=request.args.get("metric_type"),
		))
		return _ok([m.model_dump() for m in metrics])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/productivity")
def record_productivity():
	"""Record a productivity metric.
	---
	POST /api/mob/rwf/productivity
	Permission: mob_rwf:productivity:write
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = ProductivityMetricCreate(**body)
		metric = _run(_svc.record_productivity_metric(payload))
		return _ok(metric.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.get("/productivity/<employee_id>/summary")
def productivity_summary(employee_id: str):
	"""Get productivity summary for an employee.
	---
	GET /api/mob/rwf/productivity/<employee_id>/summary
	Permission: mob_rwf:productivity:view
	"""
	try:
		summary = _run(_svc.get_productivity_summary(_tenant(), employee_id))
		return _ok(summary)
	except Exception as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Equipment
# ---------------------------------------------------------------------------

@bp.get("/equipment")
def list_equipment():
	"""List equipment requisitions.
	---
	GET /api/mob/rwf/equipment
	Permission: mob_rwf:equipment:list
	"""
	try:
		items = _run(_svc.list_equipment(
			_tenant(),
			employee_id=request.args.get("employee_id"),
			state=request.args.get("state"),
		))
		return _ok([e.model_dump() for e in items])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/equipment")
def request_equipment():
	"""Submit equipment requisition.
	---
	POST /api/mob/rwf/equipment
	Permission: mob_rwf:equipment:request
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = EquipmentRequisitionCreate(**body)
		req = _run(_svc.request_equipment(payload))
		return _ok(req.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/equipment/<req_id>/approve")
def approve_equipment(req_id: str):
	"""Approve equipment requisition.
	---
	POST /api/mob/rwf/equipment/<req_id>/approve
	Permission: mob_rwf:equipment:approve
	"""
	body = request.get_json(force=True) or {}
	try:
		req = _run(_svc.approve_equipment(
			_tenant(), req_id,
			body.get("approval_reference", ""),
			body.get("approved_by", "system"),
		))
		return _ok(req.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/equipment/<req_id>/ship")
def ship_equipment(req_id: str):
	"""Mark equipment as shipped.
	---
	POST /api/mob/rwf/equipment/<req_id>/ship
	Permission: mob_rwf:equipment:manage
	"""
	body = request.get_json(force=True) or {}
	try:
		req = _run(_svc.ship_equipment(_tenant(), req_id, body.get("asset_tag", "")))
		return _ok(req.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/equipment/<req_id>/deliver")
def deliver_equipment(req_id: str):
	"""Mark equipment as delivered.
	---
	POST /api/mob/rwf/equipment/<req_id>/deliver
	Permission: mob_rwf:equipment:manage
	"""
	try:
		req = _run(_svc.deliver_equipment(_tenant(), req_id))
		return _ok(req.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/equipment/<req_id>/return")
def return_equipment(req_id: str):
	"""Mark equipment as returned.
	---
	POST /api/mob/rwf/equipment/<req_id>/return
	Permission: mob_rwf:equipment:manage
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		req = _run(_svc.return_equipment(_tenant(), req_id, body.get("returned_by", "system")))
		return _ok(req.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

@bp.get("/onboarding")
def list_onboarding():
	"""List onboarding records.
	---
	GET /api/mob/rwf/onboarding
	Permission: mob_rwf:onboarding:list
	"""
	try:
		records = _run(_svc.list_onboarding_records(
			_tenant(),
			state=request.args.get("state"),
			employee_id=request.args.get("employee_id"),
		))
		return _ok([r.model_dump() for r in records])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/onboarding")
def start_onboarding():
	"""Start digital onboarding.
	---
	POST /api/mob/rwf/onboarding
	Permission: mob_rwf:onboarding:start
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = OnboardingRecordCreate(**body)
		record = _run(_svc.start_onboarding(payload))
		return _ok(record.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.get("/onboarding/<record_id>")
def get_onboarding(record_id: str):
	"""Get onboarding record.
	---
	GET /api/mob/rwf/onboarding/<record_id>
	Permission: mob_rwf:onboarding:view
	"""
	try:
		record = _run(_svc.get_onboarding_record(_tenant(), record_id))
		return _ok(record.model_dump())
	except AssertionError as exc:
		return _err(str(exc), 404)


@bp.post("/onboarding/<record_id>/steps")
def complete_step(record_id: str):
	"""Complete an onboarding step.
	---
	POST /api/mob/rwf/onboarding/<record_id>/steps
	Permission: mob_rwf:onboarding:manage
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	body.setdefault("onboarding_id", record_id)
	try:
		payload = OnboardingStepCreate(**body)
		step = _run(_svc.complete_onboarding_step(payload))
		return _ok(step.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

@bp.get("/compliance")
def list_compliance():
	"""List remote compliance checks.
	---
	GET /api/mob/rwf/compliance
	Permission: mob_rwf:compliance:view
	"""
	try:
		checks = _run(_svc.list_compliance_checks(
			_tenant(),
			employee_id=request.args.get("employee_id"),
			check_type=request.args.get("check_type"),
			result=request.args.get("result"),
		))
		return _ok([c.model_dump() for c in checks])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/compliance")
def record_compliance():
	"""Record a compliance check result.
	---
	POST /api/mob/rwf/compliance
	Permission: mob_rwf:compliance:record
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = ComplianceCheckCreate(**body)
		check = _run(_svc.record_compliance_check(payload))
		return _ok(check.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

@bp.get("/incidents")
def list_incidents():
	"""List remote incidents.
	---
	GET /api/mob/rwf/incidents
	Permission: mob_rwf:incidents:view
	"""
	try:
		incidents = _run(_svc.list_incidents(
			_tenant(),
			employee_id=request.args.get("employee_id"),
			incident_type=request.args.get("incident_type"),
			state=request.args.get("state"),
		))
		return _ok([i.model_dump() for i in incidents])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/incidents")
def raise_incident():
	"""Raise a remote incident.
	---
	POST /api/mob/rwf/incidents
	Permission: mob_rwf:incidents:report
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = RemoteIncidentCreate(**body)
		incident = _run(_svc.raise_incident(payload))
		return _ok(incident.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/incidents/<incident_id>/resolve")
def resolve_incident(incident_id: str):
	"""Resolve an incident.
	---
	POST /api/mob/rwf/incidents/<incident_id>/resolve
	Permission: mob_rwf:incidents:resolve
	"""
	body = request.get_json(force=True) or {}
	try:
		incident = _run(_svc.resolve_incident(
			_tenant(), incident_id,
			body.get("resolution_notes", ""),
			body.get("resolved_by", "system"),
		))
		return _ok(incident.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))
