"""Flask Blueprint REST API for APG Mobile Device Management."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .capability_contract import get_capability_contract
	from .models import (
		AppDistributionCreate,
		ComplianceEvaluationCreate,
		DeviceEnrolmentCreate,
		DeviceUpdate,
		MdmProfileCreate,
		PolicyAssignmentCreate,
		PolicyCreate,
		PolicyUpdate,
		WipeRequestCreate,
	)
	from .service import MobileDeviceManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from models import (  # type: ignore
		AppDistributionCreate,
		ComplianceEvaluationCreate,
		DeviceEnrolmentCreate,
		DeviceUpdate,
		MdmProfileCreate,
		PolicyAssignmentCreate,
		PolicyCreate,
		PolicyUpdate,
		WipeRequestCreate,
	)
	from service import MobileDeviceManagementService  # type: ignore


bp = Blueprint("mob_mdm", __name__, url_prefix="/api/mob/mdm")
_svc = MobileDeviceManagementService()


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
	"""Return MDM capability contract.
	---
	GET /api/mob/mdm/contract
	"""
	return _ok(get_capability_contract(_tenant()))


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

@bp.get("/devices")
def list_devices():
	"""List enrolled devices.
	---
	GET /api/mob/mdm/devices
	Query: os_platform, enrolment_state, ownership_type
	Permission: mob_mdm:devices:list
	"""
	try:
		devices = _run(_svc.list_devices(
			_tenant(),
			os_platform=request.args.get("os_platform"),
			enrolment_state=request.args.get("enrolment_state"),
			ownership_type=request.args.get("ownership_type"),
		))
		return _ok([d.model_dump() for d in devices])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/devices")
def enrol_device():
	"""Enrol a device into MDM.
	---
	POST /api/mob/mdm/devices
	Permission: mob_mdm:enrolment:manage
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = DeviceEnrolmentCreate(**body)
		device = _run(_svc.enrol_device(payload))
		return _ok(device.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.get("/devices/<device_id>")
def get_device(device_id: str):
	"""Get a device by ID.
	---
	GET /api/mob/mdm/devices/<device_id>
	Permission: mob_mdm:devices:view
	"""
	try:
		device = _run(_svc.get_device(_tenant(), device_id))
		return _ok(device.model_dump())
	except AssertionError as exc:
		return _err(str(exc), 404)


@bp.put("/devices/<device_id>")
def update_device(device_id: str):
	"""Update a device.
	---
	PUT /api/mob/mdm/devices/<device_id>
	Permission: mob_mdm:devices:edit
	"""
	body = request.get_json(force=True) or {}
	try:
		payload = DeviceUpdate(**body)
		device = _run(_svc.update_device(_tenant(), device_id, payload))
		return _ok(device.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/devices/<device_id>/unenrol")
def unenrol_device(device_id: str):
	"""Unenrol a device.
	---
	POST /api/mob/mdm/devices/<device_id>/unenrol
	Permission: mob_mdm:enrolment:manage
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		device = _run(_svc.unenrol_device(_tenant(), device_id, body.get("unenrolled_by", "system")))
		return _ok(device.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/devices/<device_id>/suspend")
def suspend_device(device_id: str):
	"""Suspend a device.
	---
	POST /api/mob/mdm/devices/<device_id>/suspend
	Permission: mob_mdm:devices:manage
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		device = _run(_svc.suspend_device(_tenant(), device_id, body.get("suspended_by", "system")))
		return _ok(device.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@bp.get("/policies")
def list_policies():
	"""List MDM policies.
	---
	GET /api/mob/mdm/policies
	Permission: mob_mdm:policies:list
	"""
	try:
		policies = _run(_svc.list_policies(
			_tenant(),
			policy_type=request.args.get("policy_type"),
			state=request.args.get("state"),
		))
		return _ok([p.model_dump() for p in policies])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/policies")
def create_policy():
	"""Create an MDM policy.
	---
	POST /api/mob/mdm/policies
	Permission: mob_mdm:policies:create
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = PolicyCreate(**body)
		policy = _run(_svc.create_policy(payload))
		return _ok(policy.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.get("/policies/<policy_id>")
def get_policy(policy_id: str):
	"""Get a policy by ID.
	---
	GET /api/mob/mdm/policies/<policy_id>
	Permission: mob_mdm:policies:view
	"""
	try:
		policy = _run(_svc.list_policies(_tenant()))
		match = next((p for p in policy if p.id == policy_id), None)
		if not match:
			return _err("policy_not_found", 404)
		return _ok(match.model_dump())
	except Exception as exc:
		return _err(str(exc))


@bp.put("/policies/<policy_id>")
def update_policy(policy_id: str):
	"""Update an MDM policy.
	---
	PUT /api/mob/mdm/policies/<policy_id>
	Permission: mob_mdm:policies:edit
	"""
	body = request.get_json(force=True) or {}
	try:
		payload = PolicyUpdate(**body)
		policy = _run(_svc.update_policy(_tenant(), policy_id, payload))
		return _ok(policy.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/policies/<policy_id>/activate")
def activate_policy(policy_id: str):
	"""Activate a policy.
	---
	POST /api/mob/mdm/policies/<policy_id>/activate
	Permission: mob_mdm:policies:activate
	"""
	body = request.get_json(force=True) or {}
	try:
		policy = _run(_svc.activate_policy(
			_tenant(), policy_id,
			body.get("approval_reference", ""),
			body.get("activated_by", "system"),
		))
		return _ok(policy.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/policies/assign")
def assign_policy():
	"""Assign a policy to a device.
	---
	POST /api/mob/mdm/policies/assign
	Permission: mob_mdm:policies:assign
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = PolicyAssignmentCreate(**body)
		assignment = _run(_svc.assign_policy(payload))
		return _ok(assignment.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

@bp.get("/compliance")
def list_compliance():
	"""List compliance records.
	---
	GET /api/mob/mdm/compliance
	Permission: mob_mdm:compliance:view
	"""
	try:
		records = _run(_svc.list_compliance_records(
			_tenant(),
			device_id=request.args.get("device_id"),
			compliance_state=request.args.get("compliance_state"),
		))
		return _ok([r.model_dump() for r in records])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/compliance")
def evaluate_compliance():
	"""Run a compliance evaluation.
	---
	POST /api/mob/mdm/compliance
	Permission: mob_mdm:compliance:evaluate
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = ComplianceEvaluationCreate(**body)
		record = _run(_svc.evaluate_compliance(payload))
		return _ok(record.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# App Distribution
# ---------------------------------------------------------------------------

@bp.get("/apps")
def list_app_distributions():
	"""List app distributions.
	---
	GET /api/mob/mdm/apps
	Permission: mob_mdm:apps:list
	"""
	try:
		dists = _run(_svc.list_app_distributions(_tenant(), device_id=request.args.get("device_id")))
		return _ok([d.model_dump() for d in dists])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/apps")
def distribute_app():
	"""Distribute an app to a device.
	---
	POST /api/mob/mdm/apps
	Permission: mob_mdm:apps:distribute
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = AppDistributionCreate(**body)
		dist = _run(_svc.distribute_app(payload))
		return _ok(dist.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Remote Wipe
# ---------------------------------------------------------------------------

@bp.get("/remote-actions/wipes")
def list_wipes():
	"""List wipe requests.
	---
	GET /api/mob/mdm/remote-actions/wipes
	Permission: mob_mdm:remote:wipe
	"""
	try:
		wipes = _run(_svc.list_wipe_requests(
			_tenant(),
			device_id=request.args.get("device_id"),
			state=request.args.get("state"),
		))
		return _ok([w.model_dump() for w in wipes])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/remote-actions/wipes")
def request_wipe():
	"""Request a remote wipe.
	---
	POST /api/mob/mdm/remote-actions/wipes
	Permission: mob_mdm:remote:wipe
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = WipeRequestCreate(**body)
		wipe = _run(_svc.request_wipe(payload))
		return _ok(wipe.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/remote-actions/wipes/<wipe_id>/execute")
def execute_wipe(wipe_id: str):
	"""Execute a wipe request.
	---
	POST /api/mob/mdm/remote-actions/wipes/<wipe_id>/execute
	Permission: mob_mdm:remote:wipe
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		wipe = _run(_svc.execute_wipe(_tenant(), wipe_id, body.get("executed_by", "system")))
		return _ok(wipe.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@bp.get("/profiles")
def list_profiles():
	"""List MDM profiles.
	---
	GET /api/mob/mdm/profiles
	Permission: mob_mdm:profiles:list
	"""
	try:
		profiles = _run(_svc.list_profiles(_tenant(), profile_type=request.args.get("profile_type")))
		return _ok([p.model_dump() for p in profiles])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/profiles")
def create_profile():
	"""Create an MDM profile.
	---
	POST /api/mob/mdm/profiles
	Permission: mob_mdm:profiles:create
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = MdmProfileCreate(**body)
		profile = _run(_svc.create_profile(payload))
		return _ok(profile.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/profiles/<profile_id>/deploy/<device_id>")
def deploy_profile(profile_id: str, device_id: str):
	"""Deploy a profile to a device.
	---
	POST /api/mob/mdm/profiles/<profile_id>/deploy/<device_id>
	Permission: mob_mdm:profiles:deploy
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		profile = _run(_svc.deploy_profile(_tenant(), profile_id, device_id, body.get("deployed_by", "system")))
		return _ok(profile.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@bp.get("/alerts")
def list_alerts():
	"""List MDM alerts.
	---
	GET /api/mob/mdm/alerts
	Permission: mob_mdm:alerts:view
	"""
	resolved_param = request.args.get("resolved")
	resolved = None if resolved_param is None else (resolved_param.lower() == "true")
	try:
		alerts = _run(_svc.list_alerts(_tenant(), device_id=request.args.get("device_id"), resolved=resolved))
		return _ok([a.model_dump() for a in alerts])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/alerts/<alert_id>/resolve")
def resolve_alert(alert_id: str):
	"""Resolve an MDM alert.
	---
	POST /api/mob/mdm/alerts/<alert_id>/resolve
	Permission: mob_mdm:alerts:manage
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		alert = _run(_svc.resolve_alert(_tenant(), alert_id, body.get("resolved_by", "system")))
		return _ok(alert.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))
