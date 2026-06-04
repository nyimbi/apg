"""Flask Blueprint REST API for grc_pol capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_pol_api", __name__, url_prefix="/api/v1/grc/pol")


def _svc():
	from .service import PolicyManagementService
	return PolicyManagementService()


# ── Policies CRUD ─────────────────────────────────────────────────────────────

@blueprint.get("/policies")
def list_policies():
	import asyncio
	svc = _svc()
	result = asyncio.run(
		svc.policy_library(
			category=request.args.get("category"),
			status=request.args.get("status"),
		)
	)
	return jsonify(result)


@blueprint.post("/policies")
def create_policy():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.create_policy(
			title=data["title"],
			category=data["category"],
			policy_type=data["policy_type"],
			owner_id=data["owner_id"],
			effective_date=data["effective_date"],
			review_cycle_months=int(data["review_cycle_months"]),
			scope=data.get("scope", "organization_wide"),
			description=data.get("description", ""),
			version=data.get("version", "1.0"),
		)
	)
	return jsonify(result), 201


@blueprint.get("/policies/<policy_id>")
def get_policy(policy_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc._get_policy(policy_id))
	return jsonify(result)


@blueprint.put("/policies/<policy_id>")
def draft_content(policy_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.draft_policy_content(
			policy_id=policy_id,
			content_sections=data["content_sections"],
			author_id=data["author_id"],
		)
	)
	return jsonify(result)


@blueprint.delete("/policies/<policy_id>")
def retire_policy(policy_id: str):
	import asyncio
	data = request.get_json(force=True) or {}
	svc = _svc()
	result = asyncio.run(
		svc.retire_policy(
			policy_id=policy_id,
			reason=data.get("reason", "retired"),
			retired_by=data.get("retired_by", "admin"),
		)
	)
	return jsonify(result)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@blueprint.post("/policies/<policy_id>/review")
def review_policy(policy_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.policy_review(
			policy_id=policy_id,
			reviewer_id=data["reviewer_id"],
			comments=data.get("comments", ""),
			recommended_action=data["recommended_action"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/policies/<policy_id>/approve")
def approve_policy(policy_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.approve_policy(
			policy_id=policy_id,
			approver_id=data["approver_id"],
			approval_date=data["approval_date"],
			comments=data.get("comments", ""),
		)
	)
	return jsonify(result)


@blueprint.post("/policies/<policy_id>/publish")
def publish_policy(policy_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.publish_policy(
			policy_id=policy_id,
			distribution_list=data["distribution_list"],
		)
	)
	return jsonify(result)


@blueprint.post("/policies/<policy_id>/acknowledge")
def acknowledge_policy(policy_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.acknowledge_policy(
			policy_id=policy_id,
			employee_id=data["employee_id"],
			acknowledgement_date=data["acknowledgement_date"],
			method=data.get("method", "electronic_signature"),
		)
	)
	return jsonify(result)


@blueprint.post("/policies/<policy_id>/revise")
def revise_policy(policy_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.policy_revision(
			policy_id=policy_id,
			revision_reason=data["revision_reason"],
			revision_summary=data["revision_summary"],
			revised_by=data["revised_by"],
		)
	)
	return jsonify(result), 201


# ── Exceptions ────────────────────────────────────────────────────────────────

@blueprint.get("/exceptions")
def list_exceptions():
	return jsonify({"exceptions": []})


@blueprint.post("/policies/<policy_id>/exceptions")
def request_exception(policy_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.policy_exception_request(
			policy_id=policy_id,
			requestor_id=data["requestor_id"],
			reason=data["reason"],
			compensating_controls=data["compensating_controls"],
			risk_level=data["risk_level"],
			exception_type=data.get("exception_type", "temporary_exemption"),
			duration_days=int(data.get("duration_days", 90)),
		)
	)
	return jsonify(result), 201


@blueprint.post("/exceptions/<exception_id>/approve")
def approve_exception(exception_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.approve_exception(
			exception_id=exception_id,
			approver_id=data["approver_id"],
			approved_until=data["approved_until"],
			conditions=data["conditions"],
		)
	)
	return jsonify(result)


# ── Analytics and search ──────────────────────────────────────────────────────

@blueprint.get("/analytics")
def analytics():
	import asyncio
	period = request.args.get("period", "2026-06")
	svc = _svc()
	result = asyncio.run(svc.policy_analytics(period))
	return jsonify(result)


@blueprint.get("/dashboard")
def dashboard():
	import asyncio
	entity_id = request.args.get("entity_id", "default")
	svc = _svc()
	result = asyncio.run(svc.policy_dashboard(entity_id))
	return jsonify(result)


@blueprint.get("/search")
def search():
	import asyncio
	query = request.args.get("q", "")
	svc = _svc()
	result = asyncio.run(svc.policy_search(query))
	return jsonify(result)


@blueprint.get("/gap-analysis")
def gap_analysis():
	import asyncio
	entity_id = request.args["entity_id"]
	framework = request.args["framework"]
	svc = _svc()
	result = asyncio.run(
		svc.policy_gap_analysis(entity_id=entity_id, framework=framework)
	)
	return jsonify(result)


@blueprint.get("/expiry-report")
def expiry_report():
	import asyncio
	days_ahead = int(request.args.get("days_ahead", 90))
	svc = _svc()
	result = asyncio.run(svc.policy_expiry_report(days_ahead))
	return jsonify(result)
