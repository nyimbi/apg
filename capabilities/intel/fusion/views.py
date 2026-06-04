"""Flask Blueprint UI views for APG Intelligence Fusion.

Routes:
  /intel-fusion/                  — dashboard
  /intel-fusion/workspaces        — workspace list
  /intel-fusion/workspaces/<id>   — workspace detail
  /intel-fusion/items             — intelligence item list
  /intel-fusion/items/<id>        — item detail
  /intel-fusion/correlations      — correlation set list
  /intel-fusion/correlations/<id> — correlation detail
  /intel-fusion/assessments       — assessment picture list
  /intel-fusion/assessments/<id>  — assessment detail
  /intel-fusion/products          — intelligence product list
  /intel-fusion/products/<id>     — product detail
  /intel-fusion/hypotheses        — hypothesis test list
  /intel-fusion/hypotheses/<id>   — hypothesis detail (incl. ACH matrix)
  /intel-fusion/evidence          — evidence list
  /intel-fusion/evidence/<id>     — evidence detail
  /intel-fusion/judgements        — analytical judgement list
  /intel-fusion/reports           — reports hub

Views return JSON view-models. Rendering is handled by the frontend.

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .capability_contract import get_capability_contract
	from .service import IntelligenceFusionService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import IntelligenceFusionService  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Process-local view model helpers — used by test_package_contract.py
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_model(svc: IntelligenceFusionService, tenant_id: str = "default") -> dict:
	"""Return a dashboard view model dict for the given tenant."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Intelligence Fusion",
		"tenant_id": tenant_id,
		"summary": svc.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def fusion_console_model(svc: IntelligenceFusionService, tenant_id: str = "default") -> dict:
	"""Return the full fusion console view model for the given tenant."""
	return {
		"tenant_id": tenant_id,
		"authorities": _tenant_items(svc._sync_authorities, tenant_id),
		"workspaces": _tenant_items(svc._sync_workspaces, tenant_id),
		"sources": _tenant_items(svc._sync_sources, tenant_id),
		"artifacts": _tenant_items(svc._sync_artifacts, tenant_id),
		"correlations": _tenant_items(svc._sync_correlations, tenant_id),
		"hypotheses": _tenant_items(svc._sync_hypotheses, tenant_id),
		"assessments": _tenant_items(svc._sync_assessments, tenant_id),
		"referrals": _tenant_items(svc._sync_referrals, tenant_id),
		"disseminations": _tenant_items(svc._sync_disseminations, tenant_id),
		"reviews": _tenant_items(svc._sync_reviews, tenant_id),
	}


def agent_workbench_model(svc: IntelligenceFusionService, tenant_id: str = "default") -> dict:
	"""Return agent workbench view model for the given tenant."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in svc._sync_agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(store: dict, tenant_id: str) -> list:
	return [
		item.to_dict()
		for item in sorted(store.values(), key=lambda v: v.id)
		if item.tenant_id == tenant_id
	]

ui = Blueprint("intel_fusion_ui", __name__, url_prefix="/intel-fusion")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _svc() -> IntelligenceFusionService:
	tenant_id = request.headers.get("X-Tenant-ID", "default")
	actor_id = request.headers.get("X-Actor-ID", "ui")
	return IntelligenceFusionService(tenant_id=tenant_id, actor_id=actor_id)


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _qs(key: str, default: Any = None) -> Any:
	return request.args.get(key, default)


def _page() -> tuple[int, int]:
	return int(_qs("page", 1)), int(_qs("page_size", 20))


def _kpis(report: Any) -> list[dict[str, Any]]:
	"""Extract top-level KPIs from a dashboard report for the UI card row."""
	d = report.model_dump(mode="json") if hasattr(report, "model_dump") else report
	return [
		{"label": "Total Items",          "value": d.get("total_items", 0),          "color": "blue"},
		{"label": "Active Workspaces",    "value": d.get("active_workspaces", 0),    "color": "green"},
		{"label": "Open Hypotheses",      "value": d.get("open_hypotheses", 0),      "color": "orange"},
		{"label": "Critical Assessments", "value": d.get("critical_assessments", 0), "color": "red"},
		{"label": "Released Products",    "value": d.get("released_products", 0),    "color": "purple"},
		{"label": "Total Evidence",       "value": d.get("total_evidence", 0),       "color": "teal"},
	]


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/")
def dashboard():
	"""
	Dashboard view.

	Returns KPIs, recent workspace summaries, and navigation links.
	"""
	svc = _svc()
	report = _run(svc.dashboard_report())
	workspaces = _run(svc.list_workspaces(status="active", page=1, page_size=5))
	report_dict = report.model_dump(mode="json")
	return jsonify({
		"view": "dashboard",
		"title": "Intelligence Fusion — Dashboard",
		"tenant_id": svc.tenant_id,
		"kpis": _kpis(report),
		"report": report_dict,
		"recent_workspaces": workspaces.items[:5],
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# FusionWorkspace views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/workspaces")
def list_workspaces():
	"""Workspace list view with pagination and filter support."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_workspaces(
		status=_qs("status"),
		workspace_type=_qs("workspace_type"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "workspace_list",
		"title": "Fusion Workspaces",
		"tenant_id": svc.tenant_id,
		"workspaces": result.model_dump(mode="json"),
		"filters": {"status": _qs("status"), "workspace_type": _qs("workspace_type")},
		"nav": _nav(),
	})


@ui.get("/workspaces/<workspace_id>")
def detail_workspace(workspace_id: str):
	"""Workspace detail view including summary counts."""
	svc = _svc()
	try:
		ws = _run(svc.get_workspace(workspace_id))
		summary = _run(svc.workspace_summary(workspace_id))
		items = _run(svc.list_intel_items(workspace_id=workspace_id, page=1, page_size=10))
		correlations = _run(svc.list_correlations(workspace_id=workspace_id, page=1, page_size=10))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify({
		"view": "workspace_detail",
		"title": ws.name,
		"workspace": ws.model_dump(mode="json"),
		"summary": summary.model_dump(mode="json"),
		"recent_items": items.items[:10],
		"recent_correlations": correlations.items[:10],
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceItem views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/items")
def list_items():
	"""Intelligence item list view."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_intel_items(
		workspace_id=_qs("workspace_id"),
		source_type=_qs("source_type"),
		status=_qs("status"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "item_list",
		"title": "Intelligence Items",
		"items": result.model_dump(mode="json"),
		"filters": {
			"workspace_id": _qs("workspace_id"),
			"source_type": _qs("source_type"),
			"status": _qs("status"),
		},
		"nav": _nav(),
	})


@ui.get("/items/<item_id>")
def detail_item(item_id: str):
	"""Intelligence item detail view."""
	svc = _svc()
	try:
		item = _run(svc.get_intel_item(item_id))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify({
		"view": "item_detail",
		"title": f"Intel Item — {item.source_type.value.upper()}",
		"item": item.model_dump(mode="json"),
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# CorrelationSet views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/correlations")
def list_correlations():
	"""Correlation set list view."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_correlations(
		workspace_id=_qs("workspace_id"),
		status=_qs("status"),
		correlation_type=_qs("correlation_type"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "correlation_list",
		"title": "Correlation Sets",
		"correlations": result.model_dump(mode="json"),
		"filters": {
			"workspace_id": _qs("workspace_id"),
			"status": _qs("status"),
			"correlation_type": _qs("correlation_type"),
		},
		"nav": _nav(),
	})


@ui.get("/correlations/<correlation_id>")
def detail_correlation(correlation_id: str):
	"""Correlation set detail view."""
	svc = _svc()
	try:
		corr = _run(svc.get_correlation(correlation_id))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify({
		"view": "correlation_detail",
		"title": f"Correlation — {corr.correlation_type.value.replace('_', ' ').title()}",
		"correlation": corr.model_dump(mode="json"),
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# AssessmentPicture views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/assessments")
def list_assessments():
	"""Assessment picture list view."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_assessments(
		workspace_id=_qs("workspace_id"),
		risk_level=_qs("risk_level"),
		assessment_type=_qs("assessment_type"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "assessment_list",
		"title": "Assessment Pictures",
		"assessments": result.model_dump(mode="json"),
		"filters": {
			"workspace_id": _qs("workspace_id"),
			"risk_level": _qs("risk_level"),
			"assessment_type": _qs("assessment_type"),
		},
		"nav": _nav(),
	})


@ui.get("/assessments/<assessment_id>")
def detail_assessment(assessment_id: str):
	"""Assessment picture detail view."""
	svc = _svc()
	try:
		assessment = _run(svc.get_assessment(assessment_id))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify({
		"view": "assessment_detail",
		"title": f"Assessment — {assessment.risk_level.value.upper()} {assessment.assessment_type.value.title()}",
		"assessment": assessment.model_dump(mode="json"),
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceProduct views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/products")
def list_products():
	"""Intelligence product list view."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_products(
		workspace_id=_qs("workspace_id"),
		status=_qs("status"),
		product_type=_qs("product_type"),
		tlp=_qs("tlp"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "product_list",
		"title": "Intelligence Products",
		"products": result.model_dump(mode="json"),
		"filters": {
			"status": _qs("status"),
			"product_type": _qs("product_type"),
			"tlp": _qs("tlp"),
		},
		"nav": _nav(),
	})


@ui.get("/products/<product_id>")
def detail_product(product_id: str):
	"""Intelligence product detail view."""
	svc = _svc()
	try:
		product = _run(svc.get_product(product_id))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify({
		"view": "product_detail",
		"title": product.title,
		"product": product.model_dump(mode="json"),
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisTest views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/hypotheses")
def list_hypotheses():
	"""Hypothesis test list view."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_hypotheses(
		workspace_id=_qs("workspace_id"),
		status=_qs("status"),
		sat_method=_qs("sat_method"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "hypothesis_list",
		"title": "Hypothesis Tests",
		"hypotheses": result.model_dump(mode="json"),
		"filters": {
			"workspace_id": _qs("workspace_id"),
			"status": _qs("status"),
			"sat_method": _qs("sat_method"),
		},
		"nav": _nav(),
	})


@ui.get("/hypotheses/<hypothesis_id>")
def detail_hypothesis(hypothesis_id: str):
	"""Hypothesis detail view including ACH matrix if available."""
	svc = _svc()
	try:
		hyp = _run(svc.get_hypothesis(hypothesis_id))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404

	hyp_dict = hyp.model_dump(mode="json")
	ach_view = None
	if hyp.ach_matrix and hyp.alternative_hypotheses:
		all_hyps = [hyp.statement] + hyp.alternative_hypotheses
		ach_view = _format_ach_matrix(hyp_dict.get("ach_matrix", {}), all_hyps)

	return jsonify({
		"view": "hypothesis_detail",
		"title": f"Hypothesis — {hyp.sat_method.value.replace('_', ' ').title()}",
		"hypothesis": hyp_dict,
		"ach_matrix_view": ach_view,
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# Evidence views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/evidence")
def list_evidence():
	"""Evidence list view."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_evidence(
		workspace_id=_qs("workspace_id"),
		evidence_type=_qs("evidence_type"),
		status=_qs("status"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "evidence_list",
		"title": "Evidence",
		"evidence": result.model_dump(mode="json"),
		"filters": {
			"workspace_id": _qs("workspace_id"),
			"evidence_type": _qs("evidence_type"),
			"status": _qs("status"),
		},
		"nav": _nav(),
	})


@ui.get("/evidence/<evidence_id>")
def detail_evidence(evidence_id: str):
	"""Evidence detail view with chain-of-custody."""
	svc = _svc()
	try:
		ev = _run(svc.get_evidence(evidence_id))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify({
		"view": "evidence_detail",
		"title": f"Evidence — {ev.evidence_type.value.title()}",
		"evidence": ev.model_dump(mode="json"),
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticalJudgement views
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/judgements")
def list_judgements():
	"""Analytical judgement list view."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_judgements(
		workspace_id=_qs("workspace_id"),
		judgement_type=_qs("judgement_type"),
		page=page,
		page_size=page_size,
	))
	return jsonify({
		"view": "judgement_list",
		"title": "Analytical Judgements",
		"judgements": result.model_dump(mode="json"),
		"filters": {
			"workspace_id": _qs("workspace_id"),
			"judgement_type": _qs("judgement_type"),
		},
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# Reports hub
# ─────────────────────────────────────────────────────────────────────────────

@ui.get("/reports")
def reports_hub():
	"""Reports hub — links to all available reports."""
	svc = _svc()
	report = _run(svc.dashboard_report())
	return jsonify({
		"view": "reports_hub",
		"title": "Fusion Reports",
		"tenant_id": svc.tenant_id,
		"kpis": _kpis(report),
		"available_reports": [
			{"id": "dashboard",         "title": "Dashboard",                "path": "/intel-fusion/"},
			{"id": "workspace_summary", "title": "Workspace Summary",        "path": "/intel-fusion/workspaces"},
			{"id": "item_pipeline",     "title": "Item Pipeline",            "path": "/intel-fusion/items"},
			{"id": "correlation_map",   "title": "Correlation Map",          "path": "/intel-fusion/correlations"},
			{"id": "assessment_matrix", "title": "Assessment Risk Matrix",   "path": "/intel-fusion/assessments"},
			{"id": "product_register",  "title": "Product Register",         "path": "/intel-fusion/products"},
			{"id": "hypothesis_board",  "title": "Hypothesis Board",         "path": "/intel-fusion/hypotheses"},
			{"id": "evidence_log",      "title": "Evidence Log",             "path": "/intel-fusion/evidence"},
			{"id": "judgement_log",     "title": "Analytical Judgements",    "path": "/intel-fusion/judgements"},
		],
		"nav": _nav(),
	})


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nav() -> list[dict[str, str]]:
	"""Standard navigation links included in every view model."""
	return [
		{"label": "Dashboard",   "path": "/intel-fusion/"},
		{"label": "Workspaces",  "path": "/intel-fusion/workspaces"},
		{"label": "Items",       "path": "/intel-fusion/items"},
		{"label": "Correlations","path": "/intel-fusion/correlations"},
		{"label": "Assessments", "path": "/intel-fusion/assessments"},
		{"label": "Products",    "path": "/intel-fusion/products"},
		{"label": "Hypotheses",  "path": "/intel-fusion/hypotheses"},
		{"label": "Evidence",    "path": "/intel-fusion/evidence"},
		{"label": "Judgements",  "path": "/intel-fusion/judgements"},
		{"label": "Reports",     "path": "/intel-fusion/reports"},
	]


def _format_ach_matrix(
	ach_matrix: dict[str, list[float]],
	hypotheses: list[str],
) -> dict[str, Any]:
	"""Format a stored ACH matrix dict for table rendering in the UI."""
	rows = []
	for ev_id, scores in ach_matrix.items():
		row = {"evidence_id": ev_id, "scores": {}}
		for i, hyp in enumerate(hypotheses):
			score = scores[i] if i < len(scores) else 0.0
			label = "C" if score > 0 else ("I" if score < 0 else "N/A")
			row["scores"][hyp[:30]] = {"value": score, "label": label}
		rows.append(row)
	return {"hypotheses": hypotheses, "rows": rows}
