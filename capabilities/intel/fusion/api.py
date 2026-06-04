"""Flask Blueprint REST API for APG Intelligence Fusion.

Endpoints:
  /api/intel-fusion/items           — IntelligenceItem CRUD
  /api/intel-fusion/workspaces      — FusionWorkspace CRUD
  /api/intel-fusion/correlations    — CorrelationSet CRUD
  /api/intel-fusion/assessments     — AssessmentPicture CRUD
  /api/intel-fusion/products        — IntelligenceProduct CRUD + lifecycle
  /api/intel-fusion/judgements      — AnalyticalJudgement CRUD
  /api/intel-fusion/evidence        — Evidence CRUD
  /api/intel-fusion/hypotheses      — HypothesisTest CRUD + ACH
  /api/intel-fusion/sat             — Structured Analytic Techniques
  /api/intel-fusion/reports         — Dashboard and workspace reports

All endpoints are sync Flask handlers wrapping async service calls via
asyncio.run().  Tenant ID is taken from the X-Tenant-ID header (default:
'default').  Actor ID from X-Actor-ID header (default: 'api').

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any

from flask import Blueprint, Response, jsonify, request

try:
	from .models import (
		AnalyticalJudgementCreate,
		AnalyticalJudgementUpdate,
		AssessmentPictureCreate,
		AssessmentPictureUpdate,
		CorrelationSetCreate,
		CorrelationSetUpdate,
		EvidenceCreate,
		EvidenceUpdate,
		FusionWorkspaceCreate,
		FusionWorkspaceUpdate,
		HypothesisTestCreate,
		HypothesisTestUpdate,
		IntelligenceItemCreate,
		IntelligenceItemUpdate,
		IntelligenceProductCreate,
		IntelligenceProductUpdate,
	)
	from .service import IntelligenceFusionService
except ImportError:
	from models import (  # type: ignore
		AnalyticalJudgementCreate, AnalyticalJudgementUpdate,
		AssessmentPictureCreate, AssessmentPictureUpdate,
		CorrelationSetCreate, CorrelationSetUpdate,
		EvidenceCreate, EvidenceUpdate,
		FusionWorkspaceCreate, FusionWorkspaceUpdate,
		HypothesisTestCreate, HypothesisTestUpdate,
		IntelligenceItemCreate, IntelligenceItemUpdate,
		IntelligenceProductCreate, IntelligenceProductUpdate,
	)
	from service import IntelligenceFusionService  # type: ignore


bp = Blueprint("intel_fusion", __name__, url_prefix="/api/intel-fusion")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _svc() -> IntelligenceFusionService:
	tenant_id = request.headers.get("X-Tenant-ID", "default")
	actor_id = request.headers.get("X-Actor-ID", "api")
	return IntelligenceFusionService(tenant_id=tenant_id, actor_id=actor_id)


def _run(coro: Any) -> Any:
	"""Execute an async coroutine in the current or a new event loop."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return asyncio.run(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _ok(data: Any, status: int = 200) -> Response:
	if hasattr(data, "model_dump"):
		return jsonify(data.model_dump(mode="json")), status
	return jsonify(data), status


def _err(msg: str, status: int = 400) -> Response:
	return jsonify({"error": msg}), status


def handle_errors(f):
	@wraps(f)
	def wrapper(*args, **kwargs):
		try:
			return f(*args, **kwargs)
		except KeyError as exc:
			return _err(str(exc), 404)
		except PermissionError as exc:
			return _err(str(exc), 403)
		except (ValueError, AssertionError) as exc:
			return _err(str(exc), 400)
		except Exception as exc:
			return _err(f"internal error: {exc}", 500)
	return wrapper


def _qs(key: str, default: Any = None) -> Any:
	return request.args.get(key, default)


def _page() -> tuple[int, int]:
	return int(_qs("page", 1)), int(_qs("page_size", 50))


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceItem
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/items")
@handle_errors
def list_items():
	"""GET /items — list intelligence items with optional filters."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_intel_items(
		workspace_id=_qs("workspace_id"),
		source_type=_qs("source_type"),
		status=_qs("status"),
		page=page,
		page_size=page_size,
	))
	return _ok(result.model_dump(mode="json"))


@bp.post("/items")
@handle_errors
def create_item():
	"""POST /items — ingest a raw intelligence item."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = IntelligenceItemCreate(**body)
	item = _run(svc.create_intel_item(payload))
	return _ok(item, 201)


@bp.get("/items/<item_id>")
@handle_errors
def get_item(item_id: str):
	"""GET /items/<id> — retrieve a single intelligence item."""
	return _ok(_run(_svc().get_intel_item(item_id)))


@bp.put("/items/<item_id>")
@handle_errors
def update_item(item_id: str):
	"""PUT /items/<id> — partial update."""
	svc = _svc()
	patch = IntelligenceItemUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_intel_item(item_id, patch)))


@bp.delete("/items/<item_id>")
@handle_errors
def delete_item(item_id: str):
	"""DELETE /items/<id> — soft delete."""
	_run(_svc().delete_intel_item(item_id))
	return _ok({"deleted": True})


@bp.post("/items/<item_id>/validate")
@handle_errors
def validate_item(item_id: str):
	"""POST /items/<id>/validate — mark item as validated."""
	return _ok(_run(_svc().validate_intel_item(item_id)))


@bp.post("/items/<item_id>/reject")
@handle_errors
def reject_item(item_id: str):
	"""POST /items/<id>/reject — reject item from pipeline."""
	return _ok(_run(_svc().reject_intel_item(item_id)))


# ─────────────────────────────────────────────────────────────────────────────
# FusionWorkspace
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/workspaces")
@handle_errors
def list_workspaces():
	"""GET /workspaces — list workspaces."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_workspaces(
		status=_qs("status"),
		workspace_type=_qs("workspace_type"),
		page=page,
		page_size=page_size,
	))
	return _ok(result.model_dump(mode="json"))


@bp.post("/workspaces")
@handle_errors
def create_workspace():
	"""POST /workspaces — create a new analytical workspace."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = FusionWorkspaceCreate(**body)
	ws = _run(svc.create_workspace(payload))
	return _ok(ws, 201)


@bp.get("/workspaces/<workspace_id>")
@handle_errors
def get_workspace(workspace_id: str):
	"""GET /workspaces/<id> — retrieve a workspace."""
	return _ok(_run(_svc().get_workspace(workspace_id)))


@bp.put("/workspaces/<workspace_id>")
@handle_errors
def update_workspace(workspace_id: str):
	"""PUT /workspaces/<id> — partial update."""
	svc = _svc()
	patch = FusionWorkspaceUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_workspace(workspace_id, patch)))


@bp.delete("/workspaces/<workspace_id>")
@handle_errors
def delete_workspace(workspace_id: str):
	"""DELETE /workspaces/<id> — soft delete."""
	_run(_svc().delete_workspace(workspace_id))
	return _ok({"deleted": True})


@bp.post("/workspaces/<workspace_id>/suspend")
@handle_errors
def suspend_workspace(workspace_id: str):
	"""POST /workspaces/<id>/suspend — suspend workspace."""
	return _ok(_run(_svc().suspend_workspace(workspace_id)))


@bp.post("/workspaces/<workspace_id>/close")
@handle_errors
def close_workspace(workspace_id: str):
	"""POST /workspaces/<id>/close — close workspace permanently."""
	return _ok(_run(_svc().close_workspace(workspace_id)))


@bp.get("/workspaces/<workspace_id>/summary")
@handle_errors
def workspace_summary(workspace_id: str):
	"""GET /workspaces/<id>/summary — workspace contents summary."""
	return _ok(_run(_svc().workspace_summary(workspace_id)))


@bp.post("/workspaces/<workspace_id>/fuse")
@handle_errors
def fuse_workspace(workspace_id: str):
	"""POST /workspaces/<id>/fuse — fuse all validated items in workspace."""
	body = request.get_json(force=True) or {}
	source_ids = body.get("source_ids")
	tw = body.get("time_window")
	time_window = tuple(tw) if tw and len(tw) == 2 else None
	result = _run(_svc().fuse_intelligence(workspace_id, source_ids, time_window))
	return _ok(result)


@bp.post("/workspaces/<workspace_id>/correlate-domains")
@handle_errors
def correlate_domains(workspace_id: str):
	"""POST /workspaces/<id>/correlate-domains — cross-domain correlation."""
	body = request.get_json(force=True) or {}
	result = _run(_svc().correlate_across_domains(
		workspace_id,
		osint_ids=body.get("osint_ids"),
		sigint_ids=body.get("sigint_ids"),
		humint_ids=body.get("humint_ids"),
		additional_domain_ids=body.get("additional_domain_ids"),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# CorrelationSet
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/correlations")
@handle_errors
def list_correlations():
	"""GET /correlations — list correlation sets."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_correlations(
		workspace_id=_qs("workspace_id"),
		status=_qs("status"),
		correlation_type=_qs("correlation_type"),
		page=page,
		page_size=page_size,
	))
	return _ok(result.model_dump(mode="json"))


@bp.post("/correlations")
@handle_errors
def create_correlation():
	"""POST /correlations — create a correlation set."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = CorrelationSetCreate(**body)
	corr = _run(svc.create_correlation(payload))
	return _ok(corr, 201)


@bp.get("/correlations/<correlation_id>")
@handle_errors
def get_correlation(correlation_id: str):
	"""GET /correlations/<id> — retrieve a correlation set."""
	return _ok(_run(_svc().get_correlation(correlation_id)))


@bp.put("/correlations/<correlation_id>")
@handle_errors
def update_correlation(correlation_id: str):
	"""PUT /correlations/<id> — partial update."""
	svc = _svc()
	patch = CorrelationSetUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_correlation(correlation_id, patch)))


@bp.delete("/correlations/<correlation_id>")
@handle_errors
def delete_correlation(correlation_id: str):
	"""DELETE /correlations/<id> — soft delete."""
	_run(_svc().delete_correlation(correlation_id))
	return _ok({"deleted": True})


@bp.post("/correlations/<correlation_id>/confirm")
@handle_errors
def confirm_correlation(correlation_id: str):
	"""POST /correlations/<id>/confirm — confirm a correlation."""
	return _ok(_run(_svc().confirm_correlation(correlation_id)))


@bp.post("/correlations/<correlation_id>/dispute")
@handle_errors
def dispute_correlation(correlation_id: str):
	"""POST /correlations/<id>/dispute — dispute a correlation."""
	return _ok(_run(_svc().dispute_correlation(correlation_id)))


# ─────────────────────────────────────────────────────────────────────────────
# AssessmentPicture
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/assessments")
@handle_errors
def list_assessments():
	"""GET /assessments — list assessment pictures."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_assessments(
		workspace_id=_qs("workspace_id"),
		risk_level=_qs("risk_level"),
		assessment_type=_qs("assessment_type"),
		page=page,
		page_size=page_size,
	))
	return _ok(result.model_dump(mode="json"))


@bp.post("/assessments")
@handle_errors
def create_assessment():
	"""POST /assessments — create a synthesised assessment picture."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = AssessmentPictureCreate(**body)
	assessment = _run(svc.create_assessment(payload))
	return _ok(assessment, 201)


@bp.get("/assessments/<assessment_id>")
@handle_errors
def get_assessment(assessment_id: str):
	"""GET /assessments/<id> — retrieve an assessment picture."""
	return _ok(_run(_svc().get_assessment(assessment_id)))


@bp.put("/assessments/<assessment_id>")
@handle_errors
def update_assessment(assessment_id: str):
	"""PUT /assessments/<id> — partial update."""
	svc = _svc()
	patch = AssessmentPictureUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_assessment(assessment_id, patch)))


@bp.delete("/assessments/<assessment_id>")
@handle_errors
def delete_assessment(assessment_id: str):
	"""DELETE /assessments/<id> — soft delete."""
	_run(_svc().delete_assessment(assessment_id))
	return _ok({"deleted": True})


@bp.post("/assessments/<assessment_id>/approve")
@handle_errors
def approve_assessment(assessment_id: str):
	"""POST /assessments/<id>/approve — approve an assessment picture."""
	body = request.get_json(force=True) or {}
	approver_id = body.get("approver_id", request.headers.get("X-Actor-ID", "approver"))
	return _ok(_run(_svc().approve_assessment(assessment_id, approver_id)))


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceProduct
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/products")
@handle_errors
def list_products():
	"""GET /products — list intelligence products."""
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
	return _ok(result.model_dump(mode="json"))


@bp.post("/products")
@handle_errors
def create_product():
	"""POST /products — create a finished intelligence product."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = IntelligenceProductCreate(**body)
	product = _run(svc.create_product(payload))
	return _ok(product, 201)


@bp.get("/products/<product_id>")
@handle_errors
def get_product(product_id: str):
	"""GET /products/<id> — retrieve a product."""
	return _ok(_run(_svc().get_product(product_id)))


@bp.put("/products/<product_id>")
@handle_errors
def update_product(product_id: str):
	"""PUT /products/<id> — partial update."""
	svc = _svc()
	patch = IntelligenceProductUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_product(product_id, patch)))


@bp.delete("/products/<product_id>")
@handle_errors
def delete_product(product_id: str):
	"""DELETE /products/<id> — soft delete."""
	_run(_svc().delete_product(product_id))
	return _ok({"deleted": True})


@bp.post("/products/<product_id>/submit")
@handle_errors
def submit_product(product_id: str):
	"""POST /products/<id>/submit — submit for peer review."""
	body = request.get_json(force=True) or {}
	reviewer_id = body.get("reviewer_id", "reviewer")
	return _ok(_run(_svc().submit_product_for_review(product_id, reviewer_id)))


@bp.post("/products/<product_id>/approve")
@handle_errors
def approve_product(product_id: str):
	"""POST /products/<id>/approve — approve a product under review."""
	body = request.get_json(force=True) or {}
	approver_id = body.get("approver_id", request.headers.get("X-Actor-ID", "approver"))
	return _ok(_run(_svc().approve_product(product_id, approver_id)))


@bp.post("/products/<product_id>/release")
@handle_errors
def release_product(product_id: str):
	"""POST /products/<id>/release — release an approved product."""
	body = request.get_json(force=True) or {}
	approval_ref = body.get("approval_reference", "")
	return _ok(_run(_svc().release_product(product_id, approval_ref)))


@bp.post("/products/<product_id>/recall")
@handle_errors
def recall_product(product_id: str):
	"""POST /products/<id>/recall — recall a released product."""
	return _ok(_run(_svc().recall_product(product_id)))


@bp.post("/products/<product_id>/disseminate")
@handle_errors
def disseminate_product(product_id: str):
	"""POST /products/<id>/disseminate — disseminate with TLP enforcement."""
	svc = _svc()
	body = request.get_json(force=True) or {}
	record = _run(svc.dissemination_with_tlp(
		product_id=product_id,
		audience=body.get("audience", ""),
		recipient_max_tlp=body.get("recipient_max_tlp", "TLP:AMBER"),
		approval_reference=body.get("approval_reference", ""),
		disseminated_by=body.get("disseminated_by", svc.actor_id),
		notes=body.get("notes", ""),
	))
	return _ok(record, 201)


@bp.get("/products/<product_id>/finished-intel")
@handle_errors
def finished_intel(product_id: str):
	"""GET /products/<id>/finished-intel — generate finished intelligence report."""
	svc = _svc()
	body = request.get_json(force=True) or {}
	workspace_id = body.get("workspace_id") or _qs("workspace_id", "")
	return _ok(_run(svc.generate_finished_intelligence(workspace_id, product_id)))


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticalJudgement
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/judgements")
@handle_errors
def list_judgements():
	"""GET /judgements — list analytical judgements."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_judgements(
		workspace_id=_qs("workspace_id"),
		judgement_type=_qs("judgement_type"),
		page=page,
		page_size=page_size,
	))
	return _ok(result.model_dump(mode="json"))


@bp.post("/judgements")
@handle_errors
def create_judgement():
	"""POST /judgements — record a calibrated analytical judgement."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = AnalyticalJudgementCreate(**body)
	return _ok(_run(svc.create_judgement(payload)), 201)


@bp.get("/judgements/<judgement_id>")
@handle_errors
def get_judgement(judgement_id: str):
	"""GET /judgements/<id> — retrieve a judgement."""
	return _ok(_run(_svc().get_judgement(judgement_id)))


@bp.put("/judgements/<judgement_id>")
@handle_errors
def update_judgement(judgement_id: str):
	"""PUT /judgements/<id> — partial update."""
	svc = _svc()
	patch = AnalyticalJudgementUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_judgement(judgement_id, patch)))


@bp.delete("/judgements/<judgement_id>")
@handle_errors
def delete_judgement(judgement_id: str):
	"""DELETE /judgements/<id> — soft delete."""
	_run(_svc().delete_judgement(judgement_id))
	return _ok({"deleted": True})


@bp.post("/judgements/<judgement_id>/challenge")
@handle_errors
def challenge_judgement(judgement_id: str):
	"""POST /judgements/<id>/challenge — register a red-team challenge."""
	body = request.get_json(force=True) or {}
	challenger_id = body.get("challenger_id", request.headers.get("X-Actor-ID", "challenger"))
	return _ok(_run(_svc().challenge_judgement(judgement_id, challenger_id)))


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/evidence")
@handle_errors
def list_evidence():
	"""GET /evidence — list evidence items."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_evidence(
		workspace_id=_qs("workspace_id"),
		evidence_type=_qs("evidence_type"),
		status=_qs("status"),
		page=page,
		page_size=page_size,
	))
	return _ok(result.model_dump(mode="json"))


@bp.post("/evidence")
@handle_errors
def create_evidence():
	"""POST /evidence — record a provenance-tracked evidence item."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = EvidenceCreate(**body)
	return _ok(_run(svc.create_evidence(payload)), 201)


@bp.get("/evidence/<evidence_id>")
@handle_errors
def get_evidence(evidence_id: str):
	"""GET /evidence/<id> — retrieve an evidence item."""
	return _ok(_run(_svc().get_evidence(evidence_id)))


@bp.put("/evidence/<evidence_id>")
@handle_errors
def update_evidence(evidence_id: str):
	"""PUT /evidence/<id> — partial update."""
	svc = _svc()
	patch = EvidenceUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_evidence(evidence_id, patch)))


@bp.delete("/evidence/<evidence_id>")
@handle_errors
def delete_evidence(evidence_id: str):
	"""DELETE /evidence/<id> — soft delete."""
	_run(_svc().delete_evidence(evidence_id))
	return _ok({"deleted": True})


@bp.post("/evidence/<evidence_id>/verify")
@handle_errors
def verify_evidence(evidence_id: str):
	"""POST /evidence/<id>/verify — mark evidence as verified."""
	return _ok(_run(_svc().verify_evidence(evidence_id)))


@bp.post("/evidence/<evidence_id>/challenge")
@handle_errors
def challenge_evidence(evidence_id: str):
	"""POST /evidence/<id>/challenge — challenge evidence."""
	return _ok(_run(_svc().challenge_evidence(evidence_id)))


@bp.post("/evidence/<evidence_id>/discredit")
@handle_errors
def discredit_evidence(evidence_id: str):
	"""POST /evidence/<id>/discredit — discredit evidence."""
	return _ok(_run(_svc().discredit_evidence(evidence_id)))


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisTest
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/hypotheses")
@handle_errors
def list_hypotheses():
	"""GET /hypotheses — list hypothesis tests."""
	svc = _svc()
	page, page_size = _page()
	result = _run(svc.list_hypotheses(
		workspace_id=_qs("workspace_id"),
		status=_qs("status"),
		sat_method=_qs("sat_method"),
		page=page,
		page_size=page_size,
	))
	return _ok(result.model_dump(mode="json"))


@bp.post("/hypotheses")
@handle_errors
def create_hypothesis():
	"""POST /hypotheses — create a structured hypothesis test."""
	svc = _svc()
	body = request.get_json(force=True)
	body.setdefault("tenant_id", svc.tenant_id)
	payload = HypothesisTestCreate(**body)
	return _ok(_run(svc.create_hypothesis(payload)), 201)


@bp.get("/hypotheses/<hypothesis_id>")
@handle_errors
def get_hypothesis(hypothesis_id: str):
	"""GET /hypotheses/<id> — retrieve a hypothesis test."""
	return _ok(_run(_svc().get_hypothesis(hypothesis_id)))


@bp.put("/hypotheses/<hypothesis_id>")
@handle_errors
def update_hypothesis(hypothesis_id: str):
	"""PUT /hypotheses/<id> — update with new evidence or conclusion."""
	svc = _svc()
	patch = HypothesisTestUpdate(**request.get_json(force=True))
	return _ok(_run(svc.update_hypothesis(hypothesis_id, patch)))


@bp.delete("/hypotheses/<hypothesis_id>")
@handle_errors
def delete_hypothesis(hypothesis_id: str):
	"""DELETE /hypotheses/<id> — soft delete."""
	_run(_svc().delete_hypothesis(hypothesis_id))
	return _ok({"deleted": True})


# ─────────────────────────────────────────────────────────────────────────────
# Structured Analytic Techniques
# ─────────────────────────────────────────────────────────────────────────────

@bp.post("/sat/ach")
@handle_errors
def sat_ach():
	"""POST /sat/ach — Analysis of Competing Hypotheses."""
	svc = _svc()
	body = request.get_json(force=True) or {}
	workspace_id = body.get("workspace_id", "")
	hypotheses = body.get("hypotheses", [])
	evidence_items = body.get("evidence_items", [])
	result = _run(svc.analysis_of_competing_hypotheses(workspace_id, hypotheses, evidence_items))
	return _ok(result)


@bp.post("/sat/kac")
@handle_errors
def sat_kac():
	"""POST /sat/kac — Key Assumptions Check."""
	svc = _svc()
	body = request.get_json(force=True) or {}
	workspace_id = body.get("workspace_id", "")
	assumptions = body.get("assumptions", [])
	confidence_scores = body.get("confidence_scores", [])
	result = _run(svc.key_assumptions_check(workspace_id, assumptions, confidence_scores))
	return _ok(result)


@bp.post("/sat/ace")
@handle_errors
def sat_ace():
	"""POST /sat/ace — ACE (Analysis, Confidence, Evidence) method."""
	svc = _svc()
	body = request.get_json(force=True) or {}
	result = _run(svc.ace_method(
		workspace_id=body.get("workspace_id", ""),
		analysis_statement=body.get("analysis_statement", ""),
		confidence_score=float(body.get("confidence_score", 0.5)),
		evidence_ids=body.get("evidence_ids", []),
	))
	return _ok(result)


@bp.post("/sat/confidence-calibration")
@handle_errors
def confidence_calibration():
	"""POST /sat/confidence-calibration — Bayesian confidence calibration."""
	svc = _svc()
	body = request.get_json(force=True) or {}
	result = _run(svc.confidence_calibration(
		prior=float(body.get("prior", 0.5)),
		likelihood_given_true=float(body.get("likelihood_given_true", 0.7)),
		likelihood_given_false=float(body.get("likelihood_given_false", 0.3)),
	))
	return _ok(result)


@bp.post("/sat/apply")
@handle_errors
def apply_sat():
	"""POST /sat/apply — apply any named SAT to a workspace."""
	svc = _svc()
	body = request.get_json(force=True) or {}
	result = _run(svc.apply_structured_analytic_techniques(
		workspace_id=body.get("workspace_id", ""),
		method=body.get("method", "analysis_of_competing_hypotheses"),
		hypotheses=body.get("hypotheses", []),
		evidence_items=body.get("evidence_items", []),
		assumptions=body.get("assumptions"),
		assumption_confidences=body.get("assumption_confidences"),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

@bp.get("/reports/dashboard")
@handle_errors
def report_dashboard():
	"""GET /reports/dashboard — tenant-level fusion dashboard."""
	return _ok(_run(_svc().dashboard_report()))


@bp.get("/reports/workspace/<workspace_id>")
@handle_errors
def report_workspace(workspace_id: str):
	"""GET /reports/workspace/<id> — workspace summary report."""
	return _ok(_run(_svc().workspace_summary(workspace_id)))
