"""Flask Blueprint REST API for APG Anti-Money Laundering.

All endpoints enforce tenant isolation via X-Tenant-ID header.
Async coroutines are dispatched via _run() for Flask sync compat.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .models import (
		AMLAlertCreate,
		AMLAlertUpdate,
		AMLCaseCreate,
		AMLCaseUpdate,
		AlertSeverity,
		AlertType,
		CTRCreate,
		CaseStatus,
		InvestigationNoteCreate,
		RegulatoryFilingCreate,
		RuleCondition,
		SARCreate,
		TransactionMonitoringRuleCreate,
		TransactionMonitoringRuleUpdate,
		WatchlistMatchStatus,
	)
	from .service import AMLService
except ImportError:  # pragma: no cover — direct file load
	from models import (  # type: ignore
		AMLAlertCreate,
		AMLAlertUpdate,
		AMLCaseCreate,
		AMLCaseUpdate,
		AlertSeverity,
		AlertType,
		CTRCreate,
		CaseStatus,
		InvestigationNoteCreate,
		RegulatoryFilingCreate,
		RuleCondition,
		SARCreate,
		TransactionMonitoringRuleCreate,
		TransactionMonitoringRuleUpdate,
		WatchlistMatchStatus,
	)
	from service import AMLService  # type: ignore

aml_bp = Blueprint("aml", __name__, url_prefix="/api/v1/aml")

# Module-level singleton — replaced in tests via monkey-patch
_SERVICE: AMLService | None = None


def _get_service() -> AMLService:
	global _SERVICE
	if _SERVICE is None:
		_SERVICE = AMLService()
	return _SERVICE


def _svc() -> AMLService:
	"""Return service scoped to the current request's tenant/actor."""
	svc = _get_service()
	svc.tenant_id = request.headers.get("X-Tenant-ID", "default")
	svc.actor_id = request.headers.get("X-Actor-ID", "system")
	return svc


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask view."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _ok(data: Any, status: int = 200) -> Any:
	if hasattr(data, "model_dump"):
		return jsonify(data.model_dump(mode="json")), status
	if isinstance(data, list):
		items = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in data]
		return jsonify({"items": items, "count": len(items)}), status
	return jsonify(data), status


def _err(msg: str, code: int = 400) -> Any:
	return jsonify({"error": str(msg)}), code


def _body() -> dict[str, Any]:
	data = request.get_json(silent=True) or {}
	assert isinstance(data, dict), "request body must be JSON object"
	return data


def _dt(val: str) -> datetime:
	return datetime.fromisoformat(val)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@aml_bp.get("/health")
def health() -> Any:
	return _ok({"status": "ok", "capability": "fintech_aml", "version": "2.0.0"})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@aml_bp.get("/dashboard")
def dashboard() -> Any:
	try:
		return _ok(_run(_svc().dashboard_summary()))
	except Exception as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Monitoring rules
# ---------------------------------------------------------------------------

@aml_bp.get("/rules/list")
def list_rules() -> Any:
	try:
		enabled_only = request.args.get("enabled_only", "false").lower() == "true"
		return _ok(_run(_svc().list_rules(enabled_only=enabled_only)))
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/rules/create")
def create_rule() -> Any:
	try:
		b = _body()
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		b.setdefault("created_by", request.headers.get("X-Actor-ID", "system"))
		if "conditions" in b:
			b["conditions"] = [c if isinstance(c, RuleCondition) else RuleCondition(**c) for c in b["conditions"]]
		rule = _run(_svc().create_rule(TransactionMonitoringRuleCreate(**b)))
		return _ok(rule, 201)
	except (AssertionError, ValueError, TypeError) as exc:
		return _err(exc)


@aml_bp.get("/rules/<rule_id>")
def get_rule(rule_id: str) -> Any:
	try:
		return _ok(_run(_svc().get_rule(rule_id)))
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.put("/rules/<rule_id>")
def update_rule(rule_id: str) -> Any:
	try:
		return _ok(_run(_svc().update_rule(rule_id, TransactionMonitoringRuleUpdate(**_body()))))
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.delete("/rules/<rule_id>")
def delete_rule(rule_id: str) -> Any:
	try:
		_run(_svc().delete_rule(rule_id))
		return _ok({"deleted": True})
	except AssertionError as exc:
		return _err(exc, 404)


# ---------------------------------------------------------------------------
# Transaction monitoring
# ---------------------------------------------------------------------------

@aml_bp.post("/monitor")
def monitor_transaction() -> Any:
	try:
		b = _body()
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		return _ok(_run(_svc().monitor_transaction(b)), 201)
	except (AssertionError, ValueError, PermissionError) as exc:
		return _err(exc)


@aml_bp.post("/evaluate")
def evaluate_rules() -> Any:
	try:
		return _ok(_run(_svc().evaluate_rules(_body())))
	except Exception as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@aml_bp.get("/alerts/list")
def list_alerts() -> Any:
	try:
		alerts = _run(_svc().list_alerts(
			status=request.args.get("status"),
			severity=request.args.get("severity"),
			alert_type=request.args.get("alert_type"),
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		))
		return _ok(alerts)
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/alerts/create")
def create_alert() -> Any:
	try:
		b = _body()
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		b.setdefault("created_by", request.headers.get("X-Actor-ID", "system"))
		return _ok(_run(_svc().create_alert(AMLAlertCreate(**b))), 201)
	except (AssertionError, ValueError, PermissionError) as exc:
		return _err(exc)


@aml_bp.get("/alerts/<alert_id>")
def get_alert(alert_id: str) -> Any:
	try:
		return _ok(_run(_svc().get_alert(alert_id)))
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.put("/alerts/<alert_id>")
def update_alert(alert_id: str) -> Any:
	try:
		return _ok(_run(_svc().update_alert(alert_id, AMLAlertUpdate(**_body()))))
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.delete("/alerts/<alert_id>")
def delete_alert(alert_id: str) -> Any:
	try:
		_run(_svc().delete_alert(alert_id))
		return _ok({"deleted": True})
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.post("/alerts/<alert_id>/approve")
def approve_alert(alert_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().approve_alert(alert_id, b.get("reviewer_id", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.post("/alerts/<alert_id>/reject")
def reject_alert(alert_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().reject_alert(alert_id, b.get("reviewer_id", ""), b.get("disposition", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.post("/alerts/<alert_id>/close")
def close_alert(alert_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().close_alert(alert_id, b.get("disposition", ""), b.get("reviewer_id", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@aml_bp.get("/cases/list")
def list_cases() -> Any:
	try:
		cases = _run(_svc().list_cases(
			status=request.args.get("status"),
			investigator_id=request.args.get("investigator_id"),
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		))
		return _ok(cases)
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/cases/create")
def create_case() -> Any:
	try:
		b = _body()
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		b.setdefault("created_by", request.headers.get("X-Actor-ID", "system"))
		return _ok(_run(_svc().create_case(AMLCaseCreate(**b))), 201)
	except (AssertionError, ValueError, PermissionError) as exc:
		return _err(exc)


@aml_bp.get("/cases/<case_id>")
def get_case(case_id: str) -> Any:
	try:
		return _ok(_run(_svc().get_case(case_id)))
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.put("/cases/<case_id>")
def update_case(case_id: str) -> Any:
	try:
		return _ok(_run(_svc().update_case(case_id, AMLCaseUpdate(**_body()))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.delete("/cases/<case_id>")
def delete_case(case_id: str) -> Any:
	try:
		_run(_svc().delete_case(case_id))
		return _ok({"deleted": True})
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.post("/cases/<alert_id>/open")
def case_management(alert_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().case_management(alert_id, b.get("investigator_id", ""))), 201)
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.post("/cases/<case_id>/investigate")
def investigate_case(case_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().investigate_case(case_id, b.get("note", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.post("/cases/<case_id>/close")
def close_case(case_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().close_case(case_id, CaseStatus(b.get("status", "closed_no_action")), b.get("notes", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.get("/cases/<case_id>/notes")
def list_notes(case_id: str) -> Any:
	try:
		return _ok(_run(_svc().list_notes(case_id)))
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/cases/<case_id>/notes")
def add_note(case_id: str) -> Any:
	try:
		b = _body()
		b["case_id"] = case_id
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		b.setdefault("created_by", request.headers.get("X-Actor-ID", "system"))
		return _ok(_run(_svc().add_note(InvestigationNoteCreate(**b))), 201)
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------

@aml_bp.get("/sar/list")
def list_sars() -> Any:
	try:
		return _ok(_run(_svc().list_sars(status=request.args.get("status"))))
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/sar/<case_id>/file")
def file_sar(case_id: str) -> Any:
	try:
		b = _body()
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		b.setdefault("created_by", request.headers.get("X-Actor-ID", "system"))
		for k in ("suspicious_activity_start", "suspicious_activity_end"):
			if isinstance(b.get(k), str):
				b[k] = _dt(b[k])
		return _ok(_run(_svc().file_sar(case_id, SARCreate(**b))), 201)
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.get("/sar/<sar_id>")
def get_sar(sar_id: str) -> Any:
	try:
		return _ok(_run(_svc().get_sar(sar_id)))
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.post("/sar/<sar_id>/approve")
def approve_sar(sar_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().approve_sar(sar_id, b.get("approved_by", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.post("/sar/<sar_id>/submit")
def submit_sar(sar_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().submit_sar(sar_id, b.get("filing_reference", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.post("/sar/<sar_id>/reject")
def reject_sar(sar_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().reject_sar(sar_id, b.get("reason", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# CTR
# ---------------------------------------------------------------------------

@aml_bp.get("/ctr/list")
def list_ctrs() -> Any:
	try:
		return _ok(_run(_svc().list_ctrs(status=request.args.get("status"))))
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/ctr/<transaction_id>/file")
def file_ctr(transaction_id: str) -> Any:
	try:
		b = _body()
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		b.setdefault("created_by", request.headers.get("X-Actor-ID", "system"))
		if isinstance(b.get("transaction_date"), str):
			b["transaction_date"] = _dt(b["transaction_date"])
		return _ok(_run(_svc().file_ctr(transaction_id, CTRCreate(**b))), 201)
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.get("/ctr/<ctr_id>")
def get_ctr(ctr_id: str) -> Any:
	try:
		return _ok(_run(_svc().get_ctr(ctr_id)))
	except AssertionError as exc:
		return _err(exc, 404)


@aml_bp.post("/ctr/<ctr_id>/submit")
def submit_ctr(ctr_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().submit_ctr(ctr_id, b.get("filing_reference", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@aml_bp.post("/watchlist/screen")
def watchlist_screening() -> Any:
	try:
		b = _body()
		matches = _run(_svc().watchlist_screening(
			subject_reference=b.get("subject_reference", ""),
			subject_name=b.get("subject_name", ""),
			kyc_profile_id=b.get("kyc_profile_id"),
			lists=b.get("lists"),
		))
		return _ok(matches)
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.get("/watchlist/list")
def list_watchlist_matches() -> Any:
	try:
		return _ok(_run(_svc().list_watchlist_matches(status=request.args.get("status"))))
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/watchlist/<match_id>/review")
def review_watchlist_match(match_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().review_watchlist_match(
			match_id,
			WatchlistMatchStatus(b.get("status", "confirmed")),
			b.get("reviewer_id", ""),
		)))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Network & patterns
# ---------------------------------------------------------------------------

@aml_bp.get("/network/<customer_id>")
def network_analysis(customer_id: str) -> Any:
	try:
		result = _run(_svc().network_analysis(customer_id))
		return _ok(result.model_dump(mode="json"))
	except Exception as exc:
		return _err(exc)


@aml_bp.get("/patterns/<customer_id>")
def pattern_detection(customer_id: str) -> Any:
	try:
		lookback = int(request.args.get("lookback_days", 90))
		result = _run(_svc().pattern_detection(customer_id, lookback_days=lookback))
		return _ok(result.model_dump(mode="json"))
	except Exception as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Risk segmentation
# ---------------------------------------------------------------------------

@aml_bp.post("/risk/segment")
def risk_segmentation() -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().risk_segmentation(
			subject_reference=b.get("subject_reference", ""),
			kyc_profile_id=b.get("kyc_profile_id"),
			contributing_factors=b.get("contributing_factors"),
			risk_score=int(b.get("risk_score", 0)),
		)))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Regulatory reports
# ---------------------------------------------------------------------------

@aml_bp.get("/reports/<report_type>")
def regulatory_reporting(report_type: str) -> Any:
	try:
		jurisdiction = request.args.get("jurisdiction", "US")
		ps = request.args.get("period_start", "")
		pe = request.args.get("period_end", "")
		period_start = _dt(ps) if ps else datetime(datetime.utcnow().year, 1, 1)
		period_end = _dt(pe) if pe else datetime.utcnow()
		report = _run(_svc().regulatory_reporting(
			jurisdiction=jurisdiction,
			period_start=period_start,
			period_end=period_end,
			report_type=report_type,
		))
		return _ok(report.model_dump(mode="json"))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------

@aml_bp.get("/filings/list")
def list_filings() -> Any:
	try:
		return _ok(_run(_svc().list_filings(jurisdiction=request.args.get("jurisdiction"))))
	except Exception as exc:
		return _err(exc)


@aml_bp.post("/filings/create")
def create_filing() -> Any:
	try:
		b = _body()
		b.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
		b.setdefault("created_by", request.headers.get("X-Actor-ID", "system"))
		for k in ("period_start", "period_end"):
			if isinstance(b.get(k), str):
				b[k] = _dt(b[k])
		return _ok(_run(_svc().create_filing(RegulatoryFilingCreate(**b))), 201)
	except (AssertionError, ValueError) as exc:
		return _err(exc)


@aml_bp.post("/filings/<filing_id>/submit")
def submit_filing(filing_id: str) -> Any:
	try:
		b = _body()
		return _ok(_run(_svc().submit_filing(filing_id, b.get("submission_reference", ""))))
	except (AssertionError, ValueError) as exc:
		return _err(exc)


# ---------------------------------------------------------------------------
# Backward-compat process-local helpers (used by existing app.py / tests)
# ---------------------------------------------------------------------------

def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	"""Process-local status — used by aml_runtime and app.py."""
	try:
		from .capability_contract import get_capability_contract
	except ImportError:
		from capability_contract import get_capability_contract  # type: ignore
	contract = get_capability_contract(tenant_id)
	svc = AMLService(tenant_id=tenant_id)
	summary = asyncio.run(svc.dashboard_summary())
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"alert_count": summary["alert_count"],
		"case_count": summary["case_count"],
	}


# ---------------------------------------------------------------------------
# Legacy process-local helpers used by test_package_contract.py
# These delegate to the AntiMoneyLaunderingService sync shim so existing
# tests that call api.monitor_transaction(payload) continue to pass.
# ---------------------------------------------------------------------------

def service() -> Any:
	"""Return the legacy sync service instance."""
	try:
		from .service import AntiMoneyLaunderingService
	except ImportError:
		from service import AntiMoneyLaunderingService  # type: ignore
	if not hasattr(service, "_instance"):
		service._instance = AntiMoneyLaunderingService()
	return service._instance


def monitor_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return service().monitor_transaction(
		str(payload["transaction_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["subject_reference"]),
		str(payload.get("kyc_profile_id") or ""),
		payload.get("amount", 0),
		str(payload.get("currency") or ""),
		str(payload.get("source_capability") or "fintech_payments"),
		str(payload.get("source_reference") or ""),
		payload.get("risk_score", 0),
		bool(payload.get("sanctions_hit", False)),
		bool(payload.get("velocity_indicator", False)),
		str(payload.get("review_id") or ""),
		bool(payload.get("policy_attached", True)),
	)


def create_alert(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_alert(
		str(payload["alert_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["alert_type"]),
		str(payload.get("severity") or "medium"),
		str(payload["subject_reference"]),
		list(payload.get("evidence_references") or []),
	)


def create_alert_from_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_alert_from_transaction(
		str(payload["alert_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["transaction_id"]),
		payload.get("alert_type"),
	)


def triage_alert(payload: dict[str, Any]) -> dict[str, Any]:
	return service().triage_alert(
		str(payload["alert_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["action"]),
		str(payload.get("disposition") or ""),
		str(payload.get("reviewer_id") or ""),
	)


def open_case(payload: dict[str, Any]) -> dict[str, Any]:
	return service().open_case(
		str(payload["case_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["alert_id"]),
		str(payload.get("case_type") or "transaction_monitoring"),
		str(payload.get("investigator_id") or ""),
		list(payload.get("evidence_references") or []),
	)


def draft_sar(payload: dict[str, Any]) -> dict[str, Any]:
	return service().draft_sar(
		str(payload["sar_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["case_id"]),
		str(payload["subject_reference"]),
		str(payload.get("jurisdiction") or ""),
		str(payload.get("narrative") or ""),
		list(payload.get("evidence_references") or []),
		str(payload.get("approved_by") or ""),
	)


def register_aml_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_aml_agent(
		str(payload["agent_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload.get("name") or payload["agent_id"]),
		str(payload.get("runtime") or "codex"),
		str(payload.get("role") or "aml_ops_reviewer"),
		str(payload.get("scope") or "triage AML alerts"),
	)


def list_alerts(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service().list_alerts(tenant_id)
