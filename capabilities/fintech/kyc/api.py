"""APG Know Your Customer — REST API Blueprint.

Mounts at ``/api/kyc``.  Every route is tenant-scoped via the
``X-Tenant-ID`` header (falls back to query-param ``tenant_id``,
then to the literal ``"default"``).

All responses follow the envelope:
    { "ok": true,  "data": {...} }
    { "ok": false, "error": "...", "code": "..." }

HTTP status codes:
    200  success (GET / action)
    201  created  (POST create)
    400  validation / rule violation
    404  not found
    409  conflict (duplicate)
    422  unprocessable entity
    500  internal error
"""
from __future__ import annotations

import asyncio
import traceback
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, jsonify, request, Response

try:
	from .service import KYCService
	from .domain.rules import RuleViolation
except ImportError:  # pragma: no cover
	from service import KYCService  # type: ignore[no-redef]
	from domain.rules import RuleViolation  # type: ignore[no-redef]


kyc_bp = Blueprint("kyc", __name__, url_prefix="/api/kyc")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tenant() -> str:
	return (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or "default"
	)


def _actor() -> str:
	return (
		request.headers.get("X-Actor-ID")
		or request.args.get("actor_id")
		or "api"
	)


def _svc() -> KYCService:
	return KYCService(tenant_id=_tenant(), actor_id=_actor())


def _ok(data: Any, status: int = 200) -> Response:
	return jsonify({"ok": True, "data": data}), status


def _err(message: str, code: str = "error", status: int = 400) -> Response:
	return jsonify({"ok": False, "error": message, "code": code}), status


def _run(coro: Any) -> Any:
	"""Execute an async coroutine from sync Flask context."""
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _handle(fn: Callable) -> Callable:
	"""Decorator: convert async KYC service exceptions to HTTP responses."""
	@wraps(fn)
	def wrapper(*args: Any, **kwargs: Any) -> Any:
		try:
			return fn(*args, **kwargs)
		except RuleViolation as exc:
			return _err(exc.reason, exc.rule_name, 400)
		except KeyError as exc:
			return _err(str(exc), "not_found", 404)
		except ValueError as exc:
			return _err(str(exc), "validation_error", 422)
		except PermissionError as exc:
			return _err(str(exc), "permission_denied", 403)
		except Exception as exc:
			traceback.print_exc()
			return _err(f"internal error: {exc}", "internal_error", 500)
	return wrapper


def _body() -> dict[str, Any]:
	data = request.get_json(silent=True) or {}
	return data


# ─────────────────────────────────────────────────────────────────────────────
# Applications
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.get("/applications")
@_handle
def list_applications() -> Response:
	"""GET /api/kyc/applications — list applications for tenant."""
	filters = {k: v for k, v in request.args.items()
	           if k not in ("tenant_id", "actor_id", "limit", "offset")}
	result = _run(_svc().list_applications(filters or None))
	return _ok(result)


@kyc_bp.post("/applications")
@_handle
def create_application() -> Response:
	"""POST /api/kyc/applications — start a new KYC application."""
	b = _body()
	result = _run(_svc().start_kyc_application(
		customer_id=b["customer_id"],
		customer_type=b["customer_type"],
		jurisdiction=b["jurisdiction"],
		legal_name=b.get("legal_name", ""),
		consent_reference=b.get("consent_reference", ""),
		kyc_tier=b.get("kyc_tier", "standard"),
		is_refugee=bool(b.get("is_refugee", False)),
		is_informal_sector=bool(b.get("is_informal_sector", False)),
		preferred_language=b.get("preferred_language", "en"),
		metadata=b.get("metadata"),
	))
	return _ok(result, 201)


@kyc_bp.get("/applications/<application_id>")
@_handle
def get_application(application_id: str) -> Response:
	"""GET /api/kyc/applications/<id> — retrieve a single application."""
	result = _run(_svc().get_application(application_id))
	return _ok(result)


@kyc_bp.put("/applications/<application_id>")
@_handle
def update_application(application_id: str) -> Response:
	"""PUT /api/kyc/applications/<id> — patch mutable fields."""
	b = _body()
	result = _run(_svc().update_application(application_id, **b))
	return _ok(result)


@kyc_bp.delete("/applications/<application_id>")
@_handle
def delete_application(application_id: str) -> Response:
	"""DELETE /api/kyc/applications/<id> — soft delete."""
	svc = _svc()
	record = _run(svc.get_application(application_id))
	record["is_deleted"] = True
	_run(svc._store.put("kyc_applications", record))
	return _ok({"id": application_id, "deleted": True})


@kyc_bp.post("/applications/<application_id>/approve")
@_handle
def approve_application(application_id: str) -> Response:
	"""POST /api/kyc/applications/<id>/approve — approve after all checks pass."""
	b = _body()
	result = _run(_svc().approve_application(
		application_id,
		reviewer_id=b.get("reviewer_id", _actor()),
		notes=b.get("notes", ""),
	))
	return _ok(result)


@kyc_bp.post("/applications/<application_id>/reject")
@_handle
def reject_application(application_id: str) -> Response:
	"""POST /api/kyc/applications/<id>/reject — reject with mandatory reason."""
	b = _body()
	result = _run(_svc().reject_application(
		application_id,
		reason=b["reason"],
		reviewer_id=b.get("reviewer_id", _actor()),
	))
	return _ok(result)


@kyc_bp.post("/applications/<application_id>/request-docs")
@_handle
def request_additional_docs(application_id: str) -> Response:
	"""POST /api/kyc/applications/<id>/request-docs — pause and request more docs."""
	b = _body()
	result = _run(_svc().request_additional_docs(
		application_id,
		required_docs=b["required_docs"],
		message=b.get("message", ""),
	))
	return _ok(result)


@kyc_bp.post("/applications/<application_id>/assign-reviewer")
@_handle
def assign_reviewer(application_id: str) -> Response:
	"""POST /api/kyc/applications/<id>/assign-reviewer."""
	b = _body()
	result = _run(_svc().assign_reviewer(
		application_id,
		reviewer_id=b["reviewer_id"],
	))
	return _ok(result)


@kyc_bp.post("/applications/<application_id>/risk-score")
@_handle
def calculate_risk_score(application_id: str) -> Response:
	"""POST /api/kyc/applications/<id>/risk-score — compute composite risk score."""
	result = _run(_svc().calculate_risk_score(application_id))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.post("/documents")
@_handle
def upload_document() -> Response:
	"""POST /api/kyc/documents — register a document upload."""
	b = _body()
	result = _run(_svc().upload_document(
		application_id=b["application_id"],
		doc_type=b["doc_type"],
		file_metadata=b["file_metadata"],
		uploaded_by=b.get("uploaded_by", _actor()),
	))
	return _ok(result, 201)


@kyc_bp.post("/documents/<document_id>/verify-authenticity")
@_handle
def verify_document_authenticity(document_id: str) -> Response:
	"""POST /api/kyc/documents/<id>/verify-authenticity."""
	result = _run(_svc().verify_document_authenticity(document_id))
	return _ok(result)


@kyc_bp.post("/documents/<document_id>/extract")
@_handle
def extract_document_data(document_id: str) -> Response:
	"""POST /api/kyc/documents/<id>/extract — run OCR extraction."""
	result = _run(_svc().extract_document_data(document_id))
	return _ok(result)


@kyc_bp.get("/documents/<document_id>/expiry")
@_handle
def check_document_expiry(document_id: str) -> Response:
	"""GET /api/kyc/documents/<id>/expiry."""
	result = _run(_svc().check_document_expiry(document_id))
	return _ok(result)


@kyc_bp.post("/documents/match")
@_handle
def document_match_check() -> Response:
	"""POST /api/kyc/documents/match — cross-check name consistency."""
	b = _body()
	result = _run(_svc().document_match_check(b["doc1_id"], b["doc2_id"]))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Verification endpoints
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.post("/verify/national-id")
@_handle
def verify_national_id() -> Response:
	"""POST /api/kyc/verify/national-id."""
	b = _body()
	result = _run(_svc().verify_national_id(
		id_number=b["id_number"],
		country=b["country"],
		name=b["name"],
		dob=b.get("dob"),
	))
	return _ok(result)


@kyc_bp.post("/verify/passport")
@_handle
def verify_passport() -> Response:
	"""POST /api/kyc/verify/passport."""
	b = _body()
	result = _run(_svc().verify_passport(
		passport_number=b["passport_number"],
		country=b["country"],
		mrz_data=b.get("mrz_data"),
	))
	return _ok(result)


@kyc_bp.post("/verify/drivers-license")
@_handle
def verify_drivers_license() -> Response:
	"""POST /api/kyc/verify/drivers-license."""
	b = _body()
	result = _run(_svc().verify_drivers_license(
		license_number=b["license_number"],
		country=b["country"],
		name=b.get("name", ""),
	))
	return _ok(result)


@kyc_bp.post("/verify/birth-certificate")
@_handle
def verify_birth_certificate() -> Response:
	"""POST /api/kyc/verify/birth-certificate."""
	b = _body()
	result = _run(_svc().verify_birth_certificate(
		cert_number=b["cert_number"],
		country=b["country"],
		full_name=b.get("full_name", ""),
		date_of_birth=b.get("date_of_birth"),
	))
	return _ok(result)


@kyc_bp.post("/verify/utility-bill")
@_handle
def verify_utility_bill() -> Response:
	"""POST /api/kyc/verify/utility-bill."""
	b = _body()
	result = _run(_svc().verify_utility_bill(
		document_id=b["document_id"],
		customer_name=b["customer_name"],
		address=b["address"],
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Biometrics
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.post("/biometrics/liveness")
@_handle
def liveness_check() -> Response:
	"""POST /api/kyc/biometrics/liveness — anti-spoofing liveness detection."""
	b = _body()
	result = _run(_svc().perform_liveness_check(
		session_id=b["session_id"],
		video_frames=b.get("video_frames", []),
	))
	return _ok(result)


@kyc_bp.post("/biometrics/face-match")
@_handle
def face_match() -> Response:
	"""POST /api/kyc/biometrics/face-match — document photo vs. live selfie."""
	b = _body()
	result = _run(_svc().face_match_id_to_selfie(
		document_id=b["document_id"],
		selfie_metadata=b["selfie_metadata"],
	))
	return _ok(result)


@kyc_bp.post("/biometrics/fingerprint")
@_handle
def fingerprint_check() -> Response:
	"""POST /api/kyc/biometrics/fingerprint."""
	b = _body()
	result = _run(_svc().fingerprint_check(fingerprint_data=b))
	return _ok(result)


@kyc_bp.post("/biometrics/voice")
@_handle
def voice_biometric() -> Response:
	"""POST /api/kyc/biometrics/voice."""
	b = _body()
	result = _run(_svc().voice_biometric(
		voice_sample=b.get("voice_sample", b),
		enrolled_voice=b.get("enrolled_voice"),
	))
	return _ok(result)


@kyc_bp.post("/biometrics/deduplication")
@_handle
def biometric_deduplication() -> Response:
	"""POST /api/kyc/biometrics/deduplication — duplicate biometric check."""
	b = _body()
	result = _run(_svc().biometric_deduplication(customer_id=b["customer_id"]))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Screening
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.post("/screening/pep")
@_handle
def pep_screening() -> Response:
	"""POST /api/kyc/screening/pep — PEP list screening."""
	b = _body()
	result = _run(_svc().pep_screening(
		name=b["name"],
		dob=b.get("dob"),
		nationality=b.get("nationality", ""),
		aliases=b.get("aliases"),
		application_id=b.get("application_id", ""),
		match_threshold=float(b.get("match_threshold", 0.85)),
	))
	return _ok(result)


@kyc_bp.post("/screening/sanctions")
@_handle
def sanctions_screening() -> Response:
	"""POST /api/kyc/screening/sanctions — OFAC/UN/EU/AU sanctions screening."""
	b = _body()
	result = _run(_svc().sanctions_screening(
		name=b["name"],
		nationality=b.get("nationality", ""),
		id_number=b.get("id_number", ""),
		application_id=b.get("application_id", ""),
		lists=b.get("lists"),
		match_threshold=float(b.get("match_threshold", 0.85)),
	))
	return _ok(result)


@kyc_bp.post("/screening/adverse-media")
@_handle
def adverse_media_screening() -> Response:
	"""POST /api/kyc/screening/adverse-media."""
	b = _body()
	result = _run(_svc().adverse_media_screening(
		name=b["name"],
		aliases=b.get("aliases"),
		application_id=b.get("application_id", ""),
		categories=b.get("categories"),
	))
	return _ok(result)


@kyc_bp.post("/screening/watchlist")
@_handle
def watchlist_screening() -> Response:
	"""POST /api/kyc/screening/watchlist — combined PEP + sanctions + adverse media."""
	b = _body()
	result = _run(_svc().watchlist_screening(
		name=b["name"],
		id_number=b.get("id_number", ""),
		dob=b.get("dob"),
		application_id=b.get("application_id", ""),
	))
	return _ok(result)


@kyc_bp.post("/screening/interpol")
@_handle
def interpol_check() -> Response:
	"""POST /api/kyc/screening/interpol."""
	b = _body()
	result = _run(_svc().interpol_check(
		name=b["name"],
		nationality=b["nationality"],
	))
	return _ok(result)


@kyc_bp.post("/screening/local-watchlist")
@_handle
def local_watchlist_check() -> Response:
	"""POST /api/kyc/screening/local-watchlist — country-level FIU watchlists."""
	b = _body()
	result = _run(_svc().local_watchlist_check(
		name=b["name"],
		id_number=b.get("id_number", ""),
		country=b["country"],
	))
	return _ok(result)


@kyc_bp.post("/screening/batch")
@_handle
def batch_screening() -> Response:
	"""POST /api/kyc/screening/batch — bulk re-screening for a list of customer IDs."""
	b = _body()
	result = _run(_svc().batch_screening(customer_ids=b["customer_ids"]))
	return _ok(result)


@kyc_bp.post("/screening/ongoing-monitoring")
@_handle
def ongoing_monitoring() -> Response:
	"""POST /api/kyc/screening/ongoing-monitoring — trigger re-screening."""
	b = _body()
	result = _run(_svc().ongoing_monitoring_trigger(
		customer_id=b["customer_id"],
		trigger_reason=b["trigger_reason"],
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Risk & Due Diligence
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.post("/risk/update-profile")
@_handle
def update_risk_profile() -> Response:
	"""POST /api/kyc/risk/update-profile — re-compute after trigger event."""
	b = _body()
	result = _run(_svc().update_risk_profile(
		customer_id=b["customer_id"],
		trigger=b["trigger"],
		new_data=b.get("new_data", {}),
	))
	return _ok(result)


@kyc_bp.get("/risk/due-diligence/<customer_id>")
@_handle
def risk_based_due_diligence(customer_id: str) -> Response:
	"""GET /api/kyc/risk/due-diligence/<customer_id> — FATF risk-based approach."""
	result = _run(_svc().risk_based_due_diligence(customer_id))
	return _ok(result)


@kyc_bp.post("/risk/edd")
@_handle
def enhanced_due_diligence() -> Response:
	"""POST /api/kyc/risk/edd — initiate Enhanced Due Diligence."""
	b = _body()
	result = _run(_svc().enhanced_due_diligence(customer_id=b["customer_id"]))
	return _ok(result)


@kyc_bp.get("/risk/ubo/<entity_id>")
@_handle
def beneficial_ownership(entity_id: str) -> Response:
	"""GET /api/kyc/risk/ubo/<entity_id> — verify beneficial ownership."""
	threshold = float(request.args.get("threshold", 25.0))
	result = _run(_svc().beneficial_ownership_verification(entity_id, ubo_threshold=threshold))
	return _ok(result)


@kyc_bp.post("/risk/source-of-funds")
@_handle
def source_of_funds() -> Response:
	"""POST /api/kyc/risk/source-of-funds — verify declared source of funds."""
	b = _body()
	result = _run(_svc().source_of_funds_verification(
		customer_id=b["customer_id"],
		declared_source=b["declared_source"],
		supporting_docs=b.get("supporting_docs", []),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Onboarding
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.post("/onboarding/start")
@_handle
def start_onboarding() -> Response:
	"""POST /api/kyc/onboarding/start — begin digital onboarding session."""
	b = _body()
	result = _run(_svc().start_digital_onboarding(
		mobile_number=b["mobile_number"],
		channel=b["channel"],
		customer_type=b.get("customer_type", "individual"),
		preferred_language=b.get("preferred_language", "en"),
		metadata=b.get("metadata"),
	))
	return _ok(result, 201)


@kyc_bp.get("/onboarding/<session_id>")
@_handle
def onboarding_status(session_id: str) -> Response:
	"""GET /api/kyc/onboarding/<session_id> — retrieve session status."""
	result = _run(_svc().onboarding_status(session_id))
	return _ok(result)


@kyc_bp.post("/onboarding/<session_id>/step")
@_handle
def onboarding_step(session_id: str) -> Response:
	"""POST /api/kyc/onboarding/<session_id>/step — mark a step complete."""
	b = _body()
	result = _run(_svc().onboarding_step_complete(
		session_id=session_id,
		step=b["step"],
		data=b.get("data", {}),
	))
	return _ok(result)


@kyc_bp.post("/onboarding/<session_id>/abandon")
@_handle
def abandon_onboarding(session_id: str) -> Response:
	"""POST /api/kyc/onboarding/<session_id>/abandon."""
	b = _body()
	result = _run(_svc().abandon_onboarding(
		session_id=session_id,
		reason=b["reason"],
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Compliance & Reporting
# ─────────────────────────────────────────────────────────────────────────────

@kyc_bp.get("/reports/dashboard")
@_handle
def dashboard_report() -> Response:
	"""GET /api/kyc/reports/dashboard — KPI summary."""
	# Build stats inline from service primitives
	svc = _svc()
	apps = _run(svc.list_applications())
	total = len(apps)
	by_status: dict[str, int] = {}
	risk_scores: list[int] = []
	for a in apps:
		s = a.get("status", "unknown")
		by_status[s] = by_status.get(s, 0) + 1
		if isinstance(a.get("risk_score"), int):
			risk_scores.append(a["risk_score"])
	avg_risk = round(sum(risk_scores) / max(len(risk_scores), 1), 2)
	return _ok({
		"tenant_id": svc.tenant_id,
		"total_applications": total,
		"by_status": by_status,
		"avg_risk_score": avg_risk,
		"generated_at": __import__("datetime").datetime.utcnow().isoformat(),
	})


@kyc_bp.get("/reports/expiry")
@_handle
def expiry_report() -> Response:
	"""GET /api/kyc/reports/expiry?days_ahead=90."""
	days_ahead = int(request.args.get("days_ahead", 90))
	result = _run(_svc().kyc_expiry_report(days_ahead=days_ahead))
	return _ok(result)


@kyc_bp.get("/reports/risk")
@_handle
def risk_report() -> Response:
	"""GET /api/kyc/reports/risk — risk band distribution."""
	svc = _svc()
	apps = _run(svc.list_applications())
	by_band: dict[str, int] = {}
	by_country: dict[str, int] = {}
	by_type: dict[str, int] = {}
	edd_pending = 0
	for a in apps:
		band = a.get("risk_band", "unknown")
		country = a.get("country_code", "UNKNOWN")
		ctype = a.get("customer_type", "unknown")
		by_band[band] = by_band.get(band, 0) + 1
		by_country[country] = by_country.get(country, 0) + 1
		by_type[ctype] = by_type.get(ctype, 0) + 1
		if a.get("status") == "pending_edd":
			edd_pending += 1
	return _ok({
		"tenant_id": svc.tenant_id,
		"by_risk_band": by_band,
		"by_country": by_country,
		"by_customer_type": by_type,
		"edd_pending": edd_pending,
	})


@kyc_bp.post("/reports/kyc-refresh")
@_handle
def kyc_refresh() -> Response:
	"""POST /api/kyc/reports/kyc-refresh — initiate periodic re-KYC."""
	b = _body()
	result = _run(_svc().kyc_refresh(
		customer_id=b["customer_id"],
		reason=b["reason"],
	))
	return _ok(result)


@kyc_bp.get("/reports/certificate/<customer_id>")
@_handle
def kyc_certificate(customer_id: str) -> Response:
	"""GET /api/kyc/reports/certificate/<customer_id> — generate KYC certificate."""
	result = _run(_svc().generate_kyc_certificate(customer_id))
	return _ok(result)


@kyc_bp.get("/reports/audit")
@_handle
def audit_report() -> Response:
	"""GET /api/kyc/reports/audit?from=YYYY-MM-DD&to=YYYY-MM-DD."""
	period_from = request.args.get("from", "2020-01-01")
	period_to = request.args.get("to", __import__("datetime").date.today().isoformat())
	result = _run(_svc().kyc_audit_report(period_from, period_to))
	return _ok(result)


@kyc_bp.get("/reports/failed")
@_handle
def failed_verification_report() -> Response:
	"""GET /api/kyc/reports/failed?period=2026-05."""
	period = request.args.get("period", "")
	if not period:
		import datetime
		period = datetime.date.today().strftime("%Y-%m")
	result = _run(_svc().failed_verification_report(period))
	return _ok(result)


@kyc_bp.get("/reports/regulator")
@_handle
def regulator_stats() -> Response:
	"""GET /api/kyc/reports/regulator?jurisdiction=KE&period=2026-Q1."""
	jurisdiction = request.args.get("jurisdiction", "KE")
	period = request.args.get("period", "2026")
	result = _run(_svc().regulator_stats_report(period, jurisdiction))
	return _ok(result)


@kyc_bp.get("/reports/onboarding-analytics")
@_handle
def onboarding_analytics() -> Response:
	"""GET /api/kyc/reports/onboarding-analytics?period=2026-05."""
	import datetime
	period = request.args.get("period", datetime.date.today().strftime("%Y-%m"))
	result = _run(_svc().onboarding_analytics(period))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Process-local helpers (backward-compatible with existing api.py callers)
# ─────────────────────────────────────────────────────────────────────────────

try:
	from .capability_contract import get_capability_contract
	from .service import KnowYourCustomerService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore[no-redef]
	from service import KnowYourCustomerService  # type: ignore[no-redef]

_SERVICE_SINGLETON: KnowYourCustomerService | None = None


def service() -> KnowYourCustomerService:
	"""Return the process-local KnowYourCustomerService singleton (sync API)."""
	global _SERVICE_SINGLETON
	if _SERVICE_SINGLETON is None:
		_SERVICE_SINGLETON = KnowYourCustomerService()
	return _SERVICE_SINGLETON


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service().dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"profile_count": summary["profile_count"],
		"verified_count": summary["verified_count"],
	}


# ── Sync process-local wrappers (mirror the old api.py surface) ───────────────

def open_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return service().open_profile(
		str(payload["profile_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["subject_reference"]),
		str(payload["legal_name"]),
		str(payload.get("customer_type") or "individual"),
		str(payload.get("country_code") or ""),
		str(payload.get("consent_reference") or ""),
		dict(payload.get("metadata") or {}),
		bool(payload.get("policy_attached", True)),
	)


def register_document(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_document(
		str(payload["document_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["profile_id"]),
		str(payload["document_type"]),
		str(payload["token_reference"]),
		str(payload.get("extracted_subject") or ""),
		payload.get("confidence", 0),
	)


def record_screening(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_screening(
		str(payload["screening_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["profile_id"]),
		bool(payload.get("sanctions_hit", False)),
		bool(payload.get("pep_hit", False)),
		bool(payload.get("watchlist_hit", False)),
		bool(payload.get("adverse_media_hit", False)),
		str(payload.get("review_id") or ""),
	)


def score_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return service().score_risk(
		str(payload["decision_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["profile_id"]),
		payload.get("risk_score", 0),
		str(payload.get("review_id") or ""),
	)


def record_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_decision(
		str(payload["decision_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload["profile_id"]),
		str(payload.get("decision") or "approve"),
		payload.get("risk_score", 0),
		str(payload.get("review_id") or ""),
	)


def register_kyc_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_kyc_agent(
		str(payload["agent_id"]),
		str(payload.get("tenant_id") or "default"),
		str(payload.get("name") or payload["agent_id"]),
		str(payload.get("runtime") or "codex"),
		str(payload.get("role") or "kyc_ops_reviewer"),
		str(payload.get("scope") or "review onboarding"),
	)


def list_profiles(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service().list_profiles(tenant_id)
