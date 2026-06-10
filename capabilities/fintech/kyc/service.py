"""KYC Service — complete Africa-ready Know Your Customer lifecycle.

Covers all 48 methods across application management, document verification,
biometric checks, screening, risk scoring, compliance reporting, and digital
onboarding workflows.

Usage (standalone / tests)::

	svc = KYCService(tenant_id="acme")
	await svc.start_kyc_application("cust_001", "individual", "KE")

Usage (platform)::

	svc = KYCService(
		tenant_id="acme",
		actor_id="ops@acme.co",
		auth=auth_service,
		audit=audit_service,
		notify=notify_service,
		db_url="postgresql+asyncpg://...",
	)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .domain.adapters import (
		AuditAdapter,
		AuthAdapter,
		NotifyAdapter,
		get_audit_adapter,
		get_auth_adapter,
		get_notify_adapter,
	)
	from .database.store import Store, get_store
	from .domain.calculations import (
		calculate_expiry_date,
		calculate_risk_band,
		calculate_risk_score,
		days_until_expiry,
		is_high_risk_country,
		is_high_risk_industry,
		name_match_score,
	)
	from .domain.events import DomainEvent
	from .domain.rules import (
		RuleViolation,
		assert_address_document_present,
		assert_biometric_match,
		assert_consent_recorded,
		assert_edd_for_high_risk,
		assert_edd_for_pep,
		assert_identity_document_present,
		assert_kyc_not_expired,
		assert_liveness_check_passed,
		assert_no_cross_tenant_access,
		assert_no_deceased_id,
		assert_no_open_reviews,
		assert_no_synthetic_identity,
		assert_no_unresolved_sanction,
		assert_risk_score_range,
		assert_screening_completed,
		assert_tenant_context,
		assert_ubo_declared,
	)
	from .models import (
		AdverseMediaCheck,
		AdverseMediaCheckCreate,
		ApplicationStatus,
		BiometricData,
		BiometricDataCreate,
		BiometricStatus,
		BiometricType,
		BusinessKYC,
		BusinessKYCCreate,
		CustomerType,
		DocumentStatus,
		DocumentType,
		IDDocument,
		IDDocumentCreate,
		IDDocumentUpdate,
		JourneyStatus,
		KYCApplication,
		KYCApplicationCreate,
		KYCApplicationUpdate,
		KYCReview,
		KYCReviewCreate,
		KYCReviewUpdate,
		OnboardingJourney,
		OnboardingJourneyCreate,
		PEPCheck,
		PEPCheckCreate,
		RiskBand,
		RiskProfile,
		RiskProfileCreate,
		ReviewStatus,
		ReviewType,
		SanctionCheck,
		SanctionCheckCreate,
		UBODeclaration,
		UBODeclarationCreate,
		uuid7str,
	)
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CUSTOMER_TYPES,
		SUPPORTED_DOCUMENT_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .kyc_runtime import normalize_code, normalize_confidence, normalize_country, normalize_risk_score, risk_band as _risk_band_fn
except ImportError:  # pragma: no cover — supports direct file loading in tests
	from domain.adapters import (  # type: ignore[no-redef]
		AuditAdapter,
		AuthAdapter,
		NotifyAdapter,
		get_audit_adapter,
		get_auth_adapter,
		get_notify_adapter,
	)
	from database.store import Store, get_store  # type: ignore[no-redef]
	from domain.calculations import (  # type: ignore[no-redef]
		calculate_expiry_date,
		calculate_risk_band,
		calculate_risk_score,
		days_until_expiry,
		is_high_risk_country,
		is_high_risk_industry,
		name_match_score,
	)
	from domain.events import DomainEvent  # type: ignore[no-redef]
	from domain.rules import (  # type: ignore[no-redef]
		RuleViolation,
		assert_address_document_present,
		assert_biometric_match,
		assert_consent_recorded,
		assert_edd_for_high_risk,
		assert_edd_for_pep,
		assert_identity_document_present,
		assert_kyc_not_expired,
		assert_liveness_check_passed,
		assert_no_cross_tenant_access,
		assert_no_deceased_id,
		assert_no_open_reviews,
		assert_no_synthetic_identity,
		assert_no_unresolved_sanction,
		assert_risk_score_range,
		assert_screening_completed,
		assert_tenant_context,
		assert_ubo_declared,
	)
	from models import (  # type: ignore[no-redef]
		AdverseMediaCheck,
		AdverseMediaCheckCreate,
		ApplicationStatus,
		BiometricData,
		BiometricDataCreate,
		BiometricStatus,
		BiometricType,
		BusinessKYC,
		BusinessKYCCreate,
		CustomerType,
		DocumentStatus,
		DocumentType,
		IDDocument,
		IDDocumentCreate,
		IDDocumentUpdate,
		JourneyStatus,
		KYCApplication,
		KYCApplicationCreate,
		KYCApplicationUpdate,
		KYCReview,
		KYCReviewCreate,
		KYCReviewUpdate,
		OnboardingJourney,
		OnboardingJourneyCreate,
		PEPCheck,
		PEPCheckCreate,
		RiskBand,
		RiskProfile,
		RiskProfileCreate,
		ReviewStatus,
		ReviewType,
		SanctionCheck,
		SanctionCheckCreate,
		UBODeclaration,
		UBODeclarationCreate,
		uuid7str,
	)
	from capability_contract import (  # type: ignore[no-redef]
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CUSTOMER_TYPES,
		SUPPORTED_DOCUMENT_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from kyc_runtime import normalize_code, normalize_confidence, normalize_country, normalize_risk_score, risk_band as _risk_band_fn  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Collection names — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
_COL_APP = "kyc_applications"
_COL_DOC = "kyc_documents"
_COL_BIO = "kyc_biometrics"
_COL_RISK = "kyc_risk_profiles"
_COL_PEP = "kyc_pep_checks"
_COL_SANC = "kyc_sanction_checks"
_COL_AMEDIA = "kyc_adverse_media"
_COL_BKYC = "kyc_business_kyc"
_COL_UBO = "kyc_ubo_declarations"
_COL_REVIEW = "kyc_reviews"
_COL_JOURNEY = "kyc_onboarding_journeys"
_COL_AUDIT = "kyc_audit_events"

# Ordered required steps per customer type
_INDIVIDUAL_STEPS = ["identity_document", "address_document", "biometrics", "pep_screening", "sanctions_screening", "risk_assessment", "review"]
_BUSINESS_STEPS = ["identity_document", "business_registration", "ubo_declaration", "address_document", "pep_screening", "sanctions_screening", "risk_assessment", "review"]

# Africa regional ID type mapping
_REGIONAL_ID_TYPES: dict[str, list[str]] = {
	"KE": ["national_id", "huduma_namba", "passport", "driver_license"],
	"NG": ["national_id", "bvn", "passport", "driver_license", "voter_id"],
	"GH": ["ghana_card", "passport", "driver_license", "voter_id"],
	"TZ": ["national_id", "passport", "driver_license"],
	"UG": ["national_id", "passport", "driver_license", "voter_id"],
	"ZA": ["national_id", "passport", "driver_license"],
	"ET": ["national_id", "passport"],
	"RW": ["national_id", "passport"],
	"ZM": ["national_id", "passport"],
	"ZW": ["national_id", "passport"],
}

# Sanctions lists screened by default
_DEFAULT_SANCTIONS_LISTS = ["OFAC_SDN", "UN_CONSOLIDATED", "EU_ASSET_FREEZE", "AU_SANCTIONS", "HMT_UK", "INTERPOL"]

# PEP category descriptors
_PEP_CATEGORIES = {
	"head_of_state": "Head of State / Government",
	"minister": "Cabinet Minister",
	"senior_civil_servant": "Senior Civil Servant",
	"judiciary": "Senior Judiciary",
	"military": "Senior Military Officer",
	"soe_exec": "State-Owned Enterprise Executive",
	"party_official": "Senior Political Party Official",
	"local_govt": "Local Government Official",
	"intl_org": "International Organisation Official",
	"family_member": "Family Member of PEP",
	"close_associate": "Close Associate of PEP",
}


def _log_pretty_path(collection: str, resource_id: str) -> str:
	"""Format a log path for structured audit lines."""
	return f"{collection}/{resource_id}"


def _now() -> datetime:
	return datetime.utcnow()


def _today() -> date:
	return date.today()


def _serialize(obj: Any) -> Any:
	"""Recursively convert Pydantic models and dates to JSON-safe primitives."""
	if hasattr(obj, "model_dump"):
		return obj.model_dump(mode="json")
	if isinstance(obj, datetime):
		return obj.isoformat()
	if isinstance(obj, date):
		return obj.isoformat()
	return obj


class KYCService:
	"""Full KYC lifecycle service — Africa-ready, adapter/store pattern.

	All public methods are async. The service is tenant-scoped: every
	operation is isolated to `tenant_id` and audited under `actor_id`.

	All methods return plain ``dict[str, Any]`` so callers don't take a
	hard dependency on internal Pydantic models.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: AuthAdapter | None = None,
		audit: AuditAdapter | None = None,
		notify: NotifyAdapter | None = None,
		db_url: str | None = None,
		store: Store | None = None,
	) -> None:
		assert tenant_id and tenant_id.strip(), "tenant_id is required"
		self.tenant_id = tenant_id.strip()
		self.actor_id = actor_id or "system"
		self._auth = auth or get_auth_adapter()
		self._audit = audit or get_audit_adapter()
		self._notify = notify or get_notify_adapter()
		self._store = store or get_store(db_url)

	# ─────────────────────────────────────────────────────────────────────────
	# Internal helpers
	# ─────────────────────────────────────────────────────────────────────────

	async def _emit(
		self,
		event_type: str,
		resource_id: str,
		resource_type: str,
		payload: dict[str, Any] | None = None,
	) -> None:
		"""Emit a domain event and write to the audit log collection."""
		evt = DomainEvent(
			event_type=event_type,
			tenant_id=self.tenant_id,
			actor_id=self.actor_id,
			payload=payload or {},
		)
		evt_dict = evt.to_dict()
		evt_dict["resource_id"] = resource_id
		evt_dict["resource_type"] = resource_type
		evt_dict["id"] = uuid7str()
		await self._store.put(_COL_AUDIT, evt_dict)
		await self._audit.log_event(
			event_type=event_type,
			actor_id=self.actor_id,
			tenant_id=self.tenant_id,
			resource_id=resource_id,
			details=payload or {},
		)

	async def _get_app(self, application_id: str) -> dict[str, Any]:
		"""Fetch an application, asserting tenant ownership."""
		record = await self._store.get(_COL_APP, application_id)
		if not record:
			raise KeyError(f"application not found: {application_id}")
		assert_no_cross_tenant_access(self.tenant_id, record["tenant_id"])
		return record

	async def _require_app(self, application_id: str) -> dict[str, Any]:
		record = await self._get_app(application_id)
		if record.get("is_deleted"):
			raise ValueError(f"application {application_id} has been deleted")
		return record

	async def _get_document(self, document_id: str) -> dict[str, Any]:
		record = await self._store.get(_COL_DOC, document_id)
		if not record:
			raise KeyError(f"document not found: {document_id}")
		assert_no_cross_tenant_access(self.tenant_id, record["tenant_id"])
		return record

	def _screening_hit_score(self, name_a: str, name_b: str) -> float:
		"""Jaro-Winkler name similarity for screening match scoring."""
		return name_match_score(name_a, name_b)

	def _document_hash(self, file_metadata: dict[str, Any]) -> str:
		"""Deterministic content hash from file metadata for deduplication."""
		key = f"{file_metadata.get('filename', '')}{file_metadata.get('size', '')}{file_metadata.get('checksum', '')}"
		return hashlib.sha256(key.encode()).hexdigest()[:16]

	async def _open_reviews_count(self, application_id: str) -> int:
		reviews = await self._store.query(_COL_REVIEW, {"application_id": application_id, "tenant_id": self.tenant_id})
		return sum(1 for r in reviews if r.get("status") in ("open", "in_progress"))

	async def _has_doc_type(self, application_id: str, doc_types: set[str]) -> bool:
		docs = await self._store.query(_COL_DOC, {"application_id": application_id, "tenant_id": self.tenant_id})
		return any(
			d.get("document_type") in doc_types and d.get("status") == DocumentStatus.verified.value
			for d in docs
		)

	async def _screening_done(self, application_id: str) -> bool:
		pep = await self._store.query(_COL_PEP, {"application_id": application_id, "tenant_id": self.tenant_id})
		sanc = await self._store.query(_COL_SANC, {"application_id": application_id, "tenant_id": self.tenant_id})
		return bool(pep) and bool(sanc)

	async def _risk_assessed(self, application_id: str) -> bool:
		risk = await self._store.query(_COL_RISK, {"application_id": application_id, "tenant_id": self.tenant_id})
		return bool(risk)

	# ─────────────────────────────────────────────────────────────────────────
	# Application Management (8 methods)
	# ─────────────────────────────────────────────────────────────────────────

	async def start_kyc_application(
		self,
		customer_id: str,
		customer_type: str,
		jurisdiction: str,
		*,
		legal_name: str = "",
		consent_reference: str = "",
		kyc_tier: str = "standard",
		is_refugee: bool = False,
		is_informal_sector: bool = False,
		preferred_language: str = "en",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Open a new KYC application and return the application record.

		Enforces tenant context, consent, and customer-type support checks
		upfront. Returns ``ApplicationStatus.draft``.

		Africa edge cases handled:
		- Refugee customers: relaxed document requirements flagged.
		- Informal sector: minimum government-issued ID required.
		- CBK Kenya: tier assigned automatically for mobile money customers.
		"""
		assert_tenant_context(self.tenant_id)
		assert customer_id and customer_id.strip(), "customer_id is required"
		assert customer_type in [e.value for e in CustomerType], f"unsupported customer_type: {customer_type}"
		if consent_reference:
			assert_consent_recorded(consent_reference)

		# Determine required steps based on customer type
		steps_required = (
			_BUSINESS_STEPS[:]
			if customer_type in ("business", "nonprofit", "trust", "partnership")
			else _INDIVIDUAL_STEPS[:]
		)

		app = KYCApplication(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			customer_id=customer_id.strip(),
			customer_type=CustomerType(customer_type),
			country_code=jurisdiction.upper(),
			legal_name=legal_name.strip() or f"Customer {customer_id}",
			consent_reference=consent_reference or "",
			kyc_tier=kyc_tier,
			is_refugee=is_refugee,
			is_informal_sector=is_informal_sector,
			preferred_language=preferred_language,
			status=ApplicationStatus.draft,
			created_by=self.actor_id,
			metadata={
				**(metadata or {}),
				"steps_required": steps_required,
				"jurisdiction": jurisdiction.upper(),
				"regional_id_types": _REGIONAL_ID_TYPES.get(jurisdiction.upper(), []),
			},
		)
		record = app.model_dump(mode="json")
		await self._store.put(_COL_APP, record)
		await self._emit(
			"kyc_application_started",
			app.id,
			"kyc_application",
			{"customer_id": customer_id, "customer_type": customer_type, "jurisdiction": jurisdiction},
		)
		logger.info("KYC application started: %s", _log_pretty_path(_COL_APP, app.id))
		return record

	async def update_application(
		self,
		application_id: str,
		**fields: Any,
	) -> dict[str, Any]:
		"""Patch mutable fields on an application.

		Accepted fields mirror ``KYCApplicationUpdate``:
		``status``, ``kyc_tier``, ``metadata``, ``legal_name``,
		``consent_reference``, ``preferred_language``.
		"""
		record = await self._require_app(application_id)
		allowed = {"status", "kyc_tier", "metadata", "legal_name", "consent_reference", "preferred_language"}
		unknown = set(fields) - allowed
		if unknown:
			raise ValueError(f"non-updatable fields: {', '.join(sorted(unknown))}")

		if "status" in fields:
			# Validate enum value
			ApplicationStatus(fields["status"])

		record.update({k: v for k, v in fields.items() if v is not None})
		record["updated_at"] = _now().isoformat()
		await self._store.put(_COL_APP, record)
		await self._emit("kyc_application_updated", application_id, "kyc_application", fields)
		return record

	async def get_application(self, application_id: str) -> dict[str, Any]:
		"""Fetch a single application by ID."""
		return await self._require_app(application_id)

	async def list_applications(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
		"""List applications for this tenant, optionally filtered.

		Supported filter keys: ``customer_id``, ``status``, ``customer_type``,
		``country_code``, ``risk_band``, ``kyc_tier``.
		"""
		base_filters: dict[str, Any] = {"tenant_id": self.tenant_id}
		if filters:
			base_filters.update({k: v for k, v in filters.items() if v is not None})
		records = await self._store.query(_COL_APP, base_filters, limit=500)
		return [r for r in records if not r.get("is_deleted")]

	async def assign_reviewer(
		self,
		application_id: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		"""Assign a human reviewer to an application and open a KYC review."""
		assert reviewer_id and reviewer_id.strip(), "reviewer_id is required"
		app = await self._require_app(application_id)

		review = KYCReview(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			review_type=ReviewType.standard_kyc,
			assigned_to=reviewer_id.strip(),
			status=ReviewStatus.open,
			created_by=self.actor_id,
			notes=f"Assigned by {self.actor_id}",
		)
		review_record = review.model_dump(mode="json")
		await self._store.put(_COL_REVIEW, review_record)

		# Advance application status if still draft
		if app.get("status") == ApplicationStatus.draft.value:
			await self.update_application(application_id, status=ApplicationStatus.pending_review.value)

		await self._emit(
			"kyc_reviewer_assigned",
			application_id,
			"kyc_application",
			{"reviewer_id": reviewer_id, "review_id": review.id},
		)
		await self._notify.send(
			recipient=reviewer_id,
			channel="email",
			subject="KYC Application Assigned",
			body=f"KYC application {application_id} has been assigned to you for review.",
			metadata={"application_id": application_id, "review_id": review.id},
		)
		return {"review_id": review.id, "application_id": application_id, "assigned_to": reviewer_id, "status": "open"}

	async def approve_application(
		self,
		application_id: str,
		reviewer_id: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Approve a KYC application after all checks pass.

		Enforces:
		- Identity + address documents present and verified.
		- Screening completed.
		- Risk assessment present.
		- EDD completed for high-risk / PEP customers.
		- No open reviews.
		- No confirmed sanctions.
		- KYC not already expired.
		"""
		assert reviewer_id and reviewer_id.strip(), "reviewer_id is required"
		app = await self._require_app(application_id)

		# Pre-approval rule enforcement
		identity_types = {"passport", "national_id", "driver_license", "resident_permit", "refugee_document", "huduma_namba", "bvn", "ghana_card", "voter_id"}
		address_types = {"utility_bill", "bank_statement", "business_registration", "lease_agreement"}
		has_id = await self._has_doc_type(application_id, identity_types)
		has_addr = await self._has_doc_type(application_id, address_types)
		assert_identity_document_present(has_id)
		assert_address_document_present(has_addr)
		assert_screening_completed(await self._screening_done(application_id))

		risk_records = await self._store.query(_COL_RISK, {"application_id": application_id, "tenant_id": self.tenant_id})
		assert_risk_score_range(int(app.get("risk_score", 0)))
		if risk_records:
			rp = risk_records[-1]
			risk_score = int(rp.get("risk_score", 0))
			# EDD gate for high-risk and PEP
			edd_reviews = await self._store.query(
				_COL_REVIEW,
				{"application_id": application_id, "tenant_id": self.tenant_id, "review_type": ReviewType.enhanced_due_diligence.value},
			)
			edd_done = any(r.get("status") == ReviewStatus.approved.value for r in edd_reviews)
			assert_edd_for_high_risk(risk_score, edd_done)
			assert_edd_for_pep(rp.get("is_pep", False), edd_done)

		# No confirmed sanctions
		sanctions = await self._store.query(_COL_SANC, {"application_id": application_id, "tenant_id": self.tenant_id})
		for s in sanctions:
			assert_no_unresolved_sanction(s.get("status") == "confirmed_hit")

		# No open reviews
		open_count = await self._open_reviews_count(application_id)
		assert_no_open_reviews(open_count)

		# Calculate expiry from risk band
		risk_band_val = app.get("risk_band", "low")
		expiry = calculate_expiry_date(risk_band_val)

		now = _now()
		updates = {
			"status": ApplicationStatus.approved.value,
			"last_verified_at": now.isoformat(),
			"expiry_date": expiry.isoformat() if expiry else None,
			"updated_at": now.isoformat(),
			"metadata": {
				**app.get("metadata", {}),
				"approved_by": reviewer_id,
				"approved_at": now.isoformat(),
				"approval_notes": notes,
			},
		}
		app.update(updates)
		await self._store.put(_COL_APP, app)

		# Close any open reviews
		open_reviews = await self._store.query(
			_COL_REVIEW,
			{"application_id": application_id, "tenant_id": self.tenant_id},
		)
		for rev in open_reviews:
			if rev.get("status") in ("open", "in_progress"):
				rev["status"] = ReviewStatus.approved.value
				rev["decision"] = "approved"
				rev["completed_at"] = now.isoformat()
				rev["notes"] = notes or rev.get("notes", "")
				await self._store.put(_COL_REVIEW, rev)

		await self._emit(
			"kyc_application_approved",
			application_id,
			"kyc_application",
			{"reviewer_id": reviewer_id, "expiry_date": expiry.isoformat() if expiry else None},
		)
		await self._notify.send(
			recipient=app.get("customer_id", application_id),
			channel="sms",
			subject="KYC Approved",
			body="Your KYC verification has been approved.",
			metadata={"application_id": application_id},
		)
		return app

	async def reject_application(
		self,
		application_id: str,
		reason: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		"""Reject a KYC application with a mandatory reason."""
		assert reason and reason.strip(), "rejection reason is required"
		assert reviewer_id and reviewer_id.strip(), "reviewer_id is required"
		app = await self._require_app(application_id)

		now = _now()
		app["status"] = ApplicationStatus.rejected.value
		app["updated_at"] = now.isoformat()
		app["metadata"] = {
			**app.get("metadata", {}),
			"rejected_by": reviewer_id,
			"rejected_at": now.isoformat(),
			"rejection_reason": reason.strip(),
		}
		await self._store.put(_COL_APP, app)

		await self._emit(
			"kyc_application_rejected",
			application_id,
			"kyc_application",
			{"reviewer_id": reviewer_id, "reason": reason},
		)
		await self._notify.send(
			recipient=app.get("customer_id", application_id),
			channel="sms",
			subject="KYC Application Update",
			body=f"Your KYC application could not be approved: {reason}",
			metadata={"application_id": application_id},
		)
		return app

	async def request_additional_docs(
		self,
		application_id: str,
		required_docs: list[str],
		message: str = "",
	) -> dict[str, Any]:
		"""Request additional documents from the customer and pause the application.

		Sets status to ``in_progress`` and stores the outstanding doc list in metadata.
		"""
		assert required_docs, "required_docs must not be empty"
		app = await self._require_app(application_id)

		now = _now()
		app["status"] = ApplicationStatus.in_progress.value
		app["updated_at"] = now.isoformat()
		app["metadata"] = {
			**app.get("metadata", {}),
			"additional_docs_requested": required_docs,
			"additional_docs_message": message or "Please provide the requested documents.",
			"additional_docs_requested_at": now.isoformat(),
			"additional_docs_requested_by": self.actor_id,
		}
		await self._store.put(_COL_APP, app)

		await self._emit(
			"kyc_additional_docs_requested",
			application_id,
			"kyc_application",
			{"required_docs": required_docs, "message": message},
		)
		await self._notify.send(
			recipient=app.get("customer_id", application_id),
			channel="email",
			subject="Additional Documents Required",
			body=message or f"Please provide: {', '.join(required_docs)}",
			metadata={"application_id": application_id, "required_docs": required_docs},
		)
		return {
			"application_id": application_id,
			"status": app["status"],
			"required_docs": required_docs,
			"message": message,
		}

	# ─────────────────────────────────────────────────────────────────────────
	# Document Verification (10 methods)
	# ─────────────────────────────────────────────────────────────────────────

	async def verify_national_id(
		self,
		id_number: str,
		country: str,
		name: str,
		dob: str | date | None = None,
	) -> dict[str, Any]:
		"""Verify a national ID against the relevant government registry.

		Registry routing:
		- KE: IPRS / Huduma Namba
		- UG: NIRA (National Identification & Registration Authority)
		- TZ: NIDA (National Identification Authority)
		- GH: NIA (National Identification Authority)
		- NG: NIN / BVN (NIMC / CBN)

		Returns: verified, name_match_score, photo_url, id_valid, expiry.
		"""
		assert id_number and id_number.strip(), "id_number is required"
		assert country and len(country) == 2, "country must be ISO-3166-1 alpha-2"
		assert name and name.strip(), "name is required"
		country = country.upper()

		# Registry routing — stub implementation returns deterministic result
		# In production: replace with HTTP client calls to each registry API.
		registry_map = {
			"KE": {"registry": "IPRS/Huduma Namba", "endpoint": "iprs.go.ke"},
			"UG": {"registry": "NIRA", "endpoint": "nira.go.ug"},
			"TZ": {"registry": "NIDA", "endpoint": "nida.go.tz"},
			"GH": {"registry": "NIA", "endpoint": "nia.gov.gh"},
			"NG": {"registry": "NIMC/BVN", "endpoint": "nimc.gov.ng"},
			"ZA": {"registry": "DHA", "endpoint": "dha.gov.za"},
			"ET": {"registry": "DPPPA", "endpoint": "dpppa.gov.et"},
			"RW": {"registry": "NIDA Rwanda", "endpoint": "nida.gov.rw"},
		}
		registry_info = registry_map.get(country, {"registry": "UNKNOWN", "endpoint": ""})

		# Deterministic confidence based on ID number format validity
		id_clean = id_number.strip().replace("-", "").replace(" ", "")
		format_valid = len(id_clean) >= 6 and id_clean[:2].isdigit() if country == "KE" else len(id_clean) >= 8

		match_score = name_match_score(name, name)  # self-match = 1.0 baseline; real impl compares against registry name
		id_valid = format_valid
		confidence = 0.92 if id_valid else 0.30

		result: dict[str, Any] = {
			"id_number": id_number.strip(),
			"country": country,
			"registry": registry_info["registry"],
			"verified": id_valid and confidence >= 0.75,
			"name_match_score": round(match_score, 4),
			"id_valid": id_valid,
			"confidence": confidence,
			"photo_url": None,  # populated by biometric adapter in production
			"expiry": None,
			"high_risk_country": is_high_risk_country(country),
			"checked_at": _now().isoformat(),
		}
		if isinstance(dob, str):
			result["dob_verified"] = bool(dob)
		elif isinstance(dob, date):
			result["dob_verified"] = True

		await self._emit("kyc_national_id_verified", id_number, "national_id", {"country": country, "verified": result["verified"]})
		return result

	async def verify_passport(
		self,
		passport_number: str,
		country: str,
		mrz_data: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Verify a passport via ICAO document validation and MRZ checksum.

		MRZ data fields: ``line1``, ``line2`` (raw MRZ strings from OCR).
		Returns: verified, mrz_valid, expiry, issuing_country, name_from_mrz.
		"""
		assert passport_number and passport_number.strip(), "passport_number is required"
		assert country and len(country) >= 2, "country is required"

		mrz = mrz_data or {}
		mrz_line1 = mrz.get("line1", "")
		mrz_line2 = mrz.get("line2", "")

		# MRZ checksum validation (simplified — real impl uses ICAO 9303 algorithm)
		mrz_valid = bool(mrz_line1 and mrz_line2 and len(mrz_line1) in (44, 36, 30))
		name_from_mrz = ""
		expiry_from_mrz = None

		if mrz_valid and len(mrz_line1) == 44:
			# TD-3 passport: positions 5-44 contain surname<<given_names
			name_field = mrz_line1[5:44].replace("<", " ").strip()
			name_from_mrz = name_field.split("  ")[0].title()
			if len(mrz_line2) == 44:
				expiry_str = mrz_line2[13:19]
				try:
					yr = int(expiry_str[:2])
					yr_full = 2000 + yr if yr < 50 else 1900 + yr
					expiry_from_mrz = date(yr_full, int(expiry_str[2:4]), int(expiry_str[4:6]))
				except (ValueError, IndexError) as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		is_expired = expiry_from_mrz is not None and expiry_from_mrz < _today()
		result = {
			"passport_number": passport_number.strip(),
			"country": country.upper(),
			"verified": mrz_valid and not is_expired,
			"mrz_valid": mrz_valid,
			"name_from_mrz": name_from_mrz,
			"expiry": expiry_from_mrz.isoformat() if expiry_from_mrz else None,
			"is_expired": is_expired,
			"high_risk_country": is_high_risk_country(country),
			"checked_at": _now().isoformat(),
		}
		await self._emit("kyc_passport_verified", passport_number, "passport", {"country": country, "verified": result["verified"]})
		return result

	async def verify_drivers_license(
		self,
		license_number: str,
		country: str,
		*,
		name: str = "",
	) -> dict[str, Any]:
		"""Verify a driving licence against national transport authority.

		Country routing: NTSA (KE), DVLA (GH/ZA), FRSC (NG), URSB (UG).
		"""
		assert license_number and license_number.strip(), "license_number is required"
		assert country and len(country) == 2, "country must be ISO-3166-1 alpha-2"

		authority_map = {
			"KE": "NTSA", "GH": "DVLA", "ZA": "RTMC",
			"NG": "FRSC", "UG": "UNRA", "TZ": "SUMATRA",
		}
		authority = authority_map.get(country.upper(), "TRANSPORT_AUTHORITY")

		# Format validation heuristic
		lic_clean = license_number.strip().upper().replace("-", "").replace(" ", "")
		format_valid = 6 <= len(lic_clean) <= 20 and any(c.isalpha() for c in lic_clean)
		confidence = 0.88 if format_valid else 0.25

		result = {
			"license_number": license_number.strip(),
			"country": country.upper(),
			"authority": authority,
			"verified": format_valid and confidence >= 0.75,
			"format_valid": format_valid,
			"confidence": confidence,
			"name_on_license": name,
			"expiry": None,
			"checked_at": _now().isoformat(),
		}
		await self._emit("kyc_drivers_license_verified", license_number, "drivers_license", {"country": country})
		return result

	async def verify_birth_certificate(
		self,
		cert_number: str,
		country: str,
		*,
		full_name: str = "",
		date_of_birth: str | date | None = None,
	) -> dict[str, Any]:
		"""Verify a birth certificate against civil registration authority.

		Used for juvenile accounts, informal sector customers lacking national ID,
		and refugees. Returns: verified, registrar, cert_valid.
		"""
		assert cert_number and cert_number.strip(), "cert_number is required"
		assert country and len(country) == 2, "country is required"

		registrar_map = {
			"KE": "Kenya Civil Registration Service",
			"NG": "National Population Commission",
			"GH": "Births and Deaths Registry",
			"TZ": "RITA (Registration, Insolvency and Trusteeship Agency)",
			"UG": "Uganda Registration Services Bureau",
			"ZA": "Department of Home Affairs",
		}
		registrar = registrar_map.get(country.upper(), "Civil Registration Authority")
		cert_clean = cert_number.strip().replace("/", "").replace("-", "")
		cert_valid = len(cert_clean) >= 6

		return {
			"cert_number": cert_number.strip(),
			"country": country.upper(),
			"registrar": registrar,
			"verified": cert_valid,
			"cert_valid": cert_valid,
			"full_name": full_name,
			"date_of_birth": date_of_birth.isoformat() if isinstance(date_of_birth, date) else str(date_of_birth or ""),
			"confidence": 0.80 if cert_valid else 0.20,
			"checked_at": _now().isoformat(),
		}

	async def upload_document(
		self,
		application_id: str,
		doc_type: str,
		file_metadata: dict[str, Any],
		uploaded_by: str,
	) -> dict[str, Any]:
		"""Register a document upload against a KYC application.

		``file_metadata`` must contain at minimum: ``filename``, ``size``,
		``mime_type``, and a ``token_reference`` (vault storage reference — never
		raw document bytes here).
		"""
		assert doc_type in [e.value for e in DocumentType], f"unsupported doc_type: {doc_type}"
		assert file_metadata.get("token_reference"), "file_metadata.token_reference is required"
		assert uploaded_by and uploaded_by.strip(), "uploaded_by is required"

		app = await self._require_app(application_id)

		content_hash = self._document_hash(file_metadata)
		doc = IDDocument(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			document_type=DocumentType(doc_type),
			token_reference=file_metadata["token_reference"],
			document_number=file_metadata.get("document_number", ""),
			issuing_country=file_metadata.get("issuing_country", app.get("country_code", "")),
			status=DocumentStatus.pending,
			confidence=0.0,
			created_by=uploaded_by.strip(),
			metadata={
				"filename": file_metadata.get("filename", ""),
				"size": file_metadata.get("size", 0),
				"mime_type": file_metadata.get("mime_type", ""),
				"content_hash": content_hash,
				"uploaded_by": uploaded_by,
				"upload_source": file_metadata.get("source", "api"),
			},
		)
		record = doc.model_dump(mode="json")
		await self._store.put(_COL_DOC, record)
		await self._emit(
			"kyc_document_uploaded",
			doc.id,
			"kyc_document",
			{"application_id": application_id, "doc_type": doc_type, "content_hash": content_hash},
		)
		logger.info("Document uploaded: %s", _log_pretty_path(_COL_DOC, doc.id))
		return record

	async def verify_document_authenticity(self, document_id: str) -> dict[str, Any]:
		"""Check document authenticity: watermarks, fonts, security features, holograms.

		In production this delegates to the ``cvsn`` (computer-vision) adapter.
		Returns: authentic, security_features_present, tampering_detected,
		synthetic_fraud_score, confidence.
		"""
		doc = await self._get_document(document_id)

		# Authenticity analysis stub — real impl calls vision adapter
		synthetic_fraud_score = doc.get("synthetic_fraud_score", 0.0)
		assert_no_synthetic_identity(synthetic_fraud_score)

		confidence = 0.91 if synthetic_fraud_score < 0.3 else 0.45
		authentic = confidence >= 0.75

		update = {
			"status": DocumentStatus.verified.value if authentic else DocumentStatus.synthetic_fraud_flagged.value,
			"confidence": confidence,
			"synthetic_fraud_score": synthetic_fraud_score,
			"updated_at": _now().isoformat(),
			"metadata": {
				**doc.get("metadata", {}),
				"authenticity_checked_at": _now().isoformat(),
				"authenticity_result": "authentic" if authentic else "suspicious",
			},
		}
		doc.update(update)
		await self._store.put(_COL_DOC, doc)

		await self._emit(
			"kyc_document_authenticity_checked",
			document_id,
			"kyc_document",
			{"authentic": authentic, "confidence": confidence},
		)
		return {
			"document_id": document_id,
			"authentic": authentic,
			"security_features_present": authentic,
			"tampering_detected": not authentic,
			"synthetic_fraud_score": synthetic_fraud_score,
			"confidence": confidence,
			"checked_at": _now().isoformat(),
		}

	async def extract_document_data(self, document_id: str) -> dict[str, Any]:
		"""Run OCR extraction on a stored document.

		In production delegates to the ``nlpc`` (NLP/OCR) adapter.
		Returns: extracted fields (name, dob, document_number, expiry, nationality, address).
		"""
		doc = await self._get_document(document_id)

		# OCR extraction stub — real impl calls OCR/NLP adapter
		extracted = {
			"document_id": document_id,
			"document_type": doc.get("document_type"),
			"extracted_name": doc.get("extracted_name", ""),
			"extracted_dob": doc.get("extracted_dob"),
			"document_number": doc.get("document_number", ""),
			"issuing_country": doc.get("issuing_country", ""),
			"expiry_date": doc.get("expiry_date"),
			"nationality": doc.get("extracted_nationality", ""),
			"address": doc.get("ocr_raw", {}).get("address", ""),
			"name_script": doc.get("name_script", "latin"),
			"name_transliterated": doc.get("name_transliterated", ""),
			"ocr_confidence": doc.get("confidence", 0.0),
			"extracted_at": _now().isoformat(),
		}

		# Update the document with extracted data
		doc["ocr_raw"] = {**doc.get("ocr_raw", {}), **{k: v for k, v in extracted.items() if v}}
		doc["updated_at"] = _now().isoformat()
		await self._store.put(_COL_DOC, doc)

		await self._emit("kyc_document_ocr_extracted", document_id, "kyc_document", {"fields_extracted": list(extracted.keys())})
		return extracted

	async def check_document_expiry(self, document_id: str) -> dict[str, Any]:
		"""Check whether a stored document has expired.

		Enforces ``assert_document_not_expired`` and updates status.
		"""
		doc = await self._get_document(document_id)
		expiry_str = doc.get("expiry_date")
		expiry_date: date | None = None
		if expiry_str:
			try:
				expiry_date = date.fromisoformat(str(expiry_str)[:10])
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		is_expired = expiry_date is not None and expiry_date < _today()
		days_remaining = days_until_expiry(expiry_date)

		if is_expired and doc.get("status") != DocumentStatus.expired.value:
			doc["status"] = DocumentStatus.expired.value
			doc["updated_at"] = _now().isoformat()
			await self._store.put(_COL_DOC, doc)
			await self._emit("kyc_document_expired", document_id, "kyc_document", {"expiry_date": str(expiry_date)})

		return {
			"document_id": document_id,
			"document_type": doc.get("document_type"),
			"expiry_date": expiry_str,
			"is_expired": is_expired,
			"days_remaining": days_remaining,
			"status": doc.get("status"),
		}

	async def document_match_check(
		self,
		doc1_id: str,
		doc2_id: str,
	) -> dict[str, Any]:
		"""Cross-check name consistency between two documents.

		Uses Jaro-Winkler to score name similarity. A score below 0.85 triggers
		a mismatch flag requiring human review.
		"""
		doc1 = await self._get_document(doc1_id)
		doc2 = await self._get_document(doc2_id)

		name1 = doc1.get("extracted_name", "")
		name2 = doc2.get("extracted_name", "")
		# Same document — trivially identical
		if doc1_id == doc2_id:
			score = 1.0
		elif name1 and name2:
			score = name_match_score(name1, name2)
		else:
			score = 0.0
		consistent = score >= 0.85

		result = {
			"doc1_id": doc1_id,
			"doc2_id": doc2_id,
			"doc1_name": name1,
			"doc2_name": name2,
			"match_score": round(score, 4),
			"names_consistent": consistent,
			"mismatch_flag": not consistent,
			"checked_at": _now().isoformat(),
		}
		if not consistent:
			await self._emit(
				"kyc_document_name_mismatch",
				doc1_id,
				"kyc_document",
				{"doc2_id": doc2_id, "score": score},
			)
		return result

	async def verify_utility_bill(
		self,
		document_id: str,
		customer_name: str,
		address: str,
	) -> dict[str, Any]:
		"""Verify a utility bill as proof of address.

		Checks: name match against customer name, address parseable, bill not older
		than 90 days. Returns: verified, name_match_score, address_confirmed,
		bill_age_days.
		"""
		assert customer_name and customer_name.strip(), "customer_name is required"
		assert address and address.strip(), "address is required"

		doc = await self._get_document(document_id)
		extracted_name = doc.get("extracted_name", "")
		extracted_address = doc.get("ocr_raw", {}).get("address", "")
		bill_date_str = doc.get("ocr_raw", {}).get("bill_date", "")

		name_score = name_match_score(customer_name, extracted_name) if extracted_name else 0.5
		address_confirmed = bool(extracted_address and len(extracted_address) > 10)

		bill_age_days = None
		if bill_date_str:
			try:
				bill_date = date.fromisoformat(str(bill_date_str)[:10])
				bill_age_days = (_today() - bill_date).days
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		too_old = bill_age_days is not None and bill_age_days > 90
		verified = name_score >= 0.80 and address_confirmed and not too_old

		return {
			"document_id": document_id,
			"customer_name": customer_name,
			"address": address,
			"extracted_name": extracted_name,
			"extracted_address": extracted_address,
			"name_match_score": round(name_score, 4),
			"address_confirmed": address_confirmed,
			"bill_age_days": bill_age_days,
			"too_old": too_old,
			"verified": verified,
			"checked_at": _now().isoformat(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# Biometric Checks (5 methods)
	# ─────────────────────────────────────────────────────────────────────────

	async def perform_liveness_check(
		self,
		session_id: str,
		video_frames: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Anti-spoofing liveness detection from video frames.

		In production delegates to ``biop`` (biometrics) adapter.
		Checks: blink detection, head movement, 3D depth cues, texture analysis.
		Returns: live, liveness_score, spoof_type, confidence.
		"""
		assert session_id and session_id.strip(), "session_id is required"
		assert video_frames is not None, "video_frames is required"

		frame_count = len(video_frames)
		# Liveness score heuristic from frame count
		liveness_score = min(0.95, 0.60 + frame_count * 0.05) if frame_count >= 5 else 0.40
		spoof_detected = liveness_score < 0.80
		spoof_type = "static_photo" if liveness_score < 0.50 else ("print_attack" if liveness_score < 0.70 else None)

		result = {
			"session_id": session_id,
			"frame_count": frame_count,
			"live": not spoof_detected,
			"liveness_score": round(liveness_score, 4),
			"spoof_detected": spoof_detected,
			"spoof_type": spoof_type,
			"confidence": round(liveness_score, 4),
			"checked_at": _now().isoformat(),
		}

		if spoof_detected:
			await self._emit("kyc_liveness_spoof_detected", session_id, "biometric_session", {"spoof_type": spoof_type})
		else:
			await self._emit("kyc_liveness_passed", session_id, "biometric_session", {"liveness_score": liveness_score})

		return result

	async def face_match_id_to_selfie(
		self,
		document_id: str,
		selfie_metadata: dict[str, Any],
	) -> dict[str, Any]:
		"""Compare face on identity document against live selfie.

		``selfie_metadata`` must contain ``token_reference`` pointing to the
		stored selfie image. Returns: match, match_score, confidence, face_quality.
		"""
		assert selfie_metadata.get("token_reference"), "selfie_metadata.token_reference is required"

		doc = await self._get_document(document_id)
		doc_token = doc.get("token_reference", "")
		selfie_token = selfie_metadata["token_reference"]

		# Face match score computed by biometrics adapter — stub returns plausible value
		# In production: call biop adapter with both tokens
		match_score = 0.92 if doc_token and selfie_token else 0.0
		confidence = 0.90
		face_quality = selfie_metadata.get("quality_score", 0.85)

		try:
			assert_biometric_match(match_score)
		except RuleViolation as exc:
			return {
				"document_id": document_id,
				"selfie_token": selfie_token,
				"match": False,
				"match_score": round(match_score, 4),
				"confidence": confidence,
				"face_quality": face_quality,
				"rule_violation": str(exc),
				"checked_at": _now().isoformat(),
			}

		# Persist biometric record
		bio = BiometricData(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=doc.get("application_id", ""),
			biometric_type=BiometricType.facial,
			token_reference=selfie_token,
			match_score=match_score,
			liveness_score=selfie_metadata.get("liveness_score", 0.0),
			spoof_score=selfie_metadata.get("spoof_score", 0.0),
			status=BiometricStatus.live if match_score >= 0.85 else BiometricStatus.failed,
			created_by=self.actor_id,
		)
		await self._store.put(_COL_BIO, bio.model_dump(mode="json"))

		await self._emit(
			"kyc_face_match_completed",
			document_id,
			"kyc_document",
			{"match_score": match_score, "biometric_id": bio.id},
		)
		return {
			"document_id": document_id,
			"biometric_id": bio.id,
			"selfie_token": selfie_token,
			"match": match_score >= 0.85,
			"match_score": round(match_score, 4),
			"confidence": confidence,
			"face_quality": face_quality,
			"checked_at": _now().isoformat(),
		}

	async def fingerprint_check(
		self,
		fingerprint_data: dict[str, Any],
	) -> dict[str, Any]:
		"""Fingerprint verification against enrolled templates.

		``fingerprint_data`` must contain: ``application_id``, ``token_reference``,
		``finger_position`` (e.g. ``right_index``), optional ``quality_score``.
		Returns: verified, match_score, duplicate_found.
		"""
		application_id = fingerprint_data.get("application_id", "")
		token_ref = fingerprint_data.get("token_reference", "")
		assert token_ref, "fingerprint_data.token_reference is required"

		quality = float(fingerprint_data.get("quality_score", 0.85))
		match_score = quality * 0.96  # Simplified — real impl uses NFIQ2 + matcher

		bio = BiometricData(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			biometric_type=BiometricType.fingerprint,
			token_reference=token_ref,
			match_score=round(match_score, 4),
			liveness_score=quality,
			status=BiometricStatus.live if match_score >= 0.85 else BiometricStatus.failed,
			created_by=self.actor_id,
			metadata={
				"finger_position": fingerprint_data.get("finger_position", "unknown"),
				"quality_score": quality,
			},
		)
		await self._store.put(_COL_BIO, bio.model_dump(mode="json"))

		await self._emit("kyc_fingerprint_checked", bio.id, "biometric", {"application_id": application_id})
		return {
			"biometric_id": bio.id,
			"application_id": application_id,
			"verified": match_score >= 0.85,
			"match_score": round(match_score, 4),
			"quality_score": quality,
			"finger_position": fingerprint_data.get("finger_position", "unknown"),
			"duplicate_found": False,  # set by deduplication job
			"checked_at": _now().isoformat(),
		}

	async def voice_biometric(
		self,
		voice_sample: dict[str, Any],
		enrolled_voice: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Voice biometric verification (text-independent speaker recognition).

		``voice_sample`` must contain: ``application_id``, ``token_reference``,
		``duration_seconds``, optional ``language``.
		Returns: verified, match_score, voice_quality, is_enrollment.
		"""
		application_id = voice_sample.get("application_id", "")
		token_ref = voice_sample.get("token_reference", "")
		assert token_ref, "voice_sample.token_reference is required"

		duration = float(voice_sample.get("duration_seconds", 3.0))
		quality = min(1.0, duration / 10.0)  # 10s = max quality
		is_enrollment = enrolled_voice is None
		match_score = quality * 0.90 if not is_enrollment else 0.0

		bio = BiometricData(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			biometric_type=BiometricType.voice,
			token_reference=token_ref,
			match_score=round(match_score, 4),
			liveness_score=quality,
			status=BiometricStatus.live if is_enrollment or match_score >= 0.85 else BiometricStatus.failed,
			created_by=self.actor_id,
			metadata={
				"duration_seconds": duration,
				"language": voice_sample.get("language", "unknown"),
				"is_enrollment": is_enrollment,
			},
		)
		await self._store.put(_COL_BIO, bio.model_dump(mode="json"))

		await self._emit("kyc_voice_biometric_recorded", bio.id, "biometric", {"is_enrollment": is_enrollment})
		return {
			"biometric_id": bio.id,
			"application_id": application_id,
			"is_enrollment": is_enrollment,
			"verified": is_enrollment or match_score >= 0.85,
			"match_score": round(match_score, 4),
			"voice_quality": round(quality, 4),
			"checked_at": _now().isoformat(),
		}

	async def biometric_deduplication(self, customer_id: str) -> dict[str, Any]:
		"""Check if biometrics for a customer already exist in the system.

		Prevents synthetic identity fraud where the same biometric is used
		across multiple accounts. Returns: duplicate_found, matching_customer_ids.
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"

		# Fetch all applications for this tenant
		all_apps = await self._store.query(_COL_APP, {"tenant_id": self.tenant_id})
		customer_app_ids = {a["id"] for a in all_apps if a.get("customer_id") == customer_id.strip()}
		other_app_ids = {a["id"] for a in all_apps if a.get("customer_id") != customer_id.strip()}

		# Fetch biometrics for this customer
		customer_bios: list[dict[str, Any]] = []
		for app_id in customer_app_ids:
			bios = await self._store.query(_COL_BIO, {"application_id": app_id, "tenant_id": self.tenant_id})
			customer_bios.extend(bios)

		# In production: compute embedding hashes and match against other customers.
		# Stub returns no duplicates.
		matching: list[str] = []

		await self._emit("kyc_biometric_deduplication_run", customer_id, "customer", {"biometric_count": len(customer_bios)})
		return {
			"customer_id": customer_id,
			"biometric_count": len(customer_bios),
			"duplicate_found": bool(matching),
			"matching_customer_ids": matching,
			"checked_at": _now().isoformat(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# Screening (8 methods)
	# ─────────────────────────────────────────────────────────────────────────

	async def pep_screening(
		self,
		name: str,
		dob: str | date | None = None,
		nationality: str = "",
		aliases: list[str] | None = None,
		*,
		application_id: str = "",
		match_threshold: float = 0.85,
	) -> dict[str, Any]:
		"""Screen a name against PEP lists (Dow Jones, World-Check, local lists).

		Africa-specific lists: Kenya Gazette, Nigeria Official Gazette,
		South Africa SAPS, AU political office holders.

		Returns: is_pep, pep_type, positions, relationships, confidence.
		"""
		assert name and name.strip(), "name is required"
		aliases = aliases or []
		all_names = [name.strip()] + [a.strip() for a in aliases if a.strip()]

		# Screening stub — real impl calls World-Check / Dow Jones API
		# Deterministically return no-hit for names that don't look politically risky
		hit_indicators = {"minister", "senator", "governor", "president", "commissioner", "general", "colonel"}
		is_hit = any(w in name.lower() for w in hit_indicators)
		match_score = 0.97 if is_hit else 0.0
		pep_category = "minister" if is_hit else ""
		pep_level = "senior" if is_hit else ""
		positions: list[dict[str, Any]] = (
			[{"title": "Cabinet Minister", "country": nationality or "UNKNOWN", "since": "2020-01-01"}] if is_hit else []
		)

		check = PEPCheck(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			full_name=name.strip(),
			date_of_birth=date.fromisoformat(str(dob)[:10]) if dob else None,
			nationality=nationality,
			match_threshold=match_threshold,
			status="hit" if is_hit else "clear",
			is_hit=is_hit,
			match_score=round(match_score, 4),
			matched_name=name if is_hit else "",
			pep_category=pep_category,
			pep_level=pep_level,
			source_list="World-Check" if is_hit else "",
			created_by=self.actor_id,
		)
		record = check.model_dump(mode="json")
		if application_id:
			await self._store.put(_COL_PEP, record)

		await self._emit("kyc_pep_screening_completed", check.id, "pep_check", {"is_hit": is_hit, "name": name})
		return {
			"check_id": check.id,
			"name": name,
			"aliases_checked": aliases,
			"is_pep": is_hit,
			"pep_type": pep_category,
			"pep_level": pep_level,
			"positions": positions,
			"relationships": [],
			"match_score": round(match_score, 4),
			"source_lists": ["World-Check", "Dow Jones", "KE_Gazette", "NG_Gazette"] if is_hit else [],
			"confidence": round(match_score, 4) if is_hit else 0.99,
			"screening_status": "hit" if is_hit else "clear",
			"screened_at": _now().isoformat(),
		}

	async def sanctions_screening(
		self,
		name: str,
		nationality: str = "",
		id_number: str = "",
		*,
		application_id: str = "",
		lists: list[str] | None = None,
		match_threshold: float = 0.85,
	) -> dict[str, Any]:
		"""Screen against OFAC SDN, UN Consolidated, EU Asset Freeze, AU Sanctions.

		Returns: match_type, list_name, confidence, matched_fields.
		"""
		assert name and name.strip(), "name is required"
		lists_to_screen = lists or _DEFAULT_SANCTIONS_LISTS

		# Sanctions screening stub — real impl calls OFAC/UN/EU APIs
		# High-confidence non-hit for standard names
		is_hit = False  # Production: fuzzy match across all lists
		matched_lists: list[str] = []
		match_score = 0.0

		check = SanctionCheck(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			full_name=name.strip(),
			nationality=nationality,
			lists_screened=lists_to_screen,
			match_threshold=match_threshold,
			status="clear",
			is_hit=is_hit,
			matched_lists=matched_lists,
			match_score=round(match_score, 4),
			created_by=self.actor_id,
		)
		record = check.model_dump(mode="json")
		if application_id:
			await self._store.put(_COL_SANC, record)

		await self._emit("kyc_sanctions_screening_completed", check.id, "sanction_check", {"is_hit": is_hit})
		return {
			"check_id": check.id,
			"name": name,
			"nationality": nationality,
			"id_number": id_number,
			"is_sanctioned": is_hit,
			"match_type": "none",
			"list_name": matched_lists[0] if matched_lists else "",
			"matched_lists": matched_lists,
			"confidence": 0.99 if not is_hit else round(match_score, 4),
			"matched_fields": [],
			"lists_screened": lists_to_screen,
			"screened_at": _now().isoformat(),
		}

	async def adverse_media_screening(
		self,
		name: str,
		aliases: list[str] | None = None,
		*,
		application_id: str = "",
		categories: list[str] | None = None,
	) -> dict[str, Any]:
		"""Web-based adverse media screening for negative news.

		Categories: fraud, money_laundering, terrorism, corruption, drug_trafficking,
		human_trafficking, financial_crime.

		Returns: hits, severity, sources, summary.
		"""
		assert name and name.strip(), "name is required"
		aliases = aliases or []
		default_categories = [
			"financial_crime", "fraud", "corruption", "terrorism",
			"drug_trafficking", "human_trafficking", "money_laundering",
		]
		cats = categories or default_categories

		# Adverse media stub — real impl calls news API / web search
		is_hit = False
		hit_categories: list[str] = []
		article_count = 0
		severity = "none"

		check = AdverseMediaCheck(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			full_name=name.strip(),
			search_terms=[name.strip()] + aliases,
			categories=cats,
			status="clear",
			is_hit=is_hit,
			hit_categories=hit_categories,
			article_count=article_count,
			summary="No adverse media found.",
			created_by=self.actor_id,
		)
		record = check.model_dump(mode="json")
		if application_id:
			await self._store.put(_COL_AMEDIA, record)

		await self._emit("kyc_adverse_media_screening_completed", check.id, "adverse_media_check", {"is_hit": is_hit})
		return {
			"check_id": check.id,
			"name": name,
			"aliases_checked": aliases,
			"hits": is_hit,
			"severity": severity,
			"hit_categories": hit_categories,
			"article_count": article_count,
			"sources": [],
			"summary": check.summary,
			"categories_screened": cats,
			"screened_at": _now().isoformat(),
		}

	async def watchlist_screening(
		self,
		name: str,
		id_number: str = "",
		dob: str | date | None = None,
		*,
		application_id: str = "",
	) -> dict[str, Any]:
		"""Combined watchlist screening: PEP + sanctions + adverse media + Interpol.

		Orchestrates the three individual screening calls and returns a unified result.
		"""
		assert name and name.strip(), "name is required"

		pep = await self.pep_screening(name, dob=dob, application_id=application_id)
		sanc = await self.sanctions_screening(name, id_number=id_number, application_id=application_id)
		amedia = await self.adverse_media_screening(name, application_id=application_id)

		overall_hit = pep["is_pep"] or sanc["is_sanctioned"] or amedia["hits"]
		risk_contribution = (
			(30 if pep["is_pep"] else 0)
			+ (50 if sanc["is_sanctioned"] else 0)
			+ (15 if amedia["hits"] else 0)
		)

		await self._emit(
			"kyc_watchlist_screening_completed",
			application_id or name,
			"watchlist",
			{"overall_hit": overall_hit, "risk_contribution": risk_contribution},
		)
		return {
			"name": name,
			"id_number": id_number,
			"application_id": application_id,
			"overall_hit": overall_hit,
			"risk_contribution": risk_contribution,
			"pep": pep,
			"sanctions": sanc,
			"adverse_media": amedia,
			"screened_at": _now().isoformat(),
		}

	async def ongoing_monitoring_trigger(
		self,
		customer_id: str,
		trigger_reason: str,
	) -> dict[str, Any]:
		"""Trigger a full re-screening and risk re-assessment.

		Trigger sources: periodic_review, transaction_alert, news_alert,
		dormancy_reactivation, regulator_request.
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"
		assert trigger_reason and trigger_reason.strip(), "trigger_reason is required"

		# Find active applications for this customer
		apps = await self._store.query(_COL_APP, {"customer_id": customer_id.strip(), "tenant_id": self.tenant_id})
		active_apps = [a for a in apps if a.get("status") == ApplicationStatus.approved.value]

		triggered: list[dict[str, Any]] = []
		for app in active_apps:
			# Create a periodic_refresh review
			review = KYCReview(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				application_id=app["id"],
				review_type=ReviewType.periodic_refresh,
				status=ReviewStatus.open,
				notes=f"Ongoing monitoring triggered: {trigger_reason}",
				created_by=self.actor_id,
			)
			await self._store.put(_COL_REVIEW, review.model_dump(mode="json"))

			# Set application to pending_review
			app["status"] = ApplicationStatus.pending_review.value
			app["updated_at"] = _now().isoformat()
			await self._store.put(_COL_APP, app)
			triggered.append({"application_id": app["id"], "review_id": review.id})

		await self._emit(
			"kyc_ongoing_monitoring_triggered",
			customer_id,
			"customer",
			{"trigger_reason": trigger_reason, "applications_triggered": len(triggered)},
		)
		return {
			"customer_id": customer_id,
			"trigger_reason": trigger_reason,
			"applications_triggered": len(triggered),
			"triggered_applications": triggered,
			"triggered_at": _now().isoformat(),
		}

	async def interpol_check(
		self,
		name: str,
		nationality: str,
	) -> dict[str, Any]:
		"""Check name against Interpol Red Notices and diffusions.

		Returns: hit, notice_type, notice_id, crime_category, issuing_country.
		"""
		assert name and name.strip(), "name is required"
		assert nationality and len(nationality) == 2, "nationality must be ISO-3166-1 alpha-2"

		# Interpol API stub
		is_hit = False
		result = {
			"name": name.strip(),
			"nationality": nationality.upper(),
			"hit": is_hit,
			"notice_type": None,
			"notice_id": None,
			"crime_category": None,
			"issuing_country": None,
			"confidence": 0.99,
			"screened_at": _now().isoformat(),
		}
		await self._emit("kyc_interpol_checked", name, "interpol_check", {"hit": is_hit})
		return result

	async def local_watchlist_check(
		self,
		name: str,
		id_number: str,
		country: str,
	) -> dict[str, Any]:
		"""Check against country-level watchlists.

		Routing:
		- KE: DCI Kenya blacklist, FRC Kenya (Financial Reporting Centre)
		- NG: EFCC watchlist, NFIU Nigeria
		- GH: FIC Ghana watchlist
		- ZA: FIC South Africa, SAPS wanted persons
		- TZ: FIU Tanzania
		"""
		assert name and name.strip(), "name is required"
		assert country and len(country) == 2, "country is required"

		local_lists_map = {
			"KE": ["DCI_Kenya", "FRC_Kenya_Blacklist"],
			"NG": ["EFCC_Watchlist", "NFIU_Nigeria"],
			"GH": ["FIC_Ghana"],
			"ZA": ["FIC_South_Africa", "SAPS_Wanted"],
			"TZ": ["FIU_Tanzania"],
			"UG": ["FIA_Uganda"],
			"RW": ["FIU_Rwanda"],
		}
		lists = local_lists_map.get(country.upper(), [f"FIU_{country.upper()}"])

		# Local watchlist stub
		is_hit = False
		result = {
			"name": name.strip(),
			"id_number": id_number,
			"country": country.upper(),
			"hit": is_hit,
			"matched_list": None,
			"lists_screened": lists,
			"confidence": 0.99,
			"screened_at": _now().isoformat(),
		}
		await self._emit("kyc_local_watchlist_checked", name, "local_watchlist", {"country": country, "hit": is_hit})
		return result

	async def batch_screening(self, customer_ids: list[str]) -> dict[str, Any]:
		"""Bulk periodic re-screening for a list of customer IDs.

		Fetches each customer's latest application and runs watchlist_screening
		concurrently (sequentially in this reference implementation for simplicity).
		Returns a summary with per-customer results.
		"""
		assert customer_ids, "customer_ids must not be empty"

		results: list[dict[str, Any]] = []
		hits = 0
		errors = 0

		for cid in customer_ids:
			try:
				apps = await self._store.query(_COL_APP, {"customer_id": cid, "tenant_id": self.tenant_id})
				if not apps:
					results.append({"customer_id": cid, "status": "no_application", "hit": False})
					continue
				latest_app = sorted(apps, key=lambda a: a.get("created_at", ""), reverse=True)[0]
				result = await self.watchlist_screening(
					latest_app.get("legal_name", cid),
					application_id=latest_app["id"],
				)
				hit = result["overall_hit"]
				if hit:
					hits += 1
				results.append({"customer_id": cid, "application_id": latest_app["id"], "hit": hit, "status": "screened"})
			except Exception as exc:
				errors += 1
				results.append({"customer_id": cid, "status": "error", "error": str(exc), "hit": False})

		await self._emit(
			"kyc_batch_screening_completed",
			self.tenant_id,
			"batch",
			{"total": len(customer_ids), "hits": hits, "errors": errors},
		)
		return {
			"total": len(customer_ids),
			"screened": len(customer_ids) - errors,
			"hits": hits,
			"errors": errors,
			"results": results,
			"screened_at": _now().isoformat(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# Risk & Scoring (6 methods)
	# ─────────────────────────────────────────────────────────────────────────

	async def calculate_risk_score(self, application_id: str) -> dict[str, Any]:
		"""Compute composite KYC risk score (0–100).

		Factors: jurisdiction risk, PEP status, sanctions, adverse media,
		document quality, biometric results, customer type, occupation,
		source of funds, complex ownership.

		Returns: score (0-100), risk_level, contributing_factors.
		"""
		app = await self._require_app(application_id)
		pep_checks = await self._store.query(_COL_PEP, {"application_id": application_id, "tenant_id": self.tenant_id})
		sanc_checks = await self._store.query(_COL_SANC, {"application_id": application_id, "tenant_id": self.tenant_id})
		amedia_checks = await self._store.query(_COL_AMEDIA, {"application_id": application_id, "tenant_id": self.tenant_id})
		docs = await self._store.query(_COL_DOC, {"application_id": application_id, "tenant_id": self.tenant_id})
		bios = await self._store.query(_COL_BIO, {"application_id": application_id, "tenant_id": self.tenant_id})

		country = app.get("country_code", "")
		customer_type = app.get("customer_type", "individual")

		is_pep = any(p.get("is_hit") for p in pep_checks)
		is_sanctioned = any(s.get("is_hit") for s in sanc_checks)
		is_adverse_media = any(a.get("is_hit") for a in amedia_checks)
		high_risk_country = is_high_risk_country(country)
		low_conf_docs = any(float(d.get("confidence", 1.0)) < 0.75 for d in docs)
		expired_doc = any(d.get("status") == DocumentStatus.expired.value for d in docs)
		liveness_fail = any(b.get("status") == BiometricStatus.spoof_detected.value for b in bios)
		biometric_mismatch = any(float(b.get("match_score", 1.0)) < 0.85 for b in bios)

		factors: dict[str, bool | int | float] = {
			"is_pep": is_pep,
			"is_sanctioned": is_sanctioned,
			"is_adverse_media": is_adverse_media,
			"high_risk_country": high_risk_country,
			"high_risk_industry": is_high_risk_industry(app.get("metadata", {}).get("industry_code", "")),
			"complex_ownership_structure": customer_type in ("business", "trust", "partnership"),
			"low_confidence_documents": low_conf_docs,
			"expired_document": expired_doc,
			"liveness_fail": liveness_fail,
			"biometric_mismatch": biometric_mismatch,
			"is_refugee": app.get("is_refugee", False),
			"is_informal_sector": app.get("is_informal_sector", False),
		}

		score, breakdown = calculate_risk_score(factors)
		band = calculate_risk_band(score)

		# Persist risk profile
		rp = RiskProfile(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			customer_type=CustomerType(customer_type),
			country_code=country,
			risk_score=score,
			risk_band=RiskBand(band),
			is_pep=is_pep,
			is_sanctioned=is_sanctioned,
			is_adverse_media=is_adverse_media,
			high_risk_country=high_risk_country,
			score_breakdown=breakdown,
			edd_required=band in ("high", "very_high", "unacceptable") or is_pep,
			created_by=self.actor_id,
		)
		await self._store.put(_COL_RISK, rp.model_dump(mode="json"))

		# Update application risk fields
		app["risk_score"] = score
		app["risk_band"] = band
		app["updated_at"] = _now().isoformat()
		if band in ("high", "very_high", "unacceptable"):
			app["edd_triggered_at"] = app.get("edd_triggered_at") or _now().isoformat()
		await self._store.put(_COL_APP, app)

		await self._emit(
			"kyc_risk_score_calculated",
			application_id,
			"kyc_application",
			{"score": score, "band": band, "edd_required": rp.edd_required},
		)
		return {
			"risk_profile_id": rp.id,
			"application_id": application_id,
			"score": score,
			"risk_level": band,
			"risk_band": band,
			"edd_required": rp.edd_required,
			"contributing_factors": breakdown,
			"factor_inputs": {k: bool(v) for k, v in factors.items()},
			"assessed_at": _now().isoformat(),
		}

	async def update_risk_profile(
		self,
		customer_id: str,
		trigger: str,
		new_data: dict[str, Any],
	) -> dict[str, Any]:
		"""Re-compute risk profile after a trigger event.

		Trigger examples: new_transaction_pattern, adverse_news_alert,
		sanctions_list_update, customer_data_change.
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"
		assert trigger and trigger.strip(), "trigger is required"

		apps = await self._store.query(_COL_APP, {"customer_id": customer_id.strip(), "tenant_id": self.tenant_id})
		updated: list[dict[str, Any]] = []
		for app in apps:
			if app.get("status") == ApplicationStatus.approved.value:
				result = await self.calculate_risk_score(app["id"])
				result["trigger"] = trigger
				updated.append(result)

		await self._emit(
			"kyc_risk_profile_updated",
			customer_id,
			"customer",
			{"trigger": trigger, "applications_updated": len(updated)},
		)
		return {
			"customer_id": customer_id,
			"trigger": trigger,
			"applications_updated": len(updated),
			"updates": updated,
			"updated_at": _now().isoformat(),
		}

	async def risk_based_due_diligence(self, customer_id: str) -> dict[str, Any]:
		"""Determine appropriate due diligence level based on risk band.

		FATF risk-based approach:
		- Low: Simplified Due Diligence (SDD)
		- Medium: Standard Customer Due Diligence (CDD)
		- High / Very High: Enhanced Due Diligence (EDD)
		- Unacceptable: Decline relationship
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"

		apps = await self._store.query(_COL_APP, {"customer_id": customer_id.strip(), "tenant_id": self.tenant_id})
		if not apps:
			raise KeyError(f"no applications found for customer: {customer_id}")

		latest_app = sorted(apps, key=lambda a: a.get("created_at", ""), reverse=True)[0]
		risk_band_val = latest_app.get("risk_band", "low")
		risk_score = int(latest_app.get("risk_score", 0))

		dd_map = {
			"low": {
				"level": "simplified",
				"label": "Simplified Due Diligence (SDD)",
				"requirements": ["government_id", "address_verification"],
				"monitoring_frequency": "annual",
			},
			"medium": {
				"level": "standard",
				"label": "Standard Customer Due Diligence (CDD)",
				"requirements": ["government_id", "address_verification", "source_of_funds", "pep_screening", "sanctions_screening"],
				"monitoring_frequency": "semi_annual",
			},
			"high": {
				"level": "enhanced",
				"label": "Enhanced Due Diligence (EDD)",
				"requirements": ["government_id", "address_verification", "source_of_funds", "source_of_wealth", "beneficial_owner", "pep_screening", "sanctions_screening", "adverse_media", "purpose_of_relationship"],
				"monitoring_frequency": "quarterly",
			},
			"very_high": {
				"level": "enhanced",
				"label": "Enhanced Due Diligence (EDD) — Very High Risk",
				"requirements": ["government_id", "address_verification", "source_of_funds", "source_of_wealth", "beneficial_owner", "pep_screening", "sanctions_screening", "adverse_media", "purpose_of_relationship", "senior_management_approval"],
				"monitoring_frequency": "monthly",
			},
			"unacceptable": {
				"level": "decline",
				"label": "Decline — Unacceptable Risk",
				"requirements": [],
				"monitoring_frequency": "none",
			},
		}

		dd = dd_map.get(risk_band_val, dd_map["medium"])
		return {
			"customer_id": customer_id,
			"application_id": latest_app["id"],
			"risk_band": risk_band_val,
			"risk_score": risk_score,
			**dd,
			"assessed_at": _now().isoformat(),
		}

	async def enhanced_due_diligence(self, customer_id: str) -> dict[str, Any]:
		"""Run Enhanced Due Diligence for high-risk customers (FATF R.19, R.20).

		Collects: source of wealth, source of funds, purpose of relationship,
		beneficial ownership, senior management approval.
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"

		apps = await self._store.query(_COL_APP, {"customer_id": customer_id.strip(), "tenant_id": self.tenant_id})
		if not apps:
			raise KeyError(f"no applications for customer: {customer_id}")

		latest_app = sorted(apps, key=lambda a: a.get("created_at", ""), reverse=True)[0]
		application_id = latest_app["id"]

		# Create EDD review record
		review = KYCReview(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=application_id,
			review_type=ReviewType.enhanced_due_diligence,
			status=ReviewStatus.open,
			notes="EDD initiated by system",
			created_by=self.actor_id,
		)
		await self._store.put(_COL_REVIEW, review.model_dump(mode="json"))

		# Update application status
		latest_app["status"] = ApplicationStatus.pending_edd.value
		latest_app["edd_triggered_at"] = _now().isoformat()
		latest_app["updated_at"] = _now().isoformat()
		await self._store.put(_COL_APP, latest_app)

		await self._emit(
			"kyc_edd_initiated",
			customer_id,
			"customer",
			{"application_id": application_id, "review_id": review.id},
		)
		return {
			"customer_id": customer_id,
			"application_id": application_id,
			"edd_review_id": review.id,
			"status": ApplicationStatus.pending_edd.value,
			"required_documents": [
				"source_of_wealth_declaration",
				"source_of_funds_evidence",
				"purpose_of_relationship_form",
				"beneficial_owner_declaration",
				"senior_management_approval",
			],
			"regulatory_reference": "FATF Recommendations 19, 20",
			"initiated_at": _now().isoformat(),
		}

	async def beneficial_ownership_verification(
		self,
		entity_id: str,
		ubo_threshold: float = 25.0,
	) -> dict[str, Any]:
		"""Identify and verify Ultimate Beneficial Owners with ≥threshold% ownership.

		Regulatory reference: FATF R.24, Companies Act (KE/NG/GH/ZA), CBK guidelines.
		"""
		assert entity_id and entity_id.strip(), "entity_id is required"

		# Fetch business KYC record
		bkyc_records = await self._store.query(_COL_BKYC, {"application_id": entity_id, "tenant_id": self.tenant_id})
		app = None
		try:
			app = await self._require_app(entity_id)
		except KeyError as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		ubo_records = await self._store.query(_COL_UBO, {"application_id": entity_id, "tenant_id": self.tenant_id})
		controlling_ubos = [
			u for u in ubo_records
			if float(u.get("ownership_percentage", 0)) >= ubo_threshold
		]
		total_declared = sum(float(u.get("ownership_percentage", 0)) for u in ubo_records)
		ownership_gap = max(0.0, 100.0 - total_declared)

		assert_ubo_declared(
			app.get("customer_type", "business") if app else "business",
			len(ubo_records),
		)

		# Screen each UBO
		ubo_screening_results: list[dict[str, Any]] = []
		for ubo in controlling_ubos:
			screening = await self.pep_screening(
				ubo.get("full_name", ""),
				dob=ubo.get("date_of_birth"),
				nationality=ubo.get("nationality", ""),
				application_id=entity_id,
			)
			ubo_screening_results.append({
				"ubo_id": ubo["id"],
				"full_name": ubo.get("full_name"),
				"ownership_pct": ubo.get("ownership_percentage"),
				"is_pep": screening["is_pep"],
				"screening_id": screening["check_id"],
			})

		await self._emit(
			"kyc_ubo_verification_completed",
			entity_id,
			"entity",
			{"ubo_count": len(ubo_records), "controlling_ubo_count": len(controlling_ubos)},
		)
		return {
			"entity_id": entity_id,
			"ubo_threshold_pct": ubo_threshold,
			"total_ubos_declared": len(ubo_records),
			"controlling_ubos": len(controlling_ubos),
			"total_ownership_declared": round(total_declared, 2),
			"ownership_gap_pct": round(ownership_gap, 2),
			"ubo_screening": ubo_screening_results,
			"verified_at": _now().isoformat(),
		}

	async def source_of_funds_verification(
		self,
		customer_id: str,
		declared_source: str,
		supporting_docs: list[str],
	) -> dict[str, Any]:
		"""Verify declared source of funds against supporting documentation.

		Declared sources: salary, business_income, investment_returns,
		inheritance, property_sale, loan, gift, pension, remittance.
		Returns: verified, corroborated, consistency_score, risk_flag.
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"
		assert declared_source and declared_source.strip(), "declared_source is required"

		valid_sources = {
			"salary", "business_income", "investment_returns", "inheritance",
			"property_sale", "loan", "gift", "pension", "remittance", "other",
		}
		source_clean = declared_source.strip().lower()
		if source_clean not in valid_sources:
			source_clean = "other"

		# High-risk source flags
		high_risk_sources = {"cash_business", "gambling_winnings", "crypto_trading", "unknown"}
		risk_flag = source_clean in high_risk_sources

		# Corroboration: check supporting docs are present in the document store
		corroborated_docs: list[str] = []
		for doc_id in supporting_docs:
			try:
				doc = await self._get_document(doc_id)
				if doc.get("status") == DocumentStatus.verified.value:
					corroborated_docs.append(doc_id)
			except KeyError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		corroboration_rate = len(corroborated_docs) / max(len(supporting_docs), 1)
		verified = corroboration_rate >= 0.5 and not risk_flag

		await self._emit(
			"kyc_source_of_funds_verified",
			customer_id,
			"customer",
			{"declared_source": source_clean, "verified": verified, "risk_flag": risk_flag},
		)
		return {
			"customer_id": customer_id,
			"declared_source": source_clean,
			"supporting_docs_provided": len(supporting_docs),
			"corroborated_docs": len(corroborated_docs),
			"corroboration_rate": round(corroboration_rate, 4),
			"verified": verified,
			"consistency_score": round(corroboration_rate, 4),
			"risk_flag": risk_flag,
			"verified_at": _now().isoformat(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# Compliance & Reporting (6 methods)
	# ─────────────────────────────────────────────────────────────────────────

	async def kyc_expiry_report(self, days_ahead: int = 90) -> dict[str, Any]:
		"""List applications expiring within `days_ahead` days.

		Buckets: expiring_within_30_days, expiring_within_90_days, already_expired.
		"""
		assert 1 <= days_ahead <= 365, "days_ahead must be between 1 and 365"

		apps = await self._store.query(_COL_APP, {"tenant_id": self.tenant_id, "status": ApplicationStatus.approved.value})
		today = _today()

		within_30: list[str] = []
		within_90: list[str] = []
		already_expired: list[str] = []

		for app in apps:
			expiry_str = app.get("expiry_date")
			if not expiry_str:
				continue
			try:
				expiry = date.fromisoformat(str(expiry_str)[:10])
			except ValueError:
				continue
			remaining = (expiry - today).days
			if remaining < 0:
				already_expired.append(app["id"])
			elif remaining <= 30:
				within_30.append(app["id"])
			elif remaining <= days_ahead:
				within_90.append(app["id"])

		await self._emit("kyc_expiry_report_generated", self.tenant_id, "tenant", {"days_ahead": days_ahead})
		return {
			"tenant_id": self.tenant_id,
			"days_ahead": days_ahead,
			"expiring_within_30_days": within_30,
			"expiring_within_90_days": within_90,
			"already_expired": already_expired,
			"total_flagged": len(within_30) + len(within_90) + len(already_expired),
			"generated_at": _now().isoformat(),
		}

	async def kyc_refresh(self, customer_id: str, reason: str) -> dict[str, Any]:
		"""Initiate periodic re-KYC for a customer.

		Creates a periodic_refresh review and resets application status to in_progress.
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"
		assert reason and reason.strip(), "reason is required"

		apps = await self._store.query(_COL_APP, {"customer_id": customer_id.strip(), "tenant_id": self.tenant_id})
		active_apps = [a for a in apps if a.get("status") == ApplicationStatus.approved.value]

		refreshed: list[str] = []
		for app in active_apps:
			review = KYCReview(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				application_id=app["id"],
				review_type=ReviewType.periodic_refresh,
				status=ReviewStatus.open,
				notes=f"KYC refresh: {reason}",
				created_by=self.actor_id,
			)
			await self._store.put(_COL_REVIEW, review.model_dump(mode="json"))
			app["status"] = ApplicationStatus.in_progress.value
			app["updated_at"] = _now().isoformat()
			await self._store.put(_COL_APP, app)
			refreshed.append(app["id"])

		await self._emit("kyc_refresh_initiated", customer_id, "customer", {"reason": reason, "count": len(refreshed)})
		return {
			"customer_id": customer_id,
			"reason": reason,
			"applications_refreshed": len(refreshed),
			"application_ids": refreshed,
			"initiated_at": _now().isoformat(),
		}

	async def generate_kyc_certificate(self, customer_id: str) -> dict[str, Any]:
		"""Generate a KYC completion certificate for a verified customer.

		Returns certificate metadata including a unique certificate reference,
		validity period, and summary of verification checks performed.
		"""
		assert customer_id and customer_id.strip(), "customer_id is required"

		apps = await self._store.query(_COL_APP, {"customer_id": customer_id.strip(), "tenant_id": self.tenant_id})
		approved = [a for a in apps if a.get("status") == ApplicationStatus.approved.value]

		if not approved:
			raise ValueError(f"no approved KYC application found for customer: {customer_id}")

		app = sorted(approved, key=lambda a: a.get("last_verified_at", ""), reverse=True)[0]
		application_id = app["id"]

		# Gather verification evidence summary
		docs = await self._store.query(_COL_DOC, {"application_id": application_id, "tenant_id": self.tenant_id})
		verified_docs = [d for d in docs if d.get("status") == DocumentStatus.verified.value]
		pep_checks = await self._store.query(_COL_PEP, {"application_id": application_id, "tenant_id": self.tenant_id})
		sanc_checks = await self._store.query(_COL_SANC, {"application_id": application_id, "tenant_id": self.tenant_id})

		cert_id = f"KYC-CERT-{uuid7str()[:8].upper()}"
		issued_at = _now()
		expiry_str = app.get("expiry_date", "")
		expiry = date.fromisoformat(str(expiry_str)[:10]) if expiry_str else (_today() + timedelta(days=365))

		certificate = {
			"certificate_id": cert_id,
			"customer_id": customer_id,
			"application_id": application_id,
			"tenant_id": self.tenant_id,
			"legal_name": app.get("legal_name", ""),
			"customer_type": app.get("customer_type", ""),
			"jurisdiction": app.get("country_code", ""),
			"kyc_tier": app.get("kyc_tier", "standard"),
			"risk_band": app.get("risk_band", "low"),
			"risk_score": app.get("risk_score", 0),
			"verified_documents": [
				{"doc_type": d.get("document_type"), "doc_id": d["id"]} for d in verified_docs
			],
			"pep_cleared": all(not p.get("is_hit") for p in pep_checks),
			"sanctions_cleared": all(not s.get("is_hit") for s in sanc_checks),
			"issued_at": issued_at.isoformat(),
			"expires_at": expiry.isoformat(),
			"issued_by": self.actor_id,
			"format": "JSON",  # In production: generate PDF via document generation service
			"checksum": hashlib.sha256(f"{cert_id}{customer_id}{issued_at.isoformat()}".encode()).hexdigest()[:16],
		}

		await self._emit("kyc_certificate_generated", cert_id, "kyc_certificate", {"customer_id": customer_id})
		return certificate

	async def kyc_audit_report(
		self,
		period_from: str | date,
		period_to: str | date,
	) -> dict[str, Any]:
		"""Generate an audit report for all KYC events in a period.

		Returns aggregated event counts, top actors, and event timeline.
		"""
		from_dt = date.fromisoformat(str(period_from)[:10]) if isinstance(period_from, str) else period_from
		to_dt = date.fromisoformat(str(period_to)[:10]) if isinstance(period_to, str) else period_to
		assert from_dt <= to_dt, "period_from must be before period_to"

		events = await self._store.query(_COL_AUDIT, {"tenant_id": self.tenant_id}, limit=10_000)

		# Filter by period
		period_events = []
		for evt in events:
			ts = evt.get("timestamp", "")
			if ts:
				evt_date = date.fromisoformat(str(ts)[:10])
				if from_dt <= evt_date <= to_dt:
					period_events.append(evt)

		# Aggregate
		by_type: dict[str, int] = {}
		by_actor: dict[str, int] = {}
		for evt in period_events:
			etype = evt.get("event_type", "unknown")
			actor = evt.get("actor_id", "unknown")
			by_type[etype] = by_type.get(etype, 0) + 1
			by_actor[actor] = by_actor.get(actor, 0) + 1

		return {
			"tenant_id": self.tenant_id,
			"period_from": str(from_dt),
			"period_to": str(to_dt),
			"total_events": len(period_events),
			"events_by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
			"events_by_actor": dict(sorted(by_actor.items(), key=lambda x: -x[1])[:10]),
			"generated_at": _now().isoformat(),
		}

	async def failed_verification_report(self, period: str) -> dict[str, Any]:
		"""Report on failed KYC verifications for a given period.

		``period``: ISO month string (``2026-05``) or a date range in
		``YYYY-MM-DD:YYYY-MM-DD`` format.
		"""
		assert period and period.strip(), "period is required"

		# Parse period
		if ":" in period:
			parts = period.split(":")
			from_dt = date.fromisoformat(parts[0].strip()[:10])
			to_dt = date.fromisoformat(parts[1].strip()[:10])
		else:
			# ISO month: YYYY-MM
			year, month = int(period[:4]), int(period[5:7])
			from_dt = date(year, month, 1)
			last_day = (date(year, month % 12 + 1, 1) - timedelta(days=1)) if month < 12 else date(year, 12, 31)
			to_dt = last_day

		apps = await self._store.query(_COL_APP, {"tenant_id": self.tenant_id}, limit=10_000)
		rejected_apps = [
			a for a in apps
			if a.get("status") == ApplicationStatus.rejected.value
		]

		# Filter by period
		period_rejected = []
		for app in rejected_apps:
			updated = app.get("updated_at", "")
			if updated:
				upd_date = date.fromisoformat(str(updated)[:10])
				if from_dt <= upd_date <= to_dt:
					period_rejected.append(app)

		by_reason: dict[str, int] = {}
		by_country: dict[str, int] = {}
		for app in period_rejected:
			reason = app.get("metadata", {}).get("rejection_reason", "unspecified")
			country = app.get("country_code", "UNKNOWN")
			by_reason[reason] = by_reason.get(reason, 0) + 1
			by_country[country] = by_country.get(country, 0) + 1

		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"period_from": str(from_dt),
			"period_to": str(to_dt),
			"total_failed": len(period_rejected),
			"by_rejection_reason": dict(sorted(by_reason.items(), key=lambda x: -x[1])),
			"by_country": dict(sorted(by_country.items(), key=lambda x: -x[1])),
			"failed_application_ids": [a["id"] for a in period_rejected],
			"generated_at": _now().isoformat(),
		}

	async def regulator_stats_report(
		self,
		period: str,
		jurisdiction: str,
	) -> dict[str, Any]:
		"""Generate regulatory statistics report for submission to financial authorities.

		Compliant with:
		- CBK Kenya Quarterly KYC Report
		- CBN Nigeria Financial System Report
		- FIC South Africa Annual Report
		- FATF Risk Assessment template
		"""
		assert period and period.strip(), "period is required"
		assert jurisdiction and len(jurisdiction) == 2, "jurisdiction must be ISO-3166-1 alpha-2"
		jurisdiction = jurisdiction.upper()

		apps = await self._store.query(
			_COL_APP,
			{"tenant_id": self.tenant_id, "country_code": jurisdiction},
			limit=10_000,
		)

		by_status: dict[str, int] = {}
		by_type: dict[str, int] = {}
		by_risk_band: dict[str, int] = {}
		pep_count = 0
		sanctions_count = 0
		edd_count = 0

		for app in apps:
			status = app.get("status", "unknown")
			ctype = app.get("customer_type", "unknown")
			band = app.get("risk_band", "unknown")
			by_status[status] = by_status.get(status, 0) + 1
			by_type[ctype] = by_type.get(ctype, 0) + 1
			by_risk_band[band] = by_risk_band.get(band, 0) + 1
			if app.get("metadata", {}).get("edd_triggered"):
				edd_count += 1

		# Screening stats
		pep_hits = await self._store.query(_COL_PEP, {"tenant_id": self.tenant_id}, limit=10_000)
		sanc_hits = await self._store.query(_COL_SANC, {"tenant_id": self.tenant_id}, limit=10_000)
		pep_count = sum(1 for p in pep_hits if p.get("is_hit"))
		sanctions_count = sum(1 for s in sanc_hits if s.get("is_hit"))

		regulator_name_map = {
			"KE": "Central Bank of Kenya (CBK)",
			"NG": "Central Bank of Nigeria (CBN)",
			"GH": "Bank of Ghana (BoG)",
			"ZA": "Financial Intelligence Centre (FIC)",
			"TZ": "Bank of Tanzania (BoT)",
			"UG": "Bank of Uganda (BoU)",
			"ET": "National Bank of Ethiopia (NBE)",
			"RW": "National Bank of Rwanda (BNR)",
		}

		return {
			"tenant_id": self.tenant_id,
			"jurisdiction": jurisdiction,
			"regulator": regulator_name_map.get(jurisdiction, f"Financial Authority {jurisdiction}"),
			"period": period,
			"total_applications": len(apps),
			"by_status": by_status,
			"by_customer_type": by_type,
			"by_risk_band": by_risk_band,
			"pep_hits": pep_count,
			"sanctions_hits": sanctions_count,
			"edd_triggered": edd_count,
			"high_risk_customers": by_risk_band.get("high", 0) + by_risk_band.get("very_high", 0) + by_risk_band.get("unacceptable", 0),
			"approval_rate": round(
				by_status.get(ApplicationStatus.approved.value, 0) / max(len(apps), 1), 4
			),
			"generated_at": _now().isoformat(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# Onboarding Workflow (5 methods)
	# ─────────────────────────────────────────────────────────────────────────

	async def start_digital_onboarding(
		self,
		mobile_number: str,
		channel: str,
		*,
		customer_type: str = "individual",
		preferred_language: str = "en",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Start a digital onboarding session.

		Channels: web, mobile, ussd, agent, branch, api.
		Returns: session_id, required_steps, expiry.

		USSD channel (ubiquitous in Africa) gets simplified step list
		to accommodate feature-phone UX constraints.
		"""
		assert mobile_number and mobile_number.strip(), "mobile_number is required"
		assert channel and channel.strip(), "channel is required"
		assert customer_type in [e.value for e in CustomerType], f"unsupported customer_type: {customer_type}"

		valid_channels = {"web", "mobile", "ussd", "agent", "branch", "api", "whatsapp", "sms"}
		if channel.lower() not in valid_channels:
			raise ValueError(f"unsupported channel: {channel}. Choose from: {', '.join(sorted(valid_channels))}")

		# USSD gets condensed steps (feature-phone constraints)
		if channel.lower() == "ussd":
			steps = ["identity_document", "pep_screening", "risk_assessment"]
		elif customer_type in ("business", "nonprofit", "trust"):
			steps = _BUSINESS_STEPS[:]
		else:
			steps = _INDIVIDUAL_STEPS[:]

		# Create a stub application — country and consent are collected during onboarding steps,
		# so we use placeholder values that satisfy NonEmptyStr validation here and get
		# replaced as the journey progresses (onboarding_step_complete updates the app record).
		app = KYCApplication(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			customer_id=mobile_number.strip(),
			customer_type=CustomerType(customer_type),
			country_code="XX",  # placeholder — updated when country step is completed
			legal_name=f"Onboarding {mobile_number[-4:]}",
			consent_reference=f"pending-{channel.lower()}",  # placeholder — replaced on consent step
			status=ApplicationStatus.draft,
			created_by=self.actor_id,
			metadata={
				**(metadata or {}),
				"channel": channel.lower(),
				"mobile_number": mobile_number.strip(),
				"preferred_language": preferred_language,
				"onboarding_stub": True,
			},
		)
		await self._store.put(_COL_APP, app.model_dump(mode="json"))

		# Create onboarding journey
		journey = OnboardingJourney(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			application_id=app.id,
			channel=channel.lower(),
			customer_type=CustomerType(customer_type),
			status=JourneyStatus.started,
			current_step=steps[0] if steps else "identity_document",
			steps_required=steps,
			steps_completed=[],
			created_by=self.actor_id,
		)
		journey_record = journey.model_dump(mode="json")
		await self._store.put(_COL_JOURNEY, journey_record)

		session_expiry = _now() + timedelta(hours=24)
		await self._emit(
			"kyc_onboarding_started",
			journey.id,
			"onboarding_journey",
			{"channel": channel, "mobile_number": mobile_number[-4:], "application_id": app.id},
		)
		return {
			"session_id": journey.id,
			"application_id": app.id,
			"channel": channel.lower(),
			"customer_type": customer_type,
			"required_steps": steps,
			"current_step": journey.current_step,
			"expiry": session_expiry.isoformat(),
			"expires_in_hours": 24,
			"started_at": journey_record["started_at"],
		}

	async def onboarding_step_complete(
		self,
		session_id: str,
		step: str,
		data: dict[str, Any],
	) -> dict[str, Any]:
		"""Mark an onboarding step as complete and advance to the next step.

		``step`` must be in the journey's ``steps_required`` list.
		``data`` contains step-specific payload (e.g. document tokens, consent).
		"""
		assert session_id and session_id.strip(), "session_id is required"
		assert step and step.strip(), "step is required"

		journey_record = await self._store.get(_COL_JOURNEY, session_id)
		if not journey_record:
			raise KeyError(f"onboarding session not found: {session_id}")
		assert_no_cross_tenant_access(self.tenant_id, journey_record["tenant_id"])

		if journey_record.get("status") in (JourneyStatus.completed.value, JourneyStatus.abandoned.value):
			raise ValueError(f"session {session_id} is {journey_record['status']}")

		steps_required: list[str] = journey_record.get("steps_required", [])
		steps_completed: list[str] = journey_record.get("steps_completed", [])

		if step not in steps_required:
			raise ValueError(f"step '{step}' is not in required steps: {steps_required}")
		if step in steps_completed:
			raise ValueError(f"step '{step}' already completed")

		steps_completed.append(step)
		remaining = [s for s in steps_required if s not in steps_completed]

		if not remaining:
			new_status = JourneyStatus.completed.value
			current_step = "done"
			completed_at = _now().isoformat()
			start_dt = datetime.fromisoformat(journey_record.get("started_at", _now().isoformat()))
			time_to_complete = int((_now() - start_dt).total_seconds())
		else:
			new_status = JourneyStatus.started.value
			current_step = remaining[0]
			completed_at = None
			time_to_complete = None

		journey_record.update({
			"steps_completed": steps_completed,
			"current_step": current_step,
			"status": new_status,
			"completed_at": completed_at,
			"time_to_complete_seconds": time_to_complete,
			"updated_at": _now().isoformat(),
			"metadata": {
				**journey_record.get("metadata", {}),
				f"step_{step}_data": {k: v for k, v in data.items() if k != "raw_bytes"},
				f"step_{step}_completed_at": _now().isoformat(),
			},
		})
		await self._store.put(_COL_JOURNEY, journey_record)

		# If journey complete, update the application status too
		if new_status == JourneyStatus.completed.value:
			app_record = await self._store.get(_COL_APP, journey_record["application_id"])
			if app_record:
				app_record["status"] = ApplicationStatus.pending_review.value
				app_record["updated_at"] = _now().isoformat()
				await self._store.put(_COL_APP, app_record)

		await self._emit(
			"kyc_onboarding_step_completed",
			session_id,
			"onboarding_journey",
			{"step": step, "steps_remaining": len(remaining), "journey_status": new_status},
		)
		return {
			"session_id": session_id,
			"step_completed": step,
			"steps_completed": steps_completed,
			"steps_remaining": remaining,
			"current_step": current_step,
			"journey_status": new_status,
			"application_id": journey_record["application_id"],
		}

	async def onboarding_status(self, session_id: str) -> dict[str, Any]:
		"""Retrieve current status of an onboarding session."""
		assert session_id and session_id.strip(), "session_id is required"

		journey_record = await self._store.get(_COL_JOURNEY, session_id)
		if not journey_record:
			raise KeyError(f"onboarding session not found: {session_id}")
		assert_no_cross_tenant_access(self.tenant_id, journey_record["tenant_id"])

		steps_required: list[str] = journey_record.get("steps_required", [])
		steps_completed: list[str] = journey_record.get("steps_completed", [])
		remaining = [s for s in steps_required if s not in steps_completed]
		progress_pct = round(len(steps_completed) / max(len(steps_required), 1) * 100, 1)

		return {
			"session_id": session_id,
			"application_id": journey_record.get("application_id"),
			"channel": journey_record.get("channel"),
			"status": journey_record.get("status"),
			"current_step": journey_record.get("current_step"),
			"steps_required": steps_required,
			"steps_completed": steps_completed,
			"steps_remaining": remaining,
			"progress_pct": progress_pct,
			"started_at": journey_record.get("started_at"),
			"completed_at": journey_record.get("completed_at"),
			"time_to_complete_seconds": journey_record.get("time_to_complete_seconds"),
		}

	async def abandon_onboarding(self, session_id: str, reason: str) -> dict[str, Any]:
		"""Mark an onboarding session as abandoned.

		Preserves partial data for analytics. Sets application status to expired.
		"""
		assert session_id and session_id.strip(), "session_id is required"
		assert reason and reason.strip(), "reason is required"

		journey_record = await self._store.get(_COL_JOURNEY, session_id)
		if not journey_record:
			raise KeyError(f"onboarding session not found: {session_id}")
		assert_no_cross_tenant_access(self.tenant_id, journey_record["tenant_id"])

		if journey_record.get("status") == JourneyStatus.completed.value:
			raise ValueError("cannot abandon a completed onboarding session")

		now = _now()
		journey_record["status"] = JourneyStatus.abandoned.value
		journey_record["abandoned_at"] = now.isoformat()
		journey_record["updated_at"] = now.isoformat()
		journey_record["metadata"] = {
			**journey_record.get("metadata", {}),
			"abandonment_reason": reason.strip(),
			"abandoned_by": self.actor_id,
			"steps_at_abandonment": journey_record.get("current_step"),
		}
		await self._store.put(_COL_JOURNEY, journey_record)

		# Expire the associated application
		app_record = await self._store.get(_COL_APP, journey_record.get("application_id", ""))
		if app_record:
			app_record["status"] = ApplicationStatus.expired.value
			app_record["updated_at"] = now.isoformat()
			await self._store.put(_COL_APP, app_record)

		await self._emit(
			"kyc_onboarding_abandoned",
			session_id,
			"onboarding_journey",
			{"reason": reason, "current_step": journey_record.get("current_step")},
		)
		return {
			"session_id": session_id,
			"status": JourneyStatus.abandoned.value,
			"reason": reason,
			"abandoned_at": now.isoformat(),
			"steps_completed": journey_record.get("steps_completed", []),
		}

	async def onboarding_analytics(self, period: str) -> dict[str, Any]:
		"""Return onboarding funnel analytics for a given period.

		Metrics: total_started, completed, abandoned, completion_rate,
		avg_time_to_complete_seconds, drop_off_by_step, by_channel.
		"""
		assert period and period.strip(), "period is required"

		# Parse period (ISO month YYYY-MM or range YYYY-MM-DD:YYYY-MM-DD)
		if ":" in period:
			parts = period.split(":")
			from_dt = date.fromisoformat(parts[0].strip()[:10])
			to_dt = date.fromisoformat(parts[1].strip()[:10])
		else:
			year, month = int(period[:4]), int(period[5:7])
			from_dt = date(year, month, 1)
			to_dt = (date(year, month % 12 + 1, 1) - timedelta(days=1)) if month < 12 else date(year, 12, 31)

		journeys = await self._store.query(_COL_JOURNEY, {"tenant_id": self.tenant_id}, limit=10_000)

		# Filter by period
		period_journeys = []
		for j in journeys:
			started = j.get("started_at", "")
			if started:
				j_date = date.fromisoformat(str(started)[:10])
				if from_dt <= j_date <= to_dt:
					period_journeys.append(j)

		total = len(period_journeys)
		completed = sum(1 for j in period_journeys if j.get("status") == JourneyStatus.completed.value)
		abandoned = sum(1 for j in period_journeys if j.get("status") == JourneyStatus.abandoned.value)

		completion_times = [
			j["time_to_complete_seconds"]
			for j in period_journeys
			if j.get("time_to_complete_seconds")
		]
		avg_time = sum(completion_times) / len(completion_times) if completion_times else 0

		# Drop-off by step: last step before abandonment
		drop_off: dict[str, int] = {}
		for j in period_journeys:
			if j.get("status") == JourneyStatus.abandoned.value:
				step = j.get("current_step", "unknown")
				drop_off[step] = drop_off.get(step, 0) + 1

		# By channel
		by_channel: dict[str, int] = {}
		for j in period_journeys:
			ch = j.get("channel", "unknown")
			by_channel[ch] = by_channel.get(ch, 0) + 1

		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"period_from": str(from_dt),
			"period_to": str(to_dt),
			"total_started": total,
			"completed": completed,
			"abandoned": abandoned,
			"in_progress": total - completed - abandoned,
			"completion_rate": round(completed / max(total, 1), 4),
			"abandonment_rate": round(abandoned / max(total, 1), 4),
			"avg_time_to_complete_seconds": round(avg_time, 1),
			"drop_off_by_step": dict(sorted(drop_off.items(), key=lambda x: -x[1])),
			"by_channel": dict(sorted(by_channel.items(), key=lambda x: -x[1])),
			"generated_at": _now().isoformat(),
		}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy dataclasses — used only by KnowYourCustomerService (sync API).
# These were originally inline in the old service.py; kept here so that
# importlib-based test loading (no package context) still works.
# ─────────────────────────────────────────────────────────────────────────────

class KycProfile:
	__slots__ = ("id", "tenant_id", "subject_reference", "legal_name", "customer_type", "country_code", "consent_reference", "status", "metadata")

	def __init__(self, id: str, tenant_id: str, subject_reference: str, legal_name: str, customer_type: str, country_code: str, consent_reference: str, status: str = "open", metadata: dict[str, Any] | None = None) -> None:
		self.id = id
		self.tenant_id = tenant_id
		self.subject_reference = subject_reference
		self.legal_name = legal_name
		self.customer_type = customer_type
		self.country_code = country_code
		self.consent_reference = consent_reference
		self.status = status
		self.metadata = metadata or {}

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "subject_reference": self.subject_reference, "legal_name": self.legal_name, "customer_type": self.customer_type, "country_code": self.country_code, "consent_reference": self.consent_reference, "status": self.status, "metadata": self.metadata}


class KycDocument:
	__slots__ = ("id", "tenant_id", "profile_id", "document_type", "token_reference", "extracted_subject", "confidence", "status")

	def __init__(self, id: str, tenant_id: str, profile_id: str, document_type: str, token_reference: str, extracted_subject: str, confidence: float, status: str = "verified") -> None:
		self.id = id
		self.tenant_id = tenant_id
		self.profile_id = profile_id
		self.document_type = document_type
		self.token_reference = token_reference
		self.extracted_subject = extracted_subject
		self.confidence = confidence
		self.status = status

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "profile_id": self.profile_id, "document_type": self.document_type, "token_reference": self.token_reference, "extracted_subject": self.extracted_subject, "confidence": self.confidence, "status": self.status}


class KycScreening:
	__slots__ = ("id", "tenant_id", "profile_id", "sanctions_hit", "pep_hit", "watchlist_hit", "adverse_media_hit", "review_id", "status")

	def __init__(self, id: str, tenant_id: str, profile_id: str, sanctions_hit: bool = False, pep_hit: bool = False, watchlist_hit: bool = False, adverse_media_hit: bool = False, review_id: str = "", status: str = "clear") -> None:
		self.id = id
		self.tenant_id = tenant_id
		self.profile_id = profile_id
		self.sanctions_hit = sanctions_hit
		self.pep_hit = pep_hit
		self.watchlist_hit = watchlist_hit
		self.adverse_media_hit = adverse_media_hit
		self.review_id = review_id
		self.status = status

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "profile_id": self.profile_id, "sanctions_hit": self.sanctions_hit, "pep_hit": self.pep_hit, "watchlist_hit": self.watchlist_hit, "adverse_media_hit": self.adverse_media_hit, "review_id": self.review_id, "status": self.status}


class KycDecision:
	__slots__ = ("id", "tenant_id", "profile_id", "decision", "risk_score", "review_id", "status")

	def __init__(self, id: str, tenant_id: str, profile_id: str, decision: str, risk_score: int, review_id: str = "", status: str = "recorded") -> None:
		self.id = id
		self.tenant_id = tenant_id
		self.profile_id = profile_id
		self.decision = decision
		self.risk_score = risk_score
		self.review_id = review_id
		self.status = status

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "profile_id": self.profile_id, "decision": self.decision, "risk_score": self.risk_score, "review_id": self.review_id, "status": self.status}


class KycEvidence:
	__slots__ = ("id", "tenant_id", "kind", "reference_id", "status", "metadata")

	def __init__(self, id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> None:
		self.id = id
		self.tenant_id = tenant_id
		self.kind = kind
		self.reference_id = reference_id
		self.status = status
		self.metadata = metadata

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": self.metadata}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy synchronous service — preserved for test_package_contract compatibility.
# The test suite loads service.py directly via importlib and calls this class
# with no arguments using the original sync API.  Do not remove.
# ─────────────────────────────────────────────────────────────────────────────

class KnowYourCustomerService:
	"""Dependency-light KYC lifecycle runtime for generated applications (sync API)."""

	def __init__(self) -> None:
		self.profiles: dict[str, KycProfile] = {}
		self.documents: dict[str, KycDocument] = {}
		self.screenings: dict[str, KycScreening] = {}
		self.decisions: dict[str, KycDecision] = {}
		self.evidence: dict[str, KycEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def open_profile(self, profile_id: str, tenant_id: str, subject_reference: str, legal_name: str, customer_type: str, country_code: str, consent_reference: str, metadata: dict[str, Any] | None = None, policy_attached: bool = True) -> dict[str, Any]:
		customer_type = normalize_code(customer_type)
		country_code = normalize_country(country_code)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "open_profile", "subject_present": bool(subject_reference), "legal_name_present": bool(legal_name), "customer_type_supported": customer_type in SUPPORTED_CUSTOMER_TYPES, "country_present": bool(country_code), "consent_recorded": bool(consent_reference)})
		if profile_id in self.profiles:
			raise ValueError(f"profile already exists: {profile_id}")
		profile = KycProfile(profile_id, tenant_id, subject_reference, legal_name, customer_type, country_code, consent_reference, metadata=dict(metadata or {}))
		self.profiles[profile_id] = profile
		self._audit(tenant_id, "kyc_profile_opened", profile_id)
		return profile.to_dict()

	def register_document(self, document_id: str, tenant_id: str, profile_id: str, document_type: str, token_reference: str, extracted_subject: str, confidence: float | int | str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		document_type = normalize_code(document_type)
		confidence_value = normalize_confidence(confidence)
		minimum = float(get_capability_contract(tenant_id)["configuration"]["documents"]["minimum_confidence"])
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_document", "profile_present": profile is not None, "document_type_supported": document_type in SUPPORTED_DOCUMENT_TYPES, "token_reference_present": bool(token_reference), "extracted_subject_present": bool(extracted_subject), "confidence_below_minimum": confidence_value < minimum})
		document = KycDocument(document_id, tenant_id, profile_id, document_type, token_reference, extracted_subject, confidence_value)
		self.documents[document_id] = document
		self._audit(tenant_id, "kyc_document_registered", document_id)
		return document.to_dict()

	def record_screening(self, screening_id: str, tenant_id: str, profile_id: str, sanctions_hit: bool = False, pep_hit: bool = False, watchlist_hit: bool = False, adverse_media_hit: bool = False, review_id: str = "") -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		screening_hit = any([sanctions_hit, pep_hit, watchlist_hit, adverse_media_hit])
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_screening", "profile_present": profile is not None, "screening_hit": screening_hit, "review_recorded": bool(review_id)})
		status = "reviewed" if screening_hit else "clear"
		screening = KycScreening(screening_id, tenant_id, profile_id, sanctions_hit, pep_hit, watchlist_hit, adverse_media_hit, review_id, status)
		self.screenings[screening_id] = screening
		self._audit(tenant_id, "kyc_screening_recorded", screening_id)
		return screening.to_dict()

	def score_risk(self, decision_id: str, tenant_id: str, profile_id: str, risk_score: int | str, review_id: str = "") -> dict[str, Any]:
		self._tenant_profile(profile_id, tenant_id)
		score = normalize_risk_score(risk_score)
		limits = get_capability_contract(tenant_id)["configuration"]["risk"]
		band = _risk_band_fn(score, int(limits["high_risk_threshold"]), int(limits["medium_risk_threshold"])) if 0 <= score <= 100 else "invalid"
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "score_risk", "risk_score_out_of_range": not 0 <= score <= 100, "high_risk": band == "high", "review_recorded": bool(review_id)})
		decision = KycDecision(decision_id, tenant_id, profile_id, band, score, review_id)
		self.decisions[decision_id] = decision
		self._audit(tenant_id, "kyc_risk_scored", decision_id)
		return decision.to_dict()

	def record_decision(self, decision_id: str, tenant_id: str, profile_id: str, decision: str, risk_score: int | str, review_id: str = "") -> dict[str, Any]:
		self._tenant_profile(profile_id, tenant_id)
		score = normalize_risk_score(risk_score)
		identity_document_present = self._has_document(tenant_id, profile_id, {"passport", "national_id", "driver_license", "resident_permit"})
		address_document_present = self._has_document(tenant_id, profile_id, {"utility_bill", "bank_statement", "business_registration"})
		screening_present = any(item.tenant_id == tenant_id and item.profile_id == profile_id for item in self.screenings.values())
		risk_present = any(item.tenant_id == tenant_id and item.profile_id == profile_id for item in self.decisions.values())
		open_review_flags = any(item.profile_id == profile_id and item.status != "clear" and not item.review_id for item in self.screenings.values())
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_decision", "identity_document_present": identity_document_present, "address_document_present": address_document_present, "screening_present": screening_present, "risk_present": risk_present, "open_review_flags": open_review_flags})
		record = KycDecision(decision_id, tenant_id, profile_id, normalize_code(decision), score, review_id, "verified" if normalize_code(decision) == "approve" else "recorded")
		self.decisions[decision_id] = record
		self.profiles[profile_id].status = "verified" if record.decision == "approve" else record.decision
		self._audit(tenant_id, "kyc_decision_recorded", decision_id)
		return record.to_dict()

	def register_kyc_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str = "") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_kyc_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "kyc_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "kyc_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.kyc.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		profiles = [profile for profile in self.profiles.values() if profile.tenant_id == tenant_id]
		return {"tenant_id": tenant_id, "profile_count": len(profiles), "document_count": sum(1 for item in self.documents.values() if item.tenant_id == tenant_id), "screening_count": sum(1 for item in self.screenings.values() if item.tenant_id == tenant_id), "decision_count": sum(1 for item in self.decisions.values() if item.tenant_id == tenant_id), "verified_count": sum(1 for item in profiles if item.status == "verified"), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_profiles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		profiles = self.profiles.values()
		if tenant_id is not None:
			profiles = [profile for profile in profiles if profile.tenant_id == tenant_id]
		return [profile.to_dict() for profile in sorted(profiles, key=lambda item: item.id)]

	def _tenant_profile_or_none(self, profile_id: str, tenant_id: str) -> KycProfile | None:
		profile = self.profiles.get(profile_id)
		if profile is None or profile.tenant_id != tenant_id:
			return None
		return profile

	def _tenant_profile(self, profile_id: str, tenant_id: str) -> KycProfile:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		if profile is None:
			raise KeyError(f"unknown KYC profile: {profile_id}")
		return profile

	def _has_document(self, tenant_id: str, profile_id: str, document_types: set[str]) -> bool:
		return any(item.tenant_id == tenant_id and item.profile_id == profile_id and item.document_type in document_types for item in self.documents.values())

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = KycEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "kyc_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "kyc_policy_denied")


# FintechKycService is the modern async service

	async def ml_kyc_risk_score(self, *args, **kwargs):
		"""AI-powered KYC customer risk scoring using behavioral and profile signals. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="kyc_customer_risk_scoring")
			return {"kyc_risk": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

FintechKycService = KYCService
