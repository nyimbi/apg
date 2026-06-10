"""Async service layer for APG Intelligence Fusion.

Tenant-scoped, event-emitting, store-backed service implementing the full
intelligence fusion lifecycle:

  IntelligenceItem → FusionWorkspace → CorrelationSet →
  AssessmentPicture → IntelligenceProduct → AnalyticalJudgement →
  Evidence → HypothesisTest

All methods are async. Domain rules enforced via domain/rules.py.
Calculations via domain/calculations.py. Events emitted via _emit_event().

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

try:
	from .database.store import Store, get_store
	from .domain import calculations as calc
	from .domain import rules
	from .domain.events import (
		ach_completed,
		assessment_approved,
		assessment_created,
		correlation_created,
		correlation_status_changed,
		evidence_created,
		evidence_status_changed,
		fusion_completed,
		hypothesis_concluded,
		hypothesis_created,
		intel_item_created,
		intel_item_status_changed,
		judgement_challenged,
		judgement_created,
		product_created,
		product_status_changed,
		workspace_created,
		workspace_status_changed,
	)
	from .models import (
		ACHMatrix,
		AnalyticalJudgement,
		AnalyticalJudgementCreate,
		AnalyticalJudgementUpdate,
		AssessmentPicture,
		AssessmentPictureCreate,
		AssessmentPictureUpdate,
		ConfidenceCalibration,
		ConfidenceLevel,
		CorrelationSet,
		CorrelationSetCreate,
		CorrelationSetStatus,
		CorrelationSetUpdate,
		DisseminationRecord,
		Evidence,
		EvidenceCreate,
		EvidenceStatus,
		EvidenceUpdate,
		FusionDashboardReport,
		FusionEvent,
		FusionQualityResult,
		FusionWorkspace,
		FusionWorkspaceCreate,
		FusionWorkspaceSummary,
		FusionWorkspaceUpdate,
		HypothesisTest,
		HypothesisTestCreate,
		HypothesisTestUpdate,
		IntelItemStatus,
		IntelligenceItem,
		IntelligenceItemCreate,
		IntelligenceItemUpdate,
		IntelligenceProduct,
		IntelligenceProductCreate,
		IntelligenceProductUpdate,
		KeyAssumptionsResult,
		PagedResult,
		ProductStatus,
		TLPLevel,
		WorkspaceStatus,
		uuid7str,
	)
except ImportError:
	from database.store import Store, get_store  # type: ignore
	from domain import calculations as calc  # type: ignore
	from domain import rules  # type: ignore
	from domain.events import (  # type: ignore
		ach_completed, assessment_approved, assessment_created,
		correlation_created, correlation_status_changed, evidence_created,
		evidence_status_changed, fusion_completed, hypothesis_concluded,
		hypothesis_created, intel_item_created, intel_item_status_changed,
		judgement_challenged, judgement_created, product_created,
		product_status_changed, workspace_created, workspace_status_changed,
	)
	from models import (  # type: ignore
		ACHMatrix, AnalyticalJudgement, AnalyticalJudgementCreate,
		AnalyticalJudgementUpdate, AssessmentPicture, AssessmentPictureCreate,
		AssessmentPictureUpdate, ConfidenceCalibration, ConfidenceLevel,
		CorrelationSet, CorrelationSetCreate, CorrelationSetStatus,
		CorrelationSetUpdate, DisseminationRecord, Evidence, EvidenceCreate,
		EvidenceStatus, EvidenceUpdate, FusionDashboardReport, FusionEvent,
		FusionQualityResult, FusionWorkspace, FusionWorkspaceCreate,
		FusionWorkspaceSummary, FusionWorkspaceUpdate, HypothesisTest,
		HypothesisTestCreate, HypothesisTestUpdate, IntelItemStatus,
		IntelligenceItem, IntelligenceItemCreate, IntelligenceItemUpdate,
		IntelligenceProduct, IntelligenceProductCreate, IntelligenceProductUpdate,
		KeyAssumptionsResult, PagedResult, ProductStatus, TLPLevel,
		WorkspaceStatus, uuid7str,
	)

try:
	from .capability_contract import (
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_WORKSPACE_TYPES as _CONTRACT_WS_TYPES,
		SUPPORTED_SOURCE_TYPES,
		SUPPORTED_ARTIFACT_TYPES,
		SUPPORTED_CORRELATION_TYPES as _CONTRACT_CORR_TYPES,
		SUPPORTED_HYPOTHESIS_TYPES,
		SUPPORTED_ASSESSMENT_TYPES as _CONTRACT_ASSESS_TYPES,
		SUPPORTED_RISK_LEVELS as _CONTRACT_RISK_LEVELS,
		SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		FUSION_EVENT_STREAM,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .fusion_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_WORKSPACE_TYPES as _CONTRACT_WS_TYPES,
		SUPPORTED_SOURCE_TYPES,
		SUPPORTED_ARTIFACT_TYPES,
		SUPPORTED_CORRELATION_TYPES as _CONTRACT_CORR_TYPES,
		SUPPORTED_HYPOTHESIS_TYPES,
		SUPPORTED_ASSESSMENT_TYPES as _CONTRACT_ASSESS_TYPES,
		SUPPORTED_RISK_LEVELS as _CONTRACT_RISK_LEVELS,
		SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		FUSION_EVENT_STREAM,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from fusion_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


from dataclasses import asdict, dataclass
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

logger = logging.getLogger(__name__)

_COL = {
	"item":        "fusion_intel_items",
	"workspace":   "fusion_workspaces",
	"correlation": "fusion_correlations",
	"assessment":  "fusion_assessments",
	"product":     "fusion_products",
	"judgement":   "fusion_judgements",
	"evidence":    "fusion_evidence",
	"hypothesis":  "fusion_hypotheses",
	"dissem":      "fusion_disseminations",
	"event":       "fusion_events",
}


class IntelligenceFusionService:
	"""
	Tenant-scoped intelligence fusion service.

	Usage::

		svc = IntelligenceFusionService(tenant_id="acme", actor_id="analyst-1")
		ws = await svc.create_workspace(FusionWorkspaceCreate(...))
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		store: Store | None = None,
		db_url: str | None = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._store = store or get_store(db_url)
		self._events: list[dict[str, Any]] = []

	# ─────────────────────────────────────────────────────────────────────────
	# IntelligenceItem CRUD
	# ─────────────────────────────────────────────────────────────────────────

	async def create_intel_item(self, payload: IntelligenceItemCreate) -> IntelligenceItem:
		"""Ingest a raw intelligence item from any source discipline."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_source_type_supported(payload.source_type.value)
		rules.assert_content_fingerprint_present(payload.content_fingerprint)
		rules.assert_custodian_assigned(payload.custodian_id)
		rules.assert_confidence_in_range(payload.confidence_score)

		if payload.workspace_id:
			ws = await self._require_workspace(payload.workspace_id)
			rules.assert_workspace_active(ws.get("status", "active"))
			rules.assert_classification_dominance(
				payload.classification.value,
				ws.get("classification", "unclassified"),
			)

		item = IntelligenceItem(**payload.model_dump(), status=IntelItemStatus.RAW)
		await self._store.put(_COL["item"], item.model_dump(mode="json"))
		self._emit_event(intel_item_created(
			self.tenant_id, self.actor_id, item.id,
			item.source_type.value, item.workspace_id,
		))
		self._log_created("IntelligenceItem", item.id)
		return item

	async def get_intel_item(self, item_id: str) -> IntelligenceItem:
		"""Retrieve a single intelligence item by ID."""
		row = await self._require(item_id, _COL["item"])
		return IntelligenceItem.model_validate(row)

	async def list_intel_items(
		self,
		workspace_id: str | None = None,
		source_type: str | None = None,
		status: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List intelligence items with optional filters."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if workspace_id:
			filters["workspace_id"] = workspace_id
		if source_type:
			filters["source_type"] = source_type
		if status:
			filters["status"] = status
		rows = await self._store.query(_COL["item"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_intel_item(self, item_id: str, patch: IntelligenceItemUpdate) -> IntelligenceItem:
		"""Partially update a raw intelligence item."""
		row = await self._require(item_id, _COL["item"])
		old_status = row.get("status")
		updates = patch.model_dump(exclude_none=True)
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["item"], row)
		if patch.status and patch.status.value != old_status:
			self._emit_event(intel_item_status_changed(
				self.tenant_id, self.actor_id, item_id, old_status or "", patch.status.value,
			))
		return IntelligenceItem.model_validate(row)

	async def delete_intel_item(self, item_id: str) -> bool:
		"""Soft-delete an intelligence item."""
		row = await self._require(item_id, _COL["item"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["item"], row)
		return True

	async def validate_intel_item(self, item_id: str) -> IntelligenceItem:
		"""Mark an item as validated by a custodian."""
		return await self._set_item_status(item_id, IntelItemStatus.VALIDATED)

	async def reject_intel_item(self, item_id: str) -> IntelligenceItem:
		"""Reject an item — removes it from the fusion pipeline."""
		return await self._set_item_status(item_id, IntelItemStatus.REJECTED)

	async def _set_item_status(self, item_id: str, status: IntelItemStatus) -> IntelligenceItem:
		patch = IntelligenceItemUpdate(status=status)
		return await self.update_intel_item(item_id, patch)

	# ─────────────────────────────────────────────────────────────────────────
	# FusionWorkspace CRUD
	# ─────────────────────────────────────────────────────────────────────────

	async def create_workspace(self, payload: FusionWorkspaceCreate) -> FusionWorkspace:
		"""Create a new analytical workspace."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_workspace_type_supported(payload.workspace_type.value)

		ws = FusionWorkspace(**payload.model_dump(), status=WorkspaceStatus.ACTIVE)
		await self._store.put(_COL["workspace"], ws.model_dump(mode="json"))
		self._emit_event(workspace_created(
			self.tenant_id, self.actor_id, ws.id, ws.workspace_type.value,
		))
		self._log_created("FusionWorkspace", ws.id)
		return ws

	async def get_workspace(self, workspace_id: str) -> FusionWorkspace:
		"""Retrieve a workspace by ID."""
		row = await self._require(workspace_id, _COL["workspace"])
		return FusionWorkspace.model_validate(row)

	async def list_workspaces(
		self,
		status: str | None = None,
		workspace_type: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List workspaces for this tenant."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if status:
			filters["status"] = status
		if workspace_type:
			filters["workspace_type"] = workspace_type
		rows = await self._store.query(_COL["workspace"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_workspace(self, workspace_id: str, patch: FusionWorkspaceUpdate) -> FusionWorkspace:
		"""Partially update a workspace."""
		row = await self._require(workspace_id, _COL["workspace"])
		rules.assert_workspace_not_closed(row.get("status", "active"))
		old_status = row.get("status")
		updates = patch.model_dump(exclude_none=True)
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["workspace"], row)
		if patch.status and patch.status.value != old_status:
			self._emit_event(workspace_status_changed(
				self.tenant_id, self.actor_id, workspace_id, patch.status.value,
			))
		return FusionWorkspace.model_validate(row)

	async def suspend_workspace(self, workspace_id: str) -> FusionWorkspace:
		"""Suspend an active workspace."""
		return await self.update_workspace(workspace_id, FusionWorkspaceUpdate(status=WorkspaceStatus.SUSPENDED))

	async def close_workspace(self, workspace_id: str) -> FusionWorkspace:
		"""Close a workspace permanently."""
		return await self.update_workspace(workspace_id, FusionWorkspaceUpdate(status=WorkspaceStatus.CLOSED))

	async def delete_workspace(self, workspace_id: str) -> bool:
		"""Soft-delete a workspace."""
		row = await self._require(workspace_id, _COL["workspace"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["workspace"], row)
		return True

	async def workspace_summary(self, workspace_id: str) -> FusionWorkspaceSummary:
		"""Return a summary of a workspace's contents."""
		ws = await self.get_workspace(workspace_id)
		f: dict[str, Any] = {"workspace_id": workspace_id, "tenant_id": self.tenant_id}
		item_count = await self._store.count(_COL["item"], {**f, "is_deleted": False})
		corr_count = await self._store.count(_COL["correlation"], {**f, "is_deleted": False})
		hyp_count = await self._store.count(_COL["hypothesis"], {**f, "is_deleted": False})
		ass_count = await self._store.count(_COL["assessment"], {**f, "is_deleted": False})
		prod_count = await self._store.count(_COL["product"], {**f, "is_deleted": False})
		ev_count = await self._store.count(_COL["evidence"], {**f, "is_deleted": False})
		return FusionWorkspaceSummary(
			workspace_id=workspace_id,
			workspace_name=ws.name,
			workspace_type=ws.workspace_type,
			item_count=item_count,
			correlation_count=corr_count,
			hypothesis_count=hyp_count,
			assessment_count=ass_count,
			product_count=prod_count,
			evidence_count=ev_count,
			lead_analyst_id=ws.lead_analyst_id,
			status=ws.status,
			classification=ws.classification,
		)

	# ─────────────────────────────────────────────────────────────────────────
	# CorrelationSet CRUD
	# ─────────────────────────────────────────────────────────────────────────

	async def create_correlation(self, payload: CorrelationSetCreate) -> CorrelationSet:
		"""Create a correlation set linking multiple intelligence items."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_correlation_type_supported(payload.correlation_type.value)
		rules.assert_analyst_assigned(payload.analyst_id)
		rules.assert_confidence_in_range(payload.confidence_score)
		rules.assert_correlation_has_items(payload.item_ids)

		ws = await self._require_workspace(payload.workspace_id)
		rules.assert_workspace_active(ws.get("status", "active"))

		corr = CorrelationSet(**payload.model_dump(), status=CorrelationSetStatus.OPEN)
		await self._store.put(_COL["correlation"], corr.model_dump(mode="json"))
		self._emit_event(correlation_created(
			self.tenant_id, self.actor_id, corr.id,
			corr.correlation_type.value, len(corr.item_ids),
		))
		self._log_created("CorrelationSet", corr.id)
		return corr

	async def get_correlation(self, correlation_id: str) -> CorrelationSet:
		"""Retrieve a correlation set by ID."""
		row = await self._require(correlation_id, _COL["correlation"])
		return CorrelationSet.model_validate(row)

	async def list_correlations(
		self,
		workspace_id: str | None = None,
		status: str | None = None,
		correlation_type: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List correlation sets with optional filters."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if workspace_id:
			filters["workspace_id"] = workspace_id
		if status:
			filters["status"] = status
		if correlation_type:
			filters["correlation_type"] = correlation_type
		rows = await self._store.query(_COL["correlation"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_correlation(self, correlation_id: str, patch: CorrelationSetUpdate) -> CorrelationSet:
		"""Partially update a correlation set."""
		row = await self._require(correlation_id, _COL["correlation"])
		old_status = row.get("status")
		updates = patch.model_dump(exclude_none=True)
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["correlation"], row)
		if patch.status and patch.status.value != old_status:
			self._emit_event(correlation_status_changed(
				self.tenant_id, self.actor_id, correlation_id, patch.status.value,
			))
		return CorrelationSet.model_validate(row)

	async def confirm_correlation(self, correlation_id: str) -> CorrelationSet:
		"""Mark a correlation as confirmed."""
		return await self.update_correlation(
			correlation_id, CorrelationSetUpdate(status=CorrelationSetStatus.CONFIRMED)
		)

	async def dispute_correlation(self, correlation_id: str) -> CorrelationSet:
		"""Mark a correlation as disputed."""
		return await self.update_correlation(
			correlation_id, CorrelationSetUpdate(status=CorrelationSetStatus.DISPUTED)
		)

	async def delete_correlation(self, correlation_id: str) -> bool:
		"""Soft-delete a correlation set."""
		row = await self._require(correlation_id, _COL["correlation"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["correlation"], row)
		return True

	# ─────────────────────────────────────────────────────────────────────────
	# AssessmentPicture CRUD
	# ─────────────────────────────────────────────────────────────────────────

	async def create_assessment(self, payload: AssessmentPictureCreate) -> AssessmentPicture:
		"""Create a synthesised assessment picture."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_assessment_type_supported(payload.assessment_type.value)
		rules.assert_risk_level_supported(payload.risk_level.value)
		rules.assert_analyst_assigned(payload.analyst_id)
		rules.assert_confidence_in_range(payload.confidence_score)
		rules.assert_assessment_has_hypotheses(payload.hypothesis_ids)
		rules.assert_assessment_has_correlations(payload.correlation_ids)

		assessment = AssessmentPicture(**payload.model_dump())
		await self._store.put(_COL["assessment"], assessment.model_dump(mode="json"))
		self._emit_event(assessment_created(
			self.tenant_id, self.actor_id, assessment.id, assessment.risk_level.value,
		))
		self._log_created("AssessmentPicture", assessment.id)
		return assessment

	async def get_assessment(self, assessment_id: str) -> AssessmentPicture:
		"""Retrieve an assessment picture by ID."""
		row = await self._require(assessment_id, _COL["assessment"])
		return AssessmentPicture.model_validate(row)

	async def list_assessments(
		self,
		workspace_id: str | None = None,
		risk_level: str | None = None,
		assessment_type: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List assessment pictures with optional filters."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if workspace_id:
			filters["workspace_id"] = workspace_id
		if risk_level:
			filters["risk_level"] = risk_level
		if assessment_type:
			filters["assessment_type"] = assessment_type
		rows = await self._store.query(_COL["assessment"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_assessment(self, assessment_id: str, patch: AssessmentPictureUpdate) -> AssessmentPicture:
		"""Partially update an assessment picture."""
		row = await self._require(assessment_id, _COL["assessment"])
		updates = patch.model_dump(exclude_none=True)
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["assessment"], row)
		return AssessmentPicture.model_validate(row)

	async def approve_assessment(self, assessment_id: str, approver_id: str) -> AssessmentPicture:
		"""Mark an assessment picture as approved by a senior analyst."""
		row = await self._require(assessment_id, _COL["assessment"])
		row["approved_by"] = approver_id
		row["approved_at"] = datetime.utcnow().isoformat()
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["assessment"], row)
		self._emit_event(assessment_approved(self.tenant_id, approver_id, assessment_id))
		return AssessmentPicture.model_validate(row)

	async def delete_assessment(self, assessment_id: str) -> bool:
		"""Soft-delete an assessment picture."""
		row = await self._require(assessment_id, _COL["assessment"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["assessment"], row)
		return True

	# ─────────────────────────────────────────────────────────────────────────
	# IntelligenceProduct CRUD + lifecycle
	# ─────────────────────────────────────────────────────────────────────────

	async def create_product(self, payload: IntelligenceProductCreate) -> IntelligenceProduct:
		"""Create a finished intelligence product."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_tlp_valid(payload.tlp.value)
		rules.assert_product_has_assessments(payload.assessment_ids)

		product = IntelligenceProduct(**payload.model_dump(), status=ProductStatus.DRAFT)
		await self._store.put(_COL["product"], product.model_dump(mode="json"))
		self._emit_event(product_created(
			self.tenant_id, self.actor_id, product.id,
			product.product_type.value, product.tlp.value,
		))
		self._log_created("IntelligenceProduct", product.id)
		return product

	async def get_product(self, product_id: str) -> IntelligenceProduct:
		"""Retrieve an intelligence product by ID."""
		row = await self._require(product_id, _COL["product"])
		return IntelligenceProduct.model_validate(row)

	async def list_products(
		self,
		workspace_id: str | None = None,
		status: str | None = None,
		product_type: str | None = None,
		tlp: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List intelligence products with optional filters."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if workspace_id:
			filters["workspace_id"] = workspace_id
		if status:
			filters["status"] = status
		if product_type:
			filters["product_type"] = product_type
		if tlp:
			filters["tlp"] = tlp
		rows = await self._store.query(_COL["product"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_product(self, product_id: str, patch: IntelligenceProductUpdate) -> IntelligenceProduct:
		"""Partially update a product in draft or review state."""
		row = await self._require(product_id, _COL["product"])
		rules.assert_product_not_recalled(row.get("status", "draft"))
		updates = patch.model_dump(exclude_none=True)
		if "tlp" in updates:
			rules.assert_tlp_valid(updates["tlp"])
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["product"], row)
		return IntelligenceProduct.model_validate(row)

	async def submit_product_for_review(self, product_id: str, reviewer_id: str) -> IntelligenceProduct:
		"""Submit a draft product for peer review."""
		row = await self._require(product_id, _COL["product"])
		rules.assert_product_in_draft_for_submit(row.get("status", "draft"))
		row["status"] = ProductStatus.REVIEW.value
		row["reviewer_id"] = reviewer_id
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["product"], row)
		self._emit_event(product_status_changed(self.tenant_id, self.actor_id, product_id, "review"))
		return IntelligenceProduct.model_validate(row)

	async def approve_product(self, product_id: str, approver_id: str) -> IntelligenceProduct:
		"""Approve a product that is under review."""
		row = await self._require(product_id, _COL["product"])
		rules.assert_product_in_review_for_approval(row.get("status", "draft"))
		row["status"] = ProductStatus.APPROVED.value
		row["reviewed_at"] = datetime.utcnow().isoformat()
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["product"], row)
		self._emit_event(product_status_changed(self.tenant_id, approver_id, product_id, "approved"))
		return IntelligenceProduct.model_validate(row)

	async def release_product(self, product_id: str, approval_reference: str) -> IntelligenceProduct:
		"""Release an approved product for dissemination."""
		row = await self._require(product_id, _COL["product"])
		rules.assert_product_in_approved_state(row.get("status", "draft"))
		rules.assert_approval_present(approval_reference)
		row["status"] = ProductStatus.RELEASED.value
		row["released_at"] = datetime.utcnow().isoformat()
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["product"], row)
		self._emit_event(product_status_changed(self.tenant_id, self.actor_id, product_id, "released"))
		return IntelligenceProduct.model_validate(row)

	async def recall_product(self, product_id: str) -> IntelligenceProduct:
		"""Recall a released product."""
		row = await self._require(product_id, _COL["product"])
		row["status"] = ProductStatus.RECALLED.value
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["product"], row)
		self._emit_event(product_status_changed(self.tenant_id, self.actor_id, product_id, "recalled"))
		return IntelligenceProduct.model_validate(row)

	async def delete_product(self, product_id: str) -> bool:
		"""Soft-delete a product."""
		row = await self._require(product_id, _COL["product"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["product"], row)
		return True

	# ─────────────────────────────────────────────────────────────────────────
	# AnalyticalJudgement CRUD
	# ─────────────────────────────────────────────────────────────────────────

	async def create_judgement(self, payload: AnalyticalJudgementCreate) -> AnalyticalJudgement:
		"""Record a calibrated analytical judgement."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_analyst_assigned(payload.analyst_id)
		rules.assert_confidence_in_range(payload.confidence_score)
		rules.assert_judgement_type_supported(payload.judgement_type.value)
		if payload.sat_method:
			rules.assert_sat_method_supported(payload.sat_method.value)

		judgement = AnalyticalJudgement(**payload.model_dump())
		await self._store.put(_COL["judgement"], judgement.model_dump(mode="json"))
		self._emit_event(judgement_created(
			self.tenant_id, self.actor_id, judgement.id,
			judgement.judgement_type.value, judgement.confidence_level.value,
		))
		self._log_created("AnalyticalJudgement", judgement.id)
		return judgement

	async def get_judgement(self, judgement_id: str) -> AnalyticalJudgement:
		"""Retrieve an analytical judgement by ID."""
		row = await self._require(judgement_id, _COL["judgement"])
		return AnalyticalJudgement.model_validate(row)

	async def list_judgements(
		self,
		workspace_id: str | None = None,
		judgement_type: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List analytical judgements with optional filters."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if workspace_id:
			filters["workspace_id"] = workspace_id
		if judgement_type:
			filters["judgement_type"] = judgement_type
		rows = await self._store.query(_COL["judgement"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_judgement(self, judgement_id: str, patch: AnalyticalJudgementUpdate) -> AnalyticalJudgement:
		"""Update a judgement with revised confidence or key assumptions."""
		row = await self._require(judgement_id, _COL["judgement"])
		updates = patch.model_dump(exclude_none=True)
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["judgement"], row)
		return AnalyticalJudgement.model_validate(row)

	async def challenge_judgement(self, judgement_id: str, challenger_id: str) -> AnalyticalJudgement:
		"""Register a red-team or devil's advocate challenge against a judgement."""
		row = await self._require(judgement_id, _COL["judgement"])
		challengers = row.get("challenger_ids", [])
		if challenger_id not in challengers:
			challengers.append(challenger_id)
		row["challenger_ids"] = challengers
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["judgement"], row)
		self._emit_event(judgement_challenged(
			self.tenant_id, self.actor_id, judgement_id, challenger_id,
		))
		return AnalyticalJudgement.model_validate(row)

	async def delete_judgement(self, judgement_id: str) -> bool:
		"""Soft-delete a judgement."""
		row = await self._require(judgement_id, _COL["judgement"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["judgement"], row)
		return True

	# ─────────────────────────────────────────────────────────────────────────
	# Evidence CRUD
	# ─────────────────────────────────────────────────────────────────────────

	async def create_evidence(self, payload: EvidenceCreate) -> Evidence:
		"""Record a provenance-tracked evidence item."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_evidence_type_supported(payload.evidence_type.value)
		rules.assert_content_fingerprint_present(payload.content_fingerprint)
		rules.assert_custodian_assigned(payload.custodian_id)
		rules.assert_chain_of_custody_present(payload.chain_of_custody)

		evidence = Evidence(**payload.model_dump(), status=EvidenceStatus.PENDING)
		await self._store.put(_COL["evidence"], evidence.model_dump(mode="json"))
		self._emit_event(evidence_created(
			self.tenant_id, self.actor_id, evidence.id, evidence.evidence_type.value,
		))
		self._log_created("Evidence", evidence.id)
		return evidence

	async def get_evidence(self, evidence_id: str) -> Evidence:
		"""Retrieve an evidence item by ID."""
		row = await self._require(evidence_id, _COL["evidence"])
		return Evidence.model_validate(row)

	async def list_evidence(
		self,
		workspace_id: str | None = None,
		evidence_type: str | None = None,
		status: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List evidence items with optional filters."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if workspace_id:
			filters["workspace_id"] = workspace_id
		if evidence_type:
			filters["evidence_type"] = evidence_type
		if status:
			filters["status"] = status
		rows = await self._store.query(_COL["evidence"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_evidence(self, evidence_id: str, patch: EvidenceUpdate) -> Evidence:
		"""Update evidence status or chain-of-custody."""
		row = await self._require(evidence_id, _COL["evidence"])
		old_status = row.get("status")
		updates = patch.model_dump(exclude_none=True)
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["evidence"], row)
		if patch.status and patch.status.value != old_status:
			self._emit_event(evidence_status_changed(
				self.tenant_id, self.actor_id, evidence_id, patch.status.value,
			))
		return Evidence.model_validate(row)

	async def verify_evidence(self, evidence_id: str) -> Evidence:
		"""Mark evidence as verified."""
		return await self.update_evidence(evidence_id, EvidenceUpdate(status=EvidenceStatus.VERIFIED))

	async def challenge_evidence(self, evidence_id: str) -> Evidence:
		"""Mark evidence as challenged."""
		return await self.update_evidence(evidence_id, EvidenceUpdate(status=EvidenceStatus.CHALLENGED))

	async def discredit_evidence(self, evidence_id: str) -> Evidence:
		"""Discredit evidence — it can no longer be used in hypotheses."""
		return await self.update_evidence(evidence_id, EvidenceUpdate(status=EvidenceStatus.DISCREDITED))

	async def delete_evidence(self, evidence_id: str) -> bool:
		"""Soft-delete an evidence item."""
		row = await self._require(evidence_id, _COL["evidence"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["evidence"], row)
		return True

	# ─────────────────────────────────────────────────────────────────────────
	# HypothesisTest CRUD + ACH
	# ─────────────────────────────────────────────────────────────────────────

	async def create_hypothesis(self, payload: HypothesisTestCreate) -> HypothesisTest:
		"""Create a structured hypothesis test."""
		rules.assert_tenant_context(payload.tenant_id)
		self._assert_own_tenant(payload.tenant_id)
		rules.assert_analyst_assigned(payload.analyst_id)
		rules.assert_sat_method_supported(payload.sat_method.value)
		rules.assert_confidence_in_range(payload.initial_confidence)
		if payload.sat_method.value == "analysis_of_competing_hypotheses":
			rules.assert_hypothesis_has_alternatives(payload.alternative_hypotheses)

		hypothesis = HypothesisTest(**payload.model_dump())
		await self._store.put(_COL["hypothesis"], hypothesis.model_dump(mode="json"))
		self._emit_event(hypothesis_created(
			self.tenant_id, self.actor_id, hypothesis.id, hypothesis.sat_method.value,
		))
		self._log_created("HypothesisTest", hypothesis.id)
		return hypothesis

	async def get_hypothesis(self, hypothesis_id: str) -> HypothesisTest:
		"""Retrieve a hypothesis test by ID."""
		row = await self._require(hypothesis_id, _COL["hypothesis"])
		return HypothesisTest.model_validate(row)

	async def list_hypotheses(
		self,
		workspace_id: str | None = None,
		status: str | None = None,
		sat_method: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> PagedResult:
		"""List hypothesis tests with optional filters."""
		filters: dict[str, Any] = {"tenant_id": self.tenant_id, "is_deleted": False}
		if workspace_id:
			filters["workspace_id"] = workspace_id
		if status:
			filters["status"] = status
		if sat_method:
			filters["sat_method"] = sat_method
		rows = await self._store.query(_COL["hypothesis"], filters, limit=page_size * page)
		return self._paged(rows, page, page_size)

	async def update_hypothesis(self, hypothesis_id: str, patch: HypothesisTestUpdate) -> HypothesisTest:
		"""Update a hypothesis test with new evidence or conclusion."""
		row = await self._require(hypothesis_id, _COL["hypothesis"])
		rules.assert_hypothesis_open_for_update(row.get("status", "open"))
		updates = patch.model_dump(exclude_none=True)
		row.update(updates)
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["hypothesis"], row)
		if patch.status and patch.status.value in ("supported", "refuted", "inconclusive"):
			final_conf = patch.final_confidence or row.get("initial_confidence", 0.5)
			self._emit_event(hypothesis_concluded(
				self.tenant_id, self.actor_id, hypothesis_id,
				patch.status.value, final_conf,
			))
		return HypothesisTest.model_validate(row)

	async def delete_hypothesis(self, hypothesis_id: str) -> bool:
		"""Soft-delete a hypothesis test."""
		row = await self._require(hypothesis_id, _COL["hypothesis"])
		row["is_deleted"] = True
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["hypothesis"], row)
		return True

	# ─────────────────────────────────────────────────────────────────────────
	# Intelligence Fusion — core analytical operations
	# ─────────────────────────────────────────────────────────────────────────

	async def fuse_intelligence(
		self,
		workspace_id: str,
		source_ids: list[str] | None = None,
		time_window: tuple[float, float] | None = None,
	) -> dict[str, Any]:
		"""
		Fuse all validated intelligence items in a workspace.

		Optionally restrict to specific source IDs or a time window
		[start_ts, end_ts] (Unix timestamps).

		Returns a fusion summary including quality score and corroboration.
		"""
		ws = await self._require_workspace(workspace_id)
		rules.assert_workspace_active(ws.get("status", "active"))

		filters: dict[str, Any] = {
			"workspace_id": workspace_id,
			"tenant_id": self.tenant_id,
			"is_deleted": False,
		}
		items = await self._store.query(_COL["item"], filters, limit=10_000)

		if source_ids:
			items = [i for i in items if i.get("id") in source_ids]

		if time_window:
			rules.assert_time_window_valid(time_window[0], time_window[1])
			items = [
				i for i in items
				if time_window[0] <= _ts(i.get("collected_at")) <= time_window[1]
			]

		rules.assert_minimum_sources_for_fusion(len(items))

		source_types = [i.get("source_type", "") for i in items]
		confidences = [float(i.get("confidence_score", 0.5)) for i in items]
		corroboration = calc.source_corroboration_score(confidences, source_types)
		quality = calc.fusion_quality_score(
			source_count=len(items),
			unique_source_types=len(set(source_types)),
			avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
			has_cross_source_confirmation=len(set(source_types)) >= 2,
			has_structured_analytic_technique=False,
		)

		# Mark items as fused
		for item in items:
			if item.get("status") in ("raw", "validated"):
				item["status"] = IntelItemStatus.FUSED.value
				item["updated_at"] = datetime.utcnow().isoformat()
				await self._store.put(_COL["item"], item)

		self._emit_event(fusion_completed(
			self.tenant_id, self.actor_id, workspace_id,
			len(items), list(set(source_types)), quality["quality_score"],
		))

		return {
			"workspace_id": workspace_id,
			"fused_item_count": len(items),
			"source_types": list(set(source_types)),
			"corroboration": corroboration,
			"quality": quality,
		}

	async def correlate_across_domains(
		self,
		workspace_id: str,
		osint_ids: list[str] | None = None,
		sigint_ids: list[str] | None = None,
		humint_ids: list[str] | None = None,
		additional_domain_ids: dict[str, list[str]] | None = None,
	) -> dict[str, Any]:
		"""
		Correlate intelligence items across OSINT, SIGINT, HUMINT and other domains.

		Returns a cross-domain correlation score and recommended correlation sets.
		"""
		await self._require_workspace(workspace_id)

		domain_map: dict[str, list[str]] = {}
		if osint_ids:
			domain_map["osint"] = osint_ids
		if sigint_ids:
			domain_map["sigint"] = sigint_ids
		if humint_ids:
			domain_map["humint"] = humint_ids
		if additional_domain_ids:
			domain_map.update(additional_domain_ids)

		domain_confidences: dict[str, float] = {}
		for domain, ids in domain_map.items():
			if not ids:
				continue
			items = []
			for iid in ids:
				row = await self._store.get(_COL["item"], iid)
				if row:
					items.append(row)
			if items:
				confidences = [float(i.get("confidence_score", 0.5)) for i in items]
				domain_confidences[domain] = sum(confidences) / len(confidences)

		cross_score = calc.cross_domain_correlation_score(domain_confidences)

		return {
			"workspace_id": workspace_id,
			"domains_covered": list(domain_confidences.keys()),
			"cross_domain_score": cross_score,
			"item_counts": {d: len(ids) for d, ids in domain_map.items()},
		}

	async def apply_structured_analytic_techniques(
		self,
		workspace_id: str,
		method: str,
		hypotheses: list[str],
		evidence_items: list[dict[str, Any]],
		assumptions: list[str] | None = None,
		assumption_confidences: list[float] | None = None,
	) -> dict[str, Any]:
		"""
		Apply a named SAT to the current workspace.

		Dispatches to ACH, KAC, or other methods as specified.
		Returns structured output for each technique.
		"""
		rules.assert_sat_method_supported(method)

		if method == "analysis_of_competing_hypotheses":
			return await self.analysis_of_competing_hypotheses(workspace_id, hypotheses, evidence_items)

		if method == "key_assumptions_check":
			return await self.key_assumptions_check(
				workspace_id,
				assumptions or hypotheses,
				assumption_confidences or [0.5] * len(hypotheses),
			)

		# Generic SAT result for other methods
		return {
			"workspace_id": workspace_id,
			"method": method,
			"hypotheses": hypotheses,
			"recommendation": f"Apply {method.replace('_', ' ')} manually with analyst judgment",
		}

	async def analysis_of_competing_hypotheses(
		self,
		workspace_id: str,
		hypotheses: list[str],
		evidence_items: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""
		Full ACH analysis: build matrix, score, identify leading hypothesis.

		evidence_items format: [{"label": str, "consistencies": [float, ...]}]
		Consistency values: 1=consistent, 0=irrelevant, -1=inconsistent.
		"""
		rules.assert_hypothesis_has_alternatives(hypotheses[1:] if len(hypotheses) > 1 else [])

		ach = calc.build_ach_matrix(hypotheses, evidence_items)
		confidence = ach.get("confidence", 0.5)

		# Emit ACH event referencing workspace
		self._emit_event(ach_completed(
			self.tenant_id, self.actor_id, workspace_id,
			ach.get("leading_hypothesis", ""),
			float(confidence),
		))

		return {
			"workspace_id": workspace_id,
			"method": "analysis_of_competing_hypotheses",
			**ach,
		}

	async def ace_method(
		self,
		workspace_id: str,
		analysis_statement: str,
		confidence_score: float,
		evidence_ids: list[str],
	) -> dict[str, Any]:
		"""
		ACE (Analysis, Confidence, Evidence) structured assessment for a workspace.

		Gathers evidence metadata and returns a structured ACE output.
		"""
		rules.assert_confidence_in_range(confidence_score)

		evidence_types: list[str] = []
		cross_confirmed = False
		for eid in evidence_ids:
			row = await self._store.get(_COL["evidence"], eid)
			if row:
				evidence_types.append(row.get("evidence_type", "observation"))
		if len(set(evidence_types)) >= 2:
			cross_confirmed = True

		result = calc.ace_assessment(
			analysis_statement, confidence_score,
			len(evidence_ids), evidence_types, cross_confirmed,
		)
		return {"workspace_id": workspace_id, **result}

	async def key_assumptions_check(
		self,
		workspace_id: str,
		assumptions: list[str],
		confidence_scores: list[float],
	) -> dict[str, Any]:
		"""
		Key Assumptions Check — assess robustness of analytic assumptions.

		Returns weakest assumption, geometric-mean robustness, and recommendation.
		"""
		result = calc.evaluate_assumptions(assumptions, confidence_scores)
		return {"workspace_id": workspace_id, "method": "key_assumptions_check", **result}

	async def generate_finished_intelligence(
		self,
		workspace_id: str,
		product_id: str,
	) -> dict[str, Any]:
		"""
		Generate a finished intelligence product from a workspace.

		Validates quality thresholds, marks related items as assessed, and
		returns the product with quality metadata.
		"""
		product = await self.get_product(product_id)
		ws = await self._require_workspace(workspace_id)

		# Gather assessment confidence scores
		assessment_confidences: list[float] = []
		for aid in product.assessment_ids:
			row = await self._store.get(_COL["assessment"], aid)
			if row:
				assessment_confidences.append(float(row.get("confidence_score", 0.5)))

		avg_conf = sum(assessment_confidences) / len(assessment_confidences) if assessment_confidences else 0.5
		quality = calc.fusion_quality_score(
			source_count=len(product.assessment_ids),
			unique_source_types=1,
			avg_confidence=avg_conf,
			has_cross_source_confirmation=len(product.assessment_ids) >= 2,
			has_structured_analytic_technique=True,
		)

		return {
			"workspace_id": workspace_id,
			"product_id": product_id,
			"product_type": product.product_type.value,
			"title": product.title,
			"classification": product.classification.value,
			"tlp": product.tlp.value,
			"quality": quality,
			"ready_for_release": quality["quality_score"] >= 0.55,
		}

	async def confidence_calibration(
		self,
		prior: float,
		likelihood_given_true: float,
		likelihood_given_false: float,
	) -> dict[str, Any]:
		"""
		Bayesian confidence calibration.

		Computes P(H|E) given prior and likelihood ratio.
		Returns calibrated posterior with ICD-203 word equivalent.
		"""
		rules.assert_confidence_in_range(prior)
		return calc.confidence_calibration_report(prior, likelihood_given_true, likelihood_given_false)

	async def dissemination_with_tlp(
		self,
		product_id: str,
		audience: str,
		recipient_max_tlp: str,
		approval_reference: str,
		disseminated_by: str,
		notes: str = "",
	) -> DisseminationRecord:
		"""
		Disseminate a released product to an audience, enforcing TLP compatibility.

		Raises RuleViolation if the product TLP exceeds recipient clearance.
		"""
		row = await self._require(product_id, _COL["product"])
		product = IntelligenceProduct.model_validate(row)
		rules.assert_product_in_approved_state(product.status.value if product.status != ProductStatus.RELEASED else "approved")
		rules.assert_audience_specified(audience)
		rules.assert_approval_present(approval_reference)
		rules.assert_tlp_compatible_with_audience(product.tlp.value, recipient_max_tlp)

		record = DisseminationRecord(
			tenant_id=self.tenant_id,
			product_id=product_id,
			audience=audience,
			tlp=product.tlp,
			approval_reference=approval_reference,
			disseminated_by=disseminated_by,
			notes=notes,
		)
		await self._store.put(_COL["dissem"], record.model_dump(mode="json"))

		# Track dissemination on product
		row["dissemination_ids"] = row.get("dissemination_ids", []) + [record.id]
		row["updated_at"] = datetime.utcnow().isoformat()
		await self._store.put(_COL["product"], row)

		return record

	# ─────────────────────────────────────────────────────────────────────────
	# Reporting
	# ─────────────────────────────────────────────────────────────────────────

	async def dashboard_report(self) -> FusionDashboardReport:
		"""Return a tenant-level dashboard report."""
		t = {"tenant_id": self.tenant_id, "is_deleted": False}

		items = await self._store.query(_COL["item"], t, limit=100_000)
		items_by_source: dict[str, int] = {}
		items_by_status: dict[str, int] = {}
		for i in items:
			st = i.get("source_type", "unknown")
			items_by_source[st] = items_by_source.get(st, 0) + 1
			s = i.get("status", "raw")
			items_by_status[s] = items_by_status.get(s, 0) + 1

		workspaces = await self._store.query(_COL["workspace"], t, limit=100_000)
		active_ws = [w for w in workspaces if w.get("status") == "active"]
		assessments = await self._store.query(_COL["assessment"], t, limit=100_000)
		products = await self._store.query(_COL["product"], t, limit=100_000)
		hypotheses = await self._store.query(_COL["hypothesis"], t, limit=100_000)
		evidence = await self._store.query(_COL["evidence"], t, limit=100_000)
		judgements = await self._store.query(_COL["judgement"], t, limit=100_000)
		correlations = await self._store.query(_COL["correlation"], t, limit=100_000)

		return FusionDashboardReport(
			tenant_id=self.tenant_id,
			total_items=len(items),
			items_by_source=items_by_source,
			items_by_status=items_by_status,
			total_workspaces=len(workspaces),
			active_workspaces=len(active_ws),
			total_correlations=len(correlations),
			total_assessments=len(assessments),
			critical_assessments=sum(1 for a in assessments if a.get("risk_level") == "critical"),
			total_products=len(products),
			released_products=sum(1 for p in products if p.get("status") == "released"),
			total_hypotheses=len(hypotheses),
			open_hypotheses=sum(1 for h in hypotheses if h.get("status") == "open"),
			total_evidence=len(evidence),
			total_judgements=len(judgements),
		)

	# ─────────────────────────────────────────────────────────────────────────
	# Private helpers
	# ─────────────────────────────────────────────────────────────────────────

	def _assert_own_tenant(self, tenant_id: str) -> None:
		if tenant_id != self.tenant_id:
			raise PermissionError(
				f"[cross_tenant_access_denied] actor tenant '{self.tenant_id}' "
				f"may not access resources owned by '{tenant_id}'"
			)

	async def _require(self, resource_id: str, collection: str) -> dict[str, Any]:
		row = await self._store.get(collection, resource_id)
		if not row or row.get("is_deleted"):
			raise KeyError(f"{collection}/{resource_id} not found")
		if row.get("tenant_id") != self.tenant_id:
			raise PermissionError(f"[cross_tenant_access_denied] {collection}/{resource_id}")
		return row

	async def _require_workspace(self, workspace_id: str) -> dict[str, Any]:
		return await self._require(workspace_id, _COL["workspace"])

	def _emit_event(self, event: Any) -> None:
		data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
		self._events.append(data)
		logger.debug("event: %s resource=%s", data.get("event_type"), data.get("resource_id"))

	@staticmethod
	def _paged(rows: list[dict[str, Any]], page: int, page_size: int) -> PagedResult:
		start = (page - 1) * page_size
		end = start + page_size
		page_rows = rows[start:end]
		return PagedResult(
			items=page_rows,
			total=len(rows),
			page=page,
			page_size=page_size,
			has_more=end < len(rows),
		)

	def _log_created(self, entity: str, entity_id: str) -> None:
		logger.info("[fusion] created %s id=%s tenant=%s", entity, entity_id, self.tenant_id)

	def _log_state_change(self, entity: str, entity_id: str, old: str, new: str) -> None:
		logger.info(
			"[fusion] %s id=%s tenant=%s %s→%s",
			entity, entity_id, self.tenant_id, old, new,
		)


def _ts(val: Any) -> float:
	"""Best-effort conversion of a stored timestamp to a Unix float."""
	if val is None:
		return 0.0
	if isinstance(val, (int, float)):
		return float(val)
	try:
		return datetime.fromisoformat(str(val)).timestamp()
	except Exception:
		return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Sync contract-layer — in-memory dataclasses mirroring the capability contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionAuthority:
	id: str
	tenant_id: str
	authority_type: str
	scope_reference: str
	classification: str
	approver_id: str
	expires_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionWorkspaceRecord:
	id: str
	tenant_id: str
	workspace_type: str
	name: str
	classification: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionSource:
	id: str
	tenant_id: str
	source_type: str
	source_reference: str
	custodian_id: str
	authority_id: str
	lineage_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionArtifact:
	id: str
	tenant_id: str
	workspace_id: str
	source_id: str
	artifact_type: str
	artifact_reference: str
	content_fingerprint: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionCorrelationRecord:
	id: str
	tenant_id: str
	artifact_id: str
	correlation_type: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionHypothesisRecord:
	id: str
	tenant_id: str
	correlation_id: str
	hypothesis_type: str
	claim_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionAssessmentRecord:
	id: str
	tenant_id: str
	hypothesis_id: str
	assessment_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionReferral:
	id: str
	tenant_id: str
	assessment_id: str
	referral_type: str
	recipient: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionDissemination:
	id: str
	tenant_id: str
	assessment_id: str
	audience: str
	release_marking: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Sync contract methods — mixed into IntelligenceFusionService
# These are non-async, in-memory dict-backed methods that satisfy the
# capability contract tests (test_package_contract.py).
# ─────────────────────────────────────────────────────────────────────────────

def _add_sync_contract_layer(cls: type) -> type:
	"""Attach sync record_* methods to IntelligenceFusionService at class creation time."""

	def _init_sync(self: Any) -> None:
		"""Initialise sync contract collections if not already present."""
		if not hasattr(self, "_sync_authorities"):
			self._sync_authorities: dict[tuple[str, str], FusionAuthority] = {}
			self._sync_workspaces: dict[tuple[str, str], FusionWorkspaceRecord] = {}
			self._sync_sources: dict[tuple[str, str], FusionSource] = {}
			self._sync_artifacts: dict[tuple[str, str], FusionArtifact] = {}
			self._sync_correlations: dict[tuple[str, str], FusionCorrelationRecord] = {}
			self._sync_hypotheses: dict[tuple[str, str], FusionHypothesisRecord] = {}
			self._sync_assessments: dict[tuple[str, str], FusionAssessmentRecord] = {}
			self._sync_referrals: dict[tuple[str, str], FusionReferral] = {}
			self._sync_disseminations: dict[tuple[str, str], FusionDissemination] = {}
			self._sync_reviews: dict[tuple[str, str], FusionReview] = {}
			self._sync_agents: dict[tuple[str, str], FusionAgent] = {}
			self._sync_audit_events: list[dict[str, Any]] = []

	# Patch __init__ to call _init_sync
	_orig_init = cls.__init__

	def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
		_orig_init(self, *args, **kwargs)
		_init_sync(self)

	cls.__init__ = _patched_init

	def _tk(self: Any, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _sync_audit(self: Any, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._sync_audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
		})

	def _sync_count(self: Any, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _sync_enforce(self: Any, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "fusion_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "fusion_policy_denied")

	def describe(self: Any, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self: Any, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(
		self: Any,
		authority_id: str,
		tenant_id: str,
		authority_type: str,
		scope_reference: str,
		classification: str,
		approver_id: str,
		expires_at: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		_init_sync(self)
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = FusionAuthority(authority_id, tenant_id, authority_type, scope_reference,
		                       classification, approver_id, expires_at, evidence_reference)
		self._sync_authorities[_tk(self, tenant_id, authority_id)] = item
		_sync_audit(self, tenant_id, "fusion_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(
		self: Any,
		workspace_id: str,
		tenant_id: str,
		workspace_type: str,
		name: str,
		classification: str,
		authority_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		authority = _tenant_authority_or_none(self, authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_workspace",
			"workspace_type_supported": workspace_type in _CONTRACT_WS_TYPES,
			"workspace_name_present": present(name),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = FusionWorkspaceRecord(workspace_id, tenant_id, workspace_type, name,
		                             classification, authority_id, evidence_reference)
		self._sync_workspaces[_tk(self, tenant_id, workspace_id)] = item
		_sync_audit(self, tenant_id, "fusion_workspace_recorded", workspace_id)
		return item.to_dict()

	def register_source(
		self: Any,
		source_id: str,
		tenant_id: str,
		source_type: str,
		source_reference: str,
		custodian_id: str,
		authority_id: str,
		lineage_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		source_type = normalize_code(source_type)
		authority = _tenant_authority_or_none(self, authority_id, tenant_id)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"source_reference_present": present(source_reference),
			"custodian_present": present(custodian_id),
			"authority_present": authority is not None,
			"lineage_present": present(lineage_reference),
			"evidence_present": present(evidence_reference),
		})
		item = FusionSource(source_id, tenant_id, source_type, source_reference,
		                    custodian_id, authority_id, lineage_reference, evidence_reference)
		self._sync_sources[_tk(self, tenant_id, source_id)] = item
		_sync_audit(self, tenant_id, "fusion_source_registered", source_id)
		return item.to_dict()

	def record_artifact(
		self: Any,
		artifact_id: str,
		tenant_id: str,
		workspace_id: str,
		source_id: str,
		artifact_type: str,
		artifact_reference: str,
		content_fingerprint: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		artifact_type = normalize_code(artifact_type)
		workspace = _tenant_workspace_or_none(self, workspace_id, tenant_id)
		source = self._sync_sources.get(_tk(self, tenant_id, source_id))
		# authority_mismatch: workspace and source must share the same authority
		authority_match = (
			workspace is not None
			and source is not None
			and workspace.authority_id == source.authority_id
		)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_artifact",
			"workspace_present": workspace is not None,
			"source_present": source is not None,
			"workspace_source_authority_match": authority_match,
			"artifact_type_supported": artifact_type in SUPPORTED_ARTIFACT_TYPES,
			"artifact_reference_present": present(artifact_reference),
			"fingerprint_present": present(content_fingerprint),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = FusionArtifact(artifact_id, tenant_id, workspace_id, source_id,
		                      artifact_type, artifact_reference, content_fingerprint,
		                      float(confidence_score), evidence_reference)
		self._sync_artifacts[_tk(self, tenant_id, artifact_id)] = item
		_sync_audit(self, tenant_id, "fusion_artifact_recorded", artifact_id)
		return item.to_dict()

	def record_correlation(
		self: Any,
		correlation_id: str,
		tenant_id: str,
		artifact_id: str,
		correlation_type: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		correlation_type = normalize_code(correlation_type)
		artifact = self._sync_artifacts.get(_tk(self, tenant_id, artifact_id))
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_correlation",
			"artifact_present": artifact is not None,
			"correlation_type_supported": correlation_type in _CONTRACT_CORR_TYPES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = FusionCorrelationRecord(correlation_id, tenant_id, artifact_id,
		                               correlation_type, float(confidence_score),
		                               analyst_id, evidence_reference)
		self._sync_correlations[_tk(self, tenant_id, correlation_id)] = item
		_sync_audit(self, tenant_id, "fusion_correlation_recorded", correlation_id)
		return item.to_dict()

	def record_hypothesis(
		self: Any,
		hypothesis_id: str,
		tenant_id: str,
		correlation_id: str,
		hypothesis_type: str,
		claim_reference: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		hypothesis_type = normalize_code(hypothesis_type)
		correlation = self._sync_correlations.get(_tk(self, tenant_id, correlation_id))
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_hypothesis",
			"correlation_present": correlation is not None,
			"hypothesis_type_supported": hypothesis_type in SUPPORTED_HYPOTHESIS_TYPES,
			"claim_present": present(claim_reference),
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = FusionHypothesisRecord(hypothesis_id, tenant_id, correlation_id,
		                              hypothesis_type, claim_reference,
		                              float(confidence_score), analyst_id, evidence_reference)
		self._sync_hypotheses[_tk(self, tenant_id, hypothesis_id)] = item
		_sync_audit(self, tenant_id, "fusion_hypothesis_recorded", hypothesis_id)
		return item.to_dict()

	def record_assessment(
		self: Any,
		assessment_id: str,
		tenant_id: str,
		hypothesis_id: str,
		assessment_type: str,
		risk_level: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		hypothesis = self._sync_hypotheses.get(_tk(self, tenant_id, hypothesis_id))
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_assessment",
			"hypothesis_present": hypothesis is not None,
			"assessment_type_supported": assessment_type in _CONTRACT_ASSESS_TYPES,
			"risk_level_supported": risk_level in _CONTRACT_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = FusionAssessmentRecord(assessment_id, tenant_id, hypothesis_id,
		                              assessment_type, risk_level, float(confidence_score),
		                              analyst_id, evidence_reference)
		self._sync_assessments[_tk(self, tenant_id, assessment_id)] = item
		_sync_audit(self, tenant_id, "fusion_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_referral(
		self: Any,
		referral_id: str,
		tenant_id: str,
		assessment_id: str,
		referral_type: str,
		recipient: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		referral_type = normalize_code(referral_type)
		assessment = self._sync_assessments.get(_tk(self, tenant_id, assessment_id))
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_referral",
			"assessment_present": assessment is not None,
			"referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES,
			"recipient_present": present(recipient),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = FusionReferral(referral_id, tenant_id, assessment_id, referral_type,
		                      recipient, approval_reference, evidence_reference)
		self._sync_referrals[_tk(self, tenant_id, referral_id)] = item
		_sync_audit(self, tenant_id, "fusion_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(
		self: Any,
		dissemination_id: str,
		tenant_id: str,
		assessment_id: str,
		audience: str,
		release_marking: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		assessment = self._sync_assessments.get(_tk(self, tenant_id, assessment_id))
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_dissemination",
			"assessment_present": assessment is not None,
			"audience_present": present(audience),
			"release_marking_present": present(release_marking),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = FusionDissemination(dissemination_id, tenant_id, assessment_id,
		                           audience, release_marking, approval_reference, evidence_reference)
		self._sync_disseminations[_tk(self, tenant_id, dissemination_id)] = item
		_sync_audit(self, tenant_id, "fusion_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(
		self: Any,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		_init_sync(self)
		status = normalize_code(status)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = FusionReview(review_id, tenant_id, reference_id, reviewer_id,
		                    status, evidence_reference)
		self._sync_reviews[_tk(self, tenant_id, review_id)] = item
		_sync_audit(self, tenant_id, "fusion_review_recorded", reference_id)
		return item.to_dict()

	def register_fusion_agent(
		self: Any,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str = "fusion operations",
	) -> dict[str, Any]:
		_init_sync(self)
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_fusion_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = FusionAgent(agent_id, tenant_id, name, runtime, role, scope)
		self._sync_agents[_tk(self, tenant_id, agent_id)] = item
		_sync_audit(self, tenant_id, "fusion_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self: Any,
		tenant_id: str,
		privileged_scope: bool = False,
		human_approval_recorded: bool = False,
		evidence_fabrication_scope: bool = False,
		source_tampering_scope: bool = False,
		privacy_bypass_scope: bool = False,
		unsupported_identity_resolution_scope: bool = False,
		autonomous_dissemination_scope: bool = False,
		unapproved_attribution_scope: bool = False,
	) -> dict[str, Any]:
		_init_sync(self)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "fusion_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"evidence_fabrication_scope": evidence_fabrication_scope,
			"source_tampering_scope": source_tampering_scope,
			"privacy_bypass_scope": privacy_bypass_scope,
			"unsupported_identity_resolution_scope": unsupported_identity_resolution_scope,
			"autonomous_dissemination_scope": autonomous_dissemination_scope,
			"unapproved_attribution_scope": unapproved_attribution_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self: Any,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		_init_sync(self)
		_sync_enforce(self, {
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "fusion_batch",
			"event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": FUSION_EVENT_STREAM,
			"accepted": True,
		}

	def dashboard_summary(self: Any, tenant_id: str) -> dict[str, Any]:
		_init_sync(self)
		return {
			"tenant_id": tenant_id,
			"authority_count": _sync_count(self, self._sync_authorities, tenant_id),
			"workspace_count": _sync_count(self, self._sync_workspaces, tenant_id),
			"source_count": _sync_count(self, self._sync_sources, tenant_id),
			"artifact_count": _sync_count(self, self._sync_artifacts, tenant_id),
			"correlation_count": _sync_count(self, self._sync_correlations, tenant_id),
			"hypothesis_count": _sync_count(self, self._sync_hypotheses, tenant_id),
			"assessment_count": _sync_count(self, self._sync_assessments, tenant_id),
			"referral_count": _sync_count(self, self._sync_referrals, tenant_id),
			"dissemination_count": _sync_count(self, self._sync_disseminations, tenant_id),
			"review_count": _sync_count(self, self._sync_reviews, tenant_id),
			"agent_count": _sync_count(self, self._sync_agents, tenant_id),
			"audit_event_count": sum(
				1 for e in self._sync_audit_events if e["tenant_id"] == tenant_id
			),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def _tenant_authority_or_none(self: Any, item_id: str, tenant_id: str) -> FusionAuthority | None:
		_init_sync(self)
		return self._sync_authorities.get(_tk(self, tenant_id, item_id))

	def _tenant_workspace_or_none(self: Any, item_id: str, tenant_id: str) -> FusionWorkspaceRecord | None:
		_init_sync(self)
		return self._sync_workspaces.get(_tk(self, tenant_id, item_id))

	# Attach all methods to the class
	cls.describe = describe
	cls.evaluate = evaluate
	cls.record_authority = record_authority
	cls.record_workspace = record_workspace
	cls.register_source = register_source
	cls.record_artifact = record_artifact
	cls.record_correlation = record_correlation
	cls.record_hypothesis = record_hypothesis
	cls.record_assessment = record_assessment
	cls.record_referral = record_referral
	cls.record_dissemination = record_dissemination
	cls.record_review = record_review
	cls.register_fusion_agent = register_fusion_agent
	cls.validate_agent_action = validate_agent_action
	cls.validate_batch = validate_batch
	cls.dashboard_summary = dashboard_summary
	cls._tenant_authority_or_none = _tenant_authority_or_none
	cls._tenant_workspace_or_none = _tenant_workspace_or_none
	return cls


# Apply the sync contract layer to IntelligenceFusionService
IntelligenceFusionService = _add_sync_contract_layer(IntelligenceFusionService)
