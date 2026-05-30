"""Dependency-light Product Information Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import PIM_EVENT_STREAM, STREAMING, SUPPORTED_ATTRIBUTE_TYPES, SUPPORTED_CHANNELS, SUPPORTED_PIM_AGENT_ROLES, SUPPORTED_PIM_AGENT_RUNTIMES, SUPPORTED_PRODUCT_TYPES, evaluate_capability_rules, get_capability_contract
except ImportError:  # pragma: no cover
	from capability_contract import PIM_EVENT_STREAM, STREAMING, SUPPORTED_ATTRIBUTE_TYPES, SUPPORTED_CHANNELS, SUPPORTED_PIM_AGENT_ROLES, SUPPORTED_PIM_AGENT_RUNTIMES, SUPPORTED_PRODUCT_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore


class PIMError(Exception):
	"""Base exception for PIM operations."""


class PIMRecordNotFoundError(PIMError):
	"""Raised when a PIM lifecycle record is not found."""


class ProductInformationLifecycleService:
	"""In-memory executable service for PIM lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.catalogs: dict[str, dict[str, Any]] = {}
		self.products: dict[str, dict[str, Any]] = {}
		self.attributes: dict[str, dict[str, Any]] = {}
		self.attribute_values: dict[str, dict[str, Any]] = {}
		self.variants: dict[str, dict[str, Any]] = {}
		self.content: dict[str, dict[str, Any]] = {}
		self.assets: dict[str, dict[str, Any]] = {}
		self.compliance: dict[str, dict[str, Any]] = {}
		self.channels: dict[str, dict[str, Any]] = {}
		self.publications: dict[str, dict[str, Any]] = {}
		self.quality_issues: dict[str, dict[str, Any]] = {}
		self.changes: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "tenant_context_present": True, "operation": operation, "operation_type": "write", "policy_attached": True, "audit_enabled": True}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "record_id": record["id"], "record_type": record["type"], "status": record["status"], "stream": PIM_EVENT_STREAM, "processor": "bytewax", "emitted_at": self._now()})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_catalog(self, catalog_id: str, tenant_id: str, code: str, name: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_catalog")
		context.update({"code_present": bool(code), "name_present": bool(name), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("catalog", catalog_id), "type": "product_catalog", "kind": "catalog", "tenant_id": tenant, "code": code.upper(), "name": name, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.catalogs[record["id"]] = record
		self._emit(tenant, "catalog_created", record)
		return deepcopy(record)

	def create_product(self, product_id: str, tenant_id: str, catalog_id: str, sku: str, name: str, product_type: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		catalog = self._get(self.catalogs, catalog_id, tenant, "catalog")
		context = self._base_context(tenant, "create_product")
		context.update({"catalog_present": bool(catalog), "sku_present": bool(sku), "name_present": bool(name), "product_type_supported": product_type in SUPPORTED_PRODUCT_TYPES, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("product", product_id), "type": "product_record", "kind": "product", "tenant_id": tenant, "catalog_id": catalog_id, "sku": sku, "name": name, "product_type": product_type, "owner_id": owner_id, "stage": "concept", "status": "draft", "created_at": self._now()}
		self.products[record["id"]] = record
		self._emit(tenant, "product_created", record)
		return deepcopy(record)

	def define_attribute(self, attribute_id: str, tenant_id: str, code: str, name: str, attribute_type: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "define_attribute")
		context.update({"code_present": bool(code), "attribute_type_supported": attribute_type in SUPPORTED_ATTRIBUTE_TYPES, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("attribute", attribute_id), "type": "product_attribute", "kind": "attribute", "tenant_id": tenant, "code": code, "name": name, "attribute_type": attribute_type, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.attributes[record["id"]] = record
		self._emit(tenant, "attribute_defined", record)
		return deepcopy(record)

	def set_attribute_value(self, value_id: str, tenant_id: str, product_id: str, attribute_id: str, value: Any, locale: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		attribute = self._get(self.attributes, attribute_id, tenant, "attribute")
		context = self._base_context(tenant, "set_attribute_value")
		context.update({"product_present": bool(product), "attribute_present": bool(attribute), "value_present": value is not None, "rich_text": attribute["attribute_type"] == "rich_text", "locale_present": bool(locale)})
		self._assert_rules(context)
		record = {"id": self._record_id("value", value_id), "type": "product_attribute_value", "kind": "attribute_value", "tenant_id": tenant, "product_id": product_id, "attribute_id": attribute_id, "value": value, "locale": locale, "status": "active", "created_at": self._now()}
		self.attribute_values[record["id"]] = record
		self._emit(tenant, "attribute_value_set", record)
		return deepcopy(record)

	def create_variant(self, variant_id: str, tenant_id: str, parent_product_id: str, sku: str, option_values: dict[str, Any]) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		parent = self._get(self.products, parent_product_id, tenant, "product")
		context = self._base_context(tenant, "create_variant")
		context.update({"parent_present": bool(parent), "sku_present": bool(sku), "options_present": bool(option_values)})
		self._assert_rules(context)
		record = {"id": self._record_id("variant", variant_id), "type": "product_variant", "kind": "variant", "tenant_id": tenant, "parent_product_id": parent_product_id, "sku": sku, "option_values": dict(option_values), "status": "draft", "created_at": self._now()}
		self.variants[record["id"]] = record
		self._emit(tenant, "variant_created", record)
		return deepcopy(record)

	def enrich_content(self, content_id: str, tenant_id: str, product_id: str, locale: str, title: str, body: str, generated: bool = False, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "enrich_content")
		context.update({"product_present": bool(product), "locale_present": bool(locale), "title_present": bool(title), "generated_content": bool(generated), "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("content", content_id), "type": "product_content", "kind": "content", "tenant_id": tenant, "product_id": product_id, "locale": locale, "title": title, "body": body, "generated": bool(generated), "reviewed_by": reviewed_by, "status": "approved" if reviewed_by or not generated else "review", "created_at": self._now()}
		self.content[record["id"]] = record
		self._emit(tenant, "content_enriched", record)
		return deepcopy(record)

	def attach_asset(self, asset_id: str, tenant_id: str, product_id: str, asset_type: str, url: str, rights_basis: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "attach_asset")
		context.update({"product_present": bool(product), "url_present": bool(url), "rights_basis_present": bool(rights_basis)})
		self._assert_rules(context)
		record = {"id": self._record_id("asset", asset_id), "type": "product_asset", "kind": "asset", "tenant_id": tenant, "product_id": product_id, "asset_type": asset_type, "url": url, "rights_basis": rights_basis, "status": "active", "created_at": self._now()}
		self.assets[record["id"]] = record
		self._emit(tenant, "asset_attached", record)
		return deepcopy(record)

	def record_compliance(self, compliance_id: str, tenant_id: str, product_id: str, framework: str, status_value: str, evidence_id: str, risk_tier: str = "low", reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "record_compliance")
		context.update({"product_present": bool(product), "evidence_present": bool(evidence_id), "high_risk": risk_tier in {"high", "critical"}, "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("compliance", compliance_id), "type": "product_compliance", "kind": "compliance", "tenant_id": tenant, "product_id": product_id, "framework": framework, "status_value": status_value, "evidence_id": evidence_id, "risk_tier": risk_tier, "reviewed_by": reviewed_by, "status": "recorded", "created_at": self._now()}
		self.compliance[record["id"]] = record
		self._emit(tenant, "compliance_recorded", record)
		return deepcopy(record)

	def create_channel_listing(self, listing_id: str, tenant_id: str, product_id: str, channel: str, external_listing_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "create_channel_listing")
		context.update({"product_present": bool(product), "channel_supported": channel in SUPPORTED_CHANNELS, "listing_id_present": bool(external_listing_id), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("listing", listing_id), "type": "channel_listing", "kind": "channel", "tenant_id": tenant, "product_id": product_id, "channel": channel, "external_listing_id": external_listing_id, "approved_by": approved_by, "status": "approved", "created_at": self._now()}
		self.channels[record["id"]] = record
		self._emit(tenant, "channel_listing_created", record)
		return deepcopy(record)

	def publish_product(self, publication_id: str, tenant_id: str, product_id: str, content_id: str, channel_listing_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		content = self.content.get(content_id)
		channel = self.channels.get(channel_listing_id)
		context = self._base_context(tenant, "publish_product")
		context.update({"product_present": bool(product), "approved_content_present": bool(content and content["tenant_id"] == tenant and content["status"] == "approved"), "approved_channel_present": bool(channel and channel["tenant_id"] == tenant and channel["status"] == "approved"), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		product["status"] = "published"
		record = {"id": self._record_id("publication", publication_id), "type": "product_publication", "kind": "publication", "tenant_id": tenant, "product_id": product_id, "content_id": content_id, "channel_listing_id": channel_listing_id, "approved_by": approved_by, "status": "published", "created_at": self._now()}
		self.publications[record["id"]] = record
		self._emit(tenant, "product_published", record)
		return deepcopy(record)

	def record_quality_issue(self, issue_id: str, tenant_id: str, product_id: str, severity: str, description: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "record_quality_issue")
		context.update({"product_present": bool(product), "high_or_critical": severity in {"high", "critical"}, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("quality", issue_id), "type": "data_quality_issue", "kind": "quality", "tenant_id": tenant, "product_id": product_id, "severity": severity, "description": description, "owner_id": owner_id, "status": "open", "created_at": self._now()}
		self.quality_issues[record["id"]] = record
		self._emit(tenant, "quality_issue_recorded", record)
		return deepcopy(record)

	def create_change_request(self, change_id: str, tenant_id: str, product_id: str, reason: str, requested_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "create_change_request")
		context.update({"product_present": bool(product), "reason_present": bool(reason)})
		self._assert_rules(context)
		record = {"id": self._record_id("change", change_id), "type": "product_change", "kind": "change", "tenant_id": tenant, "product_id": product_id, "reason": reason, "requested_by": requested_by, "approved_by": None, "status": "requested", "created_at": self._now()}
		self.changes[record["id"]] = record
		self._emit(tenant, "change_request_created", record)
		return deepcopy(record)

	def approve_change(self, change_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		change = self._get(self.changes, change_id, tenant, "change")
		context = self._base_context(tenant, "approve_change")
		context.update({"approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		change["approved_by"] = approved_by
		change["status"] = "approved"
		self._emit(tenant, "change_request_approved", change)
		return deepcopy(change)

	def register_pim_agent(self, tenant_id: str, name: str, runtime: str, role: str, purpose: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_pim_agent")
		context.update({"runtime_supported": runtime in SUPPORTED_PIM_AGENT_RUNTIMES, "role_supported": role in SUPPORTED_PIM_AGENT_ROLES})
		self._assert_rules(context)
		record = {"id": self._record_id("agent"), "type": "pim_agent", "kind": "agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "purpose": purpose, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.agents[record["id"]] = record
		self._emit(tenant, "pim_agent_registered", record)
		return deepcopy(record)

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if event_stream != "bytewax":
			self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "pim_batch", "event_stream": "queue"})
		return {"tenant_id": tenant, "record_count": int(record_count), "processor": "bytewax", "event_stream": PIM_EVENT_STREAM, "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		def count(records: dict[str, dict[str, Any]]) -> int:
			return sum(1 for record in records.values() if record["tenant_id"] == tenant)
		return {"tenant_id": tenant, "catalog_count": count(self.catalogs), "product_count": count(self.products), "attribute_count": count(self.attributes), "content_count": count(self.content), "asset_count": count(self.assets), "channel_count": count(self.channels), "publication_count": count(self.publications), "quality_issue_count": count(self.quality_issues), "change_count": count(self.changes), "agent_count": count(self.agents), "audit_event_count": sum(1 for event in self._audit_events if event["tenant_id"] == tenant), "streaming": deepcopy(STREAMING)}

	def list_records(self, tenant_id: str, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		stores = [self.catalogs, self.products, self.attributes, self.attribute_values, self.variants, self.content, self.assets, self.compliance, self.channels, self.publications, self.quality_issues, self.changes, self.agents]
		records = [record for store in stores for record in store.values() if record["tenant_id"] == tenant]
		if record_type:
			records = [record for record in records if record["type"] == record_type or record["kind"] == record_type]
		return deepcopy(records)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return deepcopy([event for event in self._audit_events if event["tenant_id"] == tenant])

	def _get(self, store: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str) -> dict[str, Any]:
		record = store.get(record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise PIMRecordNotFoundError(f"{label}_not_found")
		return record


PIMService = ProductInformationLifecycleService
PLMProductService = ProductInformationLifecycleService
ProductInformationService = ProductInformationLifecycleService
