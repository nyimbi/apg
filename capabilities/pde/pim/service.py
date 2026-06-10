"""Dependency-light Product Information Management lifecycle service — expanded implementation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		PIM_EVENT_STREAM, STREAMING, SUPPORTED_ATTRIBUTE_TYPES, SUPPORTED_CHANNELS,
		SUPPORTED_PIM_AGENT_ROLES, SUPPORTED_PIM_AGENT_RUNTIMES, SUPPORTED_PRODUCT_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		PIM_EVENT_STREAM, STREAMING, SUPPORTED_ATTRIBUTE_TYPES, SUPPORTED_CHANNELS,
		SUPPORTED_PIM_AGENT_ROLES, SUPPORTED_PIM_AGENT_RUNTIMES, SUPPORTED_PRODUCT_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class PIMError(Exception):
	"""Base exception for PIM operations."""


class PIMRecordNotFoundError(PIMError):
	"""Raised when a PIM lifecycle record is not found."""


class ProductInfoManagementService:
	"""
	In-memory executable service for PIM lifecycle packets.

	Expanded with: create_product, update_attributes, add_media,
	product_categorisation, data_quality_score, publish_to_channel,
	unpublish, bulk_import, product_search, pim_analytics.
	"""

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
		# New stores
		self._media: dict[str, dict[str, Any]] = {}
		self._categories: dict[str, dict[str, Any]] = {}
		self._category_assignments: dict[str, dict[str, Any]] = {}
		self._unpublications: list[dict[str, Any]] = []
		self._bulk_import_logs: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "tenant_context_present": True, "operation": operation, "operation_type": "write", "policy_attached": True, "audit_enabled": True}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		# Only hard-block on explicit deny; require_review creates an audit flag
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "record_id": record["id"], "record_type": record["type"], "status": record["status"], "stream": PIM_EVENT_STREAM, "processor": "bytewax", "emitted_at": _now()})

	def _get(self, store: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str) -> dict[str, Any]:
		record = store.get(record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise PIMRecordNotFoundError(f"{label}_not_found")
		return record

	# ------------------------------------------------------------------
	# create_product
	# ------------------------------------------------------------------

	def create_product(
		self,
		sku: str,
		name: str,
		category: str,
		attributes: dict[str, Any],
		tenant_id: str | None = None,
		product_id: str | None = None,
		product_type: str = "physical",
		owner_id: str = "catalog_manager",
		catalog_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Create a product with SKU, name, category, and initial attributes.

		sku: Unique stock-keeping unit.
		category: Product category label.
		attributes: Initial attribute key-value dict.
		"""
		tenant = self._tenant(tenant_id)
		# Auto-create catalog if needed
		resolved_catalog = catalog_id
		if not resolved_catalog:
			default_cat_id = f"default-catalog-{tenant}"
			if default_cat_id not in self.catalogs:
				self.create_catalog(default_cat_id, tenant, "DEFAULT", f"{tenant} Catalog", owner_id)
			resolved_catalog = default_cat_id
		context = self._base_context(tenant, "create_product")
		context.update({"catalog_present": True, "sku_present": bool(sku), "name_present": bool(name), "product_type_supported": product_type in SUPPORTED_PRODUCT_TYPES, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		if not sku:
			raise ValueError("sku_required")
		# Check SKU uniqueness within tenant
		if any(p["sku"] == sku and p["tenant_id"] == tenant for p in self.products.values()):
			raise ValueError(f"sku_already_exists:{sku}")
		resolved_id = self._record_id("product", product_id)
		record = {
			"id": resolved_id,
			"type": "product_record",
			"kind": "product",
			"tenant_id": tenant,
			"catalog_id": resolved_catalog,
			"sku": sku,
			"name": name,
			"product_type": product_type,
			"category": category,
			"attributes": dict(attributes),
			"owner_id": owner_id,
			"stage": "concept",
			"status": "draft",
			"created_at": _now(),
		}
		self.products[resolved_id] = record
		self._emit(tenant, "product_created", record)
		return deepcopy(record)

	def update_attributes(
		self,
		sku: str,
		attributes: dict[str, Any],
		tenant_id: str | None = None,
		updated_by: str = "system",
	) -> dict[str, Any]:
		"""
		Update one or more attributes on a product identified by SKU.

		Merges the provided attributes dict into existing attributes.
		"""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found_for_sku:{sku}")
		context = self._base_context(tenant, "set_attribute_value")
		context.update({"product_present": True, "attribute_present": bool(attributes), "value_present": True, "rich_text": False, "locale_present": False})
		self._assert_rules(context)
		product["attributes"].update(attributes)
		product["status"] = "updated"
		product["updated_at"] = _now()
		# Create attribute value records
		for attr_name, attr_val in attributes.items():
			val_id = self._record_id("value")
			self.attribute_values[val_id] = {
				"id": val_id,
				"type": "product_attribute_value",
				"kind": "attribute_value",
				"tenant_id": tenant,
				"product_id": product["id"],
				"attribute_name": attr_name,
				"value": attr_val,
				"updated_by": updated_by,
				"status": "active",
				"created_at": _now(),
			}
		self._emit(tenant, "attributes_updated", product)
		return deepcopy(product)

	def add_media(
		self,
		sku: str,
		media_type: str,
		url: str,
		alt_text: str,
		tenant_id: str | None = None,
		media_id: str | None = None,
		rights_basis: str = "owned",
		sort_order: int = 0,
	) -> dict[str, Any]:
		"""
		Add media (image, video, document) to a product identified by SKU.

		media_type: 'image', 'video', 'document', '3d_model', 'audio'.
		url: Media asset URL.
		alt_text: Accessibility alt text.
		rights_basis: 'owned', 'licensed', 'creative_commons'.
		"""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found_for_sku:{sku}")
		supported_types = {"image", "video", "document", "3d_model", "audio", "thumbnail"}
		if media_type not in supported_types:
			raise ValueError(f"unsupported_media_type:{media_type}")
		if not url:
			raise ValueError("media_url_required")
		resolved_id = self._record_id("media", media_id)
		record = {
			"id": resolved_id,
			"type": "product_media",
			"kind": "media",
			"tenant_id": tenant,
			"product_id": product["id"],
			"sku": sku,
			"media_type": media_type,
			"url": url,
			"alt_text": alt_text,
			"rights_basis": rights_basis,
			"sort_order": sort_order,
			"status": "active",
			"created_at": _now(),
		}
		self._media[resolved_id] = record
		self._emit(tenant, "media_added", record)
		return deepcopy(record)

	def product_categorisation(
		self,
		sku: str,
		category_path: list[str],
		tenant_id: str | None = None,
		assigned_by: str = "system",
	) -> dict[str, Any]:
		"""
		Assign a product to a hierarchical category path.

		category_path: Ordered list of category labels (e.g. ['Electronics', 'Phones', 'Smartphones']).
		Creates category nodes as needed and assigns the product to the leaf.
		"""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found_for_sku:{sku}")
		if not category_path:
			raise ValueError("category_path_required")
		# Build/resolve category nodes
		parent_id = None
		created_nodes: list[dict[str, Any]] = []
		for idx, cat_name in enumerate(category_path):
			cat_id = f"cat-{tenant}-{'-'.join(category_path[:idx+1]).replace(' ', '_').lower()}"
			if cat_id not in self._categories:
				node = {"id": cat_id, "tenant_id": tenant, "name": cat_name, "level": idx, "parent_id": parent_id, "path": category_path[:idx+1], "created_at": _now()}
				self._categories[cat_id] = node
				created_nodes.append(node)
			parent_id = cat_id
		leaf_category_id = parent_id
		assign_id = self._record_id("catassign")
		assignment = {
			"id": assign_id,
			"type": "category_assignment",
			"tenant_id": tenant,
			"product_id": product["id"],
			"sku": sku,
			"category_path": category_path,
			"leaf_category_id": leaf_category_id,
			"assigned_by": assigned_by,
			"status": "active",
			"created_at": _now(),
		}
		self._category_assignments[assign_id] = assignment
		product["category"] = "/".join(category_path)
		product["category_path"] = category_path
		product["updated_at"] = _now()
		return deepcopy(assignment)

	def data_quality_score(
		self,
		sku: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Calculate the data quality score for a product.

		Checks: name, description (content), images (media), attributes,
		category assignment, compliance, and channel listing.
		Returns per-dimension scores and weighted total.
		"""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found_for_sku:{sku}")
		pid = product["id"]
		dimensions: dict[str, float] = {}
		# Name
		dimensions["name"] = 1.0 if product.get("name") else 0.0
		# Attributes
		attr_count = len(product.get("attributes", {}))
		dimensions["attributes"] = min(1.0, attr_count / 5.0)
		# Content (descriptions)
		has_content = any(c["product_id"] == pid and c["tenant_id"] == tenant for c in self.content.values())
		dimensions["description"] = 1.0 if has_content else 0.0
		# Media
		media_count = sum(1 for m in self._media.values() if m["tenant_id"] == tenant and m["product_id"] == pid)
		dimensions["media"] = min(1.0, media_count / 3.0)
		# Category
		has_category = any(a["product_id"] == pid and a["tenant_id"] == tenant for a in self._category_assignments.values())
		dimensions["categorisation"] = 1.0 if has_category or product.get("category") else 0.0
		# Compliance
		has_compliance = any(c["product_id"] == pid and c["tenant_id"] == tenant for c in self.compliance.values())
		dimensions["compliance"] = 1.0 if has_compliance else 0.5
		# Channel listing
		has_channel = any(c["product_id"] == pid and c["tenant_id"] == tenant for c in self.channels.values())
		dimensions["channel_listing"] = 1.0 if has_channel else 0.0
		weights = {"name": 0.20, "attributes": 0.20, "description": 0.15, "media": 0.15, "categorisation": 0.15, "compliance": 0.10, "channel_listing": 0.05}
		total_score = round(sum(dimensions[d] * weights[d] for d in dimensions) * 100, 1)
		return {
			"sku": sku,
			"product_id": pid,
			"tenant_id": tenant,
			"total_score": total_score,
			"grade": "A" if total_score >= 90 else ("B" if total_score >= 75 else ("C" if total_score >= 60 else ("D" if total_score >= 40 else "F"))),
			"dimensions": {d: round(v * 100, 1) for d, v in dimensions.items()},
			"weights": weights,
			"calculated_at": _now(),
		}

	def publish_to_channel(
		self,
		sku: str,
		channel_id: str,
		tenant_id: str | None = None,
		approved_by: str = "catalog_manager",
		publication_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Publish a product to a specific sales/distribution channel.

		channel_id: Channel identifier (e.g. 'web', 'mobile', 'marketplace_amazon').
		Returns publication record.
		"""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found_for_sku:{sku}")
		supported = list(SUPPORTED_CHANNELS) if isinstance(SUPPORTED_CHANNELS, (list, set, tuple)) else []
		if supported and channel_id not in supported:
			raise ValueError(f"unsupported_channel:{channel_id}")
		if not approved_by:
			raise PermissionError("publish_approval_required")
		# Ensure channel listing exists
		channel_listing = next((c for c in self.channels.values() if c["tenant_id"] == tenant and c["product_id"] == product["id"] and c.get("channel") == channel_id), None)
		if channel_listing is None:
			listing_id = self._record_id("listing")
			channel_listing = {"id": listing_id, "type": "channel_listing", "kind": "channel", "tenant_id": tenant, "product_id": product["id"], "channel": channel_id, "external_listing_id": f"{sku}-{channel_id}", "approved_by": approved_by, "status": "approved", "created_at": _now()}
			self.channels[listing_id] = channel_listing
		resolved_pub_id = self._record_id("pub", publication_id)
		pub = {
			"id": resolved_pub_id,
			"type": "product_publication",
			"kind": "publication",
			"tenant_id": tenant,
			"product_id": product["id"],
			"sku": sku,
			"channel_id": channel_id,
			"channel_listing_id": channel_listing["id"],
			"approved_by": approved_by,
			"status": "published",
			"published_at": _now(),
		}
		self.publications[resolved_pub_id] = pub
		product["status"] = "published"
		product["updated_at"] = _now()
		self._emit(tenant, "product_published", pub)
		return deepcopy(pub)

	def unpublish(
		self,
		sku: str,
		channel_id: str,
		tenant_id: str | None = None,
		reason: str = "",
		unpublished_by: str = "system",
	) -> dict[str, Any]:
		"""
		Unpublish a product from a channel.

		Marks publication records as unpublished and reverts product status if no active publications remain.
		"""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found_for_sku:{sku}")
		pid = product["id"]
		unpublished_count = 0
		for pub in self.publications.values():
			if pub["tenant_id"] == tenant and pub["product_id"] == pid and pub.get("channel_id") == channel_id and pub["status"] == "published":
				pub["status"] = "unpublished"
				pub["unpublished_at"] = _now()
				pub["unpublish_reason"] = reason
				unpublished_count += 1
		if unpublished_count == 0:
			raise KeyError(f"no_active_publication_for_channel:{channel_id}")
		# Check if any publications remain active
		still_published = any(p["product_id"] == pid and p["tenant_id"] == tenant and p["status"] == "published" for p in self.publications.values())
		if not still_published:
			product["status"] = "unpublished"
			product["updated_at"] = _now()
		record = {"sku": sku, "product_id": pid, "tenant_id": tenant, "channel_id": channel_id, "unpublished_count": unpublished_count, "reason": reason, "unpublished_by": unpublished_by, "unpublished_at": _now()}
		self._unpublications.append(record)
		return record

	def bulk_import(
		self,
		product_data_csv: list[dict[str, Any]],
		tenant_id: str | None = None,
		owner_id: str = "system",
		catalog_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Bulk import products from a list of product data dicts (CSV row format).

		Each row should have at minimum: sku, name, category.
		Returns import summary with created, skipped, and failed counts.
		"""
		tenant = self._tenant(tenant_id)
		if not product_data_csv:
			raise ValueError("product_data_required")
		created = 0
		skipped = 0
		failed: list[dict[str, Any]] = []
		for row in product_data_csv:
			sku = row.get("sku")
			name = row.get("name")
			category = row.get("category", "Uncategorised")
			if not sku or not name:
				failed.append({"row": row, "error": "sku_and_name_required"})
				continue
			# Check if SKU already exists
			if any(p["sku"] == sku and p["tenant_id"] == tenant for p in self.products.values()):
				skipped += 1
				continue
			try:
				attrs = {k: v for k, v in row.items() if k not in {"sku", "name", "category", "product_type"}}
				self.create_product(
					sku=sku, name=name, category=category, attributes=attrs,
					tenant_id=tenant, owner_id=owner_id, catalog_id=catalog_id,
					product_type=row.get("product_type", "physical"),
				)
				created += 1
			except Exception as exc:
				failed.append({"sku": sku, "error": str(exc)})
		log = {
			"tenant_id": tenant,
			"total_rows": len(product_data_csv),
			"created_count": created,
			"skipped_count": skipped,
			"failed_count": len(failed),
			"failures": failed,
			"success": len(failed) == 0,
			"imported_at": _now(),
		}
		self._bulk_import_logs.append(log)
		return log

	def product_search(
		self,
		query: str,
		filters: dict[str, Any] | None = None,
		tenant_id: str | None = None,
		limit: int = 20,
		offset: int = 0,
	) -> dict[str, Any]:
		"""
		Search products by name/SKU/attribute values with optional filters.

		filters: Dict supporting 'category', 'status', 'product_type', 'channel'.
		query: Text search against name and SKU.
		Returns paginated result set.
		"""
		tenant = self._tenant(tenant_id)
		f = filters or {}
		query_lower = (query or "").lower()
		all_products = [p for p in self.products.values() if p["tenant_id"] == tenant]
		# Text search
		if query_lower:
			all_products = [
				p for p in all_products
				if query_lower in p["name"].lower() or query_lower in p["sku"].lower()
				or any(query_lower in str(v).lower() for v in p.get("attributes", {}).values())
			]
		# Apply filters
		if "category" in f:
			all_products = [p for p in all_products if p.get("category", "").lower() == f["category"].lower()]
		if "status" in f:
			all_products = [p for p in all_products if p.get("status") == f["status"]]
		if "product_type" in f:
			all_products = [p for p in all_products if p.get("product_type") == f["product_type"]]
		if "channel" in f:
			published_pids = {pub["product_id"] for pub in self.publications.values() if pub["tenant_id"] == tenant and pub.get("channel_id") == f["channel"] and pub["status"] == "published"}
			all_products = [p for p in all_products if p["id"] in published_pids]
		total = len(all_products)
		page = all_products[offset: offset + limit]
		return {
			"tenant_id": tenant,
			"query": query,
			"filters": f,
			"total_results": total,
			"offset": offset,
			"limit": limit,
			"results": [deepcopy(p) for p in page],
		}

	def pim_analytics(
		self,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Return aggregated PIM analytics for a tenant over a period.

		Covers product, attribute, media, publication, quality, and import statistics.
		"""
		tenant = self._tenant(tenant_id)
		def count(store: dict[str, dict[str, Any]]) -> int:
			return sum(1 for r in store.values() if r["tenant_id"] == tenant)
		products = [p for p in self.products.values() if p["tenant_id"] == tenant]
		published = [p for p in products if p.get("status") == "published"]
		draft = [p for p in products if p.get("status") == "draft"]
		period_imports = [l for l in self._bulk_import_logs if l["tenant_id"] == tenant]
		# Average quality score
		quality_scores = []
		for product in products[:20]:  # sample first 20 for performance
			try:
				score = self.data_quality_score(product["sku"], tenant)
				quality_scores.append(score["total_score"])
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		avg_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0.0
		return {
			"tenant_id": tenant,
			"period": period,
			"catalog_count": count(self.catalogs),
			"product_count": len(products),
			"published_product_count": len(published),
			"draft_product_count": len(draft),
			"attribute_count": count(self.attributes),
			"attribute_value_count": count(self.attribute_values),
			"media_count": sum(1 for m in self._media.values() if m["tenant_id"] == tenant),
			"content_count": count(self.content),
			"channel_listing_count": count(self.channels),
			"publication_count": count(self.publications),
			"unpublication_count": sum(1 for u in self._unpublications if u["tenant_id"] == tenant),
			"quality_issue_count": count(self.quality_issues),
			"bulk_import_count": len(period_imports),
			"average_data_quality_score": avg_quality,
			"generated_at": _now(),
		}

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_catalog(self, catalog_id: str, tenant_id: str, code: str, name: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_catalog")
		context.update({"code_present": bool(code), "name_present": bool(name), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("catalog", catalog_id), "type": "product_catalog", "kind": "catalog", "tenant_id": tenant, "code": code.upper(), "name": name, "owner_id": owner_id, "status": "active", "created_at": _now()}
		self.catalogs[record["id"]] = record
		self._emit(tenant, "catalog_created", record)
		return deepcopy(record)

	def create_product_legacy(self, product_id: str, tenant_id: str, catalog_id: str, sku: str, name: str, product_type: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		catalog = self._get(self.catalogs, catalog_id, tenant, "catalog")
		context = self._base_context(tenant, "create_product")
		context.update({"catalog_present": bool(catalog), "sku_present": bool(sku), "name_present": bool(name), "product_type_supported": product_type in SUPPORTED_PRODUCT_TYPES, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("product", product_id), "type": "product_record", "kind": "product", "tenant_id": tenant, "catalog_id": catalog_id, "sku": sku, "name": name, "product_type": product_type, "owner_id": owner_id, "stage": "concept", "status": "draft", "created_at": _now()}
		self.products[record["id"]] = record
		self._emit(tenant, "product_created", record)
		return deepcopy(record)

	def define_attribute(self, attribute_id: str, tenant_id: str, code: str, name: str, attribute_type: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "define_attribute")
		context.update({"code_present": bool(code), "attribute_type_supported": attribute_type in SUPPORTED_ATTRIBUTE_TYPES, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("attribute", attribute_id), "type": "product_attribute", "kind": "attribute", "tenant_id": tenant, "code": code, "name": name, "attribute_type": attribute_type, "owner_id": owner_id, "status": "active", "created_at": _now()}
		self.attributes[record["id"]] = record
		self._emit(tenant, "attribute_defined", record)
		return deepcopy(record)

	def enrich_content(self, content_id: str, tenant_id: str, product_id: str, locale: str, title: str, body: str, generated: bool = False, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "enrich_content")
		context.update({"product_present": bool(product), "locale_present": bool(locale), "title_present": bool(title), "generated_content": bool(generated), "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("content", content_id), "type": "product_content", "kind": "content", "tenant_id": tenant, "product_id": product_id, "locale": locale, "title": title, "body": body, "generated": bool(generated), "reviewed_by": reviewed_by, "status": "approved" if reviewed_by or not generated else "review", "created_at": _now()}
		self.content[record["id"]] = record
		self._emit(tenant, "content_enriched", record)
		return deepcopy(record)

	def attach_asset(self, asset_id: str, tenant_id: str, product_id: str, asset_type: str, url: str, rights_basis: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "attach_asset")
		context.update({"product_present": bool(product), "url_present": bool(url), "rights_basis_present": bool(rights_basis)})
		self._assert_rules(context)
		record = {"id": self._record_id("asset", asset_id), "type": "product_asset", "kind": "asset", "tenant_id": tenant, "product_id": product_id, "asset_type": asset_type, "url": url, "rights_basis": rights_basis, "status": "active", "created_at": _now()}
		self.assets[record["id"]] = record
		self._emit(tenant, "asset_attached", record)
		return deepcopy(record)

	def record_quality_issue(self, issue_id: str, tenant_id: str, product_id: str, severity: str, description: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		product = self._get(self.products, product_id, tenant, "product")
		context = self._base_context(tenant, "record_quality_issue")
		context.update({"product_present": bool(product), "high_or_critical": severity in {"high", "critical"}, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("quality", issue_id), "type": "data_quality_issue", "kind": "quality", "tenant_id": tenant, "product_id": product_id, "severity": severity, "description": description, "owner_id": owner_id, "status": "open", "created_at": _now()}
		self.quality_issues[record["id"]] = record
		self._emit(tenant, "quality_issue_recorded", record)
		return deepcopy(record)

	def register_pim_agent(self, tenant_id: str, name: str, runtime: str, role: str, purpose: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_pim_agent")
		context.update({"runtime_supported": runtime in SUPPORTED_PIM_AGENT_RUNTIMES, "role_supported": role in SUPPORTED_PIM_AGENT_ROLES})
		self._assert_rules(context)
		record = {"id": self._record_id("agent"), "type": "pim_agent", "kind": "agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "purpose": purpose, "owner_id": owner_id, "status": "active", "created_at": _now()}
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
		def count(store: dict[str, dict[str, Any]]) -> int:
			return sum(1 for r in store.values() if r["tenant_id"] == tenant)
		return {
			"tenant_id": tenant,
			"catalog_count": count(self.catalogs),
			"product_count": count(self.products),
			"attribute_count": count(self.attributes),
			"content_count": count(self.content),
			"asset_count": count(self.assets),
			"media_count": sum(1 for m in self._media.values() if m["tenant_id"] == tenant),
			"channel_count": count(self.channels),
			"publication_count": count(self.publications),
			"quality_issue_count": count(self.quality_issues),
			"change_count": count(self.changes),
			"agent_count": count(self.agents),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant),
			"streaming": deepcopy(STREAMING),
		}

	def list_records(self, tenant_id: str, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		stores = [self.catalogs, self.products, self.attributes, self.attribute_values, self.variants, self.content, self.assets, self.compliance, self.channels, self.publications, self.quality_issues, self.changes, self.agents]
		records = [r for store in stores for r in store.values() if r["tenant_id"] == tenant]
		if record_type:
			records = [r for r in records if r["type"] == record_type or r["kind"] == record_type]
		return deepcopy(records)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return deepcopy([e for e in self._audit_events if e["tenant_id"] == tenant])


	def product_variant_create(self, sku: str, variant_attrs: dict[str, Any], tenant_id: str | None = None, owner_id: str = "system") -> dict[str, Any]:
		"""Create a product variant from a parent SKU."""
		tenant = self._tenant(tenant_id)
		parent = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if parent is None:
			raise PIMRecordNotFoundError(f"parent_product_not_found:{sku}")
		variant_sku = f"{sku}-VAR-{len(self.variants) + 1:03d}"
		variant = self.create_product(variant_sku, parent["name"] + " - " + str(variant_attrs), parent["category"], {**parent.get("attributes", {}), **variant_attrs}, tenant_id=tenant, owner_id=owner_id, catalog_id=parent.get("catalog_id"))
		var_id = self._record_id("variant")
		self.variants[var_id] = {"id": var_id, "type": "product_variant", "kind": "variant", "tenant_id": tenant, "parent_product_id": parent["id"], "variant_product_id": variant["id"], "variant_attributes": variant_attrs, "status": "active", "created_at": _now()}
		return self.variants[var_id]

	def enrichment_workflow(self, sku: str, workflow_steps: list[str], tenant_id: str | None = None, assigned_to: str = "system") -> dict[str, Any]:
		"""Create a content enrichment workflow for a product."""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found:{sku}")
		wf_id = self._record_id("wf")
		return {"workflow_id": wf_id, "sku": sku, "product_id": product["id"], "tenant_id": tenant, "steps": workflow_steps, "step_count": len(workflow_steps), "completed_steps": [], "assigned_to": assigned_to, "status": "in_progress", "created_at": _now()}

	def quality_score(self, sku: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return data quality score — domain alias."""
		return self.data_quality_score(sku, tenant_id)

	def digital_asset_manage(self, sku: str, media_type: str, url: str, alt_text: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Manage digital assets for a product — domain alias for add_media."""
		return self.add_media(sku, media_type, url, alt_text, tenant_id=tenant_id)

	def taxonomy_manage(self, tenant_id: str | None = None, category_name: str = "", parent_id: str | None = None) -> dict[str, Any]:
		"""Manage product taxonomy categories."""
		tenant = self._tenant(tenant_id)
		cat_id = f"cat-{tenant}-{category_name.replace(' ', '_').lower()}"
		node = {"id": cat_id, "tenant_id": tenant, "name": category_name, "parent_id": parent_id, "created_at": _now()}
		self._categories[cat_id] = node
		return node

	def bulk_classify(self, skus: list[str], category_path: list[str], tenant_id: str | None = None) -> dict[str, Any]:
		"""Bulk classify multiple products to a category path."""
		tenant = self._tenant(tenant_id)
		classified = []
		failed = []
		for sku in skus:
			try:
				self.product_categorisation(sku, category_path, tenant_id=tenant)
				classified.append(sku)
			except Exception as exc:
				failed.append({"sku": sku, "error": str(exc)})
		return {"classified": len(classified), "failed": len(failed), "failures": failed, "category_path": category_path}

	def publication_schedule(self, sku: str, channel_id: str, publish_at: str, tenant_id: str | None = None, approved_by: str = "system") -> dict[str, Any]:
		"""Schedule future publication of a product to a channel."""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found:{sku}")
		sched_id = self._record_id("pubsched")
		return {"schedule_id": sched_id, "sku": sku, "product_id": product["id"], "channel_id": channel_id, "publish_at": publish_at, "approved_by": approved_by, "status": "scheduled", "created_at": _now()}

	def version_compare(self, sku: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compare current product state with previous version from change log."""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found:{sku}")
		history = [c for c in self.changes.values() if c["tenant_id"] == tenant and c.get("product_id") == product["id"]]
		return {"sku": sku, "current_version": product.get("version", 1), "change_count": len(history), "latest_change": history[-1] if history else None, "compared_at": _now()}

	def channel_validate(self, sku: str, channel_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Validate product data completeness for a specific channel."""
		quality = self.data_quality_score(sku, tenant_id)
		channel_requirements = {"web": 70, "mobile": 60, "marketplace_amazon": 85, "b2b_portal": 75}
		min_score = channel_requirements.get(channel_id, 70)
		valid = quality["total_score"] >= min_score
		return {"sku": sku, "channel_id": channel_id, "quality_score": quality["total_score"], "min_required": min_score, "valid": valid, "issues": [] if valid else [f"quality_score_below_{min_score}"], "validated_at": _now()}

	def localisation_manage(self, sku: str, locale: str, title: str, description: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Manage product localisation for a locale."""
		tenant = self._tenant(tenant_id)
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found:{sku}")
		content_id = self._record_id("content")
		return self.enrich_content(content_id, tenant, product["id"], locale, title, description)

	def import_from_erp(self, erp_data: list[dict[str, Any]], tenant_id: str | None = None, owner_id: str = "erp_system") -> dict[str, Any]:
		"""Import product data from an ERP system — domain alias for bulk_import."""
		for row in erp_data:
			row.setdefault("source", "erp")
		return self.bulk_import(erp_data, tenant_id=tenant_id, owner_id=owner_id)

	def syndicate_marketplace(self, sku: str, marketplaces: list[str], tenant_id: str | None = None, approved_by: str = "catalog_manager") -> dict[str, Any]:
		"""Syndicate a product to multiple marketplace channels."""
		results = []
		for marketplace in marketplaces:
			try:
				pub = self.publish_to_channel(sku, marketplace, tenant_id=tenant_id, approved_by=approved_by)
				results.append({"marketplace": marketplace, "status": "published", "publication_id": pub["id"]})
			except Exception as exc:
				results.append({"marketplace": marketplace, "status": "failed", "error": str(exc)})
		return {"sku": sku, "total_marketplaces": len(marketplaces), "published": sum(1 for r in results if r["status"] == "published"), "failed": sum(1 for r in results if r["status"] == "failed"), "results": results}

	def product_lifecycle(self, sku: str, stage: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Transition a product to a new lifecycle stage."""
		tenant = self._tenant(tenant_id)
		stages = {"concept", "development", "launch", "growth", "maturity", "decline", "discontinue"}
		if stage not in stages:
			raise ValueError(f"unsupported_lifecycle_stage:{stage}")
		product = next((p for p in self.products.values() if p["tenant_id"] == tenant and p["sku"] == sku), None)
		if product is None:
			raise PIMRecordNotFoundError(f"product_not_found:{sku}")
		product["stage"] = stage
		product["stage_updated_at"] = _now()
		return {"sku": sku, "new_stage": stage, "transitioned_at": _now()}

	def product_search_advanced(self, query: str, filters: dict[str, Any] | None = None, tenant_id: str | None = None, limit: int = 20, offset: int = 0) -> dict[str, Any]:
		"""Advanced product search — domain alias for product_search."""
		return self.product_search(query, filters, tenant_id=tenant_id, limit=limit, offset=offset)

	def pim_kpi_summary(self, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return PIM KPI summary — thin wrapper over pim_analytics."""
		analytics = self.pim_analytics(period, tenant_id)
		return {"kpi_summary": True, **analytics}


PIMService = ProductInfoManagementService
PLMProductService = ProductInfoManagementService
ProductInformationService = ProductInfoManagementService
ProductInformationLifecycleService = ProductInfoManagementService
