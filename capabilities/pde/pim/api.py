"""Dependency-light API helpers for Product Information Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import ProductInformationLifecycleService
except ImportError:  # pragma: no cover
	from service import ProductInformationLifecycleService  # type: ignore


_SERVICE = ProductInformationLifecycleService()


def service() -> ProductInformationLifecycleService:
	return _SERVICE


def create_catalog(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_catalog(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("code", ""), payload.get("name", ""), payload.get("owner_id", ""))


def create_product(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_product(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("catalog_id", ""), payload.get("sku", ""), payload.get("name", ""), payload.get("product_type", "physical"), payload.get("owner_id", ""))


def define_attribute(payload: dict[str, Any]) -> dict[str, Any]:
	return service().define_attribute(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("code", ""), payload.get("name", ""), payload.get("attribute_type", "text"), payload.get("owner_id", ""))


def enrich_content(payload: dict[str, Any]) -> dict[str, Any]:
	return service().enrich_content(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("product_id", ""), payload.get("locale", ""), payload.get("title", ""), payload.get("body", ""), payload.get("generated", False), payload.get("reviewed_by"))


def register_pim_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_pim_agent(payload.get("tenant_id", "default"), payload.get("name", "PIM Agent"), payload.get("runtime", "codex"), payload.get("role", "catalog_reviewer"), payload.get("purpose", "review product data"), payload.get("owner_id"))


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return service().dashboard_summary(tenant_id)
