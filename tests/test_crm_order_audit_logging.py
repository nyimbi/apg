"""Focused coverage for CRM order audit logging helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "capabilities" / "crm" / "ord" / "service.py"


def _load_order_service_module():
	package = types.ModuleType("crm_ord_testpkg")
	package.__path__ = [str(SERVICE_PATH.parent)]
	models = types.ModuleType("crm_ord_testpkg.models")

	for name in [
		"SOECustomer",
		"SOEShipToAddress",
		"SOESalesOrder",
		"SOEOrderLine",
		"SOEOrderCharge",
		"SOEPriceLevel",
		"SOEOrderTemplate",
		"SOEOrderTemplateLine",
		"SOEOrderSequence",
	]:
		setattr(models, name, type(name, (), {}))

	sys.modules["crm_ord_testpkg"] = package
	sys.modules["crm_ord_testpkg.models"] = models
	spec = importlib.util.spec_from_file_location("crm_ord_testpkg.service", SERVICE_PATH)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules["crm_ord_testpkg.service"] = module
	spec.loader.exec_module(module)
	return module


class FakeDB:
	def __init__(self):
		self.added = []

	def add(self, value):
		self.added.append(value)


def _audit_payload(internal_notes: str) -> dict:
	line = internal_notes.splitlines()[-1]
	assert line.startswith("AUDIT ")
	return json.loads(line.removeprefix("AUDIT "))


def test_order_audit_logging_appends_durable_internal_note():
	module = _load_order_service_module()
	db = FakeDB()
	service = module.OrderEntryService(db)
	order = SimpleNamespace(
		tenant_id="tenant-1",
		order_id="order-1",
		order_number="SO-001",
		status="DRAFT",
		total_amount=42,
		lines=[object(), object()],
		internal_notes="Existing note",
		updated_by_user_id=None,
		updated_at=None,
	)

	created_event = service._log_order_created(order, "user-1")
	created_payload = _audit_payload(order.internal_notes)

	assert created_event["event_type"] == "order_created"
	assert created_payload["details"]["line_count"] == 2
	assert created_payload["details"]["total_amount"] == "42"
	assert order.updated_by_user_id == "user-1"
	assert db.added == [order]

	unchanged_event = service._log_order_status_change(order, "DRAFT", "DRAFT", "user-1")
	assert unchanged_event is None
	assert len(db.added) == 1

	changed_event = service._log_order_status_change(order, "DRAFT", "SUBMITTED", "user-2")
	changed_payload = _audit_payload(order.internal_notes)

	assert changed_event["event_type"] == "order_status_changed"
	assert changed_payload["details"]["old_status"] == "DRAFT"
	assert changed_payload["details"]["new_status"] == "SUBMITTED"
	assert order.updated_by_user_id == "user-2"
	assert db.added == [order, order]
