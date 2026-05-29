"""I18N package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.i18n.service import I18nService
from capabilities.common.i18n.views import dashboard_model


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_materialized_contract_shape_is_valid():
	module = _load_module("materialized_contract_i18n", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "i18n"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_materialized_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_i18n", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "i18n" in model["capabilities"]


def test_i18n_package_compatibility_runtime_is_executable():
	service = I18nService()

	record = service.create_record(
		record_id="locale-compat",
		tenant_id="tenant-test",
		metadata={
			"locale_code": "en-US",
			"display_name": "English",
			"owner_id": "owner-test",
			"fallback_locale": "en-US",
		},
	)
	summary = service.dashboard_summary("tenant-test")
	model = dashboard_model(service, "tenant-test")

	assert record["kind"] == "locale"
	assert record["locale_code"] == "en-US"
	assert summary["locale_count"] == 1
	assert model["summary"]["locale_count"] == 1
