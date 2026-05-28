"""Resource conflict regressions for composition dependency validation."""

from __future__ import annotations

from types import SimpleNamespace

from capabilities.composition.registry.validator import DependencyValidator, ValidationErrorType


class FakeRegistry:
	def __init__(self, capabilities):
		self.capabilities = capabilities

	def discover_all(self):
		return self.capabilities

	def get_capability(self, code):
		return self.capabilities.get(code)


def _capability(code: str, resources: dict, subcapabilities: dict | None = None):
	return SimpleNamespace(
		code=code,
		version="1.0.0",
		dependencies=[],
		configuration_schema={"resources": resources},
		subcapabilities=subcapabilities or {},
		composition_keywords=[],
		primary_interfaces=[],
	)


def test_validator_rejects_duplicate_runtime_resources_before_composition():
	registry = FakeRegistry({
		"ORDER": _capability(
			"ORDER",
			{
				"api_endpoints": [{"method": "POST", "path": "/api/orders", "port": 8080}],
				"ports": [9000],
				"services": ["order-api"],
				"file_paths": ["/var/apg/runtime/orders.sock"],
				"queues": ["orders.command"],
				"topics": ["orders.changed"],
				"environment_variables": {"APG_PAYMENT_MODE": {"default": "live"}},
			},
		),
		"BILLING": _capability(
			"BILLING",
			{
				"api_endpoints": [{"method": "POST", "path": "/api/orders", "port": "8080"}],
				"ports": ["9000"],
				"services": [{"name": "order-api"}],
				"file_paths": [{"path": "/var/apg/runtime/orders.sock"}],
				"queues": [{"queue": "orders.command"}],
				"topics": [{"topic": "orders.changed"}],
				"environment_variables": [{"name": "APG_PAYMENT_MODE", "default": "test"}],
			},
		),
	})
	validator = DependencyValidator(registry=registry)

	result = validator.validate_composition(["ORDER", "BILLING"])

	assert result.valid is False
	assert all(error.error_type is ValidationErrorType.RESOURCE_CONFLICT for error in result.errors)
	resource_types = {error.details["resource_type"] for error in result.errors}
	assert {
		"api_endpoint",
		"port",
		"service_name",
		"file_path",
		"queue",
		"topic",
		"environment_variable",
	}.issubset(resource_types)
	assert {conflict["resource_type"] for conflict in result.conflicts_found} == resource_types


def test_validator_checks_subcapability_resources_and_wildcard_routes():
	order_api = SimpleNamespace(
		code="api",
		database_tables=[],
		has_api=True,
		has_models=False,
		configuration_schema={
			"resources": {
				"api_endpoints": ["/api/invoices"],
				"ports": [7001],
			}
		},
	)
	invoice_api = SimpleNamespace(
		code="api",
		database_tables=[],
		has_api=True,
		has_models=False,
		configuration_schema={
			"resources": {
				"api_endpoints": [{"method": "GET", "path": "/api/invoices"}],
				"ports": [7001],
			}
		},
	)
	registry = FakeRegistry({
		"ORDER": _capability("ORDER", {}, {"api": order_api}),
		"INVOICE": _capability("INVOICE", {}, {"api": invoice_api}),
	})
	validator = DependencyValidator(registry=registry)

	result = validator.validate_composition(["ORDER", "INVOICE"])

	assert result.valid is False
	assert any(error.details["resource_type"] == "api_endpoint" for error in result.errors)
	assert any(error.details["resource_type"] == "port" for error in result.errors)
	assert any(error.subcapability == "api" for error in result.errors)
