"""Import regressions for billing payment processors."""

from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = (
	Path(__file__).resolve().parents[1]
	/ "capabilities"
	/ "fin"
	/ "bil"
	/ "payment_processors.py"
)


def _install_fake_package(monkeypatch: pytest.MonkeyPatch) -> None:
	for package_name in ("capabilities", "capabilities.fin", "capabilities.fin.bil"):
		package = types.ModuleType(package_name)
		package.__path__ = []
		monkeypatch.setitem(sys.modules, package_name, package)

	models = types.ModuleType("capabilities.fin.bil.models")
	models.BLPayment = object
	models.PaymentStatus = types.SimpleNamespace()
	models.BillingCurrency = types.SimpleNamespace()
	monkeypatch.setitem(sys.modules, "capabilities.fin.bil.models", models)


def test_payment_processors_import_without_gateway_sdks(monkeypatch):
	_install_fake_package(monkeypatch)

	original_import = builtins.__import__

	def guarded_import(name, *args, **kwargs):
		if name in {"stripe", "aiohttp"}:
			raise ImportError(name)
		return original_import(name, *args, **kwargs)

	monkeypatch.setattr(builtins, "__import__", guarded_import)

	spec = importlib.util.spec_from_file_location(
		"capabilities.fin.bil.payment_processors",
		MODULE_PATH,
	)
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	monkeypatch.setitem(sys.modules, spec.name, module)
	spec.loader.exec_module(module)

	assert module.stripe is None
	assert module.aiohttp is None

	with pytest.raises(module.PaymentProcessorError, match="Stripe SDK"):
		module.StripePaymentProcessor("sk_test", "whsec_test")

	with pytest.raises(module.PaymentProcessorError, match="aiohttp"):
		module.PayPalPaymentProcessor("client", "secret")


def test_payment_processor_manager_imports_through_billing_package():
	from capabilities.fin.bil.payment_processors import PaymentProcessorManager

	assert PaymentProcessorManager.__name__ == "PaymentProcessorManager"
