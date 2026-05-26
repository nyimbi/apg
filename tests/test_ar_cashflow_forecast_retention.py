"""Focused regressions for AR cash-flow forecast retention."""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel


MODULE_PATH = (
	Path(__file__).resolve().parents[1]
	/ "capabilities"
	/ "fin"
	/ "arc"
	/ "accounts_receivable"
	/ "ai_cashflow_forecasting.py"
)


class _APGServiceBase:
	def __init__(self, tenant_id: str, user_id: str):
		self.tenant_id = tenant_id
		self.user_id = user_id

	async def _validate_permissions(self, permission: str) -> None:
		return None


class _AsyncFactory:
	@classmethod
	async def create(cls, **kwargs):
		return cls()


class _AuditLogger:
	def __init__(self, **kwargs):
		self.actions = []
		self.errors = []

	async def log_action(self, **kwargs):
		self.actions.append(kwargs)

	async def log_error(self, **kwargs):
		self.errors.append(kwargs)


def _install_fake_imports(monkeypatch):
	for package_name in (
		"capabilities",
		"capabilities.fin",
		"capabilities.fin.arc",
		"capabilities.fin.arc.accounts_receivable",
		"apg",
		"apg.core",
	):
		package = types.ModuleType(package_name)
		package.__path__ = []
		monkeypatch.setitem(sys.modules, package_name, package)

	base = types.ModuleType("apg.core.base")
	base.APGServiceBase = _APGServiceBase
	monkeypatch.setitem(sys.modules, "apg.core.base", base)

	models = types.ModuleType("apg.core.models")
	models.APGBaseModel = BaseModel
	monkeypatch.setitem(sys.modules, "apg.core.models", models)

	for module_name, class_name in (
		("apg.time_series_analytics", "TimeSeriesAnalyticsService"),
		("apg.ai_orchestration", "AIOrchestrationService"),
		("apg.notification_engine", "NotificationService"),
	):
		module = types.ModuleType(module_name)
		setattr(module, class_name, _AsyncFactory)
		monkeypatch.setitem(sys.modules, module_name, module)

	audit = types.ModuleType("apg.audit_compliance")
	audit.AuditLogger = _AuditLogger
	monkeypatch.setitem(sys.modules, "apg.audit_compliance", audit)

	ar_models = types.ModuleType("capabilities.fin.arc.accounts_receivable.models")
	for name in (
		"ARCustomer",
		"ARInvoice",
		"ARPayment",
		"ARCollectionActivity",
		"ARInvoiceStatus",
		"ARPaymentStatus",
		"ARCustomerStatus",
	):
		setattr(ar_models, name, type(name, (), {}))
	monkeypatch.setitem(sys.modules, ar_models.__name__, ar_models)


def _load_module(monkeypatch):
	_install_fake_imports(monkeypatch)
	spec = importlib.util.spec_from_file_location(
		"capabilities.fin.arc.accounts_receivable.ai_cashflow_forecasting",
		MODULE_PATH,
	)
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	monkeypatch.setitem(sys.modules, spec.name, module)
	spec.loader.exec_module(module)
	return module


def _data_point(module, day: int, amount: str):
	return module.CashFlowDataPoint(
		forecast_date=date(2026, 1, day),
		forecast_period=module.ForecastPeriodType.DAILY,
		total_cash_flow=Decimal(amount),
		expected_collections=Decimal(amount),
		confidence_interval_lower=Decimal("90.00"),
		confidence_interval_upper=Decimal("110.00"),
	)


def test_cashflow_forecast_store_returns_copy_and_tracks_performance(monkeypatch):
	module = _load_module(monkeypatch)
	service = module.APGCashFlowForecastingService(
		tenant_id="tenant-a",
		user_id="user-a",
		config=module.create_default_forecast_config(),
	)
	points = [_data_point(module, 1, "100.00")]
	summary = module.CashFlowForecastSummary(
		forecast_id="forecast-1",
		tenant_id="tenant-a",
		forecast_period_start=date(2026, 1, 1),
		forecast_period_end=date(2026, 1, 1),
		peak_collection_day=date(2026, 1, 1),
		overall_confidence_level=module.ForecastConfidenceLevel.HIGH,
		model_accuracy_score=0.9,
		confidence_score=0.95,
		forecast_volatility=0.1,
		next_update_due=datetime(2026, 1, 2),
	)

	import asyncio

	asyncio.run(service._store_forecast(summary, points))
	retrieved = asyncio.run(service._retrieve_forecast_by_id("forecast-1"))
	assert retrieved == points
	assert retrieved is not points

	retrieved[0].total_cash_flow = Decimal("1.00")
	assert asyncio.run(service._retrieve_forecast_by_id("forecast-1"))[0].total_cash_flow == Decimal("100.00")

	record = asyncio.run(service._update_model_performance_tracking({"accuracy": 0.91, "mape": 0.09}))
	assert record["tenant_id"] == "tenant-a"
	assert record["metrics"]["accuracy"] == 0.91
	assert isinstance(record["recorded_at"], datetime)
	assert service._model_performance_history == [record]


def test_monitor_forecast_accuracy_uses_stored_forecast(monkeypatch):
	module = _load_module(monkeypatch)
	service = module.APGCashFlowForecastingService(
		tenant_id="tenant-a",
		user_id="user-a",
		config=module.create_default_forecast_config(),
	)
	service.audit_logger = _AuditLogger()
	service._initialize_services = _AsyncFactory.create

	import asyncio

	asyncio.run(service._store_forecast(
		module.CashFlowForecastSummary(
			forecast_id="forecast-2",
			tenant_id="tenant-a",
			forecast_period_start=date(2026, 1, 1),
			forecast_period_end=date(2026, 1, 2),
			peak_collection_day=date(2026, 1, 1),
			overall_confidence_level=module.ForecastConfidenceLevel.HIGH,
			model_accuracy_score=0.9,
			confidence_score=0.95,
			forecast_volatility=0.1,
			next_update_due=datetime(2026, 1, 3),
		),
		[_data_point(module, 1, "100.00"), _data_point(module, 2, "200.00")],
	))

	report = asyncio.run(service.monitor_forecast_accuracy(
		"forecast-2",
		[_data_point(module, 1, "90.00"), _data_point(module, 2, "220.00")],
	))

	assert report["forecast_id"] == "forecast-2"
	assert report["accuracy_metrics"]["data_points"] == 2
	assert service._model_performance_history[-1]["metrics"] == report["accuracy_metrics"]
	assert service.audit_logger.actions[-1]["action"] == "monitor_forecast_accuracy"
