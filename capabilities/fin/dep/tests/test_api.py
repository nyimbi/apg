"""API handler tests for fin.dep."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from capabilities.fin.dep import api


@pytest.fixture(autouse=True)
def fresh_service():
	"""Reset the API service singleton before each test."""
	from capabilities.fin.dep.service import DepositProductsService
	api._SERVICE = DepositProductsService()
	yield


TENANT = "api_test"

BASE_PRODUCT = {
	"tenant_id":      TENANT,
	"code":           "SAV-API",
	"name":           "API Savings",
	"product_type":   "SAVINGS",
	"currency":       "KES",
	"interest_config": {
		"rate":             "7.0",
		"calculation":      "DAILY_ACCRUAL",
		"compounding":      "MONTHLY",
		"withholding_rate": "15",
	},
	"fee_config": {
		"maintenance_fee":   "100",
		"fee_frequency":     "MONTHLY",
		"minimum_balance":   "500",
		"below_minimum_fee": "30",
	},
	"terms": {
		"min_opening_amount": "500",
		"break_penalty_rate": "0",
	},
}


def test_health():
	h = api.health()
	assert h["status"] == "ok"
	assert h["capability"] == "fin.dep"


def test_create_product():
	result = api.create_product(BASE_PRODUCT)
	assert result["code"] == "SAV-API"
	assert result["product_type"] == "SAVINGS"


def test_get_product():
	api.create_product(BASE_PRODUCT)
	p = api.get_product(TENANT, "SAV-API")
	assert p["name"] == "API Savings"


def test_list_products():
	api.create_product(BASE_PRODUCT)
	products = api.list_products({"tenant_id": TENANT, "active_only": True})
	assert len(products) == 1


def test_update_product():
	api.create_product(BASE_PRODUCT)
	updated = api.update_product({"tenant_id": TENANT, "code": "SAV-API", "name": "Updated Savings"})
	assert updated["name"] == "Updated Savings"


def test_deactivate_product():
	api.create_product(BASE_PRODUCT)
	p = api.deactivate_product(TENANT, "SAV-API")
	assert p["status"] == "INACTIVE"


def test_calculate_interest_roundtrip():
	api.create_product(BASE_PRODUCT)
	api.register_account({
		"tenant_id":    TENANT,
		"account_id":   "ACC-API-1",
		"product_code": "SAV-API",
		"balance":      "20000",
		"opening_date": "2025-01-01",
	})
	result = api.calculate_interest({
		"tenant_id":    TENANT,
		"account_id":   "ACC-API-1",
		"from_date":    "2025-01-01",
		"to_date":      "2025-03-31",
		"balance":      "20000",
		"product_code": "SAV-API",
	})
	assert Decimal(result["gross_interest"]) > Decimal("0")
	assert Decimal(result["withholding_tax"]) > Decimal("0")
	assert result["calculation_type"] == "DAILY_ACCRUAL"


def test_apply_interest():
	api.create_product(BASE_PRODUCT)
	api.register_account({
		"tenant_id":    TENANT,
		"account_id":   "ACC-API-2",
		"product_code": "SAV-API",
		"balance":      "10000",
		"opening_date": "2025-01-01",
	})
	posting = api.apply_interest({
		"tenant_id":       TENANT,
		"account_id":      "ACC-API-2",
		"interest_amount": "200",
		"value_date":      "2025-01-31",
		"posting_ref":     "API-POST-1",
	})
	assert "gl_ref" in posting
	assert Decimal(posting["net_interest"]) > Decimal("0")


def test_apply_maintenance_fee():
	api.create_product(BASE_PRODUCT)
	api.register_account({
		"tenant_id":    TENANT,
		"account_id":   "ACC-API-3",
		"product_code": "SAV-API",
		"balance":      "2000",
		"opening_date": "2025-01-01",
	})
	result = api.apply_maintenance_fee({
		"tenant_id":    TENANT,
		"account_id":   "ACC-API-3",
		"posting_date": "2025-01-31",
	})
	assert result["fee_amount"] == "100"


def test_check_minimum_balance():
	api.create_product(BASE_PRODUCT)
	api.register_account({
		"tenant_id":    TENANT,
		"account_id":   "ACC-API-4",
		"product_code": "SAV-API",
		"balance":      "300",  # below 500 minimum
		"opening_date": "2025-01-01",
	})
	check = api.check_minimum_balance(TENANT, "ACC-API-4")
	assert check["meets_minimum"] is False
	assert Decimal(check["shortfall"]) == Decimal("200")


def test_simulate_maturity():
	api.create_product(BASE_PRODUCT)
	result = api.simulate_maturity({
		"tenant_id":    TENANT,
		"product_code": "SAV-API",
		"principal":    "100000",
		"tenor_days":   365,
	})
	assert Decimal(result["maturity_amount"]) > Decimal("100000")
	assert result["effective_rate"]


def test_batch_accrue_interest():
	api.create_product(BASE_PRODUCT)
	api.register_account({
		"tenant_id":    TENANT,
		"account_id":   "ACC-BATCH-1",
		"product_code": "SAV-API",
		"balance":      "50000",
		"opening_date": "2025-01-01",
	})
	result = api.batch_accrue_interest({
		"tenant_id":    TENANT,
		"accrual_date": "2025-03-15",
	})
	assert result["accounts_processed"] >= 1
	assert result["idempotent_hit"] is False
	# Second run is idempotent
	result2 = api.batch_accrue_interest({
		"tenant_id":    TENANT,
		"accrual_date": "2025-03-15",
	})
	assert result2["idempotent_hit"] is True


def test_update_product_rate():
	api.create_product(BASE_PRODUCT)
	entry = api.update_product_rate({
		"tenant_id":      TENANT,
		"product_code":   "SAV-API",
		"new_rate":        "8.0",
		"effective_date": "2025-07-01",
		"reason":          "rate_review",
	})
	assert Decimal(entry["new_rate"]) == Decimal("8.0")


def test_get_rate_schedule():
	api.create_product(BASE_PRODUCT)
	api.update_product_rate({
		"tenant_id":      TENANT,
		"product_code":   "SAV-API",
		"new_rate":        "8.0",
		"effective_date": "2025-07-01",
		"reason":          "q3_review",
	})
	schedule = api.get_rate_schedule(TENANT, "SAV-API")
	assert len(schedule) == 2  # created + 1 update


def test_get_products_by_balance():
	api.create_product(BASE_PRODUCT)
	products = api.get_products_by_balance({
		"tenant_id": TENANT,
		"balance":   "5000",
		"currency":  "KES",
	})
	assert len(products) >= 1


def test_get_product_stats():
	api.create_product(BASE_PRODUCT)
	stats = api.get_product_stats(TENANT)
	assert stats["total_products"] == 1
	assert stats["active_products"] == 1


def test_withholding_tax_report():
	api.create_product(BASE_PRODUCT)
	api.register_account({
		"tenant_id":    TENANT,
		"account_id":   "ACC-WHT-1",
		"product_code": "SAV-API",
		"balance":      "10000",
		"opening_date": "2025-01-01",
	})
	api.apply_interest({
		"tenant_id":       TENANT,
		"account_id":      "ACC-WHT-1",
		"interest_amount": "500",
		"value_date":      "2025-03-31",
		"posting_ref":     "WHT-001",
	})
	report = api.get_withholding_tax_report(TENANT, "2025-03")
	assert len(report) >= 1
