"""APG Deposit Products Engine (fin.dep).

Banking product factory and interest calculation engine for deposit accounts.
Supports CURRENT, SAVINGS, TERM_DEPOSIT, CALL_DEPOSIT, and NOTICE_DEPOSIT
products with tiered rates, daily accrual, compound interest, and WHT.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

CAPABILITY_META = {
	"capability_id":   "fin.dep",
	"name":            "Deposit Products Engine",
	"version":         "1.0.0",
	"description":     "Banking product factory and interest calculation engine for deposits",
	"category":        "fin",
	"subcapability":   "dep",
	"author":          "Nyimbi Odero",
	"company":         "Datacraft",
	"requires":        ["fin.glr", "common.auth_rbac", "common.audit_compliance"],
	"provides":        ["product_factory", "interest_engine", "fee_engine", "maturity_engine"],
	"event_streams":   ["apg.fin.dep.interest", "apg.fin.dep.maturity", "apg.fin.dep.fee"],
	"api_prefix":      "/api/fin/dep",
	"menu_category":   "Finance / Deposits",
	"menu_icon":       "fa-piggy-bank",
	"db_table_prefix": "dep_",
}

try:
	from .service import DepositProductsService
	from .models import (
		DepositProduct, ProductType, InterestCalculationType, CompoundingFrequency,
		InterestConfig, FeeConfig, ProductTerms, InterestCalculationResult,
		MaturityInstruction, BatchAccrualResult, SimulationResult,
	)
	from .api import service
except Exception:  # pragma: no cover
	pass


def get_capability_info() -> dict:
	return CAPABILITY_META
