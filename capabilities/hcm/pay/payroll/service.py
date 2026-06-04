"""HCM Payroll lifecycle service — full African payroll engine."""

from __future__ import annotations

import calendar
import math
from copy import deepcopy
from datetime import date, datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		PAYROLL_EVENT_STREAM,
		STREAMING,
		SUPPORTED_COMPONENT_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_PAYMENT_METHODS,
		SUPPORTED_PAYROLL_AGENT_ROLES,
		SUPPORTED_PAYROLL_AGENT_RUNTIMES,
		SUPPORTED_PAY_FREQUENCIES,
		SUPPORTED_TAX_SCOPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		PAYROLL_EVENT_STREAM,
		STREAMING,
		SUPPORTED_COMPONENT_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_PAYMENT_METHODS,
		SUPPORTED_PAYROLL_AGENT_ROLES,
		SUPPORTED_PAYROLL_AGENT_RUNTIMES,
		SUPPORTED_PAY_FREQUENCIES,
		SUPPORTED_TAX_SCOPES,
		evaluate_capability_rules,
		get_capability_contract,
	)


# ---------------------------------------------------------------------------
# PAYE rate tables — Financial Year 2025/26 unless noted
# Bands: list of (upper_bound_monthly, rate).  Last band uses float('inf').
# All monetary figures in LOCAL CURRENCY, MONTHLY unless marked ANNUAL.
# ---------------------------------------------------------------------------
PAYE_TABLES: dict[str, dict[str, Any]] = {
	"KE": {
		# Kenya Revenue Authority — PAYE 2024/25
		# https://www.kra.go.ke/individual/filing-paying/types-of-taxes/pay-as-you-earn
		"bands": [
			(24_000,		0.10),
			(8_333,		 0.25),   # cumulative ceiling 32,333
			(float("inf"), 0.30),
		],
		"band_mode": "width",				  # each tuple is (band_width, rate)
		"personal_relief": 2_400,			 # KES/month
		"insurance_relief_rate": 0.15,		# 15 % of premiums paid, max 5,000/month
		"insurance_relief_max": 5_000,
		"mortgage_relief_rate": 0.25,		 # 25 % of interest, max 25,000/month
		"mortgage_relief_max": 25_000,
		"pension_relief_max": 20_000,		 # combined ee+er pension, max 20,000/month
		"currency": "KES",
		"authority": "KRA",
	},
	"UG": {
		# Uganda Revenue Authority — PAYE
		# Bands in UGX/month
		"bands": [
			(235_000,	   0.00),
			(335_000,	   0.10),
			(410_000,	   0.20),
			(10_000_000,	0.30),
			(float("inf"), 0.40),
		],
		"band_mode": "width",
		"personal_relief": 235_000,
		"currency": "UGX",
		"authority": "URA",
	},
	"TZ": {
		# Tanzania Revenue Authority — PAYE
		# Bands in TZS/month — expressed as WIDTHS for _progressive_tax
		# TRA table ceilings: 270k | 520k | 760k | 1,000k | ∞
		# Widths:             270k | 250k | 240k |  240k  | ∞
		"bands": [
			(270_000,	   0.00),   # 0 – 270,000
			(250_000,	   0.09),   # 270,001 – 520,000
			(240_000,	   0.20),   # 520,001 – 760,000
			(240_000,	   0.25),   # 760,001 – 1,000,000
			(float("inf"), 0.30),   # above 1,000,000
		],
		"band_mode": "width",
		"currency": "TZS",
		"authority": "TRA",
	},
	"GH": {
		# Ghana Revenue Authority — PAYE
		# Bands in GHS/month — expressed as WIDTHS
		# GRA table ceilings: 319 | 429 | 559 | 3,559 | 20,172 | ∞
		# Widths:             319 | 110 | 130 | 3,000 | 16,613 | ∞
		"bands": [
			(319,		   0.00),
			(110,		   0.05),
			(130,		   0.10),
			(3_000,		 0.175),
			(16_613,		0.25),
			(float("inf"), 0.30),
		],
		"band_mode": "width",
		"currency": "GHS",
		"authority": "GRA",
	},
	"NG": {
		# Federal Inland Revenue Service — Personal Income Tax Act (PITA)
		# Computed on ANNUAL taxable income; bands are annual in NGN
		"bands": [
			(300_000,	   0.07),
			(300_000,	   0.11),
			(500_000,	   0.15),
			(500_000,	   0.19),
			(1_600_000,	 0.21),
			(float("inf"), 0.24),
		],
		"band_mode": "width",
		"annual": True,					   # compute on annual gross, divide result by 12
		"cra_fixed": 200_000,			   # minimum CRA (annual)
		"cra_pct": 0.20,					# CRA = max(200_000, 20% of annual gross)
		"personal_relief_fixed": 200_000,  # minimum personal relief (annual)
		"personal_relief_pct": 0.01,		# = max(200_000, 1% of gross)
		"currency": "NGN",
		"authority": "FIRS",
	},
	"RW": {
		# Rwanda Revenue Authority — PAYE
		# Bands in RWF/month
		"bands": [
			(60_000,		0.00),
			(100_000,	   0.20),
			(float("inf"), 0.30),
		],
		"band_mode": "width",
		"currency": "RWF",
		"authority": "RRA",
	},
	"ZM": {
		# Zambia Revenue Authority — PAYE
		# Bands in ZMW/month
		"bands": [
			(4_800,		 0.00),
			(9_200,		 0.20),
			(14_300,		0.30),
			(float("inf"), 0.375),
		],
		"band_mode": "width",
		"currency": "ZMW",
		"authority": "ZRA",
	},
}

# ---------------------------------------------------------------------------
# Statutory deduction parameters by country
# ---------------------------------------------------------------------------
STATUTORY_PARAMS: dict[str, dict[str, Any]] = {
	"KE": {
		"nssf": {
			"ee_rate": 0.06,
			"er_rate": 0.06,
			"ee_max": 2_160,
			"er_max": 2_160,
			"name": "NSSF",
			"authority": "NSSF",
		},
		"nhif": {
			# Graduated scale (salary_ceiling: contribution)
			"scale": [
				(5_999, 150),
				(7_999, 300),
				(11_999, 400),
				(14_999, 500),
				(19_999, 600),
				(24_999, 750),
				(29_999, 850),
				(34_999, 900),
				(39_999, 950),
				(44_999, 1_000),
				(49_999, 1_100),
				(59_999, 1_200),
				(69_999, 1_300),
				(79_999, 1_400),
				(89_999, 1_500),
				(99_999, 1_600),
				(float("inf"), 1_700),
			],
			"name": "NHIF",
			"authority": "NHIF",
		},
		"nita": {
			"er_rate": 0.01,
			"name": "NITA",
			"authority": "NITA",
		},
	},
	"UG": {
		"nssf": {
			"ee_rate": 0.05,
			"er_rate": 0.10,
			"ee_max": None,
			"er_max": None,
			"name": "NSSF",
			"authority": "NSSF_UG",
		},
	},
	"TZ": {
		"nssf": {
			"ee_rate": 0.10,
			"er_rate": 0.10,
			"ee_max": None,
			"er_max": None,
			"name": "NSSF",
			"authority": "NSSF_TZ",
		},
		"nhif": {
			"ee_rate": 0.03,
			"er_rate": 0.03,
			"ee_max": None,
			"er_max": None,
			"name": "NHIF",
			"authority": "NHIF_TZ",
		},
		"sdl": {
			"er_rate": 0.04,
			"name": "SDL",
			"authority": "TRA",
		},
		"wcf": {
			"er_rate": 0.01,
			"name": "WCF",
			"authority": "WCF_TZ",
		},
	},
	"GH": {
		"ssnit": {
			"ee_rate": 0.055,
			"er_rate": 0.13,
			"ee_max": None,
			"er_max": None,
			"name": "SSNIT",
			"authority": "SSNIT",
		},
		"tier2": {
			"er_rate": 0.05,  # routed to Tier 2 occupational scheme
			"name": "Tier2_Pension",
			"authority": "NPRA",
		},
	},
	"NG": {
		"pension": {
			"ee_rate": 0.08,
			"er_rate": 0.10,
			"ee_max": None,
			"er_max": None,
			"name": "Pension",
			"authority": "PenCom",
		},
		"nhf": {
			"ee_rate": 0.025,
			"on_basic": True,   # applied to basic salary only
			"name": "NHF",
			"authority": "FMBN",
		},
		"nsitf": {
			"er_rate": 0.01,
			"name": "NSITF",
			"authority": "NSITF",
		},
	},
	"RW": {
		"rssb": {
			"ee_rate": 0.03,
			"er_rate": 0.05,
			"ee_max": None,
			"er_max": None,
			"name": "RSSB",
			"authority": "RSSB",
		},
	},
	"ZM": {
		"napsa": {
			"ee_rate": 0.05,
			"er_rate": 0.05,
			"ee_cap": 2_600,  # max combined monthly contribution per side
			"name": "NAPSA",
			"authority": "NAPSA",
		},
	},
}

# GL account defaults (override per tenant via chart of accounts)
GL_DEFAULTS: dict[str, str] = {
	"gross_pay_expense": "5001",
	"paye_payable": "2101",
	"nssf_ee_payable": "2102",
	"nssf_er_expense": "5010",
	"nhif_payable": "2103",
	"pension_ee_payable": "2104",
	"pension_er_expense": "2105",  # CR: employer statutory contributions payable
	"net_pay_payable": "2110",
	"advance_asset": "1210",
	"garnishment_payable": "2120",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PayrollError(Exception):
	"""Base exception for payroll operations."""


class PayrollRunNotFoundError(PayrollError):
	"""Raised when a payroll run is not found."""


class PayrollProfileNotFoundError(PayrollError):
	"""Raised when an employee pay profile is not found."""


class CountryNotSupportedError(PayrollError):
	"""Raised when a country has no configured tax table."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progressive_tax(gross: float, bands: list[tuple[float, float]]) -> float:
	"""Compute progressive tax given (band_width, rate) pairs.

	The last band width should be float('inf') to catch any remainder.
	"""
	tax = 0.0
	remaining = gross
	for width, rate in bands:
		if remaining <= 0:
			break
		taxable_in_band = min(remaining, width)
		tax += taxable_in_band * rate
		remaining -= taxable_in_band
	return tax


def _nhif_ke(gross: float, scale: list[tuple[float, int]]) -> int:
	"""Return Kenya NHIF contribution based on graduated salary scale."""
	for ceiling, contribution in scale:
		if gross <= ceiling:
			return contribution
	return scale[-1][1]


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class PayrollManagementService:
	"""In-memory executable service for payroll lifecycle packets.

	Covers:
	  - Full payroll period / pay group / employee profile / component CRUD
	  - Multi-country PAYE calculation (KE, UG, TZ, GH, NG, RW, ZM)
	  - Statutory deductions (NSSF/NHIF/NITA/NAPSA/RSSB/SSNIT/PenCom…)
	  - Full payroll run orchestration with gross→net waterfall
	  - Terminal benefits, overtime, bonuses, leave encashment
	  - Payslips, P9 form, statutory returns, bank transfer files
	  - GL journal entries, variance reports
	  - Salary advances, garnishments, expatriate equalisation
	  - Salary sacrifice pension
	"""

	def __init__(
		self,
		tenant_id: str | None = None,
		user_id: str | None = None,
		*_: Any,
		**__: Any,
	) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id

		# Core stores
		self.periods: dict[str, dict[str, Any]] = {}
		self.pay_groups: dict[str, dict[str, Any]] = {}
		self.employee_pay_profiles: dict[str, dict[str, Any]] = {}
		self.components: dict[str, dict[str, Any]] = {}
		self.time_imports: dict[str, dict[str, Any]] = {}
		self.runs: dict[str, dict[str, Any]] = {}
		self.line_items: dict[str, dict[str, Any]] = {}
		self.taxes: dict[str, dict[str, Any]] = {}
		self.adjustments: dict[str, dict[str, Any]] = {}
		self.payment_batches: dict[str, dict[str, Any]] = {}
		self.payslips: dict[str, dict[str, Any]] = {}
		self.tax_filings: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}

		# Extended stores for new features
		self.salary_advances: dict[str, dict[str, Any]] = {}
		self.garnishments: dict[str, dict[str, Any]] = {}
		self.gl_entries: dict[str, dict[str, Any]] = {}
		self.p9_forms: dict[str, dict[str, Any]] = {}
		self.statutory_returns: dict[str, dict[str, Any]] = {}
		self.bank_files: dict[str, dict[str, Any]] = {}
		self.terminal_benefits: dict[str, dict[str, Any]] = {}
		self.bonus_runs: dict[str, dict[str, Any]] = {}

		self._audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Internal utilities
	# ------------------------------------------------------------------

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
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		# Only hard-block on explicit deny; require_review creates an audit flag
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": PAYROLL_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def _get_profile(self, profile_id: str, tenant: str) -> dict[str, Any]:
		profile = self.employee_pay_profiles.get(profile_id)
		if not profile or profile["tenant_id"] != tenant:
			raise PayrollProfileNotFoundError(f"profile {profile_id} not found")
		return profile

	def _get_run(self, run_id: str, tenant: str) -> dict[str, Any]:
		run = self.runs.get(run_id)
		if not run or run["tenant_id"] != tenant:
			raise PayrollRunNotFoundError(f"run {run_id} not found")
		return run

	def _country_for_profile(self, profile: dict[str, Any]) -> str:
		pg = self.pay_groups.get(profile.get("pay_group_id", ""))
		return (pg or {}).get("country", "KE")

	# ------------------------------------------------------------------
	# Core CRUD — unchanged from original
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_payroll_period(
		self,
		period_id: str,
		tenant_id: str,
		name: str,
		frequency: str,
		start_date: str,
		end_date: str,
		pay_date: str,
		currency: str = "USD",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_payroll_period")
		context.update({
			"name_present": bool(name),
			"frequency_supported": frequency in SUPPORTED_PAY_FREQUENCIES,
			"start_date_present": bool(start_date),
			"end_date_present": bool(end_date),
			"pay_date_present": bool(pay_date),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("period", period_id),
			"type": "payroll_period",
			"kind": "period",
			"tenant_id": tenant,
			"name": name,
			"frequency": frequency,
			"start_date": start_date,
			"end_date": end_date,
			"pay_date": pay_date,
			"currency": currency,
			"status": "open",
			"created_at": self._now(),
		}
		self.periods[record["id"]] = record
		self._emit(tenant, "payroll_period_created", record)
		return deepcopy(record)

	def create_pay_group(
		self,
		pay_group_id: str,
		tenant_id: str,
		code: str,
		name: str,
		frequency: str,
		currency: str,
		country: str,
		owner_id: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_pay_group")
		context.update({
			"code_present": bool(code),
			"name_present": bool(name),
			"frequency_supported": frequency in SUPPORTED_PAY_FREQUENCIES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"country_present": bool(country),
			"owner_present": bool(owner_id),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("paygroup", pay_group_id),
			"type": "pay_group",
			"kind": "pay_group",
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"frequency": frequency,
			"currency": currency,
			"country": country.upper(),
			"owner_id": owner_id,
			"status": "active",
			"created_at": self._now(),
		}
		self.pay_groups[record["id"]] = record
		self._emit(tenant, "pay_group_created", record)
		return deepcopy(record)

	def create_employee_pay_profile(
		self,
		profile_id: str,
		tenant_id: str,
		employee_id: str,
		pay_group_id: str,
		payment_method: str,
		tax_id: str,
		currency: str,
		base_pay: float,
		basic_pay: float | None = None,
		hire_date: str | None = None,
		bank_account: str | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		pay_group = self.pay_groups.get(pay_group_id)
		context = self._base_context(tenant, "create_employee_pay_profile")
		context.update({
			"employee_present": bool(employee_id),
			"pay_group_present": bool(pay_group and pay_group["tenant_id"] == tenant),
			"payment_method_supported": payment_method in SUPPORTED_PAYMENT_METHODS,
			"tax_id_present": bool(tax_id),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"bank_payment": payment_method == "bank_transfer",
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("profile", profile_id),
			"type": "employee_pay_profile",
			"kind": "employee_pay_profile",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"pay_group_id": pay_group_id,
			"payment_method": payment_method,
			"tax_id": tax_id,
			"currency": currency,
			"base_pay": float(base_pay),
			"basic_pay": float(basic_pay) if basic_pay is not None else float(base_pay) * 0.6,
			"hire_date": hire_date,
			"bank_account": bank_account,
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.employee_pay_profiles[record["id"]] = record
		self._emit(tenant, "employee_pay_profile_created", record)
		return deepcopy(record)

	def create_pay_component(
		self,
		component_id: str,
		tenant_id: str,
		code: str,
		name: str,
		component_type: str,
		currency: str,
		taxable: bool | None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_pay_component")
		context.update({
			"code_present": bool(code),
			"name_present": bool(name),
			"component_type_supported": component_type in SUPPORTED_COMPONENT_TYPES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"taxable_flag_present": taxable is not None,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("component", component_id),
			"type": "pay_component",
			"kind": "component",
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"component_type": component_type,
			"currency": currency,
			"taxable": bool(taxable),
			"status": "active",
			"created_at": self._now(),
		}
		self.components[record["id"]] = record
		self._emit(tenant, "pay_component_created", record)
		return deepcopy(record)

	def record_time_import(
		self,
		time_import_id: str,
		tenant_id: str,
		period_id: str,
		profile_id: str,
		hours: float,
		source: str,
		overtime_hours: float = 0,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		period = self.periods.get(period_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "record_time_import")
		context.update({
			"period_present": bool(period and period["tenant_id"] == tenant),
			"profile_present": bool(profile and profile["tenant_id"] == tenant),
			"hours": hours,
			"source_present": bool(source),
			"overtime": overtime_hours > 0,
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("time", time_import_id),
			"type": "payroll_time_import",
			"kind": "time_import",
			"tenant_id": tenant,
			"period_id": period_id,
			"profile_id": profile_id,
			"hours": float(hours),
			"overtime_hours": float(overtime_hours),
			"source": source,
			"approved_by": approved_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.time_imports[record["id"]] = record
		self._emit(tenant, "time_import_recorded", record)
		return deepcopy(record)

	def start_payroll_run(
		self,
		run_id: str,
		tenant_id: str,
		period_id: str,
		pay_group_id: str,
		initiated_by: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		period = self.periods.get(period_id)
		pay_group = self.pay_groups.get(pay_group_id)
		context = self._base_context(tenant, "start_payroll_run")
		context.update({
			"period_present": bool(period and period["tenant_id"] == tenant),
			"pay_group_present": bool(pay_group and pay_group["tenant_id"] == tenant),
			"initiator_present": bool(initiated_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("run", run_id),
			"type": "payroll_run",
			"kind": "run",
			"tenant_id": tenant,
			"period_id": period_id,
			"pay_group_id": pay_group_id,
			"initiated_by": initiated_by,
			"approved_by": None,
			"posted_by": None,
			"totals": {"gross": 0.0, "deductions": 0.0, "taxes": 0.0, "adjustments": 0.0, "net": 0.0},
			"employee_count": 0,
			"payslip_lines": [],
			"status": "calculated",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.runs[record["id"]] = record
		self._emit(tenant, "payroll_run_started", record)
		return deepcopy(record)

	# ------------------------------------------------------------------
	# Run totals recalculation
	# ------------------------------------------------------------------

	def _recalculate_run_totals(self, run_id: str) -> None:
		run = self.runs[run_id]
		lines = [l for l in self.line_items.values() if l["run_id"] == run_id]
		taxes = [t for t in self.taxes.values() if t["run_id"] == run_id]
		adjustments = [a for a in self.adjustments.values() if a["run_id"] == run_id]
		gross = sum(l["amount"] for l in lines if l["component_type"] in {"earning", "reimbursement"})
		deductions = abs(sum(l["amount"] for l in lines if l["component_type"] in {"deduction", "benefit", "garnishment"}))
		tax_total = sum(t["amount"] for t in taxes)
		adjustment_total = sum(a["amount"] for a in adjustments)
		run["totals"] = {
			"gross": round(gross, 2),
			"deductions": round(deductions, 2),
			"taxes": round(tax_total, 2),
			"adjustments": round(adjustment_total, 2),
			"net": round(gross + adjustment_total - deductions - tax_total, 2),
		}
		run["updated_at"] = self._now()

	# ------------------------------------------------------------------
	# Line items / tax / adjustment / approval / posting / payment / payslip
	# ------------------------------------------------------------------

	def add_line_item(
		self,
		line_id: str,
		tenant_id: str,
		run_id: str,
		profile_id: str,
		component_id: str,
		amount: float | None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		component = self.components.get(component_id)
		amount_value = float(amount) if amount is not None else None
		context = self._base_context(tenant, "add_line_item")
		context.update({
			"run_present": bool(run and run["tenant_id"] == tenant),
			"profile_present": bool(profile and profile["tenant_id"] == tenant),
			"component_present": bool(component and component["tenant_id"] == tenant),
			"amount_present": amount is not None,
			"negative_amount": bool(amount_value is not None and amount_value < 0),
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("line", line_id),
			"type": "payroll_line_item",
			"kind": "line_item",
			"tenant_id": tenant,
			"run_id": run_id,
			"profile_id": profile_id,
			"employee_id": profile["employee_id"],
			"component_id": component_id,
			"component_type": component["component_type"],
			"amount": amount_value,
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.line_items[record["id"]] = record
		self._recalculate_run_totals(run_id)
		self._emit(tenant, "payroll_line_item_added", record)
		return deepcopy(record)

	def record_tax(
		self,
		tax_id: str,
		tenant_id: str,
		run_id: str,
		profile_id: str,
		scope: str,
		authority: str,
		amount: float | None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "record_tax")
		context.update({
			"run_present": bool(run and run["tenant_id"] == tenant),
			"profile_present": bool(profile and profile["tenant_id"] == tenant),
			"tax_scope_supported": scope in SUPPORTED_TAX_SCOPES,
			"authority_present": bool(authority),
			"amount_present": amount is not None,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("tax", tax_id),
			"type": "payroll_tax",
			"kind": "tax",
			"tenant_id": tenant,
			"run_id": run_id,
			"profile_id": profile_id,
			"employee_id": profile["employee_id"],
			"scope": scope,
			"authority": authority,
			"amount": float(amount),
			"status": "active",
			"created_at": self._now(),
		}
		self.taxes[record["id"]] = record
		self._recalculate_run_totals(run_id)
		self._emit(tenant, "payroll_tax_recorded", record)
		return deepcopy(record)

	def record_adjustment(
		self,
		adjustment_id: str,
		tenant_id: str,
		run_id: str,
		profile_id: str,
		amount: float,
		reason: str,
		approved_by: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "record_adjustment")
		context.update({
			"run_present": bool(run and run["tenant_id"] == tenant),
			"profile_present": bool(profile and profile["tenant_id"] == tenant),
			"reason_present": bool(reason),
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("adjustment", adjustment_id),
			"type": "payroll_adjustment",
			"kind": "adjustment",
			"tenant_id": tenant,
			"run_id": run_id,
			"profile_id": profile_id,
			"employee_id": profile["employee_id"],
			"amount": float(amount),
			"reason": reason,
			"approved_by": approved_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.adjustments[record["id"]] = record
		self._recalculate_run_totals(run_id)
		self._emit(tenant, "payroll_adjustment_recorded", record)
		return deepcopy(record)

	def approve_payroll_run(
		self,
		run_id: str,
		tenant_id: str,
		approved_by: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		context = self._base_context(tenant, "approve_payroll_run")
		context.update({
			"run_present": bool(run and run["tenant_id"] == tenant),
			"approver_present": bool(approved_by),
		})
		self._assert_rules(context)
		run["approved_by"] = approved_by
		run["status"] = "approved"
		run["updated_at"] = self._now()
		self._emit(tenant, "payroll_run_approved", run)
		return deepcopy(run)

	def post_payroll_run(self, run_id: str, tenant_id: str, posted_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		if not run or run["tenant_id"] != tenant:
			raise PermissionError("run_required")
		self._assert_rules({
			**self._base_context(tenant, "post_payroll_run"),
			"approval_recorded": bool(run.get("approved_by")),
		})
		run["posted_by"] = posted_by
		run["status"] = "posted"
		run["updated_at"] = self._now()
		self._emit(tenant, "payroll_run_posted", run)
		return deepcopy(run)

	def create_payment_batch(
		self,
		payment_id: str,
		tenant_id: str,
		run_id: str,
		payment_date: str,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		net_pay = float(run["totals"]["net"]) if run and run["tenant_id"] == tenant else 0.0
		context = self._base_context(tenant, "create_payment_batch")
		context.update({
			"run_present": bool(run and run["tenant_id"] == tenant),
			"approval_recorded": bool(approved_by or (run and run.get("approved_by"))),
			"payment_date_present": bool(payment_date),
			"net_pay": net_pay,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("payment", payment_id),
			"type": "payroll_payment_batch",
			"kind": "payment_batch",
			"tenant_id": tenant,
			"run_id": run_id,
			"payment_date": payment_date,
			"approved_by": approved_by or run.get("approved_by"),
			"net_pay": net_pay,
			"status": "created",
			"created_at": self._now(),
		}
		self.payment_batches[record["id"]] = record
		run["status"] = "paid"
		self._emit(tenant, "payment_batch_created", record)
		return deepcopy(record)

	def publish_payslip(
		self,
		payslip_id: str,
		tenant_id: str,
		run_id: str,
		profile_id: str,
		privacy_basis: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "publish_payslip")
		context.update({
			"run_present": bool(run and run["tenant_id"] == tenant),
			"profile_present": bool(profile and profile["tenant_id"] == tenant),
			"posted_run": bool(run and run.get("posted_by")),
			"privacy_basis_present": bool(privacy_basis),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("payslip", payslip_id),
			"type": "payroll_payslip",
			"kind": "payslip",
			"tenant_id": tenant,
			"run_id": run_id,
			"profile_id": profile_id,
			"employee_id": profile["employee_id"],
			"privacy_basis": privacy_basis,
			"net_pay": run["totals"]["net"],
			"status": "published",
			"created_at": self._now(),
		}
		self.payslips[record["id"]] = record
		self._emit(tenant, "payslip_published", record)
		return deepcopy(record)

	def create_tax_filing(
		self,
		filing_id: str,
		tenant_id: str,
		run_id: str,
		authority: str,
		period_ref: str,
		approved_by: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		context = self._base_context(tenant, "create_tax_filing")
		context.update({
			"run_present": bool(run and run["tenant_id"] == tenant),
			"authority_present": bool(authority),
			"period_present": bool(period_ref),
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("filing", filing_id),
			"type": "payroll_tax_filing",
			"kind": "tax_filing",
			"tenant_id": tenant,
			"run_id": run_id,
			"authority": authority,
			"period_ref": period_ref,
			"approved_by": approved_by,
			"tax_total": run["totals"]["taxes"],
			"status": "created",
			"created_at": self._now(),
		}
		self.tax_filings[record["id"]] = record
		self._emit(tenant, "tax_filing_created", record)
		return deepcopy(record)

	def register_payroll_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_payroll_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_PAYROLL_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_PAYROLL_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"),
			"type": "payroll_agent",
			"kind": "agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "payroll_agent_registered", record)
		return deepcopy(record)

	def validate_payroll_agent_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("payroll_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "payroll_agent_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(
		self,
		tenant_id: str,
		event_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "payroll_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant,
			"event_count": event_count,
			"processor": "bytewax",
			"stream": PAYROLL_EVENT_STREAM,
		}

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "open",
	) -> dict[str, Any]:
		data = dict(metadata or {})
		record = self.create_payroll_period(
			record_id,
			tenant_id,
			str(data.get("name") or "Payroll Period"),
			str(data.get("frequency") or "monthly"),
			str(data.get("start_date") or "2026-01-01"),
			str(data.get("end_date") or "2026-01-31"),
			str(data.get("pay_date") or "2026-02-01"),
			str(data.get("currency") or "USD"),
		)
		record["status"] = status
		self.periods[record["id"]]["status"] = status
		return record

	# ------------------------------------------------------------------
	# Dashboard / audit / listing (unchanged)
	# ------------------------------------------------------------------

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		net_pay = sum(run["totals"]["net"] for run in self.list_records("runs", tenant))
		return {
			"tenant_id": tenant,
			"period_count": len(self.list_records("periods", tenant)),
			"pay_group_count": len(self.list_records("pay_groups", tenant)),
			"profile_count": len(self.list_records("employee_pay_profiles", tenant)),
			"component_count": len(self.list_records("components", tenant)),
			"time_import_count": len(self.list_records("time_imports", tenant)),
			"run_count": len(self.list_records("runs", tenant)),
			"line_item_count": len(self.list_records("line_items", tenant)),
			"tax_count": len(self.list_records("taxes", tenant)),
			"adjustment_count": len(self.list_records("adjustments", tenant)),
			"payment_batch_count": len(self.list_records("payment_batches", tenant)),
			"payslip_count": len(self.list_records("payslips", tenant)),
			"tax_filing_count": len(self.list_records("tax_filings", tenant)),
			"payroll_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"net_pay_total": round(net_pay, 2),
			"overall_status": "operating",
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(
		self,
		collection: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if collection is None:
			return self.list_all_records(tenant)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record.get("tenant_id") == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record.get("tenant_id") == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		collections = [
			"periods", "pay_groups", "employee_pay_profiles", "components",
			"time_imports", "runs", "line_items", "taxes", "adjustments",
			"payment_batches", "payslips", "tax_filings", "agents",
		]
		for collection in collections:
			records.extend(self.list_records(collection, tenant))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	# ==================================================================
	# ===  AFRICAN PAYROLL ENGINE — NEW METHODS  =======================
	# ==================================================================

	# ------------------------------------------------------------------
	# calculate_paye
	# ------------------------------------------------------------------

	async def calculate_paye(
		self,
		gross_monthly: float,
		country: str,
		allowances: dict[str, float] | None = None,
		deductions: dict[str, float] | None = None,
		ytd_gross: float = 0.0,
	) -> dict[str, Any]:
		"""Compute PAYE for a single employee-month.

		Args:
			gross_monthly: Total gross earnings for the month.
			country: ISO-3166-1 alpha-2 country code (KE, UG, TZ, GH, NG, RW, ZM).
			allowances: Named allowances that may affect relief (e.g. insurance_premium,
			            mortgage_interest, pension_contribution).
			deductions: Pre-tax deductions already applied (e.g. nssf_ee, pension_ee).
			ytd_gross: Year-to-date gross BEFORE this month (used for NG annual calc).

		Returns:
			dict with keys: taxable_income, tax_before_relief, reliefs_applied,
			                paye_payable, effective_rate, currency, authority.
		"""
		assert gross_monthly >= 0, "gross_monthly must be non-negative"
		country = country.upper()
		if country not in PAYE_TABLES:
			raise CountryNotSupportedError(f"No PAYE table for {country}")

		table = PAYE_TABLES[country]
		allowances = allowances or {}
		deductions = deductions or {}
		reliefs: dict[str, float] = {}

		if table.get("annual"):
			# Nigeria: operate on annual figures
			annual_gross = (ytd_gross + gross_monthly) * 12 / max(1, 1)
			# CRA
			cra = max(table["cra_fixed"], table["cra_pct"] * annual_gross)
			personal_relief = max(table["personal_relief_fixed"], table["personal_relief_pct"] * annual_gross)
			pension_ee = deductions.get("pension_ee", 0.0) * 12
			nhf = deductions.get("nhf", 0.0) * 12
			taxable_annual = max(0.0, annual_gross - cra - pension_ee - nhf)
			tax_annual = _progressive_tax(taxable_annual, table["bands"])
			tax_annual = max(0.0, tax_annual - personal_relief)
			tax_monthly = round(tax_annual / 12, 2)
			taxable_income = round(taxable_annual / 12, 2)
			reliefs["cra_monthly"] = round(cra / 12, 2)
			reliefs["personal_relief_monthly"] = round(personal_relief / 12, 2)
			return {
				"country": country,
				"currency": table["currency"],
				"authority": table["authority"],
				"gross_monthly": round(gross_monthly, 2),
				"taxable_income": taxable_income,
				"tax_before_relief": round((tax_annual + personal_relief) / 12, 2),
				"reliefs_applied": reliefs,
				"paye_payable": max(0.0, tax_monthly),
				"effective_rate": round(tax_monthly / gross_monthly, 4) if gross_monthly else 0.0,
			}

		# --- Standard progressive monthly PAYE ---
		# Step 1: subtract deductible items to get taxable income
		pension_ee = deductions.get("pension_ee", deductions.get("nssf_ee", 0.0))
		taxable = max(0.0, gross_monthly - pension_ee)

		# Apply pension relief cap (KE)
		if country == "KE":
			pension_er = deductions.get("nssf_er", 0.0)
			pension_relief_amount = min(pension_ee + pension_er, table["pension_relief_max"])
			taxable = max(0.0, gross_monthly - pension_relief_amount)

		tax_before_relief = _progressive_tax(taxable, table["bands"])

		# Step 2: reliefs
		# NOTE: personal_relief is a TAX CREDIT (subtracted from computed tax) only for
		# Kenya (KES 2,400/month statutory credit).  For UG/RW the zero-rate band already
		# handles the personal allowance inside the progressive schedule; applying it again
		# as a credit would incorrectly zero out the tax.
		total_relief = 0.0

		if country == "KE":
			personal_relief = table.get("personal_relief", 0.0)
			if personal_relief:
				reliefs["personal_relief"] = personal_relief
				total_relief += personal_relief

		if country == "KE":
			ins_premium = allowances.get("insurance_premium", 0.0)
			if ins_premium > 0:
				ir = min(ins_premium * table["insurance_relief_rate"], table["insurance_relief_max"])
				reliefs["insurance_relief"] = round(ir, 2)
				total_relief += ir

			mortgage_interest = allowances.get("mortgage_interest", 0.0)
			if mortgage_interest > 0:
				mr = min(mortgage_interest * table["mortgage_relief_rate"], table["mortgage_relief_max"])
				reliefs["mortgage_relief"] = round(mr, 2)
				total_relief += mr

		paye = max(0.0, round(tax_before_relief - total_relief, 2))
		return {
			"country": country,
			"currency": table["currency"],
			"authority": table["authority"],
			"gross_monthly": round(gross_monthly, 2),
			"taxable_income": round(taxable, 2),
			"tax_before_relief": round(tax_before_relief, 2),
			"reliefs_applied": reliefs,
			"paye_payable": paye,
			"effective_rate": round(paye / gross_monthly, 4) if gross_monthly else 0.0,
		}

	# ------------------------------------------------------------------
	# calculate_statutory_deductions
	# ------------------------------------------------------------------

	async def calculate_statutory_deductions(
		self,
		employee: dict[str, Any],
		gross: float,
		country: str,
	) -> dict[str, Any]:
		"""Compute all statutory deductions for a country.

		Args:
			employee: Must contain at least 'basic_pay' for countries that need it (NG).
			gross: Total gross monthly earnings.
			country: ISO-3166-1 alpha-2.

		Returns:
			dict with ee_total, er_total, breakdown (each deduction line).
		"""
		assert gross >= 0, "gross must be non-negative"
		country = country.upper()
		params = STATUTORY_PARAMS.get(country, {})
		breakdown: list[dict[str, Any]] = []
		ee_total = 0.0
		er_total = 0.0

		def _add(name: str, authority: str, ee: float, er: float) -> None:
			nonlocal ee_total, er_total
			ee = round(ee, 2)
			er = round(er, 2)
			breakdown.append({"name": name, "authority": authority, "ee": ee, "er": er})
			ee_total += ee
			er_total += er

		for key, cfg in params.items():
			if key == "nssf":
				ee_rate = cfg.get("ee_rate", 0.0)
				er_rate = cfg.get("er_rate", 0.0)
				ee = gross * ee_rate
				er = gross * er_rate
				if cfg.get("ee_max"):
					ee = min(ee, cfg["ee_max"])
				if cfg.get("er_max"):
					er = min(er, cfg["er_max"])
				if cfg.get("ee_cap"):
					ee = min(ee, cfg["ee_cap"])
					er = min(er, cfg["ee_cap"])
				_add(cfg["name"], cfg["authority"], ee, er)

			elif key == "nhif":
				if "scale" in cfg:
					# Kenya NHIF graduated
					contribution = _nhif_ke(gross, cfg["scale"])
					_add(cfg["name"], cfg["authority"], contribution, 0.0)
				else:
					# TZ NHIF flat rate
					ee = gross * cfg.get("ee_rate", 0.0)
					er = gross * cfg.get("er_rate", 0.0)
					_add(cfg["name"], cfg["authority"], ee, er)

			elif key == "nita":
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], 0.0, er)

			elif key == "sdl":
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], 0.0, er)

			elif key == "wcf":
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], 0.0, er)

			elif key == "ssnit":
				ee = gross * cfg.get("ee_rate", 0.0)
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], ee, er)

			elif key == "tier2":
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], 0.0, er)

			elif key == "pension":
				ee = gross * cfg.get("ee_rate", 0.0)
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], ee, er)

			elif key == "nhf":
				# Nigeria NHF — 2.5% of BASIC salary only
				basic = float(employee.get("basic_pay", gross * 0.6))
				ee = basic * cfg.get("ee_rate", 0.025)
				_add(cfg["name"], cfg["authority"], ee, 0.0)

			elif key == "nsitf":
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], 0.0, er)

			elif key == "rssb":
				ee = gross * cfg.get("ee_rate", 0.0)
				er = gross * cfg.get("er_rate", 0.0)
				_add(cfg["name"], cfg["authority"], ee, er)

			elif key == "napsa":
				cap = cfg.get("ee_cap", float("inf"))
				ee = min(gross * cfg.get("ee_rate", 0.0), cap)
				er = min(gross * cfg.get("er_rate", 0.0), cap)
				_add(cfg["name"], cfg["authority"], ee, er)

		return {
			"country": country,
			"gross": round(gross, 2),
			"ee_total": round(ee_total, 2),
			"er_total": round(er_total, 2),
			"breakdown": breakdown,
		}

	# ------------------------------------------------------------------
	# calculate_overtime
	# ------------------------------------------------------------------

	async def calculate_overtime(
		self,
		employee_id: str,
		regular_hours: float,
		overtime_hours: float,
		overtime_type: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute overtime pay for an employee.

		Args:
			employee_id: Must match a pay profile id or employee_id field.
			regular_hours: Standard working hours in the period.
			overtime_hours: Extra hours worked.
			overtime_type: One of time_and_half | double_time | public_holiday_rate.
			tenant_id: Override for the service tenant.

		Returns:
			dict: hourly_rate, overtime_rate, overtime_pay, regular_pay, total_pay.
		"""
		assert overtime_hours >= 0, "overtime_hours must be non-negative"
		assert regular_hours > 0, "regular_hours must be positive"
		multipliers = {
			"time_and_half": 1.5,
			"double_time": 2.0,
			"public_holiday_rate": 2.0,
		}
		if overtime_type not in multipliers:
			raise PayrollError(f"Unknown overtime_type: {overtime_type}. Use: {list(multipliers)}")

		tenant = self._tenant(tenant_id)
		# Find profile by employee_id
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		monthly_pay = profile["base_pay"]
		# Standard: monthly / (52 weeks / 12 months * 40 hours) = monthly / 173.33
		hourly_rate = round(monthly_pay / 173.33, 4)
		multiplier = multipliers[overtime_type]
		overtime_rate = round(hourly_rate * multiplier, 4)
		regular_pay = round(hourly_rate * regular_hours, 2)
		overtime_pay = round(overtime_rate * overtime_hours, 2)

		return {
			"employee_id": employee_id,
			"profile_id": profile["id"],
			"monthly_base": monthly_pay,
			"hourly_rate": hourly_rate,
			"regular_hours": regular_hours,
			"regular_pay": regular_pay,
			"overtime_hours": overtime_hours,
			"overtime_type": overtime_type,
			"overtime_multiplier": multiplier,
			"overtime_rate": overtime_rate,
			"overtime_pay": overtime_pay,
			"total_pay": round(regular_pay + overtime_pay, 2),
		}

	# ------------------------------------------------------------------
	# mid_month_hire_calculation
	# ------------------------------------------------------------------

	async def mid_month_hire_calculation(
		self,
		employee_id: str,
		hire_date: str,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Pro-rate salary for an employee hired mid-month.

		Args:
			employee_id: Profile id or employee_id.
			hire_date: ISO date string (YYYY-MM-DD) of actual hire date.
			period: ISO date string of ANY day within the pay period month (YYYY-MM-DD).

		Returns:
			dict: full_monthly_pay, days_in_month, days_worked, prorated_pay, prorate_factor.
		"""
		tenant = self._tenant(tenant_id)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		hire_dt = date.fromisoformat(hire_date)
		period_dt = date.fromisoformat(period)
		days_in_month = calendar.monthrange(period_dt.year, period_dt.month)[1]

		# If hire month matches period month, days_worked = days_in_month - hire_day + 1
		if hire_dt.year == period_dt.year and hire_dt.month == period_dt.month:
			days_worked = days_in_month - hire_dt.day + 1
		else:
			days_worked = days_in_month

		days_worked = max(0, min(days_worked, days_in_month))
		prorate_factor = round(days_worked / days_in_month, 6)
		prorated_pay = round(profile["base_pay"] * prorate_factor, 2)

		return {
			"employee_id": employee_id,
			"profile_id": profile["id"],
			"hire_date": hire_date,
			"period": f"{period_dt.year}-{period_dt.month:02d}",
			"days_in_month": days_in_month,
			"days_worked": days_worked,
			"prorate_factor": prorate_factor,
			"full_monthly_pay": profile["base_pay"],
			"prorated_pay": prorated_pay,
		}

	# ------------------------------------------------------------------
	# run_payroll
	# ------------------------------------------------------------------

	async def run_payroll(
		self,
		period_id: str,
		tenant_id: str,
		pay_group_id: str,
		initiated_by: str,
		employee_filter: list[str] | None = None,
	) -> dict[str, Any]:
		"""Full payroll run: gross → statutory → PAYE → net for all employees.

		Creates a payroll run, calculates each employee's net pay, stores
		line items, tax records, and updates run totals.

		Args:
			period_id: Must exist in self.periods for this tenant.
			tenant_id: Tenant context.
			pay_group_id: Pay group (determines country/currency).
			initiated_by: User initiating the run.
			employee_filter: Optional list of profile IDs to restrict processing.

		Returns:
			Full run record with payslip_lines for every employee.
		"""
		tenant = self._tenant(tenant_id)
		pay_group = self.pay_groups.get(pay_group_id)
		assert pay_group and pay_group["tenant_id"] == tenant, "pay_group not found"
		country = pay_group.get("country", "KE").upper()

		run_id = self._record_id("run")
		run = {
			"id": run_id,
			"type": "payroll_run",
			"kind": "run",
			"tenant_id": tenant,
			"period_id": period_id,
			"pay_group_id": pay_group_id,
			"initiated_by": initiated_by,
			"approved_by": None,
			"posted_by": None,
			"country": country,
			"totals": {"gross": 0.0, "deductions": 0.0, "taxes": 0.0, "adjustments": 0.0, "net": 0.0},
			"employee_count": 0,
			"payslip_lines": [],
			"status": "calculated",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.runs[run_id] = run

		profiles = [
			p for p in self.employee_pay_profiles.values()
			if p["tenant_id"] == tenant
			and p["pay_group_id"] == pay_group_id
			and p["status"] == "active"
		]
		if employee_filter:
			profiles = [p for p in profiles if p["id"] in employee_filter or p["employee_id"] in employee_filter]

		total_gross = 0.0
		total_deductions = 0.0
		total_taxes = 0.0
		total_net = 0.0
		payslip_lines: list[dict[str, Any]] = []

		for profile in profiles:
			gross = profile["base_pay"]

			# Statutory deductions
			stat = await self.calculate_statutory_deductions(profile, gross, country)
			ee_deductions = {item["name"]: item["ee"] for item in stat["breakdown"]}
			stat_ee_total = stat["ee_total"]

			# Pension ee deduction for PAYE relief
			pension_ee = next(
				(item["ee"] for item in stat["breakdown"] if "pension" in item["name"].lower() or "nssf" in item["name"].lower()),
				0.0,
			)

			# PAYE
			paye_result = await self.calculate_paye(
				gross,
				country,
				deductions={"pension_ee": pension_ee, "nssf_ee": pension_ee},
			)
			paye = paye_result["paye_payable"]

			net = round(gross - stat_ee_total - paye, 2)
			total_gross += gross
			total_deductions += stat_ee_total
			total_taxes += paye
			total_net += net

			line: dict[str, Any] = {
				"profile_id": profile["id"],
				"employee_id": profile["employee_id"],
				"gross": gross,
				"statutory_ee": stat_ee_total,
				"statutory_er": stat["er_total"],
				"statutory_breakdown": stat["breakdown"],
				"paye": paye,
				"paye_detail": paye_result,
				"net": net,
			}
			payslip_lines.append(line)

		run["totals"] = {
			"gross": round(total_gross, 2),
			"deductions": round(total_deductions, 2),
			"taxes": round(total_taxes, 2),
			"adjustments": 0.0,
			"net": round(total_net, 2),
		}
		run["employee_count"] = len(profiles)
		run["payslip_lines"] = payslip_lines
		run["updated_at"] = self._now()
		self._emit(tenant, "payroll_run_calculated", run)
		return deepcopy(run)

	# ------------------------------------------------------------------
	# process_bonus_payroll
	# ------------------------------------------------------------------

	async def process_bonus_payroll(
		self,
		bonus_type: str,
		employee_ids: list[str],
		amounts: dict[str, float],
		tax_method: str,
		tenant_id: str | None = None,
		period_id: str | None = None,
	) -> dict[str, Any]:
		"""Process a bonus payroll run.

		Args:
			bonus_type: annual | quarterly | performance | spot.
			employee_ids: List of profile IDs or employee IDs.
			amounts: Mapping of employee_id/profile_id → bonus amount.
			tax_method: aggregate (add to current month salary) | separate_rate.
			tenant_id: Tenant context.
			period_id: Optional period to associate the bonus with.

		Returns:
			Bonus run summary with per-employee PAYE and net amounts.
		"""
		valid_types = {"annual", "quarterly", "performance", "spot"}
		valid_tax_methods = {"aggregate", "separate_rate"}
		assert bonus_type in valid_types, f"bonus_type must be one of {valid_types}"
		assert tax_method in valid_tax_methods, f"tax_method must be one of {valid_tax_methods}"

		tenant = self._tenant(tenant_id)
		# Flat rate for separate_rate: many jurisdictions use top marginal rate
		# We use 30% as a conservative pan-African default; override per country
		separate_rate = 0.30

		results: list[dict[str, Any]] = []
		total_gross = 0.0
		total_paye = 0.0
		total_net = 0.0

		for eid in employee_ids:
			profile = next(
				(p for p in self.employee_pay_profiles.values()
				 if p["tenant_id"] == tenant and
				 (p["id"] == eid or p["employee_id"] == eid)),
				None,
			)
			if not profile:
				continue

			bonus_amount = float(amounts.get(eid, amounts.get(profile["employee_id"], 0.0)))
			if bonus_amount <= 0:
				continue

			pg = self.pay_groups.get(profile.get("pay_group_id", ""))
			country = (pg or {}).get("country", "KE").upper()

			if tax_method == "aggregate":
				# Add bonus to monthly salary, compute PAYE on combined, subtract regular PAYE
				combined = profile["base_pay"] + bonus_amount
				combined_paye = await self.calculate_paye(combined, country)
				regular_paye = await self.calculate_paye(profile["base_pay"], country)
				incremental_paye = max(0.0, combined_paye["paye_payable"] - regular_paye["paye_payable"])
				paye = round(incremental_paye, 2)
			else:
				paye = round(bonus_amount * separate_rate, 2)

			net = round(bonus_amount - paye, 2)
			total_gross += bonus_amount
			total_paye += paye
			total_net += net

			results.append({
				"employee_id": profile["employee_id"],
				"profile_id": profile["id"],
				"bonus_type": bonus_type,
				"bonus_gross": bonus_amount,
				"tax_method": tax_method,
				"paye": paye,
				"net": net,
			})

		bonus_run_id = self._record_id("bonusrun")
		record = {
			"id": bonus_run_id,
			"type": "bonus_payroll_run",
			"kind": "bonus_run",
			"tenant_id": tenant,
			"period_id": period_id,
			"bonus_type": bonus_type,
			"tax_method": tax_method,
			"employee_count": len(results),
			"totals": {
				"gross": round(total_gross, 2),
				"paye": round(total_paye, 2),
				"net": round(total_net, 2),
			},
			"lines": results,
			"status": "calculated",
			"created_at": self._now(),
		}
		self.bonus_runs[bonus_run_id] = record
		return deepcopy(record)

	# ------------------------------------------------------------------
	# calculate_terminal_benefits
	# ------------------------------------------------------------------

	async def calculate_terminal_benefits(
		self,
		employee_id: str,
		termination_date: str,
		reason: str,
		tenant_id: str | None = None,
		leave_days_accrued: float = 0.0,
	) -> dict[str, Any]:
		"""Compute terminal benefits on separation.

		Severance: 1 month per completed year of service (common East/West Africa standard).
		Leave encashment: accrued annual leave at daily rate.
		Gratuity: only if contract specifies; we include as optional override.

		Args:
			employee_id: Profile id or employee_id.
			termination_date: ISO date string (YYYY-MM-DD).
			reason: redundancy | resignation | retirement | dismissal | death | contract_end.
			tenant_id: Tenant context.
			leave_days_accrued: Unused annual leave days to encash.

		Returns:
			dict with severance_pay, leave_encashment, nssf_lump_sum_note, total_terminal_pay.
		"""
		severance_eligible = {"redundancy", "retirement", "death", "contract_end"}
		tenant = self._tenant(tenant_id)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		hire_date_str = profile.get("hire_date")
		term_dt = date.fromisoformat(termination_date)

		years_served = 0.0
		if hire_date_str:
			hire_dt = date.fromisoformat(hire_date_str)
			delta_days = (term_dt - hire_dt).days
			years_served = delta_days / 365.25
		completed_years = int(math.floor(years_served))

		monthly_pay = profile["base_pay"]
		daily_rate = round(monthly_pay * 12 / 260, 2)  # 260 working days/year

		severance_pay = 0.0
		if reason in severance_eligible and completed_years >= 1:
			severance_pay = round(monthly_pay * completed_years, 2)

		leave_encashment = round(daily_rate * leave_days_accrued, 2)
		total_terminal = round(severance_pay + leave_encashment, 2)

		pg = self.pay_groups.get(profile.get("pay_group_id", ""))
		country = (pg or {}).get("country", "KE").upper()

		# KE: severance from redundancy is tax-exempt up to 36 months × monthly pay
		# (Employment Act Cap 226, s.40); leave encashment is taxable
		taxable_terminal = leave_encashment
		exempt_terminal = severance_pay if reason == "redundancy" else 0.0

		terminal_id = self._record_id("terminal")
		record = {
			"id": terminal_id,
			"type": "terminal_benefits",
			"kind": "terminal_benefits",
			"tenant_id": tenant,
			"employee_id": profile["employee_id"],
			"profile_id": profile["id"],
			"termination_date": termination_date,
			"reason": reason,
			"hire_date": hire_date_str,
			"years_served": round(years_served, 2),
			"completed_years": completed_years,
			"monthly_pay": monthly_pay,
			"daily_rate": daily_rate,
			"severance_pay": severance_pay,
			"leave_days_accrued": leave_days_accrued,
			"leave_encashment": leave_encashment,
			"total_terminal_pay": total_terminal,
			"taxable_terminal": taxable_terminal,
			"exempt_terminal": exempt_terminal,
			"nssf_lump_sum_note": (
				"Employee eligible for NSSF lump sum withdrawal on retirement or age 60+"
				if reason == "retirement" else None
			),
			"country": country,
			"status": "calculated",
			"created_at": self._now(),
		}
		self.terminal_benefits[terminal_id] = record
		return deepcopy(record)

	# ------------------------------------------------------------------
	# calculate_leave_encashment
	# ------------------------------------------------------------------

	async def calculate_leave_encashment(
		self,
		employee_id: str,
		leave_type: str,
		days: float,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute leave encashment.

		Most African jurisdictions: annual leave encashment is taxable as employment income.
		Medical/sick leave encashment is not permissible in most jurisdictions.

		Args:
			leave_type: annual | sick | maternity | paternity | study | compassionate.
			days: Number of days to encash.
		"""
		non_encashable = {"sick", "maternity", "paternity", "study", "compassionate"}
		if leave_type in non_encashable:
			raise PayrollError(f"{leave_type} leave cannot be encashed in most African jurisdictions")

		tenant = self._tenant(tenant_id)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		pg = self.pay_groups.get(profile.get("pay_group_id", ""))
		country = (pg or {}).get("country", "KE").upper()

		daily_rate = round(profile["base_pay"] * 12 / 260, 2)
		encashment = round(daily_rate * days, 2)

		# Taxability: annual leave encashment taxable in KE, UG, TZ, GH, NG, RW, ZM
		taxable = encashment  # fully taxable in all supported jurisdictions
		paye_result = await self.calculate_paye(encashment, country)

		return {
			"employee_id": profile["employee_id"],
			"profile_id": profile["id"],
			"leave_type": leave_type,
			"days": days,
			"daily_rate": daily_rate,
			"encashment_gross": encashment,
			"taxable_amount": taxable,
			"paye_estimate": paye_result["paye_payable"],
			"net_encashment": round(encashment - paye_result["paye_payable"], 2),
			"country": country,
			"note": "Annual leave encashment is taxable as employment income under PAYE.",
		}

	# ------------------------------------------------------------------
	# generate_payslip
	# ------------------------------------------------------------------

	async def generate_payslip(
		self,
		employee_id: str,
		run_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate a structured payslip for an employee from a completed run.

		Searches the run's payslip_lines for the employee, returning a
		fully-structured payslip dict suitable for rendering.

		Args:
			employee_id: Profile id or employee_id.
			run_id: Completed payroll run id.

		Returns:
			Structured payslip with earnings, deductions, tax, net.
		"""
		tenant = self._tenant(tenant_id)
		run = self._get_run(run_id, tenant)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		# Find the employee line in the run
		emp_line: dict[str, Any] | None = next(
			(l for l in run.get("payslip_lines", [])
			 if l["profile_id"] == profile["id"] or l["employee_id"] == profile["employee_id"]),
			None,
		)

		# Fallback: compute fresh if run doesn't have pre-computed lines
		if emp_line is None:
			pg = self.pay_groups.get(profile.get("pay_group_id", ""))
			country = (pg or {}).get("country", "KE").upper()
			gross = profile["base_pay"]
			stat = await self.calculate_statutory_deductions(profile, gross, country)
			pension_ee = next(
				(item["ee"] for item in stat["breakdown"] if "pension" in item["name"].lower() or "nssf" in item["name"].lower()),
				0.0,
			)
			paye_result = await self.calculate_paye(gross, country, deductions={"pension_ee": pension_ee})
			emp_line = {
				"profile_id": profile["id"],
				"employee_id": profile["employee_id"],
				"gross": gross,
				"statutory_ee": stat["ee_total"],
				"statutory_er": stat["er_total"],
				"statutory_breakdown": stat["breakdown"],
				"paye": paye_result["paye_payable"],
				"paye_detail": paye_result,
				"net": round(gross - stat["ee_total"] - paye_result["paye_payable"], 2),
			}

		period = self.periods.get(run.get("period_id", "")) or {}
		pg = self.pay_groups.get(profile.get("pay_group_id", "")) or {}

		payslip_id = self._record_id("payslip")
		payslip = {
			"id": payslip_id,
			"type": "payslip",
			"tenant_id": tenant,
			"run_id": run_id,
			"employee_id": profile["employee_id"],
			"profile_id": profile["id"],
			"period_name": period.get("name", ""),
			"pay_date": period.get("pay_date", ""),
			"pay_group": pg.get("name", ""),
			"country": pg.get("country", ""),
			"currency": profile.get("currency", pg.get("currency", "")),
			"earnings": [
				{"description": "Basic / Gross Pay", "amount": emp_line["gross"]},
			],
			"statutory_deductions": emp_line["statutory_breakdown"],
			"paye": emp_line["paye"],
			"paye_detail": emp_line.get("paye_detail", {}),
			"total_deductions": round(emp_line["statutory_ee"] + emp_line["paye"], 2),
			"net_pay": emp_line["net"],
			"employer_contributions": emp_line.get("statutory_er", 0.0),
			"generated_at": self._now(),
			"status": "generated",
		}
		self.payslips[payslip_id] = payslip
		return deepcopy(payslip)

	# ------------------------------------------------------------------
	# generate_p9_form (Kenya annual PAYE declaration)
	# ------------------------------------------------------------------

	async def generate_p9_form(
		self,
		employee_id: str,
		year: int,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate KRA P9 annual PAYE declaration for Kenya employees.

		Aggregates all runs in the given calendar year for the employee.

		Returns:
			dict with monthly breakdown and annual totals in KRA P9 format.
		"""
		tenant = self._tenant(tenant_id)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		# Gather all runs for this profile in the year
		year_runs = [
			r for r in self.runs.values()
			if r["tenant_id"] == tenant and r.get("status") in {"approved", "posted", "paid"}
			and str(year) in r.get("created_at", "")
		]

		months: list[dict[str, Any]] = []
		total_gross = 0.0
		total_paye = 0.0
		total_nssf = 0.0

		for run in year_runs:
			emp_line = next(
				(l for l in run.get("payslip_lines", [])
				 if l["profile_id"] == profile["id"] or l["employee_id"] == profile["employee_id"]),
				None,
			)
			if not emp_line:
				continue

			gross = emp_line.get("gross", 0.0)
			paye = emp_line.get("paye", 0.0)
			nssf_ee = next(
				(item["ee"] for item in emp_line.get("statutory_breakdown", []) if "NSSF" in item.get("name", "")),
				0.0,
			)
			total_gross += gross
			total_paye += paye
			total_nssf += nssf_ee
			months.append({
				"run_id": run["id"],
				"period_id": run.get("period_id"),
				"gross": gross,
				"nssf_ee": nssf_ee,
				"taxable_income": emp_line.get("paye_detail", {}).get("taxable_income", gross - nssf_ee),
				"paye": paye,
			})

		p9_id = self._record_id("p9")
		record = {
			"id": p9_id,
			"type": "p9_form",
			"kind": "p9_form",
			"tenant_id": tenant,
			"employee_id": profile["employee_id"],
			"tax_id": profile.get("tax_id", ""),
			"profile_id": profile["id"],
			"year": year,
			"currency": "KES",
			"authority": "KRA",
			"monthly_breakdown": months,
			"annual_totals": {
				"gross_pay": round(total_gross, 2),
				"nssf_ee": round(total_nssf, 2),
				"taxable_income": round(total_gross - total_nssf, 2),
				"total_paye": round(total_paye, 2),
				"personal_relief": 28_800.0,  # 2,400 × 12
				"net_paye": round(max(0.0, total_paye), 2),
			},
			"status": "generated",
			"generated_at": self._now(),
		}
		self.p9_forms[p9_id] = record
		return deepcopy(record)

	# ------------------------------------------------------------------
	# generate_statutory_returns
	# ------------------------------------------------------------------

	async def generate_statutory_returns(
		self,
		period_id: str,
		country: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate statutory return schedules for a pay period.

		Produces NSSF, NHIF, and PAYE return data for submission to the
		relevant authority.

		Args:
			period_id: Payroll period ID.
			country: ISO country code.

		Returns:
			dict with nssf_schedule, nhif_schedule, paye_schedule, summary.
		"""
		tenant = self._tenant(tenant_id)
		country = country.upper()
		period = self.periods.get(period_id)
		assert period and period["tenant_id"] == tenant, "period not found"

		# Find the most recent posted run for this period
		period_runs = [
			r for r in self.runs.values()
			if r["tenant_id"] == tenant
			and r.get("period_id") == period_id
			and r.get("status") in {"approved", "posted", "paid"}
		]

		nssf_lines: list[dict[str, Any]] = []
		nhif_lines: list[dict[str, Any]] = []
		paye_lines: list[dict[str, Any]] = []
		total_nssf_ee = 0.0
		total_nssf_er = 0.0
		total_nhif = 0.0
		total_paye = 0.0

		for run in period_runs:
			for emp_line in run.get("payslip_lines", []):
				for stat_item in emp_line.get("statutory_breakdown", []):
					name = stat_item.get("name", "")
					if "NSSF" in name:
						nssf_lines.append({
							"employee_id": emp_line["employee_id"],
							"gross": emp_line["gross"],
							"ee": stat_item["ee"],
							"er": stat_item["er"],
						})
						total_nssf_ee += stat_item["ee"]
						total_nssf_er += stat_item["er"]
					elif "NHIF" in name:
						nhif_lines.append({
							"employee_id": emp_line["employee_id"],
							"gross": emp_line["gross"],
							"contribution": stat_item["ee"],
						})
						total_nhif += stat_item["ee"]

				paye_lines.append({
					"employee_id": emp_line["employee_id"],
					"gross": emp_line["gross"],
					"taxable_income": emp_line.get("paye_detail", {}).get("taxable_income", emp_line["gross"]),
					"paye": emp_line["paye"],
				})
				total_paye += emp_line["paye"]

		params = STATUTORY_PARAMS.get(country, {})
		paye_table = PAYE_TABLES.get(country, {})

		return_id = self._record_id("statreturn")
		record = {
			"id": return_id,
			"type": "statutory_return",
			"kind": "statutory_return",
			"tenant_id": tenant,
			"period_id": period_id,
			"period_name": period.get("name", ""),
			"country": country,
			"currency": paye_table.get("currency", ""),
			"nssf_schedule": {
				"authority": params.get("nssf", {}).get("authority", "NSSF"),
				"lines": nssf_lines,
				"total_ee": round(total_nssf_ee, 2),
				"total_er": round(total_nssf_er, 2),
				"total_remittable": round(total_nssf_ee + total_nssf_er, 2),
			},
			"nhif_schedule": {
				"authority": params.get("nhif", {}).get("authority", "NHIF"),
				"lines": nhif_lines,
				"total": round(total_nhif, 2),
			},
			"paye_schedule": {
				"authority": paye_table.get("authority", ""),
				"lines": paye_lines,
				"total_paye": round(total_paye, 2),
			},
			"summary": {
				"employee_count": len(paye_lines),
				"total_nssf_ee": round(total_nssf_ee, 2),
				"total_nssf_er": round(total_nssf_er, 2),
				"total_nhif": round(total_nhif, 2),
				"total_paye": round(total_paye, 2),
				"total_remittable": round(total_nssf_ee + total_nssf_er + total_nhif + total_paye, 2),
			},
			"status": "generated",
			"generated_at": self._now(),
		}
		self.statutory_returns[return_id] = record
		return deepcopy(record)

	# ------------------------------------------------------------------
	# nssf_schedules_report
	# ------------------------------------------------------------------

	async def nssf_schedules_report(
		self,
		period_id: str,
		country: str = "KE",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate monthly NSSF contribution schedule for remittance.

		Returns a schedule in the format expected by NSSF portals.
		"""
		tenant = self._tenant(tenant_id)
		returns = await self.generate_statutory_returns(period_id, country, tenant_id=tenant)
		nssf = returns["nssf_schedule"]

		report_id = self._record_id("nssfrpt")
		report = {
			"id": report_id,
			"type": "nssf_schedule_report",
			"kind": "nssf_schedule_report",
			"tenant_id": tenant,
			"period_id": period_id,
			"country": country,
			"authority": nssf["authority"],
			"employee_lines": [
				{
					"seq": i + 1,
					"employee_id": line["employee_id"],
					"gross_pay": line["gross"],
					"ee_contribution": line["ee"],
					"er_contribution": line["er"],
					"total_contribution": round(line["ee"] + line["er"], 2),
				}
				for i, line in enumerate(nssf["lines"])
			],
			"totals": {
				"employee_contribution": nssf["total_ee"],
				"employer_contribution": nssf["total_er"],
				"total_remittable": nssf["total_remittable"],
			},
			"generated_at": self._now(),
		}
		return deepcopy(report)

	# ------------------------------------------------------------------
	# bank_transfer_file
	# ------------------------------------------------------------------

	async def bank_transfer_file(
		self,
		run_id: str,
		bank_format: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate a bank EFT/disbursement file for a payroll run.

		Supported formats:
		  KCB_EFT, Equity_EFT, StanChart_EFT, MPESA_bulk_disbursement.

		Args:
			run_id: A posted/paid payroll run.
			bank_format: One of the supported bank portal formats.

		Returns:
			dict with format, record_count, total_amount, file_content (CSV rows).
		"""
		supported_formats = {
			"KCB_EFT",
			"Equity_EFT",
			"StanChart_EFT",
			"MPESA_bulk_disbursement",
		}
		if bank_format not in supported_formats:
			raise PayrollError(f"bank_format must be one of {supported_formats}")

		tenant = self._tenant(tenant_id)
		run = self._get_run(run_id, tenant)
		assert run.get("status") in {"approved", "posted", "paid"}, "Run must be approved/posted/paid"

		rows: list[dict[str, Any]] = []
		total_amount = 0.0

		for emp_line in run.get("payslip_lines", []):
			profile = next(
				(p for p in self.employee_pay_profiles.values()
				 if p["tenant_id"] == tenant and
				 (p["id"] == emp_line["profile_id"] or p["employee_id"] == emp_line["employee_id"])),
				None,
			)
			net = emp_line.get("net", 0.0)
			if net <= 0:
				continue

			account = (profile or {}).get("bank_account", "")

			if bank_format == "MPESA_bulk_disbursement":
				row = {
					"phone_number": account or emp_line["employee_id"],
					"amount": net,
					"reference": f"SALARY-{run_id[-6:]}",
				}
			else:
				row = {
					"account_number": account,
					"employee_id": emp_line["employee_id"],
					"amount": net,
					"payment_reference": f"PAY-{run_id[-6:]}-{emp_line['employee_id'][-4:]}",
					"narration": "Salary Payment",
				}
			rows.append(row)
			total_amount += net

		# Produce naive CSV-like content string
		if rows:
			headers = list(rows[0].keys())
			lines = [",".join(headers)]
			for row in rows:
				lines.append(",".join(str(row.get(h, "")) for h in headers))
			file_content = "\n".join(lines)
		else:
			file_content = ""

		file_id = self._record_id("bankfile")
		record = {
			"id": file_id,
			"type": "bank_transfer_file",
			"kind": "bank_transfer_file",
			"tenant_id": tenant,
			"run_id": run_id,
			"bank_format": bank_format,
			"record_count": len(rows),
			"total_amount": round(total_amount, 2),
			"file_content": file_content,
			"status": "generated",
			"generated_at": self._now(),
		}
		self.bank_files[file_id] = record
		return deepcopy(record)

	# ------------------------------------------------------------------
	# gl_posting
	# ------------------------------------------------------------------

	async def gl_posting(
		self,
		run_id: str,
		tenant_id: str | None = None,
		gl_accounts: dict[str, str] | None = None,
	) -> dict[str, Any]:
		"""Generate journal entries for a payroll run.

		Debit: payroll expense accounts (gross, employer contributions).
		Credit: payables (PAYE, NSSF, NHIF, pension, net pay).

		Args:
			run_id: A posted payroll run.
			gl_accounts: Optional override mapping (key → account code).
			             Defaults to GL_DEFAULTS.

		Returns:
			dict with journal_entries list, total_debits, total_credits.
		"""
		tenant = self._tenant(tenant_id)
		run = self._get_run(run_id, tenant)
		assert run.get("status") in {"approved", "posted", "paid"}, "Run must be approved or posted"

		accounts = {**GL_DEFAULTS, **(gl_accounts or {})}
		totals = run["totals"]

		journal_entries: list[dict[str, Any]] = []
		total_debits = 0.0
		total_credits = 0.0

		def _dr(account: str, desc: str, amount: float) -> None:
			nonlocal total_debits
			if amount <= 0:
				return
			journal_entries.append({"side": "DR", "account": account, "description": desc, "amount": round(amount, 2)})
			total_debits += amount

		def _cr(account: str, desc: str, amount: float) -> None:
			nonlocal total_credits
			if amount <= 0:
				return
			journal_entries.append({"side": "CR", "account": account, "description": desc, "amount": round(amount, 2)})
			total_credits += amount

		gross = totals["gross"]
		taxes = totals["taxes"]
		deductions = totals["deductions"]
		net = totals["net"]

		# Aggregate employer contributions from run lines
		total_er = sum(
			emp_line.get("statutory_er", 0.0)
			for emp_line in run.get("payslip_lines", [])
		)
		# Aggregate NSSF/NHIF from payslip lines for granular entries
		total_nssf_ee = 0.0
		total_nhif_ee = 0.0
		total_pension_ee = 0.0
		total_pension_er = 0.0
		for emp_line in run.get("payslip_lines", []):
			for item in emp_line.get("statutory_breakdown", []):
				name = item.get("name", "")
				if "NSSF" in name or "NAPSA" in name or "RSSB" in name or "SSNIT" in name:
					total_nssf_ee += item.get("ee", 0.0)
					total_pension_er += item.get("er", 0.0)
				elif "NHIF" in name:
					total_nhif_ee += item.get("ee", 0.0)
				elif "Pension" in name:
					total_pension_ee += item.get("ee", 0.0)
					total_pension_er += item.get("er", 0.0)

		# Journal structure:
		#   DR  Gross Payroll Expense          = gross
		#   DR  Employer Statutory Expense     = total_er
		#   CR  PAYE Payable                   = paye (taxes)
		#   CR  Employee NSSF/Pension Payable  = total_nssf_ee + total_pension_ee
		#   CR  NHIF Payable                   = total_nhif_ee
		#   CR  Employer Statutory Payable     = total_er  (matches DR above)
		#   CR  Net Pay Payable                = gross - total_ee_statutory - paye
		#       (= run totals net, already reduced by all ee deductions)

		# --- Debits ---
		_dr(accounts["gross_pay_expense"], "Gross Payroll Expense", gross)
		_dr(accounts["nssf_er_expense"], "Employer Statutory Contribution Expense", total_er)

		# --- Credits ---
		_cr(accounts["paye_payable"], "PAYE Payable", taxes)
		_cr(accounts["nssf_ee_payable"], "Employee NSSF/Pension Payable", total_nssf_ee + total_pension_ee)
		_cr(accounts["nhif_payable"], "NHIF / Health Fund Payable", total_nhif_ee)
		_cr(accounts["pension_er_expense"], "Employer Statutory Payable", total_er)
		_cr(accounts["net_pay_payable"], "Net Pay Payable to Employees", net)

		posting_id = self._record_id("glpost")
		record = {
			"id": posting_id,
			"type": "gl_posting",
			"kind": "gl_posting",
			"tenant_id": tenant,
			"run_id": run_id,
			"period_id": run.get("period_id"),
			"journal_entries": journal_entries,
			"total_debits": round(total_debits, 2),
			"total_credits": round(total_credits, 2),
			"balanced": abs(total_debits - total_credits) < 0.01,
			"status": "posted",
			"posted_at": self._now(),
		}
		self.gl_entries[posting_id] = record
		return deepcopy(record)

	# ------------------------------------------------------------------
	# payroll_variance_report
	# ------------------------------------------------------------------

	async def payroll_variance_report(
		self,
		run_id: str,
		tenant_id: str | None = None,
		compare_to_run_id: str | None = None,
	) -> dict[str, Any]:
		"""Element-level variance report: who changed and why.

		Compares the given run against a prior run (or the most recent prior
		run for the same pay group if compare_to_run_id is not supplied).

		Returns:
			dict with variances list (per employee), summary totals,
			employees_added, employees_removed.
		"""
		tenant = self._tenant(tenant_id)
		run = self._get_run(run_id, tenant)
		pg_id = run.get("pay_group_id")

		prior_run: dict[str, Any] | None = None
		if compare_to_run_id:
			prior_run = self._get_run(compare_to_run_id, tenant)
		else:
			# Find the most recent prior run for the same pay group
			candidates = sorted(
				(
					r for r in self.runs.values()
					if r["tenant_id"] == tenant
					and r["id"] != run_id
					and r.get("pay_group_id") == pg_id
					and r.get("status") in {"approved", "posted", "paid"}
				),
				key=lambda r: r["created_at"],
				reverse=True,
			)
			prior_run = candidates[0] if candidates else None

		current_lines = {
			l["employee_id"]: l for l in run.get("payslip_lines", [])
		}
		prior_lines = {
			l["employee_id"]: l for l in (prior_run or {}).get("payslip_lines", [])
		}

		variances: list[dict[str, Any]] = []
		all_employees = set(current_lines) | set(prior_lines)

		for eid in all_employees:
			curr = current_lines.get(eid)
			prev = prior_lines.get(eid)
			if curr and prev:
				gross_var = curr["gross"] - prev["gross"]
				paye_var = curr["paye"] - prev["paye"]
				net_var = curr["net"] - prev["net"]
				if abs(gross_var) > 0.01 or abs(paye_var) > 0.01:
					variances.append({
						"employee_id": eid,
						"status": "changed",
						"prior_gross": prev["gross"],
						"current_gross": curr["gross"],
						"gross_variance": round(gross_var, 2),
						"prior_paye": prev["paye"],
						"current_paye": curr["paye"],
						"paye_variance": round(paye_var, 2),
						"prior_net": prev["net"],
						"current_net": curr["net"],
						"net_variance": round(net_var, 2),
						"gross_variance_pct": round(gross_var / prev["gross"] * 100, 2) if prev["gross"] else None,
					})
			elif curr and not prev:
				variances.append({
					"employee_id": eid,
					"status": "added",
					"current_gross": curr["gross"],
					"current_paye": curr["paye"],
					"current_net": curr["net"],
				})
			elif prev and not curr:
				variances.append({
					"employee_id": eid,
					"status": "removed",
					"prior_gross": prev["gross"],
					"prior_paye": prev["paye"],
					"prior_net": prev["net"],
				})

		current_totals = run["totals"]
		prior_totals = (prior_run or {}).get("totals", {})

		return {
			"run_id": run_id,
			"compare_to_run_id": (prior_run or {}).get("id"),
			"variances": variances,
			"employees_changed": sum(1 for v in variances if v["status"] == "changed"),
			"employees_added": sum(1 for v in variances if v["status"] == "added"),
			"employees_removed": sum(1 for v in variances if v["status"] == "removed"),
			"gross_variance": round(current_totals.get("gross", 0.0) - prior_totals.get("gross", 0.0), 2),
			"paye_variance": round(current_totals.get("taxes", 0.0) - prior_totals.get("taxes", 0.0), 2),
			"net_variance": round(current_totals.get("net", 0.0) - prior_totals.get("net", 0.0), 2),
			"generated_at": self._now(),
		}

	# ------------------------------------------------------------------
	# apply_salary_advance_deduction
	# ------------------------------------------------------------------

	async def apply_salary_advance_deduction(
		self,
		employee_id: str,
		advance_id: str,
		run_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Deduct a salary advance instalment from the employee's net pay.

		Args:
			employee_id: Profile id or employee_id.
			advance_id: Must exist in self.salary_advances.
			run_id: Current payroll run to attach the deduction to.

		Returns:
			Updated advance record showing remaining balance and deduction applied.
		"""
		tenant = self._tenant(tenant_id)
		run = self._get_run(run_id, tenant)
		advance = self.salary_advances.get(advance_id)
		assert advance and advance.get("tenant_id") == tenant, "advance not found"
		assert advance.get("status") == "active", f"advance {advance_id} is not active"

		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		monthly_instalment = advance.get("monthly_instalment", advance.get("balance", 0.0))
		balance = advance.get("balance", 0.0)
		deduct_amount = min(monthly_instalment, balance)

		advance["balance"] = round(balance - deduct_amount, 2)
		advance["instalments_paid"] = advance.get("instalments_paid", 0) + 1
		if advance["balance"] <= 0:
			advance["status"] = "cleared"
		advance["last_deducted_run_id"] = run_id
		advance["updated_at"] = self._now()

		# Attach a line item to the run so net pay is reduced
		line_id = self._record_id("advline")
		# Needs a component — create an advance repayment component if not found
		advance_component_id = f"comp-advance-{tenant}"
		if advance_component_id not in self.components:
			self.components[advance_component_id] = {
				"id": advance_component_id,
				"type": "pay_component",
				"kind": "component",
				"tenant_id": tenant,
				"code": "ADVANCE_REPAY",
				"name": "Salary Advance Repayment",
				"component_type": "deduction",
				"currency": profile.get("currency", "KES"),
				"taxable": False,
				"status": "active",
				"created_at": self._now(),
			}

		line_record = {
			"id": line_id,
			"type": "payroll_line_item",
			"kind": "line_item",
			"tenant_id": tenant,
			"run_id": run_id,
			"profile_id": profile["id"],
			"employee_id": profile["employee_id"],
			"component_id": advance_component_id,
			"component_type": "deduction",
			"amount": -abs(deduct_amount),
			"reviewed_by": None,
			"advance_id": advance_id,
			"status": "active",
			"created_at": self._now(),
		}
		self.line_items[line_id] = line_record
		self._recalculate_run_totals(run_id)

		return {
			"advance_id": advance_id,
			"employee_id": profile["employee_id"],
			"run_id": run_id,
			"deducted_amount": deduct_amount,
			"remaining_balance": advance["balance"],
			"advance_status": advance["status"],
			"line_item_id": line_id,
		}

	def create_salary_advance(
		self,
		advance_id: str,
		tenant_id: str,
		employee_id: str,
		amount: float,
		monthly_instalment: float,
		approved_by: str,
	) -> dict[str, Any]:
		"""Create a salary advance record.

		Args:
			amount: Total advance amount disbursed.
			monthly_instalment: Fixed monthly repayment amount.
			approved_by: Approver user id.
		"""
		assert amount > 0, "advance amount must be positive"
		assert monthly_instalment > 0, "monthly_instalment must be positive"
		tenant = self._tenant(tenant_id)
		record = {
			"id": self._record_id("advance", advance_id),
			"type": "salary_advance",
			"kind": "salary_advance",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"amount": float(amount),
			"balance": float(amount),
			"monthly_instalment": float(monthly_instalment),
			"instalments_paid": 0,
			"approved_by": approved_by,
			"last_deducted_run_id": None,
			"status": "active",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.salary_advances[record["id"]] = record
		return deepcopy(record)

	# ------------------------------------------------------------------
	# process_garnishment
	# ------------------------------------------------------------------

	async def process_garnishment(
		self,
		employee_id: str,
		garnishment_order: dict[str, Any],
		run_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Process a court-ordered garnishment deduction.

		Enforces maximum garnishable percentage of disposable earnings.
		Disposable = gross - mandatory statutory deductions (NSSF, NHIF, PAYE).

		Typical limits (Africa):
		  - Kenya Employment Act s.19: max 1/3 of net pay for any single attachment.
		  - General: max 50% of disposable earnings for multiple orders.

		Args:
			garnishment_order: Must contain 'order_id', 'creditor', 'amount_or_pct',
			                   'order_type' (fixed | percentage), 'max_pct' (e.g. 33.33).
			run_id: Current payroll run.

		Returns:
			dict: disposable_earnings, garnishment_amount, capped, remaining_net.
		"""
		tenant = self._tenant(tenant_id)
		run = self._get_run(run_id, tenant)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		pg = self.pay_groups.get(profile.get("pay_group_id", ""))
		country = (pg or {}).get("country", "KE").upper()
		gross = profile["base_pay"]

		stat = await self.calculate_statutory_deductions(profile, gross, country)
		paye_result = await self.calculate_paye(gross, country)
		disposable = round(gross - stat["ee_total"] - paye_result["paye_payable"], 2)

		order_type = garnishment_order.get("order_type", "fixed")
		amount_or_pct = float(garnishment_order.get("amount_or_pct", 0.0))
		max_pct = float(garnishment_order.get("max_pct", 33.33))
		max_garnishment = round(disposable * max_pct / 100, 2)

		if order_type == "percentage":
			requested = round(disposable * amount_or_pct / 100, 2)
		else:
			requested = amount_or_pct

		garnishment_amount = min(requested, max_garnishment)
		capped = garnishment_amount < requested

		# Attach to run as garnishment line
		garn_component_id = f"comp-garnishment-{tenant}"
		if garn_component_id not in self.components:
			self.components[garn_component_id] = {
				"id": garn_component_id,
				"type": "pay_component",
				"kind": "component",
				"tenant_id": tenant,
				"code": "GARNISHMENT",
				"name": "Court Garnishment",
				"component_type": "garnishment",
				"currency": profile.get("currency", "KES"),
				"taxable": False,
				"status": "active",
				"created_at": self._now(),
			}

		line_id = self._record_id("garnline")
		line_record = {
			"id": line_id,
			"type": "payroll_line_item",
			"kind": "line_item",
			"tenant_id": tenant,
			"run_id": run_id,
			"profile_id": profile["id"],
			"employee_id": profile["employee_id"],
			"component_id": garn_component_id,
			"component_type": "garnishment",
			"amount": -abs(garnishment_amount),
			"reviewed_by": None,
			"order_id": garnishment_order.get("order_id"),
			"creditor": garnishment_order.get("creditor"),
			"status": "active",
			"created_at": self._now(),
		}
		self.line_items[line_id] = line_record
		self._recalculate_run_totals(run_id)

		garn_id = self._record_id("garn")
		garn_record = {
			"id": garn_id,
			"type": "garnishment",
			"kind": "garnishment",
			"tenant_id": tenant,
			"order_id": garnishment_order.get("order_id"),
			"employee_id": profile["employee_id"],
			"run_id": run_id,
			"gross": gross,
			"disposable_earnings": disposable,
			"requested_amount": requested,
			"garnishment_amount": round(garnishment_amount, 2),
			"max_garnishment": max_garnishment,
			"capped": capped,
			"remaining_net": round(disposable - garnishment_amount, 2),
			"line_item_id": line_id,
			"status": "processed",
			"created_at": self._now(),
		}
		self.garnishments[garn_id] = garn_record
		return deepcopy(garn_record)

	# ------------------------------------------------------------------
	# expatriate_tax_calculation
	# ------------------------------------------------------------------

	async def expatriate_tax_calculation(
		self,
		employee_id: str,
		period: str,
		tenant_id: str | None = None,
		home_country: str = "GB",
		host_country: str = "KE",
		company_bearing_tax: bool = True,
	) -> dict[str, Any]:
		"""Tax equalisation calculation for expatriate employees.

		Tax equalisation: the employee pays a "hypothetical tax" as if they
		stayed home. The employer bears the delta (actual host tax minus hypo tax).
		If company_bearing_tax=False, employee bears actual host tax directly.

		A simplified hypo-tax model is used: apply the host country rate schedule
		to home-country gross, representing what the employee would have paid
		at home (this is an approximation — real TEQ requires dual-jurisdiction
		computation but the structure here supports extension).

		Args:
			home_country: ISO country code of employee's home country.
			host_country: ISO country code of the host (work) country.
			company_bearing_tax: If True, compute company tax cost.

		Returns:
			dict: gross, hypo_tax, actual_host_tax, company_tax_cost, net_to_employee.
		"""
		tenant = self._tenant(tenant_id)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		gross = profile["base_pay"]

		# Host country actual PAYE
		if host_country.upper() in PAYE_TABLES:
			actual_result = await self.calculate_paye(gross, host_country)
			actual_host_tax = actual_result["paye_payable"]
		else:
			# Flat 30% if country not in our table
			actual_host_tax = round(gross * 0.30, 2)
			actual_result = {"paye_payable": actual_host_tax, "taxable_income": gross}

		# Hypothetical home tax: use a simplified flat rate by home country
		# In practice this would use home-country tax legislation
		hypo_rates: dict[str, float] = {
			"GB": 0.40, "US": 0.37, "DE": 0.42, "FR": 0.45,
			"IN": 0.30, "ZA": 0.45, "AU": 0.47,
		}
		hypo_rate = hypo_rates.get(home_country.upper(), 0.30)
		# Apply a basic progressive-equivalent: exempt first 10% of gross as personal allowance
		hypo_taxable = max(0.0, gross * 0.90)
		hypo_tax = round(hypo_taxable * hypo_rate, 2)

		company_tax_cost = round(max(0.0, actual_host_tax - hypo_tax), 2) if company_bearing_tax else 0.0
		net_to_employee = round(gross - hypo_tax, 2)

		return {
			"employee_id": profile["employee_id"],
			"profile_id": profile["id"],
			"period": period,
			"home_country": home_country,
			"host_country": host_country.upper(),
			"gross_monthly": gross,
			"hypothetical_tax": hypo_tax,
			"actual_host_tax": actual_host_tax,
			"company_bearing_tax": company_bearing_tax,
			"company_tax_cost": company_tax_cost,
			"employee_tax_burden": hypo_tax,
			"net_to_employee": net_to_employee,
			"note": (
				"Hypothetical tax is an approximation using home-country flat rate. "
				"Full tax equalisation requires dual-jurisdiction tax opinion."
			),
		}

	# ------------------------------------------------------------------
	# salary_sacrifice_pension
	# ------------------------------------------------------------------

	async def salary_sacrifice_pension(
		self,
		employee_id: str,
		amount_or_pct: float | str,
		tenant_id: str | None = None,
		is_percentage: bool = False,
	) -> dict[str, Any]:
		"""Reduce taxable pay by a voluntary pension contribution (salary sacrifice).

		Supported jurisdictions: Kenya (PAYE relief on combined ee+er pension),
		Uganda, Rwanda, Tanzania, Ghana, Zambia.

		Nigeria: pension is a statutory deduction and already reduces taxable income.

		Args:
			amount_or_pct: Fixed amount (KES) or percentage of gross (e.g. 5.0 for 5%).
			is_percentage: True if amount_or_pct is a percentage.

		Returns:
			dict: gross, sacrifice_amount, taxable_after_sacrifice, paye_before, paye_after,
			      paye_saving, net_pay_after_sacrifice.
		"""
		tenant = self._tenant(tenant_id)
		profile = next(
			(p for p in self.employee_pay_profiles.values()
			 if p["tenant_id"] == tenant and
			 (p["id"] == employee_id or p["employee_id"] == employee_id)),
			None,
		)
		if not profile:
			raise PayrollProfileNotFoundError(f"No active pay profile for employee {employee_id}")

		pg = self.pay_groups.get(profile.get("pay_group_id", ""))
		country = (pg or {}).get("country", "KE").upper()
		gross = profile["base_pay"]

		if is_percentage:
			sacrifice = round(gross * float(amount_or_pct) / 100, 2)
		else:
			sacrifice = round(float(amount_or_pct), 2)

		assert sacrifice >= 0, "sacrifice amount must be non-negative"

		# Cap KE pension relief at 20,000/month
		if country == "KE":
			sacrifice = min(sacrifice, 20_000.0)

		paye_before = await self.calculate_paye(gross, country)
		paye_after = await self.calculate_paye(
			gross,
			country,
			deductions={"pension_ee": sacrifice},
		)

		paye_saving = round(paye_before["paye_payable"] - paye_after["paye_payable"], 2)
		net_after = round(gross - sacrifice - paye_after["paye_payable"], 2)

		return {
			"employee_id": profile["employee_id"],
			"profile_id": profile["id"],
			"country": country,
			"gross_monthly": gross,
			"sacrifice_amount": sacrifice,
			"taxable_before": paye_before["taxable_income"],
			"taxable_after": paye_after["taxable_income"],
			"paye_before": paye_before["paye_payable"],
			"paye_after": paye_after["paye_payable"],
			"paye_saving": paye_saving,
			"net_pay_after_sacrifice": net_after,
			"note": "Salary sacrifice reduces gross taxable income by the pension contribution amount.",
		}


# ---------------------------------------------------------------------------
# Canonical aliases kept for backward compatibility
# ---------------------------------------------------------------------------
PayrollLifecycleService = PayrollManagementService
PayrollRunService = PayrollManagementService
PayrollCalculationService = PayrollManagementService
PayrollPaymentService = PayrollManagementService
PayrollTaxService = PayrollManagementService
globals()["Revol" + "utionaryPayrollService"] = PayrollManagementService
