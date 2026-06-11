"""SASRA Regulatory Reporting — full async service.

Implements all SASRA prudential return forms and compliance checks per the
SACCO Societies (Deposit-Taking) Regulations, 2010 and SASRA Prudential Guidelines.

Key SASRA minimums enforced:
  Capital Adequacy Ratio (CAR):  >= 10%  (institutional capital / risk-weighted assets)
  Core Capital / Total Assets:   >= 8%
  Liquidity Ratio:               >= 15%  (liquid assets / (deposits + borrowings))
  Loan to Deposit Ratio (LDR):   <= 70%
  Non-Performing Loan (NPL):     warning > 5%, breach > 10%
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

try:
	from capabilities.common.reliability import guard_tenant_id as _guard_tenant_id
	def guard_tenant_id(tenant_id: str | None, default: str = "default") -> str:
		v = tenant_id or default
		_guard_tenant_id(v)
		return v
except ImportError:
	def guard_tenant_id(tenant_id: str | None, default: str = "default") -> str:
		v = tenant_id or default
		if not v:
			raise PermissionError("tenant_context_required")
		return v

from .models import (
	AnnualReport,
	BalanceSheet,
	CapitalAdequacyResult,
	ComplianceDashboard,
	ComplianceStatus,
	FilingRecord,
	FilingStatus,
	IncomeStatement,
	LoanBand,
	LoanClassification,
	LoanClassificationBand,
	LiquidityResult,
	QuarterlyReturn,
	RatioStatus,
	RegulatoryDeadline,
	ReturnType,
	TrafficLight,
)

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_reg"

# ── SASRA Thresholds ──────────────────────────────────────────────────────────
_CAR_MIN = Decimal("10.00")         # capital adequacy ratio minimum %
_CORE_CAPITAL_RATIO_MIN = Decimal("8.00")  # core capital / total assets minimum %
_LIQUIDITY_MIN = Decimal("15.00")   # liquidity ratio minimum %
_LDR_MAX = Decimal("70.00")         # loan-to-deposit ratio maximum %
_NPL_WARN = Decimal("5.00")         # NPL ratio warning threshold %
_NPL_BREACH = Decimal("10.00")      # NPL ratio breach threshold %
_AMBER_BUFFER = Decimal("2.00")     # within 2pp of minimum → amber

# DPD band definitions: (band, dpd_min, dpd_max_incl, provision_rate, dpd_range_label)
_DPD_BANDS: list[tuple[LoanClassificationBand, int, int, Decimal, str]] = [
	(LoanClassificationBand.NORMAL,      0,   30,  Decimal("0"),    "0-30"),
	(LoanClassificationBand.WATCH,       31,  90,  Decimal("1"),    "31-90"),
	(LoanClassificationBand.SUBSTANDARD, 91,  180, Decimal("25"),   "91-180"),
	(LoanClassificationBand.DOUBTFUL,    181, 365, Decimal("50"),   "181-365"),
	(LoanClassificationBand.LOSS,        366, 9999,Decimal("100"),  ">365"),
]

# Quarter-end months
_QE_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}
# Filing due: 30 days after quarter-end
_QUARTERLY_FILING_DAYS = 30
# Annual filing: 4 months after FY end (June 30 for June FY)
_ANNUAL_FILING_MONTHS = 4


class SACCARegulatoryService:
	"""Async service for SASRA prudential returns, ratio monitoring, and filing registry.

	All financial data is sourced from the SACCO's internal ledger via injected
	data callbacks. In standalone/test mode, seed via _seed_ledger().
	"""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		# Filing registry: id -> FilingRecord
		self._filings: dict[str, FilingRecord] = {}
		# Ledger snapshots per tenant: tenant_id -> {date_str -> snapshot_dict}
		# In production these come from dep/lnd services; here we store seeded data.
		self._ledger: dict[str, dict[str, Any]] = {}
		self._audit: list[dict[str, Any]] = []

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _t(self, tenant_id: str | None) -> str:
		return guard_tenant_id(tenant_id or self.tenant_id)

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _today(self) -> str:
		return date.today().isoformat()

	def _emit(self, tenant_id: str, event: str, payload: dict[str, Any]) -> None:
		self._audit.append({
			"tenant_id": tenant_id,
			"event": event,
			"payload_id": payload.get("id", ""),
			"ts": self._now(),
		})

	def _log_ratio(self, name: str, actual: Decimal, required: Decimal, compliant: bool) -> None:
		status = "OK" if compliant else "BREACH"
		_log.info("[SASRA] %s actual=%.2f%% required=%.2f%% [%s]", name, actual, required, status)

	def _quarter_end_date(self, year: int, quarter: int) -> date:
		month = _QE_MONTH[quarter]
		if month == 3:
			return date(year, 3, 31)
		if month == 6:
			return date(year, 6, 30)
		if month == 9:
			return date(year, 9, 30)
		return date(year, 12, 31)

	def _traffic_light_ratio(
		self,
		actual: Decimal,
		minimum: Decimal | None,
		maximum: Decimal | None,
	) -> tuple[bool, TrafficLight]:
		"""Return (compliant, traffic_light) for a ratio."""
		if minimum is not None:
			if actual >= minimum:
				if actual < minimum + _AMBER_BUFFER:
					return True, TrafficLight.AMBER
				return True, TrafficLight.GREEN
			return False, TrafficLight.RED
		if maximum is not None:
			if actual <= maximum:
				if actual > maximum - _AMBER_BUFFER:
					return True, TrafficLight.AMBER
				return True, TrafficLight.GREEN
			return False, TrafficLight.RED
		return True, TrafficLight.GREEN

	def _pct(self, numerator: Decimal, denominator: Decimal) -> Decimal:
		"""Safe percentage: 0 if denominator is zero."""
		if denominator == 0:
			return Decimal("0")
		return (numerator / denominator * 100).quantize(Decimal("0.0001"), ROUND_HALF_UP)

	def _get_ledger(self, tenant_id: str, as_of_date: str) -> dict[str, Any]:
		"""Return ledger snapshot for tenant/date, or default zeros."""
		tenant_data = self._ledger.get(tenant_id, {})
		# Exact match first, then latest before date
		if as_of_date in tenant_data:
			return tenant_data[as_of_date]
		before = [d for d in sorted(tenant_data.keys()) if d <= as_of_date]
		if before:
			return tenant_data[before[-1]]
		return {}

	def seed_ledger(
		self,
		tenant_id: str,
		as_of_date: str,
		data: dict[str, Any],
	) -> None:
		"""Inject a ledger snapshot (used in tests and by adapter layer)."""
		self._ledger.setdefault(tenant_id, {})[as_of_date] = data

	# ── Loan classification helper ────────────────────────────────────────────

	def _band_for_dpd(self, dpd: int) -> tuple[LoanClassificationBand, Decimal, str]:
		"""Return (band, provision_rate_pct, label) for a DPD value."""
		for band, lo, hi, rate, label in _DPD_BANDS:
			if lo <= dpd <= hi:
				return band, rate, label
		return LoanClassificationBand.LOSS, Decimal("100"), ">365"

	# ── Core SASRA calculations ───────────────────────────────────────────────

	async def calculate_capital_adequacy(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> CapitalAdequacyResult:
		"""SASRA Form 3: Capital Adequacy Return.

		CAR = institutional_capital / risk_weighted_assets >= 10%
		Core capital / total assets >= 8%
		"""
		t = self._t(tenant_id)
		d = as_of_date or self._today()
		snap = self._get_ledger(t, d)

		core_capital = Decimal(str(snap.get("core_capital", 0)))
		secondary_capital = Decimal(str(snap.get("secondary_capital", 0)))
		institutional_capital = core_capital + secondary_capital
		total_assets = Decimal(str(snap.get("total_assets", 0)))

		# Risk weights per SASRA: loans 100%, govt securities 0%, fixed assets 100%
		gross_loans = Decimal(str(snap.get("gross_loan_portfolio", 0)))
		govt_securities = Decimal(str(snap.get("government_securities", 0)))
		fixed_assets = Decimal(str(snap.get("fixed_assets", 0)))
		other_assets = total_assets - gross_loans - govt_securities - fixed_assets
		risk_weighted_assets = (
			gross_loans * Decimal("1.0")
			+ govt_securities * Decimal("0.0")
			+ fixed_assets * Decimal("1.0")
			+ other_assets * Decimal("0.5")
		)

		car = self._pct(institutional_capital, risk_weighted_assets)
		core_ratio = self._pct(core_capital, total_assets)
		compliant = car >= _CAR_MIN and core_ratio >= _CORE_CAPITAL_RATIO_MIN
		shortfall = max(Decimal("0"), _CAR_MIN - car)

		if car >= _CAR_MIN:
			if car < _CAR_MIN + _AMBER_BUFFER:
				tl = TrafficLight.AMBER
			else:
				tl = TrafficLight.GREEN
		else:
			tl = TrafficLight.RED

		result = CapitalAdequacyResult(
			as_of_date=d,
			core_capital=core_capital,
			secondary_capital=secondary_capital,
			institutional_capital=institutional_capital,
			total_assets=total_assets,
			risk_weighted_assets=risk_weighted_assets,
			capital_adequacy_ratio=car,
			core_capital_ratio=core_ratio,
			minimum_required=_CAR_MIN,
			compliant=compliant,
			shortfall=shortfall,
			traffic_light=tl,
		)
		self._log_ratio("CAR", car, _CAR_MIN, compliant)
		return result

	async def calculate_liquidity_ratio(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> LiquidityResult:
		"""SASRA Form 4: Liquidity Return.

		Liquidity ratio = liquid_assets / (deposits + borrowings) >= 15%
		"""
		t = self._t(tenant_id)
		d = as_of_date or self._today()
		snap = self._get_ledger(t, d)

		cash = Decimal(str(snap.get("cash_on_hand", 0)))
		bank = Decimal(str(snap.get("bank_balances", 0)))
		govt = Decimal(str(snap.get("government_securities", 0)))
		other_liquid = Decimal(str(snap.get("other_liquid_assets", 0)))
		liquid = cash + bank + govt + other_liquid

		deposits = Decimal(str(snap.get("member_deposits", 0)))
		borrowings = Decimal(str(snap.get("external_borrowings", 0)))
		base = deposits + borrowings

		ratio = self._pct(liquid, base)
		compliant = ratio >= _LIQUIDITY_MIN
		shortfall = max(Decimal("0"), _LIQUIDITY_MIN - ratio)

		if compliant:
			tl = TrafficLight.AMBER if ratio < _LIQUIDITY_MIN + _AMBER_BUFFER else TrafficLight.GREEN
		else:
			tl = TrafficLight.RED

		result = LiquidityResult(
			as_of_date=d,
			cash_on_hand=cash,
			bank_balances=bank,
			government_securities=govt,
			other_liquid_assets=other_liquid,
			total_liquid_assets=liquid,
			total_deposits=deposits,
			total_borrowings=borrowings,
			total_deposits_and_borrowings=base,
			liquidity_ratio=ratio,
			minimum_required=_LIQUIDITY_MIN,
			compliant=compliant,
			shortfall=shortfall,
			traffic_light=tl,
		)
		self._log_ratio("Liquidity", ratio, _LIQUIDITY_MIN, compliant)
		return result

	async def calculate_loan_to_deposit_ratio(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> Decimal:
		"""LDR = gross_loan_portfolio / member_deposits. Maximum 70% per SASRA."""
		t = self._t(tenant_id)
		d = as_of_date or self._today()
		snap = self._get_ledger(t, d)
		loans = Decimal(str(snap.get("gross_loan_portfolio", 0)))
		deposits = Decimal(str(snap.get("member_deposits", 0)))
		return self._pct(loans, deposits)

	async def classify_loan_portfolio(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> LoanClassification:
		"""SASRA Form 5: Loan Portfolio Quality — classify loans by DPD bands."""
		t = self._t(tenant_id)
		d = as_of_date or self._today()
		snap = self._get_ledger(t, d)

		# loan_books: list of {outstanding_balance, days_past_due}
		loan_books: list[dict[str, Any]] = snap.get("loan_books", [])
		# Fall back to aggregate summary if loan_books absent
		if not loan_books:
			loan_books = snap.get("loans", [])

		# Accumulate per band
		band_data: dict[LoanClassificationBand, dict[str, Any]] = {}
		for band, lo, hi, rate, label in _DPD_BANDS:
			band_data[band] = {
				"band": band,
				"dpd_range": label,
				"number_of_loans": 0,
				"outstanding_balance": Decimal("0"),
				"provision_rate": rate,
				"required_provision": Decimal("0"),
			}

		total_gross = Decimal("0")
		for loan in loan_books:
			dpd = int(loan.get("days_past_due", 0))
			bal = Decimal(str(loan.get("outstanding_balance", 0)))
			band, rate, _ = self._band_for_dpd(dpd)
			bd = band_data[band]
			bd["number_of_loans"] += 1
			bd["outstanding_balance"] += bal
			bd["required_provision"] += (bal * rate / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
			total_gross += bal

		bands = [LoanBand(**bd) for bd in band_data.values()]
		total_provisions = sum(b.required_provision for b in bands)
		actual_provisions = Decimal(str(snap.get("loan_loss_provisions", 0)))
		coverage = self._pct(actual_provisions, total_provisions) if total_provisions else Decimal("100")

		# NPL = substandard + doubtful + loss
		npl = sum(
			bd["outstanding_balance"]
			for bnd, bd in band_data.items()
			if bnd in {LoanClassificationBand.SUBSTANDARD, LoanClassificationBand.DOUBTFUL, LoanClassificationBand.LOSS}
		)
		npl_ratio = self._pct(npl, total_gross)

		# PAR30 = watch + substandard + doubtful + loss (>30 DPD)
		par30_bal = sum(
			bd["outstanding_balance"]
			for bnd, bd in band_data.items()
			if bnd != LoanClassificationBand.NORMAL
		)
		# PAR90 = substandard + doubtful + loss (>90 DPD)
		par90_bal = sum(
			bd["outstanding_balance"]
			for bnd, bd in band_data.items()
			if bnd in {LoanClassificationBand.SUBSTANDARD, LoanClassificationBand.DOUBTFUL, LoanClassificationBand.LOSS}
		)

		return LoanClassification(
			as_of_date=d,
			bands=bands,
			total_gross_portfolio=total_gross,
			total_required_provisions=total_provisions,
			actual_provisions_held=actual_provisions,
			provisioning_coverage=coverage,
			npl_balance=npl,
			npl_ratio=npl_ratio,
			par30=self._pct(par30_bal, total_gross),
			par90=self._pct(par90_bal, total_gross),
		)

	async def calculate_required_provisions(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> Decimal:
		"""Total required loan loss provisions per SASRA DPD matrix."""
		lc = await self.classify_loan_portfolio(tenant_id, as_of_date)
		return lc.total_required_provisions

	async def calculate_provisioning_coverage(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> Decimal:
		"""Actual provisions / required provisions * 100."""
		lc = await self.classify_loan_portfolio(tenant_id, as_of_date)
		return lc.provisioning_coverage

	async def calculate_npl_ratio(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> Decimal:
		"""Non-performing loan ratio = (substandard+doubtful+loss) / gross_portfolio * 100."""
		lc = await self.classify_loan_portfolio(tenant_id, as_of_date)
		return lc.npl_ratio

	async def calculate_par(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
		days: int = 30,
	) -> Decimal:
		"""Portfolio at Risk at given DPD threshold. PAR30 or PAR90."""
		lc = await self.classify_loan_portfolio(tenant_id, as_of_date)
		if days <= 30:
			return lc.par30
		return lc.par90

	# ── Balance Sheet helper ──────────────────────────────────────────────────

	def _build_balance_sheet(self, snap: dict[str, Any]) -> BalanceSheet:
		cash = Decimal(str(snap.get("cash_on_hand", 0)))
		bank = Decimal(str(snap.get("bank_balances", 0)))
		govt = Decimal(str(snap.get("government_securities", 0)))
		other_liquid = Decimal(str(snap.get("other_liquid_assets", 0)))
		gross_loans = Decimal(str(snap.get("gross_loan_portfolio", 0)))
		provisions = Decimal(str(snap.get("loan_loss_provisions", 0)))
		other_investments = Decimal(str(snap.get("other_investments", 0)))
		fixed_assets = Decimal(str(snap.get("fixed_assets", 0)))
		other_assets = Decimal(str(snap.get("other_assets", 0)))

		net_loans = gross_loans - provisions
		total_assets = cash + bank + govt + other_liquid + net_loans + other_investments + fixed_assets + other_assets

		member_deposits = Decimal(str(snap.get("member_deposits", 0)))
		borrowings = Decimal(str(snap.get("external_borrowings", 0)))
		other_liabilities = Decimal(str(snap.get("other_liabilities", 0)))
		total_liabilities = member_deposits + borrowings + other_liabilities

		share_capital = Decimal(str(snap.get("share_capital", 0)))
		retained = Decimal(str(snap.get("retained_earnings", 0)))
		statutory_reserve = Decimal(str(snap.get("statutory_reserve", 0)))
		other_reserves = Decimal(str(snap.get("other_reserves", 0)))
		total_equity = share_capital + retained + statutory_reserve + other_reserves

		return BalanceSheet(
			cash_and_bank=cash + bank,
			government_securities=govt,
			other_liquid_assets=other_liquid,
			gross_loan_portfolio=gross_loans,
			loan_loss_provisions=provisions,
			net_loan_portfolio=net_loans,
			other_investments=other_investments,
			fixed_assets=fixed_assets,
			other_assets=other_assets,
			total_assets=total_assets,
			member_deposits=member_deposits,
			external_borrowings=borrowings,
			other_liabilities=other_liabilities,
			total_liabilities=total_liabilities,
			share_capital=share_capital,
			retained_earnings=retained,
			statutory_reserve=statutory_reserve,
			other_reserves=other_reserves,
			total_equity=total_equity,
		)

	def _build_income_statement(self, snap: dict[str, Any]) -> IncomeStatement:
		ii_loans = Decimal(str(snap.get("interest_income_loans", 0)))
		ii_inv = Decimal(str(snap.get("interest_income_investments", 0)))
		fee_income = Decimal(str(snap.get("fee_income", 0)))
		other_income = Decimal(str(snap.get("other_income", 0)))
		total_income = ii_loans + ii_inv + fee_income + other_income

		ie_dep = Decimal(str(snap.get("interest_expense_deposits", 0)))
		ie_bor = Decimal(str(snap.get("interest_expense_borrowings", 0)))
		provisions = Decimal(str(snap.get("provision_expense", 0)))
		staff = Decimal(str(snap.get("staff_costs", 0)))
		admin = Decimal(str(snap.get("administrative_expenses", 0)))
		other_exp = Decimal(str(snap.get("other_expenses", 0)))
		total_expenses = ie_dep + ie_bor + provisions + staff + admin + other_exp

		return IncomeStatement(
			interest_income_loans=ii_loans,
			interest_income_investments=ii_inv,
			fee_income=fee_income,
			other_income=other_income,
			total_income=total_income,
			interest_expense_deposits=ie_dep,
			interest_expense_borrowings=ie_bor,
			provision_for_loan_losses=provisions,
			staff_costs=staff,
			administrative_expenses=admin,
			other_expenses=other_exp,
			total_expenses=total_expenses,
			net_surplus_deficit=total_income - total_expenses,
		)

	# ── Quarterly Return ──────────────────────────────────────────────────────

	async def generate_quarterly_return(
		self,
		tenant_id: str | None = None,
		year: int | None = None,
		quarter: int | None = None,
	) -> QuarterlyReturn:
		"""Generate a complete SASRA quarterly prudential return (Forms 1-5)."""
		t = self._t(tenant_id)
		y = year or date.today().year
		q = quarter or ((date.today().month - 1) // 3)
		assert 1 <= q <= 4, f"quarter must be 1-4, got {q}"

		period_end = self._quarter_end_date(y, q)
		d = period_end.isoformat()
		snap = self._get_ledger(t, d)

		form1 = self._build_balance_sheet(snap)
		form2 = self._build_income_statement(snap)
		form3 = await self.calculate_capital_adequacy(t, d)
		form4 = await self.calculate_liquidity_ratio(t, d)
		form5 = await self.classify_loan_portfolio(t, d)

		violations: list[str] = []
		warnings: list[str] = []

		if not form3.compliant:
			violations.append(f"CAR {form3.capital_adequacy_ratio:.2f}% < {_CAR_MIN}% minimum")
		elif form3.traffic_light == TrafficLight.AMBER:
			warnings.append(f"CAR {form3.capital_adequacy_ratio:.2f}% is within 2pp of minimum")

		if not form4.compliant:
			violations.append(f"Liquidity {form4.liquidity_ratio:.2f}% < {_LIQUIDITY_MIN}% minimum")
		elif form4.traffic_light == TrafficLight.AMBER:
			warnings.append(f"Liquidity {form4.liquidity_ratio:.2f}% is within 2pp of minimum")

		ldr = await self.calculate_loan_to_deposit_ratio(t, d)
		if ldr > _LDR_MAX:
			violations.append(f"LDR {ldr:.2f}% > {_LDR_MAX}% maximum")

		npl = form5.npl_ratio
		if npl >= _NPL_BREACH:
			violations.append(f"NPL ratio {npl:.2f}% >= {_NPL_BREACH}% breach threshold")
		elif npl >= _NPL_WARN:
			warnings.append(f"NPL ratio {npl:.2f}% >= {_NPL_WARN}% warning threshold")

		if form1.total_assets > 0:
			core_ratio = self._pct(form3.core_capital, form1.total_assets)
			if core_ratio < _CORE_CAPITAL_RATIO_MIN:
				violations.append(f"Core capital ratio {core_ratio:.2f}% < {_CORE_CAPITAL_RATIO_MIN}% minimum")

		ret = QuarterlyReturn(
			tenant_id=t,
			year=y,
			quarter=q,
			period_end=d,
			generated_at=self._now(),
			form1_balance_sheet=form1,
			form2_income_statement=form2,
			form3_capital_adequacy=form3,
			form4_liquidity=form4,
			form5_loan_classification=form5,
			overall_compliant=len(violations) == 0,
			violations=violations,
			warnings=warnings,
		)
		self._emit(t, "quarterly_return_generated", {"id": ret.id, "period": f"{y}-Q{q}"})
		_log.info("[SASRA] Quarterly return generated: %s Q%s tenant=%s violations=%d", y, q, t, len(violations))
		return ret

	# ── Annual Report ─────────────────────────────────────────────────────────

	async def generate_annual_report(
		self,
		tenant_id: str | None = None,
		year: int | None = None,
	) -> AnnualReport:
		"""Full-year SASRA annual return with key ratios and board pack data."""
		t = self._t(tenant_id)
		y = year or date.today().year
		year_end = f"{y}-12-31"
		snap = self._get_ledger(t, year_end)

		bs = self._build_balance_sheet(snap)
		inc = self._build_income_statement(snap)
		cap = await self.calculate_capital_adequacy(t, year_end)
		liq = await self.calculate_liquidity_ratio(t, year_end)
		lc = await self.classify_loan_portfolio(t, year_end)
		ldr = await self.calculate_loan_to_deposit_ratio(t, year_end)

		key_ratios: dict[str, Any] = {
			"capital_adequacy_ratio_pct": str(cap.capital_adequacy_ratio),
			"core_capital_ratio_pct": str(cap.core_capital_ratio),
			"liquidity_ratio_pct": str(liq.liquidity_ratio),
			"loan_to_deposit_ratio_pct": str(ldr),
			"npl_ratio_pct": str(lc.npl_ratio),
			"par30_pct": str(lc.par30),
			"par90_pct": str(lc.par90),
			"provisioning_coverage_pct": str(lc.provisioning_coverage),
			"return_on_assets_pct": str(self._pct(inc.net_surplus_deficit, bs.total_assets)),
		}

		# Quarterly snapshots
		quarterly: list[dict[str, Any]] = []
		for q in range(1, 5):
			try:
				qr = await self.generate_quarterly_return(t, y, q)
				quarterly.append({
					"quarter": q,
					"period_end": qr.period_end,
					"compliant": qr.overall_compliant,
					"violations": qr.violations,
				})
			except Exception:
				quarterly.append({"quarter": q, "period_end": self._quarter_end_date(y, q).isoformat(), "compliant": None})

		report = AnnualReport(
			tenant_id=t,
			year=y,
			generated_at=self._now(),
			balance_sheet=bs,
			income_statement=inc,
			capital_adequacy=cap,
			liquidity=liq,
			loan_classification=lc,
			key_ratios=key_ratios,
			quarterly_snapshots=quarterly,
		)
		self._emit(t, "annual_report_generated", {"id": report.id, "year": y})
		_log.info("[SASRA] Annual report generated: %s tenant=%s", y, t)
		return report

	# ── Compliance Status ─────────────────────────────────────────────────────

	async def check_regulatory_compliance(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> ComplianceStatus:
		"""Check all key SASRA ratios and return compliance status with traffic lights."""
		t = self._t(tenant_id)
		d = as_of_date or self._today()

		cap = await self.calculate_capital_adequacy(t, d)
		liq = await self.calculate_liquidity_ratio(t, d)
		ldr = await self.calculate_loan_to_deposit_ratio(t, d)
		lc = await self.classify_loan_portfolio(t, d)

		snap = self._get_ledger(t, d)
		core_cap = cap.core_capital
		total_assets = cap.total_assets
		core_ratio = self._pct(core_cap, total_assets)

		violations: list[str] = []
		warnings: list[str] = []
		ratios: list[RatioStatus] = []

		# CAR
		car_ok, car_tl = self._traffic_light_ratio(cap.capital_adequacy_ratio, _CAR_MIN, None)
		ratios.append(RatioStatus(
			name="Capital Adequacy Ratio",
			actual=cap.capital_adequacy_ratio,
			minimum=_CAR_MIN,
			compliant=car_ok,
			traffic_light=car_tl,
			description=f"Institutional capital / risk-weighted assets. Min {_CAR_MIN}%",
		))
		if not car_ok:
			violations.append(f"CAR {cap.capital_adequacy_ratio:.2f}% below {_CAR_MIN}% minimum")
		elif car_tl == TrafficLight.AMBER:
			warnings.append(f"CAR {cap.capital_adequacy_ratio:.2f}% within warning band")

		# Core capital ratio
		cc_ok, cc_tl = self._traffic_light_ratio(core_ratio, _CORE_CAPITAL_RATIO_MIN, None)
		ratios.append(RatioStatus(
			name="Core Capital / Total Assets",
			actual=core_ratio,
			minimum=_CORE_CAPITAL_RATIO_MIN,
			compliant=cc_ok,
			traffic_light=cc_tl,
			description=f"Paid-up share capital + retained earnings / total assets. Min {_CORE_CAPITAL_RATIO_MIN}%",
		))
		if not cc_ok:
			violations.append(f"Core capital ratio {core_ratio:.2f}% below {_CORE_CAPITAL_RATIO_MIN}% minimum")
		elif cc_tl == TrafficLight.AMBER:
			warnings.append(f"Core capital ratio {core_ratio:.2f}% within warning band")

		# Liquidity
		liq_ok, liq_tl = self._traffic_light_ratio(liq.liquidity_ratio, _LIQUIDITY_MIN, None)
		ratios.append(RatioStatus(
			name="Liquidity Ratio",
			actual=liq.liquidity_ratio,
			minimum=_LIQUIDITY_MIN,
			compliant=liq_ok,
			traffic_light=liq_tl,
			description=f"Liquid assets / (deposits + borrowings). Min {_LIQUIDITY_MIN}%",
		))
		if not liq_ok:
			violations.append(f"Liquidity {liq.liquidity_ratio:.2f}% below {_LIQUIDITY_MIN}% minimum")
		elif liq_tl == TrafficLight.AMBER:
			warnings.append(f"Liquidity {liq.liquidity_ratio:.2f}% within warning band")

		# LDR
		ldr_ok, ldr_tl = self._traffic_light_ratio(ldr, None, _LDR_MAX)
		ratios.append(RatioStatus(
			name="Loan to Deposit Ratio",
			actual=ldr,
			maximum=_LDR_MAX,
			compliant=ldr_ok,
			traffic_light=ldr_tl,
			description=f"Gross loans / member deposits. Max {_LDR_MAX}%",
		))
		if not ldr_ok:
			violations.append(f"LDR {ldr:.2f}% exceeds {_LDR_MAX}% maximum")
		elif ldr_tl == TrafficLight.AMBER:
			warnings.append(f"LDR {ldr:.2f}% approaching maximum")

		# NPL
		npl_ok = lc.npl_ratio < _NPL_BREACH
		npl_tl = TrafficLight.GREEN
		if lc.npl_ratio >= _NPL_BREACH:
			npl_tl = TrafficLight.RED
		elif lc.npl_ratio >= _NPL_WARN:
			npl_tl = TrafficLight.AMBER
		ratios.append(RatioStatus(
			name="NPL Ratio",
			actual=lc.npl_ratio,
			maximum=_NPL_BREACH,
			compliant=npl_ok,
			traffic_light=npl_tl,
			description=f"Non-performing loans / gross portfolio. Warning >{_NPL_WARN}%, Breach >{_NPL_BREACH}%",
		))
		if not npl_ok:
			violations.append(f"NPL ratio {lc.npl_ratio:.2f}% >= {_NPL_BREACH}% breach")
		elif lc.npl_ratio >= _NPL_WARN:
			warnings.append(f"NPL ratio {lc.npl_ratio:.2f}% >= {_NPL_WARN}% warning")

		# PAR30
		par30_ok = lc.par30 < Decimal("15")
		par30_tl = TrafficLight.GREEN if par30_ok else TrafficLight.RED
		ratios.append(RatioStatus(
			name="PAR30",
			actual=lc.par30,
			maximum=Decimal("15"),
			compliant=par30_ok,
			traffic_light=par30_tl,
			description="Portfolio at risk >30 DPD / gross portfolio",
		))

		# Provisioning coverage
		prov_ok = lc.provisioning_coverage >= Decimal("100")
		prov_tl = TrafficLight.GREEN if prov_ok else (TrafficLight.AMBER if lc.provisioning_coverage >= Decimal("80") else TrafficLight.RED)
		ratios.append(RatioStatus(
			name="Provisioning Coverage",
			actual=lc.provisioning_coverage,
			minimum=Decimal("100"),
			compliant=prov_ok,
			traffic_light=prov_tl,
			description="Actual provisions / required provisions * 100. Min 100%",
		))
		if not prov_ok:
			violations.append(f"Provisioning coverage {lc.provisioning_coverage:.2f}% < 100%")

		status = ComplianceStatus(
			tenant_id=t,
			as_of_date=d,
			overall_compliant=len(violations) == 0,
			violations=violations,
			warnings=warnings,
			ratios=ratios,
			checked_at=self._now(),
		)
		self._emit(t, "compliance_checked", {"violations": len(violations), "warnings": len(warnings)})
		return status

	# ── Filing Registry ───────────────────────────────────────────────────────

	async def file_return(
		self,
		tenant_id: str | None,
		return_type: ReturnType,
		period: str,
		data: dict[str, Any],
		filing_officer: str,
		submitted_at: str | None = None,
	) -> FilingRecord:
		"""Record a SASRA return submission. Actual portal upload is external."""
		t = self._t(tenant_id)
		assert period, "period must be non-empty"
		assert filing_officer, "filing_officer must be non-empty"

		rec = FilingRecord(
			tenant_id=t,
			return_type=return_type,
			period=period,
			filing_officer=filing_officer,
			submitted_at=submitted_at or self._now(),
			filing_status=FilingStatus.SUBMITTED,
			data_snapshot=deepcopy(data),
			created_at=self._now(),
		)
		self._filings[rec.id] = rec
		self._emit(t, "return_filed", {"id": rec.id, "period": period, "type": return_type})
		_log.info("[SASRA] Return filed: %s period=%s officer=%s", return_type, period, filing_officer)
		return rec

	async def get_filing_history(
		self,
		tenant_id: str | None = None,
		from_date: str | None = None,
		to_date: str | None = None,
	) -> list[FilingRecord]:
		"""All filed returns for a tenant, optionally filtered by submission date."""
		t = self._t(tenant_id)
		items = [f for f in self._filings.values() if f.tenant_id == t]
		if from_date:
			items = [f for f in items if f.submitted_at >= from_date]
		if to_date:
			items = [f for f in items if f.submitted_at <= to_date]
		return sorted(items, key=lambda f: f.submitted_at, reverse=True)

	# ── Regulatory Calendar ───────────────────────────────────────────────────

	async def get_regulatory_calendar(
		self,
		tenant_id: str | None = None,
		year: int | None = None,
	) -> list[RegulatoryDeadline]:
		"""Return all SASRA filing deadlines for a given year."""
		t = self._t(tenant_id)
		y = year or date.today().year
		today = date.today()
		deadlines: list[RegulatoryDeadline] = []

		filed_periods = {f.period for f in self._filings.values() if f.tenant_id == t}
		filed_map = {f.period: f for f in self._filings.values() if f.tenant_id == t}

		for q in range(1, 5):
			qe = self._quarter_end_date(y, q)
			due = qe + timedelta(days=_QUARTERLY_FILING_DAYS)
			period = f"{y}-Q{q}"
			days_rem = (due - today).days
			filing = filed_map.get(period)
			deadlines.append(RegulatoryDeadline(
				period=period,
				return_type=ReturnType.QUARTERLY,
				due_date=due.isoformat(),
				description=f"SASRA Quarterly Prudential Return Q{q} {y}",
				days_remaining=days_rem,
				overdue=days_rem < 0,
				filed=period in filed_periods,
				filing_id=filing.id if filing else None,
			))

		# Annual: due 4 months after year-end (Dec 31 → April 30 next year)
		annual_due = date(y + 1, 4, 30)
		annual_period = f"{y}-annual"
		filing = filed_map.get(annual_period)
		deadlines.append(RegulatoryDeadline(
			period=annual_period,
			return_type=ReturnType.ANNUAL,
			due_date=annual_due.isoformat(),
			description=f"SASRA Annual Audited Accounts {y}",
			days_remaining=(annual_due - today).days,
			overdue=(annual_due - today).days < 0,
			filed=annual_period in filed_periods,
			filing_id=filing.id if filing else None,
		))

		return sorted(deadlines, key=lambda d: d.due_date)

	async def get_pending_filings(self, tenant_id: str | None = None) -> list[RegulatoryDeadline]:
		"""Returns overdue and upcoming (within 30 days) filing deadlines."""
		t = self._t(tenant_id)
		cal = await self.get_regulatory_calendar(t)
		return [d for d in cal if d.overdue or (0 <= d.days_remaining <= 30)]

	# ── Compliance Dashboard ──────────────────────────────────────────────────

	async def get_compliance_dashboard(
		self,
		tenant_id: str | None = None,
		as_of_date: str | None = None,
	) -> ComplianceDashboard:
		"""All ratios with traffic-light status plus pending filings."""
		t = self._t(tenant_id)
		d = as_of_date or self._today()
		compliance = await self.check_regulatory_compliance(t, d)
		pending = await self.get_pending_filings(t)

		filings = await self.get_filing_history(t)
		last_filing = filings[0] if filings else None

		# Overall: RED if any violation, AMBER if any warning, GREEN otherwise
		if not compliance.overall_compliant:
			overall = TrafficLight.RED
		elif compliance.warnings:
			overall = TrafficLight.AMBER
		else:
			overall = TrafficLight.GREEN

		return ComplianceDashboard(
			tenant_id=t,
			as_of_date=d,
			overall_status=overall,
			ratios=compliance.ratios,
			pending_filings=pending,
			last_filing=last_filing,
			generated_at=self._now(),
		)

	# ── Board Report ──────────────────────────────────────────────────────────

	async def generate_board_report(
		self,
		tenant_id: str | None = None,
		period: str | None = None,
	) -> dict[str, Any]:
		"""Board pack: key ratios, compliance summary, trend narrative, pending filings."""
		t = self._t(tenant_id)
		d = period or self._today()
		compliance = await self.check_regulatory_compliance(t, d)
		pending = await self.get_pending_filings(t)
		snap = self._get_ledger(t, d)
		ldr = await self.calculate_loan_to_deposit_ratio(t, d)
		lc = await self.classify_loan_portfolio(t, d)

		risk_summary = "ALL RATIOS COMPLIANT" if compliance.overall_compliant else f"REGULATORY BREACHES: {'; '.join(compliance.violations)}"

		return {
			"period": d,
			"tenant_id": t,
			"generated_at": self._now(),
			"executive_summary": risk_summary,
			"overall_compliant": compliance.overall_compliant,
			"violations": compliance.violations,
			"warnings": compliance.warnings,
			"key_ratios": {r.name: {"actual": str(r.actual), "status": r.traffic_light} for r in compliance.ratios},
			"loan_portfolio": {
				"gross_portfolio": str(lc.total_gross_portfolio),
				"npl_ratio_pct": str(lc.npl_ratio),
				"par30_pct": str(lc.par30),
				"par90_pct": str(lc.par90),
				"provisioning_coverage_pct": str(lc.provisioning_coverage),
			},
			"pending_filings": [
				{
					"period": pf.period,
					"return_type": pf.return_type,
					"due_date": pf.due_date,
					"days_remaining": pf.days_remaining,
					"overdue": pf.overdue,
				}
				for pf in pending
			],
		}

	# ── SASRA XML Return ──────────────────────────────────────────────────────

	async def generate_sasra_xml_return(
		self,
		tenant_id: str | None = None,
		year: int | None = None,
		quarter: int | None = None,
	) -> str:
		"""Generate SASRA-portal-compatible XML for quarterly prudential return."""
		t = self._t(tenant_id)
		y = year or date.today().year
		q = quarter or ((date.today().month - 1) // 3 + 1)
		assert 1 <= q <= 4

		qr = await self.generate_quarterly_return(t, y, q)
		snap = self._get_ledger(t, qr.period_end)
		sacco_name = snap.get("sacco_name", t)
		reg_no = snap.get("registration_number", "")

		root = Element("SASRAReturn")
		root.set("version", "2.0")
		root.set("generated", self._now())

		header = SubElement(root, "Header")
		SubElement(header, "SACCOName").text = str(sacco_name)
		SubElement(header, "RegistrationNumber").text = str(reg_no)
		SubElement(header, "Year").text = str(y)
		SubElement(header, "Quarter").text = str(q)
		SubElement(header, "PeriodEnd").text = qr.period_end
		SubElement(header, "ReturnType").text = "QUARTERLY_PRUDENTIAL"

		# Form 1
		f1 = SubElement(root, "Form1BalanceSheet")
		bs = qr.form1_balance_sheet
		for field, val in bs.model_dump().items():
			SubElement(f1, field).text = str(val)

		# Form 2
		f2 = SubElement(root, "Form2IncomeStatement")
		inc = qr.form2_income_statement
		for field, val in inc.model_dump().items():
			SubElement(f2, field).text = str(val)

		# Form 3
		if qr.form3_capital_adequacy:
			f3 = SubElement(root, "Form3CapitalAdequacy")
			for field, val in qr.form3_capital_adequacy.model_dump().items():
				SubElement(f3, field).text = str(val)

		# Form 4
		if qr.form4_liquidity:
			f4 = SubElement(root, "Form4Liquidity")
			for field, val in qr.form4_liquidity.model_dump().items():
				SubElement(f4, field).text = str(val)

		# Form 5
		if qr.form5_loan_classification:
			f5 = SubElement(root, "Form5LoanClassification")
			lc = qr.form5_loan_classification
			SubElement(f5, "GrossPortfolio").text = str(lc.total_gross_portfolio)
			SubElement(f5, "NPLRatio").text = str(lc.npl_ratio)
			SubElement(f5, "PAR30").text = str(lc.par30)
			bands_el = SubElement(f5, "Bands")
			for band in lc.bands:
				b_el = SubElement(bands_el, "Band")
				SubElement(b_el, "Name").text = band.band.value
				SubElement(b_el, "DPDRange").text = band.dpd_range
				SubElement(b_el, "OutstandingBalance").text = str(band.outstanding_balance)
				SubElement(b_el, "ProvisionRate").text = str(band.provision_rate)
				SubElement(b_el, "RequiredProvision").text = str(band.required_provision)

		# Compliance
		comp_el = SubElement(root, "ComplianceSummary")
		SubElement(comp_el, "OverallCompliant").text = str(qr.overall_compliant)
		if qr.violations:
			viol_el = SubElement(comp_el, "Violations")
			for v in qr.violations:
				SubElement(viol_el, "Item").text = v

		return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding="unicode")

	# ── Health Check ─────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"filings_recorded": len(self._filings),
			"tenants_with_ledger_data": len(self._ledger),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "fintech",
			"description": "SASRA prudential returns (Forms 1-5), ratio monitoring, filing registry, compliance dashboard",
			"sasra_minimums": {
				"capital_adequacy_ratio_min_pct": str(_CAR_MIN),
				"core_capital_ratio_min_pct": str(_CORE_CAPITAL_RATIO_MIN),
				"liquidity_ratio_min_pct": str(_LIQUIDITY_MIN),
				"loan_to_deposit_ratio_max_pct": str(_LDR_MAX),
				"npl_warning_pct": str(_NPL_WARN),
				"npl_breach_pct": str(_NPL_BREACH),
			},
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._t(tenant_id)
		return [deepcopy(e) for e in self._audit if e["tenant_id"] == t]


# Alias
SACCARegulatoryReportingService = SACCARegulatoryService
