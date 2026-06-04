"""
APG Payroll — Pydantic v2 data models.

Covers the full payroll lifecycle: employee configuration, pay periods,
payroll runs, payslip lines, PAYE/statutory calculations, leave balances,
GL posting, bank transfers, and terminal benefits.

All monetary values are Decimal(15,2). Tenant isolation is enforced at the
service layer; every model carries tenant_id.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Common config
# ---------------------------------------------------------------------------

_CFG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EmploymentType(str, Enum):
	PERMANENT = "permanent"
	CONTRACT = "contract"
	CASUAL = "casual"
	INTERN = "intern"
	EXPATRIATE = "expatriate"


class PayFrequency(str, Enum):
	WEEKLY = "weekly"
	BIWEEKLY = "biweekly"
	SEMIMONTHLY = "semimonthly"
	MONTHLY = "monthly"


class PeriodStatus(str, Enum):
	OPEN = "open"
	LOCKED = "locked"
	CLOSED = "closed"


class RunStatus(str, Enum):
	DRAFT = "draft"
	CALCULATED = "calculated"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	POSTED = "posted"
	PAID = "paid"
	REVERSED = "reversed"
	CANCELLED = "cancelled"


class LineElementType(str, Enum):
	BASIC = "basic"
	ALLOWANCE = "allowance"
	OVERTIME = "overtime"
	BONUS = "bonus"
	COMMISSION = "commission"
	BACK_PAY = "back_pay"
	PAYE = "paye"
	NSSF = "nssf"
	NHIF = "nhif"
	NHIF_SHI = "nhif_shi"        # Social Health Insurance (Kenya 2024+)
	NITA = "nita"
	SDL = "sdl"                  # Skills Development Levy (Tanzania)
	WCF = "wcf"                  # Workers Compensation Fund (Tanzania)
	NAPSA = "napsa"              # Zambia
	SSNIT = "ssnit"              # Ghana
	PENCOM = "pencom"            # Nigeria
	PENSION = "pension"
	LOAN_RECOVERY = "loan_recovery"
	GARNISHMENT = "garnishment"
	ADVANCE_RECOVERY = "advance_recovery"
	OTHER_DEDUCTION = "other_deduction"
	EMPLOYER_NSSF = "employer_nssf"
	EMPLOYER_NHIF = "employer_nhif"
	EMPLOYER_PENSION = "employer_pension"
	BENEFIT_IN_KIND = "benefit_in_kind"


class LeaveType(str, Enum):
	ANNUAL = "annual"
	SICK = "sick"
	MATERNITY = "maternity"
	PATERNITY = "paternity"
	COMPASSIONATE = "compassionate"
	STUDY = "study"
	UNPAID = "unpaid"


class GlEntryType(str, Enum):
	DEBIT = "debit"
	CREDIT = "credit"


class BankFormat(str, Enum):
	KENYA_EFT = "kenya_eft"
	KENYA_RTGS = "kenya_rtgs"
	MPESA_B2C = "mpesa_b2c"
	SWIFT_MT103 = "swift_mt103"
	ACH_NACHA = "ach_nacha"
	SEPA = "sepa"
	GENERIC_CSV = "generic_csv"


class Country(str, Enum):
	KE = "KE"   # Kenya
	TZ = "TZ"   # Tanzania
	UG = "UG"   # Uganda
	RW = "RW"   # Rwanda
	ZM = "ZM"   # Zambia
	GH = "GH"   # Ghana
	NG = "NG"   # Nigeria
	ZA = "ZA"   # South Africa
	ET = "ET"   # Ethiopia
	ZW = "ZW"   # Zimbabwe
	US = "US"
	GB = "GB"
	OTHER = "OTHER"


class PaymentMethod(str, Enum):
	BANK_EFT = "bank_eft"
	CASH = "cash"
	CHEQUE = "cheque"
	MPESA = "mpesa"
	AIRTEL_MONEY = "airtel_money"


class TerminalBenefitType(str, Enum):
	NOTICE_PAY = "notice_pay"
	SEVERANCE = "severance"
	GRATUITY = "gratuity"
	LEAVE_ENCASHMENT = "leave_encashment"
	UNPAID_WAGES = "unpaid_wages"
	PENSION_REFUND = "pension_refund"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _positive(v: Decimal) -> Decimal:
	if v < 0:
		raise ValueError("must be non-negative")
	return v


NonNegativeDecimal = Annotated[Decimal, AfterValidator(_positive)]


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class PRBase(BaseModel):
	"""Every payroll entity carries these audit fields."""
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# 1. Employee (payroll view — thin snapshot, not the HCM master)
# ---------------------------------------------------------------------------

class PREmployeeCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	employee_number: str
	full_name: str
	national_id: str
	tax_pin: str                          # KRA PIN / TIN
	nssf_number: str | None = None
	nhif_number: str | None = None
	bank_code: str | None = None
	bank_branch_code: str | None = None
	bank_account_number: str | None = None
	bank_account_name: str | None = None
	mobile_money_number: str | None = None
	department_id: str | None = None
	department_name: str | None = None
	cost_center: str | None = None
	employment_type: EmploymentType = EmploymentType.PERMANENT
	hire_date: date
	termination_date: date | None = None
	salary_grade: str | None = None
	basic_salary: Decimal = Decimal("0")
	currency: str = "KES"
	country: Country = Country.KE
	payment_method: PaymentMethod = PaymentMethod.BANK_EFT
	pay_frequency: PayFrequency = PayFrequency.MONTHLY
	is_expatriate: bool = False
	tax_exemption_certificate: str | None = None
	created_by: str = "system"


class PREmployeeUpdate(BaseModel):
	model_config = _CFG

	full_name: str | None = None
	bank_account_number: str | None = None
	bank_account_name: str | None = None
	bank_code: str | None = None
	bank_branch_code: str | None = None
	mobile_money_number: str | None = None
	department_id: str | None = None
	cost_center: str | None = None
	basic_salary: Decimal | None = None
	salary_grade: str | None = None
	payment_method: PaymentMethod | None = None
	termination_date: date | None = None
	nhif_number: str | None = None
	nssf_number: str | None = None
	updated_by: str = "system"


class PREmployeeResponse(PRBase):
	employee_number: str
	full_name: str
	national_id: str
	tax_pin: str
	nssf_number: str | None = None
	nhif_number: str | None = None
	bank_code: str | None = None
	bank_branch_code: str | None = None
	bank_account_number: str | None = None
	bank_account_name: str | None = None
	mobile_money_number: str | None = None
	department_id: str | None = None
	department_name: str | None = None
	cost_center: str | None = None
	employment_type: EmploymentType = EmploymentType.PERMANENT
	hire_date: date
	termination_date: date | None = None
	salary_grade: str | None = None
	basic_salary: Decimal = Decimal("0")
	currency: str = "KES"
	country: Country = Country.KE
	payment_method: PaymentMethod = PaymentMethod.BANK_EFT
	pay_frequency: PayFrequency = PayFrequency.MONTHLY
	is_expatriate: bool = False
	is_active: bool = True


# ---------------------------------------------------------------------------
# 2. Pay Period
# ---------------------------------------------------------------------------

class PRPayPeriodCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	period_code: str              # e.g. "2026-01"
	pay_frequency: PayFrequency
	start_date: date
	end_date: date
	pay_date: date
	currency: str = "KES"
	country: Country = Country.KE
	notes: str | None = None
	created_by: str = "system"


class PRPayPeriodUpdate(BaseModel):
	model_config = _CFG

	pay_date: date | None = None
	status: PeriodStatus | None = None
	notes: str | None = None
	updated_by: str = "system"


class PRPayPeriodResponse(PRBase):
	period_code: str
	pay_frequency: PayFrequency
	start_date: date
	end_date: date
	pay_date: date
	status: PeriodStatus = PeriodStatus.OPEN
	currency: str = "KES"
	country: Country = Country.KE
	notes: str | None = None


# ---------------------------------------------------------------------------
# 3. Payroll Run
# ---------------------------------------------------------------------------

class PRPayrollRunCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	period_id: str
	run_number: int = 1
	description: str | None = None
	include_employee_ids: list[str] | None = None   # None = all active
	is_bonus_run: bool = False
	is_supplementary: bool = False
	created_by: str = "system"


class PRPayrollRunUpdate(BaseModel):
	model_config = _CFG

	status: RunStatus | None = None
	description: str | None = None
	updated_by: str = "system"


class PRPayrollRunResponse(PRBase):
	period_id: str
	run_number: int
	status: RunStatus = RunStatus.DRAFT
	description: str | None = None
	is_bonus_run: bool = False
	is_supplementary: bool = False
	total_gross: Decimal = Decimal("0")
	total_deductions: Decimal = Decimal("0")
	total_taxes: Decimal = Decimal("0")
	total_net: Decimal = Decimal("0")
	employee_count: int = 0
	approved_by: str | None = None
	approved_at: datetime | None = None
	posted_by: str | None = None
	posted_at: datetime | None = None
	reversed_by: str | None = None
	reversal_reason: str | None = None


# ---------------------------------------------------------------------------
# 4. Payslip / Payroll Line Item
# ---------------------------------------------------------------------------

class PRPayslipLineCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	run_id: str
	employee_id: str
	element_type: LineElementType
	element_name: str
	amount: Decimal
	is_taxable: bool = True
	is_pensionable: bool = True
	is_employer_contribution: bool = False
	notes: str | None = None
	created_by: str = "system"


class PRPayslipLineResponse(PRBase):
	run_id: str
	employee_id: str
	element_type: LineElementType
	element_name: str
	amount: Decimal
	is_taxable: bool
	is_pensionable: bool
	is_employer_contribution: bool
	notes: str | None = None


# ---------------------------------------------------------------------------
# 5. PAYE Tax Calculation
# ---------------------------------------------------------------------------

class PRTaxBandResult(BaseModel):
	"""One tax band's contribution to PAYE."""
	model_config = _CFG

	band_min: Decimal
	band_max: Decimal | None
	rate: Decimal                   # e.g. 0.10
	taxable_in_band: Decimal
	tax_in_band: Decimal


class PRTaxCalculationCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	run_id: str
	employee_id: str
	country: Country
	gross_income: Decimal
	taxable_income: Decimal
	personal_relief: Decimal = Decimal("0")
	insurance_relief: Decimal = Decimal("0")
	mortgage_relief: Decimal = Decimal("0")
	other_relief: Decimal = Decimal("0")
	bands_applied: list[PRTaxBandResult] = Field(default_factory=list)
	gross_tax: Decimal = Decimal("0")
	tax_relief_total: Decimal = Decimal("0")
	paye_amount: Decimal = Decimal("0")
	tax_code: str | None = None
	created_by: str = "system"


class PRTaxCalculationResponse(PRBase):
	run_id: str
	employee_id: str
	country: Country
	gross_income: Decimal
	taxable_income: Decimal
	personal_relief: Decimal
	insurance_relief: Decimal
	mortgage_relief: Decimal
	other_relief: Decimal
	bands_applied: list[PRTaxBandResult]
	gross_tax: Decimal
	tax_relief_total: Decimal
	paye_amount: Decimal
	tax_code: str | None = None


# ---------------------------------------------------------------------------
# 6. Statutory Deductions
# ---------------------------------------------------------------------------

class PRStatutoryDeductionCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	run_id: str
	employee_id: str
	country: Country
	deduction_type: LineElementType   # must be a statutory type
	employee_amount: Decimal = Decimal("0")
	employer_amount: Decimal = Decimal("0")
	basis: Decimal = Decimal("0")     # wages/pensionable pay used for calc
	rate_used: Decimal | None = None
	cap_applied: bool = False
	notes: str | None = None
	created_by: str = "system"


class PRStatutoryDeductionResponse(PRBase):
	run_id: str
	employee_id: str
	country: Country
	deduction_type: LineElementType
	employee_amount: Decimal
	employer_amount: Decimal
	basis: Decimal
	rate_used: Decimal | None
	cap_applied: bool
	notes: str | None = None


# ---------------------------------------------------------------------------
# 7. Leave Balance
# ---------------------------------------------------------------------------

class PRLeaveBalanceCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	employee_id: str
	leave_type: LeaveType
	year: int
	entitled_days: Decimal
	taken_days: Decimal = Decimal("0")
	carried_forward: Decimal = Decimal("0")
	created_by: str = "system"


class PRLeaveBalanceUpdate(BaseModel):
	model_config = _CFG

	taken_days: Decimal | None = None
	carried_forward: Decimal | None = None
	updated_by: str = "system"


class PRLeaveBalanceResponse(PRBase):
	employee_id: str
	leave_type: LeaveType
	year: int
	entitled_days: Decimal
	taken_days: Decimal
	carried_forward: Decimal
	balance: Decimal          # computed by service
	encashed_days: Decimal = Decimal("0")
	encashed_amount: Decimal = Decimal("0")


# ---------------------------------------------------------------------------
# 8. GL Journal Entry
# ---------------------------------------------------------------------------

class PRGlEntryCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	run_id: str
	journal_date: date
	account_code: str
	account_name: str
	entry_type: GlEntryType
	amount: Decimal
	cost_center: str | None = None
	department_code: str | None = None
	reference: str | None = None
	narration: str | None = None
	created_by: str = "system"


class PRGlEntryResponse(PRBase):
	run_id: str
	journal_date: date
	account_code: str
	account_name: str
	entry_type: GlEntryType
	amount: Decimal
	cost_center: str | None = None
	department_code: str | None = None
	reference: str | None = None
	narration: str | None = None
	is_posted: bool = False
	posted_at: datetime | None = None


# ---------------------------------------------------------------------------
# 9. Bank Transfer File
# ---------------------------------------------------------------------------

class PRBankTransferLine(BaseModel):
	model_config = _CFG

	employee_id: str
	employee_number: str
	full_name: str
	bank_code: str
	bank_branch_code: str
	account_number: str
	account_name: str
	net_pay: Decimal
	currency: str
	payment_reference: str


class PRBankTransferFileRequest(BaseModel):
	model_config = _CFG

	tenant_id: str
	run_id: str
	bank_format: BankFormat = BankFormat.KENYA_EFT
	value_date: date
	requested_by: str


class PRBankTransferFileResponse(BaseModel):
	model_config = _CFG

	run_id: str
	bank_format: BankFormat
	value_date: date
	total_amount: Decimal
	record_count: int
	file_content: str           # raw file text (CSV/fixed-width)
	filename: str


# ---------------------------------------------------------------------------
# 10. Payroll Variance Report
# ---------------------------------------------------------------------------

class PRVarianceLine(BaseModel):
	model_config = _CFG

	employee_id: str
	employee_number: str
	full_name: str
	element_name: str
	current_amount: Decimal
	previous_amount: Decimal
	variance: Decimal
	variance_pct: Decimal | None = None
	flagged: bool = False


class PRVarianceReport(BaseModel):
	model_config = _CFG

	run_id: str
	previous_run_id: str | None = None
	generated_at: datetime
	lines: list[PRVarianceLine]
	threshold_pct: Decimal = Decimal("10")
	flagged_count: int = 0


# ---------------------------------------------------------------------------
# 11. P9 Annual Tax Return (Kenya)
# ---------------------------------------------------------------------------

class PRP9MonthDetail(BaseModel):
	model_config = _CFG

	month: int
	gross_pay: Decimal
	benefits_in_kind: Decimal = Decimal("0")
	value_of_quarters: Decimal = Decimal("0")
	total_chargeable_pay: Decimal
	paye_withheld: Decimal
	paye_paid_to_kra: Decimal


class PRP9FormResponse(BaseModel):
	model_config = _CFG

	employee_id: str
	employee_number: str
	full_name: str
	tax_pin: str
	nhif_number: str | None = None
	nssf_number: str | None = None
	year: int
	months: list[PRP9MonthDetail]
	total_gross: Decimal
	total_benefits_in_kind: Decimal
	total_chargeable_pay: Decimal
	total_paye_withheld: Decimal
	total_paye_paid: Decimal
	personal_relief_claimed: Decimal
	insurance_relief_claimed: Decimal


# ---------------------------------------------------------------------------
# 12. Statutory Returns Schedules
# ---------------------------------------------------------------------------

class PRNssfReturnLine(BaseModel):
	model_config = _CFG

	employee_id: str
	employee_number: str
	full_name: str
	nssf_number: str | None = None
	pensionable_wages: Decimal
	employee_contribution: Decimal
	employer_contribution: Decimal
	total: Decimal


class PRNssfReturnSchedule(BaseModel):
	model_config = _CFG

	run_id: str
	period_code: str
	generated_at: datetime
	lines: list[PRNssfReturnLine]
	total_employee: Decimal
	total_employer: Decimal
	total_contribution: Decimal


class PRNhifReturnLine(BaseModel):
	model_config = _CFG

	employee_id: str
	employee_number: str
	full_name: str
	nhif_number: str | None = None
	gross_salary: Decimal
	employee_contribution: Decimal
	employer_contribution: Decimal = Decimal("0")


class PRNhifReturnSchedule(BaseModel):
	model_config = _CFG

	run_id: str
	period_code: str
	generated_at: datetime
	lines: list[PRNhifReturnLine]
	total_employee: Decimal
	total_employer: Decimal


# ---------------------------------------------------------------------------
# 13. Overtime
# ---------------------------------------------------------------------------

class PROvertimeCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	employee_id: str
	run_id: str
	hours: Decimal
	rate_multiplier: Decimal = Decimal("1.5")   # 1.5x or 2x
	computed_amount: Decimal | None = None      # filled by service
	approved_by: str | None = None
	created_by: str = "system"


class PROvertimeResponse(PRBase):
	employee_id: str
	run_id: str
	hours: Decimal
	rate_multiplier: Decimal
	hourly_rate: Decimal
	computed_amount: Decimal
	approved_by: str | None = None


# ---------------------------------------------------------------------------
# 14. Salary Advance
# ---------------------------------------------------------------------------

class PRAdvanceCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	employee_id: str
	amount: Decimal
	disbursement_date: date
	recovery_start_period_id: str
	monthly_recovery: Decimal
	approved_by: str
	notes: str | None = None
	created_by: str = "system"


class PRAdvanceResponse(PRBase):
	employee_id: str
	amount: Decimal
	disbursement_date: date
	recovery_start_period_id: str
	monthly_recovery: Decimal
	amount_recovered: Decimal = Decimal("0")
	balance: Decimal
	approved_by: str
	is_fully_recovered: bool = False


# ---------------------------------------------------------------------------
# 15. Terminal Benefits / Final Settlement
# ---------------------------------------------------------------------------

class PRTerminalBenefitLine(BaseModel):
	model_config = _CFG

	benefit_type: TerminalBenefitType
	description: str
	taxable: bool
	amount: Decimal


class PRFinalSettlementCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	employee_id: str
	termination_date: date
	last_day_worked: date
	reason_for_leaving: str
	notice_period_days: int = 0
	notice_period_served_days: int = 0
	created_by: str = "system"


class PRFinalSettlementResponse(PRBase):
	employee_id: str
	termination_date: date
	last_day_worked: date
	reason_for_leaving: str
	notice_period_days: int
	notice_period_served_days: int
	prorated_salary: Decimal = Decimal("0")
	leave_encashment: Decimal = Decimal("0")
	notice_pay: Decimal = Decimal("0")
	severance_pay: Decimal = Decimal("0")
	gratuity: Decimal = Decimal("0")
	other_benefits: Decimal = Decimal("0")
	total_gross: Decimal = Decimal("0")
	paye_on_settlement: Decimal = Decimal("0")
	net_settlement: Decimal = Decimal("0")
	benefit_lines: list[PRTerminalBenefitLine] = Field(default_factory=list)
	run_id: str | None = None
	status: str = "draft"


# ---------------------------------------------------------------------------
# 16. Payroll Summary (report aggregate)
# ---------------------------------------------------------------------------

class PRPayrollSummary(BaseModel):
	model_config = _CFG

	run_id: str
	period_code: str
	pay_date: date
	employee_count: int
	total_basic: Decimal
	total_allowances: Decimal
	total_overtime: Decimal
	total_bonus: Decimal
	total_gross: Decimal
	total_paye: Decimal
	total_nssf_employee: Decimal
	total_nhif_employee: Decimal
	total_other_deductions: Decimal
	total_deductions: Decimal
	total_net: Decimal
	total_employer_nssf: Decimal
	total_employer_nhif: Decimal
	total_employer_cost: Decimal
	currency: str
	generated_at: datetime


# ---------------------------------------------------------------------------
# 17. Payroll Configuration (employer-level)
# ---------------------------------------------------------------------------

class PRPayrollConfigCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	country: Country
	currency: str
	default_pay_frequency: PayFrequency = PayFrequency.MONTHLY
	employer_name: str
	employer_tax_pin: str
	employer_nssf_code: str | None = None
	employer_nhif_code: str | None = None
	pension_scheme_type: str = "defined_contribution"  # defined_contribution | defined_benefit
	pension_employee_rate: Decimal = Decimal("0.06")
	pension_employer_rate: Decimal = Decimal("0.06")
	transport_allowance_taxfree_limit: Decimal = Decimal("2000")  # KES per month
	overtime_multiplier_standard: Decimal = Decimal("1.5")
	overtime_multiplier_holiday: Decimal = Decimal("2.0")
	gl_salary_account: str = "5100"
	gl_paye_liability_account: str = "2210"
	gl_nssf_liability_account: str = "2220"
	gl_nhif_liability_account: str = "2230"
	created_by: str = "system"


class PRPayrollConfigResponse(PRBase):
	country: Country
	currency: str
	default_pay_frequency: PayFrequency
	employer_name: str
	employer_tax_pin: str
	employer_nssf_code: str | None
	employer_nhif_code: str | None
	pension_scheme_type: str
	pension_employee_rate: Decimal
	pension_employer_rate: Decimal
	transport_allowance_taxfree_limit: Decimal
	overtime_multiplier_standard: Decimal
	overtime_multiplier_holiday: Decimal
	gl_salary_account: str
	gl_paye_liability_account: str
	gl_nssf_liability_account: str
	gl_nhif_liability_account: str
