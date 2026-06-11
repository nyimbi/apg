"""Loan Management System (LMS) — APG fin capability.

Post-origination loan lifecycle engine: disbursement → repayments →
arrears tracking → restructuring → moratorium → write-off → recovery → closure.

fintech_lending handles origination; this module takes over from disbursement.

Composition registration::

	from capabilities.fin.lms import LoanManagementService, CAPABILITY_ID

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

CAPABILITY_ID      = "fin_lms"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN  = "fin"
DISPLAY_NAME       = "Loan Management System"
DESCRIPTION        = (
	"Post-origination loan lifecycle engine: amortisation scheduling, "
	"repayment waterfall, arrears/NPA tracking, CBK/Basel classification, "
	"provisioning, restructuring, moratorium, write-off, recovery, "
	"portfolio quality reporting, and collections workflow."
)

# NATS event subjects
LMS_EVENT_STREAM      = "fin.lms.events"
LMS_LOAN_DISBURSED    = "fin.lms.loan.disbursed"
LMS_REPAYMENT_POSTED  = "fin.lms.repayment.posted"
LMS_LOAN_IN_ARREARS   = "fin.lms.loan.in_arrears"
LMS_LOAN_NPA          = "fin.lms.loan.npa"
LMS_LOAN_RESTRUCTURED = "fin.lms.loan.restructured"
LMS_LOAN_WRITTEN_OFF  = "fin.lms.loan.written_off"
LMS_LOAN_CLOSED       = "fin.lms.loan.closed"

# Public surface
from .models import (  # noqa: E402
	LoanStatus,
	AmortisationMethod,
	RestructureType,
	MoratoriumType,
	PenaltyType,
	PaymentMethod,
	LoanClassification,
	ClosureReason,
	DemandNoticeType,
	CBK_PROVISION_RATES,
	CBK_DPD_THRESHOLDS,
	Loan,
	Installment,
	Repayment,
	ArrearsPosition,
	Moratorium,
	Restructure,
	WriteOff,
	Recovery,
	LoanProvision,
	PortfolioQuality,
	StatementLine,
	GLEntry,
	uuid7str,
)
from .service import LoanManagementService  # noqa: E402

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_VERSION",
	"DISPLAY_NAME",
	"DESCRIPTION",
	"LoanManagementService",
	"LoanStatus",
	"AmortisationMethod",
	"RestructureType",
	"MoratoriumType",
	"PenaltyType",
	"PaymentMethod",
	"LoanClassification",
	"ClosureReason",
	"DemandNoticeType",
	"CBK_PROVISION_RATES",
	"CBK_DPD_THRESHOLDS",
	"Loan",
	"Installment",
	"Repayment",
	"ArrearsPosition",
	"Moratorium",
	"Restructure",
	"WriteOff",
	"Recovery",
	"LoanProvision",
	"PortfolioQuality",
	"StatementLine",
	"GLEntry",
	"uuid7str",
]
