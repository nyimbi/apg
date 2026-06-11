"""General Ledger (GL) — APG fin capability.

The GL is the system of record for every monetary event in the platform.
All double-entry journal entries flow through this module; every other
fin capability (AP, AR, Payroll, Treasury) calls back here.

Composition registration::

    from capabilities.fin.gl import GLService, CAPABILITY_ID

© 2026 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

CAPABILITY_ID = "fin_gl"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "fin"
DISPLAY_NAME = "General Ledger"
DESCRIPTION = (
	"Double-entry general ledger — accounts, journal entries, trial balance, "
	"P&L, balance sheet, period management, FX revaluation, sub-ledger, "
	"cost centres, intercompany, and regulatory reporting."
)

# NATS subjects
GL_EVENT_STREAM = "fin.gl.events"
GL_JOURNAL_POSTED = "fin.gl.journal.posted"
GL_JOURNAL_REVERSED = "fin.gl.journal.reversed"
GL_PERIOD_OPENED = "fin.gl.period.opened"
GL_PERIOD_CLOSED = "fin.gl.period.closed"
GL_ACCOUNT_CREATED = "fin.gl.account.created"
GL_FX_REVALUED = "fin.gl.fx.revalued"

# Public surface
from .models import (  # noqa: E402
	GLAccount,
	GLAccountType,
	NormalBalance,
	JournalEntry,
	JournalEntryLine,
	AccountingPeriod,
	PeriodStatus,
	JournalStatus,
	JournalType,
	TrialBalanceRow,
	BalanceSheetItem,
	PnLRow,
	SubLedgerEntry,
	AccountMovements,
	BatchEntryRequest,
	FXRate,
	GLImbalanceError,
	PostingToClosedPeriodError,
	AccountNotFoundError,
	DuplicateAccountError,
	JournalNotFoundError,
	uuid7str,
)
from .service import GLService  # noqa: E402

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_VERSION",
	"GL_EVENT_STREAM",
	"GL_JOURNAL_POSTED",
	"GLService",
	"GLAccount",
	"GLAccountType",
	"NormalBalance",
	"JournalEntry",
	"JournalEntryLine",
	"AccountingPeriod",
	"PeriodStatus",
	"JournalStatus",
	"JournalType",
	"TrialBalanceRow",
	"BalanceSheetItem",
	"PnLRow",
	"SubLedgerEntry",
	"AccountMovements",
	"BatchEntryRequest",
	"FXRate",
	"GLImbalanceError",
	"PostingToClosedPeriodError",
	"AccountNotFoundError",
	"DuplicateAccountError",
	"JournalNotFoundError",
	"uuid7str",
]
