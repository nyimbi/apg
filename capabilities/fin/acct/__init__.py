"""
Bank Account Management capability.

Manages the full regulatory bank account lifecycle: IBAN generation,
overdraft, dormancy, fund locks, GL integration, and compliance.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

CAPABILITY_META = {
	"id": "fin_acct",
	"name": "Bank Account Management",
	"version": "1.0.0",
	"description": (
		"Regulatory bank account lifecycle engine: open/close/freeze/dormancy, "
		"IBAN generation, credits/debits, internal transfers, fund locks, "
		"overdraft management, bulk payroll disbursement, and GL integration."
	),
	"parent": "fin",
	"domain": "Financial Services",
	"database_prefix": "ba_",
	"menu_category": "Banking",
	"menu_icon": "fa-university",
	"event_stream": "apg.fin.acct.lifecycle",
	"dependencies": ["fin.glr", "common.reliability", "common.nats"],
	"tags": ["banking", "accounts", "iban", "overdraft", "dormancy", "gl"],
}

from .service import BankAccountService
from .models import (
	BankAccount, AccountTransaction, AccountBalance,
	FundLock, StatementEntry, AccountSignatory,
	AccountStatus, AccountType, TransactionType,
)

__all__ = [
	"BankAccountService",
	"BankAccount",
	"AccountTransaction",
	"AccountBalance",
	"FundLock",
	"StatementEntry",
	"AccountSignatory",
	"AccountStatus",
	"AccountType",
	"TransactionType",
	"CAPABILITY_META",
]
