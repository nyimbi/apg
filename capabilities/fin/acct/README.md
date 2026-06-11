# Bank Account Management (`fin.acct`)

Regulatory bank account lifecycle engine for the APG platform.

## Features

- Open/close/freeze/unfreeze/dormant/reactivate lifecycle
- IBAN + account number generation
- Credit, debit, internal transfer (atomic)
- Fund locks (guarantees, holds, card authorisations)
- Overdraft facility with credit-approved limits
- Bulk payroll disbursement (`bulk_credit`)
- Savings sweep (`sweep_to_linked`)
- Statement generation (JSON / PDF-ready)
- Joint account signatories
- Full lifecycle audit trail
- GL journal events on every monetary operation
- NATS event emission on every state change
- CircuitBreaker on GL posting

## Quick Start

```python
from capabilities.fin.acct.service import BankAccountService
import asyncio
from decimal import Decimal

svc = BankAccountService()
acct = asyncio.run(svc.open_account("t1", "cust-001", "CURR001", "KES", opening_deposit=Decimal("5000")))
print(acct.iban, acct.available_balance)
```

## API

`POST /api/fin/acct/accounts` — see [docs/api_reference.md](docs/api_reference.md)

## Tests

```bash
python -m pytest capabilities/fin/acct/tests/ -v
```

## Docs

- [User Guide](docs/user_guide.md)
- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)
- [Installation](docs/installation_guide.md)
- [Troubleshooting](docs/troubleshooting_guide.md)
