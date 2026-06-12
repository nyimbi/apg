# SACCO FOSA — Front Office Service Activity

**Capability ID**: `fintech_sacco_fosa`  
**Domain**: fintech / sacco  
**Version**: 1.0.0

FOSA is the deposit-taking, transactional banking wing of a SACCO. Unlike BOSA (long-term savings and loans), FOSA accounts behave like current accounts — members deposit, withdraw, and transact daily via teller, M-PESA, and ATM channels.

## Features

| Feature | Description |
|---|---|
| Account types | CURRENT, SALARY, FIXED_DEPOSIT |
| Channels | Teller, M-PESA C2B/B2C, Bank transfer, ATM |
| ATM cards | VISA, Mastercard, Prepaid — issue / block / unblock |
| Standing orders | Daily, weekly, biweekly, monthly, quarterly, annually |
| Overdrafts | Request / approve workflow with expiry |
| BOSA transfers | Bi-directional FOSA↔BOSA fund movements |
| GL posting | Full double-entry on every transaction |
| Dormancy | Auto-detection at 6 months inactivity, reactivation flow |
| Teller summary | Daily cash position per teller |
| Portfolio | Aggregate statistics for management reporting |
| Audit events | Full event log per tenant |

## API Base URL

```
/api/fintech/sacco/fosa
```

Pass `X-Tenant-ID` header on every request.

## Quick Start

```python
from capabilities.fintech.sacco.fosa.service import FOSAService

svc = FOSAService(tenant_id="sacco-001")

# Open an account
acc = await svc.open_fosa_account("sacco-001", "mem-001", "CURRENT", opening_balance=Decimal("5000"))

# Deposit via M-PESA
txn = await svc.mpesa_cash_in("sacco-001", acc["id"], "MPESA-REF-001", Decimal("2500"), "0712345678")

# Check balance
bal = await svc.get_account_balance("sacco-001", acc["id"])
print(bal["available_balance"])
```

## Tests

```bash
uv run pytest -vxs capabilities/fintech/sacco/fosa/tests/
```

## Key Design Decisions

- **Idempotent M-PESA**: `mpesa_cash_in` deduplicates on `mpesa_reference` — safe for Daraja retries.
- **Daily limits**: withdrawal and transfer limits checked per calendar day, cumulative across channels.
- **GL double-entry**: every financial transaction posts a balanced debit/credit GL pair.
- **Dormancy threshold**: 6 months (configurable via `DORMANCY_MONTHS` constant).
- **BOSA transfer approval**: transfers above KES 50,000 from BOSA require `approved_by`.
- **Multi-tenancy**: all data is scoped to `tenant_id`; cross-tenant access raises `KeyError`.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Interest Accrual Engine** [Core Banking]
- **I2. Cheque Lifecycle Management** [Payments]
- **I3. RTGS / EFT Batch Payment Processing** [Payments]
- **I4. Loan Repayment Auto-Deduction** [Collections]
- **I5. Multi-Currency Accounts & FX Conversion** [International Banking]
- **I6. Transaction Dispute & Chargeback Workflow** [Risk / Compliance]
- **I7. KYC / AML Transaction Monitoring** [Compliance / AML]
- **I8. Bulk Salary Processing (Payroll Credits)** [Corporate Banking]
- **I9. Account Freeze / Unfreeze with Audit** [Governance]
- **I10. Member Notification Dispatch** [Member Experience]
- **I11. End-of-Day Teller Reconciliation & Variance Detection** [Operations]
- **I12. Fixed Deposit Maturity & Rollover Management** [Deposit Products]
- **I13. Peer-to-Peer Member Transfers** [Payments]
- **I14. Regulatory Reporting (CBK Returns)** [Compliance]
- **I15. Configurable Charge Schedules (Service Fee Engine)** [Revenue]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
