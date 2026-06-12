# SACCO Check-off Management (`ckf`)

Manages employer salary deductions and remittances to the SACCO.  Check-off accounts for ~80% of SACCO loan collections in East Africa.

## Workflow

```
generate_schedule → upload_check_off_file → reconcile_check_off → post_check_off_receipts
```

1. **Schedule** — SACCO generates a deduction list per employer per payroll month: loan installments + savings contributions + arrears.
2. **Upload** — Employer submits what was actually deducted.
3. **Reconcile** — Expected vs received.  Short-pay triggers demand notice flag; over-pay allocates excess to savings.
4. **Post** — GL entries written (DR Check-off Receivable / CR Loan & Savings Ledgers).  Idempotent.

## API

`url_prefix = /api/fintech/sacco/ckf`

| Method | Path | Action |
|--------|------|--------|
| GET | `/health` | Service health |
| GET/POST | `/employers` | List / register |
| GET/PATCH | `/employers/<id>` | Get / update |
| POST | `/employers/<id>/deactivate` | Deactivate |
| POST | `/employers/<id>/schedule` | Generate schedule |
| POST | `/employers/<id>/upload` | Upload deduction file |
| POST | `/employers/<id>/reconcile` | Reconcile |
| POST | `/employers/<id>/post` | Post GL receipts |
| GET | `/employers/<id>/status` | Period status |
| POST | `/employers/<id>/remind` | Send reminder |
| GET | `/employers/<id>/statement` | Employer statement |
| POST | `/employers/<id>/default` | Flag default |
| POST | `/links` | Add member-employer link |
| DELETE | `/links` | Remove link |
| GET | `/members/<id>/deductions` | Current deductions |
| GET | `/members/<id>/history` | Check-off history |
| GET | `/outstanding` | All unpaid remittances |
| GET | `/metrics` | Collection & compliance stats |
| POST | `/batch-schedule` | Schedule all employers |

## Header

`X-Tenant-ID: <tenant>` — required on all requests.

## Key Models

- `Employer` — registered employer with payroll contact and remittance account
- `MemberEmployerLink` — member ↔ employer association with employee number and salary
- `CheckOffSchedule` — full deduction list for a payroll period
- `ReconciliationResult` — expected vs received with member-level variance
- `RemittanceRecord` — lifecycle record per employer per period
- `GLEntry` — posted double-entry bookkeeping records

## Capability ID

`fintech_sacco_ckf`

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Bulk Reconciliation Across All Employers in Parallel** [Enhancement]
- **I2. Demand Notice Generation with Audit Trail** [Enhancement]
- **I3. Partial Remittance Acceptance with Pro-Rata Allocation** [Enhancement]
- **I4. Automated Arrears Ageing Report** [Enhancement]
- **I5. Check-off Agreement Expiry Tracking and Renewal Workflow** [Enhancement]
- **I6. Multi-Currency Support with FX Rate Pinning** [Enhancement]
- **I7. Schedule Version Control and Amendment Log** [Enhancement]
- **I8. Member Deduction Cap Enforcement (Salary Protection)** [Enhancement]
- **I9. Automated Reconciliation Discrepancy Alerting** [Enhancement]
- **I10. Employer Payment History Scoring** [Enhancement]
- **I11. GL Account Configurability per Tenant** [Enhancement]
- **I12. Salary Change Propagation with Effective Date** [Enhancement]
- **I13. Bulk Member Upload via CSV/Dict Batch** [Enhancement]
- **I14. Period Rollover with Carry-Forward Logic** [Enhancement]
- **I15. Audit Trail with Immutable Event Log** [Enhancement]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
