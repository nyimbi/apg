# SACCO Group Lending (fintech_sacco_grp)

Joint-liability group lending for Chamas, welfare groups, merry-go-rounds, and investment clubs.
Common in Kenya, Tanzania, and Uganda. The group is collectively responsible for all borrowing.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fintech/sacco/grp/health | Service health |
| GET | /api/fintech/sacco/grp/groups | List groups |
| POST | /api/fintech/sacco/grp/groups | Register group |
| GET | /api/fintech/sacco/grp/groups/{id} | Get group (full profile) |
| PUT | /api/fintech/sacco/grp/groups/{id} | Update group |
| POST | /api/fintech/sacco/grp/groups/{id}/members | Add member |
| DELETE | /api/fintech/sacco/grp/groups/{id}/members/{member_id} | Remove member |
| POST | /api/fintech/sacco/grp/groups/{id}/contributions | Record contributions |
| GET | /api/fintech/sacco/grp/groups/{id}/savings | Group savings summary |
| GET | /api/fintech/sacco/grp/groups/{id}/contributions | Contribution history |
| GET | /api/fintech/sacco/grp/groups/{id}/compliance | Contribution compliance |
| GET | /api/fintech/sacco/grp/groups/{id}/performance | Performance score |
| GET | /api/fintech/sacco/grp/groups/{id}/statement | Group ledger statement |
| GET | /api/fintech/sacco/grp/groups/{id}/mgr/schedule | MGR rotation schedule |
| PUT | /api/fintech/sacco/grp/groups/{id}/mgr/order | Set MGR rotation order |
| POST | /api/fintech/sacco/grp/groups/{id}/mgr/process | Process MGR round |
| GET | /api/fintech/sacco/grp/loans | List group loans |
| POST | /api/fintech/sacco/grp/loans | Apply for group loan |
| GET | /api/fintech/sacco/grp/loans/{id} | Get loan (per-member positions) |
| POST | /api/fintech/sacco/grp/loans/{id}/approve | Approve loan |
| POST | /api/fintech/sacco/grp/loans/{id}/disburse | Disburse to members |
| POST | /api/fintech/sacco/grp/loans/{id}/repayments | Record repayment |
| GET | /api/fintech/sacco/grp/loans/{id}/arrears | Arrears position |
| GET | /api/fintech/sacco/grp/loans/{id}/defaulting-members | Defaulting members |
| POST | /api/fintech/sacco/grp/loans/{id}/joint-liability | Trigger joint liability |

## Group Types

| Type | Description |
|------|-------------|
| `CHAMA` | Informal savings and investment group |
| `WELFARE` | Member welfare / benevolent fund |
| `MERRY_GO_ROUND` | Rotating savings (each member receives the kitty in turn) |
| `INVESTMENT` | Pooled investment club |

## Authentication

Pass `X-Tenant-ID` header on every request.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Interest-Bearing Group Loans with Amortisation Schedule** [Enhancement]
- **I2. Loan-to-Savings Ratio Enforcement** [Enhancement]
- **I3. Penalty Accrual for Late Repayments** [Enhancement]
- **I4. Group Dividend / Interest-on-Savings Distribution** [Enhancement]
- **I5. Emergency / Welfare Loan Sub-Facility** [Enhancement]
- **I6. Meeting Attendance & Quorum Tracking** [Enhancement]
- **I7. SMS/WhatsApp Contribution Reminders (Event Emission)** [Enhancement]
- **I8. Group Credit Score Export for Individual Member Loans** [Enhancement]
- **I9. Merry-Go-Round Cycle Reset & Multi-Cycle Tracking** [Enhancement]
- **I10. Group Loan Restructuring** [Enhancement]
- **I11. Bulk Group Loan Write-Off** [Enhancement]
- **I12. Contribution Projection & Savings Target Tracking** [Enhancement]
- **I13. Multi-Tier Guarantor System** [Enhancement]
- **I14. Automated Group Performance Benchmarking** [Enhancement]
- **I15. Audit Trail Export with Tamper-Evidence Hash Chain** [Enhancement]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
