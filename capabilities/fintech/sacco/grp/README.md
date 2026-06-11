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
