# leg_mat — Matter Management

Legal matter lifecycle, task management, team assignment, deadline tracking, court dockets, time entry, budget burn, conflict checking, risk scoring, and privilege log generation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/mat/health | Health check |
| GET | /api/legal/mat/describe | Capability descriptor |
| GET | /api/legal/mat/matters | List matters |
| GET | /api/legal/mat/matters/{id} | Get matter |
| POST | /api/legal/mat/matters | Create matter |
| PUT | /api/legal/mat/matters/{id} | Update matter |
| DELETE | /api/legal/mat/matters/{id} | Archive matter |
| POST | /api/legal/mat/matters/{id}/close | Close matter |
| POST | /api/legal/mat/matters/{id}/transition | FSM status transition |
| POST | /api/legal/mat/matters/{id}/template | Apply task template |
| POST | /api/legal/mat/matters/{id}/conflict-check | Run conflict check |
| GET | /api/legal/mat/matters/{id}/risk | Compute risk score |
| GET | /api/legal/mat/matters/{id}/privilege-log | Generate privilege log |
| GET | /api/legal/mat/matters/{id}/budget-burn | Budget burn report |
| GET | /api/legal/mat/tasks | List tasks |
| GET | /api/legal/mat/tasks/{id} | Get task |
| POST | /api/legal/mat/tasks | Create task |
| PUT | /api/legal/mat/tasks/{id} | Update task |
| DELETE | /api/legal/mat/tasks/{id} | Cancel task |
| GET | /api/legal/mat/deadlines | List deadlines |
| GET | /api/legal/mat/deadlines/{id} | Get deadline |
| POST | /api/legal/mat/deadlines | Create deadline |
| POST | /api/legal/mat/deadlines/chain | Create chained deadlines |
| PUT | /api/legal/mat/deadlines/{id} | Update deadline |
| DELETE | /api/legal/mat/deadlines/{id} | Remove deadline |
| GET | /api/legal/mat/docket | List docket entries |
| POST | /api/legal/mat/docket | Create docket entry |
| GET | /api/legal/mat/time-entries | List time entries |
| POST | /api/legal/mat/time-entries | Log time entry |
| GET | /api/legal/mat/dashboard | Matter dashboard |
| GET | /api/legal/mat/capacity | Team capacity report |
| GET | /api/legal/mat/risk/batch | Batch risk scores |
| GET | /api/legal/mat/audit | Audit events |

## Service Class

`MatterManagementService` — 55+ async methods covering:

- **Matter CRUD** — create, get, list, update, close, reopen, archive
- **FSM Transitions** — `transition_matter_status` with guard enforcement
- **Task Management** — create, get, list, update, complete, cancel
- **Deadline Tracking** — CRUD plus `create_chained_deadlines` for trigger-derived chains
- **Court Dockets** — create, list, update, cancel docket entries
- **Team Assignment** — assign/remove attorneys, list active team members
- **Notes** — privileged and non-privileged notes per matter
- **Time Entries** — `log_time_entry` with `Decimal` precision for hours and rates
- **Budget Tracking** — `set_time_budget`, `get_budget_burn_report` with linear burn projection
- **Conflict Checking** — `run_conflict_check` with substring party matching
- **Matter Templates** — `apply_matter_template` for bulk task seeding by matter type
- **Risk Scoring** — `compute_matter_risk_score` (0–100) and `batch_risk_scores`
- **Privilege Log** — `generate_privilege_log` with SHA-256 tamper hash
- **Capacity Planning** — `get_team_capacity_report` with load_score per attorney
- **Analytics** — `matter_dashboard`, `upcoming_deadlines`, `attorney_workload`, `search_matters`
- **Export** — `export_matter_summary` (full matter + all sub-entities)
- **Audit** — full event log via `get_audit_events`

## Matter FSM

```
open → active, on_hold, archived
active → on_hold, closed, archived
on_hold → active, closed, archived
closed → active
archived → (terminal)
```

Closing a matter with open tasks raises `ValueError`.

## Deadline Chain Rules

| Trigger Event | Derived Deadlines |
|---------------|-------------------|
| `complaint_filed` | Defendant response (d+21), Scheduling conference (d+60), Initial disclosures (d+42) |
| `defence_filed` | Plaintiff reply (d+14), Discovery cutoff (d+120) |

## Matter Templates

| Template | Tasks Generated |
|----------|----------------|
| `litigation` | Initial pleadings, Serve process, Initial disclosures, Scheduling conference |
| `transactional` | Term sheet, Due diligence, Definitive agreement, Regulatory filings |
| `advisory` | Client briefing, Research & analysis, Opinion letter |

## Risk Score Components

| Component | Max Points |
|-----------|-----------|
| Overdue tasks | 30 |
| Overdue deadlines | 30 |
| SoL within 30 days | 20 |
| Budget burn > 80% | 15 |
| Unresolved conflict flag | 25 |
| Inactivity > 30 days | 10 |

Risk levels: **low** (0–29) / **medium** (30–59) / **high** (60–79) / **critical** (80+)
