# Distribution & Agency Management (ins_dst)

Agent registry, commission management, performance tracking, compliance, bancassurance.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/dst/health | Health check |
| GET | /api/insurance/dst/describe | Capability description |
| GET | /api/insurance/dst/agents | List agents |
| POST | /api/insurance/dst/agents | Register agent |
| GET | /api/insurance/dst/agents/{id} | Get agent |
| PUT | /api/insurance/dst/agents/{id} | Update agent |
| DELETE | /api/insurance/dst/agents/{id} | Deregister agent |
| POST | /api/insurance/dst/agents/{id}/suspend | Suspend agent |
| POST | /api/insurance/dst/commissions | Compute commission |
| GET | /api/insurance/dst/commissions | List commissions |
| GET | /api/insurance/dst/commissions/{id} | Get commission |
| POST | /api/insurance/dst/commissions/{id}/approve | Approve commission |
| POST | /api/insurance/dst/commissions/{id}/pay | Pay commission |
| POST | /api/insurance/dst/compliance | Record compliance |
| GET | /api/insurance/dst/compliance | List compliance records |
| POST | /api/insurance/dst/bancassurance | Register partner |
| GET | /api/insurance/dst/bancassurance | List partners |
| POST | /api/insurance/dst/performance/{agent_id} | Performance report |
| GET | /api/insurance/dst/summary | Agency summary |
| GET | /api/insurance/dst/audit | Audit trail |

## World-Class Enhancements (v2.0)

Fifteen improvements targeting African and emerging-market distribution platforms.

**I1. Multi-Tier Hierarchy with Override Chain** — Materialised-path `hierarchy_path` + `get_hierarchy_subtree` for N-level agency trees and commission roll-ups. [Feature]

**I2. Tiered Commission Schedule Engine** — Slab-rate and retrospective bonus schemes via `commission_schedules` with `apply_commission_schedule`. [Feature]

**I3. Automated Compliance Expiry Alerts** — `scan_compliance_expiry_alerts` returns severity-bucketed alerts (critical ≤7d / warning ≤30d / notice ≤90d) for IRA licence renewals. [Compliance]

**I4. Agent Scorecard & Ranking** — `rank_agents_by_performance` computes composite score (premium 50% + persistency 30% + compliance 20%) and returns peer-group percentile ranks. [AI/ML]

**I5. Clawback / Commission Reversal with Lapse Tracking** — `initiate_clawback` creates a negative commission record, adjusts lifetime totals, and emits an audit event for IRA-mandated 90-day lapse reversals. [Compliance]

**I6. Real-Time Production Dashboard Metrics** — `production_dashboard` returns rolling 30/90/365-day premium aggregates, commission liability, top-10 agents, and product mix in one pass. [Performance]

**I7. E&O / Professional Indemnity Register** — `record_pi_coverage` / `check_pi_coverage_status` with auto-suspension to `pi_lapsed` when PI expires. [Compliance]

**I8. Geospatial Territory Assignment** — `assign_territory` and `check_territory_conflict` enforce non-overlapping county/ward assignments across active agents. [Feature]

**I9. Bulk Commission Import & Reconciliation** — `bulk_import_commissions` validates CSV/Excel rows against agent registry and returns `{imported, failed, warnings}` with per-row detail. [Feature]

**I10. Agent Wallet & Settlement Ledger** — `credit_agent_wallet` / `debit_agent_wallet` / `get_wallet_balance` maintain a double-entry ledger enabling M-Pesa and EFT settlement. [Feature]

**I11. Incentive Campaign Management** — `create_incentive_campaign` and `evaluate_agent_campaign_eligibility` automate time-bound bonus campaigns with threshold and bonus-rate configuration. [Feature]

**I12. Persistency Rate Tracking** — `record_policy_renewal` / `record_policy_lapse` / `compute_persistency_rate` track rolling 12-month renewal cohorts per agent. [AI/ML]

**I13. Digital Agent Onboarding Checklist** — `create_onboarding_checklist` / `update_onboarding_step` / `get_onboarding_status` enforce KYC, IRA upload, agreement, bank details, and training steps. [UX]

**I14. Commission Statement PDF Generation Metadata** — `generate_commission_statement` aggregates paid commissions, applies 5% WHT (KRA rules), and returns a statement dict ready for PDF rendering. [Feature]

**I15. Cross-Capability Integration Hooks** — `get_agent_for_policy_integration`, `get_commission_for_finance_integration`, and `integration_manifest` expose typed hooks for ins_pol and ins_fin without modifying service.py. [Integration]

## New Methods

The three highest-impact v2.0 methods added to `DistributionService`.

### `scan_compliance_expiry_alerts` — proactive IRA licence monitoring

```python
svc = DistributionService(tenant_id="acme")
alerts = await svc.scan_compliance_expiry_alerts(tenant_id="acme")
# Returns list of:
# {
#   "agent_id": "agt-001",
#   "compliance_type": "ira_licence",
#   "expiry_date": "2026-07-01",
#   "days_remaining": 19,
#   "severity": "warning"   # critical | warning | notice
# }
critical = [a for a in alerts if a["severity"] == "critical"]
```

### `rank_agents_by_performance` — composite scorecard with percentile ranks

```python
rankings = await svc.rank_agents_by_performance(
    tenant_id="acme",
    peer_group="nairobi_branch",   # optional scope
)
# Returns list sorted by composite_score desc:
# {
#   "agent_id": "agt-042",
#   "composite_score": 0.847,
#   "premium_attainment": 0.91,
#   "persistency_rate": 0.83,
#   "compliance_score": 0.95,
#   "percentile_rank": 92
# }
bottom_quartile = [r for r in rankings if r["percentile_rank"] < 25]
```

### `initiate_clawback` — IRA-compliant commission reversal on policy lapse

```python
result = await svc.initiate_clawback(
    tenant_id="acme",
    commission_id="com-7f3a",
    lapse_event={
        "policy_id": "pol-99x",
        "lapse_date": "2026-06-10",
        "days_on_risk": 45,
    },
    reversed_by="ops@acme.co.ke",
)
# Creates a negative dst_commission record, adjusts agent lifetime totals,
# emits "commission_clawback_initiated" audit event, returns:
# {
#   "clawback_id": "cbk-0012",
#   "original_commission_id": "com-7f3a",
#   "amount_reversed": -4250.00,
#   "agent_lifetime_total_updated": 187300.00,
#   "audit_event_id": "evt-3381"
# }
```
