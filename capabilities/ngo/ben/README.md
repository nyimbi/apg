# Beneficiary Registry (ngo_ben)

Beneficiary profiling, programme enrolment, vulnerability scoring, transfer management, deduplication.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/ben/health` | Health check |
| GET | `/api/ngo/ben/` | List beneficiaries |
| POST | `/api/ngo/ben/` | Register beneficiary |
| GET | `/api/ngo/ben/<id>` | Get beneficiary |
| PUT | `/api/ngo/ben/<id>` | Update beneficiary |
| DELETE | `/api/ngo/ben/<id>` | Deactivate beneficiary |
| GET | `/api/ngo/ben/<id>/enrolments` | List enrolments |
| POST | `/api/ngo/ben/<id>/enrolments` | Enrol in programme |
| GET | `/api/ngo/ben/<id>/assessments` | List vulnerability assessments |
| POST | `/api/ngo/ben/<id>/assessments` | Create vulnerability assessment |
| GET | `/api/ngo/ben/<id>/transfers` | List transfers |
| POST | `/api/ngo/ben/<id>/transfers` | Create transfer |
| GET | `/api/ngo/ben/<id>/dedup` | Duplicate check |
| GET | `/api/ngo/ben/analytics/vulnerability` | Vulnerability distribution |
| POST | `/api/ngo/ben/analytics/dedup-scan` | Full registry dedup scan |
| GET | `/api/ngo/ben/audit-events` | Audit log |

## World-Class Enhancements (v2.0)

**I1. AI-Powered Risk Trajectory Prediction** — Rolling time-series trend scoring flags deteriorating households before crisis, cutting response lag from weeks to days [AI/ML]

**I2. Household Graph Linkage** — `household_id` groups members with aggregated exposure/transfer rollup via `get_household_summary()` [Feature]

**I3. Biometric Hash De-duplication** — SHA-256 biometric template hash indexed at registration; raises `duplicate_biometric` on collision without storing raw biometric data (GDPR Art. 9 compliant) [Security]

**I4. Consent & Data Minimisation Ledger** — Immutable content-hashed consent records per purpose; withdrawal triggers cascading soft-purge; gates all bulk exports [Compliance]

**I5. Configurable Weighted Vulnerability Scoring** — Per-assessment `weights: dict[str, float]` parameter normalised to 1.0; raw score and weight config both persisted for cross-version comparability [AI/ML]

**I6. Grievance & Redress Tracking** — `raise_grievance()` with SLA deadline from severity; `resolve_grievance()` logs elapsed time; `list_open_grievances(days_overdue)` flags breaches [Feature]

**I7. Multi-Currency Transfer Ledger with FX Snapshot** — Captures `source_currency`, `source_amount`, `fx_rate` at creation; `programme_reach_summary` returns both local and source-currency aggregates [Feature]

**I8. Recertification & Enrolment Expiry Management** — Enrolments gain `valid_until`/`recertification_due`; `list_recertification_due(days_ahead)` and `recertify_enrolment()` automate eligibility renewals [Compliance]

**I9. Batch Disbursement with Maker-Checker Approval** — `create_disbursement_batch()` + `approve_batch()` enforcing approver != submitter; `mark_batch_processed()` bulk-confirms transfers atomically [Feature]

**I10. Longitudinal Outcome Tracking** — `record_outcome()` time-series with `outcome_trajectory()` returning baseline deltas; `programme_impact_report()` aggregates mean baseline-to-endline deltas per outcome type [Feature]

**I11. Predictive Attrition Scoring** — `predict_attrition_risk()` returns `{risk_score, risk_tier, key_factors}` from transfer gaps, vulnerability trend, grievance count, and assessment age — no external model required [AI/ML]

**I12. Offline-First Sync Protocol** — `export_sync_bundle(last_sync_at)` produces newline-delimited JSON delta; `apply_sync_bundle()` idempotently applies mutations using `updated_at` as conflict tiebreaker [UX]

**I13. Programme Eligibility Rules Engine** — JSON predicate trees per programme; `evaluate_eligibility()` returns `{eligible, reasons, score}` supporting threshold/range/set-membership/AND/OR operators [Feature]

**I14. Duplicate-Merge Workflow** — `merge_beneficiaries(primary_id, duplicate_id, merged_by)` re-parents all enrolments/assessments/transfers, soft-deletes duplicate as `status="merged"`, emits audit event with field diff [Feature]

**I15. Exit Outcome Classification** — `exit_beneficiary()` accepts structured `exit_outcome` enum (`graduated`, `relocated`, `deceased`, `dropout`, `ineligible`, `transferred_out`); `programme_graduation_report()` returns outcome-code breakdown [Compliance]

## New Methods

Three high-impact async methods added in v2.0:

### `predict_attrition_risk`

Scores dropout risk from operational signals without requiring an external model.

```python
svc = BeneficiaryRegistryService(tenant_id="ke_001")

result = await svc.predict_attrition_risk("ben_abc123")
# {
#   "risk_score": 0.74,
#   "risk_tier": "high",
#   "key_factors": ["no transfer in 45 days", "2 open grievances", "vulnerability trending up"]
# }

# Bulk pre-screen before a payment run
high_risk = [
    await svc.predict_attrition_risk(b["id"])
    for b in await svc.list_beneficiaries(programme_id="prog_xyz")
    if (await svc.predict_attrition_risk(b["id"]))["risk_tier"] == "high"
]
```

### `merge_beneficiaries`

Merges a detected duplicate into a canonical record, preserving full history from both sides.

```python
# After bulk_deduplication_scan identifies a collision
scan = await svc.bulk_deduplication_scan()
# scan["duplicate_groups"][0] -> ["ben_abc123", "ben_dup456"]

merge_result = await svc.merge_beneficiaries(
    primary_id="ben_abc123",
    duplicate_id="ben_dup456",
    merged_by="caseworker_001",
)
# All enrolments, assessments, and transfers from ben_dup456
# are re-parented to ben_abc123; duplicate is soft-deleted
# with status="merged" and a beneficiary_merged audit event is emitted.
```

### `create_vulnerability_assessment` (weighted)

Weighted pillar scoring for programme-specific targeting.

```python
# Food-security programme: weight food_security 3x
assessment = await svc.create_vulnerability_assessment(
    beneficiary_id="ben_abc123",
    food_security=8,
    income_stability=4,
    housing=3,
    health_access=5,
    education_access=6,
    weights={
        "food_security": 0.45,
        "income_stability": 0.25,
        "housing": 0.10,
        "health_access": 0.10,
        "education_access": 0.10,
    },
    assessed_by="field_officer_007",
)
# assessment["composite_score"] reflects programme-weighted priority
# assessment["weights_used"] is persisted for audit comparability
```
