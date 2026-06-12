# Product Registration

## Overview
Manages pharmaceutical product registration across global regulatory regions including dossier compilation, eCTD validation, authority interactions, approval tracking, variation management, renewal lifecycle, certificate storage, and multi-regional procedure coordination. Enforces QP sign-off, eCTD validation, and 180-day renewal alert requirements.

## Capability ID
`pharma_reg`

## Provides
- registration_application_workflow: New application and renewal filing with dossier linkage
- dossier_compilation_workflow: Module-structured CTD/eCTD dossier assembly and validation
- authority_interaction_workflow: Scientific advice, pre-submission meetings, and clarification tracking
- approval_tracking_workflow: Status tracking from submitted to approved with conditions management
- lifecycle_maintenance_workflow: Variation, renewal, transfer, and withdrawal lifecycle events
- variation_management_workflow: Type IA/IB/II variation filing with impact assessment
- renewal_management_workflow: 180-day renewal alert with automatic escalation
- procedure_management_workflow: National, MRP, DCP, and centralised procedure coordination
- registration_certificate_workflow: Certificate storage with expiry tracking
- global_dossier_alignment_workflow: Multi-region dossier consistency management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Role-based access for regulatory affairs |
| audl | Dossier and submission audit trail |
| mten | Product-level registration data isolation |
| conf | Regional deadline and procedure configuration |
| ntfy | Renewal expiry and approval notifications |
| wflo | Submission and variation approval workflow |
| comp | Regulatory submission compliance enforcement |
| schd | Renewal alert and deadline scheduling |
| mqeb | Event streaming for registration lifecycle |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| lifecycle.renewal_alert_days | Days before expiry for renewal alert | 180 |
| dossiers.ectd_validation_required | eCTD format validation mandatory | true |
| registrations.local_representative_required | Local representative required | true |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-reg/api/v1/registrations | POST | Create registration | pharma_reg:registrations |
| /pharma-reg/api/v1/registrations/<id> | PUT | Submit registration | pharma_reg:registrations |
| /pharma-reg/api/v1/registrations/<id>/approve | POST | Record approval | pharma_reg:approvals |
| /pharma-reg/api/v1/dossiers | POST | Compile dossier | pharma_reg:dossiers |
| /pharma-reg/api/v1/dossiers/<id>/validate-ectd | POST | Validate eCTD | pharma_reg:dossiers |
| /pharma-reg/api/v1/variations | POST | File variation | pharma_reg:variations |
| /pharma-reg/api/v1/registrations/renewal-alerts | GET | Get renewal alerts | pharma_reg:renewals |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| dossier_required_for_submission | Registration submitted without dossier | Deny — compile dossier |
| qp_sign_off_required | Submission without QP sign-off | Deny — obtain QP sign-off |
| ectd_validation_required | eCTD dossier not validated | Deny — validate eCTD |
| approval_before_distribution | Product distributed without registration approval | Deny — obtain registration |
| renewal_alert_180d | Registration expiring within 180 days, renewal not started | Deny — initiate renewal |
| local_representative_required | Submission without local representative | Deny — designate local rep |

## Data Models
- ProductRegistration: registration_number, product_type, registration_type, region, status, qp_signed_off, conditions_of_approval
- RegistrationDossier: dossier_number, format, version, modules_present, ectd_validated, completeness_checked
- AuthorityInteraction: interaction_type, authority, minutes_reference, action_items, follow_up_required
- RegistrationVariation: variation_number, variation_type, impact_assessed, dossier_supplement_reference
- RegistrationCertificate: certificate_number, issued_date, expiry_date, conditions
- RegistrationProcedure: procedure_type, reference_member_state, concerned_member_states, concerns

## Streaming Events
- registration_submitted, registration_approved, registration_refused
- dossier_compiled, dossier_updated
- authority_interaction_recorded, clarification_response_submitted
- variation_filed, renewal_filed
- approval_expiring, approval_renewed
- lifecycle_event_recorded, certificate_stored

## Edge Cases Handled
- eCTD validation is required even when the dossier is subsequently submitted in paper format supplementally
- QP sign-off must be a separate step from dossier compilation; the same person cannot perform both
- Renewal alert fires at 180 days but distribution is only blocked at actual expiry date
- Variations require impact assessment before filing, not before approval
- Centralised procedure registrations require reference member state tracking even for post-approval variations

## Composability Notes
Receives variation triggers from `pharma_rec` label changes and `pharma_qms` change control. Certificate expiry data feeds `pharma_dis` WDA management. Authority interaction records link to `pharma_rec` commitment tracking. Approval status gates `pharma_dis` product dispatch.

## World-Class Enhancements (v2.0)

1. **Async-First Service Layer** — All methods converted to `async def`; eliminates event-loop blocking in FastAPI/LangGraph/Bytewax contexts.
2. **Persistent Storage via Async SQLAlchemy** — In-memory dicts replaced with PostgreSQL-backed async sessions; ACID guarantees and full-text search.
3. **Strongly-Typed Variation Types** — `VariationType` `StrEnum` shared by models and service; single source of truth, fails at construction.
4. **Structured Regulatory Calendar with Deadline Engine** — `DeadlineEngine` tracks authority-specific SLAs (EMA 210-day, FDA PDUFA 12-month, TFDA 90-day) with clock-pause events.
5. **eCTD Structural Validation Beyond Flag** — `validate_ectd()` returns a structured `EctdValidationReport` with per-module ICH M8 findings and STF presence checks.
6. **Multi-Tenant Isolation at Query Level** — Tenant scoping enforced at DB query layer; cross-tenant data never reaches application tier.
7. **Regulatory Intelligence Feed Integration** — `ingest_regulatory_intelligence()` parses EMA/FDA/WHO feeds and matches against registered products by INN/brand name.
8. **PSUR / PBRER Lifecycle Tracking** — `PsurSchedule` model tracks data-lock dates, submission due dates, and assessment outcomes; surfaced in renewal alerts.
9. **Dossier Version Control with Diff Tracking** — `DossierRevision` model captures `parent_version_id`, `changed_modules`, and full lineage from sequence 0.
10. **Automated Gap Analysis Against Region-Specific Requirements** — `gap_analysis(dossier_id, target_region)` compares modules against ACTD/ANVISA/ICH region matrix with remediation actions.
11. **Parallel Multi-Region Submission Orchestration** — `bulk_submit_registrations` uses `asyncio.gather` with `return_exceptions=True`; per-region failure isolation.
12. **Condition-of-Approval Obligation Tracker** — `PostApprovalObligation` model with `due_date` and `status`; overdue obligations surface in dashboard and emit events.
13. **Rollback / Saga Support for Multi-Step Workflows** — `SagaCoordinator` records pre-state and compensates on exception across submit→query→response→approval flow.
14. **Role-Based Permission Granularity in `_enforce`** — Typed `EnforceContext` Pydantic models per operation; omitted security checks become type errors.
15. **Structured Audit Log with Retention Policy and Replay** — Immutable `AuditEvent` model persisted to append-only table; supports 7-year replay for EU GMP Annex 11 inspections.

## New Methods

### `bulk_submit_registrations` — parallel multi-region submission

```python
results = await svc.bulk_submit_registrations(
    tenant_id="acme",
    registration_ids=["reg-ke", "reg-ug", "reg-tz", "reg-rw"],
    submitted_by="jane.doe",
)
# Returns per-region success/failure dict; one region's failure does not block others
for reg_id, outcome in results["results"].items():
    print(reg_id, outcome["status"])
```

### `regulatory_compliance_report` — cross-registration GxP status

```python
report = await svc.regulatory_compliance_report(
    tenant_id="acme",
    standard="GxP",
    include_variations=True,
    include_renewals=True,
)
# report["compliance_score"], report["findings"], report["recommendations"]
print(report["compliance_score"], report["total_registrations"])
```

### `post_market_surveillance` — aggregate post-approval obligations and PSUR status

```python
surveillance = await svc.post_market_surveillance(
    tenant_id="acme",
    product_ids=["prod-001", "prod-002"],
    period_days=365,
)
# Surfaces overdue PSURs, unfulfilled conditions-of-approval, and variation outcomes
for item in surveillance["overdue_obligations"]:
    print(item["registration_id"], item["obligation_text"], item["due_date"])
```
