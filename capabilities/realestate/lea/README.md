# Lease Management

## Overview
Full lease lifecycle from heads of terms through abstraction, activation, rent escalation, option tracking, IFRS 16/ASC 842 schedule generation, rent reviews, assignments, and expiry pipeline management. AI-assisted abstraction with mandatory human verification before activation.

## Capability ID
`realestate_lea`

## Provides
- `lease_abstraction_engine`: AI-assisted extraction of key lease terms with human verification
- `rent_escalation_scheduler`: Fixed %, CPI-linked, ratchet, open market, and stepped escalations
- `lease_option_tracker`: Break, renewal, purchase, expansion, and contraction options with notice alerts
- `ifrs16_asc842_compliance`: Present-value ROU asset and lease liability amortisation schedules
- `lease_expiry_pipeline`: Rolling 12-month expiry dashboard sorted by urgency
- `rent_review_workflow`: Upward-only, indexed, and open-market reviews with backdating controls
- `lease_assignment_management`: Assignment and subletting with landlord consent enforcement
- `dilapidation_management`: Pre-lease, interim, and terminal dilapidation schedules
- `lease_renewal_workflow`: Investment committee escalation for major renewals
- `lease_performance_reporting`: WAULT, passing vs. ERV, vacancy rate

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Approval authority for activation and reviews |
| `audl` | Immutable audit for rent changes |
| `mten` | Multi-tenant isolation |
| `conf` | Lease policy configuration |
| `ntfy` | Option expiry and lease expiry alerts |
| `wflo` | Review and assignment approval workflows |
| `nlpc` | AI-assisted lease abstraction |
| `comp` | IFRS 16 / ASC 842 compliance guardrails |
| `mqeb` | Publish lease events |
| `schd` | Schedule escalation trigger dates |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `options.early_warning_days` | 180 | Days before exercise window to alert |
| `abstractions.ai_assisted` | true | Use NLP for abstraction |
| `ifrs16.asc842_categories` | 4 categories | Supported lease categories |
| `rent_reviews.notice_required_days` | 30 | Minimum notice before review |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/lea/leases` | GET/POST | List/create leases | `leases` |
| `/realestate/lea/leases/<id>/activate` | POST | Activate lease | `leases` |
| `/realestate/lea/leases/<id>/surrender` | POST | Surrender lease | `leases` |
| `/realestate/lea/abstraction` | POST | Create abstraction | `abstraction` |
| `/realestate/lea/abstraction/<id>/verify` | POST | Verify abstraction | `abstraction` |
| `/realestate/lea/escalations` | GET/POST | Escalations | `escalations` |
| `/realestate/lea/escalations/<id>/apply` | POST | Apply escalation | `escalations` |
| `/realestate/lea/options` | POST | Create option | `options` |
| `/realestate/lea/options/<id>/exercise` | POST | Exercise option | `options` |
| `/realestate/lea/options/expiring` | GET | Expiring options | `options` |
| `/realestate/lea/ifrs16` | POST | Generate schedule | `ifrs16` |
| `/realestate/lea/ifrs16/<id>/reclassify` | POST | Reclassify (auditor) | `ifrs16` |
| `/realestate/lea/expiry` | GET | Expiry pipeline | `view` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `activation_requires_verified_abstraction` | not verified | deny |
| `escalation_type_supported` | unsupported type | deny |
| `option_exercise_requires_notice` | notice not served | deny |
| `option_exercise_window_required` | outside window | deny |
| `ifrs16_requires_discount_rate` | no rate set | deny |
| `ifrs16_reclassification_requires_auditor` | no auditor approval | deny |
| `assignment_requires_landlord_consent` | no consent ref | deny |
| `forfeiture_requires_legal_process` | process incomplete | deny |
| `renewal_requires_investment_committee` | high value, no IC | deny |

## Data Models
- `LeaseCreate/Response` — full lease header with rent, dates, area, and abstraction status
- `LeaseAbstractionCreate/Response` — AI-extracted fields with exception tracking
- `RentEscalationCreate/Response` — escalation type with old/new rent tracking
- `LeaseOptionCreate/Response` — option window, notice days, exercise status
- `RentReviewCreate/Response` — review type, proposed/agreed rent, backdating auth
- `Ifrs16ScheduleCreate/Response` — ROU asset, liability, 12-month amortisation schedule
- `LeaseAssignmentCreate/Response` — assignment type with landlord consent reference

## Streaming Events
- `lease_created`, `lease_signed`, `lease_activated`, `lease_expired`, `lease_surrendered`
- `rent_escalation_applied`, `rent_review_commenced`, `rent_review_agreed`
- `option_exercised`, `option_lapsed`, `option_expiring_soon`
- `ifrs16_schedule_generated`, `lease_expiry_alert_sent`
- `assignment_completed`, `subletting_approved`

## Edge Cases Handled
- Activation blocked until abstraction is verified (not just complete)
- Escalation double-apply prevented (applied flag checked)
- IFRS 16 discount rate validated 0 < rate < 1 at model level
- Option exercise outside window returns hard denial even with notice
- Reclassification requires auditor approval to prevent silent balance sheet changes
- Expiry pipeline sorts by days_remaining ascending for urgency

## Composability Notes
- Provides IFRS 16 schedules consumed by `realestate_acc`
- Activating a lease triggers unit status update in `realestate_prm`
- Rent escalations feed into `realestate_ren` rent collection expected amounts
- Option tracking integrates with `realestate_ren` renewal pipeline
