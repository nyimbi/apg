# Property Valuation

## Overview
Full-cycle property valuation: comparable sales database, DCF model builder with range-validated discount rates, mass appraisal engine (regression, spatial, hedonic, AI AVM), valuation roll with automatic supersession, revaluation cycle management, Red Book sign-off enforcement with independent valuer validation, and structured challenge workflow requiring counter-evidence.

## Capability ID
`realestate_val`

## Provides
- `comparable_sales_analysis`: Verified comparable database with adjustment factors
- `dcf_valuation_engine`: Multi-year DCF with exit yield, rental growth, and capex allowance
- `mass_appraisal_engine`: 5 model types including AI AVM with calibration requirement
- `valuation_roll_management`: Current valuation roll with automatic supersession
- `revaluation_cycle_management`: 9 trigger types including IFRS reporting date
- `valuation_report_generation`: Desktop, restricted, Red Book, and mass appraisal reports
- `yield_analysis`: NIY, equivalent yield, reversionary yield, and 3 other types
- `valuer_panel_management`: RICS, API, and internal valuer grades
- `valuation_challenge_workflow`: Evidence-gated challenge against published valuations
- `valuation_benchmarking`: Portfolio value trends and market comparisons

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Sign-off and challenge review authority |
| `audl` | Immutable valuation audit trail |
| `mten` | Multi-tenant isolation |
| `conf` | Discount rate range and model configuration |
| `ntfy` | Revaluation due and challenge alerts |
| `wflo` | Sign-off and challenge approval workflows |
| `nlpc` | Comparable data text extraction |
| `comp` | RICS Red Book compliance |
| `mqeb` | Publish valuation lifecycle events |
| `schd` | Revaluation cycle scheduling |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `dcf.min_discount_rate` | 0.03 | Minimum discount rate (3%) |
| `dcf.max_discount_rate` | 0.30 | Maximum discount rate (30%) |
| `mass_appraisal.calibration_required` | true | Mandate model calibration |
| `valuers.independence_required_for_red_book` | true | Independent valuer for Red Book |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/val/valuations` | GET/POST | List/instruct valuations | `valuations` |
| `/realestate/val/valuations/<id>/sign-off` | POST | Sign off | `valuations` |
| `/realestate/val/valuations/<id>/publish` | POST | Publish (immutable) | `valuations` |
| `/realestate/val/comparables` | GET/POST | Comparable database | `comparables` |
| `/realestate/val/comparables/<id>/verify` | POST | Verify comparable | `comparables` |
| `/realestate/val/dcf` | POST | Run DCF model | `dcf` |
| `/realestate/val/mass-appraisal` | POST | Run mass appraisal | `mass_appraisal` |
| `/realestate/val/roll` | GET/POST | Valuation roll | `roll` |
| `/realestate/val/yields/<property_id>` | GET | Yield calculation | `yields` |
| `/realestate/val/challenges` | GET/POST | Challenges | `challenges` |
| `/realestate/val/challenges/<id>/resolve` | POST | Resolve challenge | `challenges` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `valuation_requires_qualified_valuer` | no qualified valuer | deny |
| `red_book_requires_independent_valuer` | non-independent | deny |
| `sign_off_requires_approved_valuer_grade` | internal_valuer grade | deny |
| `dcf_discount_rate_in_range` | < 3% or > 30% | deny (Pydantic) |
| `mass_appraisal_requires_calibrated_model` | not calibrated | deny |
| `challenge_requires_counter_evidence` | no evidence docs | deny (Pydantic) |
| `published_valuation_immutable` | status = published | deny |
| `challenge_requires_active_valuation` | non-challengeable status | deny |

## Data Models
- `ValuerCreate/Response` — graded valuer with independence flag and firm details
- `ComparableCreate/Response` — transaction with price, area, adjustments, verification
- `ValuationCreate/Response` — instruction with method, purpose, report type, and valuer
- `DcfModelCreate/Response` — full DCF parameters with NPV, capital value, cash flow schedule
- `ValuationRollEntryCreate/Response` — roll entry with supersession tracking
- `MassAppraisalRunCreate/Response` — model run with results per property
- `ValuationChallengeCreate/Response` — evidence-backed challenge with counter valuation

## Streaming Events
- `valuation_instructed`, `valuation_completed`, `valuation_approved`, `valuation_published`
- `comparable_added`, `comparable_verified`
- `dcf_model_run`, `mass_appraisal_run_completed`
- `revaluation_cycle_triggered`, `valuation_roll_updated`
- `valuation_challenged`, `challenge_resolved`

## Edge Cases Handled
- Published valuations are truly immutable: any write attempt denied at rule layer
- Valuation roll automatically supersedes the previous entry for the same property
- DCF discount rate range validated at Pydantic model layer (0.03–0.30)
- Challenge requires at least one counter-evidence document at Pydantic level
- Red Book requires independent valuer; internal valuers cannot publish Red Book reports
- Mass appraisal runs return a results list even in simulation mode
- Yield calculation handles zero purchase price via explicit ValueError

## Composability Notes
- Valuation figures feed `realestate_prm` property current_valuation field
- IFRS 16 commencement valuations triggered by `realestate_lea` lease activation
- Insurance reinstatement values cross-reference `realestate_ins` asset schedule
- DCF rental income inputs sourced from `realestate_ren` rent roll
