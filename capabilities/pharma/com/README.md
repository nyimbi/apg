# Commercial Operations

## Overview
Manages pharmaceutical field force activities including territory management, sales rep assignments, physician call recording, PDMA-compliant sample dispensing, HCP interaction tracking, aggregate spend management, and commercial planning. Enforces Sunshine Act reporting and PDMA compliance rules at every transactional boundary.

## Capability ID
`pharma_com`

## Provides
- territory_management_workflow: Create and manage sales territories with product alignment
- sales_rep_management_workflow: Assign and track field representatives with certification validation
- call_activity_workflow: Record physician calls with outcome and product discussion capture
- sample_management_workflow: PDMA-compliant sample dispensing with electronic signature and lot tracking
- hcp_interaction_workflow: Log all HCP interactions with value-threshold spend detection
- commercial_plan_workflow: Draft, approve, and track commercial plans with territory quotas
- target_segmentation_workflow: Tier physicians and set call frequency targets
- aggregate_spend_workflow: Track and cap cumulative HCP spend per fiscal year
- pdma_compliance_workflow: Enforce PDMA rules at sample dispensing and interaction recording
- commercial_dashboard_workflow: Consolidated field force KPI dashboard

## Requires
| Capability | Reason |
|------------|--------|
| auth | Identity and access control for all operations |
| audl | Immutable audit trail for compliance |
| mten | Multi-tenant context enforcement |
| conf | Runtime configuration management |
| ntfy | Alert notifications for compliance violations |
| wflo | Plan approval workflow |
| comp | PDMA and Sunshine Act compliance enforcement |
| schd | Call frequency scheduling |
| mqeb | Event streaming via Bytewax |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| compliance.aggregate_spend_cap | Maximum annual HCP spend (USD) | 500.0 |
| spend.receipt_required_above | Receipt threshold amount | 25.0 |
| spend.pre_approval_required_above | Pre-approval threshold | 100.0 |
| targets.review_cycle_months | Target tier review frequency | 6 |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-com/api/v1/territories | GET | List territories | pharma_com:territories |
| /pharma-com/api/v1/territories | POST | Create territory | pharma_com:territories |
| /pharma-com/api/v1/reps | GET | List sales reps | pharma_com:reps |
| /pharma-com/api/v1/calls | POST | Record call | pharma_com:calls |
| /pharma-com/api/v1/samples | POST | Dispense sample | pharma_com:samples |
| /pharma-com/api/v1/interactions | POST | Record HCP interaction | pharma_com:interactions |
| /pharma-com/api/v1/plans | POST | Create commercial plan | pharma_com:plans |
| /pharma-com/api/v1/spend/summary | GET | Aggregate spend summary | pharma_com:spend |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| sample_pdma_compliance_required | Sample dispensed without PDMA compliance | Deny — complete PDMA workflow |
| sample_signature_required | HCP signature missing | Deny — capture signature |
| aggregate_spend_cap_enforced | Annual HCP spend exceeds cap | Deny — escalate to compliance |
| spend_pre_approval_required | Spend above 100 USD without pre-approval | Deny — obtain pre-approval |
| rep_certification_required | Rep lacks valid certification | Deny — complete certification |

## Data Models
- Territory: territory_type, name, owner_id, product_ids, approval_reference
- SalesRep: rep_type, territory_id, quota, certification_reference
- CallRecord: physician_id, call_type, products_discussed, outcome
- SampleDispensing: sample_type, lot_number, expiry_date, pdma_compliant, hcp_signature_reference
- HcpInteraction: hcp_id, interaction_type, spend_amount, spend_category
- CommercialPlan: plan_period, territory_ids, product_ids, total_quota
- TargetPhysician: physician_id, tier, call_frequency_per_quarter
- AggregateSpendRecord: hcp_id, category, amount, fiscal_year

## Streaming Events
- territory_created, territory_updated, rep_assigned, call_recorded
- sample_dispensed, sample_reconciled, interaction_recorded
- spend_recorded, plan_approved, compliance_flag_raised
- pdma_violation_detected, aggregate_spend_cap_exceeded

## Edge Cases Handled
- Aggregate spend cap tracked per HCP per fiscal year, not per transaction
- Sample dispensing blocked if PDMA workflow not completed regardless of signature presence
- Territory updates do not require re-approval, only initial creation does
- Cross-tenant isolation enforced at every read/write operation
- Plan approval workflow decoupled from creation to support multi-level approval chains

## Composability Notes
Composes with `pharma_rec` for Sunshine Act reporting obligations. Feeds spend data to `grc` for compliance reporting. Territory data feeds `pharma_com` forecasting into `pharma_sup` demand planning.
