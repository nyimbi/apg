# Pharmacy Management

## Overview
Full-featured pharmacy management capability covering drug formulary management, prescription dispensing with pharmacist verification, LASA (look-alike/sound-alike) alert tracking, controlled substance logging with dual-witness enforcement, drug-drug interaction checking, inventory management with expiry tracking, and prior authorization workflows.

## Capability ID
`healthcare_pha`

## Provides
- drug_formulary_management: Add, classify, and maintain drug formulary with status tracking (preferred/non-preferred/non-formulary/prior-auth/step-therapy)
- prescription_dispensing: Full dispense lifecycle with pharmacist verification gate before release
- lasa_alert_management: Mark drugs as LASA pairs with look-alike/sound-alike/both alert types and tall-man lettering support
- controlled_substance_tracking: Schedule II–V action logging with dual-witness enforcement for waste events
- drug_interaction_checking: Record and query drug-drug interaction pairs by severity (contraindicated through informational)
- pharmacy_inventory_management: Lot-level inventory with expiry tracking, low-stock/recalled/expired status
- prior_authorization_workflow: PA request, approval, and denial lifecycle with expiry date tracking
- medication_adherence_tracking: Dispense history per patient for adherence analytics
- pharmacist_verification_workflow: Hard gate requiring pharmacist_verified=True before any dispense

## Requires
- auth: Authorization for controlled substance and PHI access
- audl: Audit trail for all dispense and controlled substance actions
- mten: Multi-tenant isolation
- conf: Tenant-specific formulary and dispensing settings
- ntfy: Low-stock, recall, and interaction alerts
- wflo: Prior authorization and formulary override approval workflows
- comp: Regulatory compliance for DEA Schedule II–V tracking
- moni: Operational monitoring for dispense queue turnaround
- mqeb: Event emission for EMR medication reconciliation and analytics

## Configuration

| Key | Description |
|-----|-------------|
| dispensing.pharmacist_verification_required | Hard-require pharmacist sign-off before dispense |
| interactions.contraindicated_blocks_dispense | Block dispense if contraindicated interaction detected |
| controlled_substances.dual_witness_required_for_waste | Require two signatures for CS waste events |
| inventory.low_stock_threshold_days | Days of supply triggering low_stock status (default: 7) |
| inventory.expiry_warning_days | Days before expiry to flag as low_stock (default: 30) |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/pha/formulary | List formulary drugs | healthcare_pha:formulary |
| POST | /api/healthcare/pha/formulary | Add drug | healthcare_pha:formulary |
| GET | /api/healthcare/pha/formulary/<id> | Drug detail | healthcare_pha:formulary |
| POST | /api/healthcare/pha/formulary/<id>/lasa | Mark as LASA | healthcare_pha:formulary |
| GET | /api/healthcare/pha/dispense | Dispense queue | healthcare_pha:dispense |
| POST | /api/healthcare/pha/dispense | Create dispense order | healthcare_pha:dispense |
| GET | /api/healthcare/pha/dispense/<id> | Order detail | healthcare_pha:dispense |
| POST | /api/healthcare/pha/dispense/<id>/verify | Pharmacist verify | healthcare_pha:dispense_verify |
| POST | /api/healthcare/pha/dispense/<id>/dispense | Dispense | healthcare_pha:dispense |
| GET | /api/healthcare/pha/interactions | List interactions | healthcare_pha:interactions |
| POST | /api/healthcare/pha/interactions | Record interaction | healthcare_pha:interactions |
| POST | /api/healthcare/pha/interactions/check | Check drug list | healthcare_pha:interactions |
| GET | /api/healthcare/pha/controlled | Controlled substance log | healthcare_pha:controlled |
| POST | /api/healthcare/pha/controlled | Log CS action | healthcare_pha:controlled |
| GET | /api/healthcare/pha/inventory | Inventory list | healthcare_pha:inventory |
| POST | /api/healthcare/pha/inventory | Add inventory | healthcare_pha:inventory |
| PUT | /api/healthcare/pha/inventory/<id>/status | Update status | healthcare_pha:inventory |
| GET | /api/healthcare/pha/prior-auth | Prior auth queue | healthcare_pha:prior_auth |
| POST | /api/healthcare/pha/prior-auth | Request PA | healthcare_pha:prior_auth |
| POST | /api/healthcare/pha/prior-auth/<id>/approve | Approve PA | healthcare_pha:prior_auth |
| POST | /api/healthcare/pha/prior-auth/<id>/deny | Deny PA | healthcare_pha:prior_auth |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| contraindicated_dispense_denied | operation=dispense, interaction_severity=contraindicated | deny |
| pharmacist_verification_required | operation=dispense, pharmacist_verified=False | deny |
| recalled_drug_dispense_denied | operation=dispense, drug_inventory_status=recalled | deny |
| expired_drug_dispense_denied | operation=dispense, drug_inventory_status=expired | deny |
| controlled_substance_dual_witness_required | operation=waste_controlled_substance, dual_witness_present=False | deny |
| prior_auth_required_for_non_formulary | operation=dispense, formulary_status=prior_auth_required, prior_auth_approved=False | deny |
| non_formulary_requires_override | operation=dispense, formulary_status=non_formulary, formulary_override_present=False | deny |
| step_therapy_required | operation=dispense, formulary_status=step_therapy, step_therapy_completed=False | deny |
| low_stock_warning | operation=dispense, inventory_days_remaining=7 | warn |

## Data Models
- DrugCreate/Response: ndc_code, rxnorm_code, drug_type, drug_schedule, dosage_form, formulary_status, is_lasa
- DispenseOrderCreate/Response: drug_id, prescription_id, quantity, pharmacist_verified, dispense lifecycle timestamps
- DrugInteractionCreate/Response: drug_a_id, drug_b_id, severity, mechanism, clinical_effect, management
- ControlledSubstanceLogCreate/Response: drug_schedule, action, quantity, witness_id, waste_amount
- InventoryItemCreate/Response: lot_number, quantity_on_hand, expiry_date, status, days_remaining
- PriorAuthCreate/Response: insurance_id, diagnosis_icd10, clinical_justification, status, expires_at

## Streaming Events
- drug_added_to_formulary, drug_dispensed, dispense_verified
- drug_interaction_detected, lasa_alert_triggered
- controlled_substance_dispensed, controlled_substance_wasted
- inventory_low_stock, inventory_recalled
- prior_auth_approved, prior_auth_denied

## Edge Cases Handled
- Contraindicated interaction is a hard deny regardless of override flags
- Recalled and expired drugs are blocked from dispense by inventory status check
- Waste events for any scheduled drug require a non-null witness_id; null witness raises PolicyViolationError
- Non-formulary dispense requires explicit formulary_override_present=True; step-therapy requires completion flag
- Inventory expiry detection uses days_remaining calculated at creation time

## Composability Notes
Formulary data feeds back to `healthcare_emr` for prescription validation and allergy-drug interaction warnings. Dispense events flow to `healthcare_ana` for medication adherence metrics. Controlled substance logs are consumed by `healthcare_reg` for DEA regulatory reporting.
