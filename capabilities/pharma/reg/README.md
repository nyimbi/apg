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
