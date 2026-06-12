# Citizen Services Portal

## Overview
Self-service citizen portal supporting application submission, status tracking, e-payment, document verification, and service delivery analytics. Provides a unified interface for all government-to-citizen service transactions across web, mobile, USSD, and kiosk channels.

## Capability ID
`government_csr`

## Provides
- citizen_self_service_workflow: Multi-channel citizen application submission
- service_application_workflow: Application intake and processing
- application_status_tracking_workflow: Real-time application status updates
- epayment_workflow: M-Pesa, card, bank, and government gateway payments
- document_verification_workflow: Identity, biometric, NIDA, and document verification
- service_notification_workflow: SMS/email/push status notifications
- service_delivery_analytics_workflow: Service performance and SLA analytics
- citizen_review_workflow: Governance review of service quality
- citizen_services_agent_workflow: Automated routing and verification agents
- service_catalogue_workflow: Service definition and fee management

## Requires
| Capability | Reason |
|---|---|
| auth | Citizen authentication (OTP, biometric) |
| audl | Audit trail of all citizen interactions |
| mten | Tenant-scoped service isolation |
| conf | Service catalogue and fee configuration |
| ntfy | Multi-channel citizen notifications |
| wflo | Application processing workflow |
| srch | Service catalogue search |
| moni | Service delivery SLA monitoring |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.payment_before_processing_enforced | Require fee payment before processing begins |
| governance.citizen_data_privacy_enforced | Data protection compliance |
| governance.duplicate_submission_check_enabled | Prevent duplicate applications |
| payments.reconciliation_enabled | Automated payment reconciliation |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-csr/services | GET | Service catalogue | government_csr:services |
| /government-csr/apply | POST | Submit application | government_csr:apply |
| /government-csr/applications | GET | Track applications | government_csr:applications |
| /government-csr/payments | GET/POST | Payment console | government_csr:payments |
| /government-csr/verifications | POST | Verify documents | government_csr:verify |
| /government-csr/analytics | GET | Service delivery analytics | government_csr:analytics |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| unauthenticated_submission_denied | authenticated=False | deny |
| cross_tenant_service_denied | cross_tenant=True | deny |
| payment_receipt_required | receipt_present=False | deny |
| citizen_id_required | citizen_id_present=False | deny |
| verification_evidence_required | evidence_present=False | deny |

## Data Models
- ServiceDefinition: id, tenant_id, service_type, name, fee_amount, sla_days
- ServiceApplication: id, tenant_id, service_id, citizen_id, channel, status, reference_number
- PaymentRecord: id, application_id, payment_method, amount, receipt_number, status
- DocumentVerification: id, application_id, verification_type, status
- CitizenNotification: id, application_id, citizen_id, notification_type, sent
- ServiceDeliveryRecord: id, application_id, certificate_reference
- ServiceReview, CitizenServicesAgent

## Streaming Events
- service_application_submitted, application_status_updated, payment_completed
- payment_failed, document_verified, service_notification_sent, service_completed

## Edge Cases Handled
- Duplicate application submission (same citizen + service + period) detected
- Payment failure triggers automatic retry notification to citizen
- Unauthenticated submissions always denied regardless of channel
- Cross-tenant service access denied even for admin users
- USSD channel has reduced document requirements vs web portal

## Composability Notes
Composes with `government_cas` (complex applications escalate to case management), `government_lic` (licence applications use CSR portal as intake), `government_bud` (collected fees credit government revenue accounts), and `government_tax` (portal payments may include tax obligations).

---

## World-Class Enhancements (v2.0)

- **I1.** Citizen Services Portal — World-Class Improvements
- **I2.** Async-Native Service Layer
- **I3.** Persistent Storage with PostgreSQL + Alembic Migrations
- **I4.** Event-Driven Architecture via Domain Events
- **I5.** Idempotency Keys and Duplicate Submission Prevention
- **I6.** Structured SLA Tracking and Breach Alerting
- **I7.** Role-Based Access Control (RBAC) with Fine-Grained Permissions
- **I8.** AI-Powered Application Pre-Screening
- **I9.** Multi-Factor OTP Authentication with TOTP Support
- **I10.** Offline-First USSD / SMS Interface
- **I11.** Document OCR and Auto-Population
- **I12.** Payment Reconciliation and Revenue Dashboard
- **I13.** Service Catalogue Versioning and Deprecation
- **I14.** Citizen Profile and History Aggregation
- **I15.** Webhook / Push Notification Integration

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
