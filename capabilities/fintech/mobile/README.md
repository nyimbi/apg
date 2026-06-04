# Mobile Banking

## Overview
Mobile Banking provides the customer-facing mobile channel layer: banking program governance, customer enrollment, trusted device binding with attestation, authentication factor registration (passcode, biometric, OTP, device binding, hardware key), account and wallet linking, mobile payment initiation, bill payment, airtime purchase, service request intake, notification preference management, and mobile fraud event recording. It is the channel capability that surfaces neobanking, payments, cards, lending, BNPL, and agency services through iOS, Android, web, USSD, and SMS interfaces.

Payment currency must match the linked account's currency. High-value payments require human approval. Devices require attestation before they can be used as a trusted payment device. All mobile banking events stream to `apg.fintech.mobile.lifecycle` via Bytewax.

## Capability ID
`fintech_mobile`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| mobile_banking_program_governance | Register mobile banking programs with country, currency, and platform controls |
| mobile_customer_enrollment | Enroll customers with KYC, consent, AML, and fraud evidence |
| trusted_device_lifecycle | Bind trusted devices with fingerprint, attestation, and risk tier |
| mobile_authentication_factor_workflow | Register auth factors (passcode, biometric, OTP, device binding, hardware key) |
| mobile_account_linking | Link deposit, wallet, card, loan, savings, BNPL, and agency float accounts |
| mobile_payment_workflow | Initiate peer transfers, merchant payments, loan repayments, and wallet cash-outs |
| mobile_bill_payment_workflow | Record bill payments with biller and payment references |
| mobile_airtime_workflow | Record airtime purchases with operator and phone references |
| mobile_service_request_workflow | Open service requests with reason, evidence, and reviewer assignment |
| mobile_notification_workflow | Manage notification preferences with channel and consent controls |
| mobile_fraud_event_workflow | Record fraud events with severity and approval gates for high-severity cases |
| mobile_banking_agent_workflow | Register AI agents for device risk, payment review, and compliance |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Customer notifications |
| nlpc | NLP for service requests |
| keym | Key management |
| fintech_payments | Payment execution |
| fintech_wallets | Wallet account linking and funding |
| fintech_cards | Card account linking |
| fintech_kyc | Customer identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Mobile fraud signal scoring |
| fintech_neobanking | Neobank account linking |
| fintech_lending | Loan account linking and repayment |
| fintech_bnpl | BNPL account linking |
| fintech_agency | Agency float account linking |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| platforms.supported | list | ios, android, web, ussd, sms | Supported mobile platforms |
| auth_factors.supported_types | list | passcode, biometric, device_binding, otp, hardware_key | Auth factor types |
| payments.supported_types | list | peer_transfer, merchant_payment, bill_payment, airtime, loan_repayment, savings_transfer, card_payment, wallet_cash_out | Payment types |
| payments.high_value_threshold | number | 100000 | Amount requiring approval |
| fraud_events.supported_severities | list | low, medium, high, critical | Fraud event severity levels |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-mobile/dashboard | GET | fintech_mobile:view | Overview |
| programs | /fintech-mobile/programs | GET/POST | fintech_mobile:manage_programs | Programs |
| customers | /fintech-mobile/customers | GET/POST | fintech_mobile:customers | Customers |
| devices | /fintech-mobile/devices | GET/POST | fintech_mobile:devices | Security |
| auth_factors | /fintech-mobile/auth-factors | GET/POST | fintech_mobile:auth | Security |
| account_links | /fintech-mobile/account-links | GET/POST | fintech_mobile:accounts | Accounts |
| payments | /fintech-mobile/payments | GET/POST | fintech_mobile:payments | Payments |
| bills | /fintech-mobile/bills | GET/POST | fintech_mobile:bills | Payments |
| airtime | /fintech-mobile/airtime | GET/POST | fintech_mobile:airtime | Payments |
| service_requests | /fintech-mobile/service-requests | GET/POST | fintech_mobile:service | Servicing |
| notifications | /fintech-mobile/notifications | GET/POST | fintech_mobile:notifications | Engagement |
| fraud_events | /fintech-mobile/fraud-events | GET/POST | fintech_mobile:fraud | Risk |
| agents | /fintech-mobile/agents | GET/POST | fintech_mobile:admin | Automation |
| settings | /fintech-mobile/settings | GET/POST | fintech_mobile:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| device_attestation_required | Device binding without attestation | deny |
| device_fingerprint_required | Device binding without fingerprint | deny |
| auth_strength_required | Auth factor without strength reference | deny |
| payment_currency_matches_link | Payment currency differs from account link | deny |
| payment_risk_reference_required | Payment without risk reference | deny |
| high_value_payment_requires_approval | Payment > 100,000 without approval | require_review |
| notification_consent_required | Notification preference without consent | deny |
| high_severity_fraud_requires_approval | High-severity fraud event without approval | require_review |
| mobile_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_mobile_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| MobileProgram | id, name, owner_id, country, currency, supported_platforms, status |
| MobileCustomer | id, customer_reference, kyc_profile_id, country, consent_reference, aml_reference, fraud_reference, status |
| TrustedDevice | id, customer_id, platform, fingerprint, attestation_reference, risk_tier, status |
| AuthFactor | id, customer_id, device_id, factor_type, strength_reference, status |
| AccountLink | id, customer_id, link_type, account_reference, currency, provider_reference, status |
| MobilePayment | id, customer_id, device_id, account_link_id, payment_type, amount, currency, recipient_reference, risk_reference, status |
| BillPayment | id, biller_reference, payment_id, payment_type |
| AirtimePurchase | id, operator_reference, phone_reference, payment_id |
| ServiceRequest | id, customer_id, reason, evidence_references, reviewer_id, status |
| NotificationPreference | id, customer_id, channel, consent_reference |
| MobileFraudEvent | id, customer_id, severity, evidence_references, human_approval_reference |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| mobile_program_registered | Program registered |
| mobile_customer_enrolled | Customer enrolled |
| trusted_device_bound | Device bound |
| auth_factor_registered | Auth factor registered |
| account_linked | Account linked to mobile profile |
| mobile_payment_initiated | Payment initiated |
| bill_payment_recorded | Bill payment recorded |
| airtime_purchased | Airtime purchased |
| service_request_opened | Service request opened |
| notification_preference_set | Notification preference saved |
| fraud_event_recorded | Fraud event recorded |
| mobile_agent_registered | AI agent registered |

## Edge Cases Handled
- Payment currency must match the linked account currency exactly — cross-currency mobile payments are denied; FX conversion must happen at the account or wallet level before the mobile payment is initiated
- Device fingerprint and attestation are both required for device binding; either one missing is a deny; attestation verifies the device is genuine hardware, fingerprint identifies the specific device instance
- Auth factor strength reference is mandatory — a strength reference ties the factor to a policy document defining its assurance level (e.g., NIST AAL2); factors without this reference cannot be used for high-assurance operations
- Notification preferences require consent even for opt-out — the consent record documents the customer's explicit communication preference choice
- High-severity fraud events (high, critical) require human approval before recording; low/medium fraud events can be recorded without approval

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide enrollment evidence; `fintech_fraud` provides per-payment risk signals; platform-specific biometric SDKs are adapter boundaries behind `auth`
- **Downstream**: All payment, wallet, card, lending, BNPL, and agency capabilities are accessed via mobile through account links; `fintech_neobanking` is the primary account backing
- **Peer**: Deployed alongside `fintech_neobanking` (the underlying account layer) and `fintech_payments` (payment execution)

## Development Notes
- USSD and SMS platforms are first-class supported platforms — this reflects the African market context where feature phones and USSD banking are primary channels
- `SUPPORTED_ACCOUNT_LINK_TYPES` includes `agency_float` — this allows agency float accounts to be linked and managed via mobile, enabling mobile-first agent operations
- Bill payment and airtime records both require a matching payment transaction (`payment_type_matches` flag) — the rule prevents a bill payment record being created for a non-bill payment transaction
- Auth factor types map to the customer authentication methods; `device_binding` is distinct from biometric — it binds the factor cryptographically to the device
