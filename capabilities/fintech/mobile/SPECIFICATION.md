# APG Mobile Banking Capability Specification

## Purpose

The Mobile Banking capability makes mobile-first banking a first-class APG
fintech package. It supports mobile enrollment, device binding, authentication
factor registration, account linking, wallet/card rail linking, in-app
payments, mobile transfers, bill payments, airtime purchase, service requests,
push notifications, fraud controls, and provider-neutral mobile-banking AI
agents.

The executable package is dependency-light and can run locally for compiler
output, demos, and package audits. Live app stores, push gateways, SMS gateways,
device-attestation providers, card issuers, core banking ledgers, mobile-money
operators, regulator filing, and durable Bytewax workers stay behind adapter
boundaries.

## Functional Scope

The package provides:

- mobile banking program governance;
- mobile customer enrollment;
- trusted device lifecycle and device-attestation evidence;
- authentication factor registration;
- account, wallet, and card linking;
- mobile payment initiation;
- bill payment and airtime purchase workflows;
- service request intake and evidence;
- notification preference management and push campaign controls;
- fraud event intake and escalation;
- provider-neutral AI-agent composition for mobile operations, device risk,
  payments, service, fraud, and compliance review.

## Composition Contract

Capability id: `fintech_mobile`

Provides:

- `mobile_banking_program_governance`
- `mobile_customer_enrollment`
- `trusted_device_lifecycle`
- `mobile_authentication_factor_workflow`
- `mobile_account_linking`
- `mobile_payment_workflow`
- `mobile_bill_payment_workflow`
- `mobile_airtime_workflow`
- `mobile_service_request_workflow`
- `mobile_notification_workflow`
- `mobile_fraud_event_workflow`
- `mobile_banking_agent_workflow`

Requires:

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_neobanking`
- `fintech_lending`
- `fintech_bnpl`
- `fintech_agency`

## Supported Domains

Supported currencies: USD, EUR, GBP, KES, ZAR, NGN, GHS, UGX, TZS.

Supported countries: KE, UG, TZ, RW, GH, NG, ZA, GB, US, AE.

Supported platforms: iOS, Android, web, USSD, SMS.

Supported authentication factors: passcode, biometric, device binding, OTP,
hardware key.

Supported payment types: peer transfer, merchant payment, bill payment, airtime,
loan repayment, savings transfer, card payment, wallet cash-out.

Supported AI-agent runtimes: Codex, Claude Code, OpenCode, Pi.

## Rule Engine

The deterministic rule engine runs before every state change. It validates
tenant context, write policy, program country/currency/platform support,
customer KYC/AML/fraud/consent evidence, device binding and attestation,
authentication factor strength, account/wallet/card link evidence, payment
amount/currency/type/risk/high-value approval, biller/airtime evidence, service
request evidence, notification consent, fraud-event severity, Bytewax lifecycle
processing, and privileged AI-agent approval.

Rules may return `allow`, `deny`, or `require_review`. Service methods raise
`PermissionError` for denied writes and unresolved required reviews.

## UI And Theming

The package publishes APG Python UI metadata for dashboard, programs, customers,
devices, authentication factors, account links, payments, bills, airtime,
service requests, notifications, fraud events, AI agents, and settings. Theme
metadata uses semantic color, density, icon, and component tokens for consistent
mobile-banking control surfaces with tenant overrides.

## Streaming

Lifecycle metadata uses Bytewax:

- processor: `bytewax`;
- stream: `apg.fintech.mobile.lifecycle`;
- key: `tenant_id`.

No alternate broker configuration is part of the contract.

## Non-Goals

This slice does not implement live app-store release automation, push gateway
delivery, SMS gateway delivery, device-attestation providers, core banking
posting, card-network actions, regulator filing, rendered UI checks, durable
worker deployment, or load testing.
