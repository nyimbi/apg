# Capability Spec: fintech_mobile

## Identity

- Name: Mobile Banking
- Capability id: `fintech_mobile`
- Version: `1.1.0`
- Target runtime: Python
- Lifecycle processor: Bytewax

## Contract Summary

`fintech_mobile` provides executable APG surfaces for mobile banking programs,
customers, devices, authentication factors, account links, payments, bills,
airtime, service requests, notifications, fraud events, and mobile AI agents.

## Main Entities

- MobileProgram
- MobileCustomer
- TrustedDevice
- AuthFactor
- AccountLink
- MobilePayment
- BillPayment
- AirtimePurchase
- ServiceRequest
- NotificationPreference
- FraudEvent
- MobileEvidence

## Main Commands

- Register mobile program.
- Enroll mobile customer.
- Bind trusted device.
- Register authentication factor.
- Link account, wallet, or card.
- Initiate mobile payment.
- Record bill payment.
- Purchase airtime.
- Open service request.
- Set notification preference.
- Record fraud event.
- Register mobile-banking AI agent.
- Validate Bytewax batch.

## UI Screens

- Dashboard
- Programs
- Customers
- Devices
- Authentication Factors
- Account Links
- Payments
- Bills
- Airtime
- Service Requests
- Notifications
- Fraud Events
- AI Agents
- Settings

## Release Evidence

This package publishes `semantic_model.json`, `package_manifest.json`, and
`release_report.json` for APG compiler/runtime tooling.
