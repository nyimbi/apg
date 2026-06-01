# Capability Spec: fintech_apis

## Identity

- Name: Banking APIs
- Capability id: `fintech_apis`
- Version: `1.1.0`
- Target runtime: Python
- Lifecycle processor: Bytewax

## Contract Summary

`fintech_apis` provides executable APG surfaces for API products, developers,
applications, consent grants, API clients, endpoint policies, webhook
subscriptions, API call audit, rate-limit controls, SLA incidents, and API
governance agents.

## Main Entities

- APIProduct
- DeveloperOrganization
- DeveloperApplication
- ConsentGrant
- APIClient
- EndpointPolicy
- WebhookSubscription
- APICallRecord
- RateLimitBucket
- SLAIncident
- APIEvidence

## Main Commands

- Register API product.
- Onboard developer organization.
- Register developer application.
- Create consent grant.
- Issue API client.
- Publish endpoint policy.
- Subscribe webhook.
- Record API call.
- Update rate-limit bucket.
- Open SLA incident.
- Register banking API agent.
- Validate Bytewax batch.

## UI Screens

- Dashboard
- API Products
- Developers
- Applications
- Consents
- Clients
- Endpoint Policies
- Webhooks
- API Calls
- Rate Limits
- SLA Incidents
- Agents
- Settings

## Release Evidence

This package publishes `semantic_model.json`, `package_manifest.json`, and
`release_report.json` for APG compiler/runtime tooling.
