# Advanced CRM Analytics Specification

## Objective

Build `crm_adv` as an executable APG capability that generated applications can compose for account lifecycle, contact consent, lead scoring and assignment, sales pipeline management, campaign governance, forecast analytics, and AI-assisted CRM review.

## Functional Surface

- Account creation with owner, segment, and territory metadata.
- Contact creation with outreach consent guardrails.
- Lead creation, scoring, and assignment policy enforcement.
- Opportunity creation with account, stage, amount, and close date.
- Activity timeline with next-step governance for open pipeline.
- Campaign launch with audience, consent, budget, and privacy review.
- Forecast recording with evidence and confidence.
- Deterministic CRM rule evaluation.
- UI routes, view models, and theme tokens for generated applications.
- AI agent registration and privileged-action validation for `codex`, `claude_code`, `opencode`, and `pi`.

## Non-Goals

- Owning the enterprise master customer record.
- Owning auth, audit, notification, or event infrastructure.
- Requiring FastAPI, Flask-AppBuilder, SQLAlchemy, or analytics-engine dependencies for package loading.

## Acceptance Criteria

- `get_capability_contract()` validates through `capabilities.capability_contract_registry.validate_contract_shape`.
- `AdvancedCRMService` can create accounts, contacts, leads, assignments, opportunities, activities, campaigns, forecasts, agents, and dashboard summaries without optional infrastructure.
- Rules deny unsafe account, contact, lead, opportunity, campaign, forecast, agent, and Bytewax lifecycle actions.
- `app.semantic_model()` exposes provides/requires, configuration, rules, screens, theme, streaming, and agent team metadata.
- Focused package tests compile and pass.
- `apg capabilities inspect`, `publish-plan`, and `implementation-audit` succeed for the package.
