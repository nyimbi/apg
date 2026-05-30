# Vendor Management Specification

## Purpose

Vendor Management lets APG applications compose supplier master data, qualification, onboarding, performance, risk, compliance, contracts, communications, portal users, scorecards, and vendor-agent review into SCM and ERP applications.

## Functional Scope

### Vendor Profile Lifecycle

- Create tenant-scoped vendor profiles.
- Require code, name, supported vendor type, category, country, and owner.
- Track lifecycle stage from prospect through active/offboarding states.

### Qualification and Onboarding

- Record qualification criteria, reviewer, and score.
- Require review for scores below the configured threshold.
- Record onboarding checklist completion and owner.

### Performance

- Record period-based scores for supported dimensions.
- Require dimensions from the configured list.
- Require score values between 0 and 100.
- Require review for low average scores.

### Risk

- Record vendor risks by type and tier.
- Support low, medium, high, and critical tiers.
- Require owners for high or critical risks.

### Compliance

- Track compliance framework, status, evidence, and review.
- Support pending, compliant, review-required, noncompliant, and expired statuses.
- Require review for noncompliant or expired compliance.

### Contracts, Communications, Portal, and Scorecards

- Create approved vendor contracts with value, currency, and date range.
- Record communication channel, subject, sentiment, and owner for negative sentiment.
- Create approved portal users for vendor collaboration.
- Generate scorecards from performance, risk, and compliance evidence.

### AI Agent Composition

- Treat vendor agents as first-class capability citizens.
- Support Codex, Claude Code, OpenCode, and Pi runtimes.
- Support onboarding, risk, performance, compliance, contract, and supplier-query reviewer roles.
- Limit autonomous scope to inspect, prepare, and recommend.
- Require human approval for privileged actions.

## Contract Requirements

The packet must expose deterministic rules, APG Python UI routes, compact visual theme tokens, configuration schema, provided/required capability metadata, Bytewax event metadata, and publishable semantic model evidence.

## Guardrails

The rule engine must reject missing tenant context, write operations without policy attachment, unaudited state changes, incomplete lifecycle records, unsupported vendor types, unsupported score dimensions, scores outside the accepted range, high risks without owners, noncompliant records without review, unapproved contracts or portal users, unsupported agent runtimes/roles, and non-Bytewax batch routing.

The rule engine must require review for low qualification scores, low performance scores, noncompliant or expired compliance, and privileged vendor-agent actions without human approval.
