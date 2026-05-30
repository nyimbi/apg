# Risk and Compliance Management Specification

## Purpose

`grc_rcm` gives APG applications a composable assurance capability for risks,
controls, obligations, assessments, evidence, issues, governance decisions,
exceptions, and AI-agent review. The package must be executable without external
services and explicit about where production adapters attach.

## Capability Identity

- Capability id: `grc_rcm`
- Display name: `Risk and Compliance Management`
- Version: `2.1.0`
- Target: `python`
- Profile: `capability`
- Event stream: `apg.grc.rcm.lifecycle`
- Stream processor: `bytewax`
- Theme: `grc_rcm_control`

## Domain Records

### Risk

Fields:

- `id`
- `tenant_id`
- `title`
- `category`
- `owner_id`
- `likelihood`
- `impact`
- `residual_score`
- `risk_level`
- `reviewed_by`
- `metadata`
- `status`
- `created_at`

Supported categories are operational, financial, technology, regulatory,
third-party, strategic, security, and privacy. Residual score is calculated as
`likelihood * impact`; levels are low, medium, high, and critical.

### Control

Fields:

- `id`
- `tenant_id`
- `name`
- `owner_id`
- `control_type`
- `mapped_risk_ids`
- `test_frequency_days`
- `last_assessment_result`
- `status`
- `created_at`

Supported control types are preventive, detective, corrective, and directive.

### Obligation

Fields:

- `id`
- `tenant_id`
- `framework`
- `requirement`
- `owner_id`
- `jurisdiction`
- `due_date`
- `mapped_control_ids`
- `status`
- `created_at`

### Assessment

Fields:

- `id`
- `tenant_id`
- `control_id`
- `assessor_id`
- `result`
- `evidence_ids`
- `findings`
- `status`
- `created_at`

Supported results are effective, partially effective, and ineffective.

### Evidence

Fields:

- `id`
- `tenant_id`
- `source`
- `linked_record_type`
- `linked_record_id`
- `encrypted`
- `retention_days`
- `status`
- `created_at`

### Issue

Fields:

- `id`
- `tenant_id`
- `title`
- `severity`
- `owner_id`
- `remediation_plan`
- `linked_assessment_id`
- `reviewed_by`
- `remediation_evidence_id`
- `status`
- `created_at`
- `remediated_at`

Supported severities are low, medium, high, and critical.

### Governance Decision

Fields:

- `id`
- `tenant_id`
- `title`
- `approver_id`
- `rationale`
- `related_risk_ids`
- `reviewed_by`
- `status`
- `created_at`

### Exception

Fields:

- `id`
- `tenant_id`
- `exception_type`
- `linked_risk_id`
- `expiration_date`
- `approved_by`
- `status`
- `created_at`

Supported exception types are risk acceptance, policy exception, control waiver,
and deadline extension.

### RCM Agent

Fields:

- `id`
- `tenant_id`
- `name`
- `runtime`
- `role`
- `scope`
- `status`
- `created_at`

Supported runtimes are Codex, Claude Code, OpenCode, and Pi. Supported roles are
risk reviewer, control reviewer, compliance reviewer, evidence reviewer, issue
reviewer, and governance reviewer.

## Lifecycle Workflows

### Risk And Control Assurance

1. Register risk.
2. Calculate residual score and risk level.
3. Require review evidence for high or critical residual risk.
4. Register controls mapped to same-tenant risks.
5. Collect encrypted evidence.
6. Assess controls.
7. Open remediation issues for control gaps.
8. Track issue remediation evidence and closure.

### Compliance Obligation Management

1. Register obligation by framework, jurisdiction, owner, due date, and mapped
   controls.
2. Link evidence and assessments to the control estate supporting the
   obligation.
3. Expose dashboard counts and work queues for overdue or weak-control areas.

### Governance Decision And Exception Management

1. Link governance decisions to risks.
2. Require approver and rationale for every decision.
3. Require review evidence for high-risk decisions.
4. Register exceptions only with supported type, approval, and expiration.

### AI-Agent Composition

1. Register RCM agents with supported runtime and role.
2. Limit agent scope to review, preparation, validation, and recommendation.
3. Require human approval for privileged actions.
4. Emit lifecycle evidence for registered agents and approved actions.

## Rule Engine

The rule engine is deterministic and returns:

- `decision`: allow, deny, or require_review;
- `matched_rules`: ordered matching rule names;
- `effects`: rule effects with reason and required action.

Rules cover tenant context, write policy attachment, risk completeness, supported
categories and ranges, high-risk review, control mapping, obligation mapping,
assessment evidence, evidence encryption and retention, issue review,
remediation evidence, governance review, exception approval, Bytewax event
routing, agent runtime and role support, and privileged-agent approval.

## UI Contract

Routes:

- `/grc-rcm/dashboard`
- `/grc-rcm/risks`
- `/grc-rcm/controls`
- `/grc-rcm/obligations`
- `/grc-rcm/assessments`
- `/grc-rcm/evidence`
- `/grc-rcm/issues`
- `/grc-rcm/governance`
- `/grc-rcm/exceptions`
- `/grc-rcm/agents`
- `/grc-rcm/settings`

Screen models are framework-neutral dictionaries so generated applications can
render them through APG's selected Python UI target.

## Events

The package emits audit-style lifecycle events with:

- tenant id;
- event type;
- record id;
- record type;
- status;
- stream name;
- processor name;
- timestamp.

Supported event types are risk registered, control registered, obligation
registered, control assessed, evidence collected, issue opened, issue
remediated, governance decision recorded, exception registered, and RCM agent
registered.

## Production Adapters

The package keeps these concerns behind adapters:

- authorization;
- audit vault;
- notification;
- document management;
- business intelligence;
- policy management;
- workflow orchestration;
- durable Bytewax topology;
- theme application;
- AI-agent runtime orchestration.

## Acceptance Criteria

- Contract shape validates through APG capability tooling.
- Service executes the full RCM lifecycle in memory.
- Guardrails reject unsafe, incomplete, cross-tenant, or unsupported actions.
- UI routes and screen models cover all primary records.
- Semantic model includes provides, requires, rules, theme, screens, and Bytewax
  streaming metadata.
- Package self-test passes.
- APG inspect, publish-plan, and implementation-audit pass for this capability.
