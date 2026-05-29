# APG RCM Capability Specification

## Summary

`grc_rcm` provides executable governance, risk, compliance, control-testing,
evidence, and governance-decision behavior for APG applications. The package is
designed as a dependency-light APG capability facade: generated applications can
compose it immediately, while production integrations for databases, analytics,
collaboration, notification, document vaults, and regulatory feeds remain
behind explicit adapters.

The package currently supports:

- tenant-scoped enterprise risk registration and residual risk scoring;
- control registration mapped to one or more risks;
- compliance obligation tracking by framework, jurisdiction, owner, due date,
  and mapped controls;
- control assessment with design effectiveness, operating effectiveness,
  findings, and evidence references;
- encrypted evidence collection with minimum retention guardrails;
- governance decisions with approver, rationale, approval state, and related
  risk references;
- deterministic rule enforcement through the executable APG capability
  contract;
- dashboard, risk register, compliance workbench, control-testing, and
  governance-board view models;
- compatibility `create_record` and `list_records` helpers for generated APG
  package surfaces.

## Capability Contract

The executable contract lives in `capability_contract.py` and is built from this
package specification through APG's spec-backed contract factory. Public
surfaces include:

- capability id: `grc_rcm`;
- package profile: `capability`;
- target: `python`;
- UI shell: `apg_python`;
- theme: `grc_rcm_operations`;
- routes: dashboard, operations, rules, and settings;
- deterministic rule engine for tenant context, write-policy enforcement, and
  high-risk review evidence.

## Runtime Surfaces

### Models

`models.py` exposes dependency-light dataclasses and enums:

- `RCMRisk`
- `RCMControl`
- `RCMComplianceObligation`
- `RCMControlAssessment`
- `RCMGovernanceDecision`
- `RCMEvidence`
- `RCMAuditEvent`
- `GRCRiskLevel`
- `GRCRiskStatus`
- `GRCComplianceStatus`
- `GRCControlType`
- `GRCGovernanceDecisionType`

### Service

`GrcRcmService` is the APG-facing runtime facade. It owns in-memory state for
the package and exposes:

- `register_risk`
- `register_control`
- `add_compliance_obligation`
- `assess_control`
- `collect_evidence`
- `record_governance_decision`
- `dashboard_summary`
- list helpers for risks, controls, obligations, assessments, decisions,
  evidence, audit events, and combined records;
- `create_record` for generated-package compatibility;
- `describe` and `evaluate` for contract and rule-engine access.

### API Helpers

`api.py` exposes dependency-light function helpers around the service for
generated applications and package probes:

- `capability_status`
- `create_record`
- `register_risk`
- `register_control`
- `add_compliance_obligation`
- `assess_control`
- `collect_evidence`
- `record_governance_decision`
- `list_records`
- `dashboard_summary`

### View Models

`views.py` deliberately avoids framework imports so it is usable by generated
Python apps, CLI probes, and tests. It exposes:

- `capability_routes`
- `dashboard_model`
- `risk_register_model`
- `control_testing_model`
- `compliance_workbench_model`
- `governance_board_model`

## Rules And Guardrails

The service enforces both contract-level rules and domain guardrails:

- write operations require tenant context;
- write operations require attached policy enforcement;
- high-risk write operations require review evidence;
- risk registration requires owner, probability, impact, and valid control
  effectiveness values;
- controls require owners, valid effectiveness values, and same-tenant mapped
  risks;
- obligations require owners, frameworks, and same-tenant mapped controls;
- failed controls require evidence references;
- evidence must be encrypted and retained for at least 365 days;
- governance decisions for high or critical risks require rationale and review
  evidence;
- cross-tenant references are rejected.

## Production Integration Boundary

The package is executable without external services. Production deployments can
replace or extend the facade behind APG adapters for:

- relational persistence and migration management;
- document and evidence vault storage;
- regulatory content feeds;
- AI risk prediction and regulatory NLP;
- workflow orchestration and approval routing;
- collaboration and notification services;
- audit vault replication;
- BI dashboards and external reporting.

Do not wire live credentials, live feeds, or external providers into the
dependency-light package surface. Keep those integrations behind named adapters
and prove them with focused integration tests when they are intentionally added.

## Verification

Focused package verification should include:

```bash
./.venv/bin/python -m py_compile capabilities/grc/rcm/__init__.py capabilities/grc/rcm/models.py capabilities/grc/rcm/service.py capabilities/grc/rcm/api.py capabilities/grc/rcm/views.py capabilities/grc/rcm/test_capability_contract.py
./.venv/bin/pytest -q capabilities/grc/rcm/test_capability_contract.py capabilities/grc/rcm/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/grc/rcm --json
./.venv/bin/apg capabilities publish-plan capabilities/grc/rcm --json
```
