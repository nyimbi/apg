# grc_rcm Capability Package

`grc_rcm` is the APG Risk and Compliance Management capability. It supplies a
dependency-light, executable assurance lifecycle for risks, controls,
obligations, assessments, evidence, issues, governance decisions, exceptions,
and AI-agent review teams.

## Contract Summary

- Capability: `grc_rcm`
- Display name: `Risk and Compliance Management`
- Version: `2.1.0`
- Target: `python`
- UI shell: `apg_python`
- Theme: `grc_rcm_control`
- Stream processor: `bytewax`
- Stream: `apg.grc.rcm.lifecycle`

## Provides

- `risk_register_lifecycle`
- `control_library_lifecycle`
- `compliance_obligation_lifecycle`
- `control_assessment_workflow`
- `evidence_management_workflow`
- `issue_remediation_workflow`
- `governance_decision_workflow`
- `exception_management_workflow`
- `rcm_dashboard_service`
- `rcm_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `document_management`
- `business_intelligence`
- `workflow_orchestration`
- `policy_management`

## Primary Workflows

1. Register tenant risks, compute residual score, and classify risk level.
2. Register controls mapped to risks.
3. Register obligations mapped to controls.
4. Collect encrypted evidence with retention guardrails.
5. Assess controls and require evidence for failed outcomes.
6. Open, review, and remediate issues.
7. Record governance decisions against risks.
8. Register approved exceptions with expiration.
9. Register RCM agents for governed review and validation work.

## Runtime Files

- `capability_contract.py`: executable APG contract and deterministic rules.
- `service.py`: in-memory lifecycle facade.
- `api.py`: generated-app helper functions.
- `views.py`: framework-neutral screen models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused package verification.

## UI Screens

- Dashboard
- Risks
- Controls
- Obligations
- Assessments
- Evidence
- Issues
- Governance
- Exceptions
- Agents
- Settings

## Guardrail Scope

Rules cover tenant context, policy attachment, supported categories and types,
numeric ranges, same-tenant links, review requirements, evidence requirements,
encryption, retention, exception approval, Bytewax event routing, supported
agent runtimes, supported agent roles, and privileged-agent human approval.
