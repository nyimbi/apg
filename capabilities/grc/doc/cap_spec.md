# grc_doc Capability Package

`grc_doc` is the APG Document Management capability. It supplies a
dependency-light, executable lifecycle for documents, templates, revisions,
approvals, publication, retention, access grants, processing jobs, and
AI-agent review teams.

## Contract Summary

- Capability: `grc_doc`
- Display name: `Document Management`
- Version: `2.1.0`
- Target: `python`
- UI shell: `apg_python`
- Theme: `grc_doc_control`
- Stream processor: `bytewax`
- Stream: `apg.grc.doc.lifecycle`

## Provides

- `document_repository_lifecycle`
- `document_template_lifecycle`
- `document_revision_workflow`
- `document_approval_workflow`
- `document_publication_workflow`
- `document_retention_workflow`
- `document_access_workflow`
- `document_processing_workflow`
- `document_dashboard_service`
- `doc_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `policy_management`
- `workflow_orchestration`
- `search`

## Primary Workflows

1. Register templates.
2. Create classified documents.
3. Create controlled revisions.
4. Approve and publish documents.
5. Assign retention and legal hold.
6. Grant access with permission and expiry controls.
7. Register and complete Bytewax-backed processing jobs.
8. Register document agents for governed review and validation work.

## Runtime Files

- `capability_contract.py`: executable APG contract and deterministic rules.
- `service.py`: in-memory lifecycle facade.
- `api.py`: generated-app helper functions.
- `views.py`: framework-neutral screen models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused package verification.

## UI Screens

- Dashboard
- Documents
- Templates
- Reviews
- Retention
- Access
- Processing
- Agents
- Settings

## Guardrail Scope

Rules cover tenant context, policy attachment, document completeness,
classification, template completeness, revision review, approval, publication,
retention, legal hold, access grants, Bytewax processing, supported agent
runtimes, supported agent roles, and privileged-agent human approval.
