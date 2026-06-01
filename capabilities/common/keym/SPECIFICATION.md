# APG KEYM Capability Specification

## Purpose

Key Management (`keym`) is the APG cryptographic key lifecycle control plane.
It must let generated APG applications create, govern, rotate, export, disable,
and audit tenant-scoped keys without binding package execution directly to live
HSMs, cloud KMS systems, vaults, blockchain ledgers, AI lifecycle services, or
hardware attestation providers.

KEYM is responsible for fail-closed governance around policy-backed key
creation, HSM attestation for root keys, dual-control export approval, overdue
rotation review, compromise response, audit evidence, visual UI composition,
and package evidence.

## First-Class Concepts

- **Managed key**: tenant-scoped key metadata with owner, algorithm, key class,
  policy reference, HSM attestation state, lifecycle state, rotation age, and
  compromise posture.
- **Key operation**: a proposed use, create, export, rotate, disable, or destroy
  operation evaluated against deterministic KEYM rules.
- **Export approval**: dual-control approval or rejection for key export.
- **Rotation evidence**: scheduled and completed rotation state with actor and
  evidence.
- **Rotation exception**: independent review for continued use when a key is
  overdue for rotation.
- **Key agent**: an accountable first-class AI agent that can review key
  policy, lifecycle posture, custody, export, rotation exception,
  compromise-response, or HSM attestation workflows.
- **Key lifecycle stream**: the Bytewax-backed event stream that must carry
  grouped key lifecycle mutations into generated APG applications.
- **Key lifecycle batch evidence**: accepted or denied batch validation record
  for grouped KEYM lifecycle mutations.
- **Audit event**: package evidence for key lifecycle and guardrail events.

Reviewable records expose a consistent evidence shape:

- `policy_decision`
- `matched_rules`
- `review_reasons`
- `review_evidence`

## Functional Requirements

1. Every mutating package operation must require tenant context.
2. Key creation must require a non-empty policy reference.
3. Root keys must require HSM attestation before activation.
4. Export operations must require approved dual-control export state owned by
   KEYM.
5. Overdue keys must require rotation or an approved rotation exception before
   continued cryptographic use.
6. Compromised keys must be blocked from cryptographic operations.
7. Export approval and rotation exception decisions must require an independent
   reviewer and reviewer notes.
8. Rotation completion must require actor and evidence.
9. Key agents must use supported APG runtimes: `codex`, `claude_code`,
   `opencode`, or `pi`.
10. Key agents must declare supported KEYM roles, owner, purpose, operating
    scope, and contribution disclosure.
11. Privileged key-agent roles without human approval must be retained as
    `pending_review` evidence.
12. Key lifecycle batch mutations must use Bytewax; denied non-Bytewax
    validations must persist evidence before raising.
13. API helpers and UI view models must expose the package lifecycle state for
   generated APG applications.
14. `app.py`, `semantic_model.json`, `release_report.json`, and
    `package_manifest.json` must reflect the live capability contract.

## Adapter Boundaries

The dependency-light package runtime must not require live external systems.
Production integrations belong behind adapters that preserve the same contract:

- APG ENCR and SECU;
- HSM, KMS, vault, cloud key-store, and software-HSM providers;
- blockchain audit ledgers;
- AI lifecycle and security-intelligence engines;
- compliance, GRC, SIEM, SOAR, DLP, monitoring, and notification systems.

## Current Lifecycle Packet

This slice maintains the executable key lifecycle governance packet and extends
it with first-class key-agent composition:

- create governed keys;
- evaluate key operations;
- request and decide export approvals;
- request and decide rotation exceptions;
- schedule and complete rotations with evidence;
- mark compromised keys and enforce fail-closed use denial;
- register key agents with runtime, role, owner, purpose, scope, disclosure, and
  privileged human-approval or pending-review evidence;
- validate Bytewax key lifecycle batches before accepting grouped mutation work;
- compose pending operation, export approval, rotation exception, rotation,
  key-agent, and lifecycle batch reviews for generated governance consoles;
- expose inventory, approvals, rotations, compromise, key-agent roster, audit,
  streaming metadata, and analytics view models with contract-derived semantic
  evidence.

## Focused Proof

Use the battery-conscious package proof while iterating:

```bash
./.venv/bin/python -m py_compile capabilities/common/keym/__init__.py capabilities/common/keym/models.py capabilities/common/keym/service.py capabilities/common/keym/api.py capabilities/common/keym/capability_contract.py capabilities/common/keym/app.py capabilities/common/keym/view_models.py capabilities/common/keym/tests/test_capability_contract.py capabilities/common/keym/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/keym/tests/test_capability_contract.py capabilities/common/keym/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/keym --json
./.venv/bin/apg capabilities publish-plan capabilities/common/keym --json
```
