# APG ENCR Capability Specification

## Purpose

Encryption Services (`encr`) is the APG cryptographic control plane for
generated applications. It must let application builders declare, execute, and
verify encryption decisions without binding generated packages directly to live
HSMs, KMS providers, post-quantum SDKs, zero-knowledge provers, homomorphic
engines, or key-management services.

The capability is responsible for fail-closed cryptographic governance:
tenant-scoped key domains, operation authorization, entropy requirements,
restricted-data quantum-safety checks, plaintext export denial, legacy
algorithm review, threat-adaptive rotation, visual UI composition, and audit
evidence.

## First-Class Concepts

- **Key domain**: a tenant-owned cryptographic boundary with algorithm,
  classification, entropy quality, quantum-safety posture, lifecycle state, and
  rotation state.
- **Crypto operation**: a proposed encrypt, decrypt, export, compute, or
  generate-key action evaluated against deterministic ENCR rules.
- **Crypto exception review**: independent reviewer approval or rejection for
  operations that require review, especially legacy algorithm use.
- **Key rotation**: scheduled and completed rotation evidence for threat
  adaptive encryption.
- **Audit event**: immutable package evidence for key-domain, operation,
  review, and rotation lifecycle transitions.

## Functional Requirements

1. Every executable package operation must require tenant context.
2. Key domains must require non-empty identifiers, owners, algorithms, and data
   classification.
3. Restricted key domains and restricted operations must use quantum-safe
   algorithms.
4. Plaintext export requests must be denied by default.
5. Key-generation operations must be denied when entropy quality is below the
   configured threshold.
6. Legacy algorithm operations must enter a review-required state unless an
   independent crypto exception review approves them.
7. Crypto exception reviews must reject self-review and missing reviewer notes.
8. Active threat signals must deny sensitive operations until affected key
   domains have completed rotation.
9. Rotation completion must require actor and evidence.
10. API helpers and UI view models must expose the lifecycle state for generated
    APG applications.
11. `app.py`, `semantic_model.json`, `release_report.json`, and
    `package_manifest.json` must reflect the live capability contract.

## Adapter Boundaries

The dependency-light package runtime must not require live external systems.
Production integrations belong behind adapters that preserve the same contract:

- APG KEYM, HSM, KMS, cloud KMS, and vault providers;
- post-quantum SDKs and hardware entropy sources;
- zero-knowledge proof systems and homomorphic engines;
- SIEM, SOAR, DLP, GRC, and audit exporters;
- AI cryptographic policy optimizers and threat-intelligence streams.

## Current Lifecycle Packet

This slice adds the executable cryptographic governance packet:

- register key domains;
- evaluate crypto operations against ENCR deterministic rules;
- request and decide crypto exception reviews;
- schedule and complete key rotations with evidence;
- expose operation queues, exception queues, rotation consoles, audit timelines,
  and contract-derived semantic evidence.

## Focused Proof

Use the battery-conscious package proof while iterating:

```bash
./.venv/bin/python -m py_compile capabilities/common/encr/__init__.py capabilities/common/encr/models.py capabilities/common/encr/service.py capabilities/common/encr/api.py capabilities/common/encr/views.py capabilities/common/encr/capability_contract.py capabilities/common/encr/app.py capabilities/common/encr/tests/test_capability_contract.py capabilities/common/encr/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/encr/tests/test_capability_contract.py capabilities/common/encr/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/encr --json
./.venv/bin/apg capabilities publish-plan capabilities/common/encr --json
```
