# Digital Forms and eSign Capability Specification

- **Capability Name**: Digital Forms and eSign
- **Capability ID**: `esgn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`esgn` provides executable digital form and electronic signature workflows for
APG applications. It manages governed form templates, schema validation,
regulated-field controls, publication approval, form submissions, multi-party
signature envelopes, identity-verified signing ceremonies, tamper seals,
encrypted evidence packages, certificates of completion, and auditable
retention evidence.

The package is intentionally dependency-light. It proves the ESGN domain model,
rule enforcement, service behavior, API helpers, UI view models, semantic model,
and publish-plan evidence while keeping production identity providers,
encryption systems, audit vaults, DLP scanners, notifications, workflow engines,
and long-term evidence stores behind APG integration boundaries.

## Provided Services

- `digital_forms`: create, validate, govern, and publish form templates.
- `form_submissions`: validate tenant-scoped submissions against published
  schemas and bind them to audit evidence references.
- `signature_envelopes`: bind submissions to recipients, routing order, consent,
  delegation policy, signature intent, and tamper seals.
- `signing_ceremonies`: capture signer identity verification, signature intent,
  ceremony timestamp, and deterministic signature evidence.
- `evidence_packages`: seal completed envelopes into encrypted evidence packages
  with retention policy, audit trail references, and certificates.
- `esgn_operations`: expose compatibility operations for generated APG package
  inspection and publish-plan tooling.

## Required Services

- `tenant_context`: all executable operations require a tenant identifier.
- `auth`: permissions and authenticated actors for form ownership, publication,
  sending, signing, and administration.
- `encr`: encrypted evidence package boundary and key-management integration.
- `audl`: append-only audit trail and evidence references.
- `comp`: compliance framework links and regulated-form review evidence.

## Optional Services

- `idfd`: production identity verification providers for signing ceremonies.
- `dlpd`: regulated-field DLP scanners and policy catalogs.
- `wflo`: approval and signing workflow orchestration.
- `ntfy`: sender, recipient, reminder, and completion notifications.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The contract includes:

- `forms`: template-owner, schema-validation, publication-approval, and
  regulated-field DLP requirements.
- `signatures`: identity verification, signature intent, tamper seal, and
  multi-party routing requirements.
- `evidence`: audit trail, encrypted evidence, certificate, and retention
  policy requirements.
- `governance`: tenant context, compliance framework, recipient consent, and
  delegated-signing policy requirements.
- `ui`: form builder, envelope console, signing room, and evidence vault
  feature toggles.
- `theme`: tenant-overridable visual theme tokens for forms and signing.

## Rules

- `tenant_context_required`: deny operations without tenant context.
- `form_template_requires_owner`: deny template creation without an accountable
  owner.
- `form_publication_requires_approval`: deny publication without explicit
  approval.
- `signing_requires_identity_verification`: deny signing without verified signer
  identity.
- `evidence_requires_encryption`: deny evidence package creation unless the
  package is encrypted.
- `regulated_form_requires_compliance_review`: require review for regulated
  forms when compliance review has not been recorded.

## UI

The package exposes APG Python view models for:

- Dashboard: route, rule, theme, summary, and recent operational state.
- Form library and builder: templates, submissions, schema state, and
  publication status.
- Envelope console: envelopes, recipients, routing, signing state, and review
  state.
- Signing room: ceremonies, signer identity status, and signature intent.
- Evidence vault: encrypted evidence packages, certificates, audit events, and
  retention controls.
- Settings: configuration, rules, permissions, and theme metadata.

## Theme

The package uses the `esgn_forms_signing` APG theme contract. It defines
compact, operational UI tokens and component-level visual contracts for the
form builder, envelope console, signing room, and evidence vault.

## External Runtime Boundary

The in-repository runtime deliberately uses deterministic hashes instead of
networked providers. Production deployments should wire APG adapters for
identity verification, KMS/encryption, audit vault persistence, DLP scans,
notifications, workflow orchestration, and long-term evidence storage without
changing the ESGN capability contract.
