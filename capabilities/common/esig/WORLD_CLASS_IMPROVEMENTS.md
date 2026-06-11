# Electronic Signature — World-Class Improvements

**Capability**: `esig` | **Regulation**: FDA 21 CFR Part 11 / GxP
**Date**: 2026-06-11 | © 2025 Datacraft

---

## 1. HMAC-SHA256 Signature with Tenant-Scoped Secret Key

Replace plain SHA-256 with HMAC-SHA256 keyed on a per-tenant secret stored in a secrets manager (Vault, AWS Secrets Manager). A hash without a key is forgeable by anyone who can read the DB — HMAC makes the signature unresettable without the key, closing a significant regulatory gap under 21 CFR §11.70.

## 2. Merkle Hash Chain (Sequential Signature Integrity)

Link each signature's `signature_hash` to the hash of the previous signature for the same document, creating a Merkle chain. Any retroactive insertion or reordering of signatures becomes cryptographically detectable. Required for continuous manufacturing records and batch genealogy.

## 3. Multi-Party Approval Workflow Engine

Model approval chains (e.g., Analyst → Reviewer → QA Manager → Regulatory Affairs) as typed workflow states with configurable routing rules, deadline escalation, and delegation. Current `list_signatures` is just a flat query; this adds structured sequencing and blocking semantics.

## 4. Biometric / MFA Second Factor Binding

Record and hash a second authentication factor (TOTP code, hardware token, push notification confirmation) at signing time. 21 CFR §11.200(a)(1) requires biometric or two-component (ID + password) authentication for each use. Binding the 2FA token into the signature hash provides proof of two-factor compliance.

## 5. Long-Term Validation (LTV) Support — OCSP / CRL Embedding

For signatures destined for regulatory submission (eCTD, eANDA), embed certificate revocation status (OCSP staple or CRL snapshot) at signing time. Signatures validated years later are verifiable even after CAs expire, per ETSI EN 319 102-1 and FDA guidance on long-term signatures.

## 6. PDF/A-3 Embedded Signature (PAdES) Export

Export signed records as PDF/A-3 with an embedded PAdES-LTV signature block. Regulators and Notified Bodies accept PAdES as a qualified electronic signature under EU eIDAS and US FDA guidance. Removes the need for a separate document management system for submission packages.

## 7. Signature Policy OID Binding

Attach a machine-readable Signature Policy Identifier (OID or URI) to each signature, encoding which SOP version, regulatory requirement, and company policy govern the signature. Enables automated policy conformance checking during audits and eliminates ambiguity about which version of a procedure a signature was made under.

## 8. Async Batch Parallel Signing with Backpressure

Replace the sequential loop in `sign_batch` with `asyncio.gather` with a semaphore-bounded concurrency limit. For batches of 100+ batch records at period-end close, sequential signing is a bottleneck; parallel coroutines with back-pressure avoid DB connection exhaustion.

## 9. Signature Template Library

Predefine meaning templates per document type (batch release, CAPA, deviation, change control, lab result) with required fields and standard wording. Reduces human error in meaning text, ensures regulatory-acceptable phrasing, and enables automated meaning classification for analytics.

## 10. Cross-Tenant Signature Federation

Support federated signatures where a contract manufacturer (CMO) signs a record in their tenant and that signature is cryptographically mirrored to the sponsor's tenant. Satisfies GMP Chapter 7 (Outsourced Activities) audit trail requirements without sharing raw data across tenants.

## 11. Tamper-Evident Audit Log with PostgreSQL Row-Level Checksums

Extend the existing append-only SQL rules with a `row_hash` column (SHA-256 of all business columns). A `verify_log_integrity` procedure chain-verifies the entire table, detecting any out-of-band DB manipulation. Current `apg_esig_no_update` rule does not protect against superuser edits.

## 12. Signature Expiry and Re-Attestation Workflow

Add an `expires_at` field and a `re_attest` method for periodic re-certification of standing authorizations (e.g., annual quality agreement re-signatures). Triggers automated notifications before expiry and logs re-attestation events to the audit trail.

## 13. Structured Meaning Validation with NLP Classification

Pass meaning text through a local Ollama model (e.g., `llama3`) to classify intent category and reject vague meanings (e.g., "approved" alone, without object). Returns a structured `meaning_category` enum (APPROVE, CERTIFY, REVIEW, AUTHOR, WITNESS) and a confidence score, supporting downstream analytics without manual tagging.

## 14. Signature Visualization and Certificate Report (DOCX/PDF)

Generate a human-readable Signature Certificate as a DOCX or PDF artifact containing signer name, timestamp, meaning, document hash, and QR-code-encoded verification URL. Required as enclosure for paper-hybrid workflows in clinical trials and notified body submissions.

## 15. OpenTelemetry Distributed Tracing

Emit OTEL spans for each signing, verification, and revocation operation with attributes `esig.document_id`, `esig.signer_id`, `esig.regulation=21cfr11`. Links to the parent request span so a full GxP audit trail can be reconstructed from OTEL traces alone — useful when the DB audit trail is inaccessible during an inspection.
