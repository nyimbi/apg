# Policy Management — World-Class Improvement Roadmap

**Capability**: `grc_pol` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Hierarchical Policy Inheritance

Policies currently exist as flat records. A parent-child hierarchy would allow an enterprise-wide "Information Security Policy" to cascade mandatory clauses into department-level policies. Child policies inherit status constraints (e.g., can't be published if parent is under revision) and automatically propagate version bumps downstream. Implement via `parent_policy_id` FK and a `cascade_revision` service method that queues revision workflows for all descendants.

---

## 2. AI-Assisted Policy Drafting (Ollama)

Call a locally hosted LLM (Mistral/Llama3) to auto-generate initial policy content sections from a structured prompt derived from: policy type, applicable framework controls, and the organisation's existing published policies. Output is treated as a draft suggestion, not a final artefact — human review remains mandatory. Reduces first-draft time from days to minutes without external data leakage.

---

## 3. Structured Two-Stage Approval Workflow

Replace the single-approver model with a configurable multi-stage approval chain: `[functional_reviewer → legal_reviewer → executive_sign_off]`. Each stage captures its own comments, decision, and timestamp. `policy_approval_chain` records the ordered list; the service enforces sequential progression, rejects out-of-order approvals, and short-circuits on any rejection. Critical for regulated industries (CBK, ISO 27001 A.5).

---

## 4. Immutable Audit Log with Cryptographic Chaining

Current `_audit_event` delegates to an adapter with no integrity guarantee. Add a `PolicyAuditChain` model where each entry stores `sha256(previous_hash + payload_json)`. The chain can be independently verified at any time. This satisfies evidence requirements for ISO 27001, SOC 2 Type II, and litigation discovery. Surface a `verify_audit_integrity` method that re-hashes the chain and reports any broken links.

---

## 5. Regulatory Framework Registry as First-Class Entity

`policy_gap_analysis` uses a hardcoded dict of framework requirements. Extract this into a `RegulatoryFramework` store table with versioned control sets (`framework_id`, `version`, `effective_date`, `control_requirements`). Allow tenants to load custom frameworks (e.g., CBK Prudential Guidelines v2024, GDPR Article mapping). Gap analysis then queries the registry rather than static code, enabling zero-code framework updates.

---

## 6. Policy Obligation Extraction and Tracking

After a policy is published, run an NLP extraction pass (spaCy + Ollama) to identify obligation sentences ("employees must...", "shall not..."). Store each obligation as a `PolicyObligation` record linked to the parent policy and section. Track obligation status against control evidence. This transforms policies from narrative documents into machine-auditable compliance artefacts.

---

## 7. Conflict Detection Between Policies

When a new policy is drafted or revised, run a cross-policy conflict check: compare scope, policy type, and obligation keywords against all `published` policies. Flag overlapping obligations or contradictory statements (e.g., two policies prescribing different password rotation periods). Return a `PolicyConflict` list that the author must resolve or acknowledge before proceeding to review.

---

## 8. Attestation Campaigns with SLA Enforcement

Current `publish_policy` creates individual acknowledgement requests. Add a `PolicyAttestationCampaign` entity that groups a batch of acknowledgement requests with: campaign start/end dates, escalation ladder, automated chase schedule, completion SLA (%), and a manager-notification when direct reports are overdue. Campaign analytics show real-time completion heat maps by department.

---

## 9. Policy Delta Reports for Revisions

When `policy_revision` bumps a version, automatically generate a structured diff between the previous `content_sections` and the new ones. Store the delta as a `PolicyDelta` record (added paragraphs, removed paragraphs, changed words). Surface this in the acknowledgement request so employees understand exactly what changed — reducing "I've already read this policy" attrition rates.

---

## 10. Risk-Linked Policy Effectiveness Scoring

`policy_effectiveness` currently computes a simple score from acknowledgement rate minus exception penalty. Extend this to pull linked risk register entries (via `linked_risk_ids`) and incorporate: residual risk reduction attributed to the policy, control test results, and incident correlations. Output a multi-dimensional `EffectivenessCard` with trended scores across last N periods.

---

## 11. Automated Review Scheduling with Calendar Integration

`policy_review_notify` sends ad-hoc emails. Replace with a structured `ReviewSchedule` entity persisted per policy: next review date, reviewer assignment, reminder intervals (T-60, T-30, T-7). Emit iCalendar `.ics` attachments so reminders land in Outlook/Google Calendar. When a review completes, automatically compute and persist the next schedule entry based on `review_cycle_months`.

---

## 12. Policy Template Versioning and Inheritance

`policy_template` stores flat templates without versioning. Add `template_version`, `superseded_by`, and `changelog` fields. When a template is updated, policies created from the old version are flagged as `template_outdated` and a bulk-update workflow is triggered. Templates can inherit from parent templates (e.g., "ISMS Base Template" → "Access Control Policy Template").

---

## 13. Bulk Import/Export with Format Normalisation

Organisations migrating to APG need to import existing policies from Word/PDF/SharePoint. Implement a `policy_bulk_import` pipeline: accept DOCX or PDF, extract title/scope/sections using docling or pdfminer, validate against `CreatePolicyRequest`, and create draft records. Complement with `policy_export` producing ISO-compliant DOCX output from stored `content_sections` using a Jinja2 template.

---

## 14. Cross-Capability Composability Hooks

Define explicit event hooks for downstream capabilities: `grc_ris` (risk) subscribes to `policy_exception_approved` to update risk residuals; `grc_ctl` (controls) subscribes to `policy_published` to trigger control mapping tasks; `grc_aud` (audit) subscribes to `policy_archived` to close linked audit findings. Implement via APG's internal event bus using `asyncio.Queue` with a `policy_event_bus` singleton.

---

## 15. Tenant-Isolated Caching with TTL Invalidation

High-traffic calls (`policy_library`, `policy_dashboard`, `policy_gap_analysis`) re-query the store on every request. Introduce a `TenantScopedCache` backed by the existing `BoundedCache` with TTL (default 60s), keyed by `(tenant_id, method_name, params_hash)`. Cache is invalidated on any mutation to the affected collection. Reduces store pressure by ~80% for read-heavy GRC portals without stale-data risk.
