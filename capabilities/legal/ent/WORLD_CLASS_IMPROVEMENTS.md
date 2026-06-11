# leg_ent — World-Class Improvement Plan

Fifteen targeted improvements that move Entity & Corporate Secretary from a capable registry tool
to a best-in-class corporate governance platform.

---

### I1. Ownership Graph & UBO Disclosure
**Category**: Compliance
**Justification**: Regulators (FATF, EU AML 6AMLD, Kenya BRS) mandate ultimate beneficial owner
(UBO) disclosure. Competitors who surface this automatically win regulated-sector deals outright.
A graph model over shareholding chains lets APG answer "who controls what?" across subsidiary
trees without manual tracing — 10× faster than spreadsheet-based approaches.
**Implementation**: Model each shareholder as a node with `entity_ref` (links to another entity
in the registry) and `ownership_pct: Decimal`; a recursive async traversal resolves the UBO
chain up to configurable depth, collapsing indirect ownership into direct-equivalent percentages.
**Competitive reference**: Dun & Bradstreet Beneficial Ownership, Acuris Risk Intelligence

---

### I2. Compliance Calendar with SLA Breach Prediction
**Category**: AI/ML
**Justification**: Chambers & Partners-rated corporate secretaries use automated deadline engines;
manual tracking loses clients. A predictive SLA model trained on historical filing latency flags
entities likely to miss their window 30/14/7 days out, enabling proactive escalation.
**Implementation**: `predict_filing_risk` aggregates entity age, jurisdiction, filing history,
and active director count into a simple logistic-regression scoring function; results feed a
`get_upcoming_deadlines` method that returns prioritised alerts with risk scores.
**Competitive reference**: Diligent Entities, Blueprint OneWorld

---

### I3. Registered Capital & Share Ledger in Decimal
**Category**: Feature
**Justification**: The current implementation stores `nominal_value` and `consideration_paid` as
Python `float`, silently accumulating rounding errors that corrupt share-premium calculations —
a reportable audit finding. Decimal arithmetic is non-negotiable for regulated financial records.
**Implementation**: Replace all monetary fields with `Decimal` throughout the shareholder and
share-transfer paths; add `compute_share_capital_summary` that returns authorised, issued, and
paid-up capital per share class using `Decimal` arithmetic throughout.
**Competitive reference**: Carta, Equiniti Share Registration

---

### I4. Board Committee Management
**Category**: Feature
**Justification**: Listed companies and large private entities run audit, remuneration, and risk
committees with separate charters. No competitor in the SME space models this at the filing level;
APG can own the governance layer that board portals (BoardPad, Diligent Boards) charge premium for.
**Implementation**: Add `committees` store; `create_committee` links to an entity and holds member
director IDs, charter text, and quorum rules; `list_committee_members` and `record_committee_meeting`
complete the cycle.
**Competitive reference**: Diligent Boards, BoardPad (Nasdaq)

---

### I5. Document Vault with Version Control
**Category**: Feature
**Justification**: Corporate secretarial work is document-heavy — certificates, MoAs, AoAs,
resolutions. Competitors like Docusign CLM and Luminance store metadata+versions; APG should
link documents to entities so the full corporate record is one API call away.
**Implementation**: `attach_document` stores `{doc_id, entity_id, doc_type, filename, storage_ref,
version, hash_sha256, uploaded_by, uploaded_at}`; `list_entity_documents` supports filtering by
doc_type; version history is maintained by never overwriting — new version creates a new record.
**Competitive reference**: ContractPodAi, Docusign CLM

---

### I6. Multi-Jurisdiction Compliance Rules Engine
**Category**: Compliance
**Justification**: A Kenyan NGO, a UK branch, and a Mauritius holding company each have different
filing cadences, director minimums, and UBO thresholds. Hard-coding jurisdiction logic into client
code is a support nightmare; a rules engine lets compliance teams configure it without engineering.
**Implementation**: `JURISDICTION_RULES` dict maps jurisdiction → `{min_directors, annual_return_months,
ubo_threshold_pct, require_company_secretary}`; `get_jurisdiction_requirements` and
`validate_entity_compliance` check an entity against its jurisdiction's rules and return a gap list.
**Competitive reference**: Vistra, TMF Group Corporate Administration

---

### I7. Power of Attorney & Signatory Register
**Category**: Feature
**Justification**: Banks and counterparties request authorised-signatory certificates constantly.
Corporate secretaries currently produce these manually; an always-current signatory register with
expiry tracking eliminates that bottleneck and integrates with the `leg_con` contract module.
**Implementation**: `grant_power_of_attorney` stores grantor, grantee, scope (bank/legal/all),
effective date, and expiry; `list_active_signatories` returns non-expired records and feeds
downstream signature-verification flows.
**Competitive reference**: DocuSign Authority Manager, Diligent Entities

---

### I8. Charges / Security Interest Register (PPSA-Aware)
**Category**: Compliance
**Justification**: The existing `charges` store is empty scaffolding. Lenders and auditors require
a live charges register aligned with PPSA (Kenya MVRB, UK Companies House CH01). Missing charges
disclosures are a financing dealbreaker.
**Implementation**: `register_charge` stores chargee, instrument date, amount (Decimal), security
description, registration deadline, and registration status; `list_charges` and `discharge_charge`
complete the lifecycle; overdue registration raises a compliance alert via the compliance calendar.
**Competitive reference**: UK Companies House charges API, Dye & Durham Corporate Search

---

### I9. Annual Return Auto-Generation
**Category**: Feature
**Justification**: The single highest-volume corporate secretary task is producing the annual return
package (CR12 in Kenya, CS14 in Zimbabwe, etc.). Automating even the data-assembly step saves
2–4 hours per entity per year; at 500 entities that is 1,000–2,000 billable hours returned to the
client.
**Implementation**: `generate_annual_return_pack` snapshots the current director list, share register,
registered address, and financial year data into a structured `AnnualReturnPack` dict that maps
directly to jurisdiction-specific form fields; the method validates completeness before returning.
**Competitive reference**: Diligent Entities auto-populate, Blueprint OneWorld annual return wizard

---

### I10. Beneficial Ownership Threshold Alerts
**Category**: Compliance / AI/ML
**Justification**: Crossing 10 %, 25 %, or 51 % ownership triggers disclosure obligations in most
jurisdictions. The current share-transfer path makes no attempt to check thresholds, exposing
clients to regulatory fines. Automated threshold checking is table-stakes for any compliance-grade
registry.
**Implementation**: After every share allotment or transfer, `_check_ownership_thresholds` computes
each shareholder's percentage of total issued shares; if a threshold boundary is crossed an alert
record is appended to the entity's compliance alerts list and emitted as an audit event.
**Competitive reference**: Acuris Risk Intelligence, Refinitiv World-Check

---

### I11. Director Conflict-of-Interest Register
**Category**: Compliance
**Justification**: Section 144 of Kenya's Companies Act 2015 (and equivalents globally) mandates
directors disclose interests. Managing this in email chains is audit-fail territory. A structured
register that links to the `leg_con` contract capability closes the loop.
**Implementation**: `declare_director_interest` stores director_id, counterparty, nature, date,
and resolution action; `list_director_interests` supports filtering by director and active/resolved
status; interests reference contract or entity IDs for cross-capability linking.
**Competitive reference**: BoardEffect Conflict Register, Diligent Conflict of Interest

---

### I12. Corporate Hierarchy Visualisation Data
**Category**: UX
**Justification**: Group legal teams need a machine-readable hierarchy to feed org-chart tools.
No open-source corporate registry currently exposes this as a clean API; APG can supply the data
layer that tools like Lucidchart and Miro consume.
**Implementation**: `get_corporate_hierarchy` does a DFS over entities where `parent_entity_id`
links subsidiaries to holding companies, returning a tree-structured dict with ownership percentages
at each edge — ready to render as a D3 hierarchy chart.
**Competitive reference**: Diligent Entities Org Chart, Corporater

---

### I13. Statutory Deadline Notification Hooks
**Category**: Integration
**Justification**: Deadlines missed because no one checked the portal. A notification hook system
lets APG push alerts to Slack, email, or the APG notification capability without coupling to any
specific transport — composability at the integration layer.
**Implementation**: `register_deadline_hook` stores `{hook_id, tenant_id, channel_type, endpoint,
days_before: list[int]}`; `fire_due_notifications` is called daily (via APG scheduler), checks
upcoming due dates, and dispatches webhook payloads to registered endpoints for each breach window.
**Competitive reference**: Diligent Entities Smart Reminder, Blueprint OneWorld notifications

---

### I14. Entity Health Score
**Category**: AI/ML
**Justification**: A single 0–100 health score surfaces the combined governance risk of an entity
— filing compliance, director adequacy, UBO disclosure completeness, active charges — so portfolio
managers and auditors can triage 200 entities in seconds instead of reviewing each manually.
**Implementation**: `compute_entity_health_score` aggregates five weighted sub-scores (filing
compliance 30 %, director adequacy 20 %, share register completeness 20 %, charges disclosure 15 %,
UBO completeness 15 %) into a composite score with a breakdown dict explaining deductions.
**Competitive reference**: Diligent Governance Intelligence Score, Gallagher Bassett compliance KPI

---

### I15. Cross-Capability Composability Hooks
**Category**: Integration
**Justification**: Corporate entities are referenced in contracts (leg_con), employment records
(hr_emp), tax filings (fin_tax), and banking mandates (fin_bank). Without declared composability
hooks, each capability re-implements entity lookup — violating DRY at the platform level and
inflating integration cost by 3–5×.
**Implementation**: `resolve_entity_ref` is a canonical lookup method returning a minimal
`EntityRef` dict `{id, legal_name, registration_number, jurisdiction, status}`; it is the
single authoritative entry point other capabilities call, enabling schema-stable cross-domain
references without tight coupling.
**Competitive reference**: Salesforce Legal Entity Object, Workday Legal Entity framework
