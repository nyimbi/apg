# World-Class Improvements — leg_dsc Document & eDiscovery

Fifteen improvements that make this capability 10x better than commercial eDiscovery platforms.

---

### I1. AI-Powered Privilege Auto-Detection
**Category**: AI/ML
**Justification**: Manual privilege review costs $50–$150/doc. Automated detection using locally-hosted LLMs cuts review cycles by 70–80%, directly eliminating the primary cost driver that forces firms to outsource to Relativity or Everlaw.
**Implementation**: `async def auto_classify_privilege(doc_id)` — sends document text metadata to a local Ollama model (llama3/mistral) with a structured prompt scoring attorney-client, work-product, and common-interest probability; results stored as `ai_privilege_scores` on the document record.
**Competitive reference**: Everlaw AI Review, Relativity aiR for Review

---

### I2. Custodian Hold Acknowledgement Workflow
**Category**: Compliance
**Justification**: Federal courts impose sanctions when custodians are not formally notified and their acknowledgements documented. Zapproved and Logikcull track this as a first-class workflow; without it, clients face spoliation motions.
**Implementation**: `async def request_hold_acknowledgement(hold_id, custodian_id)` and `async def record_acknowledgement(hold_id, custodian_id, signature_reference)` — stores `CustodianAcknowledgement` records with overdue-reminder tracking and escalation status.
**Competitive reference**: Zapproved ZDiscovery, Logikcull Legal Hold

---

### I3. Near-Duplicate & Email-Thread Detection
**Category**: AI/ML
**Justification**: Duplicate documents inflate review costs linearly; near-duplicate grouping is table-stakes in Relativity, Nuix, and IPRO. Without it, clients pay to review the same content dozens of times.
**Implementation**: Compute SimHash fingerprints at ingest; store `content_hash` and `near_dup_cluster_id` on documents; expose `async def find_near_duplicates(tenant_id, document_id, threshold)` returning cluster members sorted by similarity score.
**Competitive reference**: Relativity Near-Duplicate Analysis, Nuix Workstation

---

### I4. Forensic Integrity Verification (Chain of Custody)
**Category**: Security
**Justification**: Any document produced without a verified hash chain is attackable in court. Forensic integrity (SHA-256 at ingest, tamper-evident audit trail) is required by FRCP Rule 34 and ISO 27037.
**Implementation**: Store `content_sha256` at ingest; `async def verify_integrity(tenant_id, document_id, current_sha256)` compares against ingested hash and appends a signed integrity-check event to the immutable audit log.
**Competitive reference**: Nuix Investigate, Exterro FTK

---

### I5. Document Review Coding with Near-Dup Propagation
**Category**: Feature
**Justification**: Linear review is dead; modern eDiscovery requires batch-coded documents to propagate responsive/non-responsive calls to near-duplicates automatically, cutting review time by 40%.
**Implementation**: `async def code_document(tenant_id, document_id, coding, reviewer_id, note)` stores `ReviewCoding` records; `async def propagate_coding(tenant_id, document_id)` auto-codes the near-dup cluster with the same call and a lower confidence flag.
**Competitive reference**: Everlaw, Relativity Review

---

### I6. Redaction Engine with Audit Log
**Category**: Compliance
**Justification**: Producing unredacted PII or privileged content is a disqualifying error. A first-class redaction workflow with per-redaction audit entries is required for GDPR/CCPA compliance and privilege protection.
**Implementation**: `async def add_redaction(tenant_id, document_id, page, bbox, reason, redacted_by)` appends `redaction_log` entries; `async def list_redactions(tenant_id, document_id)` returns all redaction records for a document for privilege-log exports.
**Competitive reference**: Logikcull Redaction, Relativity Redact

---

### I7. FRCP Discovery Deadline Calendar
**Category**: Compliance
**Justification**: Missing a discovery deadline triggers sanctions. An auto-computed deadline calendar derived from case schedule data is standard in Clio and CaseFleet but absent from most document repositories.
**Implementation**: `async def create_discovery_deadline(tenant_id, matter_id, deadline_type, due_date, description, assigned_to_id)` and `async def list_overdue_deadlines(tenant_id)` returning items past `datetime.utcnow()` with days-overdue computed.
**Competitive reference**: Clio Manage, CaseFleet

---

### I8. Privilege Challenge & Dispute Tracker
**Category**: Feature
**Justification**: Opposing counsel routinely challenges privilege assertions; without a structured challenge workflow, responses are ad-hoc and poorly documented, exposing waiver risk.
**Implementation**: `async def raise_privilege_challenge(tenant_id, privilege_id, challenger_id, basis)` and `async def respond_to_challenge(tenant_id, challenge_id, response_text, supporting_doc_ids)` — tracks states `pending → responded → ruled`.
**Competitive reference**: Relativity Privilege Log, Kcura proprietary module

---

### I9. Rolling Bates Numbering (Incremental Productions)
**Category**: Feature
**Justification**: Large litigations require multiple rolling productions each picking up where the last Bates number left off. Duplicate or gap Bates numbers trigger court sanctions and spoliation inference.
**Implementation**: Persist `matter_bates_counter` keyed by `matter_id`; `create_production_set` atomically increments the counter so each new production starts at `prior_end + 1`; `async def get_bates_range(tenant_id, matter_id)` returns current high-water mark.
**Competitive reference**: Relativity Production, IPRO Eclipse SE

---

### I10. Document Family & Attachment Grouping
**Category**: Feature
**Justification**: Email attachments must be produced with their parent email (FRCP Rule 34(b)(2)(E)) or the production is deficient. Family grouping is required by every major eDiscovery tool.
**Implementation**: `async def attach_document(tenant_id, child_doc_id, parent_doc_id)` sets `parent_document_id` and shared `family_id`; `async def get_document_family(tenant_id, document_id)` returns all members in parent-first order.
**Competitive reference**: Relativity, Nuix, Everlaw (all enforce family production)

---

### I11. Data Retention & Destruction Policy Engine
**Category**: Compliance
**Justification**: GDPR Art. 17, CCPA, and corporate retention schedules require provable destruction at end-of-retention. Failure to destroy on schedule is a regulatory violation; failure to preserve under hold is spoliation.
**Implementation**: `async def set_retention_policy(tenant_id, document_id, policy_id, destroy_after_date)` and `async def list_destruction_eligible(tenant_id)` — excludes any document currently on litigation hold, preventing accidental destruction.
**Competitive reference**: Exterro Legal GRC, OpenText Information Management

---

### I12. Matter-Level eDiscovery Cost Tracking (Decimal)
**Category**: Feature
**Justification**: eDiscovery routinely runs $1M+ per matter; partners need real-time visibility into processing, hosting, and review costs. No open-source legal platform tracks this natively.
**Implementation**: `async def record_cost(tenant_id, matter_id, cost_type, amount: Decimal, vendor, description)` stores `CostEntry` records; `async def matter_cost_summary(tenant_id, matter_id)` returns totals by cost_type in Decimal with no float leakage.
**Competitive reference**: Logikcull Cost Dashboard, Relativity Billing Module

---

### I13. Semantic PII / Entity Extraction
**Category**: AI/ML
**Justification**: Automated PII detection (names, SSNs, account numbers) is required for GDPR redaction review and dramatically accelerates first-pass privilege review. Everlaw and Reveal both ship NER-based entity extraction as a core feature.
**Implementation**: `async def extract_entities(tenant_id, document_id)` — runs a local NER model (Ollama `llama3` or spaCy) to populate `document["entities"]`; `async def search_by_entity(tenant_id, entity_type, entity_value)` enables entity-scoped document discovery.
**Competitive reference**: Everlaw Entity Extraction, Reveal AI Review

---

### I14. Cross-Matter Document Deduplication & Coding Reuse
**Category**: Feature
**Justification**: Large law firms re-review the same documents across dozens of matters, wasting millions in review hours. Cross-matter deduplication identifies previously-coded documents so prior review decisions can be leveraged.
**Implementation**: Index `content_sha256` across matters; `async def find_cross_matter_copies(tenant_id, document_id)` returns all matters where the same hash appears with prior coding decisions; `async def import_coding_from_matter(tenant_id, source_matter_id, target_matter_id)` imports decisions.
**Competitive reference**: Relativity RelativityOne cross-workspace deduplication, Nuix cross-matter analysis

---

### I15. Time-Limited Secure Share Links for Production Sets
**Category**: Security
**Justification**: Sharing production sets via unencrypted email is a leading cause of data breaches in litigation. Time-limited, HMAC-signed share tokens prevent accidental or malicious unauthorised access.
**Implementation**: `async def create_share_link(tenant_id, production_id, expires_in_hours, created_by)` — generates HMAC-SHA256 tokens over `{production_id}:{expires_at}:{secret}`; `async def resolve_share_link(token)` validates expiry and returns production metadata without requiring authentication.
**Competitive reference**: Logikcull Share Links, Everlaw Secure Portal
