# leg_cpl — World-Class Improvements

Fifteen targeted improvements to make Legal Compliance Management 10x better than competitors.

---

### I1. AI-Powered Regulatory Change Detection
**Category**: AI/ML
**Justification**: Regulations change constantly (GDPR amendments, local data laws). Competitors like Clausematch charge $50k+/year to auto-detect regulatory updates; embedding this in-capability removes that dependency and gives teams a 24-72 hour advantage over manual monitoring.
**Implementation**: Ingest regulatory bulletin feeds (gazette, EUR-Lex, ODPP) via async HTTP polling; diff new text against stored `regulation_text`; flag requirements whose regulation version hash has changed and queue a `regulation_updated` event with a diff summary.
**Competitive reference**: Clausematch, Ascent RegTech

---

### I2. Obligation Decomposition Engine
**Category**: AI/ML
**Justification**: A single regulation (e.g., Kenya Data Protection Act) contains 80+ distinct obligations. Manually decomposing them is a 3-week project. ContractPodAi and Certa both offer AI-driven obligation extraction — making this table stakes for enterprise sales.
**Implementation**: Accept a raw regulation text blob; chunk it into clauses; for each clause produce a candidate `create_requirement` payload using a structured-output LLM call; return a list of pre-filled requirements ready for one-click import.
**Competitive reference**: ContractPodAi, Certa

---

### I3. Regulatory Penalty Exposure Calculator
**Category**: Compliance
**Justification**: Boards and audit committees ask "what is our maximum fine exposure?" on every compliance review. No open-source compliance tool computes this; enterprise tools like LogicGate charge per calculation. Embedding the calculator turns a risk conversation from qualitative to quantitative.
**Implementation**: Store `max_penalty_formula` per requirement (e.g., `"4% of global turnover OR EUR 20M"`); evaluate the formula at query time using `Decimal` arithmetic and return `estimated_max_exposure` and `estimated_likely_exposure` per requirement and in aggregate.
**Competitive reference**: LogicGate, Gallagher Bassett

---

### I4. Compliance Score with Trend History
**Category**: Feature
**Justification**: A single compliance_rate point-in-time snapshot is insufficient for board reporting. Drata and Vanta both provide weekly trend charts; CISOs demand "are we improving?" with statistical confidence.
**Implementation**: Persist a daily `score_snapshot` keyed by `YYYY-MM-DD` recording compliance_rate, open_breaches, critical_count; expose `get_compliance_trend(tenant_id, days=90)` returning snapshots with delta and direction indicators.
**Competitive reference**: Drata, Vanta

---

### I5. Evidence Chain-of-Custody Tracking
**Category**: Security
**Justification**: Regulators (FCA, OAG Kenya) require an auditable chain-of-custody for every evidence item — who touched it, when, what changed. Without this, evidence can be challenged in enforcement proceedings.
**Implementation**: Add an `evidence_chain` list to each evidence record; every mutation appends `{"actor_id", "action", "timestamp", "field_delta"}`; expose `get_evidence_chain(tenant_id, evidence_id)` returning the immutable custody log.
**Competitive reference**: Gallagher Bassett, Aderant

---

### I6. Cross-Requirement Dependency Graph
**Category**: Feature
**Justification**: GDPR Article 30 depends on Article 13 which depends on Article 5. A breach in one cascades. No open tool maps these; the dependency graph prevents whack-a-mole compliance failures.
**Implementation**: Add `depends_on` and `required_by` to requirement records; on `flag_non_compliant`, recursively mark dependents as `needs_review`; expose `get_dependency_graph(tenant_id, requirement_id)` returning a tree of affected nodes.
**Competitive reference**: Navex Global EthicsPoint, MetricStream

---

### I7. Automated Remediation Plan Generator
**Category**: AI/ML
**Justification**: Average time to produce a remediation plan after a breach is 5 business days. Archer GRC auto-generates plans from breach context; embedding this collapses that to minutes.
**Implementation**: On `create_breach`, auto-generate a `remediation_plan` record with steps derived from breach severity, category, and regulation; steps include SLA offsets in hours relative to discovery_date.
**Competitive reference**: Archer GRC (RSA), ServiceNow GRC

---

### I8. Regulatory Deadline Countdown with Breach Notification SLA
**Category**: Compliance
**Justification**: GDPR and Kenya DPA mandate 72-hour breach notification. Missing this SLA triggers fines larger than the original breach. No generic compliance tool enforces notification SLAs with countdown precision.
**Implementation**: On `create_breach` with `notification_required=True`, compute `notification_sla_expires_at` as `discovery_date + 72h`; expose `get_breach_sla_status()` returning `hours_remaining`, `is_overdue`, `sla_status: "green"|"amber"|"red"`.
**Competitive reference**: OneTrust, TrustArc

---

### I9. Multi-Jurisdiction Conflict Detector
**Category**: Compliance
**Justification**: Multinational entities face conflicting obligations — GDPR says delete data, AML says retain 7 years. Refinitiv World-Check and Dow Jones Risk & Compliance flag these automatically; no OSS tool does.
**Implementation**: On `create_requirement`, scan existing requirements with overlapping category for conflicting retention/frequency fields; return `conflicts: list[dict]` with conflicting requirement IDs and plain-language conflict descriptions.
**Competitive reference**: Refinitiv World-Check, Compliance.ai

---

### I10. Evidence Expiry and Gap Analysis
**Category**: Compliance
**Justification**: Certificates expire. Policies go stale. An audit that finds expired evidence is as bad as no evidence. Qualys and Drata run continuous evidence gap scans; this makes leg_cpl audit-ready at any point in time.
**Implementation**: `get_evidence_gap_report(tenant_id)` iterates all active requirements; for each checks if it has active, non-expired evidence; returns per-requirement `has_evidence`, `evidence_count`, `any_expired`, `expiring_in_30d`.
**Competitive reference**: Drata, Qualys

---

### I11. Regulator Communication Log
**Category**: Feature
**Justification**: Every correspondence with a regulator (DPA, FCA, CBK) must be logged for potential litigation. Aderant and iManage track this natively; without it, legal teams maintain ad-hoc email folders that don't survive staff turnover.
**Implementation**: Expose `log_regulator_communication(tenant_id, entity_id, regulator, direction, summary, reference, medium, actor_id)` and `list_regulator_comms(tenant_id, entity_id)` returning a chronological log.
**Competitive reference**: Aderant, iManage

---

### I12. Compliance Cost Tracking
**Category**: Feature
**Justification**: CFOs demand ROI on compliance spend. Compliance costs are tracked in spreadsheets at 80% of companies. MetricStream and Riskonnect embed cost modules; surfacing total compliance cost per regulation gives legal ops a clear budget defence.
**Implementation**: Expose `log_compliance_cost(tenant_id, requirement_id, amount: Decimal, currency, cost_type, period, recorded_by)` and `get_compliance_cost_summary(tenant_id)` returning per-regulation and per-category totals as `Decimal`.
**Competitive reference**: MetricStream, Riskonnect

---

### I13. Role-Based Assignment and Workload Balancing
**Category**: UX
**Justification**: Without workload visibility some owners are overburdened and deadlines slip. Navex Global and SAI360 both surface per-owner workload; this closes the people-management gap.
**Implementation**: `get_owner_workload(tenant_id)` returns per-owner counts of active, non-compliant, overdue, and open-breach requirements; `reassign_requirement()` transfers ownership with an audit event.
**Competitive reference**: Navex Global, SAI360

---

### I14. Bulk Import / Export (CSV/JSON)
**Category**: UX
**Justification**: Every new client arrives with an existing compliance register in Excel. Manually re-entering 200+ requirements takes 2 days and introduces errors. Vanta and Drata both offer one-click import; bulk import is a critical adoption accelerator.
**Implementation**: `bulk_import_requirements(tenant_id, records, *, dry_run=False)` validates each record, returns `{"created": int, "errors": list[dict]}`; `bulk_export_requirements(tenant_id, format="json")` serialises all requirements to JSON or CSV.
**Competitive reference**: Vanta, Drata

---

### I15. Automated Compliance Attestation Workflow
**Category**: Compliance
**Justification**: SOX, ISO 27001 and GDPR require periodic management attestation — a named officer confirming a control is in place. ServiceNow GRC and Archer automate the attestation cycle with digital sign-off; embedding this replaces a $30k/year module.
**Implementation**: `create_attestation_request()` creates a pending attestation; `submit_attestation()` records the sign-off with timestamp and updates `last_assessed_at`; `list_pending_attestations()` returns all outstanding requests.
**Competitive reference**: ServiceNow GRC, Archer GRC (RSA)
