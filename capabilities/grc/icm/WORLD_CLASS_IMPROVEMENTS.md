# World-Class Improvements — grc_icm

© 2025 Datacraft | Author: Nyimbi Odero

## Overview

15 targeted improvements to elevate the Incident & Crisis Management capability
from solid baseline to best-in-class. Ordered by risk-adjusted impact.

---

### 1. Machine-Learning Severity Triage

Current severity assignment is manual. Replace with a zero-shot Ollama classifier
that scores incident descriptions against {low, medium, high, critical} before
persistence. Confidence < 0.7 triggers human-in-the-loop confirmation. Preserves
manual override.

**Impact**: P1 mis-triage drops ~40%; on-call fatigue reduced.

---

### 2. Playbook-Driven Response Automation

Add `activate_playbook(incident_id, playbook_id)`. Each playbook is a DAG of
tasks with owners, SLAs, and dependency edges. The service evaluates the DAG on
each status change and auto-advances tasks or fires escalation webhooks. Store
playbooks in `icm_playbooks`; activations in `icm_playbook_runs`.

**Impact**: Mean time-to-contain (MTTC) cut by >50% for known incident types.

---

### 3. MITRE ATT&CK Annotation

Enrich `incident_investigation` with optional `mitre_tactics: list[str]` and
`mitre_techniques: list[str]`. Persist to evidence record and surface in
`regulatory_reporting_icm`. Enables threat-intelligence correlation across
incidents and feeds into the intel-correlation capability.

**Impact**: Structural threat-intel output; regulatory reports become
court-admissible artefacts.

---

### 4. SLA Breach Tracking and Alerting

Add `sla_profile_id` to `report_incident`. A background async task (or
capability hook) periodically evaluates each open incident against its SLA
profile and sets `sla_breached = True` plus fires a notification. Expose
`get_sla_status(incident_id)` for polling.

**Impact**: Eliminates silent SLA breaches; feeds `incident_kpi_summary`
`sla_breach_rate_pct` with real data.

---

### 5. Digital Chain-of-Custody for Evidence

Current `chain_of_custody` list is append-only in memory. Replace with a
cryptographic log: each custody transfer appends `{actor, action, timestamp,
prev_hash, entry_hash}` where `entry_hash = SHA-256(prev_hash + actor +
action + timestamp)`. Expose `verify_evidence_chain(evidence_id)` to validate
integrity. Immutable once created.

**Impact**: Evidence survives legal challenges; satisfies ISO 27037 and NIST SP
800-86 requirements.

---

### 6. Stakeholder Communication Templates

Add `send_crisis_communication(incident_id, audience, template_id)` backed by
a template registry. Audiences: internal | external | regulator | press.
Templates stored in `icm_comm_templates`; rendered with incident context via
Jinja2-style substitution. Delivery tracks opens/bounces via `ntfy` capability.

**Impact**: Consistent, legally vetted messaging; audit trail for every
stakeholder communication.

---

### 7. Multi-Framework Regulatory Mapping

SUPPORTED_REGULATORY_WINDOWS currently keyed by framework name (gdpr, pci_dss).
Replace with a `RegulatoryProfile` object that carries notification window,
required fields, template references, and jurisdiction codes. `regulatory_notify`
selects the correct profile automatically and validates required evidence before
dispatch.

**Impact**: Correct first-time filing across GDPR, CBK, GDPA, PCI-DSS, SOX
without manual lookup.

---

### 8. Crisis Communication War-Room

Add `create_war_room(incident_id, participants, channel_type)`. Provisions a
temporary, audit-logged communication channel (Matrix/Slack/Teams via ntfy
adapter). All messages are persisted in `icm_war_room_logs` with immutable
timestamps. `close_war_room(war_room_id)` archives and seals the log.

**Impact**: Eliminates off-the-record crisis comms; full communication audit
trail for regulators.

---

### 9. Automated Business Impact Assessment (BIA)

Add `business_impact_assessment(incident_id, impacted_processes)`. Evaluates
RTO/RPO against current recovery state, calculates financial exposure per hour,
and ranks recovery priorities. Feeds into BCP activation recommendation.
Store in `icm_bia_records`.

**Impact**: BCP activations become data-driven; financial exposure known within
minutes of incident declaration.

---

### 10. Incident Similarity Search

Add `find_similar_incidents(incident_id, top_k=5)`. Uses TF-IDF over description
+ root_cause text stored in the lessons-learned library to surface past incidents
with high textual overlap. Returns similarity scores and applicable lessons.
Falls back gracefully if Ollama unavailable.

**Impact**: Reduces mean investigation time by surfacing institutional memory;
prevents re-inventing remediation.

---

### 11. Immutable Audit Ledger with Hash Chaining

Current audit events are stored as independent records. Replace with a
hash-chained ledger: `{id, prev_hash, entry_hash, event_type, actor, ...}`.
Expose `verify_audit_chain(start_id, end_id)` that walks the chain and
validates every hash. Any tampered record is immediately detectable.

**Impact**: Tamper-evident audit trail that satisfies CBK Prudential Guideline
requirements for financial institutions.

---

### 12. Tabletop Exercise Simulation

Add `run_tabletop_exercise(scenario_id, participants)`. Simulates an incident
scenario through the full lifecycle (report → triage → investigate → close)
using synthetic data, without touching production records. Generates a
`tabletop_exercise_report` with timing, decision quality scores, and gap
analysis.

**Impact**: Continuous BCP/IR readiness validation without production risk.

---

### 13. Automated MTTR Forecasting

Enhance `incident_kpi_summary` with a lightweight linear regression over
historical MTTR by incident type and severity. Returns `predicted_mttr_hours`
alongside actual. Model is retrained weekly using the lessons-learned corpus.
No external ML dependency — pure Python `statistics` module.

**Impact**: Realistic SLA setting; early warning when a new incident is tracking
above historical baseline.

---

### 14. Third-Party Incident Coordination

Add `third_party_incident_notify(incident_id, vendor_id, contact_email,
notification_scope)`. Manages vendor notification obligations (data processor
notification under GDPR Art. 33, supply-chain breach notifications). Tracks
vendor acknowledgement and stores response in `icm_vendor_notifications`.

**Impact**: Closes the most common GDPR compliance gap; vendor notification
SLAs tracked automatically.

---

### 15. Executive Incident Briefing Generator

Add `generate_executive_briefing(incident_id)`. Produces a structured one-page
summary: situation, impact, containment status, regulatory exposure, and next
actions. Template-driven with Markdown output. Designed for CISO→Board comms.
Stored in `icm_exec_briefings` with versioning.

**Impact**: Eliminates manual briefing prep under pressure; consistent format
builds board confidence.
