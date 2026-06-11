# World-Class Improvements — grc_rcm

**Capability**: Risk & Compliance Management (`grc_rcm`)
**Author**: Nyimbi Odero | © 2025 Datacraft

---

## 1. Continuous Control Monitoring (CCM)

Replace point-in-time assessments with an always-on CCM engine that ingests telemetry (log streams, API call rates, configuration drift alerts) and updates control effectiveness in near-real-time. Controls shift from periodic sampling to evidence-based, time-weighted effectiveness scores. Eliminates the 90-day blind spot between scheduled assessments and reality.

## 2. Quantitative Risk Scoring (FAIR-aligned)

Replace the `likelihood × impact` proxy with a Monte Carlo simulator grounded in Factor Analysis of Information Risk (FAIR). Outputs loss exceedance curves, 95th-percentile annual loss expectancy, and confidence intervals rather than a single ordinal score. Boards can then compare risk to capital reserves on the same monetary scale.

## 3. Regulatory Change Intelligence Feed

Integrate a structured regulatory-change data bus (e.g., Wolters Kluwer, Thomson Reuters Regulatory Intelligence API) into the `regulatory_change_monitor` stub. When a new rule is published, a diff engine cross-references the existing obligation register and raises gap tickets automatically, reducing the lag from regulatory publication to internal action from weeks to hours.

## 4. Three-Lines-of-Defense Workflow Engine

Model first-line (business units), second-line (risk/compliance), and third-line (internal audit) roles as first-class entities with mandatory hand-off gates. Service methods enforce segregation-of-duties: a second-line officer cannot also be the first-line evidence preparer for the same control. Replaces ad-hoc review fields with a verified chain of custody.

## 5. Natural Language Obligation Parsing

Accept obligation text in free-form prose (e.g., pasted regulation article) and run it through a structured extraction pipeline (NER + dependency parsing) to auto-populate `framework`, `requirement`, `jurisdiction`, `due_date`, and candidate `mapped_control_ids`. Reduces obligation registration from 20-minute manual entry to a 30-second review-and-confirm loop.

## 6. Control Testing Automation Harness

Expose a `ControlTestScript` model and a `run_automated_test` service method. Scripts declare their expected evidence schema; the harness executes them against live system state (e.g., IAM policy API, firewall rule export, database audit log query) and stores typed evidence without human transcription. Reduces human labour per control test by ~70 %.

## 7. Risk Appetite Statement as Executable Policy

Represent the board-approved risk appetite statement as a machine-readable policy (OPA/Rego or a JSON-schema rule set) stored in the `_risk_appetite` store. Every `register_risk` and `risk_treatment` call evaluates the policy live, returning structured violations and required escalation actions instead of a binary within/outside flag. Closes the gap between governance documents and runtime enforcement.

## 8. Issue Ageing and SLA Breach Detection

Track `open_since` against per-severity SLA targets (e.g., critical → 48 h, high → 5 d). A `check_issue_slas` method scans all open issues, computes days overdue, and raises `issue_sla_breached` audit events automatically. Feeds dashboard heat-map with overdue counts before management reviews stale tickets manually.

## 9. Audit Evidence Chain of Custody

Add a cryptographic hash (`sha256`) and digital signature field to every `RCMEvidence` record. A `verify_evidence_chain` method replays all evidence hashes for a given control or obligation and validates an unbroken chain from collection to audit submission. Satisfies e-discovery requirements and removes manual chain-of-custody binders.

## 10. Predictive Risk Velocity

Extend the risk model with a `velocity` dimension computed from the rate of change of residual score over rolling 30/90-day windows. A `predict_risk_trajectory` method fits a linear trend and projects the residual score 90 days forward with a prediction interval. Risk owners receive early-warning signals before risks cross severity thresholds, not after.

## 11. Cross-Capability Risk Propagation Graph

Build a directed graph where nodes are risks, controls, obligations, and external capability events (e.g., a `situ_threat` event from the threat-intel capability). A `propagate_risk_impact` method walks the graph to identify downstream risks that inherit elevated likelihood when an upstream risk is re-scored. Enables cascading risk reassessment without manual dependency tracing.

## 12. Compliance Posture Benchmarking

Compare the organisation's control effectiveness distribution against an anonymised industry peer cohort (sector, size, jurisdiction). A `benchmark_compliance_posture` method returns percentile rank per framework, identifying which control domains are below-median versus peer group. Converts internal scores into externally calibrated signals for the board.

## 13. Exception Lifecycle Management with Auto-Expiry

Add a background task (`expire_exceptions`) that scans `exceptions` for records whose `expiration_date` has passed, marks them `expired`, raises an audit event, and auto-generates a follow-up issue requiring the control gap to be re-addressed or the exception renewed. Prevents stale exceptions from silently accumulating unlimited risk exposure.

## 14. Policy-as-Code Version Control

Store each compliance framework as a versioned, content-addressed policy artefact (semantic version + SHA). A `policy_diff` method compares two versions of a framework to show which requirements changed, were added, or were removed. Obligation registers link to specific policy versions so auditors can trace which exact regulation revision drove each control.

## 15. AI-Assisted Risk Narrative Generation

Add a `generate_risk_narrative` service method that takes a structured risk record and renders a board-ready narrative paragraph using a locally hosted Ollama model (e.g., `mistral` or `llama3`). The narrative explains the risk in plain language, references linked controls and their effectiveness, and suggests treatment priority. Reduces report-writing time and improves consistency of board risk packs.
