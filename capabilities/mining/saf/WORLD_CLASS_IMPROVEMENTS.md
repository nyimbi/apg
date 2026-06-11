# Mine Safety & Compliance — World-Class Improvements

**Capability**: `mining_saf` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Bowtie Risk Analysis Engine

**Current gap**: Risk assessments capture hazards and controls as flat lists with no causal structure.

**Improvement**: Implement structured bowtie analysis — explicit separation of threat sources (causes), the top event (material unwanted event), left-side prevention controls (pre-event), right-side mitigation controls (post-event), and escalation factors. Each control links to a critical control designation flag. The bowtie model enables bidirectional traceability from an incident back to the controls that should have prevented it.

**Value**: Satisfies ICMM Critical Control Management guidance; directly usable for bow-tie workshops mandated by major mining standards (ISO 31000, AS/NZS 4804).

---

## 2. Automated LTIFR / TRIFR Trend Anomaly Detection

**Current gap**: `safety_statistics()` computes point-in-time rates using a hardcoded hours-worked estimate (200,000 hrs/month).

**Improvement**: Persist actual hours-worked per period from the `mining_pro` production capability. Compute rolling 3-month and 12-month LTIFR and TRIFR. Flag statistically significant upward trend deviations using a control-chart (XmR) approach. Expose trigger thresholds as tenant configuration. Integrate with `ntfy` to push alerts when control limits are exceeded.

**Value**: Converts lagging indicator reporting into an early-warning signal. Meets requirement for continuous monitoring under ISO 45001 §9.1.1.

---

## 3. Permit-to-Work Conflict Detection

**Current gap**: `issue_permit()` and `permit_to_work()` do not check whether simultaneous permits in the same area create incompatible work conditions (e.g. hot-work adjacent to a confined-space entry with flammable gas).

**Improvement**: On permit issue, scan all active permits for the same `mine_area`. Apply a conflict matrix (hot_work × confined_space_entry → BLOCK; hot_work × electrical_isolation → WARN; etc.). Configurable matrix stored as tenant JSON. Return conflict severity in the permit response and optionally block high-severity conflicts.

**Value**: Eliminates a known fatality mechanism in mining — simultaneous incompatible permits. Cited in numerous coronial inquiries as a root-cause failure.

---

## 4. ISO 45001 Compliance Gap Assessment

**Current gap**: `compliance_report()` returns a stub with status "compliant" regardless of actual data.

**Improvement**: Map each ISO 45001 clause (4.1–10.3) to observable data points within the capability (incident investigation completion rate, corrective action closure rate, audit schedule adherence, worker participation in hazard identification, management review completion). Compute a clause-level compliance score and produce a gap register with recommended actions.

**Value**: Replaces manual compliance gap analysis; provides auditable evidence for third-party certification bodies.

---

## 5. Hierarchical Controls Effectiveness Scoring (HIRAC)

**Current gap**: Control measures are recorded but there is no mechanism to score whether higher-order controls (elimination, substitution) are being preferred over lower-order ones (PPE).

**Improvement**: Assign a hierarchy weight to each `ControlType` (elimination=1.0, substitution=0.85, engineering=0.7, administrative=0.4, PPE=0.2). Compute a Hierarchy of Controls Index (HCI) per risk register entry and per inspection finding. Flag when PPE is the sole control for a high or extreme risk.

**Value**: Incentivises more robust risk treatment and quantifies engineering-over-PPE programme effectiveness — a key requirement of modern safety management systems.

---

## 6. Stop-Work Authority (SWA) Analytics and Reporting

**Current gap**: `stop_work_invoked` is a boolean field on hazards; there is no lifecycle, no reporting, and no linkage to resumption authorisation.

**Improvement**: Add a full SWA entity: invoke (who, when, location, reason), hold status, investigation record ID, resumption authorisation (who, when, conditions), and elapsed hold time. Compute SWA frequency rate per area and per shift. Track SWA-to-resumption cycle time. Export SWA trend data for management review.

**Value**: Safety culture benchmark — high SWA invocation rates signal a healthy reporting culture. Low rates in high-risk operations are a red flag indicator per SafeWork Australia guidance.

---

## 7. Isolation and Lockout/Tagout (LOTO) Register

**Current gap**: PTW records isolation points as a plain string list with no state tracking.

**Improvement**: Introduce a dedicated LOTO isolation register: each isolation point has type (electrical, mechanical, hydraulic, pneumatic, gravitational), device ID, isolation verified by, verified at, and reinstatement verified by. PTW issuance validates that all isolation points have a `verified` state. Reinstatement records created on PTW close trigger automatic isolation removal workflow.

**Value**: LOTO failures are the leading cause of fatal electrocution and entrapment incidents in mining. A structured register with state verification closes the most common PTW compliance gap.

---

## 8. Training Competency Gate on PTW Issuance

**Current gap**: `issuer_id` is recorded but there is no validation that the issuer holds a current competency for the PTW type.

**Improvement**: Integrate with the `auth` capability's role-competency map. On PTW issuance, look up the issuer's competency records for the `ptw_type`. Block issuance if competency is absent or expired. Return competency expiry date in the permit response for audit trail. Cache competency lookups with a 15-minute TTL.

**Value**: Closes the statutory issuer qualification gap (required under most national mines regulations). Eliminates manual cross-checking against training records.

---

## 9. Real-Time Area Risk Heat Map Data

**Current gap**: Hazard and incident data exists but there is no spatial aggregation to identify hot-spot areas.

**Improvement**: Add `get_area_risk_heatmap()` that aggregates open extreme/high hazards, incidents in the past 30 days, overdue corrective actions, and active permits by `mine_area`. Return a ranked list with a composite risk score per area. Include trend (worsening / stable / improving) based on a 90-day rolling comparison. Feed data to `mining_mon` for real-time dashboard overlay.

**Value**: Shift supervisor situational awareness tool. Enables targeted pre-shift safety briefings and resource allocation to highest-risk areas.

---

## 10. Leading Indicator Dashboard Feed

**Current gap**: `safety_statistics()` computes lagging indicators (LTIFR, fatalities). Leading indicators are not tracked.

**Improvement**: Compute and expose leading indicators: near-miss reporting rate, hazard identification rate per shift, toolbox talk completion rate (sourced from `mining_pro`), SWA invocation frequency, corrective action on-time closure rate, safety inspection completion vs schedule rate, and critical control verification pass rate. Bundle as a `LeadingIndicatorSnapshot` with period comparison and trend direction.

**Value**: ISO 45001 §9.1 and major mining standards require balanced leading/lagging indicator monitoring. Leading indicators are the only mechanism to detect emerging risk before an incident occurs.

---

## 11. Incident Causal Factor Classification (Tripod Beta / ICAM)

**Current gap**: Incidents store `immediate_cause` and `root_cause` as unstructured free text.

**Improvement**: Add structured causal factor tagging using ICAM (Incident Cause Analysis Method) categories: absent/failed defences, individual/team actions, task/environment conditions, and organisational factors. Each incident investigation records one or more causal factors from a controlled taxonomy. This enables causal pattern analysis across incidents: which organisational factors recur, which defences fail most often.

**Value**: Transforms incident investigation from a one-off document exercise into a continuous organisational learning system. Required for ICMM member operations.

---

## 12. Automated Regulatory Submission Deadline Tracking

**Current gap**: `regulatory_report_safety()` records a `submission_deadline` but does not actively track or alert on it.

**Improvement**: On regulatory report creation, register the submission deadline with the `wflo` workflow engine. Generate escalating reminders at 30, 7, and 1 day before deadline. On deadline breach, automatically escalate to the defined responsible officer. Track submission status (draft → submitted → acknowledged → accepted/rejected). Record rejection reasons and resubmission history.

**Value**: Regulatory non-compliance due to missed submission deadlines is a prosecutable offence in most mining jurisdictions. Automated tracking eliminates this entirely avoidable risk.

---

## 13. Fatigue Risk Management Integration

**Current gap**: Incident records do not capture worker fatigue state; there is no fatigue risk input to hazard assessment.

**Improvement**: Add fatigue risk fields to incident reports (shift number, hours worked in past 24/48h, last rest break). Integrate with `mining_pro` shift scheduling to expose a Fatigue Risk Index per worker per shift, derived from FAID-equivalent algorithm (cumulative sleep debt model). Gate PTW issuance and critical task assignment against Fatigue Risk Index thresholds. Flag incidents where fatigue was a contributing factor.

**Value**: Industry data attributes 20-30% of mining incidents to fatigue. Structured fatigue risk management is mandated under Australian WHS Regulations 2017 and similar frameworks.

---

## 14. Chemical and Dust Exposure Registry

**Current gap**: Hazard categories include `DUST_FUMES` and `CHEMICAL` but there is no structured exposure record linked to persons or health surveillance triggers.

**Improvement**: Add a chemical/substance exposure registry: substance name, CAS number, exposure level (TWA, STEL), measurement method, date, worker IDs exposed, relevant MSDS/SDS reference, and whether the exposure exceeds occupational exposure limit (OEL). Auto-generate health surveillance triggers when OEL is exceeded. Link exposures to incident records where substance release is involved. Feed occupational hygiene trend reports.

**Value**: Satisfies regulatory occupational hygiene monitoring requirements (e.g. WHS Regulation 2017 Part 7.1) and enables proactive health surveillance programme management.

---

## 15. AI-Assisted Incident Pattern Recognition

**Current gap**: The current ML hook in `report_incident()` classifies individual incident severity but does not detect patterns across incidents.

**Improvement**: After each incident is reported, invoke an async background analysis (via Ollama-hosted LLM) that: (a) compares the incident description against the last 90 days of incidents in the same mine area; (b) identifies recurring causal themes using embedding similarity; (c) surfaces the most similar past incidents with their investigation findings and corrective actions; (d) flags whether the current incident matches a known unresolved recurring pattern. Return pattern analysis as a structured `IncidentPatternMatch` on the incident record.

**Value**: Most mining operations experience the same incident type multiple times before a fatality — pattern recognition interrupts that trajectory. Provides investigators with directly relevant prior-art from the organisation's own history rather than requiring manual search.
