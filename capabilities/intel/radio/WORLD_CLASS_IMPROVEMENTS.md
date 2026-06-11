# Radio Intelligence (intel_radio) — World-Class Improvements

15 high-leverage improvements ranked by operational impact.

---

## 1. Multi-Sensor Fusion with TDOA Geolocation

Replace the simple circular-mean DF algorithm with Time-Difference-of-Arrival
(TDOA) hyperbolic intersection. With 3+ receivers, TDOA delivers 50–200 m CEP
vs. >2 km for bearing-only fixes. Use Bancroft's method for closed-form
solution; fall back to Levenberg-Marquardt for overdetermined cases.

**Impact**: Order-of-magnitude improvement in emitter fix accuracy for any
target at ranges up to 500 km.

---

## 2. Real-Time Bytewax Stream Pipeline Integration

Replace the in-memory `_frequency_scans` / `_recordings` dicts with a proper
Bytewax dataflow that partitions by `tenant_id`, applies windowed aggregation,
and emits enriched `RadioSignalObservation` events to a Kafka / Redpanda topic.
Enables sub-second end-to-end latency from SDR capture to dashboard update.

**Impact**: Scales from single-tenant demo to multi-site, high-volume SIGINT
at >10 000 signals/s without architectural rewrites.

---

## 3. Spectrogram-Based Anomaly Detection (ML)

Integrate a lightweight ONNX model (e.g. RadioML 2018 CNN) for modulation
recognition on IQ snapshots. Replace the `identify_emitter` heuristic decision
tree with a classifier that outputs probability distributions over modulation
classes (AM, FM, BPSK, QPSK, 8PSK, QAM16/64, GFSK, CPFSK, PAM4, WBFM).

**Impact**: Raises emitter identification confidence from ~0.75 to >0.92 F1
on real-world signals; enables zero-shot detection of novel waveforms.

---

## 4. Cryptographic Evidence Chain (Hash-Linked Audit Log)

Replace the `audit_events` list with a hash-linked ledger (each entry includes
`prev_hash`). On export, the chain is verifiable end-to-end. Back with
PostgreSQL `GENERATED ALWAYS AS` expression columns for tamper evidence.

**Impact**: Satisfies legal admissibility requirements in court proceedings;
enables cross-tenant audit sharing without data exposure.

---

## 5. Adaptive Collection Scheduler with Priority Queue

Replace the static `_monitoring_schedules` dict with a heapq-backed adaptive
scheduler. Priorities are computed from: threat level of last observation,
recency, band plan classification, and spectrum occupancy delta. Urgent
frequencies jump the queue; quiet ones backoff exponentially.

**Impact**: Reduces missed-collection rate on high-priority targets by ~60 %
while keeping total receiver dwell time constant.

---

## 6. PostgreSQL Persistent Store with Alembic Migrations

The current in-memory stores (`observations`, `sessions`, etc.) are reset on
every process restart. Wire `service.py` to a SQLAlchemy async engine using the
existing `alembic/` scaffold. Use `AsyncSession` + `select()` with
tenant-scoped row-level security policies at the PostgreSQL level.

**Impact**: Enables persistent, crash-safe operation and multi-instance
deployments (containerised or bare-metal).

---

## 7. Websocket Push for Live Spectrum Dashboard

Add a `FastAPI` WebSocket endpoint (`/ws/intel-radio/spectrum/{tenant_id}`)
that pushes frequency_scan results in real time. The Flask-AppBuilder dashboard
blueprint subscribes via JS EventSource. Use Redis pub/sub as the fanout
broker.

**Impact**: Operators see spectrum changes within 100 ms instead of polling
every 30 s; dramatically improves reaction time to pop-up emitters.

---

## 8. ARDF / Foxhunt Support with Doppler DF

Add `doppler_direction_finding()` for portable single-antenna receiver support.
Implements Doppler shift estimation from a 4-element switched antenna array.
Outputs bearing with ±2° accuracy at VHF without requiring multiple
geographically separated sites.

**Impact**: Supports foot-mobile DF teams in denied-access urban environments
where deploying multiple receiver sites is impractical.

---

## 9. Inter-Capability Composition Bus Events

Every significant `_audit()` call should also publish a structured CloudEvent
to the APG composition bus (`apg.intel.radio.*` topic namespace). Other
capabilities (intel_threats, intel_correlation) subscribe and cross-correlate
SIGINT observations with HUMINT/OSINT without polling.

**Impact**: Unlocks automatic fusion with the full APG capability ecosystem;
eliminates manual data handoffs between analysts.

---

## 10. Structured ELINT/COMINT Product Templates

Add `generate_elint_product()` and `generate_comint_product()` methods that
produce NATO STANAG 4607 (GMTI), STANAG 4609 (MIIS), or national-equivalent
structured products. Include mandatory fields: classification markings,
dissemination controls (RELTO), originator, time-of-information.

**Impact**: Products are machine-ingestible by allied C2 systems without manual
reformatting; reduces analyst product-production time by ~40 %.

---

## 11. Frequency Deconfliction Engine

Before recording a new band plan or scheduling a collection, automatically
check for conflicts with existing allocations in the ITU Radio Regulations and
locally registered band plans. Return a conflict score and suggest alternative
frequencies. Integrate with the `_BAND_RANGES_MHZ` map and the tenant's own
`band_plans` store.

**Impact**: Prevents inadvertent interference with protected services (aviation
Nav-Aids, emergency services); reduces regulatory liability.

---

## 12. SDR Hardware Abstraction Layer (HAL)

Introduce an `SDRAdapter` abstract base class with concrete implementations for
RTL-SDR, HackRF, USRP, and KiwiSDR. `RadioIntelligenceService` accepts an
`sdr: SDRAdapter | None` injector. When present, `frequency_scan()` and
`signal_recording()` call the real hardware asynchronously; when absent, the
current deterministic simulation runs (preserving testability).

**Impact**: Transforms `intel_radio` from a pure-software capability into an
operationally deployable sensor management layer with no architectural changes
to callers.

---

## 13. Automated Frequency Mask (Geo-Fenced Exclusion Zones)

Add `register_exclusion_zone()` that attaches a geographic polygon + frequency
band to a tenant. All downstream scans, collections, and schedules
automatically skip or flag frequencies within the exclusion zone. Prevents
inadvertent monitoring of protected diplomatic or government frequencies.

**Impact**: Hard compliance guardrail that removes operator burden; required in
most national SIGINT frameworks.

---

## 14. Signal Correlation and Link Analysis Graph

Persist emitter–signal–session relationships as edges in a NetworkX or
neo4j-compatible graph. Expose `signal_link_graph()` returning adjacency data
for the APG `grph` capability. Supports temporal query: "which emitters were
co-active in band X between T1 and T2?"

**Impact**: Enables network analysis of adversary comms infrastructure;
correlates previously unlinked emitters via shared timing/frequency patterns.

---

## 15. Formal Specification Test Suite (Property-Based)

Replace hand-written unit tests with Hypothesis `@given` strategies covering:
frequency range invariants, tenant isolation, audit chain integrity, DF
geometry edge cases, and batch ingest idempotency. Target 100 % branch
coverage on `service.py`.

**Impact**: Catches regressions introduced during rapid capability extension;
gives confidence that guardrails hold under adversarial inputs.
