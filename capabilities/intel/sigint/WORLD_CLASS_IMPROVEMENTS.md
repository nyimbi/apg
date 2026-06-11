# SIGINT Capability — World-Class Improvements

**Capability**: `intel_sigint` | **Domain**: `intel` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Wideband IQ Recording Pipeline

Replace discrete point-in-time signal snapshots with a continuous IQ (in-phase / quadrature) sample stream per sensor. Store raw complex float32 samples with timestamps in a columnar format (Apache Arrow / Parquet over S3-compatible storage). This enables post-hoc demodulation with any algorithm without re-collection, and feeds offline SDR toolchains (GNU Radio, SoapySDR) directly.

**Impact**: Eliminates lossy one-shot observation; every demodulation decision becomes reversible.

---

## 2. Probabilistic Emitter Fingerprint Database

Maintain a tenant-scoped library of emitter fingerprints — RF fingerprints derived from transient startup behaviour, frequency drift, phase noise, and modulation imperfections. Each new signal is scored against the library with a Bayesian update step, producing posterior probabilities over known-emitter hypotheses. Fingerprints persist across collection sessions, enabling long-term re-identification of the same physical hardware even when operators change frequencies or call signs.

**Impact**: Moves emitter identification from heuristic `if/elif` chains to a data-driven, self-improving probabilistic model.

---

## 3. TDOA-Based High-Precision Geolocation

Replace the current circular-mean bearing estimator with a Time Difference of Arrival (TDOA) hyperboloid intersection solver. Accept nanosecond-precision timestamps from a GPS-disciplined sensor network. Fit least-squares intersection of multiple hyperbolas using a Levenberg-Marquardt solver. Report 50/95 % confidence ellipses in WGS-84, ready for GeoJSON export.

**Impact**: Reduces CEP from kilometres to tens of metres for cooperative sensor networks.

---

## 4. Real-Time Spectrum Anomaly Detection

Run an online one-class SVM (or Isolation Forest) over rolling spectral power density estimates. Raise an `anomaly_detected` event to the Bytewax stream when occupancy in a band deviates beyond 3-sigma from the rolling 24-hour baseline. Feed anomalies into the existing pattern workflow automatically.

**Impact**: Eliminates manual scanning duty cycle; surfaces novel signals within seconds.

---

## 5. Automated ELINT Parameter Extraction

After observation ingestion, automatically run pulse descriptor word (PDW) extraction on radar-class signals: pulse width, PRI, RF, scan type, scan period, and antenna lobe parameters. Store extracted PDWs in a structured `elint_pdw` table. Map PDWs to known system types via a configurable emitter library (EW-103 compatible schema).

**Impact**: Transforms raw radar observations into actionable Electronic Order of Battle (EOB) data without analyst intervention.

---

## 6. Lawful Intercept Compliance Ledger with Merkle Proofs

Extend the audit log to produce a tamper-evident Merkle tree over all intercept events per collection day. Store the daily Merkle root in a notarised append-only log (Rekor / Transparency Log compatible). Provide a `verify_intercept_chain(authority_id)` method that returns a Merkle proof for any intercept record, enabling third-party legal review without exposing other records.

**Impact**: Converts the current flat audit list into court-admissible evidence with cryptographic provenance.

---

## 7. Federated Multi-Tenant Signal Fusion

Introduce a `FederatedSIGINTBroker` that mediates cross-tenant signal sharing under explicit data-sharing agreements (DSAs). Tenants publish anonymised signal fingerprints to a shared correlation bus; the broker resolves matches and returns only the attributes permitted by the DSA policy. Full signal data never leaves the originating tenant boundary.

**Impact**: Enables coalition-grade intelligence fusion while preserving data sovereignty and legal boundaries.

---

## 8. Natural-Language Collection Tasking

Expose a `task_from_natural_language(instruction: str)` method that parses plain-language tasking orders (e.g., "Monitor VHF band between 136 and 174 MHz for burst transmissions from grid reference 1234") into structured `CollectionTask` objects. Use a local Ollama-hosted LLM (Mistral-7B or Llama-3.1-8B) with a constrained grammar sampler to guarantee valid output schemas.

**Impact**: Reduces analyst tasking time from minutes to seconds and eliminates transcription errors from verbal orders.

---

## 9. Modulation Classification Neural Network

Train a lightweight 1-D CNN on spectrogram slices to classify modulations (AM, FM, SSB, LSB, USB, DSB, CW, PSK31, RTTY, DMR, TETRA, APCO-P25, ADS-B, Mode-S). Expose as `classify_modulation(iq_samples: np.ndarray) -> ModulationClassification`. Use ONNX Runtime for inference so the model runs on edge hardware without GPU.

**Impact**: Replaces the current lookup-table modulation heuristic with a model achieving >95 % accuracy on RadioML2018 benchmarks.

---

## 10. Bytewax Stateful Signal Processing Topology

Replace `asyncio.gather` fan-outs with a proper Bytewax dataflow topology that maintains per-source state machines: `idle → collecting → processing → assessed`. Route signals through configurable operator chains (bandpass filter → demodulate → decode → classify → assess) with backpressure and exactly-once semantics via Kafka offsets.

**Impact**: Provides durable, resumable, horizontally scalable signal processing with replay-from-offset debugging.

---

## 11. GraphRAG Signal Relationship Index

Build a property graph (Neo4j or Memgraph) where nodes are signals, emitters, sources, targets, authorities, and assessments; edges are `detected_by`, `attributed_to`, `authorised_by`, `correlated_with`. Expose `graph_query(cypher: str)` for arbitrary relationship traversal and `shortest_path(entity_a, entity_b)` for link analysis. Feed the graph into a GraphRAG pipeline for natural-language Q&A over the signals corpus.

**Impact**: Surfaces non-obvious relationships between signals and entities that tabular queries miss entirely.

---

## 12. Adaptive Collection Scheduling with Reinforcement Learning

Train a Proximal Policy Optimisation (PPO) agent to allocate sensor dwell time across competing collection tasks. State: current spectrum activity scores per band, outstanding task priorities, sensor availability. Action: dwell duration per task. Reward: weighted sum of new-entity detections minus collection cost. Re-train daily on collected data using a local Ray RLlib instance.

**Impact**: Outperforms static round-robin scheduling by 30-50 % on entity discovery rate in contested spectrum environments.

---

## 13. Encrypted Signal Triage Queue

Before analyst review, run automated triage to classify signals as `cleartext`, `encrypted`, `compressed`, `obfuscated`, or `noise`. Use Shannon entropy thresholds and byte-frequency chi-square tests. Route encrypted signals to the decryption workflow and cleartext to the pattern workflow automatically. Maintain a triage queue with priority scoring based on target importance and signal novelty.

**Impact**: Eliminates the analyst time spent manually routing signals; high-priority encrypted traffic reaches decryption within seconds of collection.

---

## 14. Satellite Link Budget and Intercept Feasibility Calculator

Given a satellite TLE, ground station position, and antenna parameters, compute link budget (EIRP, path loss, receive G/T, Eb/N0, Shannon capacity) to determine whether intercept is technically feasible before tasking resources. Expose as `satellite_feasibility(tle: str, gs_lat: float, gs_lon: float, antenna_gain_dbi: float) -> FeasibilityReport`. Integrate with Skyfield for TLE propagation.

**Impact**: Prevents tasking of resources against targets with insufficient link margin, freeing capacity for feasible collections.

---

## 15. Differential Privacy for Analytics Export

When exporting analytics summaries to external stakeholders (partner agencies, oversight bodies), apply calibrated Laplace noise to frequency counts using differential privacy (epsilon=1.0 by default, configurable per data-sharing agreement). Provide `dp_analytics(epsilon: float) -> dict` alongside the existing `sigint_analytics()` method. Include a proof of epsilon-differential privacy in the export metadata.

**Impact**: Enables sharing of aggregate intelligence metrics with lower-classification partners without exposing collection volumes, targets, or operational patterns.
