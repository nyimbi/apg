# Signals Intelligence User Guide

**Capability ID**: `intel_sigint` | **Domain**: `intel` | **Version**: `1.2.0`
**© 2025 Datacraft** — www.datacraft.co.ke

---

## Overview

`intel_sigint` provides a governed, tenant-scoped signals intelligence runtime
for APG-generated applications. It enforces lawful authority at every step,
records a tamper-apparent audit trail, and exposes an async-first service API
that composes with other APG capabilities through the standard adapter pattern.

The capability is divided into two tiers:

1. **Governance tier** — sync CRUD methods that manage authorities, sources,
   tasks, observations, patterns, assessments, and reviews under deterministic
   policy rules.
2. **Operational tier** — async methods for real-time and batch signal
   collection, emitter analysis, direction finding, satellite tasking, and
   intelligence reporting.

---

## Installation

```bash
pip install apg-intel-sigint
```

Or within a workspace:

```bash
uv add apg-intel-sigint
```

---

## Quick Start

```python
import asyncio
from capabilities.intel.sigint import SignalsIntelligenceService

svc = SignalsIntelligenceService("tenant-alpha", actor_id="analyst-1")

# 1. Register authority (required before any collection)
auth = svc.record_authority(
    "auth-001", "tenant-alpha", "mission_order",
    "scope://op-blackbird", "secret",
    "approver-jones", "2027-01-01", "evidence://auth-001",
)

# 2. Register a VHF sensor
source = svc.register_source(
    "src-vhf-01", "tenant-alpha", "radio", "vhf",
    "sensor://tower-4/vhf", "owner-signals", auth["id"],
    "evidence://src-001",
)

# 3. Collect and analyse
async def run():
    sig = await svc.collect_signal("am_voice", 156.8e6, "sensor://tower-4/vhf", {"target": "vessel-7"})
    quality = await svc.signal_quality(sig["signal_id"])
    triage = await svc.signal_triage([sig["signal_id"]])
    report = await svc.signal_intelligence_report("secret", ["hq-watch"])
    return sig, quality, triage, report

sig, quality, triage, report = asyncio.run(run())
```

---

## Governance Tier

### Authorities

Every collection activity must be backed by a registered authority.

```python
auth = svc.record_authority(
    authority_id,   # str — unique within tenant
    tenant_id,      # str
    authority_type, # one of SUPPORTED_AUTHORITY_TYPES
    scope_reference,
    classification, # one of SUPPORTED_CLASSIFICATIONS
    approver_id,
    expires_at,     # ISO-8601 date string
    evidence_reference,
    policy_attached=True,
)
```

Raises `PermissionError` if classification or authority_type is unsupported,
approver is absent, or expiry is not provided.

### Sources

```python
source = svc.register_source(
    source_id, tenant_id, source_type, band,
    source_reference, owner_id, authority_id, evidence_reference,
)
```

`source_type` must be in `SUPPORTED_SOURCE_TYPES`; `band` in `SUPPORTED_BANDS`.
The source is linked to the authority — mismatches raise `PermissionError`.

### Collection Tasks

```python
task = svc.record_collection_task(
    task_id, tenant_id, authority_id, source_id,
    collection_mode, retention_days,
    minimization_reference, approval_reference, evidence_reference,
)
```

`collection_mode` must be in `SUPPORTED_COLLECTION_MODES`.
`retention_days` must be a positive integer.

### Observations, Processing, Patterns, Assessments

```python
obs  = svc.record_observation(obs_id, tenant_id, task_id, ref, fingerprint, 0.85, evidence)
proc = svc.record_processing_batch(batch_id, tenant_id, obs_id, "demodulate", 0.9, "analyst-2", evidence)
pat  = svc.record_pattern(pat_id, tenant_id, batch_id, "burst", 0.75, "analyst-2", evidence)
asmt = svc.record_assessment(asmt_id, tenant_id, pat_id, "hostile_comms", "secret", "analyst-2", evidence)
rev  = svc.record_review(rev_id, tenant_id, asmt_id, "supervisor-1", "approved", evidence)
```

---

## Operational Tier (Async)

### Signal Collection

```python
# Single signal
sig = await svc.collect_signal(
    signal_type="fm_voice",
    frequency=162.55e6,
    source="sensor://vhf-1",
    metadata={"target_id": "vessel-7"},
)
# sig["band"] == "VHF", sig["signal_id"] == 16-char hex fingerprint

# Bulk concurrent collection
sigs = await svc.bulk_collect_signals([
    {"signal_type": "am", "frequency": 7.2e6,  "source": "hf-rx-1", "metadata": {}},
    {"signal_type": "usb", "frequency": 14.3e6, "source": "hf-rx-1", "metadata": {}},
])

# Full frequency sweep with configurable step
sweep = await svc.spectrum_sweep(start_hz=136e6, stop_hz=174e6, step_hz=0.5e6, source="vhf-rx-1")
```

### Communication Intercepts

Intercepts require a registered authority. Attempting to intercept without one raises `PermissionError`.

```python
intercept = await svc.intercept_communication(
    target_id="vessel-7",
    channel="VHF-Ch16",
    authority_ref="auth-001",
)

# Schedule intercepts across multiple channels at once
schedule = await svc.intercept_schedule(
    target_id="vessel-7",
    channels=["VHF-Ch16", "VHF-Ch67", "VHF-Ch72"],
    authority_ref="auth-001",
)
```

### Emitter Identification and Fingerprinting

```python
# Heuristic identification from RF characteristics
emitter = await svc.emitter_identification({
    "frequency_hz": 9.4e9,
    "modulation": "pulsed",
    "pulse_width_us": 1.2,
    "pri_us": 1000.0,
    "power_dbm": 55.0,
})
# emitter["emitter_type"] == "SEARCH_RADAR"

# RF fingerprint from imperfection metrics
fp = await svc.emitter_fingerprint(signal_id, {
    "frequency_offset_ppm": 1.3,
    "phase_noise_dbc_hz": -95.0,
    "startup_transient_us": 12.5,
    "modulation_error_ratio_db": 28.0,
    "harmonic_distortion_db": -42.0,
})

# Re-identify emitter against stored fingerprints
match = await svc.emitter_reidentify({
    "frequency_offset_ppm": 1.4,
    "modulation_error_ratio_db": 27.5,
    "phase_noise_dbc_hz": -94.0,
}, tolerance_ppm=2.0)
# match["matched"] == True, match["similarity_score"] == 0.98xx
```

### Direction Finding

```python
fix = await svc.direction_finding(
    signal_id=sig["signal_id"],
    sensor_positions=[
        {"lat": -1.286, "lon": 36.817, "bearing_deg": 127.5},
        {"lat": -1.305, "lon": 36.835, "bearing_deg": 131.0},
        {"lat": -1.270, "lon": 36.800, "bearing_deg": 124.0},
    ],
)
# fix["estimated_bearing_deg"] — circular mean bearing
# fix["quality_score"] — 0-1, higher is tighter sensor agreement
```

### ELINT — Pulse Descriptor Word Extraction

```python
pdw = await svc.elint_pdw_extract(
    observation_id="obs-radar-001",
    raw_pulse_data=[
        {"toa_us": 0.0,      "pw_us": 1.2, "rf_mhz": 9400.0, "amplitude_dbm": -45.0},
        {"toa_us": 1000.0,   "pw_us": 1.2, "rf_mhz": 9400.0, "amplitude_dbm": -46.0},
        {"toa_us": 2000.0,   "pw_us": 1.2, "rf_mhz": 9400.0, "amplitude_dbm": -44.0},
    ],
)
# pdw["emitter_category"] == "LONG_RANGE_SEARCH_RADAR"
# pdw["mean_pri_us"] == 1000.0, pdw["duty_cycle"] == 0.0012
```

### Traffic and Pattern Analysis

```python
# Metadata traffic analysis between two endpoints
traffic = await svc.traffic_analysis(
    source="sensor://vhf-1",
    destination="target://vessel-7",
    period="24h",
)
# traffic["burst_detected"], traffic["transmission_count"]

# Hourly distribution and entropy for a specific target
pattern = await svc.communication_pattern_analysis(
    target_id="vessel-7",
    period="7d",
)
# pattern["high_regularity"] == True/False (entropy < 0.4 threshold)
# pattern["peak_hour_utc"] — hour with most activity
```

### Signal Triage

```python
triage_results = await svc.signal_triage([sig_a["signal_id"], sig_b["signal_id"]])
# Each result: {category: "encrypted|cleartext|compressed|noise", routing: "...", priority: 0-4}
```

### Spectrum Anomaly Detection

```python
anomaly = await svc.spectrum_anomaly_detect(
    band="VHF",
    window_size=24,
    sigma_threshold=3.0,
)
# anomaly["anomaly_detected"] == True/False
# anomaly["current_count"], anomaly["threshold"]
```

### Natural-Language Collection Tasking

```python
task_draft = await svc.task_from_natural_language(
    instruction="Monitor VHF 136-174 MHz for burst transmissions, retain for 7 days",
    authority_id="auth-001",
)
# task_draft["parsed"]["detected_band"] == "VHF"
# task_draft["parsed"]["frequency_range"] == {"start_hz": 136e6, "stop_hz": 174e6, "unit": "MHz"}
# task_draft["parsed"]["collection_mode"] == "burst"
# task_draft["parsed"]["retention_days"] == 7
```

### Satellite Operations

```python
# Task a satellite intercept
sat_task = await svc.satellite_intercept(
    target_orbit="GEO",
    frequency_band="Ku",
)

# Full link budget analysis
budget = await svc.satellite_link_budget(
    centre_frequency_hz=12.5e9,    # Ku downlink
    distance_km=35786,             # GEO slant range
    tx_eirp_dbw=55.0,              # satellite EIRP
    rx_gain_dbi=42.0,              # 1.2m dish
    system_noise_temp_k=200.0,
    bandwidth_hz=36e6,             # 36 MHz transponder
)
# budget["feasible"] == True/False (C/N >= 3 dB threshold)
# budget["cn_db"], budget["shannon_capacity_bps"]
```

### Signal Decryption

```python
result = await svc.decrypt_signal(
    raw_signal="khoor",
    decryption_key="unused",
    method="rot13",
)
# result["plaintext_fingerprint"] — SHA-256 digest of plaintext
# result["key_fingerprint"] — SHA-256 digest of key (key never returned)
```

### Intelligence Reports

```python
report = await svc.signal_intelligence_report(
    classification="secret",
    recipients=["hq-watch", "j2-intel"],
)
# report["summary"]["signals_collected"], ["emitter_types_identified"], etc.
```

### Differential Privacy Analytics

```python
# Export analytics with Laplace noise for sharing with lower-classification partners
dp = await svc.differential_privacy_analytics(epsilon=1.0)
# dp["noised_counts"]["signals_collected"] — safely noised integer
# dp["exact_counts"] — exact values retained for internal audit
# dp["privacy_guarantee"] == "epsilon-DP with epsilon=1.0"
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-sigint/dashboard` | `intel_sigint:view` | Overview |
| `/intel-sigint/authorities` | `intel_sigint:authorities` | Governance |
| `/intel-sigint/sources` | `intel_sigint:sources` | Collection |
| `/intel-sigint/collection-tasks` | `intel_sigint:collection` | Collection |
| `/intel-sigint/observations` | `intel_sigint:observations` | Processing |
| `/intel-sigint/processing` | `intel_sigint:processing` | Processing |
| `/intel-sigint/patterns` | `intel_sigint:patterns` | Analysis |
| `/intel-sigint/assessments` | `intel_sigint:assessments` | Analysis |

---

## Error Reference

| Exception | Trigger |
|-----------|---------|
| `PermissionError` | Policy rule violation (missing authority, unsupported type, etc.) |
| `ValueError` | Invalid parameter value (unsupported classification, method, orbit, etc.) |
| `KeyError` | Referenced entity (signal_id, emitter_id) not found in tenant context |
| `AssertionError` | Precondition failure (empty list, negative value, missing required key) |

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or
environment variables prefixed with `INTEL_SIGINT_`.

Key environment variables:
- `INTEL_SIGINT_DB_URL` — PostgreSQL connection string for the store adapter
- `INTEL_SIGINT_OLLAMA_URL` — Ollama endpoint for NL tasking (default: `http://localhost:11434`)
- `INTEL_SIGINT_DP_EPSILON` — Default differential privacy epsilon (default: `1.0`)
- `INTEL_SIGINT_ANOMALY_SIGMA` — Default anomaly detection sigma threshold (default: `3.0`)

---

## Interoperability

Reference in `.apg` source files:

```apg
use intel_sigint;
```

Compose with:
- `intel_radio` — hardware receiver adapters
- `intel_geoint` — geographic overlays for DF fixes
- `intel_correlation` — cross-capability signal correlation
- `intel_dashboard` — operational display
- `intel_reporting` — formal intelligence product generation

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 architectural improvements
- `cap_spec.md` — Formal capability specification
- `SPECIFICATION.md` — Detailed system specification
