# APG Signals Intelligence

`intel_sigint` is the APG package-backed capability for governed
signals-intelligence applications. It composes authorities, sources, collection
tasks, observations, processing batches, patterns, assessments, reviews, Bytewax
lifecycle metadata, UI/view models, visual theming, and provider-neutral
AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and evidence.
- Signal source registry linked to authority.
- Collection task workflow with retention, minimization, approval, and evidence.
- Observation, processing, pattern, assessment, and review workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.sigint.lifecycle`.
- RF emitter fingerprinting and probabilistic re-identification.
- ELINT Pulse Descriptor Word (PDW) extraction and emitter category classification.
- TDOA-ready direction finding with circular-mean bearing and quality scoring.
- Satellite intercept tasking with free-space path loss computation.
- Full satellite link budget (FSPL, C/N, Eb/N0, Shannon capacity, feasibility verdict).
- Natural-language collection tasking with rule-based instruction parser.
- Signal triage queue (noise / cleartext / compressed / encrypted) via Shannon entropy.
- Spectrum anomaly detection using rolling statistical baseline.
- Differential privacy analytics export (Laplace mechanism, configurable epsilon).
- Cross-band correlation and intercept scheduling across multiple channels.
- Bulk signal collection and full-range spectrum sweep.
- Signal quality assessment (SNR proxy, band-based scoring).

## Use The Service

```python
from capabilities.intel.sigint import SignalsIntelligenceService
import asyncio

service = SignalsIntelligenceService("tenant-a", actor_id="analyst-1")

# Record lawful authority before any collection
authority = service.record_authority(
    "auth-1", "tenant-a", "mission_order",
    "scope://mission", "secret", "approver-1",
    "2026-12-31", "evidence://authority",
)

# Register a signal source tied to the authority
source = service.register_source(
    "source-1", "tenant-a", "radio", "vhf",
    "sensor://vhf-1", "owner-1", authority["id"],
    "evidence://source",
)

# Collect a signal asynchronously
async def example():
    sig = await service.collect_signal("am_voice", 162.55e6, "sensor://vhf-1", {})

    # Classify modulation, triage, and link budget in one pass
    triage = await service.signal_triage([sig["signal_id"]])
    budget = await service.satellite_link_budget(
        centre_frequency_hz=1.575e9,
        distance_km=35786,
        tx_eirp_dbw=45.0,
        rx_gain_dbi=32.0,
    )
    dp_report = await service.differential_privacy_analytics(epsilon=0.5)
    return triage, budget, dp_report

asyncio.run(example())
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`,
`source_authority_mismatch`, `minimization_reference_required`, or
`bytewax_event_stream_required`.

## Key Service Methods

### Sync CRUD (original)
| Method | Purpose |
|--------|---------|
| `record_authority()` | Register a lawful collection authority |
| `register_source()` | Register a signal source tied to an authority |
| `record_collection_task()` | Create a governed collection task |
| `record_observation()` | Record a raw signal observation |
| `record_processing_batch()` | Record a processing run over an observation |
| `record_pattern()` | Record a detected signal pattern |
| `record_assessment()` | Record an intelligence assessment |
| `record_review()` | Record an analyst review |
| `register_sigint_agent()` | Register an AI agent for SIGINT tasks |
| `validate_agent_action()` | Gate a privileged agent action |
| `validate_batch()` | Validate a Bytewax batch submission |
| `dashboard_summary()` | Tenant-scoped dashboard counts |

### Core Async Operations
| Method | Purpose |
|--------|---------|
| `collect_signal()` | Collect a raw signal with band detection |
| `intercept_communication()` | Start a lawful communication intercept |
| `decrypt_signal()` | Attempt signal decryption (rot13, xor, base64, aes256_ecb_sim) |
| `traffic_analysis()` | Metadata-level traffic analysis between endpoints |
| `signal_correlation()` | Correlate multiple signals for common emitter |
| `emitter_identification()` | Identify emitter type from RF characteristics |
| `direction_finding()` | TDOA/AOA bearing estimation from sensor array |
| `satellite_intercept()` | Task a satellite intercept by orbit and band |
| `communication_pattern_analysis()` | Hourly activity entropy and peak detection |
| `signal_intelligence_report()` | Generate a classified SIGINT report |
| `frequency_scan()` | 10-step frequency scan with signal collection |
| `spectrum_sweep()` | Variable-step sweep across a frequency range |
| `bulk_collect_signals()` | Concurrent bulk signal ingestion |

### New Advanced Methods (v1.2)
| Method | Purpose |
|--------|---------|
| `emitter_fingerprint()` | Compute RF emitter fingerprint from imperfection metrics |
| `emitter_reidentify()` | Match RF features against stored fingerprints |
| `elint_pdw_extract()` | Extract PDWs and classify radar emitter category |
| `task_from_natural_language()` | Parse free-text tasking into structured task record |
| `signal_triage()` | Categorise signals as noise/cleartext/compressed/encrypted |
| `satellite_link_budget()` | Full link budget with feasibility verdict |
| `differential_privacy_analytics()` | Laplace-noise DP analytics export |
| `spectrum_anomaly_detect()` | Rolling-baseline anomaly detection per band |
| `cross_band_correlate()` | Correlate signal activity across two bands |
| `intercept_schedule()` | Schedule intercepts across multiple channels |
| `signal_quality()` | SNR-proxy quality assessment for a signal |
| `signal_triage()` | Entropy-based routing to decryption/pattern queues |
| `emitter_geolocate()` | Geolocate an emitter via direction finding |

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`
- Improvements roadmap: `WORLD_CLASS_IMPROVEMENTS.md`

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/sigint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/sigint/app.py
./.venv/bin/apg capabilities inspect intel_sigint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/sigint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/sigint --json
```

## Production Boundaries

Live receivers, lawful-intercept gateways, telecom systems, satellite feeds,
decryptors, speech processing, direction finding, storage backends, search
indexes, GraphRAG projections, dissemination delivery, and durable Bytewax
topology execution stay behind adapters.

The `task_from_natural_language()` method uses a rule-based parser stub; swap
`_parse_tasking_instruction` for a call to a locally-hosted Ollama model
(Mistral-7B or Llama-3.1-8B with constrained grammar sampling) in production.

## Copyright

© 2025 Datacraft — www.datacraft.co.ke
