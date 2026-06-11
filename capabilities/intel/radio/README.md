# APG Radio Intelligence (intel_radio)

`intel_radio` is an executable APG capability for lawful, passive
radio-monitoring, SIGINT collection, spectrum analysis, and RF direction
finding. It can be composed into generated APG applications that need
public-safety monitoring, spectrum management, interference review,
emitter identification, direction finding, jamming assessment, and
partner-feed analysis.

## What It Provides

- **Governance** — authority records, band plans, regulatory compliance audit,
  and frequency deconfliction against ITU designations and tenant allocations.
- **Collection** — receiver registration with calibration tracking, collection
  session management, and bulk observation ingest (up to 5 000 records/call).
- **Signal Analysis** — transmission decoding (14 protocols: ADS-B, AIS, ACARS,
  APRS, DMR, P25, TETRA, DSTAR, CW, RTTY, FSK, NOAA APT, SELCAL, Mode-S),
  signal classification, anomaly detection, and pattern analysis.
- **Geolocation** — bearing-only DF (circular mean), TDOA hyperbolic
  geolocation (3+ receivers), Doppler DF for single rotating-antenna platforms,
  and emitter movement tracking.
- **ELINT/COMINT Products** — structured ELINT product generation with NATO-
  compatible classification and dissemination markings, COMINT intelligence
  briefs, and radio order of battle (ORBAT) compilation.
- **Spectrum Management** — frequency scan, spectrum occupancy analysis,
  monitoring schedule (up to 500 frequencies), interference detection with
  mitigation recommendation, jamming assessment, and exclusion zone registry.
- **Cross-Border / Regional** — cross-border transmission monitoring and
  border-region frequency surveillance.
- **Link Analysis** — signal link graph (emitter → session → observation
  adjacency) for `grph` capability integration.
- **AI Agent Guardrails** — deterministic rules enforce tenant context, lawful
  authority, frequency bounds, receiver calibration, evidence requirements,
  approvals, Bytewax lifecycle routing, and strict agent-scope controls (deny
  transmit, jam, spoof, intercept protected comms, decrypt).
- **Reporting** — radio intelligence reports, COMINT briefs, signal pattern
  library, analytics dashboard, and CSV/JSON export.

## Quick Start

```bash
# Run self-test
./.venv/bin/python capabilities/intel/radio/app.py

# Run tests
./.venv/bin/pytest -q capabilities/intel/radio/tests/test_package_contract.py

# Inspect capability contract
./.venv/bin/apg capabilities inspect intel_radio --json
```

## Service Usage

```python
from capabilities.intel.radio import RadioIntelligenceListenerService
import asyncio

svc = RadioIntelligenceListenerService("tenant-a", actor_id="analyst-1")

# Record lawful authority
svc.record_authority(
    "auth-1", "tenant-a", "spectrum_license",
    "license-scope", "confidential", "approver-1",
    "2027-06-30", "evidence-ref",
)

# Register receiver and band plan
svc.record_band_plan(
    "bp-vhf", "tenant-a", "VHF", "VHF Land Mobile",
    30.0, 300.0, "auth-1", "itu-ref",
)
svc.register_receiver(
    "rx-1", "tenant-a", "SDR", "site-nairobi",
    "custodian-1", "auth-1", "cal-2025-01", "hw-evidence",
)

# Async operations
async def run():
    # Frequency scan
    scan = await svc.frequency_scan((136.0, 137.0), "nairobi-site", 60.0)

    # Decode a transmission
    decode = await svc.decode_transmission(scan["scan_id"], "ADS_B")

    # TDOA geolocation with 3 receivers
    fix = await svc.tdoa_geolocation("sig-001", [
        {"lat": -1.286, "lon": 36.817, "tdoa_us": 0.0},
        {"lat": -1.300, "lon": 36.840, "tdoa_us": 12.5},
        {"lat": -1.265, "lon": 36.800, "tdoa_us": 8.3},
    ])

    # Frequency deconfliction before allocation
    deconf = await svc.frequency_deconfliction(145.5, 25.0)

    # ELINT product for a catalogued emitter
    emitter = await svc.identify_emitter({
        "frequency_mhz": 9500.0, "modulation": "PULSE",
        "power_dbm": 60.0, "bandwidth_khz": 5000.0,
        "pulse_width_us": 2.0, "pri_us": 1000.0,
    })
    elint = await svc.elint_product(emitter["emitter_id"], "secret", "NATIONAL_ONLY")

    # Signal anomaly detection
    anomalies = await svc.signal_anomaly_detection("session-1")

    # Exclusion zone
    await svc.register_exclusion_zone(
        "ez-1", "Diplomatic Quarter", 136.0, 140.0,
        "POLYGON((36.8 -1.3, 36.9 -1.3, 36.9 -1.2, 36.8 -1.2, 36.8 -1.3))",
        "Protected diplomatic communications",
    )

    # Signal link graph for grph capability
    graph = await svc.signal_link_graph()

    # Communications pattern analysis
    patterns = await svc.comms_pattern_analysis(["session-1", "session-2"])

    # Jamming assessment
    jam = await svc.jamming_assessment(156.8, 120.0)

    # COMINT brief
    brief = await svc.comms_intelligence_brief("confidential")

asyncio.run(run())
```

## Service Methods Reference

### Sync (CRUD / Governance)

| Method | Description |
|--------|-------------|
| `record_authority()` | Register a lawful collection authority |
| `record_band_plan()` | Define a monitored frequency band |
| `register_receiver()` | Register an SDR or hardware receiver |
| `record_session()` | Open a collection session |
| `record_observation()` | Store a signal observation |
| `record_classification()` | Classify a transmission |
| `record_event()` | Record an assessed event |
| `record_referral()` | Refer an event to another agency |
| `record_dissemination()` | Record product dissemination |
| `record_review()` | Peer-review a record |
| `register_radio_agent()` | Register an AI/automated agent |
| `validate_agent_action()` | Enforce agent-scope guardrails |
| `validate_batch()` | Pre-validate a batch operation |
| `dashboard_summary()` | Tenant operational summary |
| `describe()` | Capability contract metadata |
| `evaluate()` | Evaluate context against rules |

### Async (Operational)

| Method | Description |
|--------|-------------|
| `frequency_scan()` | Scan a frequency range for signals |
| `signal_recording()` | Record IQ data metadata |
| `decode_transmission()` | Decode signal with protocol decoder |
| `identify_emitter()` | Classify emitter from RF parameters |
| `radio_direction_finding()` | Bearing-only DF from multiple receivers |
| `tdoa_geolocation()` | TDOA hyperbolic geolocation (3+ receivers) |
| `doppler_direction_finding()` | Doppler DF from rotating antenna array |
| `frequency_deconfliction()` | Check proposed allocation for conflicts |
| `elint_product()` | Generate structured ELINT product |
| `signal_anomaly_detection()` | Statistical outlier detection in session |
| `register_exclusion_zone()` | Geo-fenced frequency exclusion zone |
| `signal_link_graph()` | Emitter–session–observation graph |
| `comms_pattern_analysis()` | Traffic pattern analysis across sessions |
| `frequency_monitoring_schedule()` | Schedule periodic frequency monitoring |
| `interference_detection()` | Detect and classify interference |
| `spectrum_analysis()` | Spectrum occupancy analysis |
| `cross_border_monitoring()` | Border-region transmission surveillance |
| `jamming_assessment()` | Assess intentional jamming |
| `geo_emitter_tracking()` | Track emitter movement from DF fixes |
| `radio_order_of_battle()` | Compile regional electronic ORBAT |
| `radio_intelligence_report()` | Generate classified RINT report |
| `comms_intelligence_brief()` | Generate COMINT/RINT brief |
| `signal_pattern_library()` | Tenant signal pattern catalogue |
| `signal_classification_batch()` | Batch-classify observations |
| `bulk_observation_ingest()` | Bulk ingest up to 5 000 observations |
| `export_observations()` | Export observations (CSV/JSON) |
| `frequency_compliance_audit()` | Regulatory compliance audit |
| `receiver_calibration_status()` | Receiver calibration health check |
| `radio_analytics()` | Aggregate analytics |
| `health_check()` | Service health and operational metrics |

## Guardrails

The capability is defensive, passive, and compliance-first. The rule engine
denies AI-agent actions requesting: transmission, jamming, spoofing, interference
generation, decryption, or interception of protected communications.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/intel/radio/*.py \
    capabilities/intel/radio/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/radio/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/radio --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/radio --json
```

## Improvement Roadmap

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 prioritised engineering improvements
covering TDOA geolocation, real-time Bytewax streaming, ML-based modulation
recognition, cryptographic audit chains, PostgreSQL persistence, WebSocket
push, Doppler DF, inter-capability bus events, structured ELINT/COMINT
products, frequency deconfliction, SDR HAL, geo-fenced exclusion zones,
signal link analysis, and property-based testing.
