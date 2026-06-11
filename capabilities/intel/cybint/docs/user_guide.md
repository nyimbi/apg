# Cyber Intelligence — User Guide

**Capability**: `intel_cybint` | **Version**: `1.2.0` | **Domain**: `intel`

© 2025 Datacraft (www.datacraft.co.ke)

---

## Overview

`intel_cybint` delivers a governed, tenant-scoped defensive cyber-intelligence runtime inside
APG-generated applications. It covers the full intelligence lifecycle from raw IOC ingestion
through threat actor attribution, kill-chain classification, MITRE ATT&CK mapping, STIX 2.1
export, and automated threat briefing — all with audit trails, TLP enforcement, and policy
guardrails.

This guide is organised in workflow order: install → configure → operate → share → review.

---

## Installation

```bash
pip install apg-intel-cybint
```

Or via the APG workspace:

```bash
uv add apg-intel-cybint
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Authority** | A lawful mandate (e.g. defensive_operations_authority) that gates indicator recording. |
| **Indicator** | An IOC with type, value, TLP, confidence score, and lifecycle state. |
| **Sighting** | Evidence of an indicator observed in the environment. |
| **Enrichment** | Contextual data attached to an indicator (GeoIP, ASN, malware family, etc.). |
| **Threat Profile** | A named threat actor or campaign entity with classification and confidence. |
| **Risk Assessment** | A scored linkage between an indicator and a threat profile. |
| **Incident Link** | Binding a risk assessment to an active incident with response priority. |
| **Dissemination** | A controlled release of intelligence to a defined audience under TLP marking. |
| **Lifecycle State** | ACTIVE → UNDER_REVIEW → RETIRED → ARCHIVED (governed state machine). |
| **STIX Bundle** | STIX 2.1 export of indicators, threat profiles, and sightings for TAXII sharing. |

---

## Service Instantiation

```python
from capabilities.intel.cybint import CyberIntelligenceService

# Minimal
svc = CyberIntelligenceService("tenant-a")

# Full collaborator injection
svc = CyberIntelligenceService(
    tenant_id="tenant-a",
    actor_id="analyst-007",
    auth=my_auth_adapter,
    audit=my_audit_adapter,
    notify=my_notify_adapter,
    db_url="postgresql+asyncpg://user:pass@localhost/cybint",
    store=my_store_adapter,
)
```

---

## Governance Workflow

### 1. Record a Lawful Authority

Every indicator requires a pre-existing authority in the same tenant.

```python
authority = svc.record_authority(
    authority_id="auth-ops-2025",
    tenant_id="tenant-a",
    authority_type="defensive_operations_authority",
    scope_reference="scope://threat-intel-defensive",
    classification="confidential",
    approver_id="ciso-001",
    expires_at="2027-01-01",
    evidence_reference="policy://cybint-authority-2025",
)
print(authority["id"])  # "auth-ops-2025"
```

Supported `authority_type` values: see `SUPPORTED_AUTHORITY_TYPES` in `capability_contract.py`.

### 2. Register an Indicator

```python
indicator = svc.record_indicator(
    indicator_id="ioc-evil-domain",
    tenant_id="tenant-a",
    indicator_type="domain",
    indicator_value="malicious-c2.invalid",
    tlp="amber",
    confidence_score=0.88,
    authority_id="auth-ops-2025",
    evidence_reference="feed://abuse-ch/2025-06-01",
)
```

Supported `indicator_type`: `ip`, `domain`, `url`, `hash`, `email`, `cve`, `yara`, `registry`, `mutex`, `asn`.
Supported `tlp`: `white`, `green`, `amber`, `red`.

---

## Bulk Operations

### Bulk IOC Ingest

```python
import asyncio

ioc_list = [
    {"indicator_type": "ip", "indicator_value": "198.51.100.42", "tlp": "amber",
     "confidence_score": 0.75, "evidence_reference": "feed://et/2025-06"},
    {"indicator_type": "domain", "indicator_value": "bad-actor.example", "tlp": "amber",
     "confidence_score": 0.90, "evidence_reference": "feed://otx/2025-06"},
]

result = asyncio.run(svc.ioc_bulk_ingest(ioc_list))
print(result["succeeded"], "/", result["submitted"])  # 2 / 2
```

Cap: 5,000 IOCs per call. Failures are reported per-index, successes continue.

---

## Threat Intelligence Operations

### Attack Surface Scan

```python
surface = asyncio.run(svc.scan_attack_surface("datacraft.co.ke"))
# Keys: scan_id, dns_records, exposed_services, web_technologies,
#       subdomain_count_estimate, certificate_transparency_entries
```

### Vulnerability Discovery

```python
vulns = asyncio.run(svc.vulnerability_discovery("10.0.0.0/24", "version"))
# Keys: scan_id, hosts_scanned, hosts_responsive, vulnerabilities, critical_count
```

Scan types: `syn_stealth`, `full_connect`, `udp`, `version`, `script`.

### Malware Analysis

```python
analysis = asyncio.run(svc.malware_analysis(
    "d41d8cd98f00b204e9800998ecf8427e",
    {"file_type": "exe", "size_bytes": 123456, "entropy": 7.8,
     "import_count": 250, "section_count": 8},
))
# Keys: malware_family, confidence, mitre_techniques, ioc_hashes
```

### Network Traffic Analysis

```python
result = asyncio.run(svc.network_traffic_analysis({
    "packet_count": 100000,
    "bytes_total": 5_000_000,
    "proto_counts": {"TCP": 80000, "UDP": 20000},
    "src_ips": ["10.0.0.1", "10.0.0.2"],
    "dst_ips": [f"192.168.1.{i}" for i in range(200)],
    "duration_s": 300.0,
}))
# Keys: scan_suspected, beaconing_suspected, throughput_bps, protocol_distribution
```

### Intrusion Detection

```python
events = [
    {"src_ip": "10.1.1.100", "dst_ip": "10.0.0.1", "dst_port": 22,
     "proto": "TCP", "bytes": 512, "timestamp": "2025-06-01T10:00:00Z"},
    # ... more events
]
detection = asyncio.run(svc.intrusion_detection(events))
# Keys: alerts, alert_count, high_severity, protocol_byte_totals
```

Cap: 10,000 events per call.

### Threat Actor Attribution

```python
result = asyncio.run(svc.threat_actor_attribution([
    "198.51.100.42", "malicious-c2.invalid", "CVE-2021-44228",
    "d41d8cd98f00b204e9800998ecf8427e",
]))
# Keys: actor_scores (dict), top_actor, top_confidence
```

### Dark Web Monitoring

```python
result = asyncio.run(svc.dark_web_monitoring(
    keywords=["datacraft", "breach", "credentials"],
    onion_sites=["marketplace1.onion", "forums2.onion"],
))
# Keys: hits (list of {site, matched_keywords, hit_count}), risk_score
```

### Honeypot Alert Processing

```python
result = asyncio.run(svc.honeypot_alert({
    "src_ip": "203.0.113.55",
    "dst_port": 3389,
    "payload_hex": "deadbeef" * 20,
    "protocol": "TCP",
    "timestamp": "2025-06-01T08:30:00Z",
}))
# Keys: interaction_type, origin_country_estimate, payload_fingerprint
```

### Zero-Day Tracking

```python
# Each call advances the stage: DISCOVERED → CONFIRMED → PATCH_AVAILABLE → MITIGATED → CLOSED
result = asyncio.run(svc.zero_day_tracking("CVE-2024-99999"))
# Keys: cvss_score, severity, stage, in_the_wild_exploitation, patch_available
```

---

## Advanced Intelligence Methods

### MITRE ATT&CK Navigator Layer

```python
# First create a threat profile
svc.record_profile(
    "prof-apt29", "tenant-a", "apt_group", "APT29",
    "confidential", 0.91, "analyst-1", "evidence://apt29",
)

layer = asyncio.run(svc.map_to_attack_navigator("prof-apt29"))
# layer["layer"] is Navigator-compatible JSON
# POST to /api/v1/attack-navigator/layer or serve via blueprint
```

### Kill Chain Stage Classifier

```python
result = asyncio.run(svc.classify_kill_chain_stage("ioc-evil-domain"))
# Keys: kill_chain_stages, stage_confidences, primary_stage
# Example: {"kill_chain_stages": ["RECONNAISSANCE", "COMMAND_AND_CONTROL"],
#            "primary_stage": "COMMAND_AND_CONTROL"}
```

### Indicator Lifecycle Management

```python
# Move to under review
asyncio.run(svc.transition_indicator_lifecycle(
    "ioc-evil-domain", "UNDER_REVIEW", "reviewer-001"
))

# Retire (sets confidence to 0.0)
asyncio.run(svc.transition_indicator_lifecycle(
    "ioc-evil-domain", "RETIRED", "reviewer-001"
))
```

State machine: `ACTIVE → UNDER_REVIEW → {ACTIVE | RETIRED} → ARCHIVED`

### Confidence Decay

```python
# Halve confidence every 60 days; retire indicators below 0.10
result = asyncio.run(svc.apply_confidence_decay(half_life_days=60))
print(result["retirement_eligible"])  # count of indicators below threshold
```

### Threat Hunt

```python
result = asyncio.run(svc.threat_hunt(
    hypothesis="APT29 lateral movement via compromised VPN credentials",
    data_sources=["endpoint_logs", "auth_logs", "dns_logs", "proxy_logs"],
))
# Keys: findings, verdict (THREAT_CONFIRMED | INCONCLUSIVE | NO_EVIDENCE)
```

### OSINT Enrichment

```python
result = asyncio.run(svc.osint_enrichment("198.51.100.42"))
# Keys: geo_country, asn, reputation_score, known_malicious, threat_feeds_matched
```

### Phishing Campaign Detection

```python
headers = [
    {"from": "support@evil.example", "subject": "Urgent: verify your account",
     "sender_ip": "203.0.113.1", "timestamp": "2025-06-01T06:00:00Z"},
    # ... more headers
]
result = asyncio.run(svc.phishing_campaign_detection(headers))
# Keys: campaign_detected, campaign_confidence, campaign_indicators
```

### Lateral Movement Detection

```python
logs = [
    {"user": "jdoe", "src_host": "ws-101", "dst_host": "srv-db-01",
     "timestamp": "2025-06-01T09:00:00Z", "auth_type": "NTLM"},
    # ... more entries
]
result = asyncio.run(svc.lateral_movement_detection(logs))
# Keys: lateral_movement_suspected, alerts
```

### Deduplication

```python
result = asyncio.run(svc.deduplicate_indicators())
# Keys: indicators_kept, indicators_removed, removed_ids
```

### STIX 2.1 Bundle Export

```python
bundle = asyncio.run(svc.export_stix_bundle(tlp_filter="amber"))
import json
print(json.dumps(bundle["bundle"], indent=2))
# Standard STIX 2.1 bundle, ready for TAXII push or sharing with ISACs
```

### Behavioural Baseline

```python
result = asyncio.run(svc.compute_behavioural_baseline(
    entity_id="user-jdoe",
    metric_series=[5.0, 6.0, 5.5, 4.8, 97.0, 6.0, 5.2],  # login attempts/hour
))
# anomalies[0] will flag index=4 (value=97.0) as 3σ outlier
```

---

## Reporting and Sharing

### Automated Threat Brief

```python
brief = asyncio.run(svc.generate_threat_brief("confidential", period_days=7))
# Keys: executive_summary, key_findings, recommended_actions, tlp_marking
print(brief["executive_summary"]["overall_risk_level"])
```

### Intelligence Sharing (TLP-governed)

```python
result = asyncio.run(svc.threat_intelligence_sharing(
    indicator_ids=["ioc-evil-domain", "ioc-ip-42"],
    recipients=["partner-isac", "sector-cert"],
    tlp="amber",
))
# Keys: share_records (list), indicator_count, recipient_count
```

### CYBINT Report

```python
report = asyncio.run(svc.cybint_report("confidential"))
# Keys: report_id, summary (full aggregate across all operational methods)
```

---

## Compliance and Posture

### Compliance Check

```python
result = asyncio.run(svc.compliance_check("NIST_CSF"))
# Supported frameworks: NIST_CSF, ISO27001, SOC2, MITRE_ATTCK
# Keys: controls_total, controls_covered, coverage_pct, compliance_status, gaps
```

### Security Posture Assessment

```python
posture = asyncio.run(svc.security_posture_assessment())
# Keys: posture_score (0–100), posture_rating (EXCELLENT/GOOD/FAIR/POOR)
print(posture["posture_rating"])
```

### Supply Chain Risk

```python
risk = asyncio.run(svc.supply_chain_risk_assessment("vendor-cloudprovider"))
# Keys: risk_score, risk_tier (HIGH/MEDIUM/LOW), known_breaches,
#       days_since_last_patch, has_security_certifications
```

---

## Incident Response

### Trigger Playbook

```python
response = asyncio.run(svc.incident_response_trigger(
    indicator_id="ioc-evil-domain",
    playbook="BLOCK_IP",
))
# Playbooks: ISOLATE, BLOCK_IP, RESET_CREDENTIALS, ESCALATE, CONTAIN
# Keys: response_id, status ("TRIGGERED"), triggered_by, triggered_at
```

---

## Dashboard and Analytics

```python
summary = svc.dashboard_summary("tenant-a")
analytics = asyncio.run(svc.cyber_analytics())
health = asyncio.run(svc.health_check())
```

---

## Export

```python
# JSON export
export = asyncio.run(svc.export_indicators(fmt="json"))

# TLP-filtered CSV
export = asyncio.run(svc.export_indicators(fmt="csv", tlp_filter="amber"))

# STIX lite
export = asyncio.run(svc.export_indicators(fmt="stix"))
```

---

## Error Handling

| Exception | Cause |
|-----------|-------|
| `PermissionError` | Policy rule denied (tenant_context, authority, TLP, policy) |
| `ValueError` | Invalid enum value (scan_type, framework, playbook, classification) |
| `KeyError` | Referenced entity not found (indicator_id, profile_id) |
| `AssertionError` | Required argument missing or batch cap exceeded |

All enforced errors include reason strings from `capability_contract.py` for structured logging.

---

## Streaming Integration (NATS + bytewax)

The service emits audit events with `"processor": "bytewax"`. Wire the bytewax dataflow to
consume from `apg.intel.cybint.iocs.inbound` and publish to `apg.intel.cybint.iocs.acked`.
Playbook dispatch publishes to `apg.intel.cybint.playbooks.dispatch`.

```python
# Example: NATS audit consumer pattern
async def nats_audit_consumer(msg):
    event = json.loads(msg.data)
    # Route to SIEM, alert manager, or durable store
    await process_audit_event(event)
```

---

## Further Reading

- `service.py` — Full business logic (1,700+ lines)
- `models.py` — In-memory dataclass models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Policy rules and supported value sets
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised capability improvements
- `SPECIFICATION.md` — Formal capability specification
- `PLAN.md` — Implementation plan
