# APG Cyber Intelligence (intel_cybint)

**Capability ID**: `intel_cybint` | **Domain**: `intel` | **Version**: `1.2.0`

`intel_cybint` is the APG package-backed capability for governed defensive cyber-intelligence.
It composes authorities, indicators, sightings, enrichment, threat profiles, risk assessments,
incident links, dissemination, reviews, MITRE ATT&CK mapping, STIX 2.1 export, lifecycle
management, NATS-based streaming, and AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and evidence.
- Indicator registry with type, value, TLP, confidence, authority, evidence, and lifecycle state.
- Sighting, enrichment, profile, risk, incident-link, dissemination, and review workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.cybint.lifecycle`.
- MITRE ATT&CK Navigator layer generation from ThreatProfiles.
- STIX 2.1 bundle export for TAXII-compatible sharing.
- Confidence decay engine with configurable half-life.
- Indicator lifecycle state machine (ACTIVE → UNDER_REVIEW → RETIRED → ARCHIVED).
- Lockheed Martin Kill Chain stage classifier per indicator.
- Automated threat intelligence brief generation.
- Indicator deduplication via content-addressed canonical keys.
- Behavioural baseline computation with 3σ anomaly detection.

## Quick Start

```python
from capabilities.intel.cybint import CyberIntelligenceService

svc = CyberIntelligenceService("tenant-a")

# Authority + indicator (sync)
authority = svc.record_authority(
    "auth-1", "tenant-a", "defensive_operations_authority",
    "scope://defensive-threat-intel", "confidential",
    "approver-1", "2027-12-31", "evidence://authority",
)
indicator = svc.record_indicator(
    "ioc-1", "tenant-a", "domain", "evil.example.invalid",
    "amber", 0.92, authority["id"], "evidence://ioc",
)

# Async operations
import asyncio

async def main():
    # Attack surface scan
    surface = await svc.scan_attack_surface("datacraft.co.ke")

    # MITRE ATT&CK Navigator layer
    profile = svc.record_profile(
        "prof-1", "tenant-a", "apt_group", "Lazarus",
        "confidential", 0.85, "analyst-1", "evidence://profile",
    )
    layer = await svc.map_to_attack_navigator("prof-1")

    # STIX 2.1 bundle
    bundle = await svc.export_stix_bundle(tlp_filter="amber")

    # Confidence decay
    decay = await svc.apply_confidence_decay(half_life_days=60)

    # Kill chain classification
    kc = await svc.classify_kill_chain_stage("ioc-1")

    # Threat brief
    brief = await svc.generate_threat_brief("confidential", period_days=7)

    # Behavioural baseline
    baseline = await svc.compute_behavioural_baseline(
        "host-web-01", [12.0, 14.0, 11.0, 87.0, 13.0, 10.0]
    )

asyncio.run(main())
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`, `confidence_score_invalid`,
`offensive_or_exploit_scope_denied`, or `bytewax_event_stream_required`.

## Service Methods

### Sync (CRUD + Governance)

| Method | Description |
|--------|-------------|
| `describe(tenant_id)` | Return capability contract dict |
| `evaluate(context)` | Evaluate capability rules |
| `record_authority(...)` | Record a lawful cyber authority |
| `record_indicator(...)` | Register an indicator of compromise |
| `record_sighting(...)` | Record an indicator sighting |
| `record_enrichment(...)` | Attach enrichment data to an indicator |
| `record_profile(...)` | Create or update a threat actor profile |
| `record_risk(...)` | Record a cyber risk assessment |
| `record_incident_link(...)` | Link a risk assessment to an incident |
| `record_dissemination(...)` | Record an intelligence dissemination |
| `record_review(...)` | Record an analyst review |
| `register_cybint_agent(...)` | Register an AI agent for CYBINT tasks |
| `validate_agent_action(...)` | Validate an agent action against policy |
| `validate_batch(...)` | Validate a batch operation |
| `dashboard_summary(tenant_id)` | Aggregate dashboard counts |

### Async (Operational + Advanced)

| Method | Description |
|--------|-------------|
| `scan_attack_surface(domain)` | Enumerate external attack surface |
| `vulnerability_discovery(ip_range, scan_type)` | Discover CVEs across a network range |
| `dark_web_monitoring(keywords, onion_sites)` | Search dark web for keyword hits |
| `malware_analysis(hash, metadata)` | Classify malware by static features |
| `network_traffic_analysis(pcap_data)` | Detect scan/beaconing patterns in traffic |
| `intrusion_detection(events)` | Alert on suspicious port access and port scans |
| `threat_actor_attribution(iocs)` | Attribute IOCs to known threat actors |
| `honeypot_alert(alert_data)` | Classify inbound honeypot interactions |
| `zero_day_tracking(vuln_id)` | Track a zero-day through its lifecycle |
| `cybint_report(classification)` | Generate a full CYBINT intelligence report |
| `ioc_bulk_ingest(ioc_list)` | Bulk-ingest up to 5,000 IOCs |
| `threat_intelligence_sharing(ids, recipients, tlp)` | Share indicators with partners |
| `vulnerability_prioritisation(scan_ids)` | Rank CVEs by severity and exploitability |
| `phishing_campaign_detection(email_headers)` | Detect phishing campaign patterns |
| `lateral_movement_detection(auth_logs)` | Detect lateral movement in auth logs |
| `threat_hunt(hypothesis, data_sources)` | Execute a hypothesis-driven threat hunt |
| `osint_enrichment(indicator_value)` | Enrich an IOC with OSINT feed data |
| `security_posture_assessment()` | Compute tenant security posture score |
| `export_indicators(fmt, tlp_filter)` | Export indicators to JSON/CSV/STIX |
| `health_check()` | Service health and metrics |
| `cyber_analytics()` | Aggregate analytics across all domains |
| `compliance_check(framework)` | Check compliance against NIST/ISO/SOC2/MITRE |
| `incident_response_trigger(indicator_id, playbook)` | Trigger an IR playbook |
| `supply_chain_risk_assessment(vendor_id)` | Assess third-party vendor cyber risk |
| `bulk_sighting_update(updates)` | Bulk-update sighting severity/confidence |
| **`map_to_attack_navigator(profile_id)`** | Generate MITRE ATT&CK Navigator layer |
| **`apply_confidence_decay(half_life_days)`** | Apply exponential confidence decay |
| **`classify_kill_chain_stage(indicator_id)`** | Assign Kill Chain stages to indicator |
| **`transition_indicator_lifecycle(id, state, reviewer)`** | Advance indicator lifecycle |
| **`generate_threat_brief(classification, days)`** | Automated threat intelligence brief |
| **`deduplicate_indicators()`** | Content-addressed IOC deduplication |
| **`export_stix_bundle(tlp_filter)`** | STIX 2.1 bundle export |
| **`compute_behavioural_baseline(entity_id, series)`** | 3σ anomaly detection baseline |

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Streaming (NATS + bytewax)

The capability emits lifecycle events to NATS subjects:

| Subject | When |
|---------|------|
| `apg.intel.cybint.lifecycle` | All CRUD and audit events |
| `apg.intel.cybint.iocs.inbound` | IOC ingestion subscription |
| `apg.intel.cybint.iocs.acked` | Ingestion confirmation |
| `apg.intel.cybint.playbooks.dispatch` | IR playbook dispatch |
| `apg.intel.cybint.playbooks.completed` | Playbook completion |

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/cybint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/cybint/app.py
./.venv/bin/apg capabilities inspect intel_cybint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/cybint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/cybint --json
```

## Production Boundaries

Exploit development, payload generation, intrusion tooling, vulnerability exploitation,
credential collection, command-and-control, live SIEM/EDR/SOAR integrations, malware
sandboxes, vulnerability scanners, ticketing systems, asset inventories, blocklist
deployment, containment execution, storage backends, GraphRAG projections,
dissemination delivery, and durable bytewax topology execution stay behind adapters.

## Improvements Roadmap

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 prioritised improvements targeting parity with
commercial platforms (Recorded Future, CrowdStrike, Mandiant, Palo Alto).
