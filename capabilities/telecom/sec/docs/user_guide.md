# Telecom Security — User Guide

**Capability ID**: `telecom_sec` | **Domain**: `telecom` | **Version**: `1.1.0`

## Description

Comprehensive telecom security management: fraud detection (WANGIRI, IRSF, SIM swap, OTT bypass), SS7/Diameter signalling security, roaming abuse detection, VoIP fraud analysis, lawful intercept lifecycle management, security incident response, and threat intelligence sharing. All writes are policy-gated with immutable audit trails. Warrant and evidence requirements are hard constraints — not configurable.

## Installation

```bash
pip install apg-telecom-sec
```

## Quick Start

```python
from apg_telecom_sec.service import TelecomSecurityService

svc = TelecomSecurityService()

# Raise a fraud case
case = svc.raise_fraud_case(
    case_id="FC-001",
    tenant_id="acme",
    fraud_type="irsf",
    msisdn="+254700000001",
    confidence_score=0.88,
    evidence_reference="cdr://store/2026-06-11/FC-001",
    detected_at="2026-06-11T08:00:00Z",
)

# Compute multi-signal fraud risk score
import asyncio
risk = asyncio.run(svc.evaluate_fraud_risk_score(
    msisdn="+254700000001",
    features={"calls_per_hour": 80, "geo_anomaly_score": 0.6, "recent_fraud_flag": True},
    tenant_id="acme",
))
print(risk["recommended_action"])  # "block"
```

## Provides

| Workflow | Description |
|----------|-------------|
| `fraud_management_workflow` | Real-time fraud detection, scoring, and blocking |
| `ss7_security_workflow` | SS7 protocol attack detection and mitigation |
| `diameter_security_workflow` | Diameter protocol attack detection |
| `lawful_intercept_workflow` | Warrant-gated lawful intercept management |
| `security_incident_workflow` | Incident lifecycle and regulatory reporting |
| `threat_intel_workflow` | IOC management with TLP enforcement |
| `voip_fraud_detection_workflow` | VoIP-specific fraud pattern detection |
| `roaming_security_workflow` | Roaming abuse and bypass fraud detection |
| `sec_agent_workflow` | Security automation agent management |

## Requires

| Capability | Reason |
|------------|--------|
| `auth` | Authentication and privilege checks |
| `audl` | Immutable audit trail for all security actions |
| `mten` | Tenant context enforcement |
| `conf` | Security policy configuration |
| `ntfy` | Security incident and fraud alerts |
| `wflo` | Incident and intercept workflow states |
| `moni` | Real-time security monitoring |
| `mqeb` | Event streaming via Bytewax |
| `comp` | Regulatory compliance and lawful intercept |

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-sec/dashboard` | `telecom_sec:view` | Overview |
| `/telecom-sec/fraud` | `telecom_sec:fraud` | Fraud |
| `/telecom-sec/fraud-rules` | `telecom_sec:fraud_rules` | Fraud |
| `/telecom-sec/ss7` | `telecom_sec:ss7` | Signalling Security |
| `/telecom-sec/diameter` | `telecom_sec:diameter` | Signalling Security |
| `/telecom-sec/intercept` | `telecom_sec:intercept` | Legal |
| `/telecom-sec/incidents` | `telecom_sec:incidents` | Incidents |
| `/telecom-sec/threat-intel` | `telecom_sec:threat_intel` | Intelligence |

## Core Service Methods

### Fraud Management

```python
# Raise a fraud case
svc.raise_fraud_case(case_id, tenant_id, fraud_type, msisdn, confidence_score,
                     evidence_reference, detected_at)

# Block a confirmed fraud case (evidence mandatory)
svc.apply_fraud_block(case_id, tenant_id, evidence_reference)

# Multi-signal risk scoring
await svc.evaluate_fraud_risk_score(msisdn, features, tenant_id)
# features: calls_per_hour, sms_per_hour, geo_anomaly_score, is_roaming,
#           recent_fraud_flag, account_age_days
# Returns: risk_score (0–1), contributing_signals, recommended_action, auto_block
```

**Supported fraud types**: `irsf`, `wangiri`, `pbx_hacking`, `sim_swap`, `subscription_fraud`, `roaming_fraud`, `premium_rate_abuse`, `bypass_fraud`, `cli_spoofing`, `account_takeover`

### Signalling Security (SS7 / Diameter)

```python
# Record SS7 attack
svc.record_ss7_attack(attack_id, tenant_id, attack_type, source_reference,
                      target_reference, evidence_reference, detected_at)

# Async SS7 detection from raw signaling event
await svc.ss7_attack_detection(signaling_event, tenant_id)
# signaling_event keys: message_type, source_gt, destination_gt, opcode, imsi

# Diameter fraud detection
await svc.diameter_fraud_detection(request, tenant_id)
# request keys: command_code, origin_realm, destination_realm, avp_list

# Correlate attacks into campaigns
await svc.correlate_signaling_attacks(tenant_id, window_minutes=60)
# Returns: cluster_count, coordinated_campaign_detected, suggested_severity, clusters
```

**SS7 attack types**: `location_tracking`, `location_hijacking`, `subscriber_data_manipulation`, `subscriber_info_disclosure`, `call_interception`, `sms_interception`

**Diameter attack types**: `realm_spoofing`, `roaming_fraud_ulr`, `ulr_flooding`, `hss_enumeration`

### VoIP Fraud Detection

```python
# Pattern-based VoIP fraud detection (IRSF, PBX hacking, Wangiri)
await svc.voip_fraud_detection(cdr, tenant_id)
# cdr keys: cdr_id, destination, duration_secs, call_count_last_hour, calling_number

# Simpler interface for single call analysis
await svc.detect_voip_fraud(call_id, calling_number, called_number,
                             duration_seconds, cost, tenant_id)
```

### SIM Swap Detection

```python
# Risk-scored SIM swap assessment
await svc.sim_swap_detection(customer_id, event, tenant_id)
# event keys: recent_password_reset, geographic_anomaly,
#             multiple_swaps_30d, high_value_transaction_after
# Returns: risk_score, verdict ("normal"/"low_risk"/"suspicious"), risk_factors

# Simpler swap evaluation
await svc.detect_sim_swap(msisdn, old_iccid, new_iccid, swap_channel,
                           tenant_id, kyc_verified)
```

### OTT Bypass / SIM Box Detection

```python
await svc.ott_bypass_detection(traffic_pattern, tenant_id)
# traffic_pattern keys: call_duration_variance, destination_entropy,
#                       flash_call_ratio, cli_spoofing_indicators
# Detected when score >= 0.5 across 4 indicator dimensions
```

### Lawful Intercept

Lawful intercept requires both `warrant_reference` and `regulatory_authority` — these are hard requirements enforced by the policy engine and cannot be bypassed.

```python
# Activate an intercept
svc.activate_intercept(intercept_id, tenant_id, intercept_type, target_msisdn,
                        warrant_reference, regulatory_authority, activated_at, expires_at)

# Register an LI order (higher-level helper)
await svc.lawful_intercept_order(target_id, authority_ref, scope, expiry, tenant_id)

# Update intercept status
svc.update_intercept_status(intercept_id, tenant_id, new_status)

# Audit expiry — marks expired, warns pre-expiry
await svc.manage_intercept_expiry(tenant_id, warn_days_before=7)
# Returns: expired_intercepts, expiring_soon (with days_until_expiry)
```

**Intercept types**: `call_content`, `sms_content`, `data_content`, `metadata_only`, `combined`

**Intercept statuses**: `active`, `suspended`, `expired`, `terminated`

### Security Incidents

```python
# Open an incident
svc.open_incident(incident_id, tenant_id, incident_type, severity, description,
                   evidence_reference, opened_at)

# Update status / resolve
svc.update_incident_status(incident_id, tenant_id, new_status, resolved_at=None)

# Execute a response action
await svc.security_incident_response(incident_id, action, tenant_id, performed_by, notes)
# actions: contain, eradicate, recover, close, escalate
```

**Severity levels**: `low`, `medium`, `high`, `critical`

**Incident types**: `fraud`, `intrusion`, `data_breach`, `signalling_attack`, `service_disruption`, `policy_violation`

### Threat Intelligence

```python
# Record an IOC
svc.record_threat_intel(intel_id, tenant_id, source, ioc_type, ioc_value,
                         tlp_level, valid_from, valid_to, shared)

# Cross-reference IOC against all internal signals
await svc.enrich_threat_intel_ioc(ioc_value, ioc_type, tenant_id)
# Returns: correlation_count, correlations (fraud/SS7/Diameter/intrusion),
#          enriched_confidence, recommended_tlp

# Bulk block threats
await svc.bulk_block_threats(threat_ids, tenant_id)
```

**TLP levels**: `white`, `green`, `amber`, `red`

**IOC types**: `ipv4`, `ipv6`, `domain`, `url`, `msisdn`, `imsi`, `realm`, `hash`

### Analytics & Posture

```python
# Unified security posture score (0–100, graded A–F)
await svc.generate_security_posture_score(tenant_id)
# Dimensions: incident_management (0–25), signaling_security (0–25),
#             fraud_control (0–25), compliance (0–25)

# Per-jurisdiction compliance matrix
await svc.multi_jurisdiction_compliance_matrix(tenant_id, jurisdictions=["KE","TZ","EU"])
# Per-jurisdiction: retention check, warrant completeness, audit trail presence

# Data retention compliance check
await svc.data_retention_compliance(jurisdiction, retention_days, tenant_id)

# Security KPI analytics
await svc.security_incident_analytics(tenant_id, period="monthly")

# Threat intel summary
await svc.threat_intel_analytics(tenant_id, period="weekly")

# Dashboard summary
svc.dashboard_summary(tenant_id)
```

### Subscriber Anomaly Detection

```python
await svc.subscriber_anomaly_detection(
    msisdn="+254700000001",
    current_metrics={"call_volume": 450, "data_usage_mb": 8000, "sms_count": 5, ...},
    baseline_metrics={
        "call_volume": {"mean": 50, "std": 15},
        "data_usage_mb": {"mean": 500, "std": 100},
        ...
    },
    tenant_id="acme",
)
# Metrics: call_volume, data_usage_mb, sms_count, roaming_duration_min,
#          international_call_ratio
# Flags dimensions with |z| > 3.0; returns severity: none/medium/high/critical
```

### Red Team Testing

```python
# Replay known GSMA attack scenarios against live detection logic
result = await svc.run_red_team_scenario(
    scenario_name="gsma_sri_tracking",
    tenant_id="acme",
    dry_run=True,
)
print(result["passed"])  # True if detection worked correctly
```

**Built-in scenarios**: `gsma_sri_tracking`, `ss7_location_hijack`, `diameter_realm_spoof`, `irsf_highrate`, `wangiri_burst`, `sim_swap_chain`

### Network Security

```python
# Roaming partner security assessment
await svc.roaming_security_check(roaming_partner_id, tenant_id)
# Returns: risk_rating (low/medium/high), ss7_attack_count, diameter_attack_count

# Network intrusion detection from traffic events
await svc.network_intrusion_detection(traffic_event, tenant_id)
# traffic_event keys: source_ip, dst_port, protocol, byte_count, packet_count, tcp_flags
# Detects: port_scan, ddos_indicator, syn_flood, protocol_anomaly, ioc_match
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed `TELECOM_SEC_`.

| Key | Default | Description |
|-----|---------|-------------|
| `fraud.block_threshold` | `0.85` | Confidence threshold for auto-block |
| `lawful_intercept.warrant_required` | `true` | Cannot be overridden |
| `lawful_intercept.regulatory_authority_required` | `true` | Cannot be overridden |
| `governance.evidence_fabrication_denied` | `true` | Hard block — not configurable |
| `intercept.warn_days_before_expiry` | `7` | Pre-expiry notification window |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `tenant_context_required` | `tenant_context_present=False` | deny |
| `intercept_without_warrant_denied` | `warrant_present=False` | deny |
| `intercept_regulatory_authority_required` | `authority_present=False` | deny |
| `fraud_block_requires_evidence` | `evidence_present=False` | deny |
| `evidence_fabrication_denied` | agent fabrication scope | deny |
| `cross_tenant_access_denied` | cross-tenant agent scope | deny |
| `sec_batch_requires_bytewax` | non-bytewax stream | deny |

## Streaming Events

All write operations emit events to `apg.telecom.sec.lifecycle` via Bytewax:

`fraud_case_raised` · `fraud_block_applied` · `ss7_attack_detected` · `diameter_attack_detected` · `intercept_activated` · `intercept_expired` · `security_incident_opened` · `security_incident_resolved` · `threat_ioc_shared` · `threat_ioc_enriched` · `voip_fraud_detected` · `sim_swap_suspicious` · `ott_bypass_detected` · `network_intrusion_detected` · `red_team_scenario_run` · `sec_agent_registered`

## Composability

```apg
use telecom_sec;
```

- Feeds fraud signals → `telecom_ana` (fraud analytics)
- Consumes network events ← `telecom_net` (SS7/Diameter correlation)
- Lawful intercept → `comp` (regulatory reporting chains)
- Threat intel → `grph` (adversary network mapping)
- Posture scores → `moni` (security monitoring dashboards)
- Anomaly events → `ntfy` (subscriber alert notifications)

## Further Reading

- `service.py` — Business logic implementation (all service methods)
- `models.py` — Data models (SecFraudCase, SecIncident, etc.)
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Policy rules and capability contract
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised enhancement proposals
- `SPECIFICATION.md` — Full capability specification
