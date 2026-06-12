# Telecom Security

## Overview
Provides comprehensive telecom security management covering fraud detection (WANGIRI, IRSF, SIM swap), SS7/Diameter signalling security, roaming security, VoIP fraud detection, lawful intercept management, security incident response, and threat intelligence sharing. Enforces strict warrant and evidence requirements throughout.

## Capability ID
`telecom_sec`

## Provides
- fraud_management_workflow: Real-time fraud detection, scoring, and blocking
- ss7_security_workflow: SS7 protocol attack detection and mitigation
- diameter_security_workflow: Diameter protocol attack detection
- lawful_intercept_workflow: Warrant-gated lawful intercept management
- security_incident_workflow: Incident lifecycle and regulatory reporting
- threat_intel_workflow: IOC management with TLP enforcement
- voip_fraud_detection_workflow: VoIP-specific fraud pattern detection
- roaming_security_workflow: Roaming abuse and bypass fraud detection
- sec_agent_workflow: Security automation agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication and privilege checks |
| audl | Immutable audit trail for all security actions |
| mten | Tenant context enforcement |
| conf | Security policy configuration |
| ntfy | Security incident and fraud alerts |
| wflo | Incident and intercept workflow states |
| moni | Real-time security monitoring |
| mqeb | Event streaming via bytewax |
| comp | Regulatory compliance and lawful intercept |

## Configuration
| Key | Description |
|-----|-------------|
| fraud.supported_types | 10 fraud pattern types |
| fraud.block_threshold | Confidence threshold for auto-block (0.85) |
| lawful_intercept.warrant_required | Always true — cannot be overridden |
| lawful_intercept.regulatory_authority_required | Always true |
| governance.evidence_fabrication_denied | Hard block on evidence tampering |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-sec/fraud | GET/POST | Fraud case management | telecom_sec:fraud |
| /telecom-sec/ss7 | GET/POST | SS7 attack records | telecom_sec:ss7 |
| /telecom-sec/diameter | GET/POST | Diameter attack records | telecom_sec:diameter |
| /telecom-sec/intercept | GET/POST | Lawful intercept console | telecom_sec:intercept |
| /telecom-sec/incidents | GET/POST | Security incident queue | telecom_sec:incidents |
| /telecom-sec/threat-intel | GET/POST | Threat intelligence | telecom_sec:threat_intel |
| /telecom-sec/agents | GET/POST | Security agent workbench | telecom_sec:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| intercept_without_warrant_denied | warrant_present=False | deny |
| intercept_regulatory_authority_required | authority_present=False | deny |
| fraud_block_requires_evidence | evidence_present=False | deny |
| evidence_fabrication_denied | agent fabrication scope | deny |
| cross_tenant_access_denied | cross-tenant agent scope | deny |
| sec_batch_requires_bytewax | non-bytewax stream | deny |

## Data Models
- SecFraudCase: id, tenant_id, fraud_type, msisdn, confidence_score, evidence_reference, status
- SecSs7Attack: id, tenant_id, attack_type, source_reference, target_reference, evidence_reference
- SecDiameterAttack: id, tenant_id, attack_type, source_realm, target_realm, evidence_reference
- SecLawfulIntercept: id, tenant_id, intercept_type, target_msisdn, warrant_reference, regulatory_authority, status
- SecIncident: id, tenant_id, incident_type, severity, description, evidence_reference, status
- SecThreatIntel: id, tenant_id, source, ioc_type, ioc_value, tlp_level, valid_from, shared
- SecAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- fraud_case_raised, fraud_block_applied, ss7_attack_detected
- diameter_attack_detected, intercept_activated, security_incident_opened
- security_incident_resolved, threat_ioc_shared, voip_fraud_detected, sec_agent_registered

## Edge Cases Handled
- Warrant missing → intercept activation denied, no partial warrants accepted
- Fraud block without evidence → blocked even if confidence score is 0.99
- Evidence fabrication by agents → hard deny, not configurable
- Cross-tenant IOC sharing must be explicit (shared=True), not automatic
- Intercept expiry tracked — expired intercepts must be renewed via new warrant
- Regulatory authority field separate from warrant to support dual-sign jurisdictions

## New Methods (v1.1)
| Method | Description |
|--------|-------------|
| `evaluate_fraud_risk_score()` | Multi-signal 0–1 risk score for an MSISDN (velocity, geo, IOC, SIM swap) |
| `correlate_signaling_attacks()` | Cluster SS7/Diameter attacks to detect coordinated campaigns |
| `generate_security_posture_score()` | Unified 0–100 posture score with grade across 4 security dimensions |
| `enrich_threat_intel_ioc()` | Cross-reference an IOC against all internal signal stores |
| `run_red_team_scenario()` | Replay GSMA attack scenarios against live detection logic |
| `manage_intercept_expiry()` | Audit LI expiry: mark expired, warn pre-expiry, idempotent |
| `multi_jurisdiction_compliance_matrix()` | Per-jurisdiction compliance matrix (KE/TZ/UG/ZA/EU/US) |
| `subscriber_anomaly_detection()` | Z-score behavioral anomaly detection across 5 subscriber metrics |

## World-Class Enhancements (v2.0)

Full details in `WORLD_CLASS_IMPROVEMENTS.md`.

1. **ML Fraud Scoring** — local Ollama model replaces threshold heuristics for CDR/SS7/SIM swap scoring
2. **Bytewax Streaming Pipeline** — persistent dataflow replacing in-memory lists; sub-second fraud decisions at CDR rate
3. **GSMA FS.11 SS7 Firewall** — full Category 1/2/3 MAP/TCAP enforcement with per-PLMN whitelist/blacklist
4. **Diameter Edge Agent (DEA)** — S6a/S6d/S13 proxy with Origin-Realm validation and HSS enumeration detection
5. **SUPI/SUCI Audit Trail** — per-3GPP TS 33.501 SUPI exposure logging with purpose-limitation enforcement
6. **SIM Box Graph Detection** — NetworkX/pgvector call-graph clustering for coordinated SIM box blocking
7. **Zero-Trust NE Access Control** — declarative per-NE/per-PLMN MAP opcode allowlist; policy violations logged
8. **Cryptographic Evidence Chain** — SHA-256 + HMAC linked-list evidence chain admissible in ETSI LI proceedings
9. **Automated LI Lifecycle** — asyncio TaskGroup scheduler for warrant expiry enforcement and pre-expiry ntfy alerts
10. **STIX/TAXII Federation** — STIX 2.1 PostgreSQL JSONB store with TAXII 2.1 endpoint and TLP-aware sharing
11. **CALEA/ETSI LI DF2/DF3** — HI2/HI3 delivery function with S/MIME key exchange and delivery receipts
12. **Subscriber Anomaly Baselines** — rolling 30-day P50/P90 baselines; z-score >3σ flags; Bytewax incremental updates
13. **Multi-Jurisdiction Compliance** — `SecJurisdictionPolicy` table covering KE/TZ/UG/ZA/EU/US with per-jurisdiction matrix
14. **Red Team Regression Framework** — `SecRedTeamScenario` replaying GSMA CVD vectors; pass/fail stored for regression detection
15. **Unified Security Posture Score** — 0–100 tenant score aggregating all signals; daily trend tracking; alerts on >10pt drop in 24h

## New Methods

### `evaluate_fraud_risk_score` — multi-signal fraud scoring per MSISDN

```python
svc = TelecomSecService()
result = await svc.evaluate_fraud_risk_score(
    msisdn="+254700000001",
    features={
        "calls_per_hour": 45,
        "sms_per_hour": 120,
        "geo_anomaly_score": 0.82,
        "is_roaming": True,
        "recent_fraud_flag": False,
        "account_age_days": 14,
    },
    tenant_id="safaricom",
)
# result["risk_score"]        -> 0.0–1.0 composite score
# result["auto_block"]        -> True if score >= 0.85
# result["recommended_action"] -> "block" | "flag" | "monitor" | "pass"
# result["signals"]           -> per-signal breakdown dict
```

### `generate_security_posture_score` — unified 0–100 tenant posture score

```python
posture = await svc.generate_security_posture_score(tenant_id="safaricom")
# posture["score"]      -> 0–100 int
# posture["grade"]      -> "A" | "B" | "C" | "D" | "F"
# posture["dimensions"] -> {"incidents": int, "signaling": int, "fraud": int, "compliance": int}
# posture["summary"]    -> human-readable assessment string
```

### `subscriber_anomaly_detection` — z-score behavioral anomaly detection

```python
anomaly = await svc.subscriber_anomaly_detection(
    msisdn="+254700000002",
    current_metrics={
        "call_volume": 320,
        "data_usage_mb": 4800,
        "sms_count": 15,
        "roaming_duration_min": 0,
        "international_call_ratio": 0.95,
    },
    baseline_metrics={
        "call_volume":             {"mean": 40.0,  "std": 12.0},
        "data_usage_mb":           {"mean": 500.0, "std": 150.0},
        "sms_count":               {"mean": 20.0,  "std": 8.0},
        "roaming_duration_min":    {"mean": 0.0,   "std": 5.0},
        "international_call_ratio":{"mean": 0.05,  "std": 0.04},
    },
    tenant_id="safaricom",
)
# anomaly["is_anomalous"]     -> True
# anomaly["flagged_dimensions"] -> ["call_volume", "international_call_ratio"]
# anomaly["z_scores"]         -> per-metric z-score dict
```

## Composability Notes
Feeds fraud signals to telecom_ana (fraud analytics). Consumes network events from telecom_net for SS7/Diameter correlation. Lawful intercept integrates with comp for regulatory reporting chains. Threat intel composes with grph for adversary network mapping.
