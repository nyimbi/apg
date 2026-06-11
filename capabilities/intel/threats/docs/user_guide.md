# Threat Intelligence

**Capability ID**: `intel_threats` | **Domain**: `intel` | **Version**: `1.2.0`

## Description

`intel_threats` is an executable APG capability for building governed
threat-intelligence applications. It provides a full async runtime for
IOC lifecycle management, threat actor profiling, campaign tracking, MITRE
ATT&CK integration, vulnerability triage, feed management, PIR satisfaction
scoring, threat-hunting hypothesis generation, temporal confidence decay,
and longitudinal trend analysis.

---

## Installation

```bash
pip install apg-intel-threats
```

---

## Provides

- `threat_authority_workflow`
- `threat_workspace_workflow`
- `threat_source_workflow`
- `threat_indicator_workflow`
- `threat_actor_workflow`
- `threat_campaign_workflow`
- `threat_assessment_workflow`
- `threat_report_workflow`
- `threat_mitigation_workflow`
- `threat_review_workflow`
- `threat_agent_workflow`

---

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-threats/dashboard` | `intel_threats:view` | Overview |
| `/intel-threats/authorities` | `intel_threats:authorities` | Governance |
| `/intel-threats/workspaces` | `intel_threats:workspaces` | Planning |
| `/intel-threats/sources` | `intel_threats:sources` | Evidence |
| `/intel-threats/indicators` | `intel_threats:indicators` | Evidence |
| `/intel-threats/actors` | `intel_threats:actors` | Analysis |
| `/intel-threats/campaigns` | `intel_threats:campaigns` | Analysis |
| `/intel-threats/assessments` | `intel_threats:assessments` | Analysis |
| `/intel-threats/reports` | `intel_threats:reports` | Reporting |
| `/intel-threats/feeds` | `intel_threats:feeds` | Feeds |
| `/intel-threats/hunting` | `intel_threats:hunting` | Hunting |
| `/intel-threats/trends` | `intel_threats:trends` | Analytics |

---

## Key Service Methods

### Governance

```python
svc.record_authority(authority_id, tenant_id, authority_type, scope_reference,
                     classification, approver_id, expires_at, evidence_reference)
svc.record_workspace(workspace_id, tenant_id, workspace_type, name,
                     classification, authority_id, evidence_reference)
svc.register_source(source_id, tenant_id, workspace_id, source_type,
                    source_reference, custodian_id, lineage_reference, evidence_reference)
```

All write operations evaluate deterministic policy rules. Violations raise
`PermissionError`.

### Indicator Management

```python
ioc = await svc.create_indicator(
    ioc_type="ip_address",   # ip_address | domain | url | file_hash_{md5,sha1,sha256}
    value="185.234.218.1",   # | email | cve_id | yara_rule | sigma_rule
    confidence=0.85,         # 0.0–1.0
    tlp="amber",             # white | green | amber | red
    source="my-feed",
    context={"campaign": "op-sandstorm"},
)

enriched = await svc.enrich_indicator(ioc["id"])
# Returns type-specific enrichment: GeoIP/ASN for IPs, WHOIS for domains,
# CVSS+EPSS for CVEs, detection ratio for file hashes, etc.

await svc.retire_indicator(ioc["id"], reason="false_positive_confirmed")

result = await svc.bulk_import_indicators(stix_bundle)
# Returns {imported_count, skipped_count, error_count, imported_ids, ...}

export = await svc.export_indicators(
    filters={"ioc_types": ["ip_address", "domain"], "confidence_min": 0.6},
    format="stix",  # stix | misp | csv | openioc
)

hits = await svc.search_indicators(
    query="185.234",
    ioc_types=["ip_address"],
    confidence_min=0.5,
)

overlaps = await svc.indicator_overlap_check("185.234.218.1")

staleness = await svc.staleness_management(older_than_days=60)
# Retires all active indicators not updated within 60 days

decay = await svc.apply_confidence_decay(ioc["id"])
# Applies exponential half-life decay (IP: 14d, domain: 30d, file hash: 180d, ...)
# Returns {original_confidence, decay_factor, new_confidence, retired_by_decay}
```

### Threat Actor Profiling

```python
actor = await svc.create_threat_actor(
    name="APT-Example",
    aliases=["DarkBear"],
    motivation="espionage",       # espionage | financial | hacktivism | terrorism | disruption
    sophistication="advanced",    # minimal | intermediate | advanced | nation-state
    origin_country="RU",
)

await svc.update_actor_profile(
    actor["id"],
    ttps=["T1566.001", "T1059.001", "T1041"],
    target_sectors=["finance", "government"],
    known_tools=["Cobalt Strike", "Mimikatz"],
)
# Unrecognised technique IDs are stored but flagged as unverified.

await svc.link_actor_to_indicator(actor["id"], ioc["id"],
    relationship_type="uses", confidence=0.80)

await svc.link_actor_to_campaign(actor["id"], campaign["id"], role="operator")

dossier = await svc.actor_attribution_report(actor["id"])
# Returns full dossier: linked indicators, campaigns, tactic coverage, avg confidence

candidates = await svc.reverse_attribution(indicator_id)
# Walks indicator -> campaign -> actor graph; returns ranked list with confidence intervals
```

### Campaign Tracking

```python
campaign = await svc.create_campaign(
    name="Operation Sandstorm",
    start_date="2026-01-15",
    objective="Data exfiltration from financial sector",
    target_sectors=["finance", "insurance"],
    target_regions=["EU", "NA"],
)

await svc.add_campaign_indicator(campaign["id"], ioc["id"],
    first_seen="2026-01-20T00:00:00Z",
    last_seen="2026-04-01T00:00:00Z")

await svc.add_campaign_technique(campaign["id"], "T1566.001",
    notes="Spearphishing attachments targeting CFOs")

timeline = await svc.campaign_timeline(campaign["id"])
# Returns chronological events: indicator first/last seen, technique observations

similarity = await svc.campaign_similarity(campaign1["id"], campaign2["id"])
# Returns Jaccard similarity for IOC overlap, technique overlap, and composite score
```

### MITRE ATT&CK Integration

```python
tech = await svc.map_technique("T1566.001")
# Returns metadata: name, tactic, kill_chain_phase, sub_techniques, MITRE URL

techniques = await svc.get_techniques_for_actor(actor["id"])

gaps = await svc.coverage_analysis(["T1566", "T1059.001", "T1041"])
# Returns covered_tactics, uncovered_tactics, coverage_ratio, recommended_techniques

phases = await svc.kill_chain_mapping(indicator_ids)
# Heuristically maps IOC types to Lockheed Martin Kill Chain phases

path = await svc.attack_path_analysis(["T1566", "T1059.001", "T1055", "T1041"])
# Returns tactic_groups, kill_chain_coverage, current_phase, predicted_next_phase,
# attack_progression, mitigation_hints
```

### Vulnerability Intelligence

```python
# EPSS-augmented CVE prioritisation
triaged = await svc.vulnerability_triage(["CVE-2024-1234", "CVE-2024-9999"])
# Returns list sorted by priority_score desc, each entry includes:
# {cve_id, cvss_base_score, epss_probability, cisa_kev, exploit_in_wild,
#  priority_score, priority_label, recommendation, nvd_url}

# Ingest behavioral sandbox report (Cuckoo / Any.run / Triage JSON)
result = await svc.ingest_sandbox_report(
    sandbox_report={
        "type": "cuckoo",
        "network": {
            "hosts": ["185.234.218.1"],
            "domains": [{"domain": "malicious-example.com"}],
        },
        "dropped": [{"sha256": "abcdef1234...", "name": "loader.exe"}],
        "signatures": [{"ttp": ["T1059.001", "T1041"]}],
    },
    campaign_id=campaign["id"],   # optional; links extracted IOCs to campaign
)
# Returns {extracted_ioc_count, techniques_linked, campaign_links_created, errors}
```

### Adversary Simulation

```python
playbook = await svc.generate_simulation_playbook(
    actor_id=actor["id"],
    target_platform="windows",   # windows | linux | macos | cloud
)
# Returns ordered steps compatible with CALDERA / Atomic Red Team / VECTR:
# {step_number, technique_id, technique_name, tactic, kill_chain_phase,
#  atomic_test_ref, expected_artifacts, detection_opportunity}
```

### Temporal Intelligence

```python
# Confidence decay -- exponential half-life by IOC type
decay = await svc.apply_confidence_decay(ioc["id"])
# {original_confidence: 0.85, decay_factor: 0.612, new_confidence: 0.52,
#  retired_by_decay: False, age_days: 35.2, half_life_days: 30}

# Longitudinal trend analysis
trends = await svc.longitudinal_trend_analysis(period_days=90, bucket="weekly")
# Returns time-series {labels, indicators.counts, indicators.sma, indicators.trend,
#                      campaigns.counts, campaigns.trend, narrative}
# trend values: "accelerating" | "decelerating" | "stable" | "emerging" | "flat"
```

### PIR Management

```python
pir = await svc.intelligence_requirement(
    requirement_text="Identify C2 infrastructure used by APT-Example targeting EU finance",
    priority="critical",   # critical | high | medium | low
    requester="analyst-001",
)

# Score how well collected intelligence satisfies the PIR
score = await svc.score_pir_satisfaction(pir["id"])
# Returns {satisfaction_score: 0.82, satisfaction_label: "satisfied",
#          contributing_artifacts: [{artifact_type, artifact_id, relevance}, ...]}
```

### Threat-Hunting Hypothesis Generation

```python
hypotheses = await svc.generate_hunting_hypotheses(
    actor_id=actor["id"],
    target_platform="windows",
)
# Each hypothesis includes:
# {technique_id, technique_name, hunt_priority, sigma_sketch (dict),
#  observable_fields, false_positive_notes, mitre_url}
```

### Reporting & Sharing

```python
report = await svc.generate_threat_report(
    classification="tlp:amber",
    report_type="flash_report",   # flash_report | assessment | weekly_digest | attribution_report
    target_audience="SOC Team",
    title="APT-Example Campaign Update",
    summary="...",
    indicator_ids=[ioc["id"]],
    actor_ids=[actor["id"]],
    campaign_ids=[campaign["id"]],
)

await svc.share_via_taxii(report["id"],
    taxii_server_url="https://taxii.example.com",
    collection_id="apt-sharing",
)

misp_event = await svc.export_misp_event([ioc["id"]])

log = await svc.dissemination_log(report["id"])

calibration = await svc.confidence_calibration_report(
    analyst_id="analyst-001",
    period="2026-Q1",
)
# Returns {brier_score, overconfidence_bias, recommendation, ...}
```

### Feed Management

```python
feed = await svc.register_feed(
    name="AlienVault OTX",
    url="https://otx.alienvault.com/taxii/discovery",
    format="taxii",        # stix | misp | csv | taxii | openioc | json
    auth_method="api_key", # none | api_key | bearer_token | basic | mtls
    update_frequency="*/6 * * * *",
)

batch = await svc.ingest_feed(feed["id"])
quality = await svc.feed_quality_report(feed["id"])
# {quality_score, quality_grade (A/B/C/D), false_positive_rate, staleness_ratio, ...}

dedup = await svc.deduplicate_from_feed(feed["id"], batch["batch_id"])
dashboard = await svc.feeds_dashboard()
```

---

## Confidence Decay Half-Lives

| IOC Type | Half-Life |
|----------|-----------|
| ip_address | 14 days |
| domain | 30 days |
| url | 21 days |
| file_hash_* | 180 days |
| email | 60 days |
| cve_id | 365 days |
| yara_rule / sigma_rule | 120 days |

An IOC with `new_confidence < 0.10` after decay is flagged `retired_by_decay: true`
and should be removed from active blocking feeds.

---

## Guardrails

The capability denies the following via deterministic policy rules:

- Unsupported attribution without evidence
- Fabricated indicator injection
- Source record tampering
- Privacy bypass in enrichment pipelines
- Autonomous mitigation actions without human approval
- Unapproved publication of TLP-restricted material
- Privileged agent scope without human approval record

Supported AI agent runtimes: `codex`, `claude_code`, `opencode`, `pi`.

---

## Interoperability

```apg
use intel_threats;
```

Composes with: `intel_alerts`, `intel_correlation`, `intel_prediction`,
`intel_dashboard`, `intel_reporting`.

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment
variables prefixed with `INTEL_THREATS_`.

| Variable | Description | Default |
|----------|-------------|---------|
| `INTEL_THREATS_DEFAULT_TLP` | Default TLP for new indicators | `white` |
| `INTEL_THREATS_STALENESS_DAYS` | Staleness threshold for auto-retire | `90` |
| `INTEL_THREATS_DECAY_ENABLED` | Enable scheduled confidence decay | `true` |
| `OLLAMA_BASE_URL` | Enable ML-powered threat scoring via local Ollama | unset |

---

## Further Reading

- `service.py` -- Business logic (all async methods)
- `models.py` -- Pydantic v2 data models
- `api.py` -- REST API endpoints
- `views.py` -- Flask-AppBuilder views and Pydantic schemas
- `README.md` -- Quick method reference
- `WORLD_CLASS_IMPROVEMENTS.md` -- 15 prioritised enhancement proposals
- `cap_spec.md` -- Full capability specification
