# Threat Intelligence

`intel_threats` is an executable APG capability package for building governed
threat-intelligence applications. It provides a concrete async runtime for
lawful authority, threat workspaces, source lineage, indicators, actors,
campaigns, assessments, reports, mitigations, reviews, Bytewax lifecycle
checks, UI models, and provider-neutral AI-agent support.

## What It Provides

### Governance & Lifecycle

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

### Indicator Management

| Method | Description |
|--------|-------------|
| `create_indicator()` | Create and store an IOC with TLP, confidence, and context |
| `enrich_indicator()` | Type-aware enrichment (GeoIP, WHOIS, CVSS, EPSS, sandbox data) |
| `retire_indicator()` | Soft-retire an IOC with reason |
| `bulk_import_indicators()` | Parse STIX 2.1 bundle and import all indicator objects |
| `export_indicators()` | Export IOCs as STIX, MISP, CSV, or OpenIOC |
| `search_indicators()` | Full-text search across values, sources, and context |
| `indicator_overlap_check()` | Find campaigns sharing a given indicator value |
| `staleness_management()` | Auto-retire indicators older than N days |
| `apply_confidence_decay()` | Apply type-specific half-life decay to a confidence score |

### Threat Actor Profiling

| Method | Description |
|--------|-------------|
| `create_threat_actor()` | Create actor profile with motivation and sophistication |
| `link_actor_to_indicator()` | Associate an actor with an IOC with relationship type |
| `link_actor_to_campaign()` | Associate an actor with a campaign and role |
| `update_actor_profile()` | Update TTPs, target sectors, and known tools |
| `actor_attribution_report()` | Full attribution dossier with MITRE tactic coverage |
| `actor_search()` | Search actors by name, alias, country, motivation, sophistication |
| `reverse_attribution()` | Rank candidate actors for an unknown indicator via graph walk |

### Campaign Tracking

| Method | Description |
|--------|-------------|
| `create_campaign()` | Create a campaign record with sectors and regions |
| `add_campaign_indicator()` | Associate an IOC with a campaign with temporal bounds |
| `add_campaign_technique()` | Associate an ATT&CK technique with a campaign |
| `campaign_timeline()` | Chronological event timeline for a campaign |
| `active_campaigns_report()` | All active campaigns with indicator and technique counts |
| `campaign_similarity()` | Jaccard-based IOC and technique overlap between two campaigns |

### MITRE ATT&CK Integration

| Method | Description |
|--------|-------------|
| `map_technique()` | Look up ATT&CK technique metadata by ID |
| `get_techniques_for_actor()` | All techniques attributed to an actor with metadata |
| `coverage_analysis()` | Identify detection gaps from observed techniques |
| `kill_chain_mapping()` | Map indicators to Lockheed Martin Kill Chain phases |
| `attack_path_analysis()` | Reconstruct likely attack paths; predict next phase |

### Vulnerability Intelligence

| Method | Description |
|--------|-------------|
| `vulnerability_triage()` | EPSS-augmented CVE scoring for patch prioritisation |
| `ingest_sandbox_report()` | Parse Cuckoo/Any.run/Triage JSON; extract IOCs and techniques |

### Attribution Intelligence

| Method | Description |
|--------|-------------|
| `reverse_attribution()` | Reverse-walk indicator → campaign → actor graph |
| `generate_simulation_playbook()` | CALDERA/Atomic Red Team playbook from actor TTPs |

### Temporal Intelligence

| Method | Description |
|--------|-------------|
| `apply_confidence_decay()` | Exponential half-life decay by IOC type |
| `longitudinal_trend_analysis()` | Weekly/monthly time-series with SMA smoothing |

### PIR & Hunting

| Method | Description |
|--------|-------------|
| `intelligence_requirement()` | Register a Priority Intelligence Requirement |
| `score_pir_satisfaction()` | Keyword-overlap satisfaction scoring for open PIRs |
| `generate_hunting_hypotheses()` | Sigma sketches and observables per actor TTP |

### Reporting & Sharing

| Method | Description |
|--------|-------------|
| `generate_threat_report()` | Structured report with indicators, actors, and campaigns |
| `share_via_taxii()` | Push report to TAXII 2.1 collection |
| `export_misp_event()` | Export indicators as MISP JSON event |
| `dissemination_log()` | Retrieve all dissemination events for a report |
| `confidence_calibration_report()` | Brier-score analyst calibration analysis |

### Feed Management

| Method | Description |
|--------|-------------|
| `register_feed()` | Register an external TI feed (STIX, MISP, CSV, TAXII) |
| `ingest_feed()` | Pull and parse the latest indicators from a feed |
| `feed_quality_report()` | FP rate, staleness, dedup rate, quality grade |
| `deduplicate_from_feed()` | Fingerprint-based duplicate retirement |
| `feeds_dashboard()` | Summary across all registered feeds |

## Quick Start

```python
from capabilities.intel.threats import ThreatIntelligenceService

svc = ThreatIntelligenceService()

# Governance setup
authority = svc.record_authority(
    "auth-1", "tenant-a", "mission_order",
    "scope-ref", "confidential", "approver-1",
    "2027-12-31", "authority-evidence",
)
workspace = svc.record_workspace(
    "ws-1", "tenant-a", "cyber_threat", "Ops Workspace",
    "confidential", "auth-1", "workspace-evidence",
)

# Async indicator and actor operations
import asyncio

async def main():
    ioc = await svc.create_indicator(
        ioc_type="ip_address",
        value="185.234.218.1",
        confidence=0.85,
        tlp="amber",
        source="threat-feed-alpha",
    )
    enriched = await svc.enrich_indicator(ioc["id"])

    actor = await svc.create_threat_actor(
        name="APT-Example",
        aliases=["DarkBear", "UNC9999"],
        motivation="espionage",
        sophistication="advanced",
        origin_country="XX",
    )
    await svc.update_actor_profile(
        actor["id"],
        ttps=["T1566.001", "T1059.001", "T1041"],
        target_sectors=["finance", "government"],
        known_tools=["Cobalt Strike", "Mimikatz"],
    )

    # Patch prioritisation
    triaged = await svc.vulnerability_triage(["CVE-2024-1234", "CVE-2024-5678"])
    print(triaged[0]["priority_label"])  # "CRITICAL", "HIGH", etc.

    # Temporal decay
    decay = await svc.apply_confidence_decay(ioc["id"])
    print(decay["new_confidence"])

    # Trend analysis
    trends = await svc.longitudinal_trend_analysis(period_days=90, bucket="weekly")
    print(trends["narrative"])

asyncio.run(main())
```

## Guardrails

The capability denies: unsupported attribution, fabricated indicators, source
tampering, privacy bypasses, autonomous mitigation, unapproved publication, and
privileged agent actions without human approval. AI agents are first-class but
bounded; supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.

## Generated Application Surfaces

- `app.semantic_model()` — APG semantic model for compiler output
- `app.component_manifest()` — publishable component manifest
- `app.self_test()` — package entrypoint and invariant verification
- `api.py` — process-local helpers for generated applications
- `views.py` — Flask-AppBuilder dashboard, console, and agent-workbench view models

## Improvement Roadmap

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 prioritised enhancements covering:
Diamond Model attribution, shared-infrastructure graph detection, behavioral
malware clustering, STIX 2.1 relationship graph export, multi-tenant TLP
enforcement, adversary simulation playbook generation, and more.

## Verification

Focused verification covers Python compilation, app self-test, manifest JSON
validation, package tests, APG inspect, APG publish-plan, package implementation
audit, lifecycle audit, global implementation audit, strict package-artifact
audit, stale-marker scan, disallowed messaging scan, and `git diff --check`.
