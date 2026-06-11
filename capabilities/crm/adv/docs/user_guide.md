# Advanced CRM Analytics — User Guide

**Capability ID**: `crm_adv` | **Domain**: `crm` | **Version**: `1.1.0`
**Company**: Datacraft | **Author**: Nyimbi Odero | **Copyright**: © 2025

---

## Overview

Advanced CRM Analytics (`crm.adv`) is the full-lifecycle customer relationship management capability for the APG platform. It covers account management, contact relationship mapping, AI-powered lead scoring and assignment, sales pipeline tracking, CPQ, activity timelines, campaign governance, forecast analytics, 360-degree customer views, proactive deal risk alerting, conversation intelligence, multi-touch attribution, and ARR waterfall reporting — all wired to the APG event bus via Bytewax+NATS.

---

## Installation

```bash
pip install apg-crm-adv
```

For local AI features (copilot, NBA engine, call analysis), install and start Ollama:

```bash
brew install ollama
ollama serve
ollama pull llama3
export OLLAMA_BASE_URL=http://localhost:11434
```

---

## Quick Start

```python
from capabilities.crm.adv.service import AdvancedCRMService
import asyncio

svc = AdvancedCRMService()

# Create an account
account = svc.create_account(
    account_id="acct_001",
    tenant_id="acme",
    name="Acme Corp",
    owner="rep_jane",
    segment="enterprise",
    territory="east_africa",
)

# Capture and score a lead
lead = svc.lead_capture(
    source="web_form",
    contact_details={"name": "Bob Smith", "email": "bob@acme.com", "company": "Acme Corp"},
    campaign_id="camp_q2_2025",
    tenant_id="acme",
)

score = svc.lead_scoring(lead["id"], model_type="predictive", tenant_id="acme")
print(f"Lead score: {score['score']} | Qualified: {score['qualified']}")

# Create an opportunity from the qualified lead
opp = svc.opportunity_create(
    lead_id=lead["id"],
    product_id="product_saas_pro",
    value=50000.0,
    probability=0.65,
    close_date="2026-09-30",
    tenant_id="acme",
)

# Advance through stages
svc.opportunity_stage_advance(opp["id"], "proposal", "Sent proposal deck", tenant_id="acme")

# Run async AI features
async def run_ai():
    # Get 360-degree view
    view = await svc.get_360_view("acct_001", tenant_id="acme")
    print(f"Contacts: {len(view['contacts'])} | Health: {view['health_index']}/100")

    # Assess deal risk
    risk = await svc.compute_deal_risk(opp["id"], tenant_id="acme")
    print(f"Deal risk: {risk['risk_level']} ({risk['risk_score']})")

    # Get next-best actions
    actions = await svc.next_best_action(opp["id"], "opportunity", tenant_id="acme")
    for action in actions:
        print(f"  [{action['confidence']:.0%}] {action['action_type']}: {action['rationale']}")

    # ARR waterfall
    waterfall = await svc.arr_waterfall("2026-06", tenant_id="acme")
    print(f"Net ARR: {waterfall['net_arr']}")

asyncio.run(run_ai())
```

---

## Core Workflows

### Lead-to-Opportunity Pipeline

```
lead_capture()
    -> lead_scoring() [or ml_lead_scoring() for AI scoring]
    -> lead_assignment()
    -> opportunity_create()
    -> opportunity_stage_advance() [repeated]
    -> create_quote() + apply_discount_governance()
```

All transitions emit events to `apg.crm.adv.lifecycle` via Bytewax+NATS.

### Campaign Attribution Flow

```
launch_campaign()
    -> lead_capture() [with campaign_id attribution]
    -> campaign_analytics()
    -> compute_multi_touch_attribution()
```

### Revenue Intelligence Flow

```
opportunity_stage_advance("closed_won" / "closed_lost")
    -> win_loss_analysis()
    -> arr_waterfall()
    -> revenue_forecast()
```

---

## Service API Reference

### Account Management

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_account` | `(account_id, tenant_id, name, owner, segment, territory?)` | Account record |
| `list_accounts` | `(tenant_id)` | List of accounts |
| `get_360_view` *(async)* | `(account_id, tenant_id)` | Full 360 view dict |
| `account_health_index` *(async)* | `(account_id, period, tenant_id)` | Float 0–100 |
| `churn_predict` *(async)* | `(customer_id, tenant_id)` | Churn probability dict |
| `build_abm_target_list` *(async)* | `(icp_definition, limit, tenant_id)` | Ranked target list |

### Lead Management

| Method | Signature | Returns |
|--------|-----------|---------|
| `lead_capture` | `(source, contact_details, campaign_id, tenant_id, ...)` | Lead record |
| `lead_scoring` | `(lead_id, model_type, tenant_id, scoring_factors?)` | Score record |
| `ml_lead_scoring` *(async)* | `(lead_id, tenant_id, scoring_factors?)` | AI score record |
| `lead_assignment` | `(lead_id, rep_id, reason, tenant_id, assignment_policy?)` | Updated lead |

### Opportunity & Pipeline

| Method | Signature | Returns |
|--------|-----------|---------|
| `opportunity_create` | `(lead_id, product_id, value, probability, close_date, tenant_id, ...)` | Opportunity record |
| `opportunity_stage_advance` | `(opportunity_id, new_stage, notes, tenant_id, advanced_by?)` | Updated opportunity |
| `pipeline_report` | `(rep_id, period, tenant_id)` | Pipeline summary dict |
| `compute_deal_risk` *(async)* | `(opportunity_id, tenant_id)` | Risk assessment dict |
| `run_deal_risk_scan` *(async)* | `(tenant_id, risk_threshold?)` | Scan results dict |
| `deal_health_score` *(async)* | `(opportunity_id, tenant_id)` | Float 0–1 |

### CPQ (Configure-Price-Quote)

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_quote` | `(opportunity_id, line_items, tenant_id, discount_pct?, valid_days?, notes?)` | Quote record |
| `apply_discount_governance` | `(quote_id, requested_discount_pct, rep_id, tenant_id)` | Approval status dict |
| `list_quotes` | `(tenant_id)` | List of quotes |

#### Discount governance thresholds

| Discount Range | Required Approval |
|---------------|-------------------|
| 0–10% | Auto-approved (rep) |
| 10–20% | Sales manager |
| 20–30% | VP Sales |
| >30% | Executive (CEO) |

### Analytics & Intelligence

| Method | Signature | Returns |
|--------|-----------|---------|
| `campaign_analytics` | `(campaign_id, tenant_id)` | Campaign ROI dict |
| `win_loss_analysis` | `(period, reason_codes?, tenant_id)` | Win/loss stats dict |
| `revenue_forecast` *(async)* | `(period, tenant_id, filters?)` | Forecast dict |
| `arr_waterfall` *(async)* | `(period, tenant_id)` | ARR waterfall dict |
| `crm_dashboard` | `(rep_id, period, tenant_id)` | Dashboard summary |
| `crm_executive_dashboard` *(async)* | `(period, tenant_id)` | Exec dashboard |
| `cohort_retention` *(async)* | `(cohort_definition, periods, tenant_id)` | Retention curve |
| `customer_journey_map` *(async)* | `(customer_id, tenant_id)` | Journey touchpoints |

### AI Features (require `OLLAMA_BASE_URL`)

| Method | Signature | Returns |
|--------|-----------|---------|
| `copilot_query` *(async)* | `(prompt, context_ids, tenant_id)` | Copilot response dict |
| `next_best_action` *(async)* | `(entity_id, entity_type, tenant_id)` | Action list |
| `analyze_call_transcript` *(async)* | `(activity_id, transcript, tenant_id)` | Analysis dict |
| `compute_multi_touch_attribution` *(async)* | `(opportunity_id, model_type, tenant_id)` | Attribution dict |
| `build_abm_target_list` *(async)* | `(icp_definition, limit, tenant_id)` | Target list dict |

### Salesforce Sync

| Method | Signature | Returns |
|--------|-----------|---------|
| `sync_lead_to_salesforce` *(async)* | `(lead_id, tenant_id)` | Sync result |
| `sync_contact_to_salesforce` *(async)* | `(account_id, tenant_id)` | Sync result |

Requires env vars: `SFDC_CLIENT_ID`, `SFDC_CLIENT_SECRET`, `SFDC_USERNAME`, `SFDC_PASSWORD`.

---

## AI Sales Copilot

The copilot uses a local Ollama model to answer natural-language CRM queries. No data leaves your infrastructure.

```python
response = await svc.copilot_query(
    prompt="Which deals are most likely to close this quarter and what should I do next?",
    context_ids=["opp_001", "opp_002", "acct_abc"],
    tenant_id="acme",
)
print(response["response"])
```

NATS stream: `crm.adv.copilot.{tenant_id}` — subscribe here for streaming token delivery to a UI.

---

## Deal Risk Management

### Single deal assessment

```python
risk = await svc.compute_deal_risk("opp_001", tenant_id="acme")
# {
#   "risk_score": 0.72,
#   "risk_level": "high",
#   "drivers": {
#     "days_inactive": 14,
#     "days_to_close": 8,
#     "inactivity_risk": 0.667,
#     "close_date_risk": 0.733,
#     "probability_risk": 0.1,
#   },
#   "recommended_action": "Immediate manager review",
# }
```

### Scheduled scan (APG cron — every 6 hours)

```python
scan = await svc.run_deal_risk_scan(tenant_id="acme", risk_threshold=0.65)
print(f"{scan['at_risk_count']}/{scan['total_scanned']} deals at risk")
```

Deals exceeding the threshold emit `deal_at_risk` events to NATS subject `crm.adv.risk.{tenant_id}`.

---

## Multi-Touch Attribution

```python
attribution = await svc.compute_multi_touch_attribution(
    opportunity_id="opp_001",
    model_type="time_decay",  # first_touch | last_touch | linear | time_decay | data_driven
    tenant_id="acme",
)
for tp in attribution["touchpoints"]:
    print(f"  {tp['event']:30s} {tp['credit']:.2%}")
```

`data_driven` uses Shapley value approximation via Ollama; falls back to `linear` when Ollama is unavailable.

---

## ARR Waterfall

```python
waterfall = await svc.arr_waterfall("2026-06", tenant_id="acme")
# {
#   "new_arr": 120000.00,
#   "expansion_arr": 25000.00,
#   "churn_arr": 15000.00,
#   "net_arr": 130000.00,
#   "gross_arr_added": 145000.00,
# }
```

---

## ABM Target List

```python
targets = await svc.build_abm_target_list(
    icp_definition={
        "industry": "fintech",
        "segment": "enterprise",
        "min_arr": 500_000,
        "geography": "east_africa",
        "min_employees": 100,
    },
    limit=25,
    tenant_id="acme",
)
for t in targets["targets"][:5]:
    print(f"  {t['account_name']:30s}  ICP score: {t['icp_score']}")
```

---

## Streaming Architecture

All CRM events flow through Bytewax+NATS. The primary stream is `apg.crm.adv.lifecycle`.

```
CRM write operation
    -> AdvancedCRMService._emit()
    -> apg.crm.adv.lifecycle (NATS subject, key=tenant_id)
    -> Bytewax pipeline consumers:
        intel.alerts       (anomaly detection)
        intel.correlation  (pipeline drift)
        fintech.terminal   (revenue projections)
        ntfy               (rep notifications)
```

NATS subjects used:

| Subject | Content |
|---------|---------|
| `apg.crm.adv.lifecycle` | All CRM lifecycle events |
| `crm.adv.copilot.{tenant_id}` | Copilot token stream |
| `crm.adv.risk.{tenant_id}` | Deal at-risk alerts |
| `crm.adv.revenue.{tenant_id}` | Revenue events |
| `crm.adv.nba.{tenant_id}` | Next-best-action recommendations |
| `crm.adv.federation.{tenant_id}` | Federated query audit |

---

## Configuration Reference

All keys are tenant-scoped. Defaults shown.

| Key | Default | Description |
|-----|---------|-------------|
| `default_lead_score_threshold` | `70.0` | Minimum score for auto-qualification |
| `default_opportunity_probability` | `50.0` | Default win probability (%) |
| `customer_health_score_enabled` | `true` | Enable account health index |
| `ai_recommendations_enabled` | `true` | Enable next-best-action engine |
| `predictive_analytics_enabled` | `true` | Enable ML lead scoring and risk scan |
| `email_integration_enabled` | `true` | Enable email activity sync |
| `max_records_per_page` | `100` | Pagination cap (10–1000) |
| `cache_ttl_seconds` | `300` | Cache TTL for 360 views (60–3600) |
| `background_job_timeout` | `3600` | Max background job runtime (300–7200) |

---

## Business Rules Summary

| Rule | Effect |
|------|--------|
| `tenant_context_required` | deny — all operations require tenant_id |
| `crm_write_requires_policy` | deny — write operations require policy attachment |
| `lead_assignment_requires_score` | deny — score must be present before assignment |
| `opportunity_amount_must_be_positive` | deny — $0 opportunities rejected |
| `forecast_requires_confidence` | deny — confidence value [0,1] mandatory |
| `bulk_outreach_requires_privacy_review` | require_review — human can override |
| `crm_batch_import_requires_bytewax` | deny — batch imports must route through Bytewax |
| `privileged_agent_crm_action_requires_human_approval` | deny — no autonomous privileged agent actions |

---

## Composability

| Direction | Capability | Integration |
|-----------|-----------|-------------|
| Upstream | `auth` | User identity resolution for owner/assignment fields |
| Upstream | `common_mdm` | Territory, segment, industry reference data |
| Upstream | `composition_config` | Tenant-level runtime config overrides |
| Downstream | `intel.alerts` | Subscribes to lifecycle stream for anomaly detection |
| Downstream | `intel.correlation` | Pipeline drift detection |
| Downstream | `fintech.terminal` | Consumes opportunity and forecast data |
| Downstream | `ntfy` | Activity reminders, assignment alerts, risk notifications |
| Peer | `crm.ord` | Order management — shares account model |
| Peer | `crm.mkt` | Marketing campaigns — shares consent and audience model |

---

## Testing

```bash
# Run CI tests
uv run pytest -vxs tests/ci

# Type check
uv run pyright
```

Test files are in `tests/ci/`. Use real objects and pytest fixtures — no mocks except for LLM calls.

---

## Further Reading

- `/capabilities/crm/adv/service.py` — Business logic implementation
- `/capabilities/crm/adv/models.py` — Pydantic v2 data models
- `/capabilities/crm/adv/api.py` — REST API endpoints
- `/capabilities/crm/adv/views.py` — Flask-AppBuilder views and Pydantic schemas
- `/capabilities/crm/adv/README.md` — Quick reference
- `/capabilities/crm/adv/WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 strategic improvements
- `/capabilities/crm/adv/cap_spec.md` — Formal capability specification
