# Advanced CRM Analytics

**Capability ID**: `crm_adv` | **Domain**: `crm` | **Version**: `1.0.0`

## Description

Advanced CRM Analytics (`crm.adv`) is the full-lifecycle customer relationship management capability for the APG platform. It provides a governed, multi-tenant surface covering account management, contact relationship mapping, lead scoring and assignment, sales pipeline tracking, activity timelines, campaign governance, and forecast analytics — all wired to the APG event bus via Bytewax for real-time state propagation.

## Installation

```bash
pip install apg-crm-adv
```

## Provides

- `account_lifecycle`
- `contact_relationship_management`
- `lead_scoring_and_assignment`
- `sales_pipeline_management`
- `activity_timeline`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/crm-adv/dashboard` | `crm_adv:view` | Overview |
| `/crm-adv/accounts` | `crm_adv:manage_accounts` | Accounts |
| `/crm-adv/contacts` | `crm_adv:manage_contacts` | Contacts |
| `/crm-adv/leads` | `crm_adv:manage_leads` | Pipeline |
| `/crm-adv/pipeline` | `crm_adv:manage_pipeline` | Pipeline |
| `/crm-adv/activities` | `crm_adv:manage_activities` | Engagement |
| `/crm-adv/campaigns` | `crm_adv:manage_campaigns` | Engagement |
| `/crm-adv/forecasts` | `crm_adv:forecast` | Analytics |

## Key Service Methods

- `lead_capture()`
- `lead_scoring()`
- `lead_assignment()`
- `opportunity_create()`
- `opportunity_stage_advance()`
- `pipeline_report()`
- `customer_segmentation()`
- `campaign_analytics()`
- `win_loss_analysis()`
- `crm_dashboard()`

_(See `service.py` for complete API.)_

## Interoperability

`crm_adv` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use crm_adv;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `CRM_ADV_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
