# APG Advanced CRM Analytics

`crm_adv` gives APG applications a governed CRM operating surface for accounts, contacts, leads, pipeline, activities, campaigns, forecasts, and AI-assisted sales review.

## What It Provides

- Account lifecycle.
- Contact relationship management.
- Lead scoring and assignment.
- Sales pipeline management.
- Activity timeline.
- Campaign governance.
- Forecast analytics.
- CRM AI agents.

## How To Use It

Import the service for in-process generated applications:

```python
from capabilities.crm.adv import AdvancedCRMService

service = AdvancedCRMService()
account = service.create_account(
    "acme",
    "tenant-a",
    "Acme",
    "seller-owner",
    "enterprise",
    "north",
)
```

Create and assign a scored lead:

```python
lead = service.create_lead("lead-1", "tenant-a", "Acme Expansion", "web", 82)
assigned = service.assign_lead("tenant-a", lead["id"], "seller-1", "round_robin")
```

Inspect compiler-facing package evidence:

```bash
./.venv/bin/apg capabilities inspect crm_adv --json
./.venv/bin/apg capabilities publish-plan capabilities/crm/adv --json
```

## Lifecycle

1. Create accounts with owner, segment, and territory.
2. Create contacts with consent for outreach.
3. Create leads with source and score.
4. Assign scored leads through a policy.
5. Create opportunities with account, stage, amount, and close date.
6. Record activities with next steps for open pipeline.
7. Launch campaigns with audience and consent evidence.
8. Record forecasts with evidence and confidence.
9. Register AI agents for CRM review work.

## Screens

- Dashboard
- Accounts
- Contacts
- Leads
- Pipeline
- Activities
- Campaigns
- Forecasts
- Agents
- Settings

## Guardrails

The deterministic rule engine blocks missing tenant context, writes without policy, incomplete accounts, outreach without consent, leads without source, lead assignment without score or policy, incomplete opportunities, forecasts without evidence or confidence, campaigns without audience or consent, non-Bytewax CRM imports/events, unsupported agent runtimes, unsupported agent roles, and privileged agent actions without human approval.

## AI Agent Support

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles are pipeline analyst, lead quality reviewer, account strategist, forecast reviewer, campaign reviewer, and privacy reviewer.
