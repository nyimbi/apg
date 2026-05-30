# APG Advanced CRM Analytics Specification

- **Capability Name**: Advanced CRM Analytics
- **Category**: CRM
- **Version**: 2.1.0
- **Capability ID**: `crm_adv`

## Purpose

Advanced CRM Analytics gives APG applications a governed customer-relationship lifecycle for account management, contact consent, lead scoring, lead assignment, opportunity pipeline management, activity tracking, campaign governance, forecast analytics, Bytewax lifecycle events, generated UI surfaces, and AI-assisted sales review.

## Capability Boundaries

The capability owns account, contact, lead, opportunity, activity, campaign, forecast, rule, theme, and CRM-agent package surfaces. It does not own authentication, audit persistence, notification delivery, master customer records, or event infrastructure; those remain adapter dependencies.

## Provides

- `account_lifecycle`
- `contact_relationship_management`
- `lead_scoring_and_assignment`
- `sales_pipeline_management`
- `activity_timeline`
- `campaign_governance`
- `forecast_analytics`
- `crm_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `common_mdm`

## Lifecycle

1. Create accounts with owner, segment, and optional territory.
2. Create contacts with outreach consent when outreach is enabled.
3. Create leads with source and optional score.
4. Assign scored leads through an assignment policy.
5. Create opportunities with account, stage, amount, and close date.
6. Record activities and next steps for open pipeline work.
7. Launch campaigns with audience and consent evidence, requiring privacy review for bulk outreach.
8. Record forecasts with evidence and confidence.
9. Register AI agents for pipeline, lead quality, account strategy, forecast, campaign, and privacy review.

## Rule Engine

The deterministic rule engine denies missing tenant context, writes without policy, incomplete account records, outreach without consent, leads without source, lead assignment without score or policy, incomplete opportunities, forecasts without evidence or confidence, campaigns without audience or consent, non-Bytewax CRM imports/events, unsupported CRM-agent runtime or role, and privileged agent actions without human approval. Open-pipeline activities without a next step and bulk campaigns without privacy review require review.

## UI Contract

The capability exposes screens for dashboard, accounts, contacts, leads, pipeline, activities, campaigns, forecasts, agents, and settings. Theme metadata defines compact CRM surfaces for account segmentation, relationship maps, scoring lanes, sales stages, timelines, campaign privacy, forecast confidence, and agent review lanes.

## Streaming

CRM lifecycle events use the Bytewax processor and stream `apg.crm.adv.lifecycle`. The stream key is `tenant_id`. Events include account creation, contact creation, lead creation, lead assignment, opportunity creation, activity recording, campaign launch, forecast recording, and CRM agent registration.

## AI Agent Composition

CRM agents are first-class capability records. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles are pipeline analyst, lead quality reviewer, account strategist, forecast reviewer, campaign reviewer, and privacy reviewer. Privileged CRM actions require recorded human approval.
