# APG Documentation

APG is the Agentic Platform Generator: an Africa-first, Python-first compiler
and capability platform that turns `.apg` specifications into self-contained
Flask applications.

Use this index for current developer-facing documentation. Historical reports,
older plans, archived grammar drafts, and generated UI research artifacts remain
under their existing folders, but the current executable path is documented in
the files below.

## Start Here

| Need | Document |
| --- | --- |
| Install APG and inspect the CLI | [Installation and CLI reference](installation.md) |
| Compile and run a generated app | [Quick start](quickstart.md) |
| Understand the platform architecture | [Architecture](architecture.md) |
| Change compiler, UI, capabilities, or docs | [Developer guide](developer_guide.md) |
| Understand generated UI workspaces | [Generated UI](generated_ui.md) |

## Core Platform

- [APG Language Guide](apg_language.md)
- [Language Reference](language_reference.md)
- [Language Manual](language_manual.md)
- [APG Grammar Guide](apg_grammar_guide.md)
- [APG Cheat Sheet](apg_cheat_sheet.md)
- [Workflow Reference](workflow_reference.md)
- [Application Composition](application_composition.md)
- [Screen Composition](screen_composition.md)
- [AI Agent Composition](ai_agent_composition.md)

## Capabilities

- [Capabilities Overview](capabilities/README.md)
- [Capability Standards](capability_standards.md)
- [Capability Contracts](capability_contracts.md)
- [Capability Development Guide](capability_development_guide.md)
- [Capability Integration Guide](capability_integration_guide.md)
- [Composability Contract](composability_contract.md)
- [Marketplace Microservices Guide](marketplace_microservices_guide.md)

## Developer Operations

- [Tooling Specification](tooling.md)
- [Repository Hygiene](repository_hygiene.md)
- [Contributors Guide](contributors_guide.md)
- [Capacity Development Guide](capacity_development_guide.md)
- [Deployment Guide](deployment.md)
- [API Reference](api/README.md)

## Planning, Reports, And Research

- [Progress Log](progress_log.md)
- [Reports](reports/README.md)
- [Roadmaps](roadmaps/README.md)
- [Specifications](specifications/README.md)
- [Reference Documents](reference/README.md)
- [Documentation Archive](archive/README.md)
- [Generated UI Research](research/generated-ui-workspaces/SUMMARY.md)
- [Docs Refresh Research](research/docs-update/)

## Current Facts To Preserve

- The current CLI entry point is `apg=cli.main:cli` in `setup.py`.
- The advertised compiler target is `python`.
- Generated apps are Flask apps with local `static/` assets and no CDN
  requirement.
- `apg baseline examples --refresh` is the baseline refresh command.
- The current full test target is `uv run pytest tests/ -q`; the latest run
  completed with 1486 passed, 1 skipped, and 3 warnings.
- Capability inventory should be described as source-tree inventory unless a
  specific audit proves runtime depth for a subset.
