# APG (Application Program Generator) Documentation

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

## Overview

The Application Program Generator (APG) is a Python-first application generation
platform built around a terse, readable DSL, a compiler, generated Python
application artifacts, and a composable capability system. The current
engineering priority is closing the gap between intended APG platform behavior
and executable, verified repository behavior.

## Key Features

- **APG DSL and Compiler**: Parse `.apg` source, build a semantic model, and
  generate dependency-light Python application artifacts.
- **Composable Capabilities**: Define package-backed capabilities with
  configuration, deterministic rules, UI contracts, theme tokens, tests, and
  publish-plan evidence.
- **AI Agent Composition**: Model AI agents and adapter runtimes as first-class
  application components.
- **Screens, Workflows, and Rules**: Express business records, UI composition,
  stateful workflows, and rule contracts in APG source.
- **Tooling Evidence**: Use CLI commands for linting, validation, formatting,
  graphs, release evidence, packaging, language-server checks, Studio
  projections, and aggregate tooling audits.
- **Examples and Capacity Development**: Build from numbered examples toward
  executable ERP and platform capacities.

## 📚 Documentation Structure

### Core Platform
- [Installation & Setup](./installation.md) - Getting started with APG
- [Quick Start Guide](./quickstart.md) - Build your first APG application
- [Architecture Overview](./architecture.md) - System design and components
- [APG Language Guide](./apg_language.md) - Current executable APG language model, entity patterns, capabilities, agents, screens, workflows, rules, i18n, and generated runtime behavior
- [APG Tutorial](./apg_tutorial.md) - Step-by-step path from a table to a composed executable application
- [Language Reference](./language_reference.md) - APG syntax, constructs, and compilation model
- [APG Grammar Guide](./apg_grammar_guide.md) - How `spec/apg.g4` is structured and how to extend it safely
- [APG Cheat Sheet](./apg_cheat_sheet.md) - Compact syntax, command, route, and authoring reference
- [Workflow Reference](./workflow_reference.md) - Workflow-oriented language and runtime reference

### Capabilities & Features
- [Capabilities Overview](./capabilities/README.md) - Complete capability catalog
- [Capability Building Standards](./capability_standards.md) - Standards for capability contracts, rules, UI, theming, i18n, streaming, package shape, tests, and documentation
- [Executable Capability Contracts](./capability_contracts.md) - Configuration, rule-engine, UI, and theme contract registry
- [Application Composition](./application_composition.md) - First-class application shells over capabilities, agents, routes, and deployment metadata
- [Screen Composition](./screen_composition.md) - First-class screen contracts, UI relationships, and generated composition graphs
- [AI Agent Composition](./ai_agent_composition.md) - First-class AI agents, swarms, handoffs, and generated runtime manifests
- [Proposed Capability Architecture](./proposed_capability_architecture.md) - Capability boundaries and platform architecture
- [Marketplace Microservices Guide](./marketplace_microservices_guide.md) - Marketplace service design and integration

### Development
- [Developer Guide](./developer_guide.md) - Immediate developer onboarding for changing APG grammar, compiler, generator, CLI, tooling, language server, Studio, examples, tests, and docs
- [Contributors Guide](./contributors_guide.md) - First-30-minutes setup, contribution workflow, evidence standards, testing expectations, commit protocol, and review checklist
- [Capacity Development Guide](./capacity_development_guide.md) - How to build new executable APG capacities from records, capabilities, rules, screens, workflows, agents, Bytewax streaming, tests, and docs
- [Repository Hygiene](./repository_hygiene.md) - Canonical root, docs, tests, reports, examples, generated-output, and local-artifact placement rules
- [API Reference](./api/README.md) - Complete API documentation
- [Goal Progress Log](./progress_log.md) - Durable progress, verification evidence, and next work for the active APG closure goal
- [Roadmaps](./roadmaps/) - Implementation roadmaps and active planning artifacts moved out of the repository root
- [Specifications](./specifications/) - Capability specification artifacts and executable specification summaries
- [Deployment Guide](./deployment.md) - Production deployment strategies
- [Reports](./reports/README.md) - Historical implementation and validation reports
- [Documentation Archive](./archive/README.md) - Older root README variants and planning references

### Administration
- [Deployment Guide](./deployment.md) - Production deployment strategies

## 🎯 Quick Navigation

| Category | Documentation |
|----------|---------------|
| **Getting Started** | [Installation](./installation.md) → [Quick Start](./quickstart.md) → [Tutorial](./apg_tutorial.md) |
| **Core Platform** | [Architecture](./architecture.md) → [APG Language Guide](./apg_language.md) → [Grammar Guide](./apg_grammar_guide.md) → [Cheat Sheet](./apg_cheat_sheet.md) → [API Reference](./api/README.md) |
| **Capabilities** | [Capabilities](./capabilities/README.md) → [Capability Standards](./capability_standards.md) → [Capability Contracts](./capability_contracts.md) → [Application Composition](./application_composition.md) → [Screen Composition](./screen_composition.md) → [AI Agent Composition](./ai_agent_composition.md) → [Marketplace Guide](./marketplace_microservices_guide.md) |
| **Contributing** | [Developer Guide](./developer_guide.md) → [Contributors Guide](./contributors_guide.md) → [Capacity Development Guide](./capacity_development_guide.md) → [Repository Hygiene](./repository_hygiene.md) |
| **Planning** | [Roadmaps](./roadmaps/) → [Specifications](./specifications/) → [Progress Log](./progress_log.md) |
| **Operations** | [Deployment](./deployment.md) → [Reports](./reports/README.md) → [Progress Log](./progress_log.md) |

## 🏗️ System Requirements

### Minimum Developer Requirements
- Python 3.10+
- `uv`
- 4GB RAM
- 20GB disk space

### Recommended Developer Requirements
- Python 3.11+
- Docker or another container runtime for package/deployment verification
- Enough battery or power budget for focused pytest slices and generated-app
  smoke tests

### Optional Runtime Dependencies
- PostgreSQL or compatible database services for database-backed deployment
  profiles
- Redis or equivalent cache/event infrastructure for capabilities that require
  it
- BeeWare toolkit for mobile packaging work
- provider SDKs for specific AI, payment, blockchain, or integration
  capabilities

## 🔧 Core Technologies

| Component | Technology Stack |
|-----------|------------------|
| **Compiler** | ANTLR grammar, Python parser artifacts, semantic model, Python generator |
| **Backend** | Generated Python artifacts, capability contracts, service helpers |
| **UI** | Screen contracts, UI manifests, theming contracts |
| **AI Agents** | Adapter-oriented runtimes such as Codex, Claude Code, OpenCode, OpenAI, Ollama, and Pi |
| **Streaming** | Bytewax-oriented APG streaming metadata |
| **Packaging** | Web, desktop, mobile, container, and Python package profiles over generated Python artifacts |
| **Tooling** | CLI, language server, Studio projections, fixture audits, release evidence |

## 📖 Documentation Standards

All documentation follows these principles:
- **Production-Ready**: Covers real implementations, not mock or placeholder code
- **Comprehensive**: Complete coverage of features and functionality
- **Up-to-Date**: Reflects current implementation status
- **Actionable**: Includes practical examples and code samples
- **Accessible**: Clear structure and navigation for all skill levels

## 🤝 Support & Community

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas
- **Contact**: nyimbi@gmail.com for direct support

## 📄 License

This project is proprietary software owned by Datacraft. All rights reserved.

---

*Last Updated: May 2026*
*Version: 2.0.0*
