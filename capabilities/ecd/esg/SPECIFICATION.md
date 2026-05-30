# Sustainability and ESG Specification

## Purpose

Sustainability and ESG Management lets APG applications compose ESG profile, framework, metric, measurement, target, supplier assessment, initiative, risk, report, stakeholder, engagement, and agent workflows into reporting and governance applications.

## Functional Scope

- Create tenant ESG profiles with name, industry, country, reporting year, and owner.
- Attach reporting frameworks from the supported framework list.
- Define ESG metrics by pillar, type, unit, name, and owner.
- Record measurements with period, value, source, evidence, and review where source quality requires it.
- Set ESG targets with baseline, target, due date, type, and owner.
- Record supplier ESG assessments with score, risk tier, evidence, and high-risk owner.
- Record initiatives, risks, reports, stakeholders, and engagements.
- Treat ESG agents as first-class capability citizens with supported runtime, role, scope, and human approval requirements.

## Guardrails

The rule engine rejects missing tenant context, write operations without policy attachment, missing evidence, unsupported frameworks, unsupported metric pillars/types/units, unsupported measurement sources, unsupported report types, invalid supplier scores, high risks without owners, unapproved reports, missing stakeholder consent, unsupported agent runtimes or roles, unaudited state changes, and non-Bytewax batch routing.

The rule engine requires review for supplier/calculated measurements and privileged ESG-agent actions.
