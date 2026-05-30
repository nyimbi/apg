# Product Information Management Specification

## Purpose

Product Information Management lets APG applications compose catalog, product, attribute, variant, content, asset, compliance, channel listing, publication, data quality, change control, and agent workflows into commerce, ERP, and product data applications.

## Functional Scope

- Create catalogs and product records.
- Define reusable product attributes and localized values.
- Create product variants.
- Enrich content with review gates for generated content.
- Attach assets with rights basis.
- Record compliance evidence and high-risk review.
- Create approved channel listings and publish products only when content and channel evidence are approved.
- Record data-quality issues and change requests.
- Treat PIM agents as first-class capability citizens with supported runtime, role, scope, and human approval requirements.

## Guardrails

The rule engine rejects missing tenant context, write operations without policy attachment, incomplete catalogs/products/attributes/values/content/assets/compliance/channel listings/publications/quality records/change records, unsupported product and attribute types, unsupported channels, high-risk records without review or owner evidence, unsupported agent runtimes or roles, unaudited state changes, and non-Bytewax batch routing.
