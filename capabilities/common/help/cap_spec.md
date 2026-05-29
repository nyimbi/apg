# Help and Knowledge Base Capability Specification

- **Capability Name**: Help and Knowledge Base
- **Capability ID**: `help`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `help`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `help_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `article_requires_owner`
- `publication_requires_approval`
- `answer_requires_citations`
- `restricted_content_requires_filtering`
- `stale_article_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `help_support_knowledge` APG theme contract.
