# DVRL Works Specification

## Purpose

DVRL Works is a documentation and evidence packet for the APG Data
Virtualization capability. It preserves working reports that help contributors
understand prior analysis, remediation, deployment readiness, validation, and
Singer integration decisions.

## Non-Goals

DVRL Works does not define a separate capability, service, UI, workflow,
runtime, rule engine, event stream, package manifest, semantic model, or APG
compiler target.

## Authoritative Runtime

The parent `capabilities/common/dvrl` package owns the runtime lifecycle:

- source registration and activation
- schema refresh review
- virtual table publication
- query guardrail evaluation
- cache decision lifecycle
- policy review
- source retirement
- audit events
- UI routes and view models
- Bytewax-backed event metadata
- package inspection and publish evidence

## Artifact Requirements

This folder may contain Markdown reports and checklists. Reports should state
whether they are background evidence, current operating guidance, or superseded
analysis. Runtime behavior described in a report is only authoritative after it
is reflected in the parent DVRL contract, service, app metadata, and tests.

## Contributor Requirements

When adding or updating a DVRL Works report:

- keep the report tied to the parent DVRL package
- avoid introducing runtime entrypoints in this folder
- avoid duplicating the root capability specification
- cite the parent specification or plan when describing active behavior
- update the parent package when a finding changes executable behavior

## Acceptance Criteria

- This folder has a README, specification, plan, and source-of-truth
  `cap_spec.md` pointer.
- The parent DVRL package remains the only executable APG DVRL capability.
- Package-gap scans no longer misclassify this folder as an undocumented
  runtime capability.
