# Rationale

## Decision

Ship a capability intelligence strip above the three existing operation forms: rule test-bench, dry-run diff, and approval SLA. Compute it from default rule context, default/resolved configuration, approval policy, and current operation result.

## Why this beats the benchmark

The benchmarks split policy testing, configuration validation, approvals, history, and SLA tracking across specialized tools. APG can compress the governance loop into one generated capability console because it owns the capability contract and operation forms.

## Rejected alternatives

- Server-side test suite registry: rejected to avoid persistence and permissions scope.
- Live SLA scheduler: rejected because generated apps do not have a scheduler contract for this surface.
- Rule authoring editor: rejected because the console is for evaluating generated capability contracts, not modifying them.

## Validation target

Generated capability console HTML must still preserve operation inputs and summarize rules/configuration/approval results while adding rule test-bench, dry-run diff, and approval SLA content.
