# Capability Console Rules Config Approval

Date: 2026-07-04

## Best-In-Class Patterns

- Policy simulators make the request context explicit, keep the submitted input visible, and summarize allow/deny effects before exposing raw policy JSON.
- OPA decision-log and debugging patterns emphasize traceability: input, decision, matched rule path, and related metadata should be inspectable after evaluation.
- Feature-flag and configuration consoles separate resolved effective values from default values so operators can preview the impact of overrides.
- Audit and approval tools make reviewer requirements visible before execution and keep raw event detail available as secondary evidence.

## Live Audit

Representative app: `examples/09_capability_rules_configuration/output/app.py`.

Before server: `127.0.0.1:20899`.

Observed defects:

- The console rendered three blank JSON textareas, so operators needed to know the payload schema before they could evaluate anything.
- Results were mostly raw JSON or a generic key/value list, with no operation-specific summary for decisions, configuration, or approval requirements.
- Submitted JSON was not preserved in the operation form after POST, making repeated scenario testing awkward.
- Capability rules and default configuration were available only inside raw capability JSON.
- Approval output did not promote required levels or approver identities.

After server: `127.0.0.1:20900`.

After verification:

- The default console renders sample rule and approval contexts plus the declared default configuration.
- Rules evaluation preserves submitted context and summarizes the decision, matched rules, and actions.
- Configuration resolution preserves submitted overrides and renders the effective configuration before raw JSON.
- Approval planning preserves submitted context and promotes required/not-required state plus approver badges.
- Raw capability and result JSON remain available through disclosure panels for debugging.

## Fix List

Must-fix:

- Replace blank payload boxes with generated defaults derived from the capability description.
- Preserve submitted JSON independently for rule, configuration, and approval operations.
- Render operation-specific summaries for decisions, resolved configuration, and approval plans.

High-value polish:

- Promote default configuration and declared rules into a capability profile panel.
- Keep raw JSON inspectable but secondary.
- Add regression coverage for GET and all three POST flows using the generated companion capability module.

## Validation

- Regenerated all 20 numbered examples.
- Live after audit: `assets/after-console.html`, `assets/after-rules.html`, `assets/after-config.html`, and `assets/after-approval.html`.
- Targeted tests: `3 passed` across the capability console regression, template route coverage, and CSS class coverage.
- Full suite: `1483 passed, 1 skipped, 3 warnings in 730.43s`.
