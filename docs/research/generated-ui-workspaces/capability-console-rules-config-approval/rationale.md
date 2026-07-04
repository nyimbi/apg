# Rationale

## Decisions

- Generate default rule and approval contexts from the capability configuration so the console is immediately testable.
- Use the capability's declared default configuration as the configuration override starting point.
- Preserve the exact submitted JSON text for the active operation so operators can iterate without retyping.
- Split result rendering by operation: matched rules and actions for rules, key/value rows for configuration, and approver badges for approvals.
- Promote default configuration and declared rules into a profile panel, while keeping raw capability JSON in a disclosure.
- Load the generated app from a temporary output directory in the regression test so optional generated companion modules resolve the same way they do in the live app.

## Rejected Alternatives

- Adding a browser JSON editor dependency: rejected because generated apps must stay self-contained and within asset budgets.
- Storing submitted JSON in server-side sessions: rejected because the POST response can render preserved values directly.
- Removing raw JSON entirely: rejected because APG users still need inspectable contract and operation payloads for debugging.
- Hardcoding a credit-control-only context: rejected because the compiler should derive safe defaults from the capability description.
