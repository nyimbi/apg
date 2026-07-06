# Capability Console Round-2 Research

## Commercial leader

Open Policy Agent is the best-in-class reference for policy decision testing and decision-log auditability. LaunchDarkly is the strongest adjacent reference for change approvals and history, AWS AppConfig is the benchmark for validated configuration rollout, and ServiceNow-style SLA tracking is the operational reference for approval timing.

## Leader weaknesses

- OPA is powerful, but its test and decision-log workflows are developer-oriented rather than embedded in generated business capability consoles.
- LaunchDarkly approvals and history are excellent for feature changes, but they are not automatically tied to APG capability rules, configuration, and approval planning.
- AWS AppConfig validates and deploys configuration, but dry-run differences are not embedded beside APG rule and approval tools.
- SLA platforms track approval timing, but they live apart from the capability rule/configuration workspace.

## Differentiators proposed

1. Rule Test-bench: generated one-click contexts for baseline, high-risk, and international policy scenarios.
2. Dry-run Diff: compare default configuration to resolved configuration and highlight changed keys.
3. Approval SLA Countdown: show approver count, target window, remaining time, and required/not-required state inline.
4. Local Bench Persistence: remember the last loaded test case in browser storage without new dependencies.

## Shipped verdict

APG now upgrades the capability console from three standalone forms into an integrated governance cockpit. Before, users could evaluate rules, resolve config, and plan approval. After, they can stage policy scenarios, inspect config deltas, and watch approval SLA context before submitting operations.
