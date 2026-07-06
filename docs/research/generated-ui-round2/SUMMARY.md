# Round 2 Generated UI Summary

Date: 2026-07-06

## Per-workspace Verdicts

| Workspace | Leader | Differentiators shipped | Before/after verdict | Residual gaps |
|---|---|---|---|---|
| home-dashboard | Grafana | Composable Tiles, Threshold Alerts, Annotation Pins, Scheduled Export | From static overview to operator command center | Real scheduled delivery still needs backend jobs |
| entity-list | Linear | Shareable State, Column Memory, Virtual Window, Keyboard Fuzzy Filter | From table view to durable list workspace | Server-side virtualization remains future work for huge datasets |
| entity-analytics | Tableau | Annotation Pin, Comparative Overlay, Forecast Band | From charts to decision intelligence | Forecasts are heuristic, not statistical models |
| kanban | Jira Software | Cumulative Flow, Swimlanes, WIP Policy | From board to flow-control cockpit | Drag conflict resolution is still local-first |
| record-detail | Notion database pages | Change Diff Timeline, Related Record Graph, Create Sibling Context | From detail page to record cockpit | Notifications are local UI only |
| forms | Typeform | Autosave Draft, Async Field Validation, Smart Defaults, Undoable Submit, Dependency Tree | From generated form to resilient operator entry | Validation remains generated-route scoped |
| workflow-wizard | Temporal Web UI | Live Duration Estimate, Rollback To Step, Save As Template, Step Estimate Ledger | From stepper to workflow execution assistant | Parallel branch visualization is summarized, not graph-native |
| agent-console | ChatGPT | Streaming Meter, Conversation Branching, Tool-call Inspector, Prompt Library, Run Compare | From prompt form to run intelligence console | Token/cost meters use generated estimates |
| capability-console | Open Policy Agent | Rule Test-bench, Dry-run Diff, Approval SLA Countdown, Local Bench Persistence | From forms to governance testbench | Real approval queue integration remains external |
| database-catalog | Prisma Studio | Schema Diff, ER Mini-map, Query Playground, Offline-first Catalog | From schema list to database intelligence surface | Query playground is snippet-oriented, not an executing SQL console |
| flow-debugger | Temporal Web UI | Step Replay Rail, Breakpoint Planner, Variable Inspector, Investigation Verdict | From run timeline to replay-oriented debugger | Breakpoints are local investigation state |
| login-auth | Auth0 | Passkey Readiness Tile, Magic-link Intent, Device Session Review, Lockout Recovery Flow | From login form to auth posture surface | Passkey and magic-link controls are readiness/intent only without backend providers |
| landing-marketplace | Vercel Marketplace | Capability Compare Matrix, Live Demo Boot, Install Proof Ledger, Marketplace Fit Score | From launcher/catalog to proof-backed evaluation path | Real marketplace package install remains a product/runtime decision |
| shell-chrome | Raycast | Command Center Primary Nav, Cross-workspace Recent Items, Onboarding Tour, Undo Toast Stack | From shell navigation to command-first workspace chrome | Recent and undo state are browser-local |

## Cross-workspace Delight Ledger

- Every major generated workspace now exposes a "what changed and why it matters" intelligence layer near the primary workflow.
- Local-first affordances are consistent: drafts, templates, branches, breakpoints, magic-link intent, install proof, recent items, and undo state persist without new backend dependencies.
- Navigation speed improved through shareable filters, command palette routes, generated marketplace proof, and direct inspect links.
- Trust improved through visible proof ledgers: OpenAPI, self-test, offline assets, schema status, rule diffs, workflow journals, and install proof.
- Recovery improved across surfaces: undoable submit, rollback-to-step, local breakpoints, lockout recovery, and shell undo toasts.
- The generated UI remains self-contained: no CDN runtime URLs, no SPA framework, no new generated Python dependencies, and existing examples regenerate from compiler assets/templates.

## Remaining Program Risks

- Several differentiators intentionally model readiness or local intent instead of real external integrations; this preserves offline correctness but should be called out in product docs.
- Advanced collaboration features such as real notification delivery, server-side scheduled exports, hosted passkeys, package installation, and shared recent history need explicit backend/runtime contracts.
- The generated inline shell JavaScript remains budget-friendly, but future passes should consider moving repeated shell behavior into a vendored static module if the JS surface grows.
