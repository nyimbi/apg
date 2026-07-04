# Agent And Agent-Team Consoles

Date: 2026-07-04

## Best-In-Class Patterns

- ChatGPT-style products put the conversation, composer, streaming state, and stop/retry controls in the primary workspace, with structured tool output rendered as secondary context.
- Claude-style artifact workflows separate conversational output from inspectable generated artifacts and keep raw technical detail available without making it the main reading path.
- LangSmith-style observability makes agent runs traceable: agent identity, steps, payloads, and outputs are inspectable after invocation.
- OpenAI and Agents SDK streaming guidance reinforces incremental response rendering and server-sent event style delivery for responsive UIs.

## Live Audit

Representative app: `examples/06_support_agent_team/output/app.py`.

Before server: `127.0.0.1:20895`.

Observed defects:

- `/ui/agent-teams/SupportCrew` returned `404 Unknown agent team` even though the DSL declared `SupportCrew` and the sidebar advertised agent-team routes.
- The single-agent console was form-first rather than conversation-first.
- The prompt and payload were not preserved as a visible user turn after invocation.
- Team membership and handoff flow were not visible in the console.
- Raw JSON was too prominent; it was useful but competed with the operator workflow.

After server: `127.0.0.1:20898`.

After verification:

- `/ui/agents/Planner` renders a conversation workspace with structured payload disclosure.
- `/ui/agent-teams/SupportCrew` renders successfully with team lanes and handoff flow.
- Agent and team POST flows render the user turn, response panel, and raw response disclosure.
- `/agents` includes `SupportCrew` from entity metadata when the sidecar runtime team catalog is empty.

## Fix List

Must-fix:

- Restore declared agent-team console routes by falling back to entity metadata.
- Make team invocation work when the generated runtime team catalog is missing.
- Provide semantic-model fallback for agent descriptions so in-memory generated app tests still render consoles.

High-value polish:

- Reframe the console as a conversation with an operator composer.
- Preserve submitted prompt and payload context after invocation.
- Move raw request/response details behind disclosure controls.
- Surface team members, handoff flow, tools, capabilities, runtime, and model context.

## Validation

- Regenerated all 20 numbered examples.
- Targeted tests: `2 passed in 1.61s`.
- CSS coverage: `1 passed in 0.27s`.
- Full-suite gate pending before commit.
