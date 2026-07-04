# Rationale

## Decisions

- Derive missing agent-team descriptions from `ENTITIES` when `AI_AGENTS.list_agent_teams()` returns no team. This repairs generated team routes without changing parser scope.
- Derive missing agent descriptions from `SEMANTIC_MODEL["agents"]` for in-memory generated-app execution.
- For missing runtime team invocation, return a compatible team result from entity metadata instead of a dead 404.
- Keep raw description and response JSON, but hide them behind details panels.
- Preserve the submitted message and payload so invocation results read as a conversation.
- Display team members and handoff flow in dedicated side panels.

## Rejected Alternatives

- Parser-level team extraction repair: likely correct long-term, but broader than this workspace's console scope and riskier than a generated UI fallback.
- New chat UI dependency: rejected because generated apps must stay self-contained and within JS/CSS budgets.
- Removing raw JSON entirely: rejected because APG operators and developers need inspectable payloads.
- Treating team routes as unsupported until runtime teams materialize: rejected because the generated UI already advertises those routes.
