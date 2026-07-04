# Raw Reasoning

The differentiator for APG is not a table page; it is generated operational surfaces for agents, teams, workflows, and capabilities. The agent console therefore needs to feel like a real command workspace, not a debug form.

The best references share a common structure: conversation first, structured details second, traceability always available. Chat products center the prompt and response. Observability products center run metadata and drill-in. APG needs both, but the initial generated console had raw description JSON in the side panel and a sparse form as the main event.

The hardest defect was not visual: the team route was dead. `SupportCrew` appeared in `ENTITIES` and in the app route declarations, but `AI_TEAM_DATA` was empty in the generated `ai_agents.py`. Rather than rewrite parser/team extraction in this workspace, the narrower remediation is to derive a team description from `ENTITIES` when the runtime catalog has no team. That makes the advertised route functional and keeps a future parser fix additive.

The same issue appears in in-memory tests where `app.py` is executed without sidecar imports. Agent descriptions can be derived from `SEMANTIC_MODEL["agents"]`, so the UI can still render without importing `ai_agents.py`.

Rejected: making raw JSON the main output. It is important for developers and debugging, but it should be behind disclosure controls so operators can read the conversation first.

Rejected: adding a new frontend dependency or chat library. The generated app budget and plan require server-rendered HTML plus vanilla JS/SSE, and the existing `apg-sse.js` already provides the progressive enhancement hook.
