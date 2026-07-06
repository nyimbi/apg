# Shell Chrome Raw Reasoning

The current APG shell has a topbar, sidebar, notifications, theme toggle, install/update buttons, and a basic command palette that fetches record search results. Round 2 should convert that from "search accessory" into a generated command center.

Raycast is the best leader because it proves the command palette can be the primary navigation model. The generated APG app should not try to reproduce Raycast extensions, but it can make all known generated app surfaces available from one keyboard-first panel.

Recent items matter because generated APG apps have many workspaces: dashboard, entities, workflows, agents, databases, marketplace, debug, OpenAPI. A local recent list creates continuity across those surfaces without backend state.

The onboarding tour should be shell-level, not page-level, because the shell features are global. The undo toast should demonstrate reversible chrome actions without inventing destructive operations.

Rejected: adding cmdk or another JS package. Vanilla JavaScript is sufficient and keeps the JS budget stable. Rejected: server-side recent item storage. That would require user identity/storage contracts beyond generated UI. Rejected: full guided product tour dependency. A small generated overlay is enough.
