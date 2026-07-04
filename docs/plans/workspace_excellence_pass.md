# Workspace Excellence Pass (Codex Brief)

**Date:** 2026-07-04. **Executor:** Codex. **Reviewer:** Claude (decisions only).
**Prior work:** WP0–WP7 (`docs/plans/ui_remediation_workplan.md`) — all merged through `008d2011`.

## Mission

Systematically make every generated-app workspace **functional, complete, correct, performant, and beautiful**, with workflows optimized end-to-end. Deep research per workspace precedes changes.

## Workspaces (audit each independently)

1. Home dashboard (`/ui`, `app_index.html.j2`)
2. Entity list + table + filters + saved views (`entity_list.html.j2`)
3. Entity analytics (`entity_analytics.html.j2`)
4. Kanban (`kanban_view.html.j2`)
5. Record detail + activity + related (`record_detail.html.j2`)
6. Create/edit forms + drawer + inline edit
7. Workflow list + wizard + run progress
8. Agent & agent-team consoles (streaming chat)
9. Capability console (rules/config/approval)
10. Database catalog
11. Flow debugger
12. Login/auth surfaces
13. Landing page + marketplace
14. Shell chrome: sidebar, topbar, palette, notifications, theme, i18n switcher, PWA/offline

## Per-workspace protocol

1. **Research** (save to `docs/research/generated-ui-workspaces/<workspace-slug>/` with `README.md` findings, `thinking.md` raw reasoning, `sources.md` every URL/doc consulted with titles+dates, `rationale.md` decisions & rejected alternatives, `assets/` downloads):
   - Identify the best-in-class commercial reference for this surface (e.g. Linear/Notion tables, Airtable forms, Vercel dashboards, ChatGPT/Claude chat UIs, Temporal UI for runs, Retool consoles) and enumerate the interaction patterns that make it excellent.
   - Boot a representative example app (01/05/10/13/18/20 as appropriate), exercise the workspace end-to-end as a real user (curl + reading rendered HTML; Playwright harness in `tests/ui/` where useful), and record every defect: broken/missing functionality, dead ends, confusing flows, ugly/inconsistent styling, empty/error/loading gaps, a11y issues, performance issues, mobile issues.
2. **Plan**: prioritized fix list in the workspace's `README.md` (must-fix vs polish), workflow optimizations (fewer clicks, better defaults, keyboard paths).
3. **Implement** all must-fix + high-value polish. Ground rules from `docs/plans/ui_remediation_workplan.md` §Ground Rules apply verbatim (baselines regeneration, 0 test failures, tripwires, no CDNs, JS/CSS budgets, tabs/async/typing).
4. **Verify**: re-exercise the workflow; add/extend tests covering the fixes.
5. **Commit** per workspace: `ux(<workspace-slug>): <summary>`; push after every 2–3 workspaces.

## Cross-cutting acceptance

- Every page: purposeful (answers "why would a user come here"), complete (no dead ends/raw JSON dumps), correct (no broken links/routes/stale data), beautiful (consistent tokens, spacing, empty states), fast (<1s render locally), accessible (keyboard path, landmarks), mobile-usable at 375px.
- Workflow-level: create→view→edit→related→delete/undo loop ≤ minimal clicks; agent invoke→stream→history loop polished; workflow start→steps→run history loop coherent.
- Final deliverable: `docs/research/generated-ui-workspaces/SUMMARY.md` — per-workspace verdict (before/after), remaining known gaps, and a defect ledger with resolution status.

## Reporting

After each workspace: defects found, fixes made, files, test tail, commit hash. Stop only on unresolvable design decisions; otherwise proceed through all 14.
