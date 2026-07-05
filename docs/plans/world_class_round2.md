# Round 2 — Better-Than-World-Class Pass (Codex Brief)

**Date:** 2026-07-05. **Executor:** Codex (all code + research). **Reviewer:** Claude (decisions only).
**Prior work:** WP0–WP7 (`008d2011`), 14-workspace excellence pass (`b19b4a57`…`63dcc912`), test fix (`0d5c93ec`), docs rewrite (`1f478f08`). All merged and pushed. Baseline: defect-free, 1474 tests passing, zero CDNs, 20 examples regenerated.

## Mission

Round 1 closed defects. Round 2 closes the **delight gap**: make every workspace not just functional but demonstrably **10× better than the Gartner MQ leader** for that surface. Each workspace gets a deep-research-backed enhancement set, implemented, verified, committed, pushed — one workspace per commit.

## What "better than world class" means here

For each surface, identify the commercial best-in-class reference, then ship **at least 3 differentiators** that the reference does not have or does worse — with the reasoning grounded in research, not opinion. Examples of the class of differentiator expected:

- **Entity list/table:** saved views + URL-shareable state, column resize/reorder persisted per user, fuzzy filter with keyboard `cmdk`-style palette, virtualized rows for >10k records, bulk-action bar, export-to-CSV/XLSX offline.
- **Forms:** autosave draft to `localStorage` + restore prompt, field-level async validation, smart defaults from sibling records, undoable submit, conditional-field logic visible in a dependency tree.
- **Kanban:** WIP-limit enforcement with visual warning, cross-tenant drag conflict resolution, swimlane grouping by any field, cumulative-flow diagram widget.
- **Record detail:** timeline with diff view per change, "mentioned" @-notify, related-records graph mini-map, one-click "create sibling" carrying context.
- **Workflow wizard:** live step-duration estimate from historical runs, branch/parallel steps visible, rollback-to-step, save-as-template.
- **Agent console:** streaming with token-per-second + cost meter, conversation branching/forks, tool-call inspector, prompt-library with versioning, side-by-side compare two runs.
- **Dashboards:** user-composable tiles with drag layout persisted, threshold alerts inline, annotation pins on charts, scheduled-export (offline PNG/CSV).
- **Auth:** passkey/WebAuthn option, magic-link, session device list + revoke, graceful locked-out flow.
- **Shell chrome:** command palette as primary nav (`cmdk`), cross-workspace recent-items, in-app onboarding tour, toast undo stack.

These are illustrative — Codex researches each surface's actual best-in-class and proposes the differentiators specific to it.

## Workspaces (one commit each, in order)

1. Home dashboard — composable tiles, threshold alerts, annotations, scheduled export
2. Entity list — saved views, column persistence, fuzzy filter, virtualization, bulk bar, offline export
3. Entity analytics — annotation pins, comparative overlays, forecast band
4. Kanban — WIP limits, swimlanes, cumulative-flow
5. Record detail — change-diff timeline, related graph, create-sibling
6. Forms — autosave draft, async field validation, smart defaults, undoable submit
7. Workflow wizard — duration estimates, rollback, save-as-template
8. Agent console — streaming meter, branching, tool inspector, prompt library, run compare
9. Capability console — rule test-bench, dry-run diff, approval SLA countdown
10. Database catalog — schema diff, ER mini-map, query playground
11. Flow debugger — step replay, breakpoint, variable inspector
12. Login/auth — passkey, magic-link, device session list, lockout flow
13. Landing + marketplace — capability compare, live demo boot, install proof
14. Shell chrome — `cmdk` palette as primary nav, recent items, onboarding tour, undo toasts

## Per-workspace protocol (Codex owns all steps)

1. **Research** → `docs/research/generated-ui-round2/<workspace-slug>/` with `README.md` (findings + differentiator proposal), `thinking.md` (raw reasoning), `sources.md` (every URL + title + date), `rationale.md` (why these differentiators, what was rejected), `assets/` (downloads). Identify the commercial leader for the surface; enumerate its weaknesses; propose ≥3 differentiators.
2. **Plan** → prioritized enhancement list in the workspace's `README.md`.
3. **Implement** — ground rules from `docs/plans/ui_remediation_workplan.md` §Ground Rules apply verbatim (baselines regeneration after any compiler/template change, 0 test failures before commit, tripwires, no CDNs, JS/CSS budgets, tabs/async/typing, no SPA). Generated `requirements.txt` may not gain new deps unless the research justifies it and the dep is vendored offline.
4. **Verify** — re-exercise workflow; add/extend tests covering the new behavior.
5. **Commit** `ux-r2(<workspace-slug>): <summary>`; push after every 2–3 workspaces.

## Cross-cutting acceptance

- Every differentiator has a research citation and a test.
- No new CDN/external runtime dependency (offline gate stays green).
- Total JS still ≤120 KB gzip; `apg-ui.js` ≤24 KB unless a measured, justified exception is documented in `rationale.md`.
- All 20 examples regenerated if compiler/templates touched.
- `uv run pytest tests/ -q` → 0 failures before each commit.

## Reporting

After each workspace: differentiators shipped, files, test tail, commit hash, research subfolder path. Stop only on a genuine design decision Claude must make; otherwise proceed through all 14. If a job exceeds ~2h, kill and restart with `--resume`.

## Final deliverable

`docs/research/generated-ui-round2/SUMMARY.md` — per-workspace: leader named, differentiators shipped, before/after verdict, residual gaps. Plus a cross-workspace "delight ledger".