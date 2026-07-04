# Generated-UI Remediation Workplan (Codex Handoff Brief)

**Status:** Approved for implementation
**Date:** 2026-07-04
**Parent plan:** `docs/plans/world_class_generated_ui.md` (read it first — contains the full audit, architecture decisions AD-1…AD-6, and file:line references)
**Executor:** Codex (via rescue runtime), work packages executed **in order**, one commit per work package.

---

## Ground Rules (non-negotiable)

1. **Repo:** `/Users/nyimbiodero/src/pjs/apg`. Main branch. Commit and push after each completed work package.
2. **Baseline gate:** `compiler/baseline.py:678` string-compares `examples/NN_*/output/**` against fresh generation. **Every change to `compiler/code_generator.py`, `compiler/templates/**`, or generated assets requires regenerating all 20 numbered examples** before tests pass:
   ```bash
   for d in examples/[0-9][0-9]_*/; do uv run apg compile "$d"main.apg -o "$d"output; done
   ```
   Commit regenerated outputs together with the change that caused them.
3. **Tests:** `uv run pytest tests/ -q` must be 0 failures before each commit. Slow suites (`test_tooling_audit.py`, `test_security_integration.py`) may be run once at the end of each work package rather than every iteration. The baseline test (`test_cli_baseline_json_audits_numbered_examples`) takes ~130s.
4. **Code standards:** async Python, tabs (not spaces), modern typing (`str | None`, `list[str]`). Generated `app.py` must stay stdlib+Flask+Jinja2 only — no new Python deps in generated `requirements.txt`.
5. **Existing test tripwires to respect:**
   - `tests/test_code_generator_executable_defaults.py::test_python_code_generator_does_not_use_appbuilder_ui_framework` — `PythonCodeGenerator` source must never contain `AppBuilder`, `appbuilder`, `SQLAlchemy`, `sqlalchemy`, `Pydantic`, `pydantic`, `ModelView`, `BaseView` (even in comments/strings).
   - Generated Python must contain no `TODO: Implement`, no bare `pass` lines, and must `compile()`.
6. **No SPA frameworks.** htmx + vanilla JS modules only. JS budget ≤ 24 KB min+gzip for `apg-ui.js`; total JS ≤ 120 KB gzip including vendored libs.
7. **Do not touch** `.omc/`, `capabilities/` (except if a test forces it), or unrelated files.

---

## Issue → Work Package Map

| # | Identified issue | Work package |
|---|---|---|
| 1 | CDN-dependent assets (Tailwind Play, unpkg htmx, jsDelivr Sortable) | WP0 |
| 2 | Two-tier page quality — 6 f-string-only pages | WP1 |
| 3 | Zero charts/dashboards | WP3 |
| 4 | No auth UI | WP5 |
| 5 | No real-time/streaming UI | WP4 |
| 6 | i18n modeled but never surfaced (`lang="en"` hardcoded) | WP6 |
| 7 | Weak mobile (no hamburger, sidebar-form scroll, no PWA) | WP2 + WP7 |
| 8 | Accessibility gaps (skip link, aria-current, focus, window.confirm) | WP1 + WP2 |

---

## WP0 — Self-Contained Asset Pipeline (kills issue 1)

**Goal:** zero external network requests from any generated app.

Tasks:
1. Create `compiler/assets/` with vendored, pinned files:
   - `htmx.min.js` (2.0.4 — same version currently CDN-loaded at `code_generator.py:3641`)
   - `sortable.min.js` (1.15.3 — currently :3643)
   - `apg.css` — see task 2
   - `LICENSES.md` noting each library's license (htmx BSD-2, SortableJS MIT).
   Download once from the official releases; commit the files. If network is unavailable in the runtime, stop and report — do not stub.
2. **Replace Tailwind Play CDN with a static stylesheet.** The template set is closed, so:
   - Extract every utility class used across `compiler/templates/**/*.j2` and all HTML-emitting f-strings in `code_generator.py`.
   - Produce `compiler/assets/apg.css` containing: (a) a `:root` token block (colors incl. `apg-primary #1E5B5A` / `apg-accent #D97706`, spacing, radius, shadows, font stacks), (b) the existing `.apg-*` component classes currently emitted by `theme_stylesheet()` (:3526), (c) definitions for exactly the utility classes the templates use (grep-verified closed set), with dark-mode variants driven by `.dark` OR `[data-theme="dark"]` on `<html>` plus `prefers-color-scheme` fallback.
   - Preferred: generate it with the Tailwind standalone CLI via a checked-in script `scripts/build_apg_css.sh` (pin the CLI version). If the CLI can't be fetched, hand-write the closed utility set — it is enumerable and finite.
   - Add `tests/test_generated_ui_assets.py::test_apg_css_covers_all_template_classes` — parse class attributes from templates + generated HTML builders, assert every class has a definition in `apg.css`.
3. Emit assets into generated output: extend `PythonCodeGenerator.generate()` so its returned dict includes `static/apg.css`, `static/htmx.min.js`, `static/sortable.min.js` (contents read from `compiler/assets/` at compile time). Verify every writer of that dict (CLI output writer, `baseline.py:646` refresh, `baseline.py:678` compare) handles the new paths (all are text — no bytes support needed yet).
4. Update `_html_page()` (:3636–3643): remove all `https://` asset references; link `/static/apg.css`, `/static/htmx.min.js`, `/static/sortable.min.js`. Keep `/theme.css` but slim `theme_stylesheet()` (:3526) to tokens + per-capability overrides only (component classes now live in `apg.css`).
5. Add in-app theme toggle: small inline script sets `data-theme` on `<html>` from `localStorage` (`light|dark|system`), toggle button in the topbar. Must be render-blocking-safe (inline in `<head>` before CSS to avoid flash).
6. Offline gate test: `tests/test_generated_ui_assets.py::test_no_external_urls_in_generated_output` — compile example 20 in-memory, assert no `cdn.`, `unpkg.com`, `jsdelivr.net`, `googleapis.com`, `http://`, `https://` in any generated HTML/CSS/JS (allow `https://` only inside code comments/LICENSES).
7. Regenerate all 20 examples; full test suite; commit (`feat: self-contained static asset pipeline for generated apps`); push.

**Acceptance:** offline-gate test green; visual parity (pages still styled); dark toggle works; 0 test failures.

---

## WP1 — Template Unification + A11y Baseline (kills issue 2, half of 8)

**Goal:** every `/ui/*` page renders from a Jinja2 template with consistent shell; core a11y landmarks everywhere.

Tasks:
1. Create templates (match the visual language of `entity_list.html.j2` / `record_detail.html.j2` — cards, `.apg-*` classes, badges):
   - `workflow_list.html.j2` — replaces f-string in `_ui_workflow_list_html()` (:4068). Card grid: workflow name, owning entity, step count, "Start" button.
   - `workflow_wizard.html.j2` — replaces :4109. Numbered stepper (completed=check, current=filled, future=outline), field errors under fields, back/cancel, review summary on final step.
   - `database_catalog.html.j2` — replaces :4374. Per-database card with tables → columns/types listed; relationship list.
   - `agent_console.html.j2` — replaces :5411 (both agent and team mode via a flag). Structural v1: chat-style layout — prompt input at bottom, conversation area above, collapsible "raw request/response JSON" `<details>` panels instead of bare `<pre>`. (Streaming comes in WP4 — do the layout now.)
   - `capability_console.html.j2` — replaces :5435. Three panels (rules evaluate / configuration resolve / approval plan) as proper labelled forms; results rendered as definition-list cards with a collapsible raw-JSON details.
   - `debug_console.html.j2` — replaces `_ui_debug_html()` (:5199). Run list + per-run step timeline (ordered list with status badges and durations).
2. Wire each into its renderer using the established pattern (`_render_template(...) → if not None → _html_page(...)`), then **delete the per-page f-string HTML builders**, leaving one shared minimal fallback page ("This application requires Jinja2 — pip install -r requirements.txt"). Add `tests/test_generated_ui_templates.py` asserting every `/ui` route's renderer resolves a template from `APG_UI_TEMPLATES`.
3. A11y baseline in `_html_page()` and templates:
   - Skip-to-content link as first focusable element.
   - `aria-current="page"` on active topnav links; `<main id="content">` landmark; one `<h1>` per page.
   - Replace `window.confirm()` deletes with an accessible confirm dialog component (focus-trapped `<dialog>`, Escape closes, destructive button styled `danger`) in shared inline JS.
   - Visible focus rings (`:focus-visible`) on all interactive elements in `apg.css`.
4. Regenerate 20 examples; full suite; commit; push.

**Acceptance:** grep of `code_generator.py` finds no page-level f-string HTML builders for the six pages; all routes template-rendered; keyboard-only delete flow works.

---

## WP2 — App Shell v2 + Mobile Navigation (kills half of 7 and 8)

Tasks:
1. Shell: collapsible left sidebar (entities grouped by capability, Workflows, Databases, Agents, Dashboard link) + topbar (module name, palette trigger, theme toggle). Sidebar collapses to hamburger drawer < 768px (focus-trapped, Escape closes). Persist collapse state in `localStorage`.
2. Breadcrumbs partial (`widgets/breadcrumbs.html.j2`) on every non-index page.
3. Entity-list create form (currently squeezed sidebar, `_ui_create_form_html()`:4632) moves into a drawer opened by a "New <Entity>" primary button — full-width form fields, focus-trapped, unsaved-changes guard.
4. Touch targets ≥ 44px for primary controls; tables get `overflow-x:auto` wrappers (already `.apg-table-wrap` — verify) and a stacked card view under 480px for the first 3 columns.
5. Regenerate; test; commit; push.

**Acceptance:** example 20 fully operable at 375px width with no horizontal body scroll; hamburger nav keyboard-operable.

---

## WP3 — Dashboards & Charts (kills issue 3)

Tasks:
1. Vendor `uplot.min.js` + `uplot.min.css` (1.6.x, MIT) into `compiler/assets/` → `static/`; add to LICENSES.md.
2. Create `static/apg-charts.js` (part of the ≤24 KB budget or separate ≤8 KB): hydrates `<div data-apg-chart>` elements from adjacent `<script type="application/json">` specs. Chart types: line, area, bar, sparkline (uPlot); donut, progress (inline SVG, no lib). Colors from CSS custom properties; re-render on theme change. Every chart emits a visually-hidden data table + `<details>` fallback.
3. Home dashboard v2 in `app_index.html.j2`: stat tiles gain 30-day sparkline + delta (computed server-side from record `created_at`-like fields where present; omit gracefully when absent); one donut per entity that has a `status`-semantic field; recent-activity feed card; workflow + agent summary tiles.
4. Per-entity analytics: `GET /ui/entities/ENTITY?view=analytics` tab alongside table/kanban — records-over-time line, group-by-status bar, numeric field min/avg/max stat row. Server computes specs in `_ui_entity_analytics_html()` (new), template `entity_analytics.html.j2`.
5. Empty states: every chart/tile with no data renders the empty-state component (SVG illustration + CTA), never a blank box.
6. Regenerate; test (add `tests/test_generated_ui_dashboard.py` — compile example 20, assert chart spec JSON present and valid for entities with status fields); commit; push.

**Acceptance:** examples 18 and 20 open onto a real dashboard; charts render offline; empty-data entities show empty states not errors.

---

## WP4 — Live UI: SSE + Streaming Agent Console (kills issue 5)

Tasks:
1. `/events` SSE endpoint in generated `app.py`: stdlib/Flask streaming response (`text/event-stream`), topic param (`?topics=agent:NAME,workflow:run:ID`), heartbeat comment every 15s, in-process pub/sub registry (thread-safe queue per subscriber; generated apps are single-process).
2. `static/apg-sse.js`: `EventSource` wrapper with auto-reconnect/backoff and a `data-apg-live` binding for tiles/tables ("N new records — refresh" pill; no full reload).
3. Agent console streaming: agent invoke route (`_ui_agent_console_html` POST path :5487) gains streaming mode — when the underlying runtime supports it (Ollama `stream:true`), tokens are published to `agent:NAME` topic and appended incrementally in the chat UI with a streaming cursor and a Stop button; non-streaming runtimes fall back to the existing request/response with a typing indicator. Sanitize all rendered output (escape HTML; simple markdown subset renderer for bold/code/lists — no raw HTML injection).
4. Workflow wizard/run progress: step transitions publish to `workflow:run:ID`; the stepper updates live when a run page is open.
5. Regenerate; test (SSE endpoint unit test with Flask test client reading first events; sanitizer tests with hostile input); commit; push.

**Acceptance:** agent invocation streams visibly in example 06/07; SSE survives reconnect; hostile agent output renders inert.

---

## WP5 — Authentication UX (kills issue 4)

Tasks:
1. When the module declares auth (JWT config already supported ~:1992–2013): generate `login.html.j2` (centered card, module name, error states), `POST /login` issuing the session/JWT against the app's configured credential source, `POST /logout`, and redirect-to-login middleware for `/ui/*` when unauthenticated. When no auth is declared, generate nothing — apps stay open (backward compatible; baseline examples without auth must be byte-identical except shared-template dict content).
2. Topbar user menu (initials avatar, logout) rendered only when auth active; friendly 403 page for permission failures.
3. Regenerate; test (auth-enabled fixture module: login → cookie/JWT → gated page → logout; unauthenticated redirect); commit; push.

**Acceptance:** an auth-declared test module demonstrates the full loop; auth-less examples unchanged in behavior.

---

## WP6 — i18n Surfacing (kills issue 6)

Tasks:
1. Chrome-string catalog: extract all UI literals from templates into a `_()` lookup backed by `APG_I18N: dict[lang, dict[key, str]]` embedded at compile time. Seed `en` always; when the DSL declares `i18n.supported_languages`, emit bundles for each (fallback to `en` for missing keys — machine translation is a later pass; generate the keys now).
2. `lang` attribute from active locale (fix hardcoded `en` at :3683); locale from cookie → `Accept-Language` → default. Language switcher in topbar when >1 language.
3. Locale-aware formatting in templates: dates, numbers, currency (currency semantic finally gets a symbol) via a compile-time-emitted format helper using CLDR-lite data for declared locales only (keep it tiny — pattern strings per locale, no Babel dependency).
4. RTL: `dir="rtl"` for `ar`, `he`, `fa`, `ur`; audit `apg.css` for physical properties → logical (`margin-inline-*`, `padding-inline-*`, `inset-inline-*`).
5. Regenerate (example 10 is the showcase — verify its declared languages produce bundles); test; commit; push.

**Acceptance:** example 10 renders a working language switcher; switching to `sw` changes chrome strings (en-fallback where untranslated); an RTL locale mirrors the layout.

---

## WP7 — PWA + Quality Gates (kills the rest of 7; locks everything in)

Tasks:
1. PWA: generated `static/manifest.webmanifest` (name/colors from theme, generated icon), `static/sw.js` caching static assets + last-viewed pages (read-only offline), offline banner component. `theme-color` meta.
2. Playwright harness `tests/ui/` (dev-dep only, not in generated apps): compile examples 01, 10, 20 → boot each on an ephemeral port → for every route: axe-core scan (fail CI on critical/serious), screenshot at 375/768/1440 × light/dark into `tests/ui/__screenshots__/` goldens.
3. Budgets test: assert gzip sizes — `apg.css` ≤ 60 KB, total JS ≤ 120 KB.
4. `apg baseline --refresh` CLI flag wrapping regeneration of all 20 examples (wraps `refresh_outputs=True`); document the workflow in `docs/generated_ui.md` (also document the component library and theming tokens).
5. Keyboard-only Playwright smoke: palette → navigate → create → inline edit → delete+undo.
6. Regenerate; full suite incl. slow audits; commit; push.

**Acceptance:** CI enforces a11y + budgets + offline; Lighthouse on example 20 ≥ 95 performance/accessibility (record scores in the doc).

---

## Reporting Protocol (per work package)

After each WP, report: files changed, test summary (`uv run pytest tests/ -q` tail), baseline regeneration confirmation, commit hash, any deviations from this brief with rationale. If blocked (e.g. no network to vendor an asset, a tripwire test conflicts with a required change), stop that WP and report rather than working around silently.
