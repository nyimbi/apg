# Generated UI Reference

APG generated applications ship a self-contained Flask UI. The compiler embeds
Jinja templates from `compiler/templates/` and copies browser assets from
`compiler/assets/` into each generated app's `static/` directory.

Generated pages should not depend on CDN-hosted assets.

## Asset Contract

Generated `static/` includes:

```text
apg.css
htmx.min.js
sortable.min.js
uplot.min.js
uplot.min.css
apg-charts.js
apg-sse.js
manifest.webmanifest
sw.js
icon.svg
```

The app shell loads those local files through `/static/...`, plus generated
theme CSS through `/theme.css`.

## Component Library

The generated UI is assembled from these source templates:

| Template | Component family |
| --- | --- |
| `landing.html.j2` | Landing page, app entry, marketplace links, operational stats. |
| `app_index.html.j2` | Home dashboard, charts, records, agent/team/capability summaries. |
| `entity_list.html.j2` | Entity tables, filters, saved-query links, sorting, pagination, import/export, create drawer. |
| `entity_analytics.html.j2` | Entity analytics charts, status distribution, record insights. |
| `kanban_view.html.j2` | Status/stage/state/phase kanban boards with WIP indicators. |
| `record_detail.html.j2` | Record profile, fields, inline edit, related lists, activity timeline. |
| `workflow_list.html.j2` | Workflow catalog and recent runs. |
| `workflow_wizard.html.j2` | Guided workflow execution, step progress, run state, signals. |
| `agent_console.html.j2` | Agent and team invocation consoles with streaming output. |
| `capability_console.html.j2` | Capability rules, configuration, approval, health, theme, screens, streaming metadata. |
| `database_catalog.html.j2` | Database schemas, tables, columns, indexes, relationships, status. |
| `debug_console.html.j2` | Flow debugger, workflow journal, circuit breakers, subscriptions. |
| `marketplace.html.j2` | Connector marketplace cards, search, category filters, integration blueprints. |
| `login.html.j2` | Login and auth error surfaces. |
| `widgets/breadcrumbs.html.j2` | Breadcrumb fragments. |
| `widgets/field_display.html.j2` | Typed field display helpers. |

Runtime components generated directly in `compiler/code_generator.py` include:

- persistent sidebar and topbar
- command palette
- notification tray and toast system
- confirm dialog
- theme mode toggle
- i18n language selector
- install/update PWA buttons
- offline banner
- service-worker registration
- SSE event subscriptions
- HTMX mutation fragments
- CSV export
- field-level inline edit fragments

## Theme Tokens

Generated apps expose theme values as CSS custom properties through
`/theme.css`. Capability contracts and APG theme metadata can provide tokens;
the generator falls back to safe defaults.

Common token families:

| Token family | Examples |
| --- | --- |
| Brand colors | `color.primary`, `color.accent` |
| State colors | `color.success`, `color.warning`, `color.danger` |
| Surfaces | `color.surface`, `color.surface-muted`, `color.border` |
| Text | `color.text`, `color.text-muted`, `color.link` |
| Layout | shell width, sidebar width, content spacing |
| Shape | border radius and focus ring values |
| Elevation | shadow tokens for panels, drawers, dialogs |
| Typography | font stack and weight choices |

The shell also supports browser-side theme mode:

- `system`
- `light`
- `dark`

The selected mode is stored in `localStorage` as `apg-theme`.

## I18n And Direction

Generated apps can include supported language metadata. The shell catalog
includes core labels for English and selected overrides such as Swahili (`sw`)
and Arabic (`ar`). The generated HTML `lang` and `dir` values are set from the
active locale; Arabic uses RTL direction.

The generated `/locale` route stores the selected language in the `apg_lang`
cookie and returns users to the active screen.

## The 14 Workspaces

### 1. Home Dashboard

Route: `/ui`

Shows app summary cards, entity counts, record totals, workflow shortcuts,
agent/team/capability summaries, generated charts, and activity/empty-state
guidance. It is the default operational workspace after login.

### 2. Entity Lists

Route: `/ui/entities/{Entity}`

Shows records with filters, query-preserving links, saved-view style states,
sorting, pagination, export/import controls, typed columns, create actions, and
empty states. Query parameters drive filters and views.

### 3. Kanban

Route: `/ui/entities/{Entity}?view=kanban`

Available when an entity has a `status`, `state`, `stage`, or `phase` field.
Groups records into columns, shows WIP indicators, links back to filtered lists,
and preserves the same generated data model.

### 4. Record Detail

Route: `/ui/entities/{Entity}/{id}`

Shows title, status badge, field display, inline edit controls, previous/next
record navigation, related records inferred by foreign-key naming, delete
actions, revision checks, and activity timeline.

### 5. Create/Edit Drawers

Routes: entity UI POST and field fragment routes.

Generated forms use typed inputs for text, email, phone, numeric, date,
boolean, JSON/list/dict, and textarea-style fields. HTMX swaps update the UI,
validation errors remain close to the form, and `_revision` checks protect
against stale edits.

### 6. Workflow List

Route: `/ui/workflows`

Shows declared workflows, entry points, and recent runs. Links open the guided
wizard for a workflow run.

### 7. Workflow Wizard And Run Progress

Routes:

- `/ui/workflows/{Entity}/{Workflow}`
- `/ui/workflows/{Entity}/{Workflow}/step/{n}`

Supports step navigation, run creation, progress display, journals, signals,
resume, and compensation flows. Workflow run state is also exposed through API
routes and debug surfaces.

### 8. Agent Console

Route: `/ui/agents/{Agent}`

Displays agent metadata, runtime adapter hints, prompt/payload entry, invocation
results, and streaming console behavior through generated SSE helpers where
available.

### 9. Agent Team Console

Routes:

- `/ui/agent-teams/{Team}`
- `/ui/teams/{Team}`

Shows team membership, handoff/flow metadata, capability links, invocation
payloads, and team execution output.

### 10. Capability Console

Route: `/ui/capabilities/{Capability}`

Shows capability profile, declared rules, default rule contexts,
configuration resolution and validation, approval planning, health, theme
tokens, screens, languages, and streaming metadata.

### 11. Database Catalog

Route: `/ui/databases`

Shows declared and inferred database schemas, tables, columns, primary keys,
indexes, references, relationship graph data, and generated status checks.

### 12. Flow Debugger

Route: `/ui/debug` and `/ui/debug/{run_id}`

Shows workflow runs, selected run trace, hash-chained journal entries, circuit
breaker state, event subscriptions, signals, compensation status, and debug
links from workflow screens.

### 13. Login And Auth

Routes:

- `/login`
- `/logout`
- protected `/ui` redirects when auth is required

Generated auth surfaces support session login, login errors, logout, redirects
back to the requested page, and mutation authorization checks.

### 14. Landing Page And Full Shell

Routes:

- `/`
- `/home`
- `/ui`

The full shell includes sidebar, topbar, command palette, notifications, theme
toggle, i18n switcher, PWA install/update controls, offline banner, manifest,
service worker, local static assets, and keyboard-accessible navigation.

## PWA Behavior

Generated HTML includes:

- `theme-color`
- manifest link
- service-worker registration
- install prompt handling
- update-ready handling
- offline banner
- cached same-origin GET pages and generated static assets

The manifest includes shortcuts for dashboard, workflows, and marketplace.

## Validation

Use:

```bash
apg compile <source.apg> --output /tmp/apg-ui --verify
python /tmp/apg-ui/app.py --self-test
python /tmp/apg-ui/smoke_test.py
uv run pytest tests/ -q
```

Generated UI tests cover route availability, asset vendoring, CDN-free output,
PWA files, shell behavior, workspace templates, and regression cases captured
under `docs/research/generated-ui-workspaces/`.
