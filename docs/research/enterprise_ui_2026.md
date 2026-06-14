# Enterprise UI Patterns for APG — 2026 Research

**Author:** Nyimbi Odero  
**Company:** Datacraft  
**Date:** 2026-06-15  
**Scope:** Gap analysis and roadmap for APG-generated UIs vs best-in-class enterprise platforms

---

## 1. Current APG UI Capabilities

APG generates single-file Python HTTP servers (Flask-like, stdlib only) with embedded Jinja2 templates rendered via htmx partials. The UI stack is Tailwind CSS (CDN) + htmx 2.0.4. All templates live in `compiler/templates/` and are injected verbatim into the generated `app.py` at compile time via `APG_UI_TEMPLATES`.

### What is currently implemented

| Feature | Template / Location | Notes |
|---|---|---|
| App home / dashboard | `app_index.html.j2` | Stats row (4 KPIs), entity list, capabilities, agent links |
| Entity list + create form | `entity_list.html.j2` | Sidebar create form, records table, breadcrumb |
| Full-text search | `entity_list.html.j2` | GET-param `?q=`, client-side filter rendered server-side |
| Kanban view | `kanban_view.html.j2` | Auto-detected from status/state/stage/phase fields; columns color-coded |
| Record detail — tab panels | `record_detail.html.j2` | Details / Related / Activity tabs, avatar header, status badge |
| Inline field editing | `record_detail.html.j2` | htmx `hx-get` pencil icon per field, `group-hover` visibility |
| Related lists | `record_detail.html.j2` | FK back-references, max 5 rows + "view all" |
| Activity timeline | `record_detail.html.j2` | Static "record created" entry + note composer (not yet wired) |
| Workflow wizard | Separate routes | Guided multi-step create flows per entity |
| FK dropdowns | `create_inputs` | FK fields rendered as `<select>` with live option hydration |
| Landing page | `landing.html.j2` | Per-app marketing/overview page |
| Delete with confirm | `record_detail.html.j2` | `onsubmit` confirm dialog + `expected_revision` optimistic lock |
| Status badge coloring | `record_detail.html.j2` | Green/red/yellow for known status vocabularies |
| View / List / Kanban toggle | `entity_list.html.j2` | Toggle in breadcrumb nav area |
| API JSON link | `entity_list.html.j2` | Direct link to REST endpoint |
| Topbar + global nav | `code_generator.py` (inline) | Module name, entity links, workflows link |

### Architectural constraints

- Everything generates into **one Python file** — no static asset pipeline, no bundler, no NPM.
- Tailwind via CDN limits to utility classes only; no custom component extraction unless inlined.
- htmx handles partial DOM swaps; no JS framework state management.
- Field metadata is available at compile time (field names, types, FK targets). Widget inference must happen in the compiler (`code_generator.py`) and results baked into the generated app.

---

## 2. Salesforce Lightning Design System Analysis

SLDS is the reference standard for enterprise data-dense UIs. Key patterns:

### Record page architecture
- **Highlights panel** — top bar with avatar, name, key fields (3–5), status badge, action buttons. APG has this; SLDS adds "key field" pinning where the owner can choose which 3 fields appear in the header.
- **Tab bar with lazy loading** — SLDS tabs fire separate XHR per tab (Details, Related, Activity, Chatter). APG tabs toggle `display:none`; no lazy load yet.
- **Column layout selector** — users can switch 1-col / 2-col / 3-col detail layouts. APG hard-codes 2-col.
- **Compact density vs comfortable** — SLDS ships two row-height modes toggled globally. Affects every list/form/table.

### List views
- **Pinned list views** — named saved searches per entity, user-scoped. "My Open Cases", "High Priority". APG has one unnamed search box.
- **Inline edit on list rows** — double-click a cell in the list, edit without entering detail. APG inline edit is detail-page only.
- **Column resizing and reordering** — drag column headers.
- **Row-level actions** — kebab menus (Edit, Delete, Change Owner, Clone) per row without navigating.
- **Bulk action bar** — checkbox select rows, batch-delete, batch-update field, export.
- **Sort indicators** — column header sort with asc/desc chevrons, server-side sort.
- **Column chooser** — show/hide fields from the list without changing the entity definition.

### Interaction models
- **Quick actions** — global `+` button creates any entity from anywhere; entity-level quick actions create related records pre-filled with context (e.g., "New Contact on Account X").
- **Split view** — list on left, record detail on right, no full-page navigate.
- **Keyboard shortcuts** — documented set: `n` = new record, `e` = edit, `/` = search focus, `Esc` = cancel.
- **Toast notifications** — success/error toasts that auto-dismiss. APG has an `⚠ notice` banner but no success toast.
- **Optimistic UI** — field saves appear immediate while XHR is in-flight; rollback on error.

### Component library
SLDS ships ~120 components: data tables, data trees, dueling picklists, path indicators (Kanban-lite for stages), pills, look-up (type-ahead FK search), spinners, skeletons, comboboxes, modals, popovers, tooltips, badges, progress rings, activity timeline items, feed items (Chatter).

---

## 3. Other Best Enterprise UIs — Key Differentiating Patterns

### ServiceNow
- **Record Watcher** — WebSocket-pushed live updates on the record page; if another agent edits a field the page reflects it in real time without reload.
- **Contextual sidebar** — "Related Links", "Formatter", history sidebar panel that slides in from right.
- **Form sections with collapsible panels** — long forms split into labelled accordion sections.
- **Assignment map widget** — for Location-carrying entities, map pin rendered inline in the form.
- **List decorator rules** — row background colour driven by field value (e.g., red rows for Priority=Critical). Compiler-defined.

### SAP Fiori / UI5
- **Analytical table (ALV)** — fixed headers, frozen columns, row grouping, subtotals, export to XLSX. The reference for finance/ERP density.
- **Object page layout** — header region (parallax scroll collapses to sticky bar), sections (tabs or anchor-scrolled), subsections. Closer to what APG has but with scroll-driven header collapse.
- **Flexible column layout (FCL)** — 1, 2, or 3 columns of list → master → detail, all in one viewport, navigated without full-page loads.
- **Smart controls** — `SmartTable`, `SmartForm`, `SmartFilter` auto-configure from OData metadata annotations. Equivalent to APG compiler inference but at runtime.
- **KPI tiles** — numeric headline + trend arrow + deviation percentage on the home page.
- **Draft handling** — save as draft without validation; resume from draft; conflict resolution on submit.

### Microsoft Fluent (Power Apps / Dynamics 365)
- **Copilot sidebar** — embedded AI chat pane that understands entity context ("Summarize this case", "Suggest next action"). Generative AI integrated at shell level.
- **Timeline control** — unified activity feed: emails, calls, tasks, notes, custom activities all on one chronological timeline with filters by type. APG has a stub Activity tab.
- **Business process flow bar** — horizontal stage indicator at top of record page showing where the record sits in a defined process (stages, completion %). Similar to Kanban but tied to the record detail.
- **Conditional formatting on views** — icon sets, color bands driven by field thresholds (like Excel conditional formatting but for entity lists).
- **Card gallery view** — entity list rendered as image cards, configurable primary/secondary fields.
- **Relevance search** — cross-entity full-text search with ranked results by entity type. APG search is per-entity only.

### Oracle Redwood
- **Redwood Pattern — Summarize, Act, Communicate** — every page decomposes into: data summary (KPI/chart zone) → action area → comms (notifications/comments). Structural pattern enforced by design system.
- **Conversational forms** — step-by-step question-answer flows instead of traditional forms (APG has wizard but field-group not conversation).
- **Progressive disclosure** — show only required fields; expand to see optional. APG shows all fields always.
- **Redwood charts** — inline sparklines in list cells, area charts in KPI tiles, timeline charts for time-series fields. Built on Oracle JET / D3.

---

## 4. Gap Analysis

### 4.1 Data Display Patterns

| Gap | Current State | Enterprise Norm |
|---|---|---|
| Activity / audit feed | Static "record created" stub; note textarea not wired | Real audit log: who changed what field, when, from what value to what value |
| Timeline events | None | Unified timeline: field changes, notes, related record events, file uploads |
| Summary / highlight bar | 3–5 fields in header; no user pinning | User-configurable "highlight fields" per entity |
| KPI tiles with trend | Counts only (entities, capabilities) | Delta vs prior period, trend arrow, sparkline |
| Related list column chooser | Fixed cols (all non-id, non-FK fields, max ~5) | User selects which columns appear in related list |
| Tree / hierarchy view | None | Parent-child nesting for hierarchical entities (org charts, category trees) |
| Frozen columns in table | None | First column sticky on horizontal scroll |
| Row-level color rules | None | Configurable: red if overdue, yellow if high-priority |
| Pagination | None — all records loaded | Server-side pagination with page size selector |
| Export to CSV/XLSX | None | One-click export of filtered list |

### 4.2 Interaction Patterns

| Gap | Current State | Enterprise Norm |
|---|---|---|
| Inline edit on list | Detail page only | Double-click cell in list view to edit in-place |
| Bulk select + actions | None | Checkbox column, select all, batch delete/update |
| Row-level kebab menu | None | "..." menu per row: Edit, Delete, Clone, View |
| Sort by column | None | Click column header → asc → desc → unsorted cycle |
| Keyboard shortcuts | None | `n`=new, `e`=edit, `/`=search, `Esc`=cancel, `Enter`=save |
| Toast notifications | Notice banner (warning-only) | Success/error toast with auto-dismiss + undo link |
| Drag-and-drop Kanban | Kanban is read-only; no DnD | Drag card between columns to update status field via htmx PATCH |
| Optimistic UI on field save | Blocking htmx round-trip | Instant visual update, rollback on error |
| Clone record | None | "Clone" creates prefilled create form |
| Contextual quick-create | None | "+New Contact" from Account detail prefills FK |

### 4.3 Navigation Patterns

| Gap | Current State | Enterprise Norm |
|---|---|---|
| Global search (cross-entity) | Per-entity `?q=` only | Unified search bar, results grouped by entity, ranked |
| Saved / pinned list views | None | Named filters saved per user ("My Open Tickets") |
| Recently viewed | None | Global "recents" list (last 10 records across entities) |
| Favourites / bookmarks | None | Star record to pin to sidebar |
| Breadcrumb with history | 2-level breadcrumb (app → entity → record) | Full navigation history with back/forward |
| Split view (list + detail) | Full-page navigate | Persistent list panel + detail panel side-by-side |
| Deep-link to tab | URL doesn't encode active tab | `?tab=related` in URL restores tab state |
| Command palette | None | `Cmd+K` opens fuzzy search over entities + actions |

### 4.4 Visualization

| Gap | Current State | Enterprise Norm |
|---|---|---|
| Inline sparklines | None | Trend line in list cell for time-series fields |
| KPI dashboard | Static count tiles | Numeric KPIs with %, delta, sparkline, target bar |
| Bar / line / pie charts | None | Entity-level analytics (e.g., contributions by month) |
| Map widget | None | Leaflet pin map for lat/lng entities |
| Progress rings / bars | None | % completion for tasks, loans, cycles |
| Gantt / timeline chart | None | Date-range entities (projects, cycles) rendered on Gantt |
| Heatmap calendar | None | Activity frequency (GitHub contribution graph style) |

### 4.5 Collaboration

| Gap | Current State | Enterprise Norm |
|---|---|---|
| Wired notes / comments | UI textarea present but POST not wired | Notes saved to DB, displayed in activity feed |
| @mentions | None | `@username` in comment notifies user |
| Notifications panel | None | Bell icon, unread count, notification list |
| Real-time presence | None | "John is editing this record" indicator |
| File attachments | None | Upload files, link to record, preview |
| Email-to-record | None | Inbound email creates comment/activity on matched record |

### 4.6 Mobile / Responsive

| Gap | Current State | Enterprise Norm |
|---|---|---|
| Responsive table | Table scrolls horizontally but no collapse | Priority column hide on small screen; card view below md |
| Touch-friendly targets | Small action buttons (p-1) | Min 44px tap targets on mobile |
| Swipe gestures | None | Swipe right to delete/action on mobile list |
| Bottom nav on mobile | Topbar doesn't adapt | Bottom tab bar on screens < sm |
| Offline / PWA | None | Service worker, offline-first data sync |

---

## 5. Widget Assignment Rule System Design

APG already detects one widget: entities with a `status/state/stage/phase` field get a Kanban toggle. This should be extended to a declarative rule system evaluated at compile time in `code_generator.py`.

### Rule structure

Each rule is a tuple of `(field_pattern_set, widget_type, conditions)`. The compiler evaluates all rules against the entity's field list and attaches widget metadata to the entity context dict passed to templates.

```python
# compiler/widget_rules.py

from dataclasses import dataclass, field
from typing import Callable

@dataclass
class WidgetRule:
    name: str
    # Any of these field name substrings trigger the rule
    field_patterns: set[str]
    # Widget template name
    widget: str
    # Optional extra conditions: (fields: list[dict]) -> bool
    condition: Callable[[list[dict]], bool] | None = None
    # Priority — higher wins if multiple rules match
    priority: int = 0

WIDGET_RULES: list[WidgetRule] = [
    WidgetRule("kanban",      {"status","state","stage","phase"},        "kanban_view",       priority=10),
    WidgetRule("map",         {"lat","lng","latitude","longitude"},       "map_widget",
               condition=lambda fs: _has_both(fs, {"lat","latitude"}, {"lng","longitude"}), priority=20),
    WidgetRule("chart",       {"amount","total","balance","revenue","cost","price","quantity","count"}, "trend_chart", priority=5),
    WidgetRule("progress",    {"progress","percent","completion","done_count","total_count"},  "progress_bar", priority=5),
    WidgetRule("timeline",    {"start_date","end_date","due_date","deadline","scheduled_at"}, "gantt_bar",   priority=5),
    WidgetRule("heatmap",     {"created_at","updated_at","occurred_at"},  "activity_heatmap", priority=1),
    WidgetRule("rating",      {"rating","score","stars","grade"},         "star_rating",      priority=5),
    WidgetRule("phone",       {"phone","mobile","tel"},                   "click_to_call",    priority=3),
    WidgetRule("email",       {"email"},                                  "mailto_link",      priority=3),
    WidgetRule("url",         {"url","website","link","href"},            "external_link",    priority=3),
    WidgetRule("color",       {"color","colour","hex"},                   "color_swatch",     priority=3),
    WidgetRule("image",       {"avatar","photo","image","thumbnail","picture"}, "image_preview", priority=8),
    WidgetRule("currency",    {"amount","price","cost","fee","salary","balance","revenue"}, "currency_display", priority=4),
    WidgetRule("json_viewer", {"config","metadata","settings","payload","data","extra"}, "json_viewer", priority=2),
]
```

### Compiler integration

In `_generate_entity_ui_context()` (currently within `code_generator.py` around line 4356–4377):

```python
def _detect_widgets(fields: list[dict]) -> list[str]:
    """Return list of widget names applicable to this entity's field set."""
    field_names = {f.get("name","").lower() for f in fields}
    active = []
    for rule in WIDGET_RULES:
        if rule.field_patterns & field_names:
            if rule.condition is None or rule.condition(fields):
                active.append(rule.widget)
    return sorted(active, key=lambda w: next(r.priority for r in WIDGET_RULES if r.widget == w), reverse=True)
```

The result is passed into template context as `widgets: list[str]`, and each template conditionally includes the widget partial:

```jinja
{% if 'map_widget' in widgets %}
  {% include 'widgets/map_widget.html.j2' %}
{% endif %}
```

Widget partials live in `compiler/templates/widgets/` and are injected into `APG_UI_TEMPLATES` alongside the full-page templates.

### Field-type semantic tags

Beyond name matching, APG should tag fields at compile time with semantic types. The compiler already infers FK fields (via `_id` suffix + cross-entity matching). Extend the `FieldMeta` or equivalent dict with a `semantic` key:

```
"semantic": "currency" | "datetime" | "phone" | "email" | "url" | "geo_lat" | "geo_lng"
           | "status" | "percent" | "boolean" | "text_long" | "image_url" | "json"
```

This semantic tag drives both widget selection and rendering (e.g., `currency` → right-aligned, formatted with `toLocaleString`; `datetime` → relative "3 days ago" display).

---

## 6. Prioritized Gap Closure Plan

Priority scoring: **Impact** (how much it changes perceived quality) × **Ease** (inverse of implementation complexity in single-file constraint).

### Tier 1 — Highest impact, lowest effort

1. **Drag-and-drop Kanban** (Impact: high, Effort: low)
   - SortableJS (CDN, ~16KB) gives DnD with zero config.
   - On `end` event fire `hx-patch /entities/{entity}/records/{id}` with `{status: newColumn}`.
   - Generated app already has a PATCH endpoint. Wire the JS and add `data-record-id` + `data-column` attrs to kanban cards.

2. **Column sort on list table** (Impact: high, Effort: low)
   - Add `?sort=field&dir=asc` GET params. Server-side: sort `records` list before table render.
   - Render `<th>` as `<a href="?sort=field&dir=asc">` with chevron icon.
   - Pure server-side, no JS needed.

3. **Row-level kebab menu** (Impact: high, Effort: low)
   - Replace "View →" link in table with a dropdown: View | Edit (inline) | Delete | Clone.
   - CSS-only dropdown via `group` + `group-hover` Tailwind pattern — no JS.

4. **Toast notifications** (Impact: medium-high, Effort: very low)
   - 15 lines of vanilla JS: create div, animate in, auto-dismiss after 3s.
   - htmx `HX-Trigger` response header fires `showToast` event. Server sets `HX-Trigger: {"showToast":{"message":"Saved","type":"success"}}`.

5. **Pagination** (Impact: high, Effort: low)
   - Add `?page=N&per=25` params. Slice records list server-side.
   - Render prev/next + page count in list template footer.

6. **Export to CSV** (Impact: high, Effort: very low)
   - `/entities/{entity}/records.csv` endpoint. Pure stdlib `csv` module. Already have the record list.
   - Add download link to entity list header.

7. **Wired note saving** (Impact: medium, Effort: low)
   - Notes textarea in Activity tab needs a POST handler: `/ui/entities/{e}/{id}/notes`.
   - Store in an `_notes` in-memory dict (same pattern as records). Render notes in timeline on GET.

8. **Recently viewed** (Impact: medium, Effort: low)
   - JS `localStorage` — push record ID + title on every detail page load. Max 10 entries.
   - Read in topbar JS, render "Recent" dropdown. Zero server-side changes.

### Tier 2 — Medium effort, high strategic value

9. **Progress / currency / phone / email semantic renderers** (Impact: medium, Effort: low-medium)
   - Compile-time semantic tag detection (see §5).
   - Template conditionals: `{% if field.semantic == 'currency' %}` format with commas + symbol.
   - Phone: `<a href="tel:{{ val }}">{{ val }}</a>`. Email: `<a href="mailto:{{ val }}">`. URL: external link icon.

10. **Map widget** (Impact: high for geo entities, Effort: medium)
    - Leaflet.js CDN (~40KB). Widget partial renders a `<div id="apg-map">` with `L.marker([lat,lng])`.
    - Detect at compile time: entity has both `lat`/`latitude` and `lng`/`longitude` fields.
    - Render map in record detail sidebar or as a full-width panel.

11. **KPI chart / trend line** (Impact: high, Effort: medium)
    - Chart.js CDN (~60KB, tree-shakeable) or lighter uPlot (~40KB).
    - For entities with `amount`+`created_at`: generate a "Total over time" endpoint `GET /entities/{e}/chart-data` returning `[{date, value}]`.
    - Render `<canvas>` in a "Charts" tab on record list page (not detail).

12. **Bulk select + delete** (Impact: high, Effort: medium)
    - Checkbox column in list table. "Select all" header checkbox.
    - Bulk action bar slides in at top when any checked (CSS show/hide).
    - POST to `/ui/entities/{e}/bulk-delete` with `ids[]` form array. Server iterates and deletes.

13. **Inline edit on list rows** (Impact: high, Effort: medium)
    - `hx-trigger="dblclick"` on `<td>` fires htmx GET for inline edit input.
    - Same pattern as record detail field editing, applied to list table cells.

14. **Global search** (Impact: high, Effort: medium)
    - Topbar search input. POST to `/ui/search?q=term`.
    - Server iterates all entity stores, full-text matches, returns grouped results.
    - Rendered as grouped list: "Accounts (3)", "Contacts (1)".

15. **Business process flow bar** (Impact: high for workflow-heavy apps, Effort: medium)
    - For entities with status field + known status vocabulary, render a horizontal progress bar showing current stage.
    - Compiler generates ordered status list from field constraints or semantic model.

16. **Saved list views** (Impact: high, Effort: medium)
    - `localStorage` key `apg_views_{entity}` stores `{name, q, sort, dir}` objects.
    - "Save this view" button. Dropdown of saved views replaces the search bar label.

### Tier 3 — Strategic, weeks-to-months

17. **Split view (FCL)** — three-panel layout: entity list / record list / record detail. Requires CSS grid restructure of the shell.

18. **Real-time WebSocket updates** — for ServiceNow-style record watcher. Add a `/ws` endpoint via `asyncio` + stdlib `http.server` or minimal `websockets` dep.

19. **Gantt / timeline chart** — date-range entities. Significant JS. Best deferred until Chart.js integration is stable.

20. **Activity feed with audit log** — requires storing every field change (old_value, new_value, user, timestamp) in the generated app. Audit log dict alongside the record store.

21. **@mentions + notifications** — requires user identity model. Not applicable to APG's current stateless multi-tenant model without significant auth integration.

22. **Offline / PWA** — service worker injection at compile time is feasible but adds significant complexity. Low priority until core gaps are closed.

---

## 7. Implementation Roadmap

### Quick wins — days (PR-sized)

| # | Feature | Template / Code location | Approach |
|---|---|---|---|
| QW-1 | Column sort | `entity_list.html.j2` + list route handler | `?sort=&dir=` params, sort records list, `<th>` as anchor |
| QW-2 | Row kebab menu | `entity_list.html.j2` (records_table builder) | CSS-only `group-hover` dropdown: View, Delete, Clone |
| QW-3 | Toast notifications | Inline JS in `_html_page()` shell + `HX-Trigger` header | 15-line vanilla JS toast + server header |
| QW-4 | CSV export | New route `/entities/{e}/records.csv` | stdlib `csv`, add download link to list |
| QW-5 | Pagination | `entity_list.html.j2` + list route | `?page=&per=` slicing, prev/next footer |
| QW-6 | Wired notes POST | New route `/ui/entities/{e}/{id}/notes` | In-memory `_notes` dict, render in activity tab |
| QW-7 | Recently viewed | `_html_page()` shell JS | `localStorage`, render in topbar dropdown |
| QW-8 | DnD Kanban | `kanban_view.html.j2` + PATCH endpoint | SortableJS CDN, fire htmx PATCH on drag end |
| QW-9 | Semantic field rendering | `record_detail.html.j2` + compile-time tag | Currency/phone/email/url/datetime rendering |
| QW-10 | Progress bar widget | `widgets/progress_bar.html.j2` + widget rule | Detect progress/percent field → inline HTML progress element |

### Medium term — weeks (multi-PR features)

| # | Feature | Approach |
|---|---|---|
| MT-1 | Widget rule system | `compiler/widget_rules.py`, integrate into `code_generator.py` entity context build |
| MT-2 | Map widget | `widgets/map_widget.html.j2`, Leaflet CDN, geo field detection |
| MT-3 | Chart/trend widget | `widgets/trend_chart.html.j2`, Chart.js CDN, `amount`+`date` detection, chart-data endpoint |
| MT-4 | Bulk select + actions | Checkbox column, bulk action bar, bulk-delete POST endpoint |
| MT-5 | Inline list edit | `hx-trigger="dblclick"` on list cells, same edit-field partial as detail page |
| MT-6 | Global search | `/ui/search` endpoint iterating all entity stores, grouped results template |
| MT-7 | Saved list views | `localStorage` view storage, save/load UI in entity list header |
| MT-8 | Business process bar | Horizontal stage indicator in record detail header for status-field entities |
| MT-9 | Audit log | Per-record change log dict `{field, old, new, ts}`, render in Activity tab timeline |
| MT-10 | Column chooser | Per-entity field visibility stored in `localStorage`, toggle in list header |

### Strategic — months (architectural changes)

| # | Feature | Key decisions |
|---|---|---|
| S-1 | Flexible column layout (FCL) | Shell restructure; list pane + detail pane side-by-side; URL-driven state |
| S-2 | Command palette | `Cmd+K` modal, fuzzy search entities + actions, keyboard navigation of results |
| S-3 | WebSocket live updates | `asyncio` WS server in generated app; field-change events pushed to open detail pages |
| S-4 | Gantt chart | Heavy JS; evaluate Frappe Gantt (MIT) vs custom canvas; only for date-range entities |
| S-5 | PWA / offline | Service worker injected at compile time; IndexedDB local cache; sync on reconnect |
| S-6 | Copilot sidebar | Per-record AI chat pane calling an LLM endpoint; requires `LLM_URL` env var in generated app |
| S-7 | Notification system | In-app notification store, bell icon, cross-entity event bus within generated app |

---

## 8. Compiler Implementation Notes

### How to add a widget partial

1. Create `compiler/templates/widgets/<name>.html.j2`.
2. The `_load_ui_templates()` function in `code_generator.py` recursively loads all `.j2` files from `compiler/templates/` — widget partials are automatically picked up if placed in a `widgets/` subdirectory (update the glob in `_load_ui_templates` to be recursive).
3. In the entity context dict passed to `_render_template()`, include `widgets: list[str]` from `_detect_widgets(fields)`.
4. In `record_detail.html.j2`, add include blocks for each widget after the Details panel.

### Template rendering pipeline

Generated app embeds all templates in `APG_UI_TEMPLATES: dict[str, str]`. At runtime, the generated app does:

```python
env = jinja2.Environment(loader=jinja2.DictLoader(APG_UI_TEMPLATES))
tmpl = env.get_template("record_detail.html.j2")
body = tmpl.render(**ctx)
```

Jinja2 `{% include %}` works with `DictLoader` — widget partials referenced as `{% include 'widgets/map_widget.html.j2' %}` resolve correctly as long as the key exists in `APG_UI_TEMPLATES`.

### Avoiding CDN bloat

Current CDN load: Tailwind (~350KB unminified via CDN), htmx (~60KB). Adding SortableJS (+16KB), Chart.js (+200KB), Leaflet (+140KB) is reasonable for an enterprise tool. Each additional CDN script should be conditional: only injected into `_html_page()` when the generated app actually has entities requiring that widget. The compiler knows this at code-gen time.

```python
# In code_generator.py — _html_page() template string builder
extra_scripts = []
if any_entity_has_geo:
    extra_scripts.append('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">')
    extra_scripts.append('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>')
if any_entity_has_kanban:
    extra_scripts.append('<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js"></script>')
if any_entity_has_chart:
    extra_scripts.append('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>')
```

---

## Open Questions

1. **Persistent storage for notes/audit log** — current generated apps are in-memory only. Notes and audit log need a durable store. Options: (a) SQLite via stdlib `sqlite3` in the generated app, (b) flat JSON file, (c) PostgreSQL as per `CLAUDE.local.md`. SQLite is the pragmatic single-file answer.

2. **User identity for collaboration features** — @mentions and per-user saved views require a user model. APG currently has no auth layer in generated apps. Should collaboration features degrade gracefully when no user session exists?

3. **Semantic model annotations vs field name heuristics** — the `semantic_model.json` per capability (e.g., in `chama/`) likely contains richer field metadata than name-matching heuristics. Should widget detection read semantic model annotations at compile time instead of / in addition to name patterns?

4. **SortableJS licensing** — MIT, safe for commercial use. Chart.js MIT. Leaflet BSD-2. All clear.

5. **Which charts library** — Chart.js is the most familiar but 200KB. uPlot (MIT, 40KB) is faster for time-series. Recharts (React-only, irrelevant). ECharts (Apache 2, 900KB, too heavy). Recommendation: Chart.js for first integration due to ecosystem; revisit uPlot for performance-sensitive apps.

---

*End of research document. Implementation should begin with Tier 1 Quick Wins (QW-1 through QW-10) in sequence, each as an independently testable PR against the APG compiler.*
