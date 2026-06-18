# APG Enterprise UI Gap Closure Plan

## Research Basis
See: docs/research/enterprise-ui-gap-analysis.md

## Current APG UI State
- Entity list: search, sort, pagination, bulk select/delete/export ✅
- Record detail: tab panels (Details, Related, Activity), semantic fields ✅
- Kanban: SortableJS DnD ✅
- Toast notifications ✅
- Floating bulk action bar ✅

## Gap Analysis (Priority Order)

### CRITICAL (makes app feel non-enterprise)

#### U1: Highlights Panel (Salesforce / HubSpot pattern)
**What Salesforce does**: Every record page opens with a compact 3-6 field summary ("highlights panel") at the top of the record header — the most important fields at a glance before the user digs into tabs.
**APG current**: Jumps straight to full Details tab grid
**Plan**: Generate a highlights panel from the first 4-6 non-id, non-revision fields. Add `highlight: true` field annotation support.
**Effort**: M — modify `record_detail.html.j2` + highlights logic in code_generator

#### U2: Inline Field Editing (click-to-edit)
**What Salesforce does**: Every field in the record detail has a pencil icon on hover. Click → field becomes an input in place. Save individually or cancel. No page reload.
**APG current**: Edit pencil icon exists but triggers `hx-get` to a separate edit form
**Plan**: Full inline edit via htmx PATCH per field — the existing `/fields/{name}/edit` and `/fields/{name}/patch` routes are the right foundation; improve the returned HTML to be inline (not a full form page).
**Effort**: M — improve `_ui_field_edit_html()` response

#### U3: Command Palette (Cmd+K)
**What Linear/Notion/ServiceNow do**: Universal search + action runner. Press Cmd+K → fuzzy-search overlay shows recent items, entities, actions. Instantly navigate anywhere.
**APG current**: None
**Plan**: 
- Add `GET /api/search?q=` endpoint returning {entities, records, workflows}
- Add command palette overlay (vanilla JS, no framework needed)
- Wire Cmd+K keyboard shortcut
**Effort**: L — JS overlay + search endpoint

#### U4: Skeleton Screens
**What Carbon/Salesforce do**: While htmx loads content, show grey shimmer placeholders (not blank space or spinner)
**APG current**: No loading state
**Plan**: Add `hx-indicator` class + CSS shimmer animation. Global CSS in `_html_page()`.
**Effort**: S — pure CSS + hx-indicator attribute

#### U5: Rich Activity Feed
**What Salesforce does**: Activity timeline with typed events (created, updated, status changed, commented), user avatars, timestamps, filter by type, compose box
**APG current**: Static "Record created" stub
**Plan**:
- Add `apg_activity` in-memory store per entity record (list of events)
- Generate events on record create/update/delete/status-change
- Render typed events with icons in the Activity tab
- Add "Add Note" compose functionality (POST to store note)
**Effort**: L

### HIGH (significantly improves feel)

#### U6: Field Metadata Annotation System
**What Salesforce Lightning App Builder does**: Every field has metadata (widget type, compact layout inclusion, required, help text). The page layout builder assigns widgets based on field type + metadata.
**APG current**: Field type → widget mapping is hardcoded in `field_display.html.j2` based on `semantic` string
**Plan**: Add `field_meta` dict per field in the APG spec. Compiler passes this as `field_semantics` to templates. Add new widget types: `rich_text`, `file_upload`, `color_picker`, `date_picker`, `autocomplete`.
**Effort**: L

#### U7: Compact Layout / Two-Column Field Grid
**What SAP Fiori does**: Fields in record detail are arranged in a responsive 2-column grid. Labels left-aligned. Compact density option.
**APG current**: 2-column `md:grid-cols-2` already ✅ — needs density toggle

#### U8: Global Navigation Improvements
- **Recent items** dropdown (last 10 viewed records stored in localStorage)
- **Entity search** in sidebar/topbar
- **Keyboard shortcuts** help modal (`?` key)

#### U9: Empty State Illustrations
**What Linear/Notion do**: Delightful empty state with context-specific message and CTA
**APG current**: Basic icon + text
**Plan**: Improve empty state templates with better messaging

#### U10: Responsive Mobile
**APG current**: Tailwind responsive classes but not mobile-optimized
**Plan**: Stack create form below table on mobile, collapsible sidebar

## Widget Assignment Rules

Following the Salesforce/SAP pattern, APG field type → widget mapping:

| Field name pattern | Semantic type | Widget |
|-------------------|---------------|--------|
| `*_email` / contains `email` | email | mailto link |
| `*_phone` / `*_tel` | phone | tel link |
| `*_url` / `*_link` | url | external link |
| `*_image` / `*_photo` | image_url | thumbnail |
| `amount` / `price` / `cost` / `*_ksh` / `*_usd` | currency | formatted number |
| `*_percent` / `*_rate` | percent | progress bar |
| `*_rating` / `*_score` (0-5) | rating | star display |
| `*_color` / `*_colour` | color | swatch |
| `status` / `state` / `*_status` | status | colored badge |
| `is_*` / `has_*` / `*_enabled` | boolean | check/x icon |
| `*_json` / `*_config` / `*_metadata` | json | expandable pre |
| `description` / `notes` / `*_body` | text_long | multiline |
| `*_id` (foreign key) | fk | link to related record |

## Implementation Phases

### Phase 1 (Now — immediate quality wins)
1. Skeleton screens — S effort, huge perceived quality improvement
2. Highlights panel — M effort, immediately enterprise-feeling
3. Rich activity feed with typed events — L effort
4. Command palette (Cmd+K) — L effort

### Phase 2
5. Full inline field edit improvements
6. Recent items in localStorage
7. Keyboard shortcuts modal
8. Better empty states

### Phase 3
9. Field metadata annotation system in DSL
10. New widget types (rich_text, file_upload, date_picker)
11. Mobile-optimized layouts
