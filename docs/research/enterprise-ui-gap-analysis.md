# Enterprise UI Gap Analysis: APG vs World-Class Platforms
**Date:** 2026-06-15  
**Author:** Research Agent (claude-sonnet-4-6)  
**Scope:** Salesforce Lightning, ServiceNow Horizon, SAP Fiori, Microsoft Dynamics 365, HubSpot CRM, Monday.com Vibe, Notion, Linear, Retool, Appsmith/Budibase vs APG generated Tailwind+htmx UI

---

## 1. Executive Summary — Top 20 UI Gaps (Priority Order)

APG currently generates: entity list pages (search/sort/pagination/bulk-select), record detail pages (Details/Related/Activity tab panels), Kanban board, sidebar create form, toast notifications, semantic field display (email links, phone links, status badges, rating stars), floating bulk action bar (delete, CSV export), FK dropdowns, landing pages, guided workflow wizard.

The following gaps are ordered by expected user-perceived quality impact:

| # | Gap | Severity | Analogous Platform Feature |
|---|-----|----------|---------------------------|
| 1 | **No highlights panel / compact record header** | Critical | Salesforce compact layout (7 key fields at top of record) |
| 2 | **No inline field editing** (click-to-edit in place) | Critical | Salesforce inline edit, Notion click-to-edit, Linear inline fields |
| 3 | **No command palette / global search** (Cmd+K) | Critical | Linear Cmd+K, Salesforce Einstein Search, Notion / |
| 4 | **No skeleton screens** — raw spinners or blank flashes | High | All platforms; Carbon Design System shimmer pattern |
| 5 | **Activity feed is a stub** — no typed events, no filtering | High | HubSpot 3-column timeline, Salesforce Activity Timeline |
| 6 | **No KPI / metric card widgets** on list/home pages | High | Monday.com board stats, Salesforce report charts, SAP OVP |
| 7 | **No column management** — no pin, resize, show/hide | High | Salesforce enhanced related list, Retool table, monday.com |
| 8 | **No saved views** — filter+sort combos not persisted | High | Linear saved filters, Salesforce list views, monday.com views |
| 9 | **No breadcrumb trail** on record detail pages | High | SAP Fiori dynamic header, ServiceNow workspace breadcrumb |
| 10 | **No related list panel** on record page (only tab) | High | Salesforce Related List Single, SAP Object Page sections |
| 11 | **No global recent items** in navigation | Medium | Salesforce global header recent items (5 MRU records) |
| 12 | **No empty state designs** — blank tables look broken | Medium | Linear, Notion, HubSpot — illustrated empty states + CTA |
| 13 | **No progress/process indicator** (path / stepper) | Medium | Salesforce Path component, SAP BPF, Linear cycle progress |
| 14 | **No notification inbox** — only transient toasts | Medium | Salesforce notification bell, ServiceNow agent notification list |
| 15 | **No dark mode / theme system** | Medium | Linear dark-first, SLDS 2 styling hooks, SAP design tokens |
| 16 | **No dense vs comfortable layout toggle** | Medium | SAP Fiori compact/cozy/condensed density, Retool density |
| 17 | **No keyboard shortcut system** | Medium | Linear (30+ shortcuts), Salesforce (keyboard nav) |
| 18 | **No pivot table / grouped aggregation view** | Low-Med | Monday.com dashboards, Retool table grouping, SAP ALP |
| 19 | **No Gantt / timeline chart widget** | Low | Monday.com Gantt, Retool Timeline component |
| 20 | **No field-level widget type assignment** (beyond auto-detect) | Low | Salesforce Dynamic Forms, SAP Fiori field metadata |

---

## 2. Platform-by-Platform Analysis

### 2.1 Salesforce Lightning Design System (SLDS)

**Visual Design Language**
- Color system: design tokens via CSS custom properties. SLDS 1 uses `--lwc-*` variables; SLDS 2 moves to global styling hooks (`--sds-c-*`) with CSS `var()`. Primary brand blue `#0176d3`, neutral grays `#3e3e3c` / `#706e6b` / `#dddbda`, background `#f3f2f2`.
- Typography: Salesforce Sans (custom), fallback system-ui. Body 13px/1.5, label 12px uppercase tracked, heading 20px/600.
- Spacing: 4px base unit. Scale: 4, 8, 12, 16, 20, 24, 32.
- Border radius: 4px standard, 0.25rem.
- Elevation: flat cards with `box-shadow: 0 2px 2px rgba(0,0,0,.1)`.

**Key SLDS Components (full list)**
Accordion, Activity Timeline, Alert, Avatar, Badge, Breadcrumb, Button, Button Group, Button Icon, Card, Carousel, Chat, Checkbox, Checkbox Button, Chip, Color Picker, Combobox, Data Table, Date Picker, Datetime Picker, Docked Composer, Docked Form Footer, Docked Utility Bar, Dropdown Menu, Dueling Picklist, Dynamic Icon, Expression, Feed, File Selector, Files, Form Element, Global Header, Global Navigation, Icon, Input, Listbox, Lookup, Map, Menus, Modal, Notification, Paginator, Path, Picklist, Pills, Popover, Progress Bar, Progress Indicator, Progress Ring, Prompt, Radio Button Group, Rich Text Editor, Scoped Notification, Select, Setup Assistant, Slider, Spinner, Split View, Summary Detail, Tabs, Textarea, Time Picker, Toast, Toggle, Tooltip, Tree, Tree Grid, Vertical Navigation, Visual Picker, Welcome Mat.

**Record Detail Page Components**
A Salesforce Lightning record page contains in order:
1. **Global Header** — app switcher, nav bar with tabs (MRU dropdowns), Einstein global search, favorites star, bell notification, utility bar toggle, user avatar.
2. **Record Highlights Panel** — driven by Compact Layout; shows object icon + color, record name (h1), up to 7 key fields in a 2-row grid, action buttons (Edit, Delete, Follow, Clone, custom quick actions).
3. **Path Component** (optional) — horizontal stage tracker with key fields per stage; chevron steps.
4. **Tab Bar** — Details | Related | Activity | Chatter (custom tabs configurable).
5. **Details Tab** — field sections driven by Page Layout; 2-column responsive grid; read mode + inline edit pencil icons; "Edit All" mode.
6. **Related Tab** — Related List Quick Links (count badges) + individual related lists with Enhanced List (10 columns, resize, mass actions).
7. **Activity Tab** — Activity Timeline: New Task/Event/Log a Call/Email quick compose; "Next Steps" bucket; "Past Activity" chronological list with type icons (call, email, task, event, meeting); filter by type/date; expand/collapse items; compact layout fields on expansion.
8. **Related Record panels** — sidebar cards showing parent/child record summaries.
9. **Dynamic Forms** — Fields and sections draggable individually onto canvas; visibility rules per field (field value / profile / device).

**What APG has vs Salesforce record page:**
- APG has: Details/Related/Activity tabs ✓, basic field display ✓, status badges ✓, rating stars ✓.
- APG missing: Highlights panel ✗, compact layout system ✗, action button bar ✗, Path component ✗, inline edit ✗, Related List Quick Links ✗, enhanced related lists (column resize/pin) ✗, rich Activity Timeline (typed events, filtering, compose) ✗, Dynamic Forms visibility rules ✗.

**List/Table UX**
- `slds-table`: sticky header, row-level checkboxes (shift-click range), inline edit per cell (pencil icon on hover → input → check/x icons), bulk action toolbar (appears on selection), column header sort (tri-state: neutral/asc/desc), column resize handle (drag right edge).
- List views: named views, filter logic (up to 10 filter rows, AND/OR), column chooser, pin/lock columns, split-view (list + record preview side-by-side).
- CSS classes: `slds-table slds-table_cell-buffer slds-table_bordered slds-table_fixed-layout`, row `slds-hint-parent`, checkbox cell `slds-th__action`, sort `slds-is-sortable slds-is-sorted_asc`.

**Global Navigation**
- Global search: `slds-global-search` in `slds-global-header`. On click: dropdown shows 5 MRU records + recent searches. As you type: instant results (Cmd+K shortcut to focus). Scope selector to filter by object type. Natural language query cards.
- Favorites: star icon per tab; favorites dropdown in nav bar items.
- Recent items: automatic 5-item MRU per object shown in nav tab dropdowns.
- Notification bell: count badge; dropdown panel with timestamped items; "See All" link to full notification page.

**Inline Edit Pattern**
Click a field value → pencil icon appears on hover → click pencil or double-click value → field converts to input widget appropriate for type → Save (checkmark) / Cancel (X) buttons appear in field → optimistic update shown immediately → `notifyRecordUpdateAvailable()` called → success toast fires. "Edit All" button puts entire Details tab into edit mode simultaneously.

**Widget Assignment Rules (Lightning App Builder / Dynamic Forms)**
Salesforce maps field metadata types to input widgets automatically:
| Field Type | Widget Rendered |
|---|---|
| Text | `<lightning-input type="text">` |
| Number / Currency / Percent | `<lightning-input type="number">` with formatter |
| Date | `<lightning-datepicker>` calendar popover |
| DateTime | `<lightning-datetimepicker>` |
| Checkbox / Boolean | `<lightning-input type="checkbox">` toggle |
| Picklist (single) | `<lightning-combobox>` |
| Picklist (multi) | `<lightning-dual-listbox>` |
| Lookup (FK) | `<lightning-lookup>` typeahead search with record card preview |
| Long Text / Rich Text | `<lightning-textarea>` or `<lightning-input-rich-text>` |
| Email | `<lightning-input type="email">` + mailto link in read mode |
| Phone | `<lightning-input type="tel">` + tel: link in read mode |
| URL | `<lightning-input type="url">` + external link in read mode |
| Rating (custom) | Star widget `⭐⭐⭐⭐⭐` |
| Address | Compound field → street/city/state/zip/country sub-fields |
| Name | Compound field → salutation/first/last |
| Geolocation | Map widget |

In **Dynamic Forms**, field-type detection is fully automatic; the administrator cannot override the widget type — only visibility rules, required status, and placement are configurable per field per profile/record type.

---

### 2.2 SAP Fiori (Horizon Design System)

**Visual Design Language**
- Based on SAP's design token system. Primary color: `#0070f2` (SAP Blue), secondary actions `#0064d9`, semantic: success `#107e3e`, warning `#e9730c`, error `#bb0000`, information `#0070f2`.
- Typography: `72` font (SAP's custom typeface, licensed); web fallback `"72", Arial, Helvetica, sans-serif`. Body 14px/1.4, label 12px, heading 20px, display 28px.
- Spacing: 0.25rem (4px) base. Named tokens: `--sapContent_IconHeight`, `--sapFontSize`, `--sapFontFamily`.
- Density modes: **Cozy** (44px touch targets, default for all), **Compact** (32px, desktop power users), **Condensed** (24px, data-heavy screens). Set via `sapUiSizeCompact` / `sapUiSizeCozy` CSS class on container.
- Border radius: `0.25rem` standard, `0.5rem` cards.
- Elevation: cards use `box-shadow: 0 0 0 1px rgba(0,0,0,.15), 0 2px 4px rgba(0,0,0,.15)`.

**Floorplans (Page-Level Templates)**
SAP provides five standard page-level templates (floorplans):
1. **List Report** — filter bar + table/chart. Header: title bar + filter bar (SmartFilterBar with search fields). Content: SmartTable or chart. Actions: "Go" to apply filters, "Adapt Filters" to show/hide filter fields. Toolbar: Create, Edit, Delete, Export, group actions.
2. **Object Page** — master detail for a single record. Dynamic page header (collapses on scroll), title area with key fields, header facets (KPI tiles, contact card, address block), content area with sections and subsections. Flexible column layout (1/2/3 columns responsive). Section types: form, table, chart, custom.
3. **Worklist** — simplified list for tasks requiring action; no filter bar; direct-action focus.
4. **Overview Page (OVP)** — card-based dashboard; role-driven cards (List Card, Table Card, Bar Chart Card, Stack Card, Object Stream Card, Timeline Card, KPI Header Card, Analytical Card). Cards auto-refresh; drag-to-rearrange in edit mode.
5. **Analytical List Page (ALP)** — chart + table in same view; bidirectional filtering (chart → table, table → chart); microcharts in table cells.

**Object Page Header** (what APG needs to emulate with its record pages):
- `sap.uxap.ObjectPageDynamicHeaderTitle` — avatar/image, title, subtitle, tags, breadcrumb, KPI tiles row, header actions (Edit/Save/Cancel/Delete).
- Collapses from expanded (full detail) to collapsed (just title + key actions) on scroll.
- Section navigation: horizontal tab bar pinned below collapsed header, or left side anchor navigation for long pages.

**Density Toggle Implementation**
```css
/* Cozy (default) */
.sapUiSizeCozy .sapMInputBase { height: 2.75rem; }
/* Compact */
.sapUiSizeCompact .sapMInputBase { height: 2rem; }
/* Condensed */
.sapUiSizeCondensed .sapMInputBase { height: 1.5rem; }
```
APG equivalent: add a density toggle button in the app header; store preference in `localStorage`; toggle a CSS class on `<body>`.

---

### 2.3 ServiceNow Horizon Design System

**Workspace Structure**
Three page types in Horizon workspaces:
1. **Landing page** — actionable dashboard; KPI tiles, activity queues, charts. `snc-landing-page` web component.
2. **List page** — filterable, sortable record list. Columns managed per user. Saved personal views. `snc-list` component.
3. **Record page** — three-panel layout: left sidebar (related links, related records), center (form fields, rich text body), right sidebar (activity feed, related lists, attachments, approvals).

**Navigation Architecture**
- **Unified Navigation Header** — global search (magnifier icon, Cmd+K), notification bell, help, user avatar, app switcher.
- **Application Navigation** — left sidebar with collapsible module groups; breadcrumb trail at top of content area.
- **Page Navigation** — tab bar within a workspace record page.
- Workspace supports both tab-based multitasking (multiple open records as tabs, like browser tabs) and single-record breadcrumb navigation.

**Messaging Patterns**
- Alert list: stacked banners above content, expandable/collapsible, LIFO order.
- Inline validation: field-level error messages below inputs.
- Toast: slide-in bottom-right, auto-dismiss 5s, manual close X.
- Notification inbox: bell with badge count; slide-out panel; timestamped items with "mark read".

**Components (Horizon)**
`snc-alert`, `snc-avatar`, `snc-badge`, `snc-button`, `snc-card`, `snc-checkbox`, `snc-chip`, `snc-combobox`, `snc-context-menu`, `snc-data-cell`, `snc-data-table`, `snc-date-picker`, `snc-dialog`, `snc-drawer`, `snc-dropdown`, `snc-file-attachment`, `snc-form`, `snc-icon`, `snc-input`, `snc-label`, `snc-list`, `snc-loading`, `snc-modal`, `snc-navigation`, `snc-notification`, `snc-pagination`, `snc-panel`, `snc-popover`, `snc-progress-bar`, `snc-radio`, `snc-select`, `snc-skeleton`, `snc-slider`, `snc-split-button`, `snc-tab-bar`, `snc-tag`, `snc-textarea`, `snc-time-picker`, `snc-toast`, `snc-toggle`, `snc-tooltip`, `snc-tree`.

---

### 2.4 Microsoft Dynamics 365

**UI Structure**
- **Model-Driven Apps**: Navigation bar (top) + left nav pane + main page area + optional right pane.
- Main page types: Dashboard (charts + lists), Form (record detail), View (list), Business Process Flow (guided process bar across top of form).
- **Command Bar / Ribbon** — contextual action buttons at top of each page; adapts to object and selection state (no record selected vs. one selected vs. multiple).
- **Business Process Flow (BPF)** — horizontal multi-stage progress bar pinned below the command bar; each stage is clickable; fields appear below per stage (like Salesforce Path but more detailed).
- **Power Apps Fluent UI** — Dynamics 365 is migrating to Microsoft's Fluent 2 design system. Primary `#0078d4` (Microsoft blue), typography `"Segoe UI"`, spacing 4/8/12/16/24/32px.
- **Timeline control** on forms: standard component showing emails, calls, appointments, notes, custom activities. Filter by type/date. "New" button expands to compose area for notes/tasks. Collapse/expand items. Pinned notes. Sort asc/desc.
- **Dashboards** — chart types: Bar, Column, Line, Pie, Funnel, Multi-Stream (streaming activity wall). Up to 6 widgets per dashboard, configurable grid.

**Accessibility** — WCAG 2.1 AA required. Full keyboard navigation. ARIA live regions for dynamic updates. High-contrast mode supported.

---

### 2.5 HubSpot CRM

**Record Page Layout (3-column)**
- **Left sidebar** (200px): Actions (Log Activity, Note, Call, Email, Meeting, Task), property card with key fields (collapsed by default), website activity, communication subscriptions. Each card collapsible, draggable.
- **Center column**: Activity timeline (chronological; types: calls, emails, notes, meetings, tasks; icons per type; filter by type/date; "Recent 3" and "Upcoming 3" summary cards; full timeline below; compact card state → expand on click). Custom activity cards. App Extension cards.
- **Right sidebar** (280px): Associated records (contacts on a deal, deals on a contact), attachments, segment memberships, playbooks, Salesforce sync, custom app cards.

**Design Principles from HubSpot's Own Product Blog**
- Remove icon clutter: HubSpot removed association card icons because they added visual noise with no scanning value.
- Information density: reduced margins within/between timeline cards; less whitespace.
- Progressive disclosure: compact timeline cards (title + date + type icon only) → expand for full body.
- Fewer uses of color: badge/indicator color reserved for status signals only; avoids decorative color.
- Quick-copy buttons on contact fields: hover reveals copy-to-clipboard icon for email/phone instead of linking.

**Typography / Colors**
- Font: `Lexend` (primary), fallback `sans-serif`. Body 14px, label 12px, heading 18px/600.
- Primary blue `#ff7a59` (HubSpot orange-red for CTA), secondary `#0091ae` (teal).
- Card background `#ffffff`, page background `#f5f8fa`, border `#cbd6e2`.

---

### 2.6 Monday.com (Vibe Design System)

**Views Available per Board**
Table (default spreadsheet), Kanban, Gantt, Calendar, Map, Files, Workload, Chart, Timeline, Form. Each view is independently configured with its own filter/grouping/column visibility. "Saved Views" pinned as tabs.

**Vibe Design System**
- Open source React component library, 50+ components, Storybook-documented.
- Colors: `--color-brand` is monday's bright orange `#FF3D57`; backgrounds `#F6F7FB` (light), `#1C1F3B` (dark).
- Typography: `Figtree` font. Body 14px/400, label 12px/500, heading 18px/700.
- Spacing: 4px base. Named tokens: `--spacing-xs: 4px`, `--spacing-s: 8px`, `--spacing-m: 16px`, `--spacing-l: 24px`, `--spacing-xl: 48px`.
- Components include: `Avatar`, `AvatarGroup`, `Badge`, `Box`, `Button`, `Checkbox`, `Chips`, `Combobox`, `ColorPicker`, `Counter`, `DatePicker`, `Dialog`, `Divider`, `Dropdown`, `EditableHeading`, `ExpandCollapse`, `Flex`, `Form`, `Heading`, `Icon`, `IconButton`, `Label`, `Link`, `List`, `Loader`, `Menu`, `Modal`, `MultiStepIndicator`, `RadioButton`, `Search`, `Skeleton`, `Slider`, `SplitButton`, `Steps`, `Table`, `Tabs`, `Tag`, `Text`, `TextArea`, `TextField`, `TimePicker`, `Toast`, `Toggle`, `Tooltip`, `VirtualizedList`.

**Key patterns absent from APG**:
- Board header with multiple view tabs + "Add View" + saved view management.
- Group-by rows in table with collapse/expand + aggregate row (sum/avg/count per group).
- Status column: color-coded pill labels, click to cycle through states.
- Person column: avatar + name autocomplete.
- Board-level dashboard: drag-and-drop chart widgets over the board data.

---

### 2.7 Notion

**Design Language**
- Font: `NotionInter` (custom Inter variant). Body 16px/400, heading 20px+/500-600, display serif (optional).
- Spacing: 4px base. Scale: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64px.
- Border radius: 4px (buttons/inputs), 8px (cards), 12px (modals).
- Background: `#ffffff` (default light), `#191919` (dark). Sidebar: `#f7f7f5` (light).
- Primary interactive: `#097fe8` (blue).
- Shadows: soft multi-layer `box-shadow: rgba(15,15,15,.05) 0 0 0 1px, rgba(15,15,15,.1) 0 3px 6px, rgba(15,15,15,.2) 0 9px 24px`.

**Block Editor (/ command menu)**
Notion's most-copied pattern: typing `/` in a text block opens a block picker floating menu. Categories: Text, Heading 1/2/3, Quote, Callout, Divider, Table, List, Code, Toggle, Database (inline/full), Link. Keyboard navigable. Search within menu. Each block type has an icon + name + description.

APG equivalent opportunity: A `/` command menu in rich text fields or a form-builder context.

**Database Views**
Each database supports: Table, Board (Kanban), Gallery, List, Calendar, Timeline, Chart. View settings panel: layout, property visibility, filters, sorts, grouping. Views saved and tabbed at top of database. Property types: Text, Number, Select, Multi-select, Date, Person, Files, Checkbox, URL, Email, Phone, Formula, Relation (FK), Rollup, Created time, Last edited, Created by, Last edited by.

**Sidebar**
224px fixed width. Nested page tree (infinite depth). Workspace switcher top-left. Search (Cmd+P), Inbox (notification center), Settings. Hover shows "+" add sub-page, "..." options menu. Drag-to-reorder. Collapsed state: 0px (hidden), toggle with left-edge hover.

---

### 2.8 Linear (Best-in-class UX)

**Design Language**
- Dark-first. Light mode exists but is secondary. Canvas `#16161A`, elevated surfaces `#1E1E24` / `#26262C` / `#2E2E36`. Accent: `#5E6AD2` (periwinkle purple). Semantic: success `#4CAF50`, warning `#F5A623`, error `#E5484D`.
- Typography: `Inter Variable`, weights 510 (regular UI) / 590 (emphasis) / 300 (display). Berkeley Mono for code.
- Color generated with LCH color space for perceptual uniformity across light/dark.
- Spacing: 4px base. Tight defaults: sidebar items 28px height, list rows 36px.
- Surfaces have 1px inset borders (`inset 0 0 0 1px rgba(255,255,255,.08)`) instead of fill differences.

**Keyboard System (all 30+)**
- `C` — create issue anywhere
- `Cmd+K` — command palette (fuzzy finder: issues, projects, cycles, commands)
- `X` — select issue
- `Shift+↑/↓`, `Shift+Click` — multi-select
- `Esc` — go back / clear selection
- `G then I` — go to Inbox
- `G then V` — go to current cycle
- `G then B` — go to Backlog
- `G then M` — go to My Issues
- `O then F` — open Favorites
- `O then P` — open Projects
- `1-4` — set priority (Urgent/High/Medium/Low)
- `S` — set status
- `A` — assign
- `L` — add label
- `T` — set cycle/iteration

**Command Palette (Cmd+K)**
Fuzzy-search modal, centered overlay, semi-transparent backdrop. Three sections: Recent, Commands, Search Results. Each item shows: icon, name, keyboard shortcut hint on right. Arrow keys navigate, Enter executes, Esc dismisses. Tab switches between search types.

**Shortcut Discovery System**
Hovering any UI element for 1.5s shows a tooltip with the keyboard shortcut. Bottom-right corner shows contextual shortcut hints for current view.

**Issue Detail Page**
- No tab panels. Single-page scrollable layout.
- Top: breadcrumb (Team > Project > Issue ID). Status button (inline editable, cycles states). Priority icon. Title (inline editable H1). Description (rich text / markdown with / command menu).
- Right sidebar panel (280px fixed): Status, Priority, Assignee, Labels, Cycle, Project, Estimate, Due date, Relations (blocked by / blocking / duplicate of). All inline-editable on click.
- Bottom: Activity feed — chronological list of: status changes, comment additions, label changes, assignee changes, relation changes. Each event shows: avatar, action text, timestamp. Comment input at bottom (markdown, @mention, emoji, file attach).

**Performance**
Near-instant navigation (<100ms perceived). Virtual scrolling for long lists. Optimistic updates — UI updates before server confirms.

---

### 2.9 Retool

**Component Library (100+)**
Tables: Table (inline edit, column pin, freeze, sort, filter, group-by, row expand, virtualization), Listview. Charts: Chart (bar/line/area/scatter/pie), Statistic (KPI number + trend). Forms: Form, JSON Form, JSON Schema Form. Inputs: TextInput, NumberInput, CurrencyInput, PasswordInput, TextArea, RichTextEditor, DatePicker, DateRangePicker, TimePicker, Select, Multiselect, Combobox, TreeSelect, Slider, RangeSlider, ColorPicker, FileButton, Checkbox, Switch, RadioGroup, Rating, Signature. Navigation: Navigation, Tabs, Steps (stepper). Layout: Container, Modal, Drawer, Popover, Divider, Spacer, **Stacks** (flexbox wrapper — new Q3 2024). Feedback: Alert, Progress, Spinner, Skeleton, Toast. Display: Text, Heading, Image, Video, PDF Viewer, Map, Iframe, Icon, Badge, Tag, Avatar, Timeline (new Q3 2024), Statistic, Table of Contents.

**Table Component Details**
- Column types: text, number, currency, percent, date, boolean, badge (color-mapped), button, button group, image, link, select (inline edit), tags, rating, expand.
- Inline editing: click cell → editable input → Tab to next cell; "Save" bar appears at bottom of table.
- Column operations: drag to reorder, drag right edge to resize, right-click header for pin/hide, "Columns" button for show/hide panel.
- Bulk actions: checkbox column → floating bar appears with custom action buttons.
- Row grouping: group-by column with aggregate functions (sum/count/min/max/avg) per group.

**Stacks (Flexbox Layouts)**
New in Q3 2024. Horizontal/vertical direction, alignment (start/center/end/stretch/baseline), distribution (start/center/end/space-between/space-around/space-evenly), gap (px). Replaces manual absolute positioning.

**Timeline Component (New)**
Vertical timeline; items have: dot (color/icon), title, subtitle, timestamp, optional body. Support for multiple data sources.

---

### 2.10 Appsmith / Budibase

**Appsmith**
- 45+ widgets: Table, Chart, Form, Button, Input, Select, MultiSelect, DatePicker, TimePicker, FilePicker, CheckBox, Switch, RadioGroup, Rate (star rating), Tabs, Modal, Drawer, List, Image, Video, Iframe, Map, Text, Divider, Rich Text Editor, JSON Form (auto-generates form from JSON schema), Container, Progress Bar, Circular Progress, Statistic Box, Audio, Camera, PhoneInput, CurrencyInput, NumberSlider, RangeSlider, Category Slider, Switch Group, Checkbox Group, Tree Select, Custom Widget (HTML/CSS/JS).
- Table: inline edit (cell-level), column pin, column type override (per-column), custom column (formula), row-level actions (button column), pagination.
- JSON Form: field type auto-detected from JSON schema; renders appropriate input widget per type.

**Budibase**
- Auto-generates CRUD app from SQL schema (detect table → generate list/form/detail views automatically).
- Plug-in CLI for custom components.
- Built-in design templates (light/dark themes, brand color pickers).
- Automation builder: triggers (row created/updated/deleted, scheduled, webhook) + actions (send email, create row, update row, execute script, send notification).
- Data sources: PostgreSQL, MySQL, MongoDB, REST API, Google Sheets, Airtable.

---

## 3. Component Gap Matrix

`✓` = APG has it | `~` = partial/stub | `✗` = missing

| Component | SLDS | SAP Fiori | ServiceNow | Dynamics | HubSpot | Monday | Linear | Retool | APG |
|---|---|---|---|---|---|---|---|---|---|
| Highlights Panel / Record Header KPIs | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ~ | ✗ |
| Inline field edit (click-to-edit) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Command palette (Cmd+K) | ~ | ✗ | ✓ | ✗ | ✗ | ~ | ✓ | ~ | ✗ |
| Skeleton screens | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Rich activity feed (typed events, filter) | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ~ | ~ |
| KPI metric cards | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ~ | ✓ | ✗ |
| Column pin/resize/show-hide | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ~ | ✓ | ✗ |
| Saved views / named filters | ✓ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ✗ |
| Breadcrumb trail | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ✗ | ✗ |
| Related list panel (sidebar) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Global recent items (MRU) | ✓ | ~ | ✓ | ✓ | ✓ | ~ | ✓ | ✗ | ✗ |
| Empty state designs | ~ | ~ | ~ | ~ | ✓ | ✓ | ✓ | ~ | ✗ |
| Path / process stepper | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ~ | ✓ | ~ |
| Notification inbox | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Dark mode / theming | ✓ | ✓ | ~ | ✓ | ~ | ~ | ✓ | ✓ | ✗ |
| Density toggle (compact/comfortable) | ✗ | ✓ | ~ | ✗ | ✗ | ✗ | ~ | ✓ | ✗ |
| Keyboard shortcuts | ✓ | ~ | ~ | ~ | ~ | ✓ | ✓ | ~ | ✗ |
| Pivot / group-by aggregation | ~ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ |
| Gantt / timeline chart | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| Field-level widget type override | ✓ | ✓ | ✓ | ✓ | ~ | ~ | ✗ | ✓ | ✗ |
| Toast notifications | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kanban board | ~ | ✗ | ✓ | ~ | ✗ | ✓ | ~ | ✓ | ✓ |
| Bulk actions bar | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CSV export | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FK dropdown / lookup | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Status badges | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Star rating display | ~ | ✗ | ✗ | ~ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Responsive / mobile | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ |
| Tab panels on record | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sidebar create form | ~ | ~ | ~ | ~ | ~ | ✓ | ~ | ✓ | ✓ |

---

## 4. Widget Assignment Rules — How Platforms Map Field Types to Widgets

### 4.1 The Universal Mapping Table

Every enterprise platform implements a data-type → widget mapping. APG auto-detects some types. Here is the complete canonical mapping drawn from SLDS, SAP Fiori Elements, Retool, Appsmith, and Budibase:

| Logical Field Type | Read Mode Display | Edit/Create Widget | APG Status |
|---|---|---|---|
| `string` (short, <255) | Plain text | `<input type="text">` | ✓ |
| `string` (long, textarea) | Truncated + expand | `<textarea>` | ~ partial |
| `html` / `richtext` | Rendered HTML | Rich text editor (Quill / TipTap / ProseMirror) | ✗ |
| `integer` / `float` | Formatted number | `<input type="number">` with step | ✓ partial |
| `currency` | `$1,234.56` with symbol | Currency input (symbol prefix, 2 decimal places) | ✗ |
| `percent` | `42.3%` | Number input + `%` suffix | ✗ |
| `boolean` | "Yes" / "No" or checkmark icon | Toggle switch or checkbox | ~ |
| `date` | `15 Jun 2026` (locale-formatted) | Datepicker calendar popover | ~ basic |
| `datetime` | `15 Jun 2026, 14:32` | Datetime picker | ~ basic |
| `time` | `14:32` | Time picker | ✗ |
| `duration` | `2h 30m` | Duration input (HH:MM) | ✗ |
| `email` | `mailto:` anchor, copy icon | `<input type="email">` with validation | ✓ |
| `phone` | `tel:` anchor, copy icon | `<input type="tel">` with masking | ✓ |
| `url` | External link (↗ icon) | `<input type="url">` with validation | ✓ |
| `enum` / `status` (single) | Colored badge/pill | `<select>` or combobox with color swatches | ✓ |
| `enum` (multi-select) | Multiple pills | Multiselect combobox or tag input | ~ |
| `FK` / `lookup` | Record name (clickable link) | Typeahead search (server-side) with result card | ✓ |
| `FK[]` / `many-to-many` | Multiple record links | Multi-lookup with tags | ✗ |
| `rating` (1-5) | `★★★☆☆` stars | Clickable star widget (hover preview) | ✓ |
| `color` | Color swatch box | Color picker (hex input + palette) | ✗ |
| `json` / `object` | Collapsible JSON tree | JSON / YAML editor (Monaco / CodeMirror) | ✗ |
| `file` / `attachment` | File name + icon + size | File upload dropzone | ✗ |
| `image` | `<img>` thumbnail | Image upload + crop | ✗ |
| `geolocation` | Map pin preview | Map picker or lat/lng inputs | ✗ |
| `address` (compound) | Multi-line address block | Compound form (street/city/state/zip/country) | ✗ |
| `name` (compound) | Full name | Compound form (first/last/salutation) | ✗ |
| `uuid` | Monospace, copy icon | Auto-generated, read-only | ~ |
| `slug` | Monospace | Auto-generated from name field, editable | ✗ |
| `progress` (0-100) | Progress bar | Slider or number input | ✗ |

### 4.2 How Salesforce's "Dynamic Forms" Decides Widget Assignment

Dynamic Forms is purely **metadata-driven**. The platform reads the Salesforce object field descriptor:
1. `type` property on `FieldDefinition` → determines base widget class.
2. `htmlFormatted` flag → if true, renders HTML content instead of plain text (for rich text fields).
3. `controllerName` → if set, this field is a dependent picklist; the controller field controls which values are available.
4. `referenceTo` → if non-null, field is a lookup; renders `lightning-lookup` component with object API name for search scoping.
5. `precision` / `scale` → for number/currency fields, controls decimal places.
6. `length` → for text fields > 255 chars, upgrades to textarea automatically.
7. `required` / `updateable` / `createable` → controls whether field is mandatory or read-only.

APG's generator should implement a similar metadata annotation system in its model definitions, then drive widget selection from that metadata at template-render time.

### 4.3 SAP Fiori Elements Widget Assignment

SAP uses OData `$metadata` annotations:
- `@UI.lineItem` — appears in list report table
- `@UI.fieldGroup` — appears in object page form section
- `@UI.identification` — appears in title/header area
- `@UI.selectionField` — appears in filter bar
- `@Semantics.currencyCode` — render as currency with symbol
- `@Semantics.email.address` — render as mailto link
- `@Semantics.telephone.type` — render as tel link
- `@UI.hidden` — do not render

For APG, the equivalent is model field metadata (a `field_meta` dict or Pydantic field extra kwargs): `{"widget": "currency", "semantic": "email", "display_only": True, "hidden": False}`.

---

## 5. Color / Typography / Spacing System Recommendations for APG

### 5.1 Recommended Design Token System

APG should adopt a CSS custom property token system compatible with Tailwind's theme extension. Define in a `apg-tokens.css` file injected via the `_html_page()` base template:

```css
:root {
  /* Brand */
  --apg-color-primary:        #2563eb;   /* Tailwind blue-600 */
  --apg-color-primary-dark:   #1d4ed8;   /* blue-700 */
  --apg-color-primary-light:  #dbeafe;   /* blue-100 */

  /* Semantic */
  --apg-color-success:        #16a34a;   /* green-600 */
  --apg-color-success-light:  #dcfce7;   /* green-100 */
  --apg-color-warning:        #d97706;   /* amber-600 */
  --apg-color-warning-light:  #fef3c7;   /* amber-100 */
  --apg-color-danger:         #dc2626;   /* red-600 */
  --apg-color-danger-light:   #fee2e2;   /* red-100 */
  --apg-color-info:           #0891b2;   /* cyan-600 */
  --apg-color-info-light:     #cffafe;   /* cyan-100 */

  /* Neutrals */
  --apg-color-surface:        #ffffff;
  --apg-color-surface-alt:    #f8fafc;   /* slate-50 */
  --apg-color-border:         #e2e8f0;   /* slate-200 */
  --apg-color-text:           #0f172a;   /* slate-900 */
  --apg-color-text-muted:     #64748b;   /* slate-500 */
  --apg-color-text-subtle:    #94a3b8;   /* slate-400 */

  /* Typography */
  --apg-font-family:          "Inter var", "Inter", ui-sans-serif, system-ui, sans-serif;
  --apg-font-mono:            "JetBrains Mono", "Fira Code", ui-monospace, monospace;
  --apg-font-size-xs:         0.75rem;   /* 12px */
  --apg-font-size-sm:         0.875rem;  /* 14px */
  --apg-font-size-base:       1rem;      /* 16px */
  --apg-font-size-lg:         1.125rem;  /* 18px */
  --apg-font-size-xl:         1.25rem;   /* 20px */
  --apg-font-size-2xl:        1.5rem;    /* 24px */

  /* Spacing (4px base) */
  --apg-space-1:   0.25rem;   /* 4px */
  --apg-space-2:   0.5rem;    /* 8px */
  --apg-space-3:   0.75rem;   /* 12px */
  --apg-space-4:   1rem;      /* 16px */
  --apg-space-6:   1.5rem;    /* 24px */
  --apg-space-8:   2rem;      /* 32px */
  --apg-space-12:  3rem;      /* 48px */

  /* Radius */
  --apg-radius-sm:    0.25rem;  /* 4px */
  --apg-radius-md:    0.375rem; /* 6px */
  --apg-radius-lg:    0.5rem;   /* 8px */
  --apg-radius-xl:    0.75rem;  /* 12px */
  --apg-radius-full:  9999px;   /* pill */

  /* Shadows */
  --apg-shadow-sm:  0 1px 2px 0 rgba(0,0,0,.05);
  --apg-shadow-md:  0 4px 6px -1px rgba(0,0,0,.1), 0 2px 4px -2px rgba(0,0,0,.1);
  --apg-shadow-lg:  0 10px 15px -3px rgba(0,0,0,.1), 0 4px 6px -4px rgba(0,0,0,.1);

  /* Density: comfortable (default) */
  --apg-row-height:     2.5rem;   /* 40px */
  --apg-input-height:   2.25rem;  /* 36px */
  --apg-cell-padding-y: 0.625rem; /* 10px */
}

/* Compact density */
[data-density="compact"] {
  --apg-row-height:     2rem;    /* 32px */
  --apg-input-height:   1.75rem; /* 28px */
  --apg-cell-padding-y: 0.375rem;/* 6px */
}

/* Condensed density */
[data-density="condensed"] {
  --apg-row-height:     1.5rem;  /* 24px */
  --apg-input-height:   1.5rem;  /* 24px */
  --apg-cell-padding-y: 0.25rem; /* 4px */
}
```

Tailwind `tailwind.config.js` extension:
```js
theme: {
  extend: {
    colors: {
      'apg-primary': 'var(--apg-color-primary)',
      'apg-surface': 'var(--apg-color-surface)',
      'apg-border':  'var(--apg-color-border)',
    },
    fontFamily: {
      sans: ['var(--apg-font-family)'],
      mono: ['var(--apg-font-mono)'],
    },
  }
}
```

### 5.2 Tailwind CSS Classes for Key APG Patterns

**Highlights Panel (Compact Record Header)**
```html
<div class="bg-white border-b border-slate-200 px-6 py-4">
  <div class="flex items-start gap-4">
    <!-- Object avatar -->
    <div class="w-12 h-12 rounded-lg bg-blue-600 flex items-center justify-center text-white font-semibold text-lg flex-shrink-0">
      {{ entity_initials }}
    </div>
    <!-- Title + subtitle -->
    <div class="flex-1 min-w-0">
      <h1 class="text-xl font-semibold text-slate-900 truncate">{{ record.name }}</h1>
      <p class="text-sm text-slate-500">{{ record.subtitle_field }}</p>
    </div>
    <!-- Action buttons -->
    <div class="flex gap-2 flex-shrink-0">
      <button class="px-3 py-1.5 text-sm font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700">Edit</button>
      <button class="px-3 py-1.5 text-sm font-medium border border-slate-300 text-slate-700 rounded-md hover:bg-slate-50">...</button>
    </div>
  </div>
  <!-- Key fields row (compact layout: up to 6 fields) -->
  <dl class="mt-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
    {% for field in compact_fields %}
    <div>
      <dt class="text-xs font-medium text-slate-500 uppercase tracking-wide">{{ field.label }}</dt>
      <dd class="mt-1 text-sm text-slate-900">{{ field.value }}</dd>
    </div>
    {% endfor %}
  </dl>
</div>
```

**Skeleton Screen (shimmer animation)**
```html
<!-- In Tailwind: add to APG base styles -->
<style>
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
.apg-skeleton {
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
</style>
<!-- Skeleton row for list page -->
<tr class="border-b border-slate-100">
  <td class="px-4 py-3"><div class="apg-skeleton h-4 w-4 rounded"></div></td>
  <td class="px-4 py-3"><div class="apg-skeleton h-4 w-40 rounded"></div></td>
  <td class="px-4 py-3"><div class="apg-skeleton h-4 w-24 rounded"></div></td>
  <td class="px-4 py-3"><div class="apg-skeleton h-6 w-16 rounded-full"></div></td>
</tr>
```

**Inline Edit Field**
```html
<!-- Read mode -->
<div class="group relative" x-data="{ editing: false }">
  <span x-show="!editing"
        class="cursor-pointer hover:bg-slate-50 rounded px-1 -mx-1 py-0.5 inline-flex items-center gap-1"
        @click="editing = true">
    {{ field.value }}
    <svg class="w-3 h-3 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity" ...pencil icon...></svg>
  </span>
  <!-- Edit mode -->
  <div x-show="editing" class="flex items-center gap-1">
    <input type="text" value="{{ field.value }}"
           class="border border-blue-400 rounded px-2 py-0.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
           @keydown.enter="$dispatch('save-field', { field: '{{ field.name }}', value: $el.value }); editing = false"
           @keydown.escape="editing = false"
           x-ref="input" x-init="$nextTick(() => $refs.input.focus())">
    <button class="text-green-600 hover:text-green-700" @click="...">✓</button>
    <button class="text-red-500 hover:text-red-600" @click="editing = false">✗</button>
  </div>
</div>
```

**Command Palette**
```html
<!-- Trigger: document.addEventListener('keydown', e => { if ((e.metaKey||e.ctrlKey) && e.key==='k') showPalette() }) -->
<div id="apg-command-palette"
     class="fixed inset-0 z-50 hidden"
     role="dialog" aria-modal="true" aria-label="Command palette">
  <div class="fixed inset-0 bg-black/40 backdrop-blur-sm" onclick="hidePalette()"></div>
  <div class="fixed left-1/2 top-1/4 -translate-x-1/2 w-full max-w-lg bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
    <div class="flex items-center px-4 py-3 border-b border-slate-100">
      <svg class="w-4 h-4 text-slate-400 mr-3"><!-- search icon --></svg>
      <input id="apg-palette-input" type="text" placeholder="Search records, run actions..."
             class="flex-1 text-sm outline-none placeholder:text-slate-400"
             oninput="filterPalette(this.value)">
      <kbd class="text-xs text-slate-400 border border-slate-200 rounded px-1.5 py-0.5 ml-2">Esc</kbd>
    </div>
    <ul id="apg-palette-results" class="max-h-80 overflow-y-auto py-2">
      <!-- Items injected by filterPalette() -->
    </ul>
    <div class="px-4 py-2 border-t border-slate-100 flex gap-4 text-xs text-slate-400">
      <span>↑↓ navigate</span><span>↵ select</span><span>Esc close</span>
    </div>
  </div>
</div>
```

---

## 6. Implementation Priority List with Effort Estimates

Effort scale: S = 0.5 day, M = 1-2 days, L = 3-5 days, XL = 1-2 weeks

| # | Feature | Effort | Notes |
|---|---------|--------|-------|
| 1 | **Highlights panel** (compact record header with 6 key fields + action buttons) | M | Generate from first N fields of model; configurable via field_meta annotation |
| 2 | **Skeleton screens** (shimmer CSS + htmx `hx-indicator`) | S | Single CSS animation class; wrap list/record loaders |
| 3 | **Breadcrumb trail** on record detail page | S | Entity name → record name; auto-generated from URL params |
| 4 | **Empty state designs** (illustrated + CTA per entity) | S | One reusable empty-state component with slot for message + button |
| 5 | **Inline field edit** (click-to-edit → htmx PATCH + Alpine.js toggle) | L | Per-field endpoint; optimistic UI; type-appropriate input widget |
| 6 | **Column management** (show/hide panel + column widths via CSS vars) | M | `localStorage` persistence; flyout panel; `colgroup` widths |
| 7 | **Saved views** (named filter+sort combos persisted per user per entity) | L | DB table `apg_saved_view`; save/load/delete UI in list toolbar |
| 8 | **Rich activity feed** (typed events: create/update/comment/assign; filter; compose) | L | `apg_activity_event` table; htmx feed; event type icons; filter tabs |
| 9 | **KPI metric card widgets** on list/home pages | M | Reusable `<apg-kpi-card>` component; aggregate queries auto-generated |
| 10 | **Command palette** (Cmd+K, fuzzy search, recent items) | L | JS fuzzy search (fuse.js or simple substring); MRU in localStorage; global API endpoint `/api/search?q=` |
| 11 | **Global recent items** in nav (5 MRU records per entity) | S | localStorage MRU list; dropdown in top nav |
| 12 | **Process stepper / path component** | M | Horizontal chevron progress bar; driven by status field values order |
| 13 | **Notification inbox** (bell + slide panel + persistent records) | L | `apg_notification` table; htmx polling or SSE; mark-read; bell badge count |
| 14 | **Design token CSS + density toggle** | M | CSS custom properties file; density data-attribute toggle button in header |
| 15 | **Related list panel** (sidebar on record page, not just tab) | M | Configurable related entity list; collapsible; count badge |
| 16 | **Richer field widget types** (currency, percent, duration, color, JSON tree, file upload) | L | Extend `_render_field_value()` and form widget dispatch functions |
| 17 | **Keyboard shortcuts** (C=create, X=select, ?=help, Esc=back) | M | Document-level keydown listener; shortcut map; help modal (?key) |
| 18 | **Dark mode** (CSS custom property swap via `[data-theme="dark"]`) | M | Second token set; `prefers-color-scheme` media + manual toggle |
| 19 | **Pivot / group-by table** (aggregate rows, collapse groups) | XL | SQL GROUP BY query generation; nested table HTML; toggle expand |
| 20 | **Field-level widget type override** in APG spec | M | `field_meta = {"widget": "currency|rating|progress|json|..."}` annotation in model; generator reads this |

---

## 7. Specific Implementation Recommendations

### 7.1 Highlights Panel — APG Generator Change
In the service generator, identify the first 6 non-`id`/non-`created_at`/non-`updated_at` fields and expose them as `compact_fields` in the record detail template context. Add a `@field_meta(highlight=True)` decorator or `highlight: bool = False` field annotation to let developers pin specific fields to the panel.

### 7.2 Activity Feed Architecture
Create a shared `apg_activity` table:
```sql
CREATE TABLE apg_activity_event (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(100) NOT NULL,  -- e.g. "Member", "Transaction"
    entity_id   UUID NOT NULL,
    event_type  VARCHAR(50) NOT NULL,   -- create|update|comment|assign|status_change|delete
    actor_id    UUID REFERENCES apg_user(id),
    field_name  VARCHAR(100),           -- for update events
    old_value   TEXT,
    new_value   TEXT,
    body        TEXT,                   -- for comment events (markdown)
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON apg_activity_event (entity_type, entity_id, created_at DESC);
```
Event icons (Tailwind + SVG): comment=chat-bubble, update=pencil, create=plus-circle, status-change=arrow-right-circle, assign=user-plus, delete=trash.

### 7.3 Command Palette Data Architecture
The palette needs a `/api/v1/search` endpoint returning:
```json
{
  "results": [
    {"type": "record", "entity": "Member", "id": "uuid", "label": "John Doe", "subtitle": "member@example.com", "url": "/member/uuid"},
    {"type": "action", "label": "Create Member", "shortcut": "C", "url": "/member/new"}
  ]
}
```
Generator auto-registers all entities with their display fields into a search registry. Front-end: debounced 150ms input → htmx GET → replace palette results. Keyboard: ArrowUp/Down navigate highlighted item; Enter follows `url`; Escape closes.

### 7.4 Inline Edit — htmx Pattern
```python
# Generated route per entity+field
@bp.route('/api/<entity>/<uuid:record_id>/field/<field_name>', methods=['PATCH'])
def update_field(entity, record_id, field_name):
    value = request.json.get('value')
    # validate field_name is in allowlist
    record = db.session.get(ModelClass, record_id)
    setattr(record, field_name, value)
    db.session.commit()
    # Return updated field fragment for htmx swap
    return render_template('fragments/field_value.html', record=record, field=field_name)
```
Front-end: `hx-patch="/api/member/{{id}}/field/name" hx-vals="{'value': inputEl.value}" hx-swap="outerHTML"` on the field container.

### 7.5 Skeleton Screen — htmx Integration
htmx has a built-in `hx-indicator` mechanism. Add a skeleton template that matches the shape of the real content:
```html
<div hx-get="/entity/list" hx-trigger="load" hx-indicator="#skeleton-rows">
  <!-- Content appears here after load -->
</div>
<div id="skeleton-rows" class="htmx-indicator">
  <!-- 5 skeleton rows matching table structure -->
</div>
```
CSS: `.htmx-indicator { display: none; } .htmx-request .htmx-indicator { display: block; }`

### 7.6 Saved Views — DB Schema
```sql
CREATE TABLE apg_saved_view (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES apg_user(id),
    entity_type VARCHAR(100) NOT NULL,
    name        VARCHAR(200) NOT NULL,
    filters     JSONB NOT NULL DEFAULT '[]',
    sort_field  VARCHAR(100),
    sort_dir    VARCHAR(4) DEFAULT 'asc',
    columns     JSONB,            -- ordered list of visible column names
    is_default  BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT now()
);
```

---

## 8. Mobile Responsiveness Approach

**Current APG state:** Tailwind CSS provides responsive utilities but generated templates use fixed sidebar layouts that break below 768px.

**Target pattern (drawn from enterprise platforms):**
- `< 640px (sm)`: Single column. Sidebar hidden. Top nav collapses to hamburger. Table becomes card-list (each row is a card, top field is the title). Bulk actions → bottom sheet. Kanban → single-column scroll. Highlights panel → 2 fields only.
- `640-1024px (md)`: Two-column possible. Sidebar collapses to icon-only (48px). Table shows 3-4 priority columns only (hide via `hidden md:table-cell` on low-priority columns). Detail page: tabs stack vertically.
- `> 1024px (lg)`: Full layout. Sidebar expanded (240px). Table shows all columns. Record detail: 2-column (content + sidebar panel).

**Implementation:** Add `sm:hidden`, `md:table-cell`, `lg:block` Tailwind modifiers on generated column/sidebar elements. Generator should mark which columns are "priority" (first 3) and "secondary" (rest).

**Bottom sheet pattern** for mobile bulk actions:
```html
<div id="bulk-sheet" class="fixed bottom-0 left-0 right-0 z-40 bg-white border-t border-slate-200 p-4 lg:hidden transform translate-y-full transition-transform">
  <!-- Bulk action buttons -->
</div>
```

---

## 9. Accessibility Standards

All enterprise platforms target WCAG 2.1 AA. Key requirements for APG-generated UIs:
- All form inputs must have associated `<label>` (via `for`/`id` or `aria-label`).
- Status badges must not rely on color alone — include text or icon.
- Modals must trap focus and restore focus on close. `role="dialog"` + `aria-modal="true"`.
- Toasts must use `role="status"` (non-urgent) or `role="alert"` (urgent). Auto-dismiss toasts must not be the only means of communicating critical info.
- Tables must have `<thead>` with `scope="col"` on `<th>` elements. Sortable columns: `aria-sort="ascending|descending|none"`.
- Keyboard nav: Tab moves between interactive elements. Arrow keys move within compound widgets (tab panels, dropdown menus, data tables). Esc closes overlays.
- Skip-to-content link (`<a href="#main-content" class="sr-only focus:not-sr-only">Skip to content</a>`) at top of every page.
- Color contrast: 4.5:1 for normal text, 3:1 for large text and UI components.
- Focus visible: `focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2` (Tailwind) on all interactive elements.

---

## 10. Research Gaps & Open Questions

1. APG's actual `_html_page()` and UI template generation code was not found in the searched files. The current generated UI appears to be in-memory Python string generation, not templated HTML files. The Tailwind+htmx UI referenced in recent commits needs to be located for precise gap analysis.
2. APG's field metadata system is unclear — whether models carry enough type annotations to drive automatic widget selection (currency vs plain number, etc.) needs verification.
3. Whether APG generates unique per-entity Flask routes for field-level PATCH (needed for inline edit) or only GET/POST CRUD endpoints.
4. The wizard/guided-workflow feature added in recent commits should be reviewed — it may partially address the "process stepper" gap.
5. Mobile CSS breakpoint behavior of current generated templates has not been tested.

---

## Sources

- [Salesforce Lightning Design System — Component Overview](https://www.lightningdesignsystem.com/components/overview/)
- [SLDS Design Tokens (LWC Guide)](https://developer.salesforce.com/docs/platform/lwc/guide/create-components-css-design-tokens.html)
- [SLDS Styling Hooks (CSS Custom Properties)](https://developer.salesforce.com/docs/platform/lwc/guide/create-components-css-custom-properties.html)
- [SLDS Standard Design Tokens — force:base](https://developer.salesforce.com/docs/atlas.en-us.lightning.meta/lightning/tokens_standard_force_base.htm)
- [Salesforce LWC Inline Edit (Data Table)](https://developer.salesforce.com/docs/platform/lwc/guide/data-table-inline-edit.html)
- [Salesforce Inline Edit in List Views](https://help.salesforce.com/s/articleView?id=xcloud.basics_customviews_lv_lex_considerations.htm&language=en_US&type=5)
- [Salesforce Compact Layouts — Trailhead](https://trailhead.salesforce.com/content/learn/modules/lex_customization/lex_customization_compact_layouts)
- [Salesforce Lightning App Builder — Record Pages](https://trailhead.salesforce.com/content/learn/modules/lightning_app_builder/lightning_app_builder_recordpage)
- [Salesforce Dynamic Forms](https://hicglobalsolutions.com/blog/how-to-build-dynamic-record-pages-in-the-lightning-app-builder-with-dynamic-forms/)
- [Salesforce Activity Timeline](https://sfdcpenguin.com/blog/activity-timeline-view-activity-data-with-ease/)
- [Salesforce Global Search Guidelines](https://www.lightningdesignsystem.com/guidelines/search/global/)
- [SLDS 2 Overview — nsiqinfotech](https://nsiqinfotech.com/guide-to-salesforce-lightning-design-system/)
- [SAP Design System — experience.sap.com](https://experience.sap.com/fiori-design/)
- [SAP Fiori for Web](https://www.sap.com/design-system/fiori-design-web)
- [SAP Fiori Floorplan Overview](https://www.sap.com/design-system/fiori-design-web/v1-38/page-types/floorplan-overview)
- [SAP Fiori Object Page Info](https://www.sap.com/design-system/fiori-design-web/v1-70/page-types/floorplans/object-page/usage?external)
- [SAP Fiori Elements Overview](https://pathlock.com/blog/sap-fiori/sap-fiori-elements/)
- [ServiceNow Horizon Design System](https://horizon.servicenow.com/)
- [ServiceNow Horizon Components](https://horizon.servicenow.com/workspace/components)
- [ServiceNow Horizon Structure](https://horizon.servicenow.com/workspace/basics/structure)
- [ServiceNow Horizon What's New](https://horizon.servicenow.com/getting-started/whats-new)
- [Microsoft Dynamics 365 UI/UX Design Guide](https://learn.microsoft.com/en-us/dynamics365/guidance/develop/introduction-customer-engagement-ui-ux-design-guide)
- [Dynamics 365 UI Components (Model-Driven)](https://learn.microsoft.com/en-us/dynamics365/guidance/develop/ui-ux-component-details-model-driven-apps)
- [Dynamics 365 Key UI/UX Design Principles](https://learn.microsoft.com/en-us/dynamics365/guidance/develop/ui-ux-design-principles)
- [HubSpot Record Design Rethinking](https://product.hubspot.com/blog/rethinking-hubspots-record-design-with-usability-in-mind)
- [HubSpot Record Page Layout](https://knowledge.hubspot.com/records/work-with-records)
- [HubSpot UI Extension Components](https://developers.hubspot.com/docs/apps/developer-platform/add-features/ui-extensions/ui-components/overview)
- [HubSpot Spring 2025 UI Updates](https://developers.hubspot.com/blog/app-cards-updates-spring-spotlight-2025)
- [Monday.com Vibe Design System](https://developer.monday.com/apps/docs/vibe-design-system)
- [Monday.com Vibe Figma Kit](https://www.figma.com/community/file/940242815162888749/vibe-ui-kit-by-monday-com)
- [Monday.com GitHub — UI Style](https://github.com/mondaycom/monday-ui-style)
- [Monday.com Gantt Chart View](https://support.monday.com/hc/en-us/articles/360015643840-The-Gantt-Chart-View-and-Widget)
- [Notion Design Tokens & Typography — DesignMD](https://designmd.cc/benchmarks/notion)
- [Notion Database Views Help](https://www.notion.com/help/views-filters-and-sorts)
- [Notion UI Breakdown — Sidebar](https://medium.com/@quickmasum/ui-breakdown-of-notions-sidebar-2121364ec78d)
- [Linear Delightful Design Patterns](https://gunpowderlabs.com/2024/12/22/linear-delightful-patterns)
- [Linear Design Analysis — getdesign.md](https://getdesign.md/linear.app/design-md)
- [Linear Design Refresh — linear.app](https://linear.app/now/behind-the-latest-design-refresh)
- [Linear Redesign Part II](https://linear.app/now/how-we-redesigned-the-linear-ui)
- [Linear Keyboard Shortcuts](https://keycombiner.com/collections/linear/)
- [Linear UX Patterns — SaaSUI](https://www.saasui.design/application/linear)
- [Retool Rebuilt UI Component Library](https://retool.com/blog/redesigned-ui-component-library)
- [Retool Q3 2024 Developer Day](https://retool.com/blog/retool-developer-day-q3-2024)
- [Retool Components Docs](https://docs.retool.com/apps/concepts/components/)
- [Appsmith vs Budibase Comparison](https://budibase.com/blog/alternatives/appsmith-vs-budibase/)
- [Budibase vs Appsmith — dhiwise](https://www.dhiwise.com/post/budibase-vs-appsmith-internal-tools-comparison)
- [Enterprise Data Table UX Patterns](https://www.pencilandpaper.io/articles/ux-pattern-analysis-enterprise-data-tables)
- [Bulk Action UX — eleken](https://www.eleken.co/blog-posts/bulk-actions-ux)
- [Inline Edit Best Practices — UXDWorld](https://uxdworld.com/inline-editing-in-tables-design/)
- [PatternFly Inline Edit Guidelines](https://www.patternfly.org/components/inline-edit/design-guidelines/)
- [Cloudscape Inline Edit Pattern](https://cloudscape.design/patterns/resource-management/edit/inline-edit/)
- [Command Palette UX Patterns — Medium](https://medium.com/design-bootcamp/command-palette-ux-patterns-1-d6b6e68f30c1)
- [Command Palette Design — Solomon](https://solomon.io/designing-command-palettes/)
- [Enterprise UI 2026 — hashbyt](https://hashbyt.com/blog/enterprise-ui-design)
- [Carbon Design System — Loading Pattern](https://carbondesignsystem.com/patterns/loading-pattern/)
- [Carbon Design System — Notification Pattern](https://carbondesignsystem.com/patterns/notification-pattern/)
- [Skeleton Screen Design — LogRocket](https://blog.logrocket.com/ux-design/skeleton-loading-screen-design/)
- [Empty State Best Practices — Pencil & Paper](https://www.pencilandpaper.io/articles/empty-states)
- [Empty State UX — eleken](https://www.eleken.co/blog-posts/empty-state-ux)
- [KPI Card Types — Qlik Community](https://community.qlik.com/t5/Member-Articles/KPI-Cards-on-a-Dashboard-What-Types-Exist/ta-p/2543950)
- [Power BI KPI Dashboard Guide — EPC Group](https://www.epcgroup.net/power-bi-kpi-visuals-dashboard-guide-2026)
- [SAP Fiori Best Practices](https://www.sap.com/design-system/fiori-design-web/v1-96/discover/sap-products/sap-s4hana-only/best-practices-for-designing-sap-fiori-apps)
- [Enterprise Design Systems Best Practices](https://www.softkraft.co/enterprise-design-systems/)
- [Sidebar Design Best Practices 2026](https://www.alfdesigngroup.com/post/improve-your-sidebar-design-for-web-apps)
- [WCAG Keyboard Accessibility — UXPin](https://www.uxpin.com/studio/blog/wcag-211-keyboard-accessibility-explained/)
- [WAI-ARIA Authoring Practices Guide](https://w3c.github.io/aria/)
- [SaaS Onboarding Patterns — appcues](https://www.appcues.com/blog/saas-user-onboarding)
- [Dark Mode Design Systems — Muzli](https://muz.li/blog/dark-mode-design-systems-a-complete-guide-to-patterns-tokens-and-hierarchy/)
- [Linear Dark Mode Design Tokens](https://chyshkala.com/blog/why-linear-design-systems-break-in-dark-mode-and-how-to-fix-them)
