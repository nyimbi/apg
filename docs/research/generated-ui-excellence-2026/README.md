# Generated UI Excellence 2026
## World-class accessibility, performance, and resilience for server-rendered HTML (no build step)

**Scope**: APG DSL compiler emitting self-contained Flask apps with inline HTML/CSS/JS — dashboards, kanban boards, record detail pages, forms, login pages, workflow wizards.

**Bar**: WCAG 2.2 AA + Core Web Vitals "good" band + resilience UX patterns. Every item below is implementable in Python string-generated HTML/CSS/JS with no npm, no bundler.

---

## 1. WCAG 2.2 AA — The Actual Bar

### 1.1 The Six Universal Failures (WebAIM Million 2026)

The 2026 WebAIM Million report (1,000,000 home pages, 56,114,377 errors detected) shows 96% of all WCAG failures collapse into six categories, unchanged for seven years:

| Rank | Failure | % of pages | Notes |
|------|---------|-----------|-------|
| 1 | Low contrast text | **83.9%** | Avg 34 instances/page; 4.5:1 normal, 3:1 large text |
| 2 | Missing alt text | 53.1% | 16.2% of all images |
| 3 | Missing form labels | 51.0% | 33.1% of inputs unlabelled |
| 4 | Empty links | 46.3% | Icon-only `<a>` with no text or aria-label |
| 5 | Empty buttons | 30.6% | Icon-only `<button>` with no text |
| 6 | Missing document language | 13.5% | `<html lang="en">` absent |

Pages with ARIA present averaged **59.1 errors** vs 42 on pages without ARIA — ARIA misuse actively worsens outcomes. Generated UI must emit correct ARIA or no ARIA.

### 1.2 WCAG 2.2 New Success Criteria (ISO/IEC 40500:2025)

WCAG 2.2 became an ISO standard on 21 October 2025. The nine new criteria:

| SC | Name | Level | What generated UI breaks by default |
|----|------|-------|--------------------------------------|
| 2.4.11 | Focus Not Obscured (Minimum) | **AA** | Sticky headers/toasts cover focused element entirely |
| 2.4.12 | Focus Not Obscured (Enhanced) | AAA | No part of focused element hidden |
| 2.4.13 | Focus Appearance | **AAA** | Focus ring ≥2px thick perimeter, 3:1 contrast adjacent |
| 2.5.7 | Dragging Movements | **AA** | Kanban drag-drop has no keyboard/single-pointer alternative |
| 2.5.8 | Target Size (Minimum) | **AA** | Touch targets <24×24 CSS px with no spacing offset |
| 3.2.6 | Consistent Help | **A** | Help widget moves position across pages |
| 3.3.7 | Redundant Entry | **A** | Multi-step wizard re-asks name/email already entered |
| 3.3.8 | Accessible Authentication (Minimum) | **AA** | CAPTCHA on login; knowledge-only auth (secret questions) |
| 3.3.9 | Accessible Authentication (Enhanced) | AAA | Object recognition puzzles in auth flow |

**AA-level new failures most likely in generated admin UIs:**
- `2.4.11`: any sticky nav or toast notification covering a focused field → emitting `z-index` stacking context must account for focused elements
- `2.5.8`: action icon buttons in table rows, kanban card controls, pagination arrows — all need explicit `min-width: 44px; min-height: 44px` or 24px with `margin` spacing
- `2.5.7`: every drag interaction (kanban, sortable lists) needs an accessible alternative — a "Move" button opening a menu of column options
- `3.3.7`: login forms that re-ask username already entered; checkout/wizard flows re-asking address fields
- `3.3.8`: CAPTCHA on generated login pages is a WCAG 2.2 AA violation unless alternatives exist

### 1.3 Previously Existing AA Criteria Generated UIs Routinely Fail

These existed in WCAG 2.1 but are structurally violated by templating approaches:

- **1.3.1 Info and Relationships**: Tables used for layout; data not marked with `<th scope>`, `<caption>`, `<thead>`; form error messages not `aria-describedby` linked to their input
- **1.3.2 Meaningful Sequence**: JavaScript-reordered fields break DOM reading order
- **1.4.3 Contrast (Minimum)**: Placeholder text (`color: #aaa`) is not exempted — it still needs 4.5:1 if it's the only label
- **2.1.1 Keyboard**: Custom dropdowns, date pickers, modal dialogs built without keyboard trap management
- **2.4.1 Bypass Blocks (A)**: No skip link; navigation repeated without landmark-based bypass
- **2.4.3 Focus Order**: Modals that don't trap focus; drawers that leave focus in the document behind
- **4.1.2 Name, Role, Value**: Custom widgets (accordions, tabs, expandable rows) without role, aria-expanded, aria-controls

---

## 2. WAI-ARIA APG Patterns for Generated Widget Types

The APG patterns are **informative, not normative** — but they represent the consensus keyboard contract that screen reader users expect. Deviation is allowed only when the result still meets the underlying WCAG SC.

### 2.1 Tabs (Dashboards, Detail Pages)

```
role="tablist" on container
role="tab" aria-selected="true/false" aria-controls="panel-id" on each tab
role="tabpanel" id="panel-id" aria-labelledby="tab-id" on each panel
Keyboard: Arrow keys navigate between tabs (roving tabindex)
          Tab enters the panel, Shift+Tab returns to tab strip
```

Pattern: [WAI-ARIA Tabs](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/)

### 2.2 Dialog / Modal (Forms, Confirmations, Wizards)

```
role="dialog" aria-modal="true" aria-labelledby="dialog-title-id"
Focus trapped inside on open; first focusable element receives focus
Escape closes; focus returns to trigger element on close
Backdrop click closes (optional, but common)
Sentinel focusable elements at start/end to cycle focus
```

Pattern: [WAI-ARIA Dialog](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/)

### 2.3 Kanban / Drag-Drop (WCAG 2.5.7 + Keyboard Alternative)

The WAI-ARIA APG has no dedicated "kanban" pattern as of 2026. Best-practice synthesis from Salesforce UX, React Aria, and WCAG 2.5.7:

```
Each column: role="group" aria-label="Column: In Progress (3 items)"
Each card: role="article" or role="listitem" in role="list"
Drag handle: role="button" aria-label="Move [card title]" aria-pressed="false"
  On activate: opens a menu (role="menu") of column options
  Example items: "Move to Backlog", "Move to In Progress", "Move to Done"
  This satisfies WCAG 2.5.7 as the single-pointer alternative

Drag-and-drop if present additionally:
  aria-grabbed (deprecated but still used by some AT) or custom solution
  Announce drag start, drag over column, drop via aria-live="assertive"
  Keyboard: Space to grab, Arrow to move between columns, Space to drop, Escape to cancel
```

### 2.4 Accordion (Sidebar Filters, Detail Sections)

```
role="button" aria-expanded="true|false" aria-controls="section-id" on header
id="section-id" on collapsible content
Keyboard: Enter/Space toggles; Tab moves to next focusable element
```

Pattern: [WAI-ARIA Accordion](https://www.w3.org/WAI/ARIA/apg/patterns/accordion/)

### 2.5 Alert / Status Notifications (Toasts, Banners)

```
role="alert" aria-live="assertive" for errors (announces immediately)
role="status" aria-live="polite" for success/info (waits for quiet moment)
role="alertdialog" aria-modal="true" for confirmation dialogs requiring response
Avoid injecting into role="alert" regions repeatedly — each injection announces
```

### 2.6 Data Table (List Views, Reports)

```
<table>
  <caption>Vendor List — 47 records, sorted by Name ascending</caption>
  <thead><tr><th scope="col" aria-sort="ascending">Name</th>...</tr></thead>
  <tbody>
    <tr><th scope="row">Acme Corp</th>...</tr>
  </tbody>
</table>
aria-sort="ascending|descending|none" on sortable <th> elements
Row selection: role="checkbox" or native <input type="checkbox"> with aria-label
```

### 2.7 Landmark Structure (Every Page)

```html
<body>
  <a href="#main" class="skip-link">Skip to main content</a>
  <header role="banner">  <!-- site-level header -->
    <nav role="navigation" aria-label="Primary">...</nav>
  </header>
  <main id="main" tabindex="-1">  <!-- tabindex="-1" ensures focus lands here -->
    <nav role="navigation" aria-label="Breadcrumb">...</nav>
    <!-- page content -->
  </main>
  <aside role="complementary" aria-label="Filters">...</aside>
  <footer role="contentinfo">...</footer>
</body>
```

Skip link CSS (visible on focus, hidden otherwise):
```css
.skip-link {
  position: absolute;
  left: -9999px;
  top: 0;
  z-index: 9999;
}
.skip-link:focus {
  left: 0;
  padding: 8px 16px;
  background: #000;
  color: #fff;
}
```

---

## 3. Static HTML Checks (No Browser Required)

These checks can run on the generated HTML string in a CI pipeline before the app even starts.

### 3.1 Python-Based Static Analysis

**curlylint** (pip install curlylint): AST-based linter for Jinja/Django/Nunjucks templates, 7 accessibility rules built-in:
- `html_has_lang`: `<html lang>` required
- `aria_role`: valid role values
- `django_forms_rendering`: proper form rendering patterns
- `image_alt`: alt attribute present on `<img>`
- `meta_viewport`: no `user-scalable=no`
- `label_content`: labels have content
- `tabindex_no_positive`: no `tabindex > 0`

**html-validator** (Node, runs on generated output files):
```bash
npm install -g html-validate
echo '<generated-page.html>' | html-validate --stdin
```
Checks: proper nesting, deprecated attributes, missing required attributes on interactive elements.

**axe-core CLI** (headless, via pa11y):
```bash
npm install -g pa11y
pa11y --runner axe --level AA file:///path/to/page.html
```

### 3.2 Pure Python String/Regex Checks (Zero Dependencies)

These are fast enough to run in unit tests on every generated page:

```python
import re
from html.parser import HTMLParser

STATIC_CHECKS = [
    # SC 3.1.1 — Language of Page
    (r'<html[^>]+lang=["\']\w', "html[lang] attribute missing"),
    
    # SC 1.3.1 — skip link present
    (r'href=["\']#(main|content|maincontent)["\']', "Skip link missing"),
    
    # SC 2.4.1 — main landmark
    (r'<main[\s>]|role=["\']main["\']', "main landmark missing"),
    
    # SC 1.3.1 — no positive tabindex
    (r'tabindex=["\'][1-9]', "Positive tabindex found (breaks focus order)"),
    
    # SC 1.1.1 — img without alt
    (r'<img(?![^>]*\balt=)[^>]*>', "img missing alt attribute"),
    
    # SC 4.1.2 — button without text or aria-label
    (r'<button[^>]*>\s*</button>', "Empty button"),
    
    # SC 3.3.2 — input without label
    # (complex, use HTMLParser for this)
    
    # SC 1.4.12 — viewport disables zoom
    (r'user-scalable\s*=\s*no', "user-scalable=no violates 1.4.4"),
]
```

### 3.3 Heading Hierarchy Check

```python
def check_heading_hierarchy(html: str) -> list[str]:
    """Detect heading level skips (h1 -> h3 without h2)."""
    headings = re.findall(r'<h([1-6])', html, re.IGNORECASE)
    errors = []
    if headings and headings[0] != '1':
        errors.append(f"First heading is h{headings[0]}, expected h1")
    for i in range(1, len(headings)):
        diff = int(headings[i]) - int(headings[i-1])
        if diff > 1:
            errors.append(f"Heading jumps from h{headings[i-1]} to h{headings[i]}")
    if headings.count('1') > 1:
        errors.append("Multiple h1 elements found")
    return errors
```

### 3.4 Form Label Association Check

```python
def check_form_labels(html: str) -> list[str]:
    """Every input (except hidden/submit/button/image) must have a label."""
    errors = []
    # Find all inputs with id
    inputs = re.findall(r'<input[^>]+id=["\']([^"\']+)["\'][^>]*>', html)
    for input_id in inputs:
        if not re.search(rf'for=["\']({re.escape(input_id)})["\']', html):
            if not re.search(rf'id=["\']({re.escape(input_id)})["\'][^>]*aria-label', html):
                errors.append(f"Input #{input_id} has no associated label")
    return errors
```

---

## 4. Core Web Vitals 2026 — Server-Rendered HTML

### 4.1 Thresholds (Google, 75th percentile)

| Metric | Good | Needs Improvement | Poor |
|--------|------|------------------|------|
| LCP (Largest Contentful Paint) | ≤ 2.5s | 2.5s–4.0s | > 4.0s |
| INP (Interaction to Next Paint) | ≤ 200ms | 200ms–500ms | > 500ms |
| CLS (Cumulative Layout Shift) | ≤ 0.10 | 0.10–0.25 | > 0.25 |

INP replaced FID in March 2024. It measures every interaction across the page lifetime (75th percentile), not just the first.

### 4.2 LCP for Server-Rendered Pages

The LCP element is typically the page's `<h1>`, a hero image, or a large block of text. For server-rendered HTML:

**What matters:**
- **Critical CSS inline in `<head>`**: Everything needed to render above-the-fold content without a network round-trip. External CSS is render-blocking.
- **System font stacks**: Eliminate web font LCP delays entirely.
  ```css
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, "Noto Sans", sans-serif,
               "Apple Color Emoji", "Segoe UI Emoji";
  ```
- **No render-blocking JS in `<head>`**: All `<script>` tags get `defer` or `async`.
  - `defer`: executes after HTML parsed, in order — correct for app code
  - `async`: executes immediately when loaded, out of order — for independent 3rd-party scripts only
- **Image dimensions**: Always set `width` and `height` on `<img>`. Prevents CLS from image layout shifts. Browser computes aspect ratio from these before image loads.
- **`<link rel="preload">` for above-fold images**: `<link rel="preload" as="image" href="/hero.webp">`

**For inline HTML/CSS/JS generation** (APG's model):
- Emit all CSS inline in `<style>` blocks in `<head>` — already zero extra network requests
- System fonts only — no Google Fonts
- Inline SVG for icons (no icon font, no external sprite)
- Any JS that doesn't affect initial render: `defer` or at end of `<body>`

### 4.3 CLS for Server-Rendered Pages

CLS measures unexpected layout shifts. Common causes in generated UI:

| Cause | Fix |
|-------|-----|
| Images without width/height | Always emit `width="X" height="Y"` |
| Web fonts loading | Use system fonts; if web fonts needed, `font-display: swap` + `font-size-adjust` |
| Dynamic content injected above existing content | Reserve space with `min-height`; inject below or use `position: absolute` overlays |
| `content-visibility: auto` without `contain-intrinsic-size` | Pair as `content-visibility: auto; contain-intrinsic-size: auto 300px` |
| Ads / embeds without dimensions | Not applicable to generated admin UI |

**`content-visibility: auto` is Baseline (September 2025)** — safe to use in all modern browsers. Gives up to 7× rendering improvement on long pages (reports, audit logs, activity feeds). Pair with `contain-intrinsic-size` to avoid CLS:
```css
.lazy-section {
  content-visibility: auto;
  contain-intrinsic-size: auto 400px;  /* estimated height placeholder */
}
```

### 4.4 INP for Server-Rendered Vanilla JS

INP = input delay + processing time + presentation delay, measured at 75th percentile.

**Key rules:**
1. **Break up long tasks**: Any JS work >50ms on the main thread delays the next interaction. Use `scheduler.yield()` (Chrome 115+) or `setTimeout(fn, 0)` to yield between chunks.
2. **DOM size**: Large DOM (>1,500 nodes) inflates style recalculation on every interaction. Generated pages with 47-column tables, deep nested accordions, large kanban boards are at risk.
3. **Layout thrashing**: Don't read layout properties (offsetHeight, getBoundingClientRect) then write styles in the same synchronous block. Batch reads, then writes.
4. **Avoid client-side HTML rendering**: For interactions that need to insert content, prefer server-side rendering via form submission or fetch + innerHTML from a server-rendered fragment. Client-side HTML string rendering via `innerHTML = template(data)` blocks paint.
5. **Event delegation**: One event listener on `document` or `<main>` rather than N listeners on N elements. Reduces listener overhead on large tables/lists.

```js
// Pattern: yield on long tasks
async function processLargeDataset(items) {
  for (let i = 0; i < items.length; i++) {
    process(items[i]);
    if (i % 50 === 0) {
      await new Promise(r => setTimeout(r, 0));  // yield to browser
    }
  }
}
```

---

## 5. Resilience UX

### 5.1 Offline Detection

```js
// Minimal, production-grade pattern
const networkStatus = {
  online: navigator.onLine,
  init() {
    window.addEventListener('online',  () => this.setOnline(true));
    window.addEventListener('offline', () => this.setOnline(false));
  },
  async verify() {
    // navigator.onLine=true is unreliable (connected to router, no internet)
    try {
      await fetch('/health', { method: 'HEAD', cache: 'no-store' });
      return true;
    } catch { return false; }
  },
  setOnline(state) {
    this.online = state;
    document.querySelector('[data-offline-banner]')
      ?.classList.toggle('hidden', state);
    if (state) this.flushQueue();
  }
};
```

**Offline banner pattern** (always emit in generated HTML):
```html
<div data-offline-banner role="alert" aria-live="assertive" class="offline-banner hidden">
  You're offline. Changes will be saved when reconnected.
</div>
```

**Key caveat**: `navigator.onLine = false` is reliable (definitely offline). `navigator.onLine = true` means "connected to a network" not "connected to the internet." Always verify with a real fetch for critical operations.

### 5.2 Optimistic UI with Rollback (Vanilla JS)

For form submissions and actions (status changes, record saves):

```js
function optimisticUpdate(element, newState, rollbackState, serverFetch) {
  // 1. Apply optimistic state immediately
  applyState(element, newState);
  
  // 2. Send to server
  serverFetch()
    .then(response => {
      if (!response.ok) throw new Error(response.statusText);
      // optionally update with server-returned data
    })
    .catch(err => {
      // 3. Rollback on failure
      applyState(element, rollbackState);
      showError(`Failed: ${err.message}. Your change was not saved.`);
    });
}
```

Use optimistic updates for: status changes, star/bookmark toggles, soft deletes, ordering.
Use pessimistic (wait for server) for: financial transactions, irreversible deletes, payment submissions.

### 5.3 Error Boundaries in Vanilla JS

No `try/catch` equivalent for DOM rendering. Strategies:

```js
// Global error boundary
window.addEventListener('error', (event) => {
  console.error('Uncaught error:', event.error);
  showFallbackUI('Something went wrong. Please refresh.');
  event.preventDefault();  // prevent default browser error display
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  showFallbackUI('A background operation failed.');
  event.preventDefault();
});

function showFallbackUI(message) {
  const fallback = document.getElementById('error-fallback');
  if (fallback) {
    fallback.textContent = message;
    fallback.hidden = false;
    fallback.focus();  // announce to screen reader
  }
}
```

Always emit an `id="error-fallback"` hidden element in generated pages.

### 5.4 Form Draft Persistence

```js
class FormDraft {
  constructor(formId, storageKey = `draft:${formId}`) {
    this.form = document.getElementById(formId);
    this.key = storageKey;
    if (!this.form) return;
    this.restore();
    this.form.addEventListener('input', this.debounce(() => this.save(), 800));
    this.form.addEventListener('submit', () => this.clear());
  }
  
  save() {
    const data = Object.fromEntries(new FormData(this.form));
    // Never persist passwords or sensitive fields
    delete data.password; delete data.password_confirm; delete data.cvv;
    try {
      localStorage.setItem(this.key, JSON.stringify({
        ts: Date.now(), data
      }));
    } catch (e) { /* localStorage full — fail silently */ }
  }
  
  restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(this.key));
      if (!saved) return;
      const age = Date.now() - saved.ts;
      if (age > 7 * 24 * 60 * 60 * 1000) { this.clear(); return; } // 7-day expiry
      Object.entries(saved.data).forEach(([name, value]) => {
        const field = this.form.querySelector(`[name="${name}"]`);
        if (field && field.type !== 'password') field.value = value;
      });
      this.showRestoredBanner(saved.ts);
    } catch { /* corrupt storage — ignore */ }
  }
  
  clear() { localStorage.removeItem(this.key); }
  
  showRestoredBanner(ts) {
    const banner = this.form.querySelector('[data-draft-banner]');
    if (banner) {
      banner.textContent = `Draft restored from ${new Date(ts).toLocaleString()}`;
      banner.hidden = false;
    }
  }
  
  debounce(fn, ms) {
    let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }
}
```

**Security**: Never store `password`, `token`, `cvv`, `secret` field values. Use sessionStorage (cleared on tab close) for auth flows.

### 5.5 `prefers-reduced-motion`

Every animation in generated UI must respect this:

```css
/* Base: no animations */
.transition-base {
  transition: none;
}

/* Only animate if user hasn't requested reduced motion */
@media (prefers-reduced-motion: no-preference) {
  .transition-base {
    transition: opacity 200ms ease, transform 200ms ease;
  }
  .toast { animation: slide-in 200ms ease; }
  .kanban-card-drag { transition: box-shadow 150ms ease; }
}
```

**Generated HTML/CSS template pattern**: emit all animations inside `@media (prefers-reduced-motion: no-preference)`. Baseline: no motion.

### 5.6 Dark Mode (`prefers-color-scheme`)

CSS custom properties + system preference detection, no JS required:

```css
:root {
  --color-bg: #ffffff;
  --color-text: #111111;
  --color-surface: #f5f5f5;
  --color-border: #d1d5db;
  --color-primary: #2563eb;
  --color-primary-text: #ffffff;
  /* ... */
}

@media (prefers-color-scheme: dark) {
  :root {
    --color-bg: #0f172a;
    --color-text: #f1f5f9;
    --color-surface: #1e293b;
    --color-border: #334155;
    --color-primary: #3b82f6;
    --color-primary-text: #ffffff;
  }
}
```

**Modern CSS (2025)**: `light-dark()` function allows same-property dark/light definition:
```css
:root { color-scheme: light dark; }
body { background: light-dark(#fff, #0f172a); color: light-dark(#111, #f1f5f9); }
```
Baseline Newly Available September 2025 — safe in all modern browsers.

**User override**: Provide a `<button>` that adds `data-theme="dark"` to `<html>` and persists to localStorage. This overrides `prefers-color-scheme` via CSS `[data-theme="dark"] :root { ... }`.

### 5.7 Print Stylesheets

Admin-facing generated apps often need print-ready reports and record detail pages:

```css
@media print {
  /* Hide navigation chrome */
  header, nav, aside, footer,
  .btn, .action-bar, .pagination,
  [data-offline-banner], [data-draft-banner] { display: none !important; }
  
  /* Expand collapsed content */
  details { open: true; }
  
  /* Page setup */
  @page { margin: 2cm; size: A4 portrait; }
  
  /* Typography */
  body { font-size: 11pt; color: #000; background: #fff; }
  a[href]::after { content: " (" attr(href) ")"; font-size: 9pt; }
  
  /* Prevent orphan headings */
  h1, h2, h3, h4, h5, h6 { break-after: avoid; }
  tr, li { break-inside: avoid; }
  
  /* Tables */
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #000; padding: 4pt; }
  thead { display: table-header-group; }  /* repeat on each page */
  
  /* Show URLs for links */
  .no-print-url a[href]::after { content: ""; }
}
```

---

## 6. Competitive Bar Analysis

### 6.1 Django Admin (2024-2025)

**Targeting**: WCAG 2.2 AA + ATAG 2.0 (authoring tool accessibility).

**What Django admin does by default:**
- Form field labels properly associated via `for`/`id`
- `<main>` landmark, `<header>` with `role="banner"`, navigation landmarks added in 5.0
- Help text and errors programmatically associated with fields (fixed Django 5.0)
- Admin action log communicates entry types to screen readers
- Viewport meta does not disable text scaling

**Known gaps (as of early 2024):**
- Windows High Contrast mode — 5 identified failures (team blog)
- Overall accessibility score: 80.5/100 vs industry average higher
- Significant room for improvement in admin components

**Django admin score**: Adequate foundation, not aspirational. Many generated apps using it score below industry average on accessibility.

### 6.2 shadcn/ui (React, build-step) — The Component Standard

shadcn/ui is the current best-practice reference for component-level accessibility (via Radix UI primitives):

**Passes (34/48 components, WCAG 2.2 AA)**: Buttons, inputs, dialogs, most form elements.

**Notable failures** (relevant as anti-patterns to avoid generating):
- **Focus ring**: `focus-visible:ring-1 ring-ring/50` fails 3:1 non-text contrast ratio (2.4.11, 2.4.13)
- **Combobox**: Missing `aria-haspopup`, `aria-expanded`, `aria-controls`
- **Data Table**: No `<caption>`, no row count announcements, no `aria-sort`
- **Charts**: SVG with no accessible alternative — screen readers see empty
- **Carousel**: Autoplay ignores `prefers-reduced-motion`
- **Toast**: 4-second duration too brief for cognitive accessibility (should be 5–7s minimum, dismissible)
- **Date Picker**: Navigation buttons below 24×24px minimum (2.5.8)

**Lesson**: Even the best-regarded component library has ~10% components with significant gaps. Generated HTML must be more explicit than component libraries that depend on runtime composition.

### 6.3 Phoenix LiveView / Hotwire (Rails) — The Interaction Model

Both are server-rendered HTML with progressive enhancement — the same architectural model as APG.

**What they do well:**
- Form submissions return server-rendered HTML fragments — no client-side state management
- Turbo/LiveView morphing preserves focus during DOM updates (critical for keyboard users — avoids focus loss on re-render)
- Server-side validation with inline error messages is the default, not an afterthought
- Hotwire modals with accessible implementation are documented by community

**What they don't do by default:**
- ARIA patterns are left to the developer — no enforced landmark structure
- Accessibility is framework-optional, not framework-enforced
- No built-in offline detection or draft persistence

**Lesson for APG**: The compiler can enforce more than any framework by generating guaranteed-correct ARIA in the DSL-to-HTML translation layer.

### 6.4 Refine (React, build-step)

Refine uses Radix + shadcn with AI-assisted scaffolding. Key properties:
- Accessibility built-in via Radix primitives
- Tree-shakable, server-component-friendly
- AI-generated Refine code inherits component-level accessibility guarantees

**Gap that APG can beat Refine on**: Refine requires a build step, npm, React hydration overhead. APG inline HTML/CSS/JS has **zero hydration cost** — page is interactive immediately on HTML parse. This makes LCP and INP intrinsically better for static interaction patterns.

---

## 7. Prioritized Implementation Checklist

### TIER 1 — Must-Emit (Every Generated Page)

These are non-negotiable; any generated page missing these fails WCAG AA:

- [ ] `<html lang="en">` (or locale-appropriate code) — SC 3.1.1
- [ ] `<meta charset="UTF-8">` in `<head>`
- [ ] `<meta name="viewport" content="width=device-width, initial-scale=1">` — **no** `user-scalable=no` — SC 1.4.4
- [ ] Skip link as first child of `<body>`: `<a href="#main" class="skip-link">Skip to main content</a>` — SC 2.4.1
- [ ] `<main id="main" tabindex="-1">` landmark — SC 2.4.1
- [ ] `<header>`, `<nav aria-label="Primary">`, `<footer>` landmarks — SC 2.4.1
- [ ] Every `<img>` has `alt=""` (decorative) or `alt="description"` (informative) — SC 1.1.1
- [ ] Every `<input>` has `<label for="id">` or `aria-label` or `aria-labelledby` — SC 1.3.1, 4.1.2
- [ ] Every `<button>` has visible text or `aria-label` — SC 4.1.2
- [ ] Every `<a>` has visible text or `aria-label` — SC 2.4.4
- [ ] `<title>` is unique and descriptive: `"Entity Name | App Name"` — SC 2.4.2
- [ ] Heading hierarchy: single `<h1>`, no skipped levels — SC 1.3.1
- [ ] Error messages linked to their input via `aria-describedby` — SC 1.3.1, 3.3.1
- [ ] All CSS in `<style>` in `<head>` (no render-blocking `<link>`) — LCP
- [ ] System font stack only — LCP
- [ ] All `<script>` tags have `defer` (or placed at end of `<body>`) — LCP
- [ ] All `<img>` have explicit `width` and `height` attributes — CLS
- [ ] Color contrast ≥ 4.5:1 for normal text, ≥ 3:1 for large text and UI components — SC 1.4.3
- [ ] Focus ring with ≥ 3:1 contrast against adjacent background — SC 2.4.11, 2.4.13
- [ ] Touch targets ≥ 24×24 CSS px (or with spacing offset) — SC 2.5.8
- [ ] `@media (prefers-reduced-motion: no-preference)` wraps all animations — WCAG, UX
- [ ] `@media (prefers-color-scheme: dark)` dark mode tokens — UX
- [ ] `@media print` stylesheet — UX
- [ ] `<div data-offline-banner role="alert" class="hidden">` in body — Resilience
- [ ] `<div id="error-fallback" hidden>` in body — Resilience

### TIER 2 — Widget-Specific (Emit When Widget Present)

- [ ] **Tables**: `<caption>`, `<thead>`, `<th scope="col|row">`, `aria-sort` on sortable columns
- [ ] **Tabs**: `role="tablist"`, `role="tab" aria-selected aria-controls`, `role="tabpanel" aria-labelledby`
- [ ] **Dialogs**: `role="dialog" aria-modal="true" aria-labelledby`, focus trap, Escape closes, focus returns
- [ ] **Kanban columns**: `role="group" aria-label`, "Move" button alternative to drag, `aria-live` for announcements
- [ ] **Accordions**: `aria-expanded`, `aria-controls` on trigger; `id` matching `aria-controls` on panel
- [ ] **Toast notifications**: `role="status" aria-live="polite"` (success/info) or `role="alert" aria-live="assertive"` (error); minimum 5s duration + manual dismiss
- [ ] **Forms with multi-step**: prefill previous answers (SC 3.3.7); no re-asking same field
- [ ] **Login**: no cognitive CAPTCHA without alternative (SC 3.3.8); magic link or biometric alternative
- [ ] **Drag interactions**: keyboard alternative via button+menu (SC 2.5.7)
- [ ] **Long pages** (>50 elements): `content-visibility: auto; contain-intrinsic-size: auto Xpx` on off-screen sections
- [ ] **Forms**: FormDraft autosave for forms with >3 fields or long content; excludes password fields

### TIER 3 — Excellence Differentiators

- [ ] `aria-live` status region for form validation (announces errors without focus move)
- [ ] Column summary in table caption: "47 vendors, sorted by Name"
- [ ] Row count update in `aria-live` region on filter: "Showing 12 of 47 vendors"
- [ ] `prefers-color-scheme` with user override toggle (persisted to localStorage)
- [ ] `font-size-adjust` for system font stack metric consistency
- [ ] Global `window.onerror` + `unhandledrejection` error boundary
- [ ] Offline queue: queue failed mutations, flush on reconnect
- [ ] INP: event delegation pattern (single listener on container, not N on N elements)
- [ ] INP: `scheduler.yield()` for any processing >50ms (polyfill with `setTimeout(0)`)
- [ ] Print stylesheet with repeat table headers, page break rules
- [ ] Reduced motion: instant transitions for all state changes under `prefers-reduced-motion: reduce`

---

## 8. Research Gaps & Open Questions

1. **WCAG 2.3 timeline**: W3C has a working draft for WCAG 2.3/3.0 (Silver). No stable SC numbers yet. Nothing to implement for compliance, but worth monitoring for stricter cognitive/COGA requirements.

2. **EU Accessibility Act (EAA) June 2025**: Requires WCAG 2.1 AA for EU market digital products/services. WCAG 2.2 AA is a superset and satisfies EAA. Specific enforcement by member state varies — no single definitive penalty structure yet.

3. **`scheduler.yield()` availability**: Available in Chrome 115+, Firefox not yet (as of early 2026). Polyfill with `await new Promise(r => setTimeout(r, 0))` is safe but less efficient. Monitor caniuse.

4. **`content-visibility` and search indexing**: Google has stated Googlebot respects `content-visibility: auto`. Monitor for changes — hidden content ranking implications unresolved.

5. **Accessible authentication without CAPTCHA for generated Flask apps**: Flask-Login + email magic link is the cleanest implementation. Rate-limiting + device fingerprinting as fraud alternative to CAPTCHA. No widely-adopted turnkey Flask solution as of 2025.

6. **Kanban keyboard pattern official standardization**: WAI-ARIA APG has no formal "kanban" pattern. React Aria's implementation (Adobe) is the current de-facto reference. Should be checked periodically for official APG addition.

7. **`light-dark()` CSS function browser parity**: Baseline Newly Available September 2025 — Chrome 101+, Firefox 120+, Safari 17.5+. Safe for modern browsers; older Safari versions need `@media` fallback.

8. **Storage persistence across origins**: The `beforeunload` + `navigator.sendBeacon()` pattern for draft saves is unreliable on mobile (background tabs get killed). IndexedDB with Service Worker is more robust but adds significant complexity for a no-build-step generator.
