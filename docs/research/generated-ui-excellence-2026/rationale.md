# Rationale — Design Decisions & Tradeoffs

Decisions made during this research and why. Covers what was considered, what was rejected, and what assumptions were made.

---

## Decision 1: Tier Structure for the Checklist

**Decision**: Three tiers — Must-Emit (every page), Widget-Specific, Excellence Differentiators.

**Alternatives considered**:
- Flat checklist (all items equal priority) — rejected because it provides no guidance on what to do first
- WCAG-level organization (Level A, AA, AAA) — rejected because WCAG level doesn't map to implementation effort or impact
- Page-type organization (form page, list page, kanban) — rejected because many items apply across all page types

**Why three tiers**: Tier 1 items are blockers — a generated page missing any of them fails a basic compliance audit. Tier 2 items are conditional — only relevant when the widget is present. Tier 3 items are differentiators — they separate "compliant" from "excellent." This maps directly to a phased implementation plan: ship Tier 1 first, Tier 2 as each widget type is added, Tier 3 as polish.

---

## Decision 2: System Fonts Only (No Web Fonts)

**Decision**: Generated pages use system font stacks exclusively. No Google Fonts, no self-hosted WOFF2.

**Alternatives considered**:
- Google Fonts with `font-display: swap` — provides aesthetic control but adds a network dependency and LCP risk; `font-display: swap` causes CLS during font swap
- Self-hosted WOFF2 with preload — eliminates third-party dependency but requires font files in the generated app and adds build complexity (subsetting, WOFF2 encoding)
- Variable fonts — requires font files; same issues as self-hosted

**Why system fonts**: Admin/internal tooling has no brand requirement for custom typography. System fonts are already cached by the OS, render at zero LCP cost, match the OS's rendering quality, and work in print. The aesthetic difference between Inter and `-apple-system, BlinkMacSystemFont, "Segoe UI"` is negligible in data-dense UI. The performance difference is significant. For any generated UI that needs custom fonts (white-label, brand-critical), the caller can override via CSS custom properties — the generator's default is system fonts.

---

## Decision 3: All CSS Inline, No External Stylesheets

**Decision**: Emit all CSS in `<style>` blocks in `<head>`. No `<link rel="stylesheet">`.

**Alternatives considered**:
- Single external CSS file per generated app — cleaner HTML, browser can cache it; requires a static file server and a separate asset pipeline
- CSS-in-JS equivalent (style attributes) — per-element inline styles; defeats cascade, prevents media queries, defeats `prefers-color-scheme`
- CSS custom properties with a small external base file + inline overrides — hybrid; still has one external request on first load

**Why inline**: APG generates self-contained files. "Self-contained" means no external dependencies. An external CSS file breaks this constraint. Inline `<style>` blocks are render-blocking in the sense that they must be parsed before paint — but since they're inline, there's no network round-trip. The browser parses them synchronously as part of HTML parsing. This is the optimal critical CSS pattern. Deduplication across pages is a non-issue for generated admin apps where each page is independently served.

**Tradeoff accepted**: Larger HTML payload per page. For typical admin UI (a few KB of CSS), this is irrelevant. If a generated page's CSS exceeds ~20KB, it should be investigated as a generator bug (excessive specificity, duplicated rules).

---

## Decision 4: No ARIA Unless Complete

**Decision**: Generated HTML emits ARIA attributes only when the full pattern is implemented (role + keyboard contract + states + labels). Native HTML elements preferred over ARIA-decorated divs.

**Alternatives considered**:
- Always emit ARIA for richness — rejected by data: pages with ARIA average 59 errors vs 42 without; incomplete ARIA is actively harmful
- Developer-provided ARIA — rejected; APG is a generator, not a framework; the developer shouldn't need to understand ARIA patterns to get correct output

**Why native-first**: `<button>` gives you role="button", focusability, keyboard activation (Enter/Space), and disabled state for free. `<div role="button" tabindex="0">` gives you the role but requires manual keyboard handling, manual disabled management, and is easily broken. The generator should use `<button>` for interactive elements, `<a>` for navigation, `<input>` for inputs, `<select>` for dropdowns, `<details>`/`<summary>` for accordions — and only reach for ARIA when native elements genuinely can't express the semantic (tabs, dialogs, live regions, kanban columns).

---

## Decision 5: "Move" Button Pattern for Kanban, Not Aria-Grabbed

**Decision**: Kanban cards have a visible "Move" button (or accessible action menu) as the primary interaction model, with optional mouse drag-drop as enhancement.

**Alternatives considered**:
- `aria-grabbed` on draggable elements — deprecated in ARIA 1.1; still used but no guaranteed AT support
- `role="gridcell"` with keyboard arrow navigation — valid but complex; requires full grid pattern implementation
- Custom `aria-roledescription="draggable card"` — descriptive only, no keyboard behavior implied

**Why "Move" button**: WCAG 2.5.7 (AA) requires a single-pointer alternative to dragging. A "Move" button satisfying this requirement also satisfies keyboard users and AT users. The button+menu pattern is the recommendation from Salesforce UX (the most cited expert source on accessible DnD) and is what React Aria implements. It's also simpler to generate correctly than a full keyboard-drag implementation. Mouse drag-drop (via HTML5 DnD API or pointer events) can be layered on as progressive enhancement without touching the ARIA structure.

---

## Decision 6: FormDraft Excludes Password and Sensitive Fields

**Decision**: The FormDraft class explicitly deletes `password`, `password_confirm`, `cvv`, `secret`, and `token` fields before writing to localStorage.

**Alternatives considered**:
- Encrypt localStorage data — rejected; key management in client JS is unsolvable; XSS that can read localStorage can also read the decryption key
- Use sessionStorage instead — sessionStorage clears on tab close; appropriate for within-session persistence but not cross-session draft recovery
- Allowlist approach (only save fields listed as safe) — more robust but requires field metadata from the generator schema; viable enhancement

**Why explicit exclusion**: Defense-in-depth. The generator knows which fields are type="password". Excluding them prevents any possibility of credentials persisting in plaintext localStorage. The exclusion should be name-based (not type-based) because `type="text"` inputs with name="token" or name="api_key" are equally sensitive.

---

## Decision 7: prefers-reduced-motion as Default-Off

**Decision**: All animations are inside `@media (prefers-reduced-motion: no-preference)`. The zero-motion state is the default.

**Alternatives considered**:
- Default to animated, reduce inside `@media (prefers-reduced-motion: reduce)` — common pattern; requires explicitly disabling every animation for reduced-motion users
- No animations at all — simplest; appropriate for dense data UI, may look unpolished

**Why default-off**: Admin/data UI is not a marketing site. Motion serves utility purposes (drawer open/close, toast entry, drag feedback). Starting from no-motion and adding animation for users who want it is the correct accessibility-first approach. It also means the CSS is simpler: `transition: none` by default, `transition: X 200ms ease` in the media query. No risk of forgetting to disable an animation.

---

## Decision 8: Landmark Structure as Generator Invariant

**Decision**: Every generated page shell emits the same landmark structure: `<header>`, `<nav aria-label="Primary">`, `<main id="main" tabindex="-1">`, optional `<aside>`, `<footer>`. Skip link is always the first element in `<body>`.

**Alternatives considered**:
- Opt-in landmark structure (developer configures) — rejected; landmark structure should not be optional
- Django-style retrofit (add landmarks as bugs are filed) — rejected; get it right from day one
- Role attributes on divs instead of semantic elements — rejected; semantic elements express the same landmark roles without ARIA and are more robust

**Why invariant**: The value of landmarks is consistency. Screen reader users navigate by landmark (usually `F6` or swipe gesture). If some pages have `<main>` and some don't, or some have labeled `<nav>` and some don't, users can't build a mental model of the app's structure. The generator enforces this invariant for free — every page gets the same shell.

**tabindex="-1" on `<main>`**: Required to ensure programmatic focus (from the skip link `href="#main"`) actually lands in the main element. Without it, some browsers focus the element but don't scroll correctly. This is a subtle but important detail.

---

## Decision 9: content-visibility Only with contain-intrinsic-size

**Decision**: Never emit `content-visibility: auto` without `contain-intrinsic-size: auto <estimated-height>`.

**Alternatives considered**:
- Emit content-visibility alone — found a specific community bug report confirming CLS spikes without contain-intrinsic-size; rejected
- Use virtual scrolling instead — requires significant JS; incompatible with server-rendered HTML and no-build-step constraint

**Why paired**: Without `contain-intrinsic-size`, the browser treats off-screen content as having zero height. When the user scrolls near it, the actual height is calculated and the page shifts (CLS). With `contain-intrinsic-size: auto <estimated>`, the browser uses the estimate as a placeholder and then the actual height after rendering, with the `auto` keyword caching the measured size after first render. This makes content-visibility: auto CLS-safe.

The estimated height for `contain-intrinsic-size` should be a reasonable guess (e.g., 400px for a section with mixed content). It doesn't need to be exact — the `auto` prefix means the browser will cache the real value after the first render.

---

## Decision 10: Scope Limited to WCAG 2.2 AA (Not 3.0/Silver)

**Decision**: Research targeted WCAG 2.2 AA. WCAG 3.0 (Silver) was not researched in depth.

**Rationale**: WCAG 3.0 is still in Working Draft as of July 2026. No stable success criterion numbers exist. The EU Accessibility Act references WCAG 2.1 AA. WCAG 2.2 AA is the current legal compliance target, is a superset of 2.1, and has been ISO-standardized (ISO/IEC 40500:2025). Implementing 2.2 AA provides maximum coverage against current requirements. When WCAG 3.0 reaches Candidate Recommendation status with stable SCs, a follow-up research pass is warranted.

---

## Assumptions Made

1. **Generated apps serve internal/admin users, not general public.** This affects the resilience requirements (full offline-first vs. simple offline detection) and visual requirements (system fonts acceptable, no brand requirements).

2. **Python/Flask environment without npm available at runtime.** All JS must be vanilla or CDN-loaded (deferred). No Webpack, Vite, or PostCSS build step.

3. **English as primary language, but lang attribute should use the actual locale.** The generator should accept a locale parameter and emit the correct BCP 47 language tag.

4. **Forms are the primary interaction mode.** Optimistic UI and draft persistence are most valuable for forms, not for read-only views.

5. **localStorage is available.** Generated apps are served over HTTPS to modern browsers. localStorage availability is a reasonable assumption. The implementation handles localStorage errors (full storage, private browsing mode) gracefully with try/catch.
