# World-class examination of the generated UI surface

**Date:** 2026-10-25
**Mode:** Audit — read-only against the generator as it stands after `86479adc`.
**Method:** inspected generator templates (`compiler/templates/*.j2`), asset contract (`compiler/assets/*`), the generated-app CSS (`apg.css`), and a compiled sample app launched on `127.0.0.1:8080`. Cross-referenced with the `ux-excellence` checklist (spatial coherence, token topology, state-space, content, motion, APCA, cross-platform, cognitive load).

## Verdict

**Not world-class in absolute terms. World-class *relative to what the project scope is honest about* (a single self-contained Flask app with no build step).**

The surface APG ships is genuinely deep for its constraints — breadth of screen families, dark mode, CSP nonces, PWA/offline, HTMX partials, command palette, density toggle, OpenUI generative UI — but it carries several principled gaps vs. current world-class baselines (Shadcn/Linear/Vercel baseline): no `prefers-reduced-motion` in CSS, no `clamp()` fluid type, no logical properties for RTL, no ARIA live on most regions, no container queries, no motion system beyond fade/opacities, and no `aria-live` pairing for toast/SSE updates. These are style-bound, additive fixes — none requires architectural change.

## Evidence drawn

### Asset contract
Generated apps ship these locally (asset contract documented at `docs/generated_ui.md`):

```
apg.css  htmx.min.js  sortable.min.js  uplot.min.js  uplot.min.css
apg-charts.js  apg-sse.js  manifest.webmanifest  sw.js  icon.svg
apg-openui.js (since the OpenUI commits)
```

Zero CDN dependencies. Vendored with license tracking in `licenses.md`.

### Token topology
- Theme via CSS custom properties through `/theme.css`: `--apg-bg`, `--apg-bg-panel`, `--apg-text`, `--apg-primary` — plus `color-mix(in srgb, ...)` for hover states (9 occurrences).
- Light + dark: `@media (prefers-color-scheme: dark)`, `:root[data-theme="dark"]`, `:root.dark` — three dark-mode selectors present at `apg.css:45-67`.
- Semantic tokens: yes, surfaces / borders / text / hover / success-warning-danger families.
- **Gap:** dark mode is `prefers-color-scheme` only — no JS-driven systematic dark mode with persistence, no `color-scheme: light dark` baseline on `:root` (only on dark selectors). Toggle exists (`.apg-theme-toggle` at `:138`), but persistence path isn't visible as a classically documented ARIA toggle.
- **Gap:** no APCA / Lc contrast verification. Plain `color-mix` 92% background blend relies on visual judgment, not measured luminance.

### Spatial grid
- Based on 8px scale (`--apg-space-3`, `--apg-space-4`, `--apg-space-5`), grid templates at `repeat(4, minmax(0,1fr))`. Good.
- No container queries (`container-type` = 0 hits). Responsive grids use plain media queries only. For compound screens with sidebar this is fine; nested cards can't respond to parent width. Modern baseline violation.

### Typography
- `font-size:` declarations at 15 sites, all in `rem` units, no `clamp()` (0 occurrences). **Gap:** no fluid topology. At large widths type stays fixed, at small widths may wrap. Suggest `clamp(1rem, 0.875rem + 0.45vw, 1.25rem)`.
- `em` is used 417x for padding and margins — fine for spacing that scales.
- Line length / reading-science rules (60 chars) not visible.

### Motion & choreography
- `@keyframes apg-shimmer` (skeleton), 4 keyframes total in `apg.css`.
- `transition:` 9 sites. No `prefers-reduced-motion` (0 hits). **A11y blocker**: users with vestibular restrictions see every animation.
- No layout-thrashing check in code (no `will-change` (0 hits), `transform:` 29 — so perf-friendly).

### State-space (loading / empty / error)
- Skeleton loader classes: `.apg-skeleton`, `.apg-skeleton-row`; `.htmx-request .apg-content-area { opacity .6 }` for in-flight loading.
- Empty/error states: HTML structure in entity_list.html.j2 has empty-state markup with guidance.
- No optimistic-UI rollback. No SWR (stale-while-revalidate) beyond Cache-Control being present.
- **State coverage gap:** multi-state surface is shippable but no offline *writes* queue (only cached read-only GET via service worker). And `IndexedDB` is never used (`0` references).
- SSE event stream via `apg-sse.js`, real-time subscription present.

### Content & microcopy
- Tables have caption containing entity name + `col scope="col"` (test `test_generated_table_caption_contains_entity_name`, `test_generated_table_header_cells_have_col_scope`).
- `role="banner"`, `role="navigation"`. ARIA expanded on density toggle button.
- **Gap:** no automated microcopy audit; `role="dialog"` only in 1 of the 15 templates. So most screens lack modal semantic regions except agent console.
- `aria-live` = 0 hits in templates. SSE/live updates aren't announced to assistive tech. **A11y gap.**

### Accessibility
- Skip link (`.apg-skip-link`) present.
- CSP nonce wired without `unsafe-inline` (test).
- Tables: captions + col scope.
- Density toggle button with ARIA.
- **Gaps:** no skip-to-nav for deep tables, no `aria-live="polite"` on toast or SSE, no focus management on HTMX-driven fragment insertions (a known challenge with HTMX), focus outline on links is browser-default. Color contrast not verified (no APCA in pipeline, no axe run).
- The CTA test (`tests/test_generated_app_a11y.py` 4 assertions only) is not comprehensive for the listed critical-paths. Real-world a11y requires screen-reader testing.

### OpenUI (the innovation)
- Library spec emission (`openui_library.json`, `openui_prompt.txt`) since commit 381e7050.
- Browser `apg-openui.js` renderer mounted on `component-console`/agent-console result surfaces.
- Spec conformance tests (`test_openui_conformance.py`, `test_openui_render.py`) cover 53 of 58 fixtures (5 server-only).
- This is a genuinely novel artifact in the codegen space — no comparable upstream low-code generator ships a model-facing UI component library derived from its own schema.

### Cross-platform / input modality
- Touch targets ≥ 44px: verified at `.apg-status-row` (min-height 40px — **slightly below** 44px WCAG), `.apg-sidebar-link` likely ≥ 40.
- Logical properties for RTL: 0 hits — uses `margin-left`/`padding-right`-style; `margin-inline` absent. SoRTL not truly supported despite `dir="ltr"` markup trying.
- Safe-area insets: 0 hits.
- Keyboard navigation: command palette uses keyboard events, tabbing should work. No documented keyboard shortcut help.

### Cognitive load
- Tables are dense (4+ columns, status rows). Command palette provides global search — a genuine assist. Filters panel with saved-query links.
- Density toggle (compact) exists.
- Forms should be ≤7 fields — the `entity_list` create drawer flows off the main screen.

## Strengths (what's world-class already)

1. **Zero CDN / offline-ready** PWA with service worker + manifest — rare for codegen.
2. **OpenUIlang render-surface** (server + browser), with a model-facing component library.
3. **CSP-nonce security** with no `unsafe-inline`.
4. **Information architecture breadth**: 15 screen families, 5 dashboard types, 6 data views, ER-relationship navigator — *far more screen variety than most generators*.
5. **Single self-contained file** (~12k emitted lines) means no build system can break — theme tokens are at compile time coupled with JS runtime.
6. **Dark mode** tokens + toggle, with system-preference fallback.

## Gaps to close for "worldclass"

| Priority | Gap | Effort | Evidence |
| --- | --- | --- | --- |
| P1 | `prefers-reduced-motion` wrapping (zero) | S | `grep -c` = 0 |
| P1 | `aria-live` on dynamic regions (SSE toast, status) | S | 0 in templates |
| P1 | `clamp()` fluid typography | M | 0 in CSS |
| P2 | RTL logical properties (`margin-inline-start`, etc.) | M (315 rem sites, 90 margin-* lines) | 0 logical-property uses |
| P2 | Container queries for nested cards | M | 0 |
| P3 | APCA/LWC contrast audit in build gate | L | Currently no color-science verification |
| P3 | Mapped keyboard shortcut help + focus management on HTMX swap | M | Partial in command palette |
| P3 | Tailor skeleton geometry to final layout (CLS < 0.1) | M | Shimmer present, geometry not matched |
| P4 | `safe-area-inset` for notched displays | S | 0 |
| P4 | Touch target guarantee >= 44px | S | `.apg-status-row` is 40px Min-height |

## Conclusion

The generated UI has the *breadth* and *security* of a world-class product — 15 screen families, CSP-hard, dark-mode-tokenised, PWA shell, OpenUI generative surface — but lacks the *polish baselines* of motion-safety, accessibility on dynamic updates, fluid typography, and true RTL. These gaps are additive CSS/HTML improvements on top of an already-deep surface, not structural changes. None of them require touching the (massive, production-risk) `compiler/code_generator.py` f-string, only template/CSS hygiene. The OpenUI integration is the most defensible novel UI feature and should be marketed as such.

**Recommendation:** schedule a "Wave UI-polish" that runs `prefers-reduced-motion`, `clamp()`, `aria-live`, logical properties, safe-area, and `>=44px` touch targets through the generated CSS without generator changes. Then add an automated Lighthouse gate (CLS + lighthouse score >= 90) for generated apps.