# Sources

All URLs accessed during this research session. Date: 2026-07-10.

---

## WCAG 2.2 Standards & Criteria

| Title | URL | Summary |
|-------|-----|---------|
| Web Content Accessibility Guidelines (WCAG) 2.2 | https://www.w3.org/TR/WCAG22/ | Normative spec; all 87 SC with understanding docs |
| What's New in WCAG 2.2 (WAI) | https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/ | Authoritative list of all 9 new SC with level and plain description — fetched directly |
| WCAG 2 Overview (WAI) | https://www.w3.org/WAI/standards-guidelines/wcag/ | Entry point for WCAG family |
| WCAG 2.2 vs 2.1: 9 New Requirements Your Site Probably Fails | https://adaquickscan.com/blog/wcag-2-2-iso-standard-2025 | ISO/IEC 40500:2025 adoption date (Oct 2025) confirmed here |
| WCAG 2.2 Complete Compliance Guide 2025 | https://www.allaccessible.org/blog/wcag-22-complete-guide-2025 | Summary of all 56 AA criteria |
| WCAG 2.2 AA Summary and Checklist | https://www.levelaccess.com/blog/wcag-2-2-aa-summary-and-checklist-for-website-owners/ | 2026 compliance checklist from Level Access |
| WCAG 2.2 New Success Criteria: Complete Implementation Guide | https://testparty.ai/blog/wcag-22-new-success-criteria | Implementation patterns for all 9 new SC |
| What Are All 87 WCAG 2.2 Success Criteria? | https://testparty.ai/blog/wcag-22-success-criteria-list | Full enumeration |
| WCAG 2.2 Focus Appearance (Minimum) | https://testparty.ai/blog/wcag-focus-appearance-minimum | Detailed 2.4.11 implementation |
| WCAG 2.2 for Brand Sites: 2026 Fix List | https://www.monotonomo.com/journal/wcag-2-2-brand-sites-2026/ | Focus, targets, dragging, authentication failures |
| WCAG 2.2 Updates — Deque University | https://dequeuniversity.com/resources/wcag-2.2/ | Code examples for each new SC |
| Contrast requirements for WCAG 2.2 Level AA | https://www.makethingsaccessible.com/guides/contrast-requirements-for-wcag-2-2-level-aa/ | 3:1 non-text contrast rationale |
| WCAG 2.2 Explained: AudioEye | https://www.audioeye.com/post/wcag-22/ | Focus/target/auth implementation |
| Understanding SC 2.4.1 Bypass Blocks (W3C) | https://www.w3.org/TR/UNDERSTANDING-WCAG20/navigation-mechanisms-skip.html | Normative understanding doc |
| WCAG 2.4.1 Bypass Blocks — 2025 Guide | https://testparty.ai/blog/wcag-2-4-1-bypass-blocks-2025-guide | Skip link + landmark dual approach |
| WebAIM: Skip Navigation Links | https://webaim.org/techniques/skipnav/ | Implementation best practices |

---

## Failure Statistics

| Title | URL | Summary |
|-------|-----|---------|
| WebAIM Million 2026 Report | https://webaim.org/projects/million/ | **Fetched directly.** 56.1M errors/1M pages; 6 failure types = 96% of errors; 95.9% pages fail; pages with ARIA avg 59 errors vs 42 without |
| Accessibility Risks in AI-Generated Interfaces | https://brics-econ.org/accessibility-risks-in-ai-generated-interfaces-wcag-and-real-world-failures | **Fetched directly.** SC mapping for AI-generated failures; 73% of AI alt text wrong/meaningless |
| 95% of Websites Still Fail Basic WCAG Standards | https://signalscv.com/2026/07/95-percent-of-websites-still-fail-basic-wcag-standards-and-accessibe-research-explains-why | accessiBe research on ongoing failure rates |
| Web Accessibility 2026 Complete Guide | https://assist-software.net/business-insights/web-accessibility-2026-complete-guide-wcag-compliance | Compliance landscape summary |
| Common WCAG Failures (craigabbott.co.uk) | https://www.craigabbott.co.uk/decks/common-wcag-failures/ | Practitioner deck on recurring failures |

---

## WAI-ARIA APG Patterns

| Title | URL | Summary |
|-------|-----|---------|
| ARIA Authoring Practices Guide (APG) | https://www.w3.org/WAI/ARIA/apg/ | Root of all APG patterns |
| APG Patterns Index | https://www.w3.org/WAI/ARIA/apg/patterns/ | **Fetched directly.** Full list of 30 patterns |
| Dialog (Modal) Pattern | https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/ | Focus trap, aria-modal, escape handling |
| Alert and Message Dialogs Pattern | https://www.w3.org/WAI/ARIA/apg/patterns/alertdialog/ | alertdialog vs dialog |
| Tabs Pattern | https://www.w3.org/WAI/ARIA/apg/patterns/tabs/ | tablist/tab/tabpanel keyboard contract |
| Listbox Pattern | https://www.w3.org/WAI/ARIA/apg/patterns/listbox/ | role=listbox for selectables |
| Keyboard Navigation Patterns for Complex Widgets (UXPin, 2026) | https://www.uxpin.com/studio/blog/keyboard-navigation-patterns-complex-widgets/ | Roving tabindex, composite widget contracts |
| 4 Major Patterns for Accessible Drag and Drop (Salesforce UX) | https://medium.com/salesforce-ux/4-major-patterns-for-accessible-drag-and-drop-1d43f64ebf09 | Definitive survey of drag-drop a11y approaches |
| React Aria: Kanban Board Example | https://react-aria.adobe.com/examples/kanban | Adobe's reference implementation |
| React Aria: Drag and Drop | https://react-aria.adobe.com/dnd | Full DnD keyboard + screen reader contract |
| The Road to Accessible Drag and Drop Part 2 (Vispero) | https://vispero.com/resources/the-road-to-accessible-drag-and-drop-part-2/ | Screen reader AT perspective |
| Drag-and-Drop Design: Accessibility Best Practices | https://appinstitute.com/drag-and-drop-design-accessibility-best-practices/ | "Move" button pattern for WCAG 2.5.7 |
| Design patterns and WCAG (TetraLogical) | https://tetralogical.com/blog/2024/08/09/design-patterns-wcag/ | APG patterns vs WCAG normative requirements |
| APG Explained (Elementor) | https://elementor.com/blog/apg-explained/ | 2025 overview of APG usage |

---

## Static Analysis Tools

| Title | URL | Summary |
|-------|-----|---------|
| curlylint (official site) | https://www.curlylint.org/ | Python AST linter for curly-brace templates; 7 a11y rules |
| curlylint accessibility linting rules | https://www.curlylint.org/blog/accessibility-linting-rules/ | Per-rule documentation |
| curlylint GitHub | https://github.com/thibaudcolas/curlylint | Source; last significant activity 2023 |
| curlylint on PyPI | https://pypi.org/project/curlylint/ | Install: `pip install curlylint` |
| pa11y (GitHub) | https://github.com/pa11y/pa11y | Node.js CLI accessibility runner; Puppeteer-based |
| pa11y (npm) | https://www.npmjs.com/package/pa11y | Version 9 requires Node 20/22/24 |
| A review of HTML linters (chezsoi.org) | https://chezsoi.org/lucas/blog/a-review-of-html-linters.html | Comparative benchmark: html-validate wins on thoroughness |
| Automated testing with axe-core and pa11y (DWP) | https://accessibility-manual.dwp.gov.uk/best-practice/automated-testing-using-axe-core-and-pa11y | UK gov manual on combined axe + pa11y |
| Web Accessibility Testing with axe and pa11y (2025) | https://johal.in/web-accessibility-testing-automated-compliance-checking-with-axe-and-pa11y-2025/ | CI integration guide |
| axe accessibility check (webhint) | https://webhint.io/docs/user-guide/hints/hint-axe/ | webhint/axe integration |
| Web Accessibility Python (PyCon 2021) | https://accessibility-loves-python.vercel.app/ | Python-based accessibility tools survey |
| 36 HTML Static Analysis Tools (analysis-tools.dev) | https://analysis-tools.dev/tag/html | Catalog of HTML linters |
| Automated Accessibility Part 1: Linting (DEV) | https://dev.to/steady5063/automated-accessibility-part-1-linting-5378 | curlylint + eslint-jsx-a11y comparison |
| Combining axe-core and PA11Y (Craig Abbott) | https://craigabbott.co.uk/blog/combining-axe-core-and-pa11y/ | How to run both in one CI pass |

---

## Core Web Vitals

| Title | URL | Summary |
|-------|-----|---------|
| Core Web Vitals (Google Search Central) | https://developers.google.com/search/docs/appearance/core-web-vitals | Authoritative thresholds + ranking impact |
| How Core Web Vitals Thresholds Were Defined (web.dev) | https://web.dev/articles/defining-core-web-vitals-thresholds | 75th percentile methodology explanation |
| Optimize INP (web.dev) | https://web.dev/articles/optimize-inp | **Fetched directly.** Input delay, processing time, presentation delay; yield patterns |
| Optimize Long Tasks (web.dev) | https://web.dev/articles/optimize-long-tasks | 50ms task budget, scheduler.yield() |
| content-visibility: the new CSS property (web.dev) | https://web.dev/articles/content-visibility | 7× rendering improvement on long pages |
| content-visibility causing CLS (web-vitals-feedback) | https://groups.google.com/g/web-vitals-feedback/c/Gr41J4pYzoE | contain-intrinsic-size fix documented here |
| content-visibility (MDN) | https://developer.mozilla.org/en-US/docs/Web/CSS/content-visibility | Baseline Newly Available September 2025 |
| content-visibility (DebugBear) | https://www.debugbear.com/blog/content-visibility-api | Performance measurement with before/after |
| content-visibility for INP (NitroPack) | https://nitropack.io/blog/content-visibility-inp/ | INP improvement case studies |
| Render-blocking requests (Chrome DevTools) | https://developer.chrome.com/docs/performance/insights/render-blocking | Critical CSS inline guidance |
| Optimize resource loading (web.dev) | https://web.dev/learn/performance/optimize-resource-loading | defer/async/preload patterns |
| Optimize CSS Delivery (Google PageSpeed) | https://developers.google.com/speed/docs/insights/OptimizeCSSDelivery | Inline critical CSS guidance |
| How to eliminate render-blocking resources (LogRocket) | https://blog.logrocket.com/eliminate-render-blocking-resources-css-javascript/ | Practical defer/async examples |
| Core Web Vitals 2026 guide (corewebvitals.io) | https://www.corewebvitals.io/core-web-vitals | 2026 threshold summary |
| Core Web Vitals 2026 (webhelpagency) | https://webhelpagency.com/blog/core-web-vitals-2026/amp/ | INP/LCP/CLS optimization strategies |
| INP Input Delay Causes and Fixes | https://www.corewebvitals.io/core-web-vitals/interaction-to-next-paint/input-delay | Long task anatomy |
| CSS Optimization Guide 2025 (DEV) | https://dev.to/satyam_gupta_0d1ff2152dcc/css-optimization-guide-2025-speed-up-your-website-best-practices-code-examples-31ib | System fonts, critical CSS |

---

## Resilience UX

| Title | URL | Summary |
|-------|-----|---------|
| How to Detect Online and Offline Status in JavaScript | https://blog.openreplay.com/detect-online-offline-status-javascript/ | navigator.onLine limitations, event listeners |
| Detecting network state with JavaScript (prototyp) | https://prototyp.digital/blog/detecting-network-state-with-javascript | HEAD fetch verification pattern |
| navigator.onLine reliability (xjavascript.com) | https://www.xjavascript.com/blog/fetch-api-how-to-determine-if-an-error-is-a-network-error/ | error.cause.name = "NetworkError" (Chrome 94+) |
| Error Boundaries for JS UI Patterns (NamasteDev) | https://namastedev.com/blog/error-boundaries-for-js-ui-patterns-library-agnostic/ | window.onerror + unhandledrejection pattern |
| Designing a Resilient UI (DEV) | https://dev.to/istealersn_dev/designing-a-resilient-ui-handling-failures-gracefully-in-frontend-applications-1p9l | Error states, retry logic, offline fallback |
| Persist Form State in the Browser (OpenReplay) | https://blog.openreplay.com/persist-form-state-browser/ | localStorage vs sessionStorage vs IndexedDB choice |
| Implementing Auto-Save Functionality (DhiWise) | https://www.dhiwise.com/post/implementing-auto-save-on-forms | 500ms-2s debounce recommendation |
| Offline-First App Development Guide (Medium) | https://medium.com/@hashbyt/offline-first-app-development-guide-cfa7e9c36a52 | IndexedDB + sync queue patterns |
| Building an Offline-Ready Form (We Learn Code) | https://welearncode.com/offline-editor/ | Practical localStorage form editor |
| Automatic Rollback in Optimistic Updates (DEV) | https://dev.to/klis87/automatic-rollback-data-in-optimistic-updates-a-surprising-benefit-of-normalized-data-535l | Rollback pattern without React |
| Setup Auto-Save for Multi-Step Forms (reform.app) | https://www.reform.app/blog/setup-auto-save-multi-step-forms | Step-level persistence |
| prefers-color-scheme: Hello darkness (web.dev) | https://web.dev/articles/prefers-color-scheme | Definitive guide; light-dark() function |
| Theming with CSS in 2025: light-dark() (mamutlove) | https://mamutlove.com/en/blog/theming-with-css-in-2025/ | light-dark() + color-scheme property |
| Dark Mode in CSS (design.dev) | https://design.dev/guides/dark-mode-css/ | Implementation patterns |
| CSS Print Media Queries Complete Guide (CodeLucky) | https://codelucky.com/css-print-media-queries/ | @page, break-* properties |
| CSS Print Styles for PDFs (DiDoesDigital) | https://didoesdigital.com/blog/print-styles/ | thead repeat, orphans/widows |
| Designing for Print with CSS Tips 2025 | https://618media.com/en/blog/designing-for-print-with-css-tips/ | break-* modern property names |

---

## Competitive Analysis: Frameworks

| Title | URL | Summary |
|-------|-----|---------|
| Django Accessibility Statement | https://www.djangoproject.com/accessibility/ | Official WCAG 2.2 AA + ATAG 2.0 targets |
| Django Accessibility in 2023 and Beyond | https://www.djangoproject.com/weblog/2024/feb/10/django-accessibility-in-2023-and-beyond/ | **Fetched directly.** 2023 wins, 2024 roadmap, 80.5/100 score gap |
| Django accessibility (Read the Docs) | https://django.readthedocs.io/en/6.0.x/internals/contributing/accessibility.html | Contribution guidelines |
| ActiveAdmin WCAG Compliance (Reintech) | https://reintech.io/blog/activeadmin-accessibility-wcag-compliance | Rails admin accessibility status |
| Hotwire Modals in Ruby on Rails (AppSignal) | https://blog.appsignal.com/2024/02/21/hotwire-modals-in-ruby-on-rails-with-stimulus-and-turbo-frames.html | Accessible modal patterns with Turbo |
| Hot Glue Rails Scaffolding | https://github.com/hot-glue-for-rails/hot-glue | Turbo-era scaffold generator capabilities |
| shadcn/ui Accessibility Audit 2026 (TheFrontKit) | https://thefrontkit.com/blogs/shadcn-ui-accessibility-audit-2026 | **Fetched directly.** 34/48 pass, 5 significant gaps, focus ring failure |
| shadcn/ui Handbook 2026 | https://shadcnspace.com/blog/shadcn-ui-handbook | Server component compatibility, tree-shaking |
| AI-First UIs: shadcn/ui model (Refine) | https://refine.dev/blog/shadcn-blog/ | Refine + shadcn integration status |
| React UI libraries 2025 comparison (Makers' Den) | https://makersden.io/blog/react-ui-libs-2025-comparing-shadcn-radix-mantine-mui-chakra | shadcn vs Radix vs MUI accessibility comparison |
| shadcn/ui Best Practices 2026 (Medium) | https://medium.com/write-a-catalyst/shadcn-ui-best-practices-for-2026-444efd204f44 | EU Accessibility Act June 2025 enforcement context |
| shadcn/ui Changelog | https://ui.shadcn.com/docs/changelog | Base UI rebuild December 2025 |
