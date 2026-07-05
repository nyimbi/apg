# Shell Chrome Sidebar Topbar Palette Notifications Theme I18n PWA

## Verdict

Before: the generated shell had the foundation of a topbar, sidebar, theme toggle, locale switcher, command palette, offline banner, and service worker. The gaps were mostly discoverability and completeness: command palette access was keyboard-only and lacked dialog semantics, sidebar active state depended on client-side topnav scripting or template internals, notifications were transient toasts only, and the PWA install/update flow had no UI affordance.

After: the shell exposes a visible command trigger, server-side active navigation, notification history panel, PWA install/update buttons, richer manifest shortcuts/categories, service-worker update handling, and offline/online notifications. The existing theme toggle, locale switcher, sidebar drawer, skip link, and offline caching remain self-contained.

## Live Surface Audit

- Before app: example 20 booted on `127.0.0.1:20909`.
- Before `/ui`: shell rendered topbar/sidebar/theme/locale/offline banner, but no visible command trigger, no notification center, no install/update controls.
- Before `/ui/entities/Vendor`: entity template had local active breadcrumb, but sidebar active state was not server-rendered in direct shell markup.
- Before PWA assets: manifest lacked `id`, app shortcuts, categories, and service-worker update messaging.
- After app: regenerated example 20 booted on `127.0.0.1:20910`.
- After `/ui`: visible `Search` command button, notification panel, install/update buttons, and active dashboard nav rendered.
- After `/ui/entities/Vendor`: sidebar link for `Vendor` rendered `aria-current="page"`.
- After manifest/service worker: manifest contains shortcuts and app metadata; service worker uses `apg-ui-v2` and handles `SKIP_WAITING`.

## Must-Fix Items Completed

- Added visible command-palette trigger with `aria-haspopup="dialog"` and dialog semantics on the palette container.
- Added notification center/history panel and routed toast/offline/PWA events into it.
- Added install/update shell controls and service-worker update handling.
- Added server-side active state for topnav/sidebar links.
- Added PWA manifest `id`, categories, orientation, and shortcuts.
- Added regression coverage for shell controls, active sidebar, and PWA metadata/update behavior.

## Evidence

- `assets/before-ui.html`
- `assets/before-entity.html`
- `assets/before-marketplace.html`
- `assets/before-manifest.webmanifest`
- `assets/before-sw.js`
- `assets/before-search.json`
- `assets/after-ui.html`
- `assets/after-entity.html`
- `assets/after-marketplace.html`
- `assets/after-manifest.webmanifest`
- `assets/after-sw.js`
- `assets/after-search.json`
