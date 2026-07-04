# Generated UI

APG generated applications ship a self-contained UI asset pipeline. The compiler emits all CSS, JavaScript, charting, streaming, PWA metadata, and icon files into each generated `static/` directory; generated pages should not depend on CDN-hosted assets.

## Asset Contract

The generated UI includes:

- `static/apg.css` for shell, forms, tables, dashboards, responsive layout, RTL, auth, i18n, live updates, and offline state.
- `static/htmx.min.js`, `static/sortable.min.js`, `static/uplot.min.js`, and `static/uplot.min.css` as vendored browser assets.
- `static/apg-charts.js` and `static/apg-sse.js` for generated dashboards and server-sent events.
- `static/manifest.webmanifest`, `static/sw.js`, and `static/icon.svg` for installable, offline-capable generated apps.

The current test budget is gzip `apg.css` at 60 KB or less and combined generated JavaScript at 120 KB or less.

## PWA Behavior

Generated HTML pages include `theme-color`, `manifest`, and service-worker registration tags. The service worker caches generated static assets and the last viewed same-origin GET pages. When the browser reports offline status, the generated shell shows an offline banner and keeps cached UI screens available.

Manifest names, short names, descriptions, theme colors, and icons are generated from the APG module and theme metadata so each output directory is portable without external build steps.

## Baseline Refresh

Use the baseline command to regenerate and verify all numbered examples:

```bash
apg baseline examples --refresh
```

`--refresh` is an alias for `--refresh-outputs`; both rewrite numbered example `output/` directories from the current compiler before running the compiler baseline audit.

For remediation work packages that change compiler or template output, regenerate all numbered examples and then run:

```bash
uv run pytest tests/ -q
```

The test gate also checks that generated UI output remains CDN-free, budgeted, and covered by the static asset contract.
