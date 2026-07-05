# Rationale

## Decisions

- Keep the existing generated shell architecture and improve the shared `_html_page()` output. This keeps the change small and ensures every `/ui/*` page benefits.
- Preserve the existing command search implementation, but expose it through a visible button and dialog semantics.
- Add a client-side notification history panel rather than a backend notification model. This covers shell events without inventing persistence semantics.
- Keep install/update buttons hidden until browser or service-worker events make them actionable.
- Generate PWA shortcuts in the manifest for dashboard, workflows, and marketplace because those are stable generated shell routes.
- Use server-side current-path detection in real Flask requests for topnav/sidebar `aria-current`.

## Rejected Alternatives

- Browser push notifications were rejected as out of scope for self-contained generated apps.
- New JavaScript bundles were rejected; the existing inline shell script is enough and avoids extra asset budget pressure.
- Reworking sidebar grouping by capability was rejected for this final pass because capability ownership metadata is inconsistent across entity kinds; active-state and shell controls had higher value.

## Validation Plan

- Targeted PWA/shell regression tests.
- Template/CSS coverage checks.
- Regenerate all 20 numbered examples.
- Live after-audit of `/ui`, `/ui/entities/Vendor`, `/ui/marketplace`, manifest, service worker, and search endpoint.
- Full `uv run pytest tests/ -q`.
- PythonCodeGenerator hardcoded-literal tripwire.
