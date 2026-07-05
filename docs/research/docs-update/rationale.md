# Docs Update Rationale

The previous top-level and developer documentation mixed current compiler
behavior with older platform claims. The update narrows public docs around
source-backed facts and marks inventory counts as inventory rather than uniform
production-readiness claims.

## Decisions

### APG Name And Positioning

The root README now uses "Agentic Platform Generator" and describes APG as
Africa-first because the requested positioning matches the current direction of
the generated app and capability ecosystem. The older "Application Programming
Generation" wording still appears where it is embedded in package metadata or
legacy command descriptions, but current docs lead with the requested name.

### CLI Surface

The current CLI is the Click app exposed by `setup.py` as `apg=cli.main:cli`.
The legacy root `cli.py` still exists, but it is documented only as a
compatibility surface. This avoids teaching developers older `apg build` and
`python cli.py run` flows when the current compiler path is `apg compile`.

### Generated Apps

Generated apps are documented as Flask apps because `compiler/code_generator.py`
imports Flask and emits Flask routes. The docs no longer present React,
Flask-AppBuilder, Django, Redis, PostgreSQL, or Docker Compose as mandatory for
basic APG compiler usage.

### Static Assets

The generated UI section explicitly lists local static assets because
`compiler/assets/` contains the vendored files and generated pages link to
`/static/...`. The docs state that generated apps should not require CDN
dependencies.

### Capability Counts

The repository has multiple possible capability counts depending on whether
one counts directories, `cap_spec.md`, contracts, or build-output copies. The
README and capabilities overview therefore use precise labels:

- source-tree domain/code directories
- checked `cap_spec.md` files
- `capability_contract.py` files including build artifacts

This is more accurate than the older flat "259 production-grade capabilities"
claim.

### 14 Generated UI Workspaces

The generated UI reference maps the requested 14 workspace descriptions to
actual generated routes, templates, and helper code in `compiler/code_generator.py`
and `compiler/templates/`.

### Baseline And Tests

The docs include `apg baseline examples --refresh` because the current baseline
help exposes `--refresh` as an alias for `--refresh-outputs`. The requested
1474-pass figure was older than this checkout; fresh verification completed as
`uv run pytest tests/ -q` with 1486 passed, 1 skipped, and 3 warnings.

### Other Stale Docs

`docs/README.md`, `docs/quickstart.md`, `docs/capabilities/README.md`, and
`docs/deployment.md` were materially stale because they directed users toward
old setup flows, overstated capability maturity, or implied mandatory platform
services. They were rewritten to match the current compiler and generated-app
path.

## Deferred

Historical reports, archived docs, captured generated UI before/after assets,
and old grammar drafts were not rewritten. They are intentionally preserved as
history and research artifacts.
