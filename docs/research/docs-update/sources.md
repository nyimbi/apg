# Docs Update Sources

This refresh was grounded in repository-local sources and command output from
the APG checkout.

## Documentation Read

Inventory command:

```bash
find docs -type f | sort
find docs -type f | sort | while read -r f; do file -b --mime-type "$f"; wc -l < "$f"; done
```

Text documentation reviewed included:

- `README.md`
- `docs/README.md`
- `docs/developer_guide.md`
- `docs/installation.md`
- `docs/generated_ui.md`
- `docs/architecture.md`
- `docs/quickstart.md`
- `docs/tooling.md`
- `docs/capabilities/README.md`
- `docs/deployment.md`
- `docs/capability_standards.md`
- generated UI research summaries under `docs/research/generated-ui-workspaces/`

Binary or generated artifact files under `docs/` were inventoried rather than
quoted or edited, including `.DS_Store`, `.zip`, `.pdf`, `.pyc`, and captured
HTML/JSON assets.

## Source Files Used

- `setup.py` for Python version, package dependencies, extras, and CLI entry
  points.
- `cli/main.py` and command help for current top-level CLI.
- `cli/baseline_command.py` through `apg baseline --help` for
  `--refresh`/`--refresh-outputs`.
- `cli/capabilities_command.py` through command help for capability CLI
  groups.
- `compiler/code_generator.py` for generated Flask app routes, generated
  runtime helpers, PWA metadata, i18n catalog, shell behavior, persistence
  variables, and generated README content.
- `compiler/templates/` for generated UI template names and workspace coverage.
- `compiler/assets/` for vendored generated-app browser assets.
- `compiler/baseline.py` for baseline gate behavior.
- `capabilities/` directory structure for capability domain and package
  inventory.
- `tests/` and full pytest output for the documented test target.

## Commands Used As Evidence

```bash
python -m cli.main --help
python -m cli.main baseline --help
python -m cli.main compile --help
python -m cli.main capabilities --help
python -m cli.main capabilities scaffold --help
python -m cli.main docs --help
python -m cli.main tooling --help
find capabilities -mindepth 2 -maxdepth 3 -type d ! -name '__pycache__' ! -name '.pytest_cache' ! -name '.ropeproject'
find capabilities -name capability_contract.py -not -path '*/__pycache__/*'
find capabilities -mindepth 2 -maxdepth 3 -type f -name cap_spec.md
find compiler/assets -maxdepth 1 -type f | sort
find compiler/templates -maxdepth 2 -type f | sort
rg -n "def _ui_payload|/ui/workflows|/ui/marketplace|/ui/debug|/ui/databases|kanban|analytics|agent-console|team-console|capabilities" compiler/code_generator.py
uv run pytest tests/ -q
```

## Counts Captured

- 33 top-level capability domains.
- 440 non-hidden `capabilities/<domain>/<code>` directories.
- 322 checked `cap_spec.md` files.
- 592 `capability_contract.py` files when build output copies are included.
- Generated UI assets present in `compiler/assets/`: `apg.css`,
  `htmx.min.js`, `sortable.min.js`, `uplot.min.js`, `uplot.min.css`,
  `apg-charts.js`, `apg-sse.js`, and `LICENSES.md`.
- Generated UI templates present in `compiler/templates/`: landing, app index,
  entity list, entity analytics, kanban, record detail, workflow list, workflow
  wizard, agent console, capability console, database catalog, debug console,
  marketplace, login, and widget templates.

## Not Used As Current-State Evidence

Historical completion reports under `docs/reports/` and archived root README
variants under `docs/archive/` were treated as historical context only. They
were not used as current platform claims when they conflicted with source,
command help, or generated runtime code.
