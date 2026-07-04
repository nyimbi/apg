# Generated UI Workspace Excellence Summary

## Commit Plan Ledger

Git writes are blocked in this sandbox when `.git/index.lock` cannot be created. Each completed workspace records the intended commit message and file list here.

### Prior WP7 Carry-Forward

Status: implemented and validated before this excellence pass, but commit was blocked by `.git` write permissions.

Intended commit:

```text
WP7: generated UI PWA and baseline harness
```

Validation evidence:

- Regenerated all 20 numbered examples.
- `UV_CACHE_DIR=$PWD/.uv-cache uv run pytest tests/ -q` -> `1474 passed, 1 skipped, 3 warnings`.
- PythonCodeGenerator tripwire clean.

### home-dashboard

Status: complete.

Intended commit:

```text
ux(home-dashboard): prioritize dashboard workspace actions
```

Files:

- `compiler/code_generator.py`
- `compiler/templates/app_index.html.j2`
- `tests/test_capability_composition_runtime.py`
- `tests/test_compiler_baseline.py`
- `tests/test_generated_ui_dashboard.py`
- `examples/01_minimal_customer_records/output/app.py`
- `examples/02_customer_orders_relationship/output/app.py`
- `examples/03_inventory_typed_records/output/app.py`
- `examples/04_order_fulfillment_model/output/app.py`
- `examples/05_single_support_agent/output/app.py`
- `examples/06_support_agent_team/output/app.py`
- `examples/07_multi_runtime_agent_team/output/app.py`
- `examples/08_basic_capability_contract/output/app.py`
- `examples/09_capability_rules_configuration/output/app.py`
- `examples/10_themed_i18n_streaming_capability/output/app.py`
- `examples/11_screen_composition_relationships/output/app.py`
- `examples/12_finance_general_ledger/output/app.py`
- `examples/13_procurement_approval_workbench/output/app.py`
- `examples/14_inventory_warehouse_operations/output/app.py`
- `examples/15_manufacturing_quality_control/output/app.py`
- `examples/16_hr_payroll_operations/output/app.py`
- `examples/17_crm_sales_pipeline/output/app.py`
- `examples/18_operations_dashboard_capability/output/app.py`
- `examples/19_multi_capability_dependency_suite/output/app.py`
- `examples/20_enterprise_erp_platform/output/app.py`
- `docs/research/generated-ui-workspaces/home-dashboard/README.md`
- `docs/research/generated-ui-workspaces/home-dashboard/thinking.md`
- `docs/research/generated-ui-workspaces/home-dashboard/sources.md`
- `docs/research/generated-ui-workspaces/home-dashboard/rationale.md`
- `docs/research/generated-ui-workspaces/home-dashboard/assets/before-example20-ui.html`
- `docs/research/generated-ui-workspaces/home-dashboard/assets/before-example20-ui.headers`
- `docs/research/generated-ui-workspaces/home-dashboard/assets/after-example20-ui.html`
- `docs/research/generated-ui-workspaces/home-dashboard/assets/after-example20-ui.headers`

Validation evidence:

- Live audit before/after: example 20 `/ui` booted at `127.0.0.1:20881`.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `4 passed in 43.21s`.
- Full suite: `1475 passed, 1 skipped, 3 warnings in 737.69s`.
- PythonCodeGenerator tripwire clean.

## Verdicts

| Workspace | Before | After | Status |
| --- | --- | --- | --- |
| home-dashboard | API-oriented shortcut row, inaccurate agent/capability summaries for example 20, passive empty activity state, stats mixed record and non-record constructs. | Workspace-first shortcuts, record-focused KPI cards, accurate generated agent/team/capability counts, linked summaries, actionable empty state. | Complete |

## Defect Ledger

| Workspace | Defect | Resolution | Status |
| --- | --- | --- | --- |
| home-dashboard | Quick navigation prioritized generated API/internal links over user work. | Reordered shortcuts around first entity, workflows, database catalog, marketplace, and retained API/debug links as secondary. | Resolved |
| home-dashboard | Home template depended on missing `describe_application()` keys for capabilities and agents. | Dashboard context now classifies generated `ENTITIES` directly. | Resolved |
| home-dashboard | KPI cards included non-record constructs before business records. | Stats now include only record-owning entity/table types. | Resolved |
| home-dashboard | Empty recent activity state had no next action. | Added CTA to first primary entity. | Resolved |
| home-dashboard | Generated app HTTP tests had readiness windows too short for heavier generated UI imports under load. | Normalized generated-app health polling loops to 80 attempts. | Resolved |
