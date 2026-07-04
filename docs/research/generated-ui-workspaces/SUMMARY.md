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

### entity-list-table-filters-saved-views

Status: complete.

Intended commit:

```text
ux(entity-list-table-filters-saved-views): improve saved views and filter ergonomics
```

Files:

- `compiler/assets/apg.css`
- `compiler/code_generator.py`
- `compiler/templates/entity_list.html.j2`
- `tests/test_generated_ui_dashboard.py`
- `examples/01_minimal_customer_records/output/app.py`
- `examples/01_minimal_customer_records/output/static/apg.css`
- `examples/02_customer_orders_relationship/output/app.py`
- `examples/02_customer_orders_relationship/output/static/apg.css`
- `examples/03_inventory_typed_records/output/app.py`
- `examples/03_inventory_typed_records/output/static/apg.css`
- `examples/04_order_fulfillment_model/output/app.py`
- `examples/04_order_fulfillment_model/output/static/apg.css`
- `examples/05_single_support_agent/output/app.py`
- `examples/05_single_support_agent/output/static/apg.css`
- `examples/06_support_agent_team/output/app.py`
- `examples/06_support_agent_team/output/static/apg.css`
- `examples/07_multi_runtime_agent_team/output/app.py`
- `examples/07_multi_runtime_agent_team/output/static/apg.css`
- `examples/08_basic_capability_contract/output/app.py`
- `examples/08_basic_capability_contract/output/static/apg.css`
- `examples/09_capability_rules_configuration/output/app.py`
- `examples/09_capability_rules_configuration/output/static/apg.css`
- `examples/10_themed_i18n_streaming_capability/output/app.py`
- `examples/10_themed_i18n_streaming_capability/output/static/apg.css`
- `examples/11_screen_composition_relationships/output/app.py`
- `examples/11_screen_composition_relationships/output/static/apg.css`
- `examples/12_finance_general_ledger/output/app.py`
- `examples/12_finance_general_ledger/output/static/apg.css`
- `examples/13_procurement_approval_workbench/output/app.py`
- `examples/13_procurement_approval_workbench/output/static/apg.css`
- `examples/14_inventory_warehouse_operations/output/app.py`
- `examples/14_inventory_warehouse_operations/output/static/apg.css`
- `examples/15_manufacturing_quality_control/output/app.py`
- `examples/15_manufacturing_quality_control/output/static/apg.css`
- `examples/16_hr_payroll_operations/output/app.py`
- `examples/16_hr_payroll_operations/output/static/apg.css`
- `examples/17_crm_sales_pipeline/output/app.py`
- `examples/17_crm_sales_pipeline/output/static/apg.css`
- `examples/18_operations_dashboard_capability/output/app.py`
- `examples/18_operations_dashboard_capability/output/static/apg.css`
- `examples/19_multi_capability_dependency_suite/output/app.py`
- `examples/19_multi_capability_dependency_suite/output/static/apg.css`
- `examples/20_enterprise_erp_platform/output/app.py`
- `examples/20_enterprise_erp_platform/output/static/apg.css`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/README.md`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/thinking.md`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/sources.md`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/rationale.md`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/assets/before-vendor-list.html`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/assets/before-vendor-list.headers`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/assets/after-vendor-list.html`
- `docs/research/generated-ui-workspaces/entity-list-table-filters-saved-views/assets/after-vendor-list.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 20 `/ui/entities/Vendor` booted at `127.0.0.1:20882`.
- Live after audit: regenerated example 20 `/ui/entities/Vendor?filter.status=active&q=Acme&sort=id&dir=desc` booted at `127.0.0.1:20883`.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `2 passed in 5.29s`.
- Full suite: `1476 passed, 1 skipped, 3 warnings in 733.24s`.
- PythonCodeGenerator tripwire clean.

### entity-analytics

Status: complete.

Intended commit:

```text
ux(entity-analytics): make analytics data-backed and drillable
```

Files:

- `compiler/assets/apg.css`
- `compiler/code_generator.py`
- `compiler/templates/entity_analytics.html.j2`
- `tests/test_generated_ui_dashboard.py`
- `examples/01_minimal_customer_records/output/app.py`
- `examples/01_minimal_customer_records/output/static/apg.css`
- `examples/02_customer_orders_relationship/output/app.py`
- `examples/02_customer_orders_relationship/output/static/apg.css`
- `examples/03_inventory_typed_records/output/app.py`
- `examples/03_inventory_typed_records/output/static/apg.css`
- `examples/04_order_fulfillment_model/output/app.py`
- `examples/04_order_fulfillment_model/output/static/apg.css`
- `examples/05_single_support_agent/output/app.py`
- `examples/05_single_support_agent/output/static/apg.css`
- `examples/06_support_agent_team/output/app.py`
- `examples/06_support_agent_team/output/static/apg.css`
- `examples/07_multi_runtime_agent_team/output/app.py`
- `examples/07_multi_runtime_agent_team/output/static/apg.css`
- `examples/08_basic_capability_contract/output/app.py`
- `examples/08_basic_capability_contract/output/static/apg.css`
- `examples/09_capability_rules_configuration/output/app.py`
- `examples/09_capability_rules_configuration/output/static/apg.css`
- `examples/10_themed_i18n_streaming_capability/output/app.py`
- `examples/10_themed_i18n_streaming_capability/output/static/apg.css`
- `examples/11_screen_composition_relationships/output/app.py`
- `examples/11_screen_composition_relationships/output/static/apg.css`
- `examples/12_finance_general_ledger/output/app.py`
- `examples/12_finance_general_ledger/output/static/apg.css`
- `examples/13_procurement_approval_workbench/output/app.py`
- `examples/13_procurement_approval_workbench/output/static/apg.css`
- `examples/14_inventory_warehouse_operations/output/app.py`
- `examples/14_inventory_warehouse_operations/output/static/apg.css`
- `examples/15_manufacturing_quality_control/output/app.py`
- `examples/15_manufacturing_quality_control/output/static/apg.css`
- `examples/16_hr_payroll_operations/output/app.py`
- `examples/16_hr_payroll_operations/output/static/apg.css`
- `examples/17_crm_sales_pipeline/output/app.py`
- `examples/17_crm_sales_pipeline/output/static/apg.css`
- `examples/18_operations_dashboard_capability/output/app.py`
- `examples/18_operations_dashboard_capability/output/static/apg.css`
- `examples/19_multi_capability_dependency_suite/output/app.py`
- `examples/19_multi_capability_dependency_suite/output/static/apg.css`
- `examples/20_enterprise_erp_platform/output/app.py`
- `examples/20_enterprise_erp_platform/output/static/apg.css`
- `docs/research/generated-ui-workspaces/entity-analytics/README.md`
- `docs/research/generated-ui-workspaces/entity-analytics/thinking.md`
- `docs/research/generated-ui-workspaces/entity-analytics/sources.md`
- `docs/research/generated-ui-workspaces/entity-analytics/rationale.md`
- `docs/research/generated-ui-workspaces/entity-analytics/assets/before-vendor-analytics.html`
- `docs/research/generated-ui-workspaces/entity-analytics/assets/before-vendor-analytics.headers`
- `docs/research/generated-ui-workspaces/entity-analytics/assets/after-vendor-analytics.html`
- `docs/research/generated-ui-workspaces/entity-analytics/assets/after-vendor-analytics.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 20 `/ui/entities/Vendor?view=analytics` booted at `127.0.0.1:20884`.
- Live after audit: regenerated example 20 `/ui/entities/Vendor?view=analytics` booted at `127.0.0.1:20885`.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `2 passed in 5.65s`.
- Full suite: `1477 passed, 1 skipped, 3 warnings in 799.09s`.
- PythonCodeGenerator tripwire clean.

### kanban

Status: complete.

Intended commit:

```text
ux(kanban): restore board rendering and keyboard moves
```

Files:

- `compiler/assets/apg.css`
- `compiler/code_generator.py`
- `compiler/templates/kanban_view.html.j2`
- `tests/test_generated_ui_dashboard.py`
- `examples/01_minimal_customer_records/output/app.py`
- `examples/01_minimal_customer_records/output/static/apg.css`
- `examples/02_customer_orders_relationship/output/app.py`
- `examples/02_customer_orders_relationship/output/static/apg.css`
- `examples/03_inventory_typed_records/output/app.py`
- `examples/03_inventory_typed_records/output/static/apg.css`
- `examples/04_order_fulfillment_model/output/app.py`
- `examples/04_order_fulfillment_model/output/static/apg.css`
- `examples/05_single_support_agent/output/app.py`
- `examples/05_single_support_agent/output/static/apg.css`
- `examples/06_support_agent_team/output/app.py`
- `examples/06_support_agent_team/output/static/apg.css`
- `examples/07_multi_runtime_agent_team/output/app.py`
- `examples/07_multi_runtime_agent_team/output/static/apg.css`
- `examples/08_basic_capability_contract/output/app.py`
- `examples/08_basic_capability_contract/output/static/apg.css`
- `examples/09_capability_rules_configuration/output/app.py`
- `examples/09_capability_rules_configuration/output/static/apg.css`
- `examples/10_themed_i18n_streaming_capability/output/app.py`
- `examples/10_themed_i18n_streaming_capability/output/static/apg.css`
- `examples/11_screen_composition_relationships/output/app.py`
- `examples/11_screen_composition_relationships/output/static/apg.css`
- `examples/12_finance_general_ledger/output/app.py`
- `examples/12_finance_general_ledger/output/static/apg.css`
- `examples/13_procurement_approval_workbench/output/app.py`
- `examples/13_procurement_approval_workbench/output/static/apg.css`
- `examples/14_inventory_warehouse_operations/output/app.py`
- `examples/14_inventory_warehouse_operations/output/static/apg.css`
- `examples/15_manufacturing_quality_control/output/app.py`
- `examples/15_manufacturing_quality_control/output/static/apg.css`
- `examples/16_hr_payroll_operations/output/app.py`
- `examples/16_hr_payroll_operations/output/static/apg.css`
- `examples/17_crm_sales_pipeline/output/app.py`
- `examples/17_crm_sales_pipeline/output/static/apg.css`
- `examples/18_operations_dashboard_capability/output/app.py`
- `examples/18_operations_dashboard_capability/output/static/apg.css`
- `examples/19_multi_capability_dependency_suite/output/app.py`
- `examples/19_multi_capability_dependency_suite/output/static/apg.css`
- `examples/20_enterprise_erp_platform/output/app.py`
- `examples/20_enterprise_erp_platform/output/static/apg.css`
- `docs/research/generated-ui-workspaces/kanban/README.md`
- `docs/research/generated-ui-workspaces/kanban/thinking.md`
- `docs/research/generated-ui-workspaces/kanban/sources.md`
- `docs/research/generated-ui-workspaces/kanban/rationale.md`
- `docs/research/generated-ui-workspaces/kanban/assets/before-vendor-kanban.html`
- `docs/research/generated-ui-workspaces/kanban/assets/before-vendor-kanban.headers`
- `docs/research/generated-ui-workspaces/kanban/assets/after-vendor-kanban.html`
- `docs/research/generated-ui-workspaces/kanban/assets/after-vendor-kanban.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 20 `/ui/entities/Vendor?view=kanban` booted at `127.0.0.1:20886` and fell back to the list page.
- Live after audit: regenerated example 20 `/ui/entities/Vendor?view=kanban` booted at `127.0.0.1:20887` and rendered the board.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `2 passed in 7.99s`.
- Full suite: `1478 passed, 1 skipped, 3 warnings in 799.95s`.
- PythonCodeGenerator tripwire clean.

## Verdicts

| Workspace | Before | After | Status |
| --- | --- | --- | --- |
| home-dashboard | API-oriented shortcut row, inaccurate agent/capability summaries for example 20, passive empty activity state, stats mixed record and non-record constructs. | Workspace-first shortcuts, record-focused KPI cards, accurate generated agent/team/capability counts, linked summaries, actionable empty state. | Complete |
| entity-list-table-filters-saved-views | Generic table page with no saved views, hidden filter state, top-level API JSON link, and filter-dropping pagination/sort links. | Saved-view tabs, semantic `Active` preset, active filter chips, query-preserving pagination/sort/page-size links, canonical table wrapper, and developer exports disclosure. | Complete |
| entity-analytics | Chart-present but shallow: flat placeholder trend, no drill-through, no headline metrics, and no actionable insights. | Date-bucketed trend data, summary metrics, status drill-through links, largest-segment and trend-window insights, and clearer empty states. | Complete |
| kanban | `view=kanban` silently fell back to the entity list because the template used unsupported Jinja loop control; movement was pointer-only. | Template renders with default Jinja, cards have keyboard/server move controls, board context is preserved after moves, columns have drill-through and WIP guidance. | Complete |

## Defect Ledger

| Workspace | Defect | Resolution | Status |
| --- | --- | --- | --- |
| home-dashboard | Quick navigation prioritized generated API/internal links over user work. | Reordered shortcuts around first entity, workflows, database catalog, marketplace, and retained API/debug links as secondary. | Resolved |
| home-dashboard | Home template depended on missing `describe_application()` keys for capabilities and agents. | Dashboard context now classifies generated `ENTITIES` directly. | Resolved |
| home-dashboard | KPI cards included non-record constructs before business records. | Stats now include only record-owning entity/table types. | Resolved |
| home-dashboard | Empty recent activity state had no next action. | Added CTA to first primary entity. | Resolved |
| home-dashboard | Generated app HTTP tests had readiness windows too short for heavier generated UI imports under load. | Normalized generated-app health polling loops to 80 attempts. | Resolved |
| entity-list-table-filters-saved-views | Saved views were absent. | Added generated semantic saved-view presets per entity. | Resolved |
| entity-list-table-filters-saved-views | Field filters and sorting were not visible once applied. | Added active chips with direct clear links for search, field filters, and sort. | Resolved |
| entity-list-table-filters-saved-views | Pagination, page-size, and column sorting dropped current filter state. | Centralized entity query URL generation and reused it across table controls. | Resolved |
| entity-list-table-filters-saved-views | API JSON competed with operator navigation. | Moved CSV/API/page JSON into a developer exports disclosure. | Resolved |
| entity-list-table-filters-saved-views | Rendered tables did not expose the canonical overflow wrapper class. | Changed generated table wrapper to `apg-table-wrap`. | Resolved |
| entity-analytics | Records-over-time chart used fake flat data. | Built a deterministic 30-day date bucket series from the first available date-like field or record key. | Resolved |
| entity-analytics | Status chart had no path back to records. | Added status drill-through rows linking to filtered entity tables. | Resolved |
| entity-analytics | Page lacked headline context before charts. | Added record, recent, status, and measure metric tiles. | Resolved |
| entity-analytics | Analytics did not surface what mattered. | Added largest-segment and trend-window insight cards. | Resolved |
| kanban | Kanban route silently rendered the list page. | Removed unsupported `{% break %}` from the template and added regression coverage. | Resolved |
| kanban | Card movement required pointer drag/drop. | Added native per-card move forms with status select and submit button. | Resolved |
| kanban | Server-rendered card moves redirected to the list. | Added `return_view=kanban` handling for UI record updates. | Resolved |
| kanban | Columns had no drill-through or bottleneck signal. | Added filtered list links, board summary metrics, and generated WIP guidance. | Resolved |
