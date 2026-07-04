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

### record-detail-activity-related

Status: complete.

Intended commit:

```text
ux(record-detail-activity-related): render related records and navigation
```

Files:

- `compiler/code_generator.py`
- `compiler/templates/record_detail.html.j2`
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
- `docs/research/generated-ui-workspaces/record-detail-activity-related/README.md`
- `docs/research/generated-ui-workspaces/record-detail-activity-related/thinking.md`
- `docs/research/generated-ui-workspaces/record-detail-activity-related/sources.md`
- `docs/research/generated-ui-workspaces/record-detail-activity-related/rationale.md`
- `docs/research/generated-ui-workspaces/record-detail-activity-related/assets/before-customer-detail.html`
- `docs/research/generated-ui-workspaces/record-detail-activity-related/assets/before-customer-detail.headers`
- `docs/research/generated-ui-workspaces/record-detail-activity-related/assets/after-customer-detail.html`
- `docs/research/generated-ui-workspaces/record-detail-activity-related/assets/after-customer-detail.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 02 `/ui/entities/Customer/1` booted at `127.0.0.1:20888` and fell back to raw JSON when related records existed.
- Live after audit: regenerated example 02 `/ui/entities/Customer/1` booted at `127.0.0.1:20889` and rendered the full record page.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `2 passed in 1.19s`.
- Full suite: `1479 passed, 1 skipped, 3 warnings in 790.16s`.
- PythonCodeGenerator tripwire clean.

### create-edit-forms-drawer-inline-edit

Status: complete.

Intended commit:

```text
ux(create-edit-forms-drawer-inline-edit): improve form validation and editing
```

Files:

- `compiler/code_generator.py`
- `compiler/templates/entity_list.html.j2`
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
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/README.md`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/thinking.md`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/sources.md`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/rationale.md`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/before-customer-list.html`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/before-customer-list.headers`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/before-create-error.html`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/before-create-error.headers`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/after-customer-list.html`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/after-customer-list.headers`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/after-create-error.html`
- `docs/research/generated-ui-workspaces/create-edit-forms-drawer-inline-edit/assets/after-create-error.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 01 `/ui/entities/Customer` and failed create request booted at `127.0.0.1:20890`.
- Live after audit: regenerated example 01 `/ui/entities/Customer` and failed create request booted at `127.0.0.1:20891`.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `2 passed in 1.07s`.
- Full suite: `1480 passed, 1 skipped, 3 warnings in 742.32s`.
- PythonCodeGenerator tripwire clean.

### workflow-list-wizard-run-progress

Status: complete.

Intended commit:

```text
ux(workflow-list-wizard-run-progress): record wizard runs and fix step flow
```

Files:

- `compiler/code_generator.py`
- `compiler/templates/workflow_list.html.j2`
- `compiler/templates/workflow_wizard.html.j2`
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
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/README.md`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/thinking.md`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/sources.md`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/rationale.md`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/before-workflow-list.html`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/before-workflow-list.headers`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/before-wizard-step.html`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/before-wizard-step.headers`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/before-wizard-post.html`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/before-wizard-post.headers`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-wizard-step.html`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-wizard-step.headers`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-wizard-complete.html`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-workflow-list.html`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-workflow-list.headers`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-workflow-runs.json`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-debug-run.html`
- `docs/research/generated-ui-workspaces/workflow-list-wizard-run-progress/assets/after-debug-run.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 01 `/ui/workflows`, `/ui/workflows/Customer/create_customer`, and first wizard POST booted at `127.0.0.1:20892`.
- Live after audit: regenerated example 01 full customer wizard booted at `127.0.0.1:20894`, advanced through all six steps, created `workflow-run-1`, and rendered `/ui/debug/workflow-run-1`.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `8 passed in 15.99s`.
- Full suite: `1481 passed, 1 skipped, 3 warnings in 714.85s`.
- PythonCodeGenerator tripwire clean.

### agent-and-agent-team-consoles

Status: complete.

Intended commit:

```text
ux(agent-and-agent-team-consoles): restore team consoles and chat layout
```

Files:

- `compiler/code_generator.py`
- `compiler/templates/agent_console.html.j2`
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
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/README.md`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/thinking.md`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/sources.md`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/rationale.md`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-agent-console.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-agent-console.headers`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-team-console.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-team-console.headers`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-agent-post.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-agent-post.headers`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-team-post.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/before-team-post.headers`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-agent-console.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-agent-console.headers`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-team-console.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-team-console.headers`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-agent-post.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-agent-post.headers`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-team-post.html`
- `docs/research/generated-ui-workspaces/agent-and-agent-team-consoles/assets/after-team-post.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 06 `/ui/agents/Planner`, `/ui/agent-teams/SupportCrew`, and invoke POSTs booted at `127.0.0.1:20895`; team console and team POST returned 404.
- Live after audit: regenerated example 06 booted at `127.0.0.1:20898`; agent and team consoles returned 200 and both POST flows rendered response panels.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `2 passed in 1.61s`.
- CSS coverage: `1 passed in 0.27s`.
- Full suite: `1482 passed, 1 skipped, 3 warnings in 716.48s`.
- PythonCodeGenerator tripwire clean.

### capability-console-rules-config-approval

Status: complete.

Intended commit:

```text
ux(capability-console-rules-config-approval): clarify capability operations
```

Files:

- `compiler/code_generator.py`
- `compiler/templates/capability_console.html.j2`
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
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/README.md`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/thinking.md`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/sources.md`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/rationale.md`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-console.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-console.headers`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-rules.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-rules.headers`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-config.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-config.headers`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-approval.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/before-approval.headers`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-console.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-console.headers`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-rules.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-rules.headers`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-config.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-config.headers`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-approval.html`
- `docs/research/generated-ui-workspaces/capability-console-rules-config-approval/assets/after-approval.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 09 `/ui/capabilities/CreditControl` and the rules/configuration/approval POST flows booted at `127.0.0.1:20899`.
- Live after audit: regenerated example 09 booted at `127.0.0.1:20900`; the console and all three POST flows returned 200 and rendered operation-specific summaries.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `3 passed` across capability console regression, CSS class coverage, and required template route coverage.
- Full suite: `1483 passed, 1 skipped, 3 warnings in 730.43s`.
- PythonCodeGenerator tripwire clean.

### database-catalog

Status: complete.

Intended commit:

```text
ux(database-catalog): infer schemas and render table metadata
```

Files:

- `compiler/code_generator.py`
- `compiler/templates/database_catalog.html.j2`
- `tests/test_compiler_database_ast.py`
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
- `docs/research/generated-ui-workspaces/database-catalog/README.md`
- `docs/research/generated-ui-workspaces/database-catalog/thinking.md`
- `docs/research/generated-ui-workspaces/database-catalog/sources.md`
- `docs/research/generated-ui-workspaces/database-catalog/rationale.md`
- `docs/research/generated-ui-workspaces/database-catalog/assets/before-database-catalog.html`
- `docs/research/generated-ui-workspaces/database-catalog/assets/before-database-catalog.headers`
- `docs/research/generated-ui-workspaces/database-catalog/assets/before-schema-json.json`
- `docs/research/generated-ui-workspaces/database-catalog/assets/before-schema-json.headers`
- `docs/research/generated-ui-workspaces/database-catalog/assets/after-database-catalog.html`
- `docs/research/generated-ui-workspaces/database-catalog/assets/after-database-catalog.headers`
- `docs/research/generated-ui-workspaces/database-catalog/assets/after-schema-json.json`
- `docs/research/generated-ui-workspaces/database-catalog/assets/after-schema-json.headers`
- `docs/research/generated-ui-workspaces/SUMMARY.md`

Validation evidence:

- Live before audit: example 20 `/ui/databases` and `/databases/ERPDB/schemas` booted at `127.0.0.1:20901`; UI reported 0 schemas/tables and JSON returned `schemas: []`.
- Live after audit: regenerated example 20 booted at `127.0.0.1:20902`; UI rendered `ERPDB / erp_platform` with Vendor, Customer, and Employee tables, and JSON returned the inferred schema.
- Regenerated all 20 numbered examples through `APGCompiler.compile_file()`.
- Targeted tests: `3 passed` across database catalog regression, CSS class coverage, and required template route coverage.
- Full suite: `1484 passed, 1 skipped, 3 warnings in 750.06s`.
- PythonCodeGenerator tripwire clean.

## Verdicts

| Workspace | Before | After | Status |
| --- | --- | --- | --- |
| home-dashboard | API-oriented shortcut row, inaccurate agent/capability summaries for example 20, passive empty activity state, stats mixed record and non-record constructs. | Workspace-first shortcuts, record-focused KPI cards, accurate generated agent/team/capability counts, linked summaries, actionable empty state. | Complete |
| entity-list-table-filters-saved-views | Generic table page with no saved views, hidden filter state, top-level API JSON link, and filter-dropping pagination/sort links. | Saved-view tabs, semantic `Active` preset, active filter chips, query-preserving pagination/sort/page-size links, canonical table wrapper, and developer exports disclosure. | Complete |
| entity-analytics | Chart-present but shallow: flat placeholder trend, no drill-through, no headline metrics, and no actionable insights. | Date-bucketed trend data, summary metrics, status drill-through links, largest-segment and trend-window insights, and clearer empty states. | Complete |
| kanban | `view=kanban` silently fell back to the entity list because the template used unsupported Jinja loop control; movement was pointer-only. | Template renders with default Jinja, cards have keyboard/server move controls, board context is preserved after moves, columns have drill-through and WIP guidance. | Complete |
| record-detail-activity-related | Related records caused raw JSON fallback; no copy link or prev/next navigation; title selection preferred generated numbers. | Full record template renders with related records, filtered related links, activity/notes, copy link, next/previous navigation, and human-readable titles. | Complete |
| create-edit-forms-drawer-inline-edit | Create drawer bypassed native validation, structured fields used single-line text inputs, failed creates returned a contextless fragment, and inline edit controls were not type-aware. | Native required/type validation, helper text, JSON textareas, contextual error recovery, draft guard, Ctrl/Cmd-S submit, and semantic inline edit controls. | Complete |
| workflow-list-wizard-run-progress | Wizard skipped every other step, successful UI workflows left no run history/debug trace, completion had no run link, and structured textarea values could fail final validation. | Wizard advances sequentially, successful completions record shared workflow runs, recent runs appear on the list, completion links to record/debugger, and array/object strings are coerced. | Complete |
| agent-and-agent-team-consoles | Single-agent console was form-first and raw-JSON-heavy; declared team console and team POST were 404 dead ends. | Conversation-first agent/team consoles, preserved prompt context, team lanes and handoff flow, raw JSON disclosures, and metadata fallback for team routes/invocation. | Complete |
| capability-console-rules-config-approval | Three blank JSON boxes and generic/raw results made rules, configuration, and approvals hard to test without knowing the payload contract. | Prefilled generated contexts, preserved submitted JSON, operation-specific summaries, capability profile, and secondary raw JSON disclosures. | Complete |
| database-catalog | Declared databases could render as empty: example 20 showed 0 schemas/tables and `/databases/ERPDB/schemas` returned `[]`. | Generated schemas are inferred from record entities, UI renders tables/columns/indexes/constraints, and schema JSON exposes the same metadata. | Complete |

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
| record-detail-activity-related | Related-record pages fell back to raw JSON. | Moved related count computation into Python and removed brittle Jinja list summing. | Resolved |
| record-detail-activity-related | Related entity sections were hidden unless child records already existed. | Added related candidates with empty-state CTAs and filtered list links. | Resolved |
| record-detail-activity-related | Record review lacked copy and adjacent navigation. | Added copy-link, previous, and next header controls. | Resolved |
| record-detail-activity-related | Display title selected generated identifiers before meaningful names. | Preferred `legal_name`, `full_name`, `name`, and similar fields. | Resolved |
| create-edit-forms-drawer-inline-edit | Create form disabled native browser validation. | Removed `novalidate` and generated required/type attributes plus field helper text. | Resolved |
| create-edit-forms-drawer-inline-edit | JSON/list/dict fields were hard to edit in one-line controls. | Generated textareas with structured placeholders for create, edit, and inline editors. | Resolved |
| create-edit-forms-drawer-inline-edit | Failed creates lost the surrounding workspace context. | Re-rendered the entity workspace with the validation notice instead of returning a bare fragment. | Resolved |
| create-edit-forms-drawer-inline-edit | Inline editing treated typed values as generic text. | Generated semantic controls for numbers, dates, email/phone/url, and structured fields. | Resolved |
| create-edit-forms-drawer-inline-edit | Drawer edits could be discarded accidentally and had no keyboard submit affordance. | Added dirty-state guard through the shared confirm modal and Ctrl/Cmd-S form submission. | Resolved |
| workflow-list-wizard-run-progress | Wizard form action and POST handler both advanced the step, skipping every other step. | Rendered form actions to the current step and kept advancement in the POST handler. | Resolved |
| workflow-list-wizard-run-progress | Completed UI wizards did not create workflow runs. | Recorded successful wizard completion in the shared `WORKFLOW_RUNS` store with a trace and created-record metadata. | Resolved |
| workflow-list-wizard-run-progress | Workflow list had no run history context. | Added recorded run counts and a recent-runs panel linking to the debugger. | Resolved |
| workflow-list-wizard-run-progress | Completion state had no direct debugger or created-record path. | Added recorded-run summary plus `Open created record` and `Inspect run` links. | Resolved |
| workflow-list-wizard-run-progress | Structured textarea values remained strings through record coercion. | Parsed array/object JSON strings in `_coerce_value_for_type()`. | Resolved |
| agent-and-agent-team-consoles | Declared team routes returned `Unknown agent team`. | Added team-description fallback from agent-team entity metadata. | Resolved |
| agent-and-agent-team-consoles | Team invocation returned 404 when sidecar team catalog was empty. | Added entity-metadata team invocation fallback that invokes declared member agents when available. | Resolved |
| agent-and-agent-team-consoles | In-memory generated apps could not render agent consoles without `ai_agents.py`. | Added semantic-model fallback for agent descriptions. | Resolved |
| agent-and-agent-team-consoles | Console placed form/raw JSON before conversational output. | Reworked template into conversation, composer, structured payload disclosure, and secondary raw JSON details. | Resolved |
| agent-and-agent-team-consoles | Team membership and handoff flow were invisible. | Added team lanes and handoff flow side panels. | Resolved |
| capability-console-rules-config-approval | Console started with blank JSON payloads. | Generated rule, configuration, and approval defaults from the capability description. | Resolved |
| capability-console-rules-config-approval | Submitted operation JSON was not preserved after POST. | Passed the active raw JSON field back into the capability console template. | Resolved |
| capability-console-rules-config-approval | Rule/configuration/approval results were raw or generic. | Added operation-specific summaries for matched rules/actions, resolved configuration, and approvers. | Resolved |
| capability-console-rules-config-approval | Capability rules and defaults were hidden in raw JSON. | Added a capability profile panel with default configuration and declared rules. | Resolved |
| capability-console-rules-config-approval | Regression coverage did not load generated capability companion modules. | Compiled the test app to a temporary output directory and imported the generated module with that directory on `sys.path`. | Resolved |
| database-catalog | Database declarations without explicit schemas produced empty schema lists. | Inferred schemas from generated record entities when no authored schema exists. | Resolved |
| database-catalog | Schema JSON endpoint returned `schemas: []` for example 20. | Reused enriched `list_databases()` output for `/databases/<name>/schemas`. | Resolved |
| database-catalog | Catalog UI did not expose table and column metadata. | Reworked the template into summary cards, database cards, table/column grids, constraints, indexes, and validation details. | Resolved |
| database-catalog | Validation warned that generated databases did not declare schemas even when entity metadata was sufficient. | Added inferred generated schema tables with synthetic primary-key columns, producing clean validation. | Resolved |
| database-catalog | Raw validation detail competed with the main catalog. | Kept warnings visible and moved raw validation JSON behind a disclosure. | Resolved |
