# Raw Reasoning

The workflow workspace should answer three user questions immediately:

1. What guided work can I start?
2. Where am I in the current workflow?
3. What happened after the workflow completed?

Before this pass, the first question was partly answered, the second was visually answered but mechanically wrong because the route skipped steps, and the third was not answered at all because UI wizard completion created only a record, not a workflow run.

Best references converge on the same pattern: workflow UIs need state and history. Temporal and Airflow make runs inspectable; Camunda makes process instance state operational; Linear makes status flow low-friction. APG already had a debugger and run store, so the strictest fix is to connect the generated wizard to that existing run model instead of creating another UI-only concept.

The route bug is subtle: `_ui_workflow_wizard_html()` rendered a form action for `step_index + 1`, and `_ui_workflow_step_post()` also advanced by one. Fixing the action to the current step makes the POST handler the single owner of advancement.

The structured-field issue surfaced during live verification rather than static inspection. The wizard rendered list/object fields as JSON textareas, but `create_record()` eventually validated Python values, and `_coerce_value_for_type()` did not parse JSON strings into list/dict values. Since `coerce_record_types()` is the shared boundary, fixing it there is better than adding a workflow-only parser.

Rejected: using `run_workflow()` for UI wizard completion. That function is for DSL-declared workflows returned by `list_workflows()`, while `APP_WORKFLOWS` are generated UI workflows over entity fields. Recording a compatible run object preserves the debugger contract without pretending the UI wizard is a DSL workflow.

Rejected: adding a separate `/ui/workflows/runs` page. The debugger already has recent run and trace views, so the workflow list only needs a recent-runs summary and drill-in link.
