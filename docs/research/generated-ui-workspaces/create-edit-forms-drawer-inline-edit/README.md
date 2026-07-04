# Create/Edit Forms, Drawer, Inline Edit

Workspace: create drawer in `entity_list.html.j2`, generated field inputs in `compiler/code_generator.py`, and inline edit fragments in `record_detail.html.j2`/generated helpers.

## Best-in-Class References

- NN/g form error guidance: users need clearly identified field problems and a path back to correction.
- Baymard inline validation research: inline/native validation prevents avoidable submit failures.
- Material Design text fields: helper/error text should occupy stable space so validation feedback does not cause disruptive layout jumps.
- Salesforce record edit forms: record editing should preserve context and use field-aware controls.

## Live Audit

Representative app: `examples/02_customer_orders_relationship/output/app.py`, booted locally at `127.0.0.1:20890`.

Routes exercised:

- `/ui/entities/Customer`
- `POST /ui/entities/Customer/records` with missing required fields

Defects found:

- Must-fix: create form used `novalidate`, disabling native browser validation.
- Must-fix: required fields had no visible required marker or helper text.
- Must-fix: list/dict fields rendered as single-line text inputs rather than JSON textareas.
- Must-fix: semantic fields such as email/phone/url did not use matching input types.
- Must-fix: drawer had no unsaved draft guard or Ctrl/Cmd-S submit path.
- Must-fix: inline edit did not consistently map decimal/list/dict/date/semantic fields to appropriate controls.

Artifacts:

- `assets/before-customer-list.html`
- `assets/before-customer-list.headers`
- `assets/before-create-error.html`
- `assets/before-create-error.headers`

## Fix Plan

Must-fix:

- Enable native validation by removing `novalidate`.
- Add required markers, `required` attributes, and helper text to generated create controls.
- Use JSON textareas for list/dict/object fields.
- Use semantic input types for email, phone, URL, date, integer, decimal, and numeric fields.
- Re-render create failures in the entity page context with a visible alert.
- Add a drawer draft guard and Ctrl/Cmd-S submit shortcut using the shared accessible confirm dialog.
- Extend inline edit widgets for numeric/date/JSON/semantic fields.

High-value polish:

- Keep the create drawer context rather than dropping users into raw error pages.
- Ensure helper text is server-rendered and testable without client JavaScript.

## After Verdict

Implemented. Create and inline-edit forms now use field-aware controls, native validation, visible required/helper text, contextual error recovery, and drawer workflow safeguards.

