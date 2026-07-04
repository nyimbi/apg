# Rationale

## Decisions

- Use native validation attributes instead of a new JavaScript validator.
  - Reason: dependency-free, accessible, and aligned with generated HTML.
- Re-render create failures in page context.
  - Reason: users should stay in the entity workflow and see what failed.
- Use JSON textareas for list/dict/object fields.
  - Reason: structured values need multiline entry and clear helper text.
- Add Ctrl/Cmd-S and draft guard to the drawer.
  - Reason: expected power-user behavior without changing data semantics.
- Extend inline edit controls to match create controls.
  - Reason: record editing should not degrade after creation.

## Rejected Alternatives

- New validation dependency.
  - Rejected because generated apps must remain dependency-light.
- Field-specific server error maps in this slice.
  - Rejected as larger than needed for the must-fix gate.
- Custom combobox/date widgets.
  - Rejected because existing native controls and FK selects are sufficient for this pass.

