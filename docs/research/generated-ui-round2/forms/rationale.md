# Rationale

## Decision

Ship the form intelligence layer inside `entity_list.html.j2` because APG create forms are rendered from the entity-list drawer. Add autosave restore, delayed native validation telemetry, smart-default chips, dependency visibility, and undoable submit.

## Why this beats the benchmark

The commercial leaders each solve a different part of form UX. APG combines those patterns with generated schema and record context, so the create form can improve repeated internal data entry without manual configuration.

## Rejected alternatives

- Remote async validators: rejected because the generated app has no validation API contract and native validity covers the current generated fields.
- A full conditional logic engine: rejected because the schema does not yet encode conditional display rules.
- Server-side undo queue: rejected because accidental-submit recovery can be delivered with a lighter client-side delay.

## Validation target

Generated entity list HTML must still include native validation, contextual error/help text, and the discard draft guard while adding autosave, validation telemetry, smart defaults, dependency tree, and undoable submit strings.
