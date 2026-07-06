# Forms Round-2 Research

## Commercial leader

Typeform is the leader for polished form completion and conditional logic. Airtable is the strongest reference for database-connected forms and prefilling from record context, while Jotform is a strong reference for save-and-continue completion flows.

## Leader weaknesses

- Typeform optimizes the respondent journey, but operational users still need table-context defaults, schema-derived validation, and fast duplicate-entry handling.
- Airtable can prefill forms through URL parameters and context, but linked-record and sibling-context behavior is limited and often requires manual setup.
- Jotform save-and-continue is useful, but the flow is explicit; users can still lose progress if they forget to save or recover the link.
- Formik and other developer libraries support async validation, but they require app-specific JavaScript architecture rather than generated, framework-free behavior.

## Differentiators proposed

1. Autosave Draft: persist in-progress drawer data to local storage and restore it automatically when the user reopens the form.
2. Async Field Validation: run delayed field checks against native validity and expose form readiness before submit.
3. Smart Defaults: infer safe values from sibling records and apply them field by field without copying IDs or revision metadata.
4. Undoable Submit: delay final submission briefly so accidental creates can be canceled with Escape.
5. Dependency Tree: show the generated field influences that drive submit readiness.

## Shipped verdict

APG now turns the generated create drawer into an operational form cockpit. Before, the surface had native validation and a discard guard. After, it adds local recovery, validation telemetry, schema-derived sibling defaults, dependency visibility, and an undo window without adding a dependency or runtime URL.
