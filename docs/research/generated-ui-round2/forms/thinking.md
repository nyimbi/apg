# Raw Thinking

The generated form lives inside the entity list drawer, so the highest-impact change is to improve that drawer rather than create a separate forms page. The existing implementation already has native input types, contextual JSON help, and a discard confirmation. Round 2 should make it resilient and faster for repeated data entry.

Typeform is the commercial polish benchmark, but APG can beat it for internal applications because it knows the entity schema and current records. Airtable's prefill model points toward context-aware defaults, while Jotform's save-and-continue pattern points toward recovery. The differentiator is combining those into a generated, zero-dependency drawer.

The undoable submit uses a short client-side delay instead of a server-side tombstone because generated create semantics are simple and the requirement is to stay narrow. Async validation is simulated with delayed native validity checks because no remote validation endpoint exists yet; this is honest and still improves the user's feedback loop.

Rejected ideas:

- Adding a validation API route per entity. Useful later, but it would widen the backend contract and test surface.
- Adding a clone-from-record route. That belongs with explicit unique-field policy.
- Adding a form library. The ground rules prohibit SPA frameworks and there is no need for a dependency.
