# Raw Reasoning

WP2 already moved create forms into a drawer, so the remaining quality gap was form correctness and recovery. The generated drawer looked polished but disabled native validation and provided little guidance for required or structured fields.

The conservative fix is to enrich the generated controls rather than adding a client validation framework. HTML required/type attributes, helper text, and server-rendered contextual errors are durable and dependency-free.

Inline edit should match the same field semantics as create. Decimal and JSON fields are common in APG examples, so treating them as plain text creates avoidable data entry errors.

Rejected for this workspace:

- A new client-side validation library. Native validation plus server errors covers the must-fix scope.
- Full field-level server error mapping back into each control. Valuable, but requires carrying submitted values/error maps through the form renderer; the current contextual alert is the safe step.
- Custom date picker. Native date input is sufficient and keeps generated apps dependency-light.

