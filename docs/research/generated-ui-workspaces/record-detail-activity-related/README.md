# Record Detail, Activity, Related

Workspace: `/ui/entities/<Entity>/<record_id>` via `compiler/templates/record_detail.html.j2`.

## Best-in-Class References

- Salesforce Lightning: record pages lead with a highlights panel and primary actions so users understand the object before diving into fields.
- HubSpot CRM: record timelines make activity history and notes part of the record workflow, not a separate debug log.
- Salesforce related lists: related records are a core navigation surface and should remain visible even when empty.
- Notion pages: backlinks/relations help users understand connected information without leaving the current page.

## Live Audit

Representative app: `examples/02_customer_orders_relationship/output/app.py`, booted locally at `127.0.0.1:20888`.

Route exercised: `/ui/entities/Customer/1`, with Customer and related Order records seeded through the generated API.

Defects found:

- Must-fix: record detail fell back to raw JSON when related records existed because `related_lists | sum(attribute='records')` attempted to sum lists.
- Must-fix: related entities only appeared when records existed, leaving no empty related-list CTA for modeled relationships.
- Must-fix: no copy-link action and no previous/next record navigation.
- Must-fix: the display title preferred the first string field (`customer_number`) over clearer name fields such as `legal_name`.
- Polish: related-list links were generic rather than filtered to the current record.

Artifacts:

- `assets/before-customer-detail.html`
- `assets/before-customer-detail.headers`

## Fix Plan

Must-fix:

- Move related counts into the generated render context.
- Include related-list candidates even when they have zero records.
- Add filtered related-list URLs using `filter.<fk_field>=<record_id>`.
- Add copy-link, previous, and next controls to the record header.
- Prefer semantic title fields (`legal_name`, `full_name`, `name`, `title`) over generic first string fields.

High-value polish:

- Add related empty-state CTAs for filtered list and related create workflow.
- Preserve activity timeline and note entry while keeping the record page out of raw JSON fallback.

## After Verdict

Implemented. Record detail now renders reliably with related records, highlights a readable title, exposes copy/next navigation, and turns related lists into filtered workflow links instead of a brittle tab count.

