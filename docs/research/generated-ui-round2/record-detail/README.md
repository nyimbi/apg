# Record Detail Round-2 Research

## Commercial leader

Notion database pages are the best-in-class reference for this surface because every database item opens as a full page with properties plus collaborative context. Airtable record-detail interfaces are the strongest adjacent benchmark for linked-record traversal, and Linear issue detail is the benchmark for high-signal relationships and activity.

## Leader weaknesses

- Notion pages are flexible, but important operational relationships can become buried across properties, comments, and page body content.
- Airtable linked records are strong for navigation, but the user still has to infer downstream impact and safe duplication context.
- Linear issue details are excellent for software work, but their relationship model is domain-specific and does not generalize automatically to arbitrary APG entities.
- None of the leaders produce a compact, generated, zero-dependency record cockpit from the entity schema without app-specific configuration.

## Differentiators proposed

1. Change Diff Timeline: summarize populated fields and current revision beside the activity count so reviewers can scan what matters before opening the activity tab.
2. Related Record Graph: convert FK-derived related lists into a compact relationship graph with filtered drilldown links.
3. Create Sibling Context: expose safe default fields from the current record to accelerate similar-record creation without copying IDs or relationship internals.
4. Generated everywhere: derive the intelligence layer from schema, record, relationship, and activity data already available in generated apps.

## Shipped verdict

APG now adds a record intelligence command center above the detail tabs. The before state had solid details, related records, and activity tabs; the after state surfaces review, graph, and duplication decisions before the user chooses a tab.
