# Raw Thinking

The record-detail workspace already had a good foundation: title header, status badge, previous and next navigation, copy link, inline field editing, related lists, and activity notes. The missing layer was decision compression. A world-class detail page should answer "what changed?", "what else will this affect?", and "how do I create a similar item?" without forcing the user to hunt through tabs.

Notion is the leader because database rows become rich pages, but generic generated APG apps can beat that for operational data by using structured schema facts. Airtable and Linear are useful references for linked records and issue relations, yet both still assume the user already knows the domain model. APG can infer downstream lists from FK naming conventions, activity from the generated event log, and safe sibling fields from the visible record.

Rejected ideas:

- Building a true graph canvas. It would add JS and layout weight while the generated app only needs a compact relation map.
- Adding a new clone backend route. That would be useful later, but a safe visual context panel fits the current allowed surface and avoids route/test blast radius.
- Calculating real historical field diffs. The generated event log does not retain prior field values, so the honest version is a current snapshot timeline tied to revision and activity count.

The shipped implementation should preserve all existing tests because it only adds optional context passed into the existing record-detail template.
