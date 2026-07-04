# Raw Reasoning

This workspace had a hidden correctness failure: the page looked acceptable when no relationships existed but collapsed to raw JSON when related records were present. The root cause was a Jinja aggregation expression adding lists to an integer. Counts belong in the generator where the data structure is explicit.

The generated UI should not hide relationship possibilities just because there are no child records yet. Showing empty related sections with clear CTAs makes the schema visible and avoids dead ends.

Copy link and previous/next navigation are small additions, but they matter for record-review workflows. They also do not require new persistence or dependencies.

Rejected for this workspace:

- Full attach-existing/create-new related record drawers. Valuable but better handled with the create/edit forms workspace.
- Threaded activity comments. The current note form is enough for this slice once the page renders reliably.
- Delete undo. Important, but broader than this record-detail repair and tied to delete semantics across list/detail.

