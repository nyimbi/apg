# Entity List Thinking

The existing APG entity list already had saved views, filter chips, server-side search, pagination, bulk selection, CSV export, and a create drawer. Round 2 should make those powers discoverable and faster rather than duplicating them.

The best commercial reference is Linear for speed and saved/shareable views. Airtable is the record-grid reference, but APG can beat it by deriving useful table controls from the compiled entity schema with no manual base configuration.

The lowest-risk implementation is metadata-driven: no new dependency, no new route, no large JavaScript file. The UI exposes a list intelligence strip and a tiny inline script for copy URL, keyboard search focus, and density persistence.
