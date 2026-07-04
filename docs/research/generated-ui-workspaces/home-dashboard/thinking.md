# Raw Thinking

The Home dashboard should answer: "What changed, what needs attention, and where do I start?" The current generated surface already has charts and entity lists from WP3, but it still behaves partly like an API index. Example 20 is the right audit target because it has entities, capabilities, agents, a team, a database, and workflows.

Live audit facts:

- `/ui` returned HTTP 200 and a 24,068 byte HTML payload before Home changes.
- The page had an `h1`, skip link, offline banner, 6 stat values, and 6 chart hooks.
- Recent activity was empty but only displayed a passive empty message.
- "Quick nav" linked to API/internal surfaces before user workflow surfaces.
- `describe_application()` in example 20 only returned `name`, `version`, `description`, `entities`, and `databases`; template inputs for capabilities, agents, routes, and screens were therefore unreliable.

Implementation reasoning:

- The generated `ENTITIES` list is the most reliable source for the home workspace because all APG constructs are present there with `type`.
- Stat cards should represent record-owning entities only; otherwise databases, apps, workflows, and agent/team constructs consume the first four KPI slots.
- API links remain useful for generated apps, but Home should put "Start with first entity", workflows, database catalog, and marketplace first.
- Empty states should be actionable. For Home, the first useful action is create/view the first primary record, because that creates activity and data for charts.

Risk notes:

- Kept changes scoped to `app_index.html.j2` and `_ui_dashboard_context()` to avoid disturbing entity-list or shell behavior.
- Used existing classes to avoid expanding the CSS utility surface unnecessarily.
