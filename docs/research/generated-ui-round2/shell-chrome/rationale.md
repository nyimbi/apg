# Shell Chrome Rationale

## Decisions

- Named Raycast as the leader because command-palette-first productivity is the core benchmark for shell chrome.
- Implemented command metadata in generated Python from known app surfaces. This keeps commands accurate without scanning the DOM.
- Implemented recent items in localStorage. The feature is cross-workspace and requires no backend identity or persistence contract.
- Implemented a shell tour overlay and an undo toast for clearing recents. These are reversible and local.

## Rejected alternatives

- `cmdk` dependency: rejected because APG generated output must avoid SPA frameworks and new runtime packages.
- Server-side recent history: rejected because it needs user/account storage decisions beyond this pass.
- Large third-party guided-tour library: rejected by dependency and JS budget constraints.
- Fake AI launcher commands: rejected because shell commands should point to real generated routes.

## Budget note

The implementation adds only inline shell JavaScript and CSS. It adds no generated Python dependencies and no external runtime URLs.
