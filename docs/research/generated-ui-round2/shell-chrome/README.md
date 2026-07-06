# Shell Chrome Round-2 Research

Date accessed: 2026-07-06

## Best-in-class reference

Commercial leader: Raycast. Raycast is the strongest reference for shell chrome because it makes the command palette the primary product surface, not a secondary search box.

Adjacent references:

- Linear for product navigation and command-menu expectations.
- Notion for keyboard-first slash/command workflows.
- Jira for enterprise keyboard shortcuts and quick operations.

## Leader weaknesses

- Raycast is desktop-native and extension-driven; generated browser apps need the same command speed without an installed launcher.
- Linear and Notion are fast for their own objects, but their recent-item and undo behaviors are product-specific. APG can make these cross-workspace and generated from actual app metadata.
- Jira exposes shortcuts, but they are often help-page oriented. APG can show onboarding directly in the shell and make it dismissible.

## Differentiators proposed

1. Command Center Primary Nav: upgrade `⌘K` from record search to generated app commands, entities, workflows, marketplace, APIs, and agent surfaces.
2. Cross-workspace Recent Items: persist visited generated pages in localStorage and surface them directly in the command palette.
3. Onboarding Tour: expose a shell-level tour for command center, recent items, notifications, theme, install, and offline mode.
4. Undo Toast Stack: add a shell undo toast for reversible chrome actions, starting with clearing recent items.

## Prioritized implementation

Ship all four in `_html_page()` and `apg.css` so every generated shell benefits without touching page templates.
