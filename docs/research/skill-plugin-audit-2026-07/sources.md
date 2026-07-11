# Sources

Accessed: 2026-07-11. All sources are local files; no web sources were used.

## Configuration Files

| Source | Use |
|---|---|
| `/Users/nyimbiodero/.claude/settings.json` | Read enabled plugin list, known marketplaces, model/hooks/permission context. Not modified. |
| `/Users/nyimbiodero/.claude/settings.local.json` | Checked local settings overlay. Not modified. |
| `/Users/nyimbiodero/.claude/plugins/installed_plugins.json` | Read installed plugin records, versions, install paths, update timestamps. |
| `/Users/nyimbiodero/.claude/CLAUDE.md` | Read global Claude Code instructions and confirmed it does not contain the pjs routing table. |
| `/Users/nyimbiodero/src/pjs/CLAUDE.md` | Read current pjs skill routing table and old aliases. Target for optimized table. |
| `/Users/nyimbiodero/.claude/commands` | Checked for custom user slash commands; directory does not exist. |

## Plugin Manifests And Package Metadata

| Source | Use |
|---|---|
| `/Users/nyimbiodero/.claude/plugins/cache/omc/oh-my-claudecode/4.15.3/.claude-plugin/plugin.json` | Current OMC plugin manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/omc/oh-my-claudecode/4.15.3/package.json` | Current OMC package metadata. |
| `/Users/nyimbiodero/.claude/plugins/cache/openai-codex/codex/1.0.5/.claude-plugin/plugin.json` | Codex plugin manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/frontend-design/unknown/.claude-plugin/plugin.json` | Active frontend-design plugin manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/claude-md-management/1.0.0/.claude-plugin/plugin.json` | CLAUDE.md management plugin manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/claude-code-setup/1.0.0/.claude-plugin/plugin.json` | Claude Code setup plugin manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/remember/0.8.3/.claude-plugin/plugin.json` | Active remember plugin manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/anthropic-agent-skills/document-skills/9d2f1ae18723` | Active document-skills plugin skill source. |
| `/Users/nyimbiodero/.claude/plugins/cache/ui-ux-pro-max-skill/ui-ux-pro-max/2.5.0/.claude-plugin/plugin.json` | UI/UX Pro Max plugin manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/ui-ux-pro-max-skill/ui-ux-pro-max/2.5.0/skill.json` | UI/UX Pro Max skill metadata. |
| `/Users/nyimbiodero/.claude/plugins/cache/knowledge-work-plugins/design/1.2.0/.claude-plugin/plugin.json` | Design skill-source manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/knowledge-work-plugins/sales/1.3.0/.claude-plugin/plugin.json` | Sales skill-source manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/knowledge-work-plugins/product-management/1.2.0/.claude-plugin/plugin.json` | Product-management skill-source manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/knowledge-work-plugins/marketing/1.2.0/.claude-plugin/plugin.json` | Marketing skill-source manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/knowledge-work-plugins/finance/1.3.0/.claude-plugin/plugin.json` | Finance skill-source manifest. |
| `/Users/nyimbiodero/.claude/plugins/cache/knowledge-work-plugins/engineering/1.2.0/.claude-plugin/plugin.json` | Engineering skill-source manifest. |

## Skill And Command Sources

| Source | Use |
|---|---|
| `/Users/nyimbiodero/.claude/skills/*/SKILL.md` | Catalogued installed user-level skills and inferred source plugins. |
| `/Users/nyimbiodero/.claude/plugins/cache/omc/oh-my-claudecode/4.15.3/skills/*/SKILL.md` | Catalogued active OMC skills. |
| `/Users/nyimbiodero/.claude/plugins/cache/omc/oh-my-claudecode/4.15.3/commands/*.md` | Catalogued active OMC slash commands. |
| `/Users/nyimbiodero/.claude/plugins/cache/openai-codex/codex/1.0.5/skills/*/SKILL.md` | Catalogued Codex helper skills. |
| `/Users/nyimbiodero/.claude/plugins/cache/openai-codex/codex/1.0.5/commands/*.md` | Catalogued Codex slash commands. |
| `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/*/*/skills/*/SKILL.md` | Catalogued official plugin skills. |
| `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/*/*/commands/*.md` | Catalogued official plugin commands. |
| `/Users/nyimbiodero/.claude/plugins/cache/anthropic-agent-skills/document-skills/9d2f1ae18723/skills/*/SKILL.md` | Catalogued document and artifact skills. |
| `/Users/nyimbiodero/.claude/plugins/cache/ui-ux-pro-max-skill/ui-ux-pro-max/2.5.0/.claude/skills/*/SKILL.md` | Catalogued UI/UX Pro Max CKM skills. |
| `/Users/nyimbiodero/.claude/plugins/cache/knowledge-work-plugins/*/*/skills/*/SKILL.md` | Catalogued knowledge-work skills mirrored into `~/.claude/skills`. |

## Redundant Cache Copies Observed

These were detected during inventory but excluded from the optimized routing recommendation because newer/current copies exist and no uninstall was requested:

- `/Users/nyimbiodero/.claude/plugins/cache/omc/oh-my-claudecode/4.15.2`
- `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/remember/0.7.3`
- `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/frontend-design/6d578313aa15`
- `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/frontend-design/7d0e5f5aae16`
- `/Users/nyimbiodero/.claude/plugins/cache/claude-plugins-official/frontend-design/8326199c6ec6`
- `/Users/nyimbiodero/.claude/plugins/cache/anthropic-agent-skills/document-skills/35414756ca55`
- `/Users/nyimbiodero/.claude/plugins/cache/anthropic-agent-skills/document-skills/575462609294`
