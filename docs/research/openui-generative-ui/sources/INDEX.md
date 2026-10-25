# Sources — OpenUI research

Accessed 2026-10-24. Local copies are authoritative; URLs are provenance only.

## Primary specification (archived)

| Document | Local copy | Upstream URL |
| --- | --- | --- |
| OpenUI Overview, v1.0 — what OpenUI is, and a tour of every feature | `openui-lang-1.0-overview.md` | https://github.com/thesysdev/openui/blob/main/spec/overview.md |
| The OpenUI Language, v1.0 — grammar, program model, evaluation, streaming, state, actions, editing, errors | `openui-lang-1.0-language.md` | https://github.com/thesysdev/openui/blob/main/spec/language.md |
| The OpenUI Prompt and LibrarySpec, v1.0 — LibrarySpec format, library rules, prompt generation, signature format, message protocol | `openui-lang-1.0-prompt.md` | https://github.com/thesysdev/openui/blob/main/spec/prompt.md |
| Conformance test library (12 components, 1 function, 1 action) | `openui-fixture-library.json` | https://github.com/thesysdev/openui/blob/main/spec/fixtures/1.0/library.json |

Section numbers cited in `README.md` refer to `language.md` and `prompt.md`.

## Secondary

| Title | URL | Note |
| --- | --- | --- |
| OpenUI project site | https://www.openui.com | Marketing surface. Claims: 3× faster generation, 67% fewer tokens vs JSON, 1,362 tokens/screen average over 46 interfaces. Unverified independently. |
| thesysdev/openui repository | https://github.com/thesysdev/openui | MIT. Packages: `lang-core`, `react-lang`, `react-headless`, `react-ui`, `react-email`, `langchain`, `assistant-ui`, `vue-lang`, `svelte-lang`, `angular-lang`, `browser-bundle`, `a2ui`, `server`, `observability`, `devtools`, `openui-cli`. |

## Caveat on the secondary sources

Token-efficiency numbers come from the vendor's own benchmark page and are not reproduced here. The architecture and grammar claims above are taken from the archived spec text, which is normative.
