# Skill And Plugin Audit - 2026-07

## Scope

Audit target: Claude Code configuration for `/Users/nyimbiodero`, with the pjs project routing table at `/Users/nyimbiodero/src/pjs/CLAUDE.md`.

Constraints honored:

- No plugins were removed or uninstalled.
- `~/.claude/settings.json` was read only and not changed.
- `~/.claude/settings.local.json` was read only and not changed.
- Custom command directory `~/.claude/commands` does not exist, so there are no user-level custom slash commands to catalogue from that location.
- The global `~/.claude/CLAUDE.md` does not contain the pjs skill routing table; the routing table is in `/Users/nyimbiodero/src/pjs/CLAUDE.md`.

## Inventory Summary

Enabled plugins from `settings.json` and `installed_plugins.json`:

| Plugin | Version / path | Primary surface |
|---|---|---|
| `oh-my-claudecode@omc` | `4.15.3` | OMC orchestration, planning, research, verification, runtime workflows |
| `frontend-design@claude-plugins-official` | `unknown` cache version | Frontend implementation design |
| `claude-md-management@claude-plugins-official` | `1.0.0` | CLAUDE.md maintenance |
| `claude-code-setup@claude-plugins-official` | `1.0.0` | Claude Code automation recommendations |
| `remember@claude-plugins-official` | `0.8.3` | Continuous Claude Code memory |
| `document-skills@anthropic-agent-skills` | `9d2f1ae18723` | PDF/DOCX/PPTX/XLSX, artifacts, MCP, skill creation |
| `ui-ux-pro-max@ui-ux-pro-max-skill` | `2.5.0` | UI/UX design intelligence |
| `codex@openai-codex` | `1.0.5` | Codex delegation and review commands |

Additional installed skill sources found under `~/.claude/skills` map to cached knowledge-work plugin surfaces: `design`, `sales`, `product-management`, `marketing`, `finance`, and `engineering`. These are treated as routable because the skills are physically installed in the user skill directory.

Stale/redundant cache copies were observed but not used for the clean routing recommendation: OMC `4.15.2`, remember `0.7.3`, older `frontend-design` cache hashes, and older `document-skills` cache hashes. They should be ignored for routing unless the plugin manager later marks them current.

## Coverage Summary

| Surface | Best current coverage |
|---|---|
| Codebase discovery | `codebase-memory`, `trace`, `deep-dive` |
| Planning / architecture | `system-design`, `architecture`, `ralplan`, `team`, `omc-plan` |
| Frontend / UI / UX | `frontend-design`, `ui-ux-pro-max`, `design-system`, `accessibility-review`, `webapp-testing` |
| Code review / verification | `code-review`, `testing-strategy`, `verify`, `ultraqa`, `/codex:review` |
| Debugging / incident response | `debug`, `incident-response`, `/codex:rescue` |
| Docs / specs / project memory | `documentation`, `write-spec`, `wiki`, `remember`, `doc-coauthoring` |
| Research | `autoresearch`, `external-context`, `research-synthesis`, `synthesize-research` |
| Document files | `pdf`, `docx`, `pptx`, `xlsx` |
| Deployment / release | `deploy-checklist`, `release` |
| Skill / MCP creation | `skill-creator`, `skillify`, `mcp-builder` |
| Codex bridge | `/codex:transfer`, `/codex:review`, `/codex:rescue`, `codex-cli-runtime` |

## Redundancies And Keep Decisions

| Redundancy | Keep / route to | Why |
|---|---|---|
| Old pjs `web-design-guidelines` vs installed UI skills | `ui-ux-pro-max` for broad UI/UX; `frontend-design` for implementation | The old alias is not installed. The two kept skills split design intelligence from implementation guidance. |
| Official `frontend-design`, document-skills `frontend-design`, and UI/UX Pro Max | `frontend-design` and `ui-ux-pro-max`; avoid routing to duplicate document-skill frontend aliases | Official frontend-design is the narrow frontend implementation skill; UI/UX Pro Max is the richer design/polish surface. |
| `design-system` vs `ckm:design-system` | `design-system` for pjs routing; `ckm:design-system` only when explicitly using UI/UX Pro Max CKM workflows | The plain skill is simpler and already in `~/.claude/skills`. |
| `research-synthesis`, `synthesize-research`, `autoresearch`, `external-context` | `autoresearch` for new research folders; `external-context` for lookup; synthesis skills for user/product research data | They overlap on research but differ by workflow depth and input type. |
| `code-review`, `/codex:review`, `/codex:adversarial-review`, OMC reviewer agents | `code-review` for interactive review; `/codex:review` for local git-state review; adversarial review only for high-risk changes | Avoids making every review a multi-agent or Codex job while preserving stronger review paths. |
| `debug` engineering skill vs `/oh-my-claudecode:debug` runtime command | `debug` for application failures; `/oh-my-claudecode:debug` for OMC runtime/session state | Same label, different target system. |
| Official `remember` vs OMC `remember` | Official plugin for passive memory; OMC `remember` for explicit project-knowledge triage | Avoids confusing memory extraction with project note routing. |
| `marketing:competitive-brief`, `product-management:competitive-brief`, `competitive-brief`, `competitive-intelligence` | Keep namespaced business skills out of pjs engineering routing | These are business-workflow tools, not default engineering research skills. |
| Stale cache versions | Do not route to stale copies | Current enabled/copy-installed surfaces cover the same functionality. |

## Gaps

Exact aliases in the existing pjs routing table with no installed skill under that exact name:

| Missing alias | Covered by now | Recommended future addition |
|---|---|---|
| `vercel-react-best-practices` | `frontend-design`, `webapp-testing` | Add a React/Next.js best-practices skill if this remains a common route. |
| `web-design-guidelines` | `ui-ux-pro-max`, `frontend-design` | No add needed unless a lighter local style-guide skill is desired. |
| `api-design-principles` | `system-design` | Add an API-design skill if OpenAPI/resource-shape work is frequent. |
| `async-python-patterns` | `system-design`, `code-review` | Add an async Python/FastAPI concurrency skill. |
| `fastapi-templates` | `system-design`, `testing-strategy` | Add a FastAPI implementation/template skill. |
| `postgresql-table-design` | `system-design` | Add a PostgreSQL schema/data-modeling skill. |
| `sql-optimization-patterns` | `tech-debt`, `code-review` | Add a SQL query-planning and index-optimization skill. |
| `langchain-architecture` | `architecture` | Add a LangChain/LangGraph architecture skill if agent framework work is active. |
| `mermaid-diagrams` | `documentation` | Add a Mermaid diagram skill if diagram generation is frequent. |

## Proposed CLAUDE.md Skill Routing Table

| Task | Skill |
|---|---|
| Codebase discovery / call graph / impact analysis | `codebase-memory` |
| React/Next.js / frontend implementation | `frontend-design` |
| UI/UX planning, review, visual polish | `ui-ux-pro-max` |
| Design systems / accessibility | `design-system`, `accessibility-review` |
| Browser-based web app testing | `webapp-testing` |
| API, service, and boundary design | `system-design` |
| Architecture decisions / ADRs | `architecture` |
| Python async / FastAPI implementation | `system-design` |
| PostgreSQL schema / data modeling | `system-design` |
| SQL optimization / query performance | `tech-debt` |
| LangChain/LangGraph / agent architecture | `architecture` |
| Testing strategy / coverage planning | `testing-strategy` |
| Debugging / failures / stack traces | `debug` |
| Code review / security / performance / correctness | `code-review` |
| Cleanup / refactor / AI slop | `ai-slop-cleaner`, `tech-debt` |
| Deployment / release readiness | `deploy-checklist`, `release` |
| Incident response / postmortems | `incident-response` |
| Technical docs / READMEs / runbooks | `documentation` |
| Product specs / PRDs | `write-spec` |
| Research / external reference lookup | `autoresearch`, `external-context` |
| Research synthesis | `research-synthesis`, `synthesize-research` |
| Diagrams / Mermaid | `documentation` |
| PDF / DOCX / PPTX / XLSX | `pdf`, `docx`, `pptx`, `xlsx` |
| CLAUDE.md / Claude Code setup | `claude-md-improver`, `claude-automation-recommender` |
| Skill / MCP creation | `skill-creator`, `mcp-builder`, `skillify` |
| Codex delegation / review | `/codex:transfer`, `/codex:review`, `/codex:rescue` |

Application status: replacing the table in `/Users/nyimbiodero/src/pjs/CLAUDE.md` was attempted from the `apg` workspace, but the write was rejected because that file is outside the writable repo root. The table above is the exact optimized replacement.

## Full Catalogue

| Name | Source | Purpose | Surface |
|---|---|---|---|
| `/claude-md-management:revise-claude-md` | claude-md-management@claude-plugins-official | Update CLAUDE.md with learnings from this session | plugin command |
| `/codex:adversarial-review` | codex@openai-codex | Run a Codex review that challenges the implementation approach and design choices | Codex command |
| `/codex:cancel` | codex@openai-codex | Cancel an active background Codex job in this repository | Codex command |
| `/codex:rescue` | codex@openai-codex | Delegate investigation, an explicit fix request, or follow-up rescue work to the Codex rescue subagent | Codex command |
| `/codex:result` | codex@openai-codex | Show the stored final output for a finished Codex job in this repository | Codex command |
| `/codex:review` | codex@openai-codex | Run a Codex code review against local git state | Codex command |
| `/codex:setup` | codex@openai-codex | Check whether the local Codex CLI is ready and optionally toggle the stop-time review gate | Codex command |
| `/codex:status` | codex@openai-codex | Show active and recent Codex jobs for this repository, including review-gate status | Codex command |
| `/codex:transfer` | codex@openai-codex | Transfer the current Claude Code session into a resumable Codex thread | Codex command |
| `/oh-my-claudecode:ask` | oh-my-claudecode@omc | OMC ask | OMC command |
| `/oh-my-claudecode:autoresearch` | oh-my-claudecode@omc | OMC autoresearch | OMC command |
| `/oh-my-claudecode:ccg` | oh-my-claudecode@omc | OMC ccg | OMC command |
| `/oh-my-claudecode:compact` | oh-my-claudecode@omc | Prepare OMC context for a manual Claude Code /compact handoff. | OMC command |
| `/oh-my-claudecode:configure-notifications` | oh-my-claudecode@omc | OMC configure-notifications | OMC command |
| `/oh-my-claudecode:debug` | oh-my-claudecode@omc | OMC debug | OMC command |
| `/oh-my-claudecode:deep-dive` | oh-my-claudecode@omc | OMC deep-dive | OMC command |
| `/oh-my-claudecode:deepinit` | oh-my-claudecode@omc | OMC deepinit | OMC command |
| `/oh-my-claudecode:external-context` | oh-my-claudecode@omc | OMC external-context | OMC command |
| `/oh-my-claudecode:hud` | oh-my-claudecode@omc | OMC hud | OMC command |
| `/oh-my-claudecode:learner` | oh-my-claudecode@omc | OMC learner | OMC command |
| `/oh-my-claudecode:mcp-setup` | oh-my-claudecode@omc | OMC mcp-setup | OMC command |
| `/oh-my-claudecode:omc-doctor` | oh-my-claudecode@omc | OMC omc-doctor | OMC command |
| `/oh-my-claudecode:omc-setup` | oh-my-claudecode@omc | OMC omc-setup | OMC command |
| `/oh-my-claudecode:omc-teams` | oh-my-claudecode@omc | OMC omc-teams | OMC command |
| `/oh-my-claudecode:project-session-manager` | oh-my-claudecode@omc | OMC project-session-manager | OMC command |
| `/oh-my-claudecode:psm` | oh-my-claudecode@omc | OMC psm | OMC command |
| `/oh-my-claudecode:release` | oh-my-claudecode@omc | OMC release | OMC command |
| `/oh-my-claudecode:remember` | oh-my-claudecode@omc | OMC remember | OMC command |
| `/oh-my-claudecode:sciomc` | oh-my-claudecode@omc | OMC sciomc | OMC command |
| `/oh-my-claudecode:self-improve` | oh-my-claudecode@omc | OMC self-improve | OMC command |
| `/oh-my-claudecode:skill` | oh-my-claudecode@omc | OMC skill | OMC command |
| `/oh-my-claudecode:skillify` | oh-my-claudecode@omc | OMC skillify | OMC command |
| `/oh-my-claudecode:trace` | oh-my-claudecode@omc | OMC trace | OMC command |
| `/oh-my-claudecode:verify` | oh-my-claudecode@omc | OMC verify | OMC command |
| `/oh-my-claudecode:visual-verdict` | oh-my-claudecode@omc | OMC visual-verdict | OMC command |
| `/oh-my-claudecode:wiki` | oh-my-claudecode@omc | OMC wiki | OMC command |
| `/oh-my-claudecode:writer-memory` | oh-my-claudecode@omc | OMC writer-memory | OMC command |
| `/product-management:brainstorm` | knowledge-work-plugins/product-management | Brainstorm a product idea, problem space, or strategic question with a sharp thinking partner | plugin command |
| `accessibility-review` | knowledge-work-plugins/design | Run a WCAG 2.1 AA accessibility audit on a design or page. Trigger with "audit accessibility", "check a11y", "is this accessible?", or when reviewing a design for color contrast, keyboard navigation, touch target size, or screen reader behavior before handoff. | UI/UX/accessibility |
| `account-research` | knowledge-work-plugins/sales | Research a company or person and get actionable sales intel. Works standalone with web search, supercharged when you connect enrichment tools or your CRM. Trigger with "research [company]", "look up [person]", "intel on [prospect]", "who is [name] at [company]", or "tell me about [company]". | research/intelligence |
| `ai-slop-cleaner` | oh-my-claudecode@omc | Clean AI-generated code slop with a regression-safe, deletion-first workflow and optional reviewer-only mode | refactor/cleanup |
| `algorithmic-art` | document-skills@anthropic-agent-skills | Creating algorithmic art using p5.js with seeded randomness and interactive parameter exploration. Use this when users request creating art using code, generative art, algorithmic art, flow fields, or particle systems. Create original algorithmic art rather than copying existing artists' work to avoid copyright violations. | creative/artifacts |
| `api-design-principles` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | API design (missing) |
| `architecture` | knowledge-work-plugins/engineering | Create or evaluate an architecture decision record (ADR). Use when choosing between technologies (e.g., Kafka vs SQS), documenting a design decision with trade-offs and consequences, reviewing a system design proposal, or designing a new component from requirements and constraints. | planning/architecture |
| `ask` | oh-my-claudecode@omc | Process-first advisor routing for Claude, Codex, Gemini, Antigravity, Grok, or Cursor via 'omc ask', with artifact capture and no raw CLI assembly | advisory/orchestration |
| `async-python-patterns` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | Python/backend (missing) |
| `audit-support` | knowledge-work-plugins/finance | Support SOX 404 compliance with control testing methodology, sample selection, and documentation standards. Use when generating testing workpapers, selecting audit samples, classifying control deficiencies, or preparing for internal or external audits. | finance/audit |
| `autopilot` | oh-my-claudecode@omc | Full autonomous execution from idea to working code | planning/orchestration |
| `autoresearch` | oh-my-claudecode@omc | Stateful single-mission improvement loop with strict evaluator contract, markdown decision logs, and max-runtime stop behavior | research/intelligence |
| `brand-guidelines` | document-skills@anthropic-agent-skills | Applies Anthropic's official brand colors and typography to any sort of artifact that may benefit from having Anthropic's look-and-feel. Use it when brand colors or style guidelines, visual formatting, or company design standards apply. | brand/design |
| `brand-review` | knowledge-work-plugins/marketing | Review content against your brand voice, style guide, and messaging pillars, flagging deviations by severity with specific before/after fixes. Use when checking a draft before it ships, when auditing copy for voice consistency and terminology, or when screening for unsubstantiated claims, missing disclaimers, and other legal flags. | brand/content review |
| `call-prep` | knowledge-work-plugins/sales | Prepare for a sales call with account context, attendee research, and suggested agenda. Works standalone with user input and web research, supercharged when you connect your CRM, email, chat, or transcripts. Trigger with "prep me for my call with [company]", "I'm meeting with [company] prep me", "call prep [company]", or "get me ready for [meeting]". | sales/research |
| `call-summary` | knowledge-work-plugins/sales | Process call notes or a transcript - extract action items, draft follow-up email, generate internal summary. Use when pasting rough notes or a transcript after a discovery, demo, or negotiation call, drafting a customer follow-up, logging the activity for your CRM, or capturing objections and next steps for your team. | sales/meeting notes |
| `campaign-plan` | knowledge-work-plugins/marketing | Generate a full campaign brief with objectives, audience, messaging, channel strategy, content calendar, and success metrics. Use when planning a product launch, lead-gen push, or awareness campaign, when you need a week-by-week content calendar with dependencies, or when translating a marketing goal into a structured, executable plan. | marketing/planning |
| `cancel` | oh-my-claudecode@omc | Cancel any active OMC mode (autopilot, ralph, ultrawork, ultraqa, swarm, ultrapilot, pipeline, team) | workflow control |
| `canvas-design` | document-skills@anthropic-agent-skills | Create beautiful visual art in .png and .pdf documents using design philosophy. You should use this skill when the user asks to create a poster, piece of art, design, or other static piece. Create original visual designs, never copying existing artists' work to avoid copyright violations. | creative/artifacts |
| `ccg` | oh-my-claudecode@omc | Claude-Codex-Gemini tri-model orchestration via /ask codex + /ask antigravity (or gemini), then Claude synthesizes results | advisory/orchestration |
| `ckm:banner-design` | ui-ux-pro-max@ui-ux-pro-max-skill | Design banners for social media, ads, website heroes, creative assets, and print. Multiple art direction options with AI-generated visuals. Actions: design, create, generate banner. Platforms: Facebook, Twitter/X, LinkedIn, YouTube, Instagram, Google Display, website hero, print. Styles: minimalist, gradient, bold typography, photo-based, illustrated, geometric, retro, glassmorphism, 3D, neon, duotone, editorial, collage. Uses ui-ux-pro-max, frontend-design, ai-artist, ai-multimodal skills. | UI/UX/visual design |
| `ckm:brand` | ui-ux-pro-max@ui-ux-pro-max-skill | Brand voice, visual identity, messaging frameworks, asset management, brand consistency. Activate for branded content, tone of voice, marketing assets, brand compliance, style guides. | brand/design |
| `ckm:design` | ui-ux-pro-max@ui-ux-pro-max-skill | Comprehensive design skill: brand identity, design tokens, UI styling, logo generation (55 styles, Gemini AI), corporate identity program (50 deliverables, CIP mockups), HTML presentations (Chart.js), banner design (22 styles, social/ads/web/print), icon design (15 styles, SVG, Gemini 3.1 Pro), social photos (HTML->screenshot, multi-platform). Actions: design logo, create CIP, generate mockups, build slides, design banner, generate icon, create social photos, social media images, brand identity, design system. Platforms: Facebook, Twitter, LinkedIn, YouTube, Instagram, Pinterest, TikTok, Threads, Google Ads. | UI/UX/visual design |
| `ckm:design-system` | ui-ux-pro-max@ui-ux-pro-max-skill | Token architecture, component specifications, and slide generation. Three-layer tokens (primitive->semantic->component), CSS variables, spacing/typography scales, component specs, strategic slide creation. Use for design tokens, systematic design, brand-compliant presentations. | design system |
| `ckm:slides` | ui-ux-pro-max@ui-ux-pro-max-skill | Create strategic HTML presentations with Chart.js, design tokens, responsive layouts, copywriting formulas, and contextual slide strategies. | presentation design |
| `ckm:ui-styling` | ui-ux-pro-max@ui-ux-pro-max-skill | Create beautiful, accessible user interfaces with shadcn/ui components (built on Radix UI + Tailwind), Tailwind CSS utility-first styling, and canvas-based visual designs. Use when building user interfaces, implementing design systems, creating responsive layouts, adding accessible components (dialogs, dropdowns, forms, tables), customizing themes and colors, implementing dark mode, generating visual designs and posters, or establishing consistent styling patterns across applications. | UI/UX/frontend styling |
| `claude-api` | document-skills@anthropic-agent-skills | /- | API/integration tooling |
| `claude-automation-recommender` | claude-code-setup@claude-plugins-official | Analyze a codebase and recommend Claude Code automations (hooks, subagents, skills, plugins, MCP servers). Use when user asks for automation recommendations, wants to optimize their Claude Code setup, mentions improving Claude Code workflows, asks how to first set up Claude Code for a project, or wants to know what Claude Code features they should use. | Claude Code setup |
| `claude-md-improver` | claude-md-management@claude-plugins-official | Audit and improve CLAUDE.md files in repositories. Use when user asks to check, audit, update, improve, or fix CLAUDE.md files. Scans for all CLAUDE.md files, evaluates quality against templates, outputs quality report, then makes targeted updates. Also use when the user mentions "CLAUDE.md maintenance" or "project memory optimization". | CLAUDE.md maintenance |
| `close-management` | knowledge-work-plugins/finance | Manage the month-end close process with task sequencing, dependencies, and status tracking. Use when planning the close calendar, tracking close progress, identifying blockers, or sequencing close activities by day. | finance/close |
| `code-review` | knowledge-work-plugins/engineering | Review code changes for security, performance, and correctness. Trigger with a PR URL or diff, "review this before I merge", "is this code safe?", or when checking a change for N+1 queries, injection risks, missing edge cases, or error handling gaps. | code review |
| `codebase-memory` | custom/user skill | Use the codebase knowledge graph for structural code queries. Triggers on: explore the codebase, understand the architecture, what functions exist, show me the structure, who calls this function, what does X call, trace the call chain, find callers of, show dependencies, impact analysis, dead code, unused functions, high fan-out, refactor candidates, code quality audit, graph query syntax, Cypher query examples, edge types, how to use search_graph. | codebase discovery |
| `codex-cli-runtime` | codex@openai-codex | Internal helper contract for calling the codex-companion runtime from Claude Code | Codex delegation |
| `codex-result-handling` | codex@openai-codex | Internal guidance for presenting Codex helper output back to the user | Codex delegation |
| `competitive-brief` | knowledge-work-plugins/product-management | Create a competitive analysis brief for one or more competitors or a feature area. Use when informing product strategy or feature prioritization, building sales battle cards, prepping board or investor materials, or deciding where to differentiate vs. achieve parity. | competitive research |
| `competitive-intelligence` | knowledge-work-plugins/sales | Research your competitors and build an interactive battlecard. Outputs an HTML artifact with clickable competitor cards and a comparison matrix. Trigger with "competitive intel", "research competitors", "how do we compare to [competitor]", "battlecard for [competitor]", or "what's new with [competitor]". | competitive research |
| `configure-notifications` | oh-my-claudecode@omc | Configure notification integrations (Telegram, Discord, Slack) via natural language | tooling/ops |
| `content-creation` | knowledge-work-plugins/marketing | Draft marketing content across channels - blog posts, social media, email newsletters, landing pages, press releases, and case studies. Use when writing any marketing content, when you need channel-specific formatting, SEO-optimized copy, headline options, or calls to action. | marketing/content |
| `create-an-asset` | knowledge-work-plugins/sales | Generate tailored sales assets (landing pages, decks, one-pagers, workflow demos) from your deal context. Describe your prospect, audience, and goal - get a polished, branded asset ready to share with customers. | sales/assets |
| `daily-briefing` | knowledge-work-plugins/sales | Start your day with a prioritized sales briefing. Works standalone when you tell me your meetings and priorities, supercharged when you connect your calendar, CRM, and email. Trigger with "morning briefing", "daily brief", "what's on my plate today", "prep my day", or "start my day". | sales/planning |
| `debug` | knowledge-work-plugins/engineering | Structured debugging session - reproduce, isolate, diagnose, and fix. Trigger with an error message or stack trace, "this works in staging but not prod", "something broke after the deploy", or when behavior diverges from expected and the cause isn't obvious. | debugging |
| `deep-dive` | oh-my-claudecode@omc | 2-stage pipeline: trace (causal investigation) -> deep-interview (requirements crystallization) with 3-point injection | codebase discovery |
| `deep-interview` | oh-my-claudecode@omc | Socratic deep interview with mathematical ambiguity gating before explicit execution approval | requirements clarification |
| `deepinit` | oh-my-claudecode@omc | Deep codebase initialization with hierarchical AGENTS.md documentation | repository documentation |
| `deploy-checklist` | knowledge-work-plugins/engineering | Pre-deployment verification checklist. Use when about to ship a release, deploying a change with database migrations or feature flags, verifying CI status and approvals before going to production, or documenting rollback triggers ahead of time. | deployment/release |
| `design-critique` | knowledge-work-plugins/design | Get structured design feedback on usability, hierarchy, and consistency. Trigger with "review this design", "critique this mockup", "what do you think of this screen?", or when sharing a Figma link or screenshot for feedback at any stage from exploration to final polish. | UI/UX/design review |
| `design-handoff` | knowledge-work-plugins/design | Generate developer handoff specs from a design. Use when a design is ready for engineering and needs a spec sheet covering layout, design tokens, component props, interaction states, responsive breakpoints, edge cases, and animation details. | UI/UX/handoff |
| `design-system` | knowledge-work-plugins/design | Audit, document, or extend your design system. Use when checking for naming inconsistencies or hardcoded values across components, writing documentation for a component's variants, states, and accessibility notes, or designing a new pattern that fits the existing system. | design system |
| `doc-coauthoring` | document-skills@anthropic-agent-skills | Guide users through a structured workflow for co-authoring documentation. Use when user wants to write documentation, proposals, technical specs, decision docs, or similar structured content. This workflow helps users efficiently transfer context, refine content through iteration, and verify the doc works for readers. Trigger when user mentions writing docs, creating proposals, drafting specs, or similar documentation tasks. | documentation/writing |
| `documentation` | knowledge-work-plugins/engineering | Write and maintain technical documentation. Trigger with "write docs for", "document this", "create a README", "write a runbook", "onboarding guide", or when the user needs help with any form of technical writing - API docs, architecture docs, or operational runbooks. | technical documentation |
| `docx` | document-skills@anthropic-agent-skills | Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx files). Triggers include: any mention of 'Word doc', 'word document', '.docx', or requests to produce professional documents with formatting like tables of contents, headings, page numbers, or letterheads. Also use when extracting or reorganizing content from .docx files, inserting or replacing images in documents, performing find-and-replace in Word files, working with tracked changes or comments, or converting content into a polished Word document. If the user asks for a 'report', 'memo', 'letter', 'template', or similar deliverable as a Word or .docx file, use this skill. Do NOT use for PDFs, spreadsheets, Google Docs, or general coding tasks unrelated to document generation. | document processing |
| `draft-content` | knowledge-work-plugins/marketing | Draft blog posts, social media, email newsletters, landing pages, press releases, and case studies with channel-specific formatting and SEO recommendations. Use when writing any marketing content, when you need headline or subject line options, or when adapting a message for a specific platform, audience, and brand voice. | marketing/content |
| `draft-outreach` | knowledge-work-plugins/sales | Research a prospect then draft personalized outreach. Uses web research by default, supercharged with enrichment and CRM. Trigger with "draft outreach to [person/company]", "write cold email to [prospect]", "reach out to [name]". | sales/outreach |
| `email-sequence` | knowledge-work-plugins/marketing | Design and draft multi-email sequences with full copy, timing, branching logic, exit conditions, and performance benchmarks. Use when building onboarding, lead nurture, re-engagement, win-back, or product launch flows, when you need a complete drip campaign with A/B test suggestions, or when mapping a sequence end-to-end with a flow diagram. | marketing/content |
| `external-context` | oh-my-claudecode@omc | Invoke parallel document-specialist agents for external web searches and documentation lookup | research/reference lookup |
| `fastapi-templates` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | FastAPI/backend (missing) |
| `financial-statements` | knowledge-work-plugins/finance | Generate financial statements (income statement, balance sheet, cash flow) with period-over-period comparison and variance analysis. Use when preparing a monthly or quarterly P&L, closing the books and need to flag material variances, comparing actuals to budget, building a financial summary for leadership review, or looking up GAAP presentation requirements and period-end adjustments. | finance/reporting |
| `forecast` | knowledge-work-plugins/sales | Generate a weighted sales forecast with best/likely/worst scenarios, commit vs. upside breakdown, and gap analysis. Use when preparing a quarterly forecast call, assessing gap-to-quota from a pipeline CSV, deciding which deals to commit vs. call upside, or checking pipeline coverage against your number. | sales/forecasting |
| `frontend-design` | frontend-design@claude-plugins-official | Create distinctive, production-grade frontend interfaces with high design quality. Use this skill when the user asks to build web components, pages, or applications. Generates creative, polished code that avoids generic AI aesthetics. | UI/UX/frontend implementation |
| `gpt-5-4-prompting` | codex@openai-codex | Internal guidance for composing Codex and GPT-5.4 prompts for coding, review, diagnosis, and research tasks inside the Codex Claude Code plugin | Codex prompting |
| `hud` | oh-my-claudecode@omc | Configure HUD display options (layout, presets, display elements) | tooling/ops |
| `incident-response` | knowledge-work-plugins/engineering | Run an incident response workflow - triage, communicate, and write postmortem. Trigger with "we have an incident", "production is down", an alert that needs severity assessment, a status update mid-incident, or when writing a blameless postmortem after resolution. | incident response |
| `internal-comms` | document-skills@anthropic-agent-skills | A set of resources to help me write all kinds of internal communications, using the formats that my company likes to use. Claude should use this skill whenever asked to write some sort of internal communications (status reports, leadership updates, 3P updates, company newsletters, FAQs, incident reports, project updates, etc.). | internal communications |
| `journal-entry` | knowledge-work-plugins/finance | Prepare journal entries with proper debits, credits, and supporting detail. Use when booking month-end accruals (AP, payroll, prepaid), recording depreciation or amortization, posting revenue recognition or deferred revenue adjustments, or documenting an entry for audit review. | finance/accounting |
| `journal-entry-prep` | knowledge-work-plugins/finance | Prepare journal entries with proper debits, credits, and supporting documentation for month-end close. Use when booking accruals, prepaid amortization, fixed asset depreciation, payroll entries, revenue recognition, or any manual journal entry. | finance/accounting |
| `langchain-architecture` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | AI framework architecture (missing) |
| `learner` | oh-my-claudecode@omc | Extract a learned skill from the current conversation | skill creation |
| `local-build-reminder` | oh-my-claudecode@omc | Remind the user to rebuild OMC after editing TypeScript when running from a local fork. Triggered automatically by the AI whenever it notices it (or the user) just changed a src/**/*.ts file in an OMC dev install. | tooling/ops |
| `marketing:competitive-brief` | custom/user skill | Research competitors and generate a positioning and messaging comparison with content gaps, opportunities, and threats. Use when building sales battlecards, when finding positioning gaps and messaging angles competitors haven't claimed, or when a competitor makes a move and you need to assess the impact. | competitive research |
| `mcp-builder` | document-skills@anthropic-agent-skills | Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services, whether in Python (FastMCP) or Node/TypeScript (MCP SDK). | API/integration tooling |
| `mcp-setup` | oh-my-claudecode@omc | Configure popular MCP servers for enhanced agent capabilities | tooling/ops |
| `mermaid-diagrams` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | diagrams/documentation (missing) |
| `metrics-review` | knowledge-work-plugins/product-management | Review and analyze product metrics with trend analysis and actionable insights. Use when running a weekly, monthly, or quarterly metrics review, investigating a sudden spike or drop, comparing performance against targets, or turning raw numbers into a scorecard with recommended actions. | product analytics |
| `omc-doctor` | oh-my-claudecode@omc | Diagnose and fix oh-my-claudecode installation issues | tooling/ops |
| `omc-plan` | oh-my-claudecode@omc | Strategic planning with optional interview workflow | planning/orchestration |
| `omc-reference` | oh-my-claudecode@omc | OMC agent catalog, available tools, team pipeline routing, commit protocol, and skills registry. Auto-loads when delegating to agents, using OMC tools, orchestrating teams, making commits, or invoking skills. | OMC reference |
| `omc-setup` | oh-my-claudecode@omc | Install or refresh oh-my-claudecode for plugin, npm, and local-dev setups from the canonical setup flow | tooling/ops |
| `omc-teams` | oh-my-claudecode@omc | CLI-team runtime for claude, codex, gemini, antigravity, grok, or cursor workers in tmux panes when you need process-based parallel execution | planning/orchestration |
| `pdf` | document-skills@anthropic-agent-skills | Use this skill whenever the user wants to do anything with PDF files. This includes reading or extracting text/tables from PDFs, combining or merging multiple PDFs into one, splitting PDFs apart, rotating pages, adding watermarks, creating new PDFs, filling PDF forms, encrypting/decrypting PDFs, extracting images, and OCR on scanned PDFs to make them searchable. If the user mentions a .pdf file or asks to produce one, use this skill. | document processing |
| `performance-report` | knowledge-work-plugins/marketing | Build a marketing performance report with key metrics, trend analysis, wins and misses, and prioritized optimization recommendations. Use when wrapping a campaign, when preparing weekly, monthly, or quarterly channel summaries for stakeholders, or when you need data translated into an executive summary with next-period priorities. | marketing/reporting |
| `pipeline-review` | knowledge-work-plugins/sales | Analyze pipeline health - prioritize deals, flag risks, get a weekly action plan. Use when running a weekly pipeline review, deciding which deals to focus on this week, spotting stale or stuck opportunities, auditing for hygiene issues like bad close dates, or identifying single-threaded deals. | sales/pipeline |
| `postgresql-table-design` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | database design (missing) |
| `pptx` | document-skills@anthropic-agent-skills | Use this skill any time a .pptx file is involved in any way - as input, output, or both. This includes: creating slide decks, pitch decks, or presentations; reading, parsing, or extracting text from any .pptx file (even if the extracted content will be used elsewhere, like in an email or summary); editing, modifying, or updating existing presentations; combining or splitting slide files; working with templates, layouts, speaker notes, or comments. Trigger whenever the user mentions \"deck,\" \"slides,\" \"presentation,\" or references a .pptx filename, regardless of what they plan to do with the content afterward. If a .pptx file needs to be opened, created, or touched, use this skill. | document processing |
| `product-brainstorming` | knowledge-work-plugins/product-management | Brainstorm product ideas, explore problem spaces, and challenge assumptions as a thinking partner. Use when exploring a new opportunity, generating solutions to a product problem, stress-testing an idea, or when a PM needs to think out loud with a sharp sparring partner before converging on a direction. | product strategy |
| `product-management:competitive-brief` | custom/user skill | Create a competitive analysis brief for one or more competitors or a feature area. Use when informing product strategy or feature prioritization, building sales battle cards, prepping board or investor materials, or deciding where to differentiate vs. achieve parity. | competitive research |
| `project-session-manager` | oh-my-claudecode@omc | Worktree-first dev environment manager for issues, PRs, and features with optional tmux sessions | tooling/ops |
| `ralph` | oh-my-claudecode@omc | Self-referential loop until task completion with configurable verification reviewer | persistent execution |
| `ralplan` | oh-my-claudecode@omc | Consensus planning entrypoint that auto-gates vague ralph/autopilot/team requests before execution | planning/orchestration |
| `reconciliation` | knowledge-work-plugins/finance | Reconcile accounts by comparing GL balances to subledgers, bank statements, or third-party data. Use when performing bank reconciliations, GL-to-subledger recs, intercompany reconciliations, or identifying and categorizing reconciling items. | finance/accounting |
| `release` | oh-my-claudecode@omc | Generic release assistant - analyzes repo release rules, caches them in .omc/RELEASE_RULE.md, then guides the release | deployment/release |
| `remember` | oh-my-claudecode@omc | Review reusable project knowledge and decide what belongs in project memory, notepad, or durable docs | memory/project knowledge |
| `research-synthesis` | knowledge-work-plugins/design | Synthesize user research into themes, insights, and recommendations. Use when you have interview transcripts, survey results, usability test notes, support tickets, or NPS responses that need to be distilled into patterns, user segments, and prioritized next steps. | research synthesis |
| `roadmap-update` | knowledge-work-plugins/product-management | Update, create, or reprioritize your product roadmap. Use when adding a new initiative and deciding what moves to make room, shifting priorities after new information comes in, moving timelines due to a dependency slip, or building a Now/Next/Later view from scratch. | product planning |
| `sciomc` | oh-my-claudecode@omc | Orchestrate parallel scientist agents for comprehensive analysis with AUTO mode | analysis/orchestration |
| `self-improve` | oh-my-claudecode@omc | Autonomous evolutionary code improvement engine with tournament selection | refactor/optimization |
| `seo-audit` | knowledge-work-plugins/marketing | Run a comprehensive SEO audit - keyword research, on-page analysis, content gaps, technical checks, and competitor comparison. Use when assessing a site's SEO health, when finding keyword opportunities and content gaps competitors own, or when you need a prioritized action plan split into quick wins and strategic investments. | marketing/SEO |
| `setup` | oh-my-claudecode@omc | Use first for install/update routing - sends setup, doctor, or MCP requests to the correct OMC setup flow | tooling/ops |
| `skill` | oh-my-claudecode@omc | Manage local skills - list, add, remove, search, edit, setup wizard | skill management |
| `skill-creator` | document-skills@anthropic-agent-skills | Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, edit, or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy. | skill creation |
| `skillify` | oh-my-claudecode@omc | Turn a repeatable workflow from the current session into a reusable OMC skill draft | skill creation |
| `slack-gif-creator` | document-skills@anthropic-agent-skills | Knowledge and utilities for creating animated GIFs optimized for Slack. Provides constraints, validation tools, and animation concepts. Use when users request animated GIFs for Slack like "make me a GIF of X doing Y for Slack. | creative/artifacts |
| `sox-testing` | knowledge-work-plugins/finance | Generate SOX sample selections, testing workpapers, and control assessments. Use when planning quarterly or annual SOX 404 testing, pulling a sample for a control (revenue, P2P, ITGC, close), building a testing workpaper template, or evaluating and classifying a control deficiency. | finance/audit |
| `sprint-planning` | knowledge-work-plugins/product-management | Plan a sprint - scope work, estimate capacity, set goals, and draft a sprint plan. Use when kicking off a new sprint, sizing a backlog against team availability (accounting for PTO and meetings), deciding what's P0 vs. stretch, or handling carryover from the last sprint. | product planning |
| `sql-optimization-patterns` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | database optimization (missing) |
| `stakeholder-update` | knowledge-work-plugins/product-management | Generate a stakeholder update tailored to audience and cadence. Use when writing a weekly or monthly status for leadership, announcing a launch, escalating a risk or blocker, or translating the same progress into exec-brief, engineering-detail, or customer-facing versions. | product communications |
| `standup` | knowledge-work-plugins/engineering | Generate a standup update from recent activity. Use when preparing for daily standup, summarizing yesterday's commits and PRs and ticket moves, formatting work into yesterday/today/blockers, or structuring a few rough notes into a shareable update. | engineering communications |
| `synthesize-research` | knowledge-work-plugins/product-management | Synthesize user research from interviews, surveys, and feedback into structured insights. Use when you have a pile of interview notes, survey responses, or support tickets to make sense of, need to extract themes and rank findings by frequency and impact, or want to turn raw feedback into roadmap recommendations. | research synthesis |
| `system-design` | knowledge-work-plugins/engineering | Design systems, services, and architectures. Trigger with "design a system for", "how should we architect", "system design for", "what's the right architecture for", or when the user needs help with API design, data modeling, or service boundaries. | planning/architecture |
| `team` | oh-my-claudecode@omc | N coordinated agents on shared task list using Claude Code implicit agent teams | planning/orchestration |
| `tech-debt` | knowledge-work-plugins/engineering | Identify, categorize, and prioritize technical debt. Trigger with "tech debt", "technical debt audit", "what should we refactor", "code health", or when the user asks about code quality, refactoring priorities, or maintenance backlog. | refactor/tech debt |
| `testing-strategy` | knowledge-work-plugins/engineering | Design test strategies and test plans. Trigger with "how should we test", "test strategy for", "write tests for", "test plan", "what tests do we need", or when the user needs help with testing approaches, coverage, or test architecture. | testing strategy |
| `theme-factory` | document-skills@anthropic-agent-skills | Toolkit for styling artifacts with a theme. These artifacts can be slides, docs, reportings, HTML landing pages, etc. There are 10 pre-set themes with colors/fonts that you can apply to any artifact that has been creating, or can generate a new theme on-the-fly. | UI/UX/theme |
| `trace` | oh-my-claudecode@omc | Evidence-driven tracing lane that orchestrates competing tracer hypotheses in Claude built-in team mode | codebase discovery |
| `ui-ux-pro-max` | ui-ux-pro-max@ui-ux-pro-max-skill | UI/UX design intelligence for web and mobile. Includes 50+ styles, 161 color palettes, 57 font pairings, 161 product types, 99 UX guidelines, and 25 chart types across 10 stacks (React, Next.js, Vue, Svelte, SwiftUI, React Native, Flutter, Tailwind, shadcn/ui, and HTML/CSS). Actions: plan, build, create, design, implement, review, fix, improve, optimize, enhance, refactor, and check UI/UX code. Projects: website, landing page, dashboard, admin panel, e-commerce, SaaS, portfolio, blog, and mobile app. Elements: button, modal, navbar, sidebar, card, table, form, and chart. Styles: glassmorphism, claymorphism, minimalism, brutalism, neumorphism, bento grid, dark mode, responsive, skeuomorphism, and flat design. Topics: color systems, accessibility, animation, layout, typography, font pairing, spacing, interaction states, shadow, and gradient. Integrations: shadcn/ui MCP for component search and examples. | UI/UX/frontend |
| `ultragoal` | oh-my-claudecode@omc | Durable multi-goal workflow that persists plan/ledger artifacts under .omc/ultragoal and prints Claude /goal handoff text for the active session | goal management |
| `ultraqa` | oh-my-claudecode@omc | QA cycling workflow - test, verify, fix, repeat until goal met | QA/verification |
| `ultrawork` | oh-my-claudecode@omc | Parallel execution engine for high-throughput task completion | execution/orchestration |
| `user-research` | knowledge-work-plugins/design | Plan, conduct, and synthesize user research. Trigger with "user research plan", "interview guide", "usability test", "survey design", "research questions", or when the user needs help with any aspect of understanding their users through research. | user research |
| `ux-copy` | knowledge-work-plugins/design | Write or review UX copy - microcopy, error messages, empty states, CTAs. Trigger with "write copy for", "what should this button say?", "review this error message", or when naming a CTA, wording a confirmation dialog, filling an empty state, or writing onboarding text. | UX writing |
| `variance-analysis` | knowledge-work-plugins/finance | Decompose financial variances into drivers with narrative explanations and waterfall analysis. Use when analyzing budget vs. actual, period-over-period changes, revenue or expense variances, or preparing variance commentary for leadership. | finance/reporting |
| `vercel-react-best-practices` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | React/Next.js (missing) |
| `verify` | oh-my-claudecode@omc | Verify that a change really works before you claim completion | verification |
| `visual-verdict` | oh-my-claudecode@omc | Structured visual QA verdict for screenshot-to-reference comparisons | visual QA |
| `web-artifacts-builder` | document-skills@anthropic-agent-skills | Suite of tools for creating elaborate, multi-component claude.ai HTML artifacts using modern frontend web technologies (React, Tailwind CSS, shadcn/ui). Use for complex artifacts requiring state management, routing, or shadcn/ui components - not for simple single-file HTML/JSX artifacts. | web artifacts |
| `web-design-guidelines` | pjs CLAUDE.md routing hint | Referenced by existing routing table; no installed skill found under this exact name. | UI/UX/frontend (missing) |
| `webapp-testing` | document-skills@anthropic-agent-skills | Toolkit for interacting with and testing local web applications using Playwright. Supports verifying frontend functionality, debugging UI behavior, capturing browser screenshots, and viewing browser logs. | web testing |
| `wiki` | oh-my-claudecode@omc | LLM Wiki - persistent markdown knowledge base that compounds across sessions (Karpathy model) | knowledge base |
| `write-spec` | knowledge-work-plugins/product-management | Write a feature spec or PRD from a problem statement or feature idea. Use when turning a vague idea or user request into a structured document, scoping a feature with goals and non-goals, defining success metrics and acceptance criteria, or breaking a big ask into a phased spec. | product/spec writing |
| `writer-memory` | oh-my-claudecode@omc | Agentic memory system for writers - track characters, relationships, scenes, and themes | writing/memory |
| `xlsx` | document-skills@anthropic-agent-skills | Use this skill any time a spreadsheet file is the primary input or output. This means any task where the user wants to: open, read, edit, or fix an existing .xlsx, .xlsm, .csv, or .tsv file (e.g., adding columns, computing formulas, formatting, charting, cleaning messy data); create a new spreadsheet from scratch or from other data sources; or convert between tabular file formats. Trigger especially when the user references a spreadsheet file by name or path - even casually (like \"the xlsx in my downloads\") - and wants something done to it or produced from it. Also trigger for cleaning or restructuring messy tabular data files (malformed rows, misplaced headers, junk data) into proper spreadsheets. The deliverable must be a spreadsheet file. Do NOT trigger when the primary deliverable is a Word document, HTML report, standalone Python script, database pipeline, or Google Sheets API integration, even if tabular data is involved. | document processing |
