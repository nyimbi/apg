# Rationale: Scope, Inclusion, and Exclusion Decisions

---

## Why These Competitors

### Included: Code Generators (JHipster, Amplication, Wasp, Refine.dev)

These four were chosen because they are the closest architectural peers to APG — tools that emit ownable source code from a higher-level specification. They compete for the same developer mindshare even if their stacks differ.

**JHipster**: The longest-running and most mature full-stack generator. Java/Spring Boot stack is different from APG's Python/Flask, but the architecture philosophy (DSL/JDL → full app) is identical. JHipster's upgrade pain and merge conflict problem is the single most instructive failure mode for APG to avoid — it is what happens when roundtrip safety is not designed in from the start.

**Amplication**: The closest match to APG's ambition: model-driven, plugin-extensible, generates production code to your repo. The `.amplicationignore` custom code model is the most sophisticated in the market. The domain redirect anomaly (amplication.com → overcut.ai) is a live signal worth monitoring. Their backend-only scope (no UI) is the key gap APG exploits.

**Wasp**: Included because it is a DSL-to-full-stack-app compiler — exactly what APG is. Wasp's explicit admission that "inventing a new programming language was a mistake" is the most relevant cautionary data point for APG's own DSL design. Their stack (React + Node.js + Prisma) overlaps with modern web development defaults in a way APG (Flask) does not, making Wasp a useful proxy for "what happens when a DSL targets the dominant stack and still fails on adoption."

**Refine.dev**: Included because it occupies the "frontend-only code generator" niche and demonstrates what AI-assisted React scaffolding from existing APIs looks like. Its limitation (no backend, no database) is instructive — it shows the market gap for full-stack generation with Python backends.

### Included: Runtime Platforms (Retool, Appsmith, Budibase, ToolJet, Superblocks)

These five were included not because they are architectural peers, but because they compete for budget and are what non-technical buyers compare APG against. Understanding their criticisms is essential for positioning APG to buyers who say "we're evaluating Retool vs. Appsmith vs. building custom."

The coverage of their limitations (no code export, internal tools only, per-user pricing, SSO gating) provides direct material for APG positioning documents and sales narratives.

**Retool**: The category-defining "leader" (by brand recognition and pricing discussion). Its criticisms define the genre's ceiling.

**Appsmith**: The open-source reference implementation of the category. Its governance-gating model (RBAC/SSO behind paid tiers) is a repeated pain point across multiple review sources.

**Budibase**: Included for the "no JSON in REST API response" limitation — a specific, reproducible technical complaint that signals the platform's architectural rigidity. Also the clearest example of "internal tools only" positioning.

**ToolJet**: The current AGPL leader in the category. Most complete free-tier feature set in 2026. Its AI generation (natural language → working draft) is the most advanced in the runtime platform category.

**Superblocks**: Included because of the $23M raise (June 2025) specifically for "secure AI app generation for the enterprise." This is a direct bet on the thesis that enterprise buyers want governed AI-generated apps. Superblocks' "Clark" AI agent generating React/TypeScript with full RBAC/SSO/audit logs is the best-executed enterprise story in the runtime category.

---

## What Was Excluded and Why

### Excluded: OutSystems, Mendix, Appian, ServiceNow Creator

These are enterprise no-code/low-code platforms in the $100K+/year license bracket. They were mentioned in HN discussions but are not in the same buyer context as APG. Their buyers are Fortune 500 IT departments with dedicated platform teams and budget cycles measured in quarters. APG's go-to-market is developer-led and Africa-focused. Including these would dilute the analysis without adding actionable intelligence.

### Excluded: Bubble, Adalo, Webflow, Framer

No-code visual builders targeting non-developers. The research briefly touches Bubble because its lock-in and scaling ceiling ($50K-$150K migration cost, performance degradation at 5,000-8,000 users) is frequently cited in developer criticism of the category. But Bubble is not a competitive threat to APG — its buyers and APG's buyers have essentially no overlap.

### Excluded: Supabase, Firebase, Nhost

Backend-as-a-service platforms. They are frequently the backend layer for Lovable/Bolt-generated apps. Not generators in themselves, though they generate some auth and database client code. Including them would require a separate research track on BaaS vs. generated backend — a worthwhile future spike but out of scope here.

### Excluded: GitHub Copilot, Cursor, Claude Code, Windsurf

AI coding assistants are not app generators. They assist human-driven development rather than generating from a specification. The distinction matters: an app generator produces a predictable, repeatable artifact from a typed spec; an AI coding assistant produces variable output from a prompt. The former is what APG is. Including AI assistants would conflate "AI-assisted development" (what every developer already uses) with "AI-native generation from spec" (what APG does).

### Excluded: AWS Amplify, Azure Static Web Apps generators

Cloud-vendor-specific generators. Excluded because they create cloud vendor lock-in that is distinct from platform lock-in — the buyer must use AWS or Azure as infrastructure. APG generates cloud-agnostic output. The comparison would require infrastructure cost modeling that is out of scope.

### Excluded: Laravel (PHP), Rails, Django generators/scaffolding

Framework-level scaffolding tools. `rails generate`, `django-admin startproject`, `artisan make:model` are in-framework generators, not app generators. They produce CRUD stubs that require significant manual work to become production apps. The distinction is: app generators claim to produce production-ready output; framework scaffolders explicitly produce development starting points. APG is an app generator, not a scaffolder.

---

## Methodology Choices

### Why competitor blogs were used as primary sources for feature matrices

Official documentation frequently omits limitations and feature gating. Competitor comparison blogs (particularly ToolJet's, Superblocks', and UI Bakery's) are written with the explicit intent of surfacing differentiators — including the competitor's weaknesses. While these sources are biased toward the author's product, the specific claims about competitors are verifiable and tend to be accurate (false claims invite legal challenges). Cross-referencing three independent comparison blogs against G2 reviews and HN discussions provided sufficient triangulation.

### Why G2 reviews were weighted less heavily

G2 reviews are buyer-segment skewed toward enterprise IT purchasers, not developers. APG's target audience is developers and technical founders. G2 data was used to confirm pricing complaints (Retool per-user pricing) and SSO gating patterns — facts that are consistent across sources — rather than for subjective assessments.

### Why the Hacker News "RIP Low-Code 2014-2025" thread was prioritized

HN threads on platform criticism are primary source developer opinions, unmediated by marketing. The thread (item 46767440) surfaced specific named platforms (Retool abandoned, n8n pivoting, Mendix/OutSystems complexity) and specific pain points (no code export, proprietary ecosystems, per-run cost models) that do not appear in vendor or analyst writing.

### Why Wasp's own blog was treated as a primary source

Wasp's "5 Years and $5M Later: Inventing a New Programming Language for Web Development Was a Mistake" is founder-authored self-criticism, which is a higher-quality signal than analyst criticism because founders have privileged information about adoption data and user interviews. The specific admission — "whatever we tried, it would always come back to 'why a custom language?'" — is directly applicable to APG's DSL design decisions.

### On the Amplication domain redirect

Observed during research: `amplication.com` 301-redirects to `overcut.ai`. The `docs.amplication.com` subdomain returns ENOTFOUND. The GitHub repository at `github.com/amplication/amplication` is still active and has recent commits. This is ambiguous — possible interpretations: acquisition, brand pivot, domain misconfiguration. Noted as an open question rather than a conclusion. If Amplication is being wound down, this is an addressable market event.

---

## Confidence Levels by Finding

**High confidence** (multiple independent sources, direct observation):
- Runtime platforms are internal-tools-only — stated by vendors themselves
- Retool SSO is enterprise-only — consistent across G2, pricing page, and comparison blogs
- Lovable CVE-2025-48757 — multiple security publications, The Register coverage
- JHipster upgrade pain / merge conflicts — community forum, official docs, practitioner blog
- Wasp custom language adoption problem — Wasp's own blog post (primary source)
- No competitor generates OTel instrumentation or SBOM — verified by absence across all sources

**Medium confidence** (single or secondary sources, plausible but not triple-verified):
- Amplication domain redirect to overcut.ai — direct observation, but interpretation uncertain
- Market size figures ($4.2B AI code generation in 2025, $230B African fintech) — single analyst sources
- "86.3% of developers did not use low-code in 2024" — single survey, source cited as "MELA survey" in secondary coverage

**Lower confidence** (inferred or extrapolated):
- That APG's section-level regeneration markers would be technically superior to `.amplicationignore` — logical argument, not tested
- That a JHipster user base is addressable by APG — different stacks, uncertain conversion rate
- Timeline estimate of "18-36 months" for African competitor emergence — educated guess, no primary data
