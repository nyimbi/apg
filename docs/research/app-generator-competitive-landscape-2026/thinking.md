# Thinking: Raw Reasoning Chain

Stream-of-consciousness notes during research, 2026-07-10.

---

## Initial Framing

The question is "what would it take to be decisively better" — not just competitive. That raises the bar. "Better" means winning on axes that matter to the customer, not winning on feature count.

Two very different markets are being compared here:
1. Code generators (JHipster, Amplication, Wasp, Refine) — emit ownable source artifacts
2. Runtime low-code platforms (Retool, Appsmith, Budibase, ToolJet, Superblocks) — apps live inside a proprietary runtime

APG is firmly category 1. The category-2 platforms are not really competitors in the traditional sense — they serve a different buyer (internal tooling ops teams vs. developers building production apps). But they ARE competing for budget and mind-share, so understanding why they're criticized matters.

---

## Hypothesis Going In

APG's advantages are probably:
- Africa connectors (unique)
- Capability breadth (wide)
- Full-stack (vs. backend-only or frontend-only)
- Open code output (vs. runtime lock-in)
- Offline-first (unique in market)

What I'm less sure about:
- Is APG's DSL a strength or a liability? (Wasp's experience suggests DSLs create tooling adoption friction)
- What do enterprise buyers actually demand that APG doesn't yet emit? (SBOM? OTel? IaC?)
- What's Amplication doing? The domain redirect is suspicious.

---

## Surprises Found During Research

**Surprise 1: Amplication domain redirecting to overcut.ai**
When trying to fetch amplication.com, it 301 redirected to overcut.ai. This is significant. Overcut.ai appears to be an unrelated AI service. Either: (a) Amplication was acquired, (b) they're pivoting their brand, or (c) there's a domain misconfiguration. The GitHub repo is still active. The docs subdomain (docs.amplication.com) was ENOTFOUND — confirms domain migration or shutdown of the original brand. If Amplication is being wound down or acquired, their user base (who need Node.js/NestJS backend generation) is addressable.

**Surprise 2: Wasp's self-inflicted blog post**
Wasp published "5 Years and $5M Later: Inventing a New Programming Language for Web Development Was a Mistake" in May 2026. This is remarkable candor. The lesson: a custom DSL/language creates adoption friction that is nearly impossible to overcome once it becomes perceived as "another thing to learn." APG needs to take this seriously — its APG language/DSL must either (a) be so minimal and YAML-like that it doesn't feel like a language, or (b) be completely hideable behind LLM-assisted authoring so developers never see the syntax.

**Surprise 3: Lovable CVE-2025-48757 scale**
170+ production apps with RLS disabled by default. 18,000 user records exposed including students from UC Berkeley and UC Davis. This happened because Lovable's AI generator optimized for "working demo" and disabled RLS for convenience. The generated app appeared functional but was fundamentally insecure. The market lesson: AI-generated code that is secure-by-default is a defensible moat. APG already emits security headers (from recent commits). This should be made explicit in marketing.

**Surprise 4: The "internal tools only" wall**
Every single runtime platform (Retool, Appsmith, Budibase, ToolJet, Superblocks) is explicitly designed for internal tools only. Their own documentation says they are "not designed for apps that anonymous clients use." This means the entire category of customer-facing apps — e-commerce, citizen portals, farmer apps, mobile banking — is unserved by the major low-code platforms. APG's customer-facing app generation is not just a feature, it's targeting a different market segment.

**Surprise 5: No competitor generates OTel instrumentation**
I searched specifically for this. Zero competitors generate OpenTelemetry SDK initialization or trace context propagation in their output. This is a clean gap. In 2026, enterprise SRE teams require OTel as a baseline — "any observability tool that doesn't natively support OpenTelemetry should be a red flag." Generating OTel-instrumented apps would be a differentiator that is currently unoccupied.

**Surprise 6: No competitor generates SBOM**
Despite SBOM generation being mandated by US EO 14028 and increasingly required for enterprise procurement, not a single app generator in the survey produces an SBOM. Generating a CycloneDX SBOM via `cyclonedx-bom` as part of the build pipeline is a one-tool addition, not a large engineering effort. The enterprise signal-to-noise ratio on this is very high.

---

## What I Got Wrong Initially

First instinct was to focus on feature parity with competitors. Wrong framing. The runtime platforms (Retool etc.) are not the right comparison class — they serve ops/IT teams building internal dashboards, APG serves developers building production applications. The comparison should be: APG vs. JHipster (for teams that would consider a full-stack generator) and APG vs. Lovable/Bolt (for teams that want AI-assisted generation). These are the actual decision contexts.

The other initial wrong instinct was thinking "Africa connectors" is a narrow niche. It's not — Africa's fintech market is projected at $230B by 2025 (McKinsey). The mobile money infrastructure (M-PESA, MTN MoMo) is more advanced in feature breadth than most Western payment rails. A generator that natively composes M-PESA STK Push, C2B webhooks, balance queries, and reconciliation reports is not serving a niche — it's serving the dominant payment rail of a $1T+ GDP region.

---

## Remaining Uncertainties

**On Wasp's lesson for APG:**
Is APG's DSL perceived as "another language to learn"? The Wasp analogy is strong — both are DSLs that compile to web apps. The difference is Wasp tried to replace the JavaScript ecosystem, while APG's DSL sits above Python/Flask (which developers still write directly). But developers who aren't familiar with APG will face the same adoption barrier. The mitigation is LLM-assisted authoring (describe in English, get APG spec). Whether this adequately addresses the friction is unknown.

**On regeneration safety:**
The `.amplicationignore` model (Amplication) is file-level — you mark whole files as "don't touch." This is simple but coarse. Section-level markers (my suggestion in the README) are more powerful but require the parser to understand the markers and preserve them. I don't know how APG currently handles this — whether it has any regeneration protection at all, or whether every regeneration overwrites custom code. This is a critical gap to investigate.

**On multi-tenancy:**
I stated "no competitor supports multi-tenancy from DSL declaration." That's correct for the platforms surveyed. But I don't know if APG supports it either. If APG has a multi-tenancy capability in its 335, that would be a significant differentiator to document explicitly. If it doesn't, adding it should be on the roadmap.

**On the MELA (2025 developer survey) stat:**
"86.3% of developers did not use any low-code platforms in 2024" — this cuts both ways. It means low-code adoption is still very low among professional developers, which is a market education problem for APG. But it also means the market is uncrowded from a developer tools perspective — most developers who would benefit from APG are not using any generator. This is an opportunity, not a threat.

**On JHipster's position:**
JHipster is Java/Spring Boot — a completely different language stack from APG (Python/Flask). The overlap in buyer is moderate — both serve backend developers who want generated scaffolding. But JHipster is not a threat to APG's core market because Java shops and Python shops have very different hiring pools and preferences. The risk of confusion with JHipster is minimal.

---

## Key Inference: The Convergence Point

The market is bifurcating into:
1. LLM-native generators (Lovable, Bolt) — fast, unsafe, good for demos
2. DSL/model-native generators (JHipster, Amplication, APG) — slower to configure, secure, production-grade

The winning position in 2026-2028 is: DSL/model-native + LLM-assisted authoring. Meaning: the output is deterministic and secure (from a typed model), but the authoring is LLM-driven (describe in English, get a typed model, compile to app). This is what Amplication's Jovu partially does for backends. APG doing this for full-stack apps would combine the speed of vibe-coding with the safety of model-driven generation.

This is the strategic insight: APG should not compete with Lovable on "generate faster." It should compete on "generate safer, more completely, and for markets Lovable can't touch (Africa, offline, regulated industries)."

---

## Search Strategy Notes

- 15 searches performed total
- 8 WebFetch calls (3 blocked by 403/DNS errors: JHipster main, amplication.com redirect, JHipster control center, G2 Retool, Amplication blog post)
- Most productive searches: platform-specific comparison blogs (ToolJet blog is surprisingly comprehensive for competitor data), HN discussion threads, security post-mortems
- Least productive: queries looking for "world class generator checklist" — had to synthesize this from multiple sources rather than find it directly
- Africa query produced the Datacraft link (own site) but validated no competitor covers this space
- Wasp's own blog was the most candid source — the "mistake" post is a primary source confession, not secondary analysis
