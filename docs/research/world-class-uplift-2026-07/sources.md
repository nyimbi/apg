# Sources — World-Class Uplift Mission

This mission folder synthesizes three dedicated research folders (each with its own complete
`sources.md` — canonical citation lists live there):

| Folder | Topic | Sources |
|--------|-------|---------|
| `../generated-app-runtime-baseline/sources.md` | OWASP ASVS 5.0, password storage, session mgmt, login hardening, k8s probes, framework defaults | 45 URLs |
| `../app-generator-competitive-landscape-2026/sources.md` | Amplication, JHipster, Refine, Wasp, Retool, Appsmith, Budibase, ToolJet, Superblocks; SBOM/OTel standards | 60+ URLs |
| `../generated-ui-excellence-2026/sources.md` | WCAG 2.2, WebAIM Million 2026, WAI-ARIA APG, Core Web Vitals, resilience patterns | 30+ URLs |

## Repo-internal evidence (accessed 2026-07-10)

- `compiler/code_generator.py` @ commit 5f53f346 — line-level evidence cited in findings-state-of-play.md
- `docs/research/composition-systems-gap-analysis.md` (2026-06-15)
- `docs/research/enterprise-ui-gap-analysis.md` (2026-06-15)
- `docs/research/ui_integration_gaps_2026.md`, `ui_integration_gaps_2026_v1.md` (2026-06-13)
- `docs/research/generated-ui-workspaces/SUMMARY.md`, `docs/research/generated-ui-round2/` (2026-07-06)
- `tests/test_generated_ui_*.py`, `tests/test_generated_workflow_runtime.py` — coverage map

## Headline external citations (full lists in the folders above)

- OWASP ASVS 5.0: https://github.com/OWASP/ASVS/blob/master/5.0/en/0x15-V6-Authentication.md
- OWASP Password Storage Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
- OWASP Session Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
- NIST SP 800-63B Rev 4: https://pages.nist.gov/800-63-4/sp800-63b.html
- Python hashlib (scrypt/pbkdf2_hmac): https://docs.python.org/3/library/hashlib.html
- Kubernetes probes: https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/
- WebAIM Million 2026: https://webaim.org/projects/million/
- WCAG 2.2 new criteria: https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/
- WAI-ARIA Authoring Practices: https://www.w3.org/WAI/ARIA/apg/patterns/
- Wasp DSL post-mortem: https://wasp.sh/blog/2026/05/13/new-language-for-web-dev-was-a-mistake
- Lovable RLS exposure: https://www.theregister.com/2026/02/27/lovable_app_vulnerabilities/
- OpenSSF SBOM tooling: https://openssf.org/blog/2025/06/05/choosing-an-sbom-generation-tool/
