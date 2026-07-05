# Raw Reasoning

This workspace combines a public-ish root landing page with an in-app connector marketplace. The root page should orient the user and move them into the generated app quickly; it should not be a decorative marketing screen disconnected from actual generated routes. The marketplace should work even when no external connectors are installed, because generated apps still have local integration surfaces: OpenAPI, record APIs, workflows, events, and agents.

The live audit revealed the highest-priority defect was functional, not visual: example 20's root and `/home` routes crashed. The crash came from assuming `MODULE_NAME` is a capability name. The strict fix is a guarded theme lookup with default tokens.

For the landing page, the highest-value repair is to expose real generated capabilities: primary CTA, first entity, readiness metrics, action cards, integration blueprints, and developer links. The page remains standalone and self-contained, but no longer relies on a gradient hero or generic API link strip.

For marketplace, the no-connector state should not stop at a CLI instruction. Generated connector blueprints provide a useful catalog baseline without adding dependencies or external URLs. Search and category filters make the surface behave like a marketplace even with local generated cards.

Rejected ideas:

- Adding live connector installation: outside current compiler semantics and would require new backend state.
- Fetching remote marketplace catalog data: violates the self-contained/no external URL constraint for generated apps.
- Adding large illustration assets: unnecessary for a utilitarian generated-app surface and increases generated output noise.
