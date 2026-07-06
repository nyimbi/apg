# Login/Auth Rationale

## Decisions

- Named Auth0 as the leader because it has first-party documentation for passkeys and email magic links, the two advanced authentication methods requested for this workspace.
- Implemented generated auth intelligence instead of hosted-auth integration. APG generated apps must remain offline, dependency-light, and self-contained.
- Kept username/password as the only credential verifier. Passkey and magic-link controls are honest readiness/intent controls, not fake bypasses.
- Implemented local lockout recovery affordances in the browser. This gives users a visible path after repeated failures without changing server security semantics.
- Added session-device review on the login page so auth posture is visible before sign-in.

## Rejected alternatives

- Real WebAuthn registration/sign-in: rejected because it needs challenge generation, credential storage, and verification APIs beyond this UI pass.
- Real email magic links: rejected because it needs token persistence and a mail provider.
- Hosted Auth0/Clerk/WorkOS widgets: rejected because generated output must not depend on CDN/runtime URLs or new dependencies.
- Account unlock API: rejected because the current generated auth model has no account state store to unlock.

## Budget note

The implementation uses vanilla JavaScript embedded in the login template and CSS in `apg.css`. It adds no generated Python dependencies and no external runtime URLs.
