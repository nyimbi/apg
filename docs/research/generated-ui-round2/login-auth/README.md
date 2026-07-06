# Login/Auth Round-2 Research

Date accessed: 2026-07-06

## Best-in-class reference

Commercial leader: Auth0. Auth0 is the strongest benchmark for a generated login surface because it covers hosted authentication, database-connection passkeys, and passwordless email magic links at broad production scale.

Adjacent references:

- Clerk for custom passkey UI flows and session-oriented developer ergonomics.
- WorkOS AuthKit for passwordless Magic Auth and polished B2B auth presentation.
- Okta for passkey governance and account recovery/lockout operating posture.

## Leader weaknesses

- Auth0 and Okta are powerful but configuration-heavy. Small generated applications can feel over-served by admin-console concepts before the operator understands the local auth posture.
- Clerk and WorkOS provide polished components, but generated offline apps cannot depend on their hosted runtime, SDKs, or external scripts under the APG ground rules.
- Most leader login boxes hide device/session context until after sign-in. APG can show the auth posture directly on the login screen.
- Lockout and recovery paths are often policy-driven and opaque to the end user. APG can expose a clear local recovery path without implying that an email/SMS provider exists.

## Differentiators proposed

1. Passkey Readiness Tile: detect browser passkey/WebAuthn support and explain whether the generated app can start a passkey enrollment handoff.
2. Magic-link Intent: stage a local magic-link request for the typed username and redirect target so operators can wire an email provider later without changing the generated UI.
3. Device Session Review: show the current browser/device session context on the login screen and let users locally dismiss stale device rows.
4. Lockout Recovery Flow: track failed sign-in attempts locally, reveal a recovery panel after repeated failures, and offer a safe local reset of the attempt counter.

## Prioritized implementation

Ship all four in `login.html.j2` with a generated `auth_intelligence` object. Keep the actual credential check unchanged and make every enhanced control explicit about its generated/offline nature.
