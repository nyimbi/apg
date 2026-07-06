# Login/Auth Raw Reasoning

The current APG login page is deliberately small: username, password, redirect context. Round 2 should make it feel like a modern auth product without violating the ground rules. The main constraint is that generated apps cannot call hosted Auth0, Clerk, WorkOS, or Okta services, cannot load their SDKs, and cannot add backend dependencies.

Auth0 is the best leader because it spans passkeys and email magic links while also representing a mature hosted authentication product. Clerk is important for custom passkey UI thinking. WorkOS is important because Magic Auth is designed for B2B login boxes. Okta is important for passkey governance and recovery/lockout operations.

Real passkey authentication requires a server challenge and credential verification. Real magic links require a delivery channel and one-time token persistence. APG should not fake either. The right implementation is a generated readiness/control plane: detect browser support, stage intent locally, explain session posture, and reveal recovery steps after repeated failed sign-ins. This creates a much better generated login experience while keeping the actual auth boundary honest.

Rejected: adding WebAuthn libraries. That adds dependencies and still needs a server challenge/store. Rejected: adding SMTP/email magic links. That would require secrets and external infrastructure. Rejected: bypass-login magic URLs. That would weaken the generated auth model.
