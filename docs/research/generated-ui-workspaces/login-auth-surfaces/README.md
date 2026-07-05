# Login Auth Surfaces

## Verdict

Before: auth-required generated apps correctly redirected unauthenticated `/ui` requests, accepted valid credentials, rejected invalid credentials, and showed the user menu after login. The sign-in page itself was too application-shell-like: it rendered inside the normal generated app chrome with sidebar/topbar affordances visible to unauthenticated users, had sparse field labeling, did not preserve the attempted username on error, and used a terse credential failure message.

After: the generated login route renders in a standalone asset shell with the same local CSS/JS stack but without the app sidebar/topbar, labels the authentication context, preserves the username after a failed attempt, uses a generic recovery-oriented error, and shows the post-login destination. Authenticated UI and logout behavior remain unchanged.

## Live Surface Audit

- Before app: generated auth-required sample booted on `127.0.0.1:20905`.
- After app: regenerated auth-required sample booted on `127.0.0.1:20906`.
- Redirect: unauthenticated `/ui` returns `302` to `/login?next=/ui`.
- Invalid credentials: `/login` returns `401` with visible error feedback.
- Valid credentials: `/login` returns `302` to `/ui`; authenticated `/ui` renders `Ops User` and `Logout`.
- Logout: `/logout` returns `302` to `/login`.

## Must-Fix Items Completed

- Removed the visible app shell from the unauthenticated sign-in page.
- Preserved submitted username after failed login.
- Replaced the old specific failure copy with a generic, recovery-oriented message.
- Added explicit field ids/labels and a visible destination cue.
- Added regression coverage for the standalone login shell, improved copy, preserved username, and successful authenticated UI.

## Evidence

- `assets/before-login.html`
- `assets/before-login-error.html`
- `assets/before-authenticated-ui.html`
- `assets/after-login.html`
- `assets/after-login-error.html`
- `assets/after-authenticated-ui.html`
