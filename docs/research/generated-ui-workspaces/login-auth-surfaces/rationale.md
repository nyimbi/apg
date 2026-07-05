# Rationale

## Decisions

- Use a shell-less `_html_page(..., shell=False)` path for login. This keeps the generated asset pipeline self-contained while removing app navigation from unauthenticated screens.
- Keep auth behavior unchanged for valid credentials, session issue, redirect, and logout. The workspace problem was presentation and error handling, not the credential contract.
- Use generic failed-login copy: `We could not sign you in with those credentials.` This avoids user enumeration detail while being clearer than a terse invalid state.
- Preserve the username field on failed login. This reduces correction effort and does not disclose anything the user did not submit.
- Show the `next` destination in the login card. Users can see where the authentication flow will continue without exposing unrelated app chrome.

## Rejected Alternatives

- Implementing reset-password, email verification, or MFA scaffolding was rejected because the APG security block does not define those flows yet.
- Moving login to a separate generated app or blueprint was rejected as needless structural churn for a template-only surface defect.
- Removing all shared local scripts from shell-less pages was rejected because the budget and self-contained asset constraints are already satisfied; visible shell navigation was the concrete issue.

## Validation Plan

- Targeted auth/template/CSS tests before full suite.
- Regenerate all 20 numbered examples after the generator/template change.
- Boot a fresh auth-required generated app and capture before/after HTTP artifacts.
- Run full `uv run pytest tests/ -q`.
- Confirm PythonCodeGenerator source tripwire stays clean.
