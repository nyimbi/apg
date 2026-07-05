# Raw Reasoning

The auth workspace is narrower than most generated UI surfaces because the numbered examples do not declare authentication. I used a temporary auth-required generated app matching the existing regression source so the audit covered the real generated Flask routes and templates without changing example semantics.

The before surface passed the functional loop but violated the experience boundary: unauthenticated users saw the normal generated app shell. That makes the sidebar/topbar feel interactive before access is granted, creates duplicate landmark risk because the login template itself uses `main`, and distracts from the sign-in task. The strict remediation is to let `_html_page()` render a shell-less page for auth while retaining local assets, theme CSS, skip link, offline banner script support, and no external URLs.

The error message should stay generic. It should not disclose whether the username or password was wrong, but it can still be clear and helpful. Preserving the attempted username reduces rework without revealing account existence beyond what the user typed.

The implementation should not introduce a new auth framework or dependency. Existing generated credential/session behavior is sufficient for this workspace; the problem is the generated UI shell and form ergonomics.

Rejected ideas:

- Add password reset or MFA flows: valuable in a product auth system, but outside the generated-app contract and not backed by current compiler semantics.
- Use branded imagery or illustration: login is an operational generated app surface; the highest-value fix is clarity, not decoration.
- Hide every shared script from shell-less pages: unnecessary and riskier than preserving the existing local asset pipeline; the visible shell markup is the defect.
