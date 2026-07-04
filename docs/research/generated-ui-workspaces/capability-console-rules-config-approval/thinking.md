# Raw Reasoning

The capability workspace should answer four operator questions without requiring DSL or JSON spelunking:

1. What context can I test right now?
2. Which policy rules matched and what actions did they request?
3. What configuration values are effective after overrides?
4. Who must approve this capability operation?

The before-state technically exposed the endpoints, but it treated the UI as an API form. That misses the main console job: turn capability contract metadata into a safe operational cockpit. The APG semantic model already contains configuration, rules, and approvals, so the strict fix is to reuse those generated facts instead of inventing hardcoded examples.

The best references converge on a traceable operation pattern. OPA puts decision logs and debugging around input plus decision detail. AWS policy simulation centers the user on a test context and outcome. LaunchDarkly separates defaults and evaluated values. Retool-style audit surfaces preserve operational event data without making it the primary workflow.

The generated UI should therefore keep raw JSON, but only after summarizing the domain result. A policy operator needs badges for decision and approvers, lists for matched rules and actions, and a key/value view for resolved configuration. Raw result JSON is still important because this project generates API/debug surfaces, but it belongs behind a `details` disclosure.

The regression test initially exposed a harness mismatch: `compile_apg_file()` executes only `app.py`, while the real generated app imports companion modules such as `apg_capabilities.py`. The test now compiles to a temporary output directory and imports the generated app from there, matching the boot path used by the live audit.

Rejected: adding a generic JSON editor dependency. It would exceed the self-contained asset goal and is unnecessary for the current console. Rejected: server-side sessions for preserving submitted JSON. The POST response can pass the submitted field text back into the same template with no stateful runtime.
