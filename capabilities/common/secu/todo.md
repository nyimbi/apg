# SECU Backlog

The active specification is `SPECIFICATION.md`; the active implementation plan
is `PLAN.md`.

Future integration packets:

- Bind `SecuService` to live SIEM, SOAR, EDR, MDM, IAM, GRC, DLP, notification,
  and ticketing adapters without bypassing package guardrails.
- Attach persistent storage for security policies, device posture, threat
  indicators, risk assessments, controls, policy exceptions, incidents,
  security agents, and audit events.
- Execute the declared Bytewax `secu.lifecycle` topology and verify ordering,
  watermark, replay, and failure behavior.
- Add rendered UI verification for the dashboard, risk, threat, policy,
  exception, incident, quarantine, compliance, agent, audit, rule, and settings
  screens.
- Add load and latency tests after the executable package surface is stable.
