# Raw Thinking

The existing capability console already exposes the three core operations: rules evaluation, configuration resolution, and approval planning. The gap is that users must manually create contexts, manually compare configuration outputs, and manually infer approval urgency.

OPA is the leader for policy testing and decision logs; LaunchDarkly shows how approvals and change history make risky changes governable; AWS AppConfig validates configuration before rollout; ServiceNow SLA patterns make approval timing operational. APG can combine these references because generated capabilities already include rules, configuration, and approval metadata.

Rejected ideas:

- Adding a full policy language playground. The generated console evaluates declared rules, so an input test bench is more honest.
- Persisting test suites server-side. Browser-local persistence is enough for this pass and avoids new storage policy.
- Live SLA timers from server time. A static generated countdown conveys the concept without runtime scheduling.
