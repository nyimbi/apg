# CKM Workflow Automation Specification Pointer

The active workflow automation specification is maintained in
`SPECIFICATION.md`.

This compatibility file remains because older APG registry and compiler paths
still discover `cap_spec.md`. New design, implementation, tests, and review
work must use:

- `README.md` for practical usage and composition guidance;
- `SPECIFICATION.md` for the normative capability behavior;
- `PLAN.md` for the current implementation packet plan;
- `capability_contract.py` for executable configuration, rules, routes, theme,
  and Bytewax streaming metadata;
- `lifecycle.py` for the dependency-light lifecycle service;
- `test_capability_contract.py` for focused proof.
