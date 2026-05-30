# I18N Internationalization Specification Pointer

The active internationalization specification is maintained in
`SPECIFICATION.md`.

This compatibility file remains because older APG registry and compiler paths
still discover `cap_spec.md`. New design, implementation, test, and review work
must use:

- `README.md` for usage and composition guidance;
- `SPECIFICATION.md` for normative behavior;
- `PLAN.md` for the current implementation packet plan;
- `capability_contract.py` for executable configuration, rules, routes, theme,
  language policy, and Bytewax streaming metadata;
- `service.py` for the dependency-light lifecycle runtime;
- `test_capability_contract.py` for focused proof.
