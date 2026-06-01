# Graph Data Management Capability Packet

The GRPH capability is defined by the packet documents and executable contract:

- `README.md` explains how to use the capability.
- `SPECIFICATION.md` defines the required functionality and guardrails.
- `PLAN.md` records the implementation and verification plan.
- `capability_contract.py` is the executable contract consumed by package
  generation, tests, and composition tooling, including first-class graph-agent
  and Bytewax lifecycle batch manifests.
- Review-required graph records are durable pending-review objects with
  matched-rule and review-reason evidence; deny decisions remain hard
  guardrails.

Keep this file as a short compatibility pointer. Update the packet documents
and contract when GRPH behavior changes.
