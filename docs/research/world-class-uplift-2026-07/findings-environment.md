# Environment Repair Log — 2026-07-10

Two pre-existing breakages in the local Python environment blocked full-suite collection.
Both repaired; recorded here because they will bite again if the setup drifts.

## Topology

- `python` on PATH = `/Users/nyimbiodero/.hermes/hermes-agent/venv/bin/python` (hermes-agent venv,
  Python 3.11), created so that its `sys.path` **also includes** `/opt/homebrew/lib/python3.11/site-packages`.
- The venv has **no pip of its own**; `python -m pip` loads homebrew's pip, which targets homebrew
  site-packages for installs but **refuses to uninstall** from there ("outside environment").
  To operate on homebrew: `/opt/homebrew/bin/python3.11 -m pip ... --break-system-packages`.
  To operate on the venv: `python -m pip --python <venv-python> install ...` (`--python` must
  precede the subcommand).
- Venv site-packages precede homebrew on `sys.path` → venv packages shadow homebrew ones.

## Breakage 1: stale editable flask-appbuilder

- Symptom: `ModuleNotFoundError: No module named 'flask_appbuilder'` in 3 test modules, despite
  `pip` reporting `flask-appbuilder 4.8.0.dev1` installed.
- Cause: editable (`__editable__*.pth`) install in homebrew site-packages mapping
  `flask_appbuilder → /Users/nyimbiodero/src/pjs/fab-ext/flask_appbuilder`, a path that no longer
  exists — the fab-ext fork was restructured into `pgappforge_*` packages and no longer tracks a
  `flask_appbuilder/` directory.
- Fix: uninstalled the editable via homebrew python, installed PyPI `flask-appbuilder==5.2.2`
  (satisfies `requirements.txt` `flask-appbuilder>=4.3.0`). Pulled in `Flask-Babel 4.0.0`.
- Residual risk: FAB 4.x → 5.x API changes in capability code — validated by full suite.

## Breakage 2: pyOpenSSL ↔ cryptography version skew

- Symptom: `TypeError: deprecated() got an unexpected keyword argument 'name'` importing
  `OpenSSL` (2 test modules: whatsapp connector, fin_glr period reporting).
- Cause: cryptography **3.4.8** (2021, known CVEs) present in BOTH homebrew and the venv, while
  homebrew's pyOpenSSL 25.3.0 expected modern cryptography. The pair must move in lockstep:
  old cryptography breaks new pyOpenSSL (`deprecated(name=)`), new cryptography breaks old
  pyOpenSSL (`GEN_EMAIL` removed).
- Fix: upgraded to `cryptography 49.0.0` in both homebrew and venv, then `pyOpenSSL 26.3.0`
  in homebrew. `from OpenSSL import SSL, crypto` now imports cleanly.
- Note: pip emitted non-fatal resolver-conflict warnings (hermes-agent pins vs new versions);
  accepted deliberately — see rationale.md.
