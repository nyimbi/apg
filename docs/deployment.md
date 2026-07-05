# APG Deployment Guide

APG deployment has two different meanings:

1. Deploying the APG developer toolchain so teams can compile `.apg` source.
2. Deploying a generated Flask app created by `apg compile`, `apg package`, or
   `apg release`.

This guide focuses on the current generated-app path.

## Local Generated App

Compile:

```bash
apg compile app.apg --output generated/app --verify
```

Run:

```bash
python generated/app/app.py --host 127.0.0.1 --port 8080
```

Verify:

```bash
python generated/app/app.py --self-test
python generated/app/smoke_test.py
```

## Generated Files

Deployment-relevant generated files:

```text
app.py
__init__.py
requirements.txt
semantic_model.json
smoke_test.py
Dockerfile
.dockerignore
.env.example
static/
```

`static/` contains vendored generated UI assets. Do not add CDN requirements to
the generated deployment path.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `APG_HOST` or `HOST` | Server host default. |
| `APG_PORT` or `PORT` | Server port default. |
| `APG_DEBUG=1` | Enable generated Flask debug mode. |
| `APG_SESSION_SECRET` or `APG_JWT_SECRET` | Session secret for generated auth surfaces. |
| `APG_API_KEY` | Require API key for mutations. |
| `APG_DATA_FILE` | Persist records to a JSON file. |
| `APG_DATABASE_URL`, `APG_PG_URL`, or `DATABASE_URL` | Optional best-effort PostgreSQL persistence. |
| `APG_LANDING_STYLE` | Override generated landing style. |

## Container Deployment

Generated apps include a Dockerfile. Build and run:

```bash
cd generated/app
docker build -t apg-generated-app .
docker run --rm -p 8080:8080 --env-file .env.example apg-generated-app
```

Before promoting a container image, run:

```bash
python app.py --self-test
python smoke_test.py
```

## Package Profiles

Use package commands when you need generated evidence around delivery profiles:

```bash
apg package app.apg --target web --out /tmp/apg-web --json
apg package app.apg --target desktop --out /tmp/apg-desktop --json
apg package app.apg --target mobile --out /tmp/apg-mobile --json
apg package app.apg --target container --out /tmp/apg-container --json
apg package-verify /tmp/apg-web --json
apg deployment verify /tmp/apg-web --json
```

`web`, `desktop`, `mobile`, and `container` are packaging profiles over the
Python generated artifact. They are not compiler targets.

## Production Checklist

- Compile with `--verify`.
- Run `app.py --self-test`.
- Run `smoke_test.py`.
- Review `openapi.json`, `component.json`, and `semantic-model.json`.
- Set a real session secret if auth is enabled.
- Set `APG_API_KEY` or stronger gateway controls when exposing mutations.
- Decide whether local JSON persistence is enough or whether a database URL is
  required.
- Serve generated static assets from the generated app or copy them as a unit;
  do not split them from the app without preserving paths.
- Put TLS, rate limiting, and external identity in front of the app at the
  deployment edge when required by the environment.

## Baseline And Release Evidence

For compiler-output changes:

```bash
apg baseline examples --refresh
uv run pytest tests/ -q
```

For one app:

```bash
apg release app.apg --json
apg evidence app.apg --target web --out /tmp/apg-evidence --json
```
