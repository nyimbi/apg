#!/usr/bin/env python3
"""
APG CLI Main Entry Point
========================

Main command-line interface for APG (Application Programming Generation) language.
"""

import sys
import os
import json
from pathlib import Path

import click
from rich.console import Console

# Add APG modules to path
apg_root = Path(__file__).parent.parent
sys.path.insert(0, str(apg_root))

from cli.capabilities_command import capabilities
from cli.baseline_command import baseline
from cli.create_project import create
from cli.compile_command import compile_cmd
from cli.deployment_command import deployment
from cli.devex_command import export_cmd, serve, watch
from cli.diagnostics_command import diagnostics
from cli.doctor_command import doctor
from cli.docs_command import docs
from cli.drift_command import drift
from cli.evidence_command import evidence
from cli.explain_command import explain
from cli.format_command import format_cmd
from cli.schema_command import schema
from cli.graph_command import graph, graph_suite
from cli.hygiene_command import hygiene
from cli.ide_command import ide
from cli.lint_command import lint
from cli.migrate_plan_command import migrate_plan
from cli.model_command import model
from cli.nl_plan_command import nl_plan
from cli.package_command import package
from cli.package_verify_command import package_verify
from cli.parser_golden_command import parser_golden
from cli.release_command import release
from cli.run_command import run
from cli.studio_command import studio
from cli.tooling_command import tooling
from cli.validate_command import validate
from cli.refactor_command import refactor

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="APG")
def cli():
	"""
	APG (Application Programming Generation) Language Compiler
	
	APG is a domain-specific language for generating complete, functional web applications
	with agents, workflows, databases, and real-time interfaces.
	"""
	return None


# Add subcommands
cli.add_command(capabilities)
cli.add_command(baseline)
cli.add_command(create)
cli.add_command(compile_cmd, name='compile')
cli.add_command(deployment)
cli.add_command(export_cmd)
cli.add_command(diagnostics)
cli.add_command(doctor)
cli.add_command(docs)
cli.add_command(drift)
cli.add_command(evidence)
cli.add_command(explain)
cli.add_command(format_cmd)
cli.add_command(graph)
cli.add_command(graph_suite)
cli.add_command(hygiene)
cli.add_command(ide)
cli.add_command(lint)
cli.add_command(migrate_plan)
cli.add_command(model)
cli.add_command(nl_plan)
cli.add_command(package)
cli.add_command(package_verify)
cli.add_command(parser_golden)
cli.add_command(release)
cli.add_command(run)
cli.add_command(serve)
cli.add_command(studio)
cli.add_command(tooling)
cli.add_command(validate)
cli.add_command(schema)
cli.add_command(refactor)
cli.add_command(watch)


@cli.command()
def version():
	"""Show APG version information"""
	console.print("[bold blue]APG (Application Programming Generation)[/bold blue]")
	console.print("Version: 1.0.0")
	console.print("Language Specification: v11")
	console.print("Target Language: Python")
	console.print()
	console.print("Features:")
	console.print("  • Complete grammar with ANTLR 4.13+ support")
	console.print("  • Agent-based programming model")
	console.print("  • Workflow automation and orchestration")
	console.print("  • Database schema with DBML integration")
	console.print("  • Vector storage for AI/ML applications")
	console.print("  • Real-time web dashboards")
	console.print("  • Executable Python application artifacts")
	console.print("  • VS Code extension with Language Server")
	console.print("  • Comprehensive project templates")


@cli.command()
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option('--port', '-p', default=2087, help='Language server port')
@click.option('--host', '-h', default='127.0.0.1', help='Language server host')
@click.option("--audit-fixtures", is_flag=True, help="Audit checked-in language-server fixtures")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Language-server fixture catalog path")
@click.option("--check", is_flag=True, help="Run a dependency-light language-service check for one APG file")
@click.option("--code-actions", is_flag=True, help="Plan language-server code actions for one APG file")
@click.option("--apply-action", "action_id", help="Apply one code action id; requires --code-actions")
@click.option("--rename", "rename_symbol", help="Plan a semantic rename for the given symbol")
@click.option("--to", "rename_target", help="New symbol name for --rename")
@click.option("--kind", "rename_kind", help="Optional symbol kind discriminator for --rename")
@click.option("--write", "write_rename", is_flag=True, help="Apply a successful --rename plan to SOURCE_FILE")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.language-server-check.v1 JSON with --check")
def language_server(
	source_file: Path | None,
	port: int,
	host: str,
	audit_fixtures: bool,
	fixtures: Path | None,
	check: bool,
	code_actions: bool,
	action_id: str | None,
	rename_symbol: str | None,
	rename_target: str | None,
	rename_kind: str | None,
	write_rename: bool,
	as_json: bool,
):
	"""Start APG Language Server for IDE integration or check one APG file."""
	selected_modes = sum(bool(value) for value in [audit_fixtures, check, code_actions, rename_symbol])
	if selected_modes > 1:
		raise click.ClickException("--audit-fixtures, --check, --code-actions, and --rename cannot be combined")
	if action_id and not code_actions:
		raise click.ClickException("--apply-action requires --code-actions")
	if write_rename and not (rename_symbol or action_id):
		raise click.ClickException("--write is only valid with --rename or --code-actions --apply-action")

	if audit_fixtures:
		if source_file is not None:
			raise click.ClickException("--audit-fixtures cannot be combined with SOURCE_FILE")
		from language_server.semantic_service import audit_language_server_fixtures
		report = audit_language_server_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			summary = report["summary"]
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG language-server fixtures {status}: "
				f"{summary['passing_fixture_count']}/{summary['fixture_count']} passing, "
				f"{summary['blocking_gap_count']} blocking gaps"
			)
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return

	if check:
		if source_file is None:
			raise click.ClickException("--check requires SOURCE_FILE")
		if not source_file.exists():
			raise click.ClickException(f"APG source file not found: {source_file}")
		if not source_file.is_file():
			raise click.ClickException(f"APG language-server --check expects a file: {source_file}")
		from language_server.semantic_service import build_language_server_check
		report = build_language_server_check(source_file)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG language-server check {status}: {source_file}, "
				f"{report['diagnostic_count']} diagnostic(s), "
				f"{report['completion_count']} completion(s), "
				f"{report['document_symbol_count']} document symbol(s)"
			)
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return

	if code_actions:
		if source_file is None:
			raise click.ClickException("--code-actions requires SOURCE_FILE")
		if not source_file.exists():
			raise click.ClickException(f"APG source file not found: {source_file}")
		if not source_file.is_file():
			raise click.ClickException(f"APG language-server --code-actions expects a file: {source_file}")
		from language_server.semantic_service import build_language_server_code_actions
		report = build_language_server_code_actions(
			source_file,
			action_id=action_id,
			write=write_rename,
		)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG language-server code-actions {status}: {source_file}, "
				f"{report['action_count']} action(s)"
			)
			for action in report["actions"]:
				click.echo(f"  - {action['id']}: {action['title']}")
			if report.get("errors"):
				for error in report["errors"]:
					click.echo(f"  - {error}")
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return

	if rename_symbol:
		if source_file is None:
			raise click.ClickException("--rename requires SOURCE_FILE")
		if not rename_target:
			raise click.ClickException("--rename requires --to NEW_NAME")
		if not source_file.exists():
			raise click.ClickException(f"APG source file not found: {source_file}")
		if not source_file.is_file():
			raise click.ClickException(f"APG language-server --rename expects a file: {source_file}")
		from language_server.semantic_service import build_language_server_rename
		report = build_language_server_rename(
			source_file,
			rename_symbol,
			rename_target,
			kind=rename_kind,
			write=write_rename,
		)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			status = "OK" if report["ok"] else "FAILED"
			action = "written" if report.get("written") else "planned"
			click.echo(
				f"APG language-server rename {status}: {rename_symbol} -> {rename_target}, "
				f"{report['replacement_count']} replacement(s), {action}"
			)
			if report.get("errors"):
				for error in report["errors"]:
					click.echo(f"  - {error}")
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return

	if source_file is not None:
		raise click.ClickException("SOURCE_FILE is only accepted with --check, --code-actions, or --rename")

	_ = (host, port)
	server_script = apg_root / "language_server" / "server.py"
	os.execv(sys.executable, [sys.executable, str(server_script)])


def _module_name(app_name: str) -> str:
	normalized = "".join(
		char.lower() if char.isalnum() else "_"
		for char in app_name
	).strip("_")
	if not normalized:
		normalized = "apg_app"
	if normalized[0].isdigit():
		normalized = f"app_{normalized}"
	return normalized


def _app_class_name(app_name: str) -> str:
	parts = [part for part in _module_name(app_name).split("_") if part]
	return "".join(part.capitalize() for part in parts) or "APGApp"


def _sample_app_source(app_name: str) -> str:
	module_name = _module_name(app_name)
	class_name = _app_class_name(app_name)
	return f'''\
module {module_name} version 1.0.0 {{
    description: "{app_name} APG application";
}}

entity Customer {{
    name: str;
    email: str;
    is_active: bool = true;
    created_at: datetime;
}}

app {class_name} {{
    description: "{app_name} web application";
    routes: ["/", "/customers"];
}}
'''


def _env_example() -> str:
	return '''\
# APG runtime
APG_ENV=development
APG_ENVIRONMENT=development
APG_PRODUCTION=0
APG_SECRET_KEY=
APG_HOST=127.0.0.1
APG_PORT=5000
APG_DEBUG=1
APG_LOG_LEVEL=INFO
APG_JSON_LOGS=0
APG_BASE_URL=http://127.0.0.1:5000

# Generated application metadata
APG_APP_NAME=
APG_APP_DESCRIPTION=
APG_APP_VERSION=1.0.0
APG_MODULE_NAME=
APG_VERSION=1.0.0

# Persistence
APG_DATABASE_URL=sqlite:///apg.db
APG_SQLITE_PATH=apg.db
APG_AUTO_MIGRATE=1

# Authentication and authorization
APG_AUTH_REQUIRED=0
APG_AUTH_USERS=[]
APG_AUTH_USERNAME=
APG_AUTH_PASSWORD=
APG_AUTH_PASSWORD_HASH=
APG_AUTH_EMAIL=
APG_AUTH_DISPLAY_NAME=
APG_SESSION_SECRET=
APG_SESSION_COOKIE_NAME=apg_session
APG_SESSION_COOKIE_SECURE=0
APG_SESSION_COOKIE_SAMESITE=Lax
APG_JWT_SECRET=
APG_JWT_SECRET_KEY=
APG_API_KEY=
APG_API_KEY_USER_ID=
APG_API_KEY_TENANT_ID=
APG_API_KEY_PERMISSIONS=[]
APG_RBAC_ENABLED=0
APG_DEFAULT_ROLES=[]
APG_DEFAULT_PERMISSIONS=[]

# Tenancy
APG_TENANT_ID=default
APG_DEFAULT_TENANT_ID=default
APG_TENANT_HEADER=X-APG-Tenant
APG_TENANT_HEADER_DEFAULT=default
APG_MULTI_TENANT_ENABLED=0

# Email and notifications
APG_SMTP_HOST=
APG_SMTP_PORT=587
APG_SMTP_USER=
APG_SMTP_PASSWORD=
APG_SMTP_FROM=
APG_NOTIFY_EMAIL=
APG_ALERT_EMAIL=

# HTTP, security, and operations
APG_MAX_BODY_BYTES=1048576
APG_UPLOAD_DIR=uploads
APG_UPLOAD_MAX_BYTES=10485760
APG_UPLOAD_ALLOWED_TYPES=
APG_METRICS_ENABLED=1
APG_METRICS_TOKEN=
APG_SECURITY_HEADERS=1
APG_DISABLE_GZIP=0
APG_GZIP_MIN_BYTES=512
APG_PRIVATE_CACHE_CONTROL=no-store
APG_STATIC_CACHE_CONTROL=public, max-age=3600
APG_RECORDS_CACHE_CONTROL=no-store
APG_RATE_LIMIT_ANON=
APG_RATE_LIMIT_AUTH=
APG_RATE_EXEMPT_PATHS=
APG_OPS_QUIET_PATHS=

# External services
APG_AUTH_SERVICE_URL=
APG_AI_SERVICE_URL=
APG_NOTIFICATION_SERVICE_URL=
APG_EVENT_STREAMING_URL=
APG_RUNTIME_URL=
APG_API_BASE_URL=
APG_API_GATEWAY_URL=

# CLI developer experience
APG_WATCH_CLEAR=0
APG_VERBOSE=0
APG_RUN_PLAYWRIGHT=0

# Capability-specific APG_* variables can be added below as each capability is enabled.
'''


def _readme(app_name: str) -> str:
	return f'''\
# {app_name}

APG project scaffold generated by `apg init {app_name}`.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install apg-language flask pytest
cp .env.example .env
make compile
make run
```

## Commands

- `make compile` generates Python artifacts into `generated/`.
- `make run` compiles and runs the generated Flask app.
- `make test` runs the project tests.
- `make clean` removes generated and local cache files.
- `apg doctor` validates the local APG environment and configuration.
- `apg watch app.apg` recompiles on save.
- `apg export app.apg --format docker -o deploy` writes Docker deployment assets.

Edit `app.apg`, then run `apg compile app.apg -o generated`.
'''


def _makefile() -> str:
	return '''\
APG ?= apg
OUT ?= generated

.PHONY: run test compile clean

run: compile
	python $(OUT)/app.py

test:
	python -m pytest

compile:
	$(APG) compile app.apg -o $(OUT)

clean:
	rm -rf $(OUT) __pycache__ .pytest_cache .mypy_cache .ruff_cache
'''


def _gitignore() -> str:
	return '''\
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.DS_Store
generated/
out/
tmp/
apg.db
*.sqlite
*.sqlite3
'''


def _scaffold_named_project(app_name: str, template: str | None) -> None:
	project_dir = Path.cwd() / app_name
	files = {
		"app.apg": _INIT_TEMPLATES[template] if template else _sample_app_source(app_name),
		".env.example": _env_example(),
		"README.md": _readme(app_name),
		"Makefile": _makefile(),
		".gitignore": _gitignore(),
	}
	if project_dir.exists() and not project_dir.is_dir():
		raise click.ClickException(f"Target exists and is not a directory: {project_dir}")

	collisions = [
		str(project_dir / relative_path)
		for relative_path in files
		if (project_dir / relative_path).exists()
	]
	if collisions:
		raise click.ClickException(
			"Refusing to overwrite existing files: " + ", ".join(collisions)
		)

	project_dir.mkdir(parents=True, exist_ok=True)
	for relative_path, content in files.items():
		(project_dir / relative_path).write_text(content, encoding="utf-8")

	console.print(f"[green]Created APG project:[/green] {project_dir}")
	for relative_path in files:
		console.print(f"Created: {project_dir / relative_path}")
	console.print("\n[green]Next steps:[/green]")
	console.print(f"  1. cd {app_name}")
	console.print("  2. cp .env.example .env")
	console.print("  3. make compile")
	console.print("  4. make run")


_INIT_TEMPLATES: dict[str, str] = {
	"saas-app": '''\
module saas_app version 1.0.0 {
    description: "Multi-tenant SaaS application";
}

table Tenant {
    tenant_id: str;
    name: str;
    plan: str = "starter";
    is_active: bool = true;
    created_at: datetime;
}

table User {
    user_id: str;
    tenant_id: str;
    email: str;
    role: str = "member";
    is_active: bool = true;
}

capability TenantManagement {
    contract: {
        id: tenant_management,
        provides: [tenant_auth, user_management, billing],
        requires: [],
        configuration: {tenant_id: "default"},
        rules: [
            {name: "tenant_required", when: "tenant_id missing", action: deny}
        ]
    };
}

app SaaSApp {
    description: "Multi-tenant SaaS application";
    capabilities: [TenantManagement];
    routes: ["/", "/dashboard", "/settings"];
}
''',
	"erp-module": '''\
module erp_module version 1.0.0 {
    description: "ERP module — replace with your domain";
}

table Entity {
    entity_id: str;
    name: str;
    status: str = "active";
    created_at: datetime;
}

capability ERPModule {
    contract: {
        id: erp_module,
        provides: [entity_management],
        requires: [audit_events],
        configuration: {
            tenant_id: "default",
            currency: "USD"
        },
        rules: [
            {name: "active_only", when: "status != active", action: deny}
        ],
        ui: {shell: python, routes: [
            {name: "Entities", path: "/entities", component: "EntityList", permission: "module:view"}
        ]},
        theme: {name: module_theme, tokens: {accent: "#1565C0"}}
    };
    erp_modules: [finance];
    approvals: {levels: 1, approvers: [manager]};
    master_data: {entities: [entity]};
}

app ERPModuleApp {
    description: "ERP module application";
    capabilities: [ERPModule];
    routes: ["/module"];
}
''',
	"ai-agent": '''\
module ai_agent_app version 1.0.0 {
    description: "AI agent application";
}

agent PrimaryAgent {
    role: "primary assistant";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "You are a helpful assistant. Be concise and accurate.";
    tools: [];
    memory: vector agent_memory;
    configuration: {temperature: 0.1, max_turns: 8};
}

agent ReviewAgent {
    role: "quality reviewer";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Review responses for accuracy and completeness.";
    configuration: {temperature: 0.0, max_turns: 4};
}

agent_team MainCrew {
    agents: [PrimaryAgent, ReviewAgent];
    flow: PrimaryAgent -> ReviewAgent [condition: review_needed];
    configuration: {handoff_mode: conditional};
}

app AIAgentApp {
    description: "AI agent application";
    agent_teams: [MainCrew];
    routes: ["/chat", "/agents"];
}
''',
	"workflow-app": '''\
module workflow_app version 1.0.0 {
    description: "Approval workflow application";
}

table Request {
    request_id: str;
    title: str;
    description: str;
    amount: decimal;
    status: str = "draft";
    submitted_by: str;
    created_at: datetime;
}

capability RequestManagement {
    contract: {
        id: request_management,
        provides: [request_lifecycle, approval_queue],
        configuration: {tenant_id: "default", approval_threshold: 10000},
        rules: [
            {name: "large_request", when: "amount > approval_threshold", action: require_review}
        ]
    };
}

workflow ApprovalFlow {
    steps: str = "draft -> submitted -> reviewed -> approved";
    human_tasks: [reviewed, approved];
    assignments: {reviewed: reviewer, approved: approver};
    guards: {reviewed: "title not missing"};
    timers: {reviewed: "PT48H"};
}

app WorkflowApp {
    description: "Request approval workflow";
    capabilities: [RequestManagement];
    routes: ["/requests", "/approvals"];
}
''',
}


@cli.command()
@click.argument("app_name", required=False)
@click.option(
	"--template",
	type=click.Choice(list(_INIT_TEMPLATES.keys())),
	default=None,
	help="Project template to use (saas-app, erp-module, ai-agent, workflow-app)",
)
def init(app_name: str | None, template: str | None):
	"""Initialize an APG project."""
	if app_name:
		_scaffold_named_project(app_name, template)
		return

	current_dir = Path.cwd()

	# Check if already APG project
	if (current_dir / 'apg.json').exists():
		console.print("[yellow]Already an APG project[/yellow]")
		return

	# Create basic APG project structure
	console.print(f"[blue]Initializing APG project in {current_dir}[/blue]")

	# Basic project configuration
	project_config = {
		'name': current_dir.name,
		'version': '1.0.0',
		'description': f'APG project: {current_dir.name}',
		'author': 'APG Developer',
		'license': 'MIT',
		'template': template or 'custom',
		'target_language': 'python',
		'python_version': f'{sys.version_info.major}.{sys.version_info.minor}',
		'features': {
			'authentication': True,
			'api': True,
			'database': True,
			'testing': True
		},
		'build': {
			'source_file': 'main.apg',
			'output_directory': 'generated',
			'target_language': 'python',
			'include_runtime': True
		}
	}

	# Create apg.json
	with open(current_dir / 'apg.json', 'w') as f:
		import json
		json.dump(project_config, f, indent=2)

	# Create basic APG file if it doesn't exist
	if not (current_dir / 'main.apg').exists():
		if template and template in _INIT_TEMPLATES:
			apg_source = _INIT_TEMPLATES[template]
			console.print(f"[green]Using template: {template}[/green]")
		else:
			apg_source = f'''module {current_dir.name} version 1.0.0 {{
	description: "APG project: {current_dir.name}";
	author: "APG Developer";
	license: "MIT";
}}

agent BasicAgent {{
	name: str = "{current_dir.name} Agent";
	status: str = "inactive";
	counter: int = 0;

	initialize: () -> bool = {{
		status = "active";
		counter = 0;
		return true;
	}};

	process: () -> str = {{
		if (status == "active") {{
			counter = counter + 1;
			return "Processing request #" + str(counter);
		}}
		return "Agent is inactive";
	}};

	get_status: () -> dict = {{
		return {{
			"name": name,
			"status": status,
			"processed": counter,
			"timestamp": now()
		}};
	}};
}}'''

		with open(current_dir / 'main.apg', 'w') as f:
			f.write(apg_source)

	# Create directories
	(current_dir / 'generated').mkdir(exist_ok=True)
	(current_dir / 'templates').mkdir(exist_ok=True)
	(current_dir / 'tests').mkdir(exist_ok=True)

	console.print("APG project initialized")
	console.print(f"Created: apg.json")
	console.print(f"Created: main.apg")
	console.print(f"Created: generated/ directory")

	console.print("\n[green]Next steps:[/green]")
	console.print("  1. Edit main.apg to define your application")
	console.print("  2. Run 'apg compile' to generate Python artifacts")
	console.print("  3. Run 'python generated/app.py' to start the generated application")
	console.print("  4. Run 'python generated/app.py --describe' to inspect JSON metadata")
	console.print("  5. Run 'python generated/app.py --self-test' to verify the generated application")


if __name__ == '__main__':
	cli()
