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
from cli.create_project import create
from cli.compile_command import compile_cmd
from cli.diagnostics_command import diagnostics
from cli.drift_command import drift
from cli.explain_command import explain
from cli.format_command import format_cmd
from cli.graph_command import graph, graph_suite
from cli.ide_command import ide
from cli.lint_command import lint
from cli.migrate_plan_command import migrate_plan
from cli.model_command import model
from cli.nl_plan_command import nl_plan
from cli.package_command import package
from cli.parser_golden_command import parser_golden
from cli.release_command import release
from cli.run_command import run
from cli.validate_command import validate

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
cli.add_command(create)
cli.add_command(compile_cmd, name='compile')
cli.add_command(diagnostics)
cli.add_command(drift)
cli.add_command(explain)
cli.add_command(format_cmd)
cli.add_command(graph)
cli.add_command(graph_suite)
cli.add_command(ide)
cli.add_command(lint)
cli.add_command(migrate_plan)
cli.add_command(model)
cli.add_command(nl_plan)
cli.add_command(package)
cli.add_command(parser_golden)
cli.add_command(release)
cli.add_command(run)
cli.add_command(validate)


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
def doctor():
	"""Check APG installation and environment"""
	console.print("[bold blue]APG Environment Check[/bold blue]")
	console.print()
	
	# Check Python version
	python_version = sys.version_info
	if python_version >= (3, 10):
		console.print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
	else:
		console.print(f"❌ Python {python_version.major}.{python_version.minor} (requires 3.10+)")
	
	# Check required packages
	required_packages = [
		'antlr4',
		'click',
		'rich'
	]
	
	console.print("\n[bold]Required Packages:[/bold]")
	for package in required_packages:
		try:
			__import__(package.replace('-', '_'))
			console.print(f"✅ {package}")
		except ImportError:
			console.print(f"❌ {package} (not installed)")
	
	# Check APG components
	console.print("\n[bold]APG Components:[/bold]")
	
	components = [
		('Grammar File', apg_root / 'spec' / 'apg.g4'),
		('Compiler', apg_root / 'compiler' / 'compiler.py'),
		('Language Server', apg_root / 'language_server' / 'server.py'),
		('VS Code Extension', apg_root / 'vscode-extension' / 'package.json'),
		('Templates', apg_root / 'templates')
	]
	
	for name, path in components:
		if path.exists():
			console.print(f"✅ {name}")
		else:
			console.print(f"❌ {name} (missing: {path})")
	
	# Check ANTLR grammar compilation
	console.print("\n[bold]Grammar Compilation:[/bold]")
	grammar_file = apg_root / 'spec' / 'apg.g4'
	if grammar_file.exists():
		console.print("✅ Grammar file found")
		
		# Check if generated parser exists
		generated_dir = apg_root / 'spec'
		if all((generated_dir / name).exists() for name in ["apgLexer.py", "apgParser.py", "apgVisitor.py"]):
			console.print("✅ Generated parser found")
		else:
			console.print("⚠️  Generated parser not found (run 'apg compile --generate-parser')")
	else:
		console.print("❌ Grammar file not found")
	
	console.print("\n[green]APG environment check complete![/green]")


@cli.command()
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option('--port', '-p', default=2087, help='Language server port')
@click.option('--host', '-h', default='127.0.0.1', help='Language server host')
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
	selected_modes = sum(bool(value) for value in [check, code_actions, rename_symbol])
	if selected_modes > 1:
		raise click.ClickException("--check, --code-actions, and --rename cannot be combined")
	if action_id and not code_actions:
		raise click.ClickException("--apply-action requires --code-actions")
	if write_rename and not (rename_symbol or action_id):
		raise click.ClickException("--write is only valid with --rename or --code-actions --apply-action")

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

	console.print(f"[blue]Starting APG Language Server on {host}:{port}[/blue]")
	
	try:
		from language_server.server import start_language_server
		start_language_server(host, port)
	except ImportError:
		console.print("[red]Language server not available. Install with: pip install apg-language-server[/red]")
	except KeyboardInterrupt:
		console.print("\n[yellow]Language server stopped[/yellow]")


@cli.command()
def init():
	"""Initialize APG project in current directory"""
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
		'template': 'custom',
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
		basic_apg = f'''module {current_dir.name} version 1.0.0 {{
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
			f.write(basic_apg)
	
	# Create directories
	(current_dir / 'generated').mkdir(exist_ok=True)
	(current_dir / 'templates').mkdir(exist_ok=True)
	(current_dir / 'tests').mkdir(exist_ok=True)
	
	console.print("✅ APG project initialized")
	console.print(f"✅ Created: apg.json")
	console.print(f"✅ Created: main.apg")
	console.print(f"✅ Created: generated/ directory")
	
	console.print("\n[green]Next steps:[/green]")
	console.print("  1. Edit main.apg to define your application")
	console.print("  2. Run 'apg compile' to generate Python artifacts")
	console.print("  3. Run 'python generated/app.py' to start the generated application")
	console.print("  4. Run 'python generated/app.py --describe' to inspect JSON metadata")
	console.print("  5. Run 'python generated/app.py --self-test' to verify the generated application")


if __name__ == '__main__':
	cli()
