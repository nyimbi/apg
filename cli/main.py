#!/usr/bin/env python3
"""
APG CLI Main Entry Point
========================

Main command-line interface for APG (Application Programming Generation) language.
"""

import sys
import os
from pathlib import Path

import click
from rich.console import Console

# Add APG modules to path
apg_root = Path(__file__).parent.parent
sys.path.insert(0, str(apg_root))

from cli.create_project import create
from cli.compile_command import compile_cmd
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
	pass


# Add subcommands
cli.add_command(create)
cli.add_command(compile_cmd, name='compile')
cli.add_command(run)
cli.add_command(validate)


@cli.command()
def version():
	"""Show APG version information"""
	console.print("[bold blue]APG (Application Programming Generation)[/bold blue]")
	console.print("Version: 1.0.0")
	console.print("Language Specification: v11")
	console.print("Target Framework: Flask-AppBuilder")
	console.print()
	console.print("Features:")
	console.print("  • Complete grammar with ANTLR 4.13+ support")
	console.print("  • Agent-based programming model")
	console.print("  • Workflow automation and orchestration")
	console.print("  • Database schema with DBML integration")
	console.print("  • Vector storage for AI/ML applications")
	console.print("  • Real-time web dashboards")
	console.print("  • Professional Flask-AppBuilder output")
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
		'antlr4-python3-runtime',
		'flask',
		'flask-appbuilder',
		'sqlalchemy',
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
		generated_dir = apg_root / 'compiler' / 'generated'
		if generated_dir.exists() and any(generated_dir.glob('APG*.py')):
			console.print("✅ Generated parser found")
		else:
			console.print("⚠️  Generated parser not found (run 'apg compile --generate-parser')")
	else:
		console.print("❌ Grammar file not found")
	
	console.print("\n[green]APG environment check complete![/green]")


@cli.command()
@click.option('--port', '-p', default=2087, help='Language server port')
@click.option('--host', '-h', default='127.0.0.1', help='Language server host')
def language_server(port: int, host: str):
	"""Start APG Language Server for IDE integration"""
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
		'target_framework': 'flask-appbuilder',
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
	console.print("  2. Run 'apg compile' to generate Flask-AppBuilder application")
	console.print("  3. Run 'apg run' to start the application")


if __name__ == '__main__':
	cli()