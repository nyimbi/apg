#!/usr/bin/env python3
"""
APG Project Creation CLI
========================

Command-line interface for creating new APG projects from templates.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add APG modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from templates.template_types import TemplateType, ProjectConfig, list_available_templates, get_recommended_templates
from templates.template_manager import TemplateManager
from templates.project_scaffolder import ProjectScaffolder

console = Console()


@click.group()
def create():
	"""Create new APG projects from templates"""
	pass


@create.command()
@click.option('--name', '-n', help='Project name')
@click.option('--description', '-d', help='Project description')
@click.option('--template', '-t', help='Template type')
@click.option('--author', '-a', help='Project author')
@click.option('--output', '-o', help='Output directory')
@click.option('--interactive/--no-interactive', default=True, help='Interactive mode')
@click.option('--overwrite', is_flag=True, help='Overwrite existing project')
def project(name: Optional[str], description: Optional[str], template: Optional[str], 
           author: Optional[str], output: Optional[str], interactive: bool, overwrite: bool):
	"""Create a new APG project"""
	
	console.print(Panel("[bold blue]APG Project Creator[/bold blue]", 
					   subtitle="Generate complete APG applications from templates"))
	
	try:
		# Interactive mode
		if interactive:
			config = _interactive_project_creation(name, description, template, author, output, overwrite)
		else:
			# Non-interactive mode - require all parameters
			if not all([name, description, template]):
				console.print("[red]Error: In non-interactive mode, --name, --description, and --template are required[/red]")
				sys.exit(1)
			
			config = _create_config_from_params(name, description, template, author, output, overwrite)
		
		if not config:
			console.print("[yellow]Project creation cancelled[/yellow]")
			return
		
		# Create project
		_create_project_with_progress(config)
		
	except KeyboardInterrupt:
		console.print("\n[yellow]Project creation cancelled by user[/yellow]")
		sys.exit(1)
	except Exception as e:
		console.print(f"[red]Error creating project: {e}[/red]")
		sys.exit(1)


def _interactive_project_creation(name: Optional[str], description: Optional[str], 
								template: Optional[str], author: Optional[str], 
								output: Optional[str], overwrite: bool) -> Optional[ProjectConfig]:
	"""Interactive project creation workflow"""
	
	# Project name
	if not name:
		name = Prompt.ask("[cyan]Project name[/cyan]", default="my-apg-project")
	
	# Project description
	if not description:
		description = Prompt.ask("[cyan]Project description[/cyan]", 
							   default=f"APG application: {name}")
	
	# Show available templates
	_show_available_templates()
	
	# Template selection
	if not template:
		template_type = _select_template_interactive()
	else:
		try:
			template_type = TemplateType(template)
		except ValueError:
			console.print(f"[red]Invalid template: {template}[/red]")
			template_type = _select_template_interactive()
	
	# Author
	if not author:
		author = Prompt.ask("[cyan]Author[/cyan]", default="APG Developer")
	
	# Output directory
	if not output:
		default_output = str(Path.cwd() / name)
		output = Prompt.ask("[cyan]Output directory[/cyan]", default=default_output)
	
	# Advanced configuration
	advanced = Confirm.ask("[cyan]Configure advanced options?[/cyan]", default=False)
	
	config = ProjectConfig(
		name=name,
		description=description,
		author=author,
		template_type=template_type,
		output_directory=Path(output),
		overwrite_existing=overwrite
	)
	
	if advanced:
		config = _configure_advanced_options(config)
	
	# Show configuration summary
	_show_config_summary(config)
	
	if not Confirm.ask("[cyan]Create project with this configuration?[/cyan]", default=True):
		return None
	
	return config


def _create_config_from_params(name: str, description: str, template: str, 
							 author: Optional[str], output: Optional[str], 
							 overwrite: bool) -> ProjectConfig:
	"""Create config from command line parameters"""
	
	try:
		template_type = TemplateType(template)
	except ValueError:
		console.print(f"[red]Invalid template: {template}[/red]")
		console.print("Available templates:")
		for t in TemplateType:
			console.print(f"  - {t.value}")
		sys.exit(1)
	
	return ProjectConfig(
		name=name,
		description=description,
		author=author or "APG Developer",
		template_type=template_type,
		output_directory=Path(output) if output else Path.cwd() / name,
		overwrite_existing=overwrite
	)


def _show_available_templates():
	"""Show available templates in a table"""
	
	console.print("\n[bold]Available Templates:[/bold]")
	
	table = Table(show_header=True, header_style="bold magenta")
	table.add_column("Template", style="cyan")
	table.add_column("Name", style="green")
	table.add_column("Complexity", style="yellow")
	table.add_column("Description", style="white")
	
	templates = list_available_templates()
	for template in templates:
		table.add_row(
			template['type'],
			template.get('name', ''),
			template.get('complexity', ''),
			template.get('description', '')[:60] + "..." if len(template.get('description', '')) > 60 else template.get('description', '')
		)
	
	console.print(table)
	console.print()


def _select_template_interactive() -> TemplateType:
	"""Interactive template selection"""
	
	# Show categories
	console.print("[bold]Template Categories:[/bold]")
	console.print("1. [green]Basic[/green] - Simple templates for learning")
	console.print("2. [yellow]Intermediate[/yellow] - Feature-rich applications")
	console.print("3. [red]Advanced[/red] - Complex enterprise solutions")
	console.print("4. [blue]Industry-Specific[/blue] - Specialized domain templates")
	console.print()
	
	category = Prompt.ask("[cyan]Select category[/cyan]", choices=['1', '2', '3', '4'], default='1')
	
	# Filter templates by complexity
	complexity_map = {
		'1': 'Beginner',
		'2': 'Intermediate', 
		'3': 'Advanced',
		'4': 'Expert'
	}
	
	if category == '4':
		# Industry-specific templates
		industry_templates = [
			TemplateType.FINTECH_PLATFORM,
			TemplateType.HEALTHCARE_SYSTEM,
			TemplateType.LOGISTICS_TRACKER,
			TemplateType.SOCIAL_NETWORK
		]
		
		console.print("[bold]Industry-Specific Templates:[/bold]")
		for i, template_type in enumerate(industry_templates, 1):
			template_info = _get_template_display_info(template_type)
			console.print(f"{i}. [cyan]{template_info['name']}[/cyan] - {template_info['description']}")
		
		choice = Prompt.ask("[cyan]Select template[/cyan]", 
						  choices=[str(i) for i in range(1, len(industry_templates) + 1)])
		return industry_templates[int(choice) - 1]
	
	else:
		# Filter by complexity
		target_complexity = complexity_map[category]
		filtered_templates = get_recommended_templates(complexity=target_complexity)
		
		if not filtered_templates:
			console.print(f"[red]No templates found for complexity: {target_complexity}[/red]")
			return TemplateType.BASIC_AGENT
		
		console.print(f"[bold]{target_complexity} Templates:[/bold]")
		for i, template_type in enumerate(filtered_templates, 1):
			template_info = _get_template_display_info(template_type)
			console.print(f"{i}. [cyan]{template_info['name']}[/cyan] - {template_info['description']}")
		
		choice = Prompt.ask("[cyan]Select template[/cyan]", 
						  choices=[str(i) for i in range(1, len(filtered_templates) + 1)])
		return filtered_templates[int(choice) - 1]


def _get_template_display_info(template_type: TemplateType) -> Dict[str, str]:
	"""Get display information for template"""
	from templates.template_types import get_template_info
	
	info = get_template_info(template_type)
	return {
		'name': info.get('name', template_type.value.replace('_', ' ').title()),
		'description': info.get('description', 'Template description')
	}


def _configure_advanced_options(config: ProjectConfig) -> ProjectConfig:
	"""Configure advanced project options"""
	
	console.print("\n[bold]Advanced Configuration:[/bold]")
	
	# Target framework
	framework = Prompt.ask("[cyan]Target framework[/cyan]", 
						 default=config.target_framework,
						 choices=['flask-appbuilder', 'django', 'fastapi'])
	config.target_framework = framework
	
	# Database type
	database = Prompt.ask("[cyan]Database type[/cyan]",
						default=config.database_type,
						choices=['sqlite', 'postgresql', 'mysql'])
	config.database_type = database
	
	# Python version
	python_version = Prompt.ask("[cyan]Python version[/cyan]",
							  default=config.python_version,
							  choices=['3.10', '3.11', '3.12'])
	config.python_version = python_version
	
	# Feature flags
	console.print("\n[bold]Features:[/bold]")
	
	config.enable_authentication = Confirm.ask("[cyan]Enable authentication?[/cyan]", 
											 default=config.enable_authentication)
	
	config.enable_api = Confirm.ask("[cyan]Enable REST API?[/cyan]", 
								  default=config.enable_api)
	
	config.enable_database = Confirm.ask("[cyan]Enable database?[/cyan]", 
										default=config.enable_database)
	
	config.enable_testing = Confirm.ask("[cyan]Enable testing framework?[/cyan]", 
									  default=config.enable_testing)
	
	config.enable_docker = Confirm.ask("[cyan]Enable Docker support?[/cyan]", 
									 default=config.enable_docker)
	
	config.enable_ai_features = Confirm.ask("[cyan]Enable AI features?[/cyan]", 
										  default=config.enable_ai_features)
	
	config.enable_real_time = Confirm.ask("[cyan]Enable real-time features?[/cyan]", 
										default=config.enable_real_time)
	
	config.use_async = Confirm.ask("[cyan]Use async/await?[/cyan]", 
								 default=config.use_async)
	
	config.include_examples = Confirm.ask("[cyan]Include examples?[/cyan]", 
										default=config.include_examples)
	
	config.generate_docs = Confirm.ask("[cyan]Generate documentation?[/cyan]", 
									 default=config.generate_docs)
	
	return config


def _show_config_summary(config: ProjectConfig):
	"""Show project configuration summary"""
	
	console.print("\n[bold]Project Configuration Summary:[/bold]")
	
	table = Table(show_header=False, box=None)
	table.add_column("Property", style="cyan")
	table.add_column("Value", style="white")
	
	table.add_row("Name", config.name)
	table.add_row("Description", config.description)
	table.add_row("Author", config.author)
	table.add_row("Template", config.template_type.value)
	table.add_row("Framework", config.target_framework)
	table.add_row("Database", config.database_type)
	table.add_row("Python", config.python_version)
	table.add_row("Output", str(config.output_directory))
	
	console.print(table)
	
	# Show enabled features
	features = []
	if config.enable_authentication: features.append("Authentication")
	if config.enable_api: features.append("REST API")
	if config.enable_database: features.append("Database")
	if config.enable_testing: features.append("Testing")
	if config.enable_docker: features.append("Docker")
	if config.enable_ai_features: features.append("AI Features")
	if config.enable_real_time: features.append("Real-time")
	if config.use_async: features.append("Async")
	
	if features:
		console.print(f"\n[bold]Enabled Features:[/bold] {', '.join(features)}")
	
	console.print()


def _create_project_with_progress(config: ProjectConfig):
	"""Create project with progress indication"""
	
	scaffolder = ProjectScaffolder()
	
	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		console=console
	) as progress:
		
		task = progress.add_task("Creating project...", total=None)
		
		# Create project
		result = scaffolder.create_project(config)
		
		if result['success']:
			progress.update(task, description="✅ Project created successfully!")
			
			console.print(f"\n[green]✅ Project '{config.name}' created successfully![/green]")
			console.print(f"[cyan]Location:[/cyan] {result['project_path']}")
			console.print(f"[cyan]Template:[/cyan] {result['template_used']}")
			console.print(f"[cyan]Files generated:[/cyan] {len(result['generated_files'])}")
			
			# Show next steps
			_show_next_steps(config, result['project_path'])
			
		else:
			progress.update(task, description="❌ Project creation failed")
			
			console.print(f"\n[red]❌ Project creation failed![/red]")
			for error in result['errors']:
				console.print(f"[red]  - {error}[/red]")


def _show_next_steps(config: ProjectConfig, project_path: Path):
	"""Show next steps after project creation"""
	
	console.print("\n[bold]Next Steps:[/bold]")
	
	steps = [
		f"cd {project_path}",
		"pip install -r requirements.txt",
		"apg compile main.apg",
		"python app.py"
	]
	
	for i, step in enumerate(steps, 1):
		console.print(f"{i}. [cyan]{step}[/cyan]")
	
	console.print(f"\n[green]Then open:[/green] http://localhost:8080")
	console.print(f"[green]Login with:[/green] admin/admin (default Flask-AppBuilder credentials)")
	
	if config.template_type != TemplateType.BASIC_AGENT:
		console.print(f"\n[yellow]📚 Check the README.md for template-specific usage instructions[/yellow]")


@create.command()
def list_templates():
	"""List all available templates"""
	
	console.print(Panel("[bold blue]Available APG Templates[/bold blue]"))
	
	templates = list_available_templates()
	
	# Group by complexity
	by_complexity = {}
	for template in templates:
		complexity = template.get('complexity', 'Unknown')
		if complexity not in by_complexity:
			by_complexity[complexity] = []
		by_complexity[complexity].append(template)
	
	for complexity in ['Beginner', 'Intermediate', 'Advanced', 'Expert']:
		if complexity in by_complexity:
			console.print(f"\n[bold]{complexity} Templates:[/bold]")
			
			table = Table(show_header=True, header_style="bold magenta")
			table.add_column("ID", style="cyan", width=20)
			table.add_column("Name", style="green", width=25)
			table.add_column("Description", style="white")
			
			for template in by_complexity[complexity]:
				table.add_row(
					template['type'],
					template.get('name', ''),
					template.get('description', '')
				)
			
			console.print(table)


@create.command()
@click.argument('template_type')
def info(template_type: str):
	"""Show detailed information about a template"""
	
	try:
		template_enum = TemplateType(template_type)
	except ValueError:
		console.print(f"[red]Invalid template: {template_type}[/red]")
		console.print("Use 'apg create list-templates' to see available templates")
		return
	
	from templates.template_types import get_template_info
	
	info = get_template_info(template_enum)
	
	if not info:
		console.print(f"[red]No information available for template: {template_type}[/red]")
		return
	
	console.print(Panel(f"[bold blue]{info.get('name', template_type)}[/bold blue]"))
	
	console.print(f"[cyan]Description:[/cyan] {info.get('description', 'N/A')}")
	console.print(f"[cyan]Complexity:[/cyan] {info.get('complexity', 'N/A')}")
	
	features = info.get('features', [])
	if features:
		console.print(f"\n[cyan]Features:[/cyan]")
		for feature in features:
			console.print(f"  • {feature}")
	
	use_cases = info.get('use_cases', [])
	if use_cases:
		console.print(f"\n[cyan]Use Cases:[/cyan]")
		for use_case in use_cases:
			console.print(f"  • {use_case}")
	
	console.print(f"\n[green]Create with:[/green] apg create project --template {template_type}")


if __name__ == '__main__':
	create()