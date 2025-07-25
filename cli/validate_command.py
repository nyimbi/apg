#!/usr/bin/env python3
"""
APG Validate Command
====================

Command-line interface for validating APG source files.
"""

import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add APG modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.parser import APGParser
from compiler.semantic_analyzer import SemanticAnalyzer

console = Console()


@click.command()
@click.argument('source_file', required=False)
@click.option('--syntax-only', is_flag=True, help='Only check syntax, skip semantic analysis')
@click.option('--semantic-only', is_flag=True, help='Only check semantics, skip syntax')
@click.option('--format', '-f', default='table', 
			 type=click.Choice(['table', 'json', 'plain']),
			 help='Output format')
@click.option('--warnings', '-w', is_flag=True, help='Show warnings')
@click.option('--strict', is_flag=True, help='Strict validation mode')
@click.option('--recursive', '-r', is_flag=True, help='Validate all APG files in directory')
def validate(source_file: Optional[str], syntax_only: bool, semantic_only: bool,
			format: str, warnings: bool, strict: bool, recursive: bool):
	"""Validate APG source files for syntax and semantic errors"""
	
	# Determine files to validate
	files_to_validate = []
	
	if recursive:
		# Find all APG files in current directory and subdirectories
		current_dir = Path('.')
		files_to_validate = list(current_dir.rglob('*.apg'))
		
		if not files_to_validate:
			console.print("[yellow]No APG files found in current directory[/yellow]")
			return
			
	else:
		# Single file validation
		if not source_file:
			# Look for main.apg or check apg.json
			candidates = ['main.apg', 'src/main.apg', 'app.apg']
			
			# Check apg.json for source file
			if Path('apg.json').exists():
				with open('apg.json', 'r') as f:
					config = json.load(f)
				source_file = config.get('build', {}).get('source_file', 'main.apg')
			
			# Find first existing candidate
			if not source_file or not Path(source_file).exists():
				for candidate in candidates:
					if Path(candidate).exists():
						source_file = candidate
						break
			
			if not source_file or not Path(source_file).exists():
				console.print("[red]No APG source file found. Specify file or create main.apg[/red]")
				return
		
		source_path = Path(source_file)
		if not source_path.exists():
			console.print(f"[red]Source file not found: {source_file}[/red]")
			return
		
		files_to_validate = [source_path]
	
	# Validate files
	if len(files_to_validate) == 1:
		_validate_single_file(files_to_validate[0], syntax_only, semantic_only, 
							 format, warnings, strict)
	else:
		_validate_multiple_files(files_to_validate, syntax_only, semantic_only,
								format, warnings, strict)


def _validate_single_file(file_path: Path, syntax_only: bool, semantic_only: bool,
						 output_format: str, show_warnings: bool, strict: bool):
	"""Validate a single APG file"""
	
	console.print(Panel(f"[bold blue]APG Validator[/bold blue]", 
					   subtitle=f"Validating {file_path}"))
	
	# Read file
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			source_content = f.read()
	except Exception as e:
		console.print(f"[red]Error reading file: {e}[/red]")
		return
	
	validation_results = {
		'file': str(file_path),
		'syntax': {'valid': True, 'errors': [], 'warnings': []},
		'semantic': {'valid': True, 'errors': [], 'warnings': []}
	}
	
	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		console=console
	) as progress:
		
		# Syntax validation
		if not semantic_only:
			task = progress.add_task("Checking syntax...", total=None)
			
			try:
				parser = APGParser()
				parse_result = parser.parse_string(source_content, str(file_path))
				
				if parse_result.success:
					validation_results['syntax']['valid'] = True
					progress.update(task, description="✅ Syntax validation passed")
				else:
					validation_results['syntax']['valid'] = False
					validation_results['syntax']['errors'] = parse_result.errors
					validation_results['syntax']['warnings'] = parse_result.warnings
					progress.update(task, description="❌ Syntax validation failed")
				
			except Exception as e:
				validation_results['syntax']['valid'] = False
				validation_results['syntax']['errors'] = [f"Parser error: {str(e)}"]
				progress.update(task, description="❌ Syntax validation error")
		
		# Semantic validation
		if not syntax_only and validation_results['syntax']['valid']:
			task = progress.add_task("Checking semantics...", total=None)
			
			try:
				# Re-parse to get AST
				parser = APGParser()
				parse_result = parser.parse_string(source_content, str(file_path))
				
				if parse_result.success and parse_result.ast:
					analyzer = SemanticAnalyzer()
					semantic_result = analyzer.analyze(parse_result.ast)
					
					if semantic_result.success:
						validation_results['semantic']['valid'] = True
						validation_results['semantic']['warnings'] = semantic_result.warnings
						progress.update(task, description="✅ Semantic validation passed")
					else:
						validation_results['semantic']['valid'] = False
						validation_results['semantic']['errors'] = semantic_result.errors
						validation_results['semantic']['warnings'] = semantic_result.warnings
						progress.update(task, description="❌ Semantic validation failed")
				
			except Exception as e:
				validation_results['semantic']['valid'] = False
				validation_results['semantic']['errors'] = [f"Semantic analyzer error: {str(e)}"]
				progress.update(task, description="❌ Semantic validation error")
	
	# Output results
	_output_validation_results(validation_results, output_format, show_warnings, strict)


def _validate_multiple_files(file_paths: List[Path], syntax_only: bool, semantic_only: bool,
							output_format: str, show_warnings: bool, strict: bool):
	"""Validate multiple APG files"""
	
	console.print(Panel(f"[bold blue]APG Validator[/bold blue]", 
					   subtitle=f"Validating {len(file_paths)} files"))
	
	all_results = []
	
	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		console=console
	) as progress:
		
		task = progress.add_task("Validating files...", total=len(file_paths))
		
		for file_path in file_paths:
			progress.update(task, description=f"Validating {file_path.name}...")
			
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					source_content = f.read()
			except Exception as e:
				all_results.append({
					'file': str(file_path),
					'syntax': {'valid': False, 'errors': [f"Read error: {e}"], 'warnings': []},
					'semantic': {'valid': False, 'errors': [], 'warnings': []}
				})
				progress.update(task, advance=1)
				continue
			
			file_results = {
				'file': str(file_path),
				'syntax': {'valid': True, 'errors': [], 'warnings': []},
				'semantic': {'valid': True, 'errors': [], 'warnings': []}
			}
			
			# Syntax validation
			if not semantic_only:
				try:
					parser = APGParser()
					parse_result = parser.parse_string(source_content, str(file_path))
					
					if not parse_result.success:
						file_results['syntax']['valid'] = False
						file_results['syntax']['errors'] = parse_result.errors
						file_results['syntax']['warnings'] = parse_result.warnings
				except Exception as e:
					file_results['syntax']['valid'] = False
					file_results['syntax']['errors'] = [f"Parser error: {str(e)}"]
			
			# Semantic validation
			if not syntax_only and file_results['syntax']['valid']:
				try:
					parser = APGParser()
					parse_result = parser.parse_string(source_content, str(file_path))
					
					if parse_result.success and parse_result.ast:
						analyzer = SemanticAnalyzer()
						semantic_result = analyzer.analyze(parse_result.ast)
						
						if not semantic_result.success:
							file_results['semantic']['valid'] = False
							file_results['semantic']['errors'] = semantic_result.errors
						
						file_results['semantic']['warnings'] = semantic_result.warnings
				except Exception as e:
					file_results['semantic']['valid'] = False
					file_results['semantic']['errors'] = [f"Semantic analyzer error: {str(e)}"]
			
			all_results.append(file_results)
			progress.update(task, advance=1)
	
	# Output results
	_output_multiple_validation_results(all_results, output_format, show_warnings, strict)


def _output_validation_results(results: Dict[str, Any], output_format: str, 
							  show_warnings: bool, strict: bool):
	"""Output validation results for a single file"""
	
	syntax_valid = results['syntax']['valid']
	semantic_valid = results['semantic']['valid']
	overall_valid = syntax_valid and semantic_valid
	
	syntax_errors = results['syntax']['errors']
	syntax_warnings = results['syntax']['warnings']
	semantic_errors = results['semantic']['errors']
	semantic_warnings = results['semantic']['warnings']
	
	if output_format == 'json':
		console.print(json.dumps(results, indent=2))
		return
	
	# Summary
	if overall_valid:
		if not syntax_warnings and not semantic_warnings:
			console.print(f"\n[green]✅ Validation passed - no issues found[/green]")
		else:
			console.print(f"\n[yellow]⚠️  Validation passed with warnings[/yellow]")
	else:
		console.print(f"\n[red]❌ Validation failed[/red]")
	
	if output_format == 'plain':
		# Plain text output
		if syntax_errors:
			console.print("\nSyntax Errors:")
			for error in syntax_errors:
				console.print(f"  - {error}")
		
		if semantic_errors:
			console.print("\nSemantic Errors:")
			for error in semantic_errors:
				console.print(f"  - {error}")
		
		if show_warnings:
			if syntax_warnings:
				console.print("\nSyntax Warnings:")
				for warning in syntax_warnings:
					console.print(f"  - {warning}")
			
			if semantic_warnings:
				console.print("\nSemantic Warnings:")
				for warning in semantic_warnings:
					console.print(f"  - {warning}")
	
	else:
		# Table output
		if syntax_errors or semantic_errors or (show_warnings and (syntax_warnings or semantic_warnings)):
			
			table = Table(show_header=True, header_style="bold magenta")
			table.add_column("Type", style="cyan")
			table.add_column("Category", style="yellow")
			table.add_column("Message", style="white")
			
			# Add errors
			for error in syntax_errors:
				table.add_row("❌ Error", "Syntax", error)
			
			for error in semantic_errors:
				table.add_row("❌ Error", "Semantic", error)
			
			# Add warnings if requested
			if show_warnings:
				for warning in syntax_warnings:
					table.add_row("⚠️  Warning", "Syntax", warning)
				
				for warning in semantic_warnings:
					table.add_row("⚠️  Warning", "Semantic", warning)
			
			console.print(table)
	
	# Exit with error code if validation failed
	if not overall_valid or (strict and (syntax_warnings or semantic_warnings)):
		sys.exit(1)


def _output_multiple_validation_results(all_results: List[Dict[str, Any]], output_format: str,
									   show_warnings: bool, strict: bool):
	"""Output validation results for multiple files"""
	
	if output_format == 'json':
		console.print(json.dumps(all_results, indent=2))
		return
	
	# Summary statistics
	total_files = len(all_results)
	valid_files = sum(1 for r in all_results if r['syntax']['valid'] and r['semantic']['valid'])
	files_with_warnings = sum(1 for r in all_results if r['syntax']['warnings'] or r['semantic']['warnings'])
	
	console.print(f"\n[bold]Validation Summary:[/bold]")
	console.print(f"  Total files: {total_files}")
	console.print(f"  Valid: {valid_files}")
	console.print(f"  Invalid: {total_files - valid_files}")
	if show_warnings:
		console.print(f"  With warnings: {files_with_warnings}")
	
	if output_format == 'table':
		# Create summary table
		table = Table(show_header=True, header_style="bold magenta")
		table.add_column("File", style="cyan")
		table.add_column("Syntax", style="green")
		table.add_column("Semantic", style="green")
		table.add_column("Issues", style="yellow")
		
		for result in all_results:
			file_name = Path(result['file']).name
			
			syntax_status = "✅" if result['syntax']['valid'] else "❌"
			semantic_status = "✅" if result['semantic']['valid'] else "❌"
			
			issues = []
			if result['syntax']['errors']:
				issues.append(f"{len(result['syntax']['errors'])} syntax errors")
			if result['semantic']['errors']:
				issues.append(f"{len(result['semantic']['errors'])} semantic errors")
			if show_warnings:
				if result['syntax']['warnings']:
					issues.append(f"{len(result['syntax']['warnings'])} syntax warnings")
				if result['semantic']['warnings']:
					issues.append(f"{len(result['semantic']['warnings'])} semantic warnings")
			
			issues_str = ", ".join(issues) if issues else "None"
			
			table.add_row(file_name, syntax_status, semantic_status, issues_str)
		
		console.print(table)
		
		# Show detailed errors
		has_errors = any(not r['syntax']['valid'] or not r['semantic']['valid'] for r in all_results)
		if has_errors:
			console.print(f"\n[bold]Detailed Errors:[/bold]")
			
			for result in all_results:
				if not result['syntax']['valid'] or not result['semantic']['valid']:
					console.print(f"\n[cyan]{result['file']}:[/cyan]")
					
					for error in result['syntax']['errors']:
						console.print(f"  [red]Syntax Error:[/red] {error}")
					
					for error in result['semantic']['errors']:
						console.print(f"  [red]Semantic Error:[/red] {error}")
	
	# Exit with error code if any validation failed
	has_failures = any(not r['syntax']['valid'] or not r['semantic']['valid'] for r in all_results)
	has_warnings = any(r['syntax']['warnings'] or r['semantic']['warnings'] for r in all_results)
	
	if has_failures or (strict and has_warnings):
		sys.exit(1)


if __name__ == '__main__':
	validate()