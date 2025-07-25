#!/usr/bin/env python3
"""
APG Compile Command
===================

Command-line interface for compiling APG source files.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Add APG modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.compiler import APGCompiler, CodeGenConfig
from compiler.parser import APGParser

console = Console()


@click.command()
@click.argument('source_file', required=False)
@click.option('--output', '-o', help='Output directory')
@click.option('--target', '-t', default='flask-appbuilder', 
			 type=click.Choice(['flask-appbuilder', 'django', 'fastapi']),
			 help='Target framework')
@click.option('--generate-parser', is_flag=True, help='Generate ANTLR parser from grammar')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--watch', '-w', is_flag=True, help='Watch for file changes')
@click.option('--no-runtime', is_flag=True, help='Skip runtime generation')
@click.option('--tests', is_flag=True, help='Generate test files')
@click.option('--docs', is_flag=True, help='Generate documentation')
def compile_cmd(source_file: Optional[str], output: Optional[str], target: str,
			   generate_parser: bool, verbose: bool, watch: bool, 
			   no_runtime: bool, tests: bool, docs: bool):
	"""Compile APG source files to target framework"""
	
	if generate_parser:
		_generate_parser()
		return
	
	# Determine source file
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
			console.print("Try: apg init  # to initialize APG project")
			return
	
	source_path = Path(source_file)
	if not source_path.exists():
		console.print(f"[red]Source file not found: {source_file}[/red]")
		return
	
	# Determine output directory
	if not output:
		if Path('apg.json').exists():
			with open('apg.json', 'r') as f:
				config = json.load(f)
			output = config.get('build', {}).get('output_directory', 'generated')
		else:
			output = 'generated'
	
	output_dir = Path(output)
	
	# Create compiler configuration
	config = CodeGenConfig(
		target_language=target,
		output_directory=str(output_dir),
		generate_tests=tests,
		include_runtime=not no_runtime,
		generate_docs=docs,
		verbose=verbose
	)
	
	if watch:
		_watch_and_compile(source_path, config)
	else:
		_compile_single(source_path, config, verbose)


def _generate_parser():
	"""Generate ANTLR parser from grammar"""
	console.print("[blue]Generating ANTLR parser from grammar...[/blue]")
	
	grammar_file = Path(__file__).parent.parent / 'spec' / 'apg.g4'
	if not grammar_file.exists():
		console.print(f"[red]Grammar file not found: {grammar_file}[/red]")
		return
	
	output_dir = Path(__file__).parent.parent / 'compiler' / 'generated'
	output_dir.mkdir(exist_ok=True)
	
	try:
		import subprocess
		
		# Run ANTLR
		cmd = [
			'antlr4',
			'-Dlanguage=Python3',
			'-o', str(output_dir),
			'-visitor',
			'-listener',
			str(grammar_file)
		]
		
		with Progress(
			SpinnerColumn(),
			TextColumn("[progress.description]{task.description}"),
			console=console
		) as progress:
			task = progress.add_task("Generating parser...", total=None)
			
			result = subprocess.run(cmd, capture_output=True, text=True)
			
			if result.returncode == 0:
				progress.update(task, description="✅ Parser generated successfully")
				console.print(f"[green]✅ ANTLR parser generated in {output_dir}[/green]")
				
				# List generated files
				generated_files = list(output_dir.glob('APG*.py'))
				if generated_files:
					console.print(f"[cyan]Generated files:[/cyan]")
					for file in generated_files:
						console.print(f"  - {file.name}")
			else:
				progress.update(task, description="❌ Parser generation failed")
				console.print(f"[red]❌ ANTLR parser generation failed[/red]")
				console.print(f"[red]Error: {result.stderr}[/red]")
				
	except FileNotFoundError:
		console.print("[red]ANTLR4 not found. Install with: pip install antlr4-tools[/red]")
	except Exception as e:
		console.print(f"[red]Error generating parser: {e}[/red]")


def _compile_single(source_path: Path, config: CodeGenConfig, verbose: bool):
	"""Compile a single APG source file"""
	
	console.print(Panel(f"[bold blue]APG Compiler[/bold blue]", 
					   subtitle=f"Compiling {source_path}"))
	
	start_time = time.time()
	
	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		BarColumn(),
		TimeElapsedColumn(),
		console=console
	) as progress:
		
		# Initialize compiler
		task = progress.add_task("Initializing compiler...", total=5)
		compiler = APGCompiler(config)
		progress.update(task, advance=1)
		
		# Read source file
		progress.update(task, description="Reading source file...")
		try:
			with open(source_path, 'r', encoding='utf-8') as f:
				source_content = f.read()
		except Exception as e:
			console.print(f"[red]Error reading source file: {e}[/red]")
			return
		progress.update(task, advance=1)
		
		# Compile
		progress.update(task, description="Compiling APG source...")
		result = compiler.compile_string(source_content, source_path.stem)
		progress.update(task, advance=3)
		
		compilation_time = time.time() - start_time
		
		if result.success:
			progress.update(task, description="✅ Compilation successful!")
			
			console.print(f"\n[green]✅ Compilation successful![/green]")
			console.print(f"[cyan]Time:[/cyan] {compilation_time:.2f}s")
			console.print(f"[cyan]Generated files:[/cyan] {len(result.generated_files)}")
			
			if verbose:
				_show_compilation_details(result, config)
			
			# Write generated files
			_write_generated_files(result.generated_files, Path(config.output_directory))
			
			console.print(f"\n[green]Next steps:[/green]")
			console.print(f"  1. cd {config.output_directory}")
			console.print(f"  2. pip install -r requirements.txt")
			console.print(f"  3. python app.py")
			console.print(f"\n[green]Then open:[/green] http://localhost:8080")
			
		else:
			progress.update(task, description="❌ Compilation failed")
			
			console.print(f"\n[red]❌ Compilation failed![/red]")
			console.print(f"[cyan]Time:[/cyan] {compilation_time:.2f}s")
			
			if result.errors:
				console.print(f"\n[red]Errors:[/red]")
				for error in result.errors:
					console.print(f"  - {error}")
			
			if result.warnings:
				console.print(f"\n[yellow]Warnings:[/yellow]")
				for warning in result.warnings:
					console.print(f"  - {warning}")


def _show_compilation_details(result, config: CodeGenConfig):
	"""Show detailed compilation information"""
	
	console.print("\n[bold]Compilation Details:[/bold]")
	
	# Show phases
	phases_table = Table(show_header=True, header_style="bold magenta")
	phases_table.add_column("Phase", style="cyan")
	phases_table.add_column("Status", style="green")
	phases_table.add_column("Time", style="yellow")
	
	for phase, info in result.phase_info.items():
		status = "✅ Success" if info.get('success', True) else "❌ Failed"
		time_str = f"{info.get('time', 0):.3f}s"
		phases_table.add_row(phase, status, time_str)
	
	console.print(phases_table)
	
	# Show generated files
	if result.generated_files:
		console.print(f"\n[bold]Generated Files:[/bold]")
		files_table = Table(show_header=True, header_style="bold magenta")
		files_table.add_column("File", style="cyan")
		files_table.add_column("Lines", style="yellow")
		files_table.add_column("Size", style="green")
		
		for filename, content in result.generated_files.items():
			lines = len(content.splitlines())
			size = len(content.encode('utf-8'))
			size_str = f"{size:,} bytes"
			
			files_table.add_row(filename, str(lines), size_str)
		
		console.print(files_table)
	
	# Show statistics
	if result.statistics:
		console.print(f"\n[bold]Statistics:[/bold]")
		stats_table = Table(show_header=False, box=None)
		stats_table.add_column("Metric", style="cyan")
		stats_table.add_column("Count", style="white")
		
		for metric, count in result.statistics.items():
			stats_table.add_row(metric.replace('_', ' ').title(), str(count))
		
		console.print(stats_table)


def _write_generated_files(generated_files: dict, output_dir: Path):
	"""Write generated files to output directory"""
	
	output_dir.mkdir(parents=True, exist_ok=True)
	
	console.print(f"\n[blue]Writing files to {output_dir}...[/blue]")
	
	for filename, content in generated_files.items():
		file_path = output_dir / filename
		
		# Create subdirectories if needed
		file_path.parent.mkdir(parents=True, exist_ok=True)
		
		# Write file
		with open(file_path, 'w', encoding='utf-8') as f:
			f.write(content)
		
		console.print(f"  ✅ {filename}")


def _watch_and_compile(source_path: Path, config: CodeGenConfig):
	"""Watch source file for changes and recompile"""
	
	console.print(f"[blue]Watching {source_path} for changes...[/blue]")
	console.print("[yellow]Press Ctrl+C to stop[/yellow]")
	
	import time
	import hashlib
	
	def get_file_hash(path: Path) -> str:
		"""Get hash of file content"""
		try:
			with open(path, 'rb') as f:
				return hashlib.md5(f.read()).hexdigest()
		except:
			return ""
	
	last_hash = get_file_hash(source_path)
	
	try:
		while True:
			time.sleep(1)  # Check every second
			
			current_hash = get_file_hash(source_path)
			if current_hash != last_hash and current_hash:
				console.print(f"\n[cyan]Change detected in {source_path}[/cyan]")
				_compile_single(source_path, config, False)
				last_hash = current_hash
				console.print(f"\n[blue]Watching {source_path} for changes...[/blue]")
			
	except KeyboardInterrupt:
		console.print("\n[yellow]Stopped watching[/yellow]")


if __name__ == '__main__':
	compile_cmd()