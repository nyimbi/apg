#!/usr/bin/env python3
"""
APG Compile Command
===================

Command-line interface for compiling APG source files.
"""

import sys
import os
import json
import shutil
import subprocess
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
from compiler.code_generator import CodeGenerator
from compiler.parser import APGParser

console = Console()


@click.command()
@click.argument('source_file', required=False)
@click.option('--output', '-o', help='Output directory')
@click.option('--target', '-t', default='python',
			 type=click.Choice(['python']),
			 help='Target language')
@click.option('--generate-parser', is_flag=True, help='Generate ANTLR parser from grammar')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--watch', '-w', is_flag=True, help='Watch for file changes')
@click.option('--no-runtime', is_flag=True, help='Skip runtime generation')
@click.option('--tests', is_flag=True, help='Generate test files')
@click.option('--docs', is_flag=True, help='Generate documentation')
@click.option('--verify', is_flag=True, help='Run generated self-test and smoke test after compilation')
@click.option('--catalog', type=click.Path(path_type=Path), default=None, help='Capability contract root or local apg.capability-catalog.v1 file')
def compile_cmd(source_file: Optional[str], output: Optional[str], target: str,
			   generate_parser: bool, verbose: bool, watch: bool, 
			   no_runtime: bool, tests: bool, docs: bool, verify: bool, catalog: Path | None):
	"""Compile APG source files to Python artifacts."""
	
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
	if catalog is not None and not catalog.exists():
		raise click.ClickException(f"Capability catalog not found: {catalog}")
	
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
		target_language=CodeGenerator.normalize_target(target),
		output_directory=str(output_dir),
		generate_tests=tests,
		include_runtime=not no_runtime,
		generate_docs=docs,
		verbose=verbose
	)
	
	if watch:
		_watch_and_compile(source_path, config, catalog)
	else:
		_compile_single(source_path, config, verbose, verify, catalog)


def _generate_parser():
	"""Generate ANTLR parser from grammar"""
	console.print("[blue]Generating ANTLR parser from grammar...[/blue]")
	
	grammar_file = Path(__file__).parent.parent / 'spec' / 'apg.g4'
	if not grammar_file.exists():
		console.print(f"[red]Grammar file not found: {grammar_file}[/red]")
		return
	
	output_dir = grammar_file.parent
	
	try:
		import subprocess
		antlr_command = shutil.which('antlr4') or shutil.which('antlr')
		if not antlr_command:
			raise FileNotFoundError("antlr4")
		
		# Run ANTLR
		cmd = [
			antlr_command,
			'-Dlanguage=Python3',
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
			
			result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
			
			if result.returncode == 0:
				_strip_generated_parser_whitespace(output_dir)
				progress.update(task, description="✅ Parser generated successfully")
				console.print(f"[green]✅ ANTLR parser generated in {output_dir}[/green]")
				
				# List generated files
				generated_files = list(output_dir.glob('apg*.py'))
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


def _strip_generated_parser_whitespace(output_dir: Path):
	"""Keep generated parser artifacts compatible with repository diff checks."""
	for file_path in output_dir.glob("apg*.py"):
		text = file_path.read_text(encoding="utf-8")
		cleaned = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
		file_path.write_text(cleaned, encoding="utf-8")


def _compile_single(
	source_path: Path,
	config: CodeGenConfig,
	verbose: bool,
	verify: bool = False,
	catalog: Path | None = None,
):
	"""Compile a single APG source file"""
	if catalog is not None:
		_validate_compile_preflight(source_path, config, catalog)
	
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
			output_dir = Path(config.output_directory)
			_write_generated_files(result.generated_files, output_dir)

			if verify and not _verify_generated_app(output_dir):
				raise click.ClickException("Generated verification failed")
			
			console.print(f"\n[green]Next steps:[/green]")
			console.print(f"  1. Inspect generated files in {config.output_directory}", soft_wrap=True)
			console.print(f"  2. pip install -r {config.output_directory}/requirements.txt", soft_wrap=True)
			console.print(f"  3. python {config.output_directory}/app.py", soft_wrap=True)
			console.print(f"  4. python {config.output_directory}/app.py --describe", soft_wrap=True)
			console.print(f"  5. python {config.output_directory}/app.py --self-test", soft_wrap=True)
			console.print(f"  6. apg compile {source_path} --output {config.output_directory} --verify", soft_wrap=True)
			console.print("\n[green]The generated Python app starts a standard-library HTTP server. Use --self-test for a local health contract.[/green]")
			
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


def _validate_compile_preflight(source_path: Path, config: CodeGenConfig, catalog: Path) -> None:
	"""Run no-write generator-readiness validation before catalog-aware compile."""
	from cli.validate_command import validate_path

	report = validate_path(source_path, target=config.target_language, catalog=catalog)
	if report["ok"]:
		catalog_report = report["lint"].get("capability_catalog", {})
		console.print(
			f"[green]Capability catalog preflight OK[/green]: "
			f"{catalog_report.get('catalog_kind', 'catalog')} "
			f"with {catalog_report.get('contract_count', 0)} contract(s)"
		)
		return

	console.print("[red]Compilation preflight validation failed[/red]")
	for diagnostic in report["diagnostics"]:
		start = diagnostic["range"]["start"]
		console.print(
			f"  {diagnostic['file']}:{start['line'] + 1}:{start['character']}: "
			f"{diagnostic['code']} {diagnostic['severity']}: {diagnostic['message']}"
		)
	raise click.ClickException("Compilation preflight validation failed")


def _show_compilation_details(result, config: CodeGenConfig):
	"""Show detailed compilation information"""
	
	console.print("\n[bold]Compilation Details:[/bold]")
	
	# Show phases
	phases_table = Table(show_header=True, header_style="bold magenta")
	phases_table.add_column("Phase", style="cyan")
	phases_table.add_column("Status", style="green")
	phases_table.add_column("Time", style="yellow")
	
	for phase, info in getattr(result, "phase_info", {}).items():
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
	statistics = getattr(result, "statistics", {})
	if statistics:
		console.print(f"\n[bold]Statistics:[/bold]")
		stats_table = Table(show_header=False, box=None)
		stats_table.add_column("Metric", style="cyan")
		stats_table.add_column("Count", style="white")
		
		for metric, count in statistics.items():
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


def _verify_generated_app(output_dir: Path) -> bool:
	"""Run generated app verification commands."""
	app_path = output_dir / "app.py"
	smoke_path = output_dir / "smoke_test.py"
	if not app_path.exists():
		console.print(f"[red]Generated app not found: {app_path}[/red]")
		return False
	if not smoke_path.exists():
		console.print(f"[red]Generated smoke test not found: {smoke_path}[/red]")
		return False

	checks = [
		("generated self-test", [sys.executable, "app.py", "--self-test"]),
		("generated smoke test", [sys.executable, "smoke_test.py"]),
	]
	console.print("\n[blue]Verifying generated application...[/blue]")
	for label, command in checks:
		completed = subprocess.run(
			command,
			cwd=output_dir,
			check=False,
			capture_output=True,
			text=True,
		)
		if completed.returncode != 0:
			console.print(f"[red]❌ {label} failed[/red]")
			if completed.stdout:
				console.print(completed.stdout.rstrip())
			if completed.stderr:
				console.print(completed.stderr.rstrip())
			return False
		console.print(f"  ✅ {label}")
	console.print("[green]Generated verification passed[/green]")
	return True


try:
	from watchfiles import watch as _wfiles_watch
	HAS_WATCHFILES = True
except ImportError:
	HAS_WATCHFILES = False


def _watch_and_compile(source_path: Path, config: CodeGenConfig, catalog: Path | None = None):
	"""Watch source file for changes and recompile, printing timestamped status lines."""
	from datetime import datetime

	click.echo(f"Watching {source_path}")

	# Hoist compiler construction outside the recompile loop — avoids
	# rebuilding grammar/config objects on every file-change event.
	compiler = APGCompiler(config)

	def _do_compile() -> tuple[bool, int, float]:
		"""Run a single compile pass. Returns (success, file_count, elapsed_ms)."""
		t0 = time.monotonic()
		try:
			with open(source_path, "r", encoding="utf-8") as fh:
				source_content = fh.read()
			result = compiler.compile_string(source_content, source_path.stem)
		except Exception as exc:
			elapsed_ms = (time.monotonic() - t0) * 1000
			click.echo(f"[{_ts()}] FAILED: {exc}")
			return False, 0, elapsed_ms
		elapsed_ms = (time.monotonic() - t0) * 1000
		if result.success:
			output_dir = Path(config.output_directory)
			_write_generated_files(result.generated_files, output_dir)
			click.echo(f"[{_ts()}] Compiled OK ({len(result.generated_files)} files, {elapsed_ms:.0f}ms)")
			return True, len(result.generated_files), elapsed_ms
		else:
			n_errors = len(result.errors)
			click.echo(f"[{_ts()}] FAILED: {n_errors} error(s)")
			for err in result.errors:
				click.echo(f"  {err}")
			return False, 0, elapsed_ms

	def _ts() -> str:
		return datetime.now().strftime("%H:%M:%S")

	# Initial compile
	_do_compile()

	try:
		if HAS_WATCHFILES:
			for _changes in _wfiles_watch(source_path):
				click.echo(f"[{_ts()}] Change detected — recompiling...")
				_do_compile()
		else:
			# Polling fallback: check mtime every 500 ms.
			# Also watches for source-file deletion: if the file is missing for
			# ~5 seconds (10 consecutive polls) the watcher stops cleanly.
			try:
				last_mtime = source_path.stat().st_mtime
			except OSError:
				last_mtime = 0.0
			missing_counter = 0
			while True:
				time.sleep(0.5)
				try:
					mtime = source_path.stat().st_mtime
					missing_counter = 0
				except OSError:
					missing_counter += 1
					if missing_counter > 10:  # ~5 seconds with no source file
						click.echo(f"\n[{_ts()}] Source file not found — stopping watch.")
						return
					continue
				if mtime != last_mtime:
					last_mtime = mtime
					click.echo(f"[{_ts()}] Change detected — recompiling...")
					_do_compile()
	except KeyboardInterrupt:
		click.echo("\nStopped watching.")


if __name__ == '__main__':
	compile_cmd()
