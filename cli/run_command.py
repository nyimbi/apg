#!/usr/bin/env python3
"""
APG Run Command
===============

Command-line interface for running APG applications.
"""

import sys
import os
import json
import subprocess
import signal
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

# Add APG modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@click.command()
@click.option('--app', '-a', help='Application file (default: generated/app.py)')
@click.option('--host', '-h', default='127.0.0.1', help='Host to bind to')
@click.option('--port', '-p', default=8080, help='Port to bind to')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--auto-compile', is_flag=True, help='Auto-compile before running')
@click.option('--watch', '-w', is_flag=True, help='Watch for changes and restart')
def run(app: Optional[str], host: str, port: int, debug: bool, 
	   auto_compile: bool, watch: bool):
	"""Run APG application"""
	
	# Auto-compile if requested
	if auto_compile:
		console.print("[blue]Auto-compiling APG source...[/blue]")
		result = _auto_compile()
		if not result:
			console.print("[red]Compilation failed. Cannot run application.[/red]")
			return
	
	# Determine application file
	if not app:
		# Look for generated application
		candidates = [
			'generated/app.py',
			'app.py',
			'src/app.py'
		]
		
		# Check apg.json for output directory
		if Path('apg.json').exists():
			with open('apg.json', 'r') as f:
				config = json.load(f)
			output_dir = config.get('build', {}).get('output_directory', 'generated')
			candidates.insert(0, f'{output_dir}/app.py')
		
		# Find first existing candidate
		for candidate in candidates:
			if Path(candidate).exists():
				app = candidate
				break
		
		if not app:
			console.print("[red]No application file found.[/red]")
			console.print("Try: apg compile  # to compile APG source first")
			return
	
	app_path = Path(app)
	if not app_path.exists():
		console.print(f"[red]Application file not found: {app}[/red]")
		return
	
	if watch:
		_run_with_watch(app_path, host, port, debug)
	else:
		_run_single(app_path, host, port, debug)


def _auto_compile() -> bool:
	"""Auto-compile APG source"""
	try:
		from cli.compile_command import _compile_single
		from compiler.compiler import APGCompiler, CodeGenConfig
		
		# Find source file
		source_candidates = ['main.apg', 'src/main.apg', 'app.apg']
		
		if Path('apg.json').exists():
			with open('apg.json', 'r') as f:
				config_data = json.load(f)
			source_file = config_data.get('build', {}).get('source_file', 'main.apg')
		else:
			source_file = None
		
		if not source_file or not Path(source_file).exists():
			for candidate in source_candidates:
				if Path(candidate).exists():
					source_file = candidate
					break
		
		if not source_file or not Path(source_file).exists():
			console.print("[red]No APG source file found for auto-compilation[/red]")
			return False
		
		# Create compilation config
		output_dir = 'generated'
		if Path('apg.json').exists():
			with open('apg.json', 'r') as f:
				config_data = json.load(f)
			output_dir = config_data.get('build', {}).get('output_directory', 'generated')
		
		config = CodeGenConfig(
			target_language='flask-appbuilder',
			output_directory=output_dir,
			include_runtime=True,
			verbose=False
		)
		
		# Compile
		_compile_single(Path(source_file), config, verbose=False)
		return True
		
	except Exception as e:
		console.print(f"[red]Auto-compilation error: {e}[/red]")
		return False


def _run_single(app_path: Path, host: str, port: int, debug: bool):
	"""Run application once"""
	
	console.print(Panel(f"[bold blue]APG Application Runner[/bold blue]", 
					   subtitle=f"Starting {app_path}"))
	
	# Check if it's a Flask-AppBuilder app
	with open(app_path, 'r') as f:
		content = f.read()
		is_flask_app = 'Flask' in content or 'app.run' in content
	
	if not is_flask_app:
		console.print("[red]Application does not appear to be a Flask application[/red]")
		return
	
	# Set environment variables
	env = os.environ.copy()
	env['FLASK_HOST'] = host
	env['FLASK_PORT'] = str(port)
	env['FLASK_DEBUG'] = '1' if debug else '0'
	
	console.print(f"[green]Starting APG application...[/green]")
	console.print(f"[cyan]Host:[/cyan] {host}")
	console.print(f"[cyan]Port:[/cyan] {port}")
	console.print(f"[cyan]Debug:[/cyan] {'enabled' if debug else 'disabled'}")
	console.print(f"[cyan]Application:[/cyan] {app_path}")
	console.print(f"\n[green]Access at:[/green] http://{host}:{port}")
	console.print(f"[yellow]Press Ctrl+C to stop[/yellow]")
	console.print("-" * 50)
	
	try:
		# Change to application directory
		app_dir = app_path.parent
		
		# Run the application
		subprocess.run([sys.executable, app_path.name], 
					  cwd=app_dir, env=env, check=True)
		
	except KeyboardInterrupt:
		console.print("\n[yellow]Application stopped by user[/yellow]")
	except subprocess.CalledProcessError as e:
		console.print(f"\n[red]Application exited with error code {e.returncode}[/red]")
	except Exception as e:
		console.print(f"\n[red]Error running application: {e}[/red]")


def _run_with_watch(app_path: Path, host: str, port: int, debug: bool):
	"""Run application with file watching"""
	
	console.print(f"[blue]Running {app_path} with file watching...[/blue]")
	console.print(f"[yellow]Press Ctrl+C to stop[/yellow]")
	
	import hashlib
	
	def get_file_hash(path: Path) -> str:
		"""Get hash of file content"""
		try:
			with open(path, 'rb') as f:
				return hashlib.md5(f.read()).hexdigest()
		except:
			return ""
	
	def get_directory_hash(directory: Path) -> str:
		"""Get hash of all files in directory"""
		hashes = []
		try:
			for file_path in directory.rglob('*.py'):
				if file_path.is_file():
					hashes.append(get_file_hash(file_path))
			return hashlib.md5(''.join(sorted(hashes)).encode()).hexdigest()
		except:
			return ""
	
	app_dir = app_path.parent
	source_dir = Path('.')  # Watch current directory for APG files
	
	last_app_hash = get_directory_hash(app_dir)
	last_source_hash = get_directory_hash(source_dir)
	
	process = None
	
	def start_app():
		"""Start the application process"""
		nonlocal process
		
		env = os.environ.copy()
		env['FLASK_HOST'] = host
		env['FLASK_PORT'] = str(port)
		env['FLASK_DEBUG'] = '1' if debug else '0'
		
		console.print(f"\n[green]🚀 Starting application at http://{host}:{port}[/green]")
		
		try:
			process = subprocess.Popen(
				[sys.executable, app_path.name],
				cwd=app_dir,
				env=env,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1,
				universal_newlines=True
			)
			return process
		except Exception as e:
			console.print(f"[red]Error starting application: {e}[/red]")
			return None
	
	def stop_app():
		"""Stop the application process"""
		nonlocal process
		if process:
			try:
				process.terminate()
				process.wait(timeout=5)
			except subprocess.TimeoutExpired:
				process.kill()
				process.wait()
			except:
				pass
			process = None
			console.print("[yellow]📛 Application stopped[/yellow]")
	
	# Start initial application
	process = start_app()
	
	try:
		while True:
			time.sleep(2)  # Check every 2 seconds
			
			# Check if process is still running
			if process and process.poll() is not None:
				console.print("[red]Application process died, restarting...[/red]")
				process = start_app()
				continue
			
			# Check for source file changes
			current_source_hash = get_directory_hash(source_dir)
			if current_source_hash != last_source_hash and current_source_hash:
				console.print("\n[cyan]📝 Source file changes detected[/cyan]")
				
				# Auto-compile
				console.print("[blue]🔄 Auto-compiling...[/blue]")
				if _auto_compile():
					console.print("[green]✅ Compilation successful[/green]")
					
					# Restart application
					stop_app()
					time.sleep(1)
					process = start_app()
				else:
					console.print("[red]❌ Compilation failed, keeping current app running[/red]")
				
				last_source_hash = current_source_hash
				last_app_hash = get_directory_hash(app_dir)
				continue
			
			# Check for generated app changes (manual edits)
			current_app_hash = get_directory_hash(app_dir)
			if current_app_hash != last_app_hash and current_app_hash:
				console.print("\n[cyan]📝 Application file changes detected[/cyan]")
				
				# Restart application
				stop_app()
				time.sleep(1)
				process = start_app()
				
				last_app_hash = current_app_hash
			
	except KeyboardInterrupt:
		console.print("\n[yellow]Stopping file watcher...[/yellow]")
		stop_app()


@click.command()
@click.option('--port', '-p', default=8080, help='Port to check')
@click.option('--timeout', '-t', default=10, help='Timeout in seconds')
def check(port: int, timeout: int):
	"""Check if APG application is running"""
	
	import socket
	import urllib.request
	import urllib.error
	
	url = f"http://127.0.0.1:{port}"
	
	console.print(f"[blue]Checking APG application at {url}...[/blue]")
	
	try:
		# First check if port is open
		with socket.create_connection(('127.0.0.1', port), timeout=timeout):
			pass
		
		# Then try HTTP request
		with urllib.request.urlopen(url, timeout=timeout) as response:
			status = response.getcode()
			
		if status == 200:
			console.print(f"[green]✅ Application is running at {url}[/green]")
		else:
			console.print(f"[yellow]⚠️  Application responded with status {status}[/yellow]")
			
	except socket.timeout:
		console.print(f"[red]❌ Connection timeout - application may not be running[/red]")
	except ConnectionRefusedError:
		console.print(f"[red]❌ Connection refused - application is not running[/red]")
	except urllib.error.URLError as e:
		console.print(f"[red]❌ HTTP error: {e}[/red]")
	except Exception as e:
		console.print(f"[red]❌ Error checking application: {e}[/red]")


@click.command()
@click.option('--port', '-p', default=8080, help='Port to kill process on')
def stop(port: int):
	"""Stop APG application running on specified port"""
	
	import psutil
	
	console.print(f"[blue]Looking for processes on port {port}...[/blue]")
	
	killed = 0
	for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
		try:
			# Check if process is using the port
			connections = proc.connections()
			for conn in connections:
				if conn.laddr.port == port:
					console.print(f"[yellow]Killing process {proc.info['pid']} ({proc.info['name']})[/yellow]")
					proc.terminate()
					proc.wait(timeout=5)
					killed += 1
					break
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
			continue
	
	if killed > 0:
		console.print(f"[green]✅ Killed {killed} process(es)[/green]")
	else:
		console.print(f"[yellow]No processes found on port {port}[/yellow]")


# Add commands to a group
@click.group()
def run_group():
	"""Run and manage APG applications"""
	pass

run_group.add_command(run)
run_group.add_command(check)
run_group.add_command(stop)

if __name__ == '__main__':
	run_group()