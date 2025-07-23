#!/usr/bin/env python3
"""
APG CLI Tool
============

Command-line interface for APG project management, compilation, and development workflow.
Provides commands for creating, building, testing, and deploying APG applications.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import shutil
from datetime import datetime

# Import APG compiler components
from .compiler.compiler import APGCompiler, CodeGenConfig, CompilationResult
from .compiler.parser import APGParser
from .compiler.semantic_analyzer import SemanticAnalyzer


# ========================================
# CLI Configuration and State
# ========================================

class APGProject:
	"""Represents an APG project configuration"""
	
	def __init__(self, project_dir: Path):
		self.project_dir = project_dir
		self.config_file = project_dir / "apg.json"
		self.config = self._load_config()
	
	def _load_config(self) -> Dict[str, Any]:
		"""Load project configuration"""
		if self.config_file.exists():
			with open(self.config_file, 'r') as f:
				return json.load(f)
		return self._default_config()
	
	def _default_config(self) -> Dict[str, Any]:
		"""Default project configuration"""
		return {
			"name": self.project_dir.name,
			"version": "1.0.0",
			"description": "APG Application",
			"author": "",
			"license": "MIT",
			"target": "flask-appbuilder",
			"source_dir": "src",
			"output_dir": "generated",
			"main_file": "app.apg",
			"dependencies": [],
			"build": {
				"include_tests": True,
				"include_docs": True,
				"optimize": False
			},
			"deployment": {
				"host": "0.0.0.0",
				"port": 8080,
				"debug": True
			}
		}
	
	def save_config(self):
		"""Save project configuration"""
		with open(self.config_file, 'w') as f:
			json.dump(self.config, f, indent=2)
	
	def get_source_files(self) -> List[Path]:
		"""Get all APG source files in the project"""
		source_dir = self.project_dir / self.config["source_dir"]
		if not source_dir.exists():
			return []
		return list(source_dir.glob("**/*.apg"))
	
	def get_main_file(self) -> Optional[Path]:
		"""Get the main APG file"""
		main_file = self.project_dir / self.config["source_dir"] / self.config["main_file"]
		return main_file if main_file.exists() else None


# ========================================
# CLI Commands Implementation
# ========================================

class APGCLICommands:
	"""Implementation of APG CLI commands"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		self.compiler = APGCompiler()
	
	def init_project(self, project_name: str, target: str = "flask-appbuilder", 
					template: Optional[str] = None) -> bool:
		"""Initialize a new APG project"""
		project_dir = Path.cwd() / project_name
		
		if project_dir.exists():
			print(f"✗ Directory '{project_name}' already exists")
			return False
		
		try:
			# Create project structure
			project_dir.mkdir()
			(project_dir / "src").mkdir()
			(project_dir / "generated").mkdir()
			(project_dir / "tests").mkdir()
			(project_dir / "docs").mkdir()
			
			# Create APG project configuration
			project = APGProject(project_dir)
			project.config.update({
				"name": project_name,
				"target": target,
				"created": datetime.now().isoformat()
			})
			project.save_config()
			
			# Create sample APG file
			if template:
				self._create_from_template(project_dir, template)
			else:
				self._create_sample_app(project_dir, project_name)
			
			# Create additional project files
			self._create_gitignore(project_dir)
			self._create_readme(project_dir, project_name)
			
			print(f"✓ Created APG project '{project_name}'")
			print(f"  Location: {project_dir}")
			print(f"  Target: {target}")
			print(f"  Main file: src/{project.config['main_file']}")
			print(f"\nNext steps:")
			print(f"  cd {project_name}")
			print(f"  apg build")
			print(f"  apg serve")
			
			return True
			
		except Exception as e:
			print(f"✗ Failed to create project: {e}")
			if project_dir.exists():
				shutil.rmtree(project_dir)
			return False
	
	def build_project(self, project_dir: Path = None, clean: bool = False,
					 verbose: bool = False) -> bool:
		"""Build APG project"""
		project_dir = project_dir or Path.cwd()
		project = APGProject(project_dir)
		
		if not project.config_file.exists():
			print("✗ No APG project found. Run 'apg init' first.")
			return False
		
		try:
			print(f"Building APG project: {project.config['name']}")
			
			# Clean output directory if requested
			output_dir = project_dir / project.config["output_dir"]
			if clean and output_dir.exists():
				shutil.rmtree(output_dir)
				print("  Cleaned output directory")
			
			# Get source files
			source_files = project.get_source_files()
			if not source_files:
				print("✗ No APG source files found")
				return False
			
			print(f"  Found {len(source_files)} source file(s)")
			
			# Configure compiler
			config = CodeGenConfig(
				target_language=project.config["target"],
				output_directory=str(output_dir),
				generate_tests=project.config["build"]["include_tests"],
				use_async=True
			)
			self.compiler.set_config(**config.__dict__)
			
			# Compile all source files
			results = []
			for source_file in source_files:
				if verbose:
					print(f"  Compiling: {source_file.relative_to(project_dir)}")
				
				result = self.compiler.compile_file(source_file, output_dir, project.config["target"])
				results.append(result)
				
				if not result.success:
					print(f"✗ Failed to compile {source_file.name}")
					for error in result.errors[:3]:  # Show first 3 errors
						print(f"    {error}")
					if len(result.errors) > 3:
						print(f"    ... and {len(result.errors) - 3} more errors")
			
			# Print build summary
			successful = sum(1 for r in results if r.success)
			total_files = sum(len(r.generated_files) for r in results if r.success)
			total_time = sum(r.compilation_time for r in results)
			
			if successful == len(results):
				print(f"✓ Build successful in {total_time:.2f}s")
				print(f"  Generated {total_files} files")
				print(f"  Output: {output_dir}")
				
				# Show next steps
				if project.config["target"] == "flask-appbuilder":
					print(f"\nNext steps:")
					print(f"  cd {output_dir}")
					print(f"  pip install -r requirements.txt")
					print(f"  python app.py")
				
				return True
			else:
				print(f"✗ Build failed: {successful}/{len(results)} files compiled")
				return False
				
		except Exception as e:
			print(f"✗ Build failed: {e}")
			return False
	
	def serve_project(self, project_dir: Path = None, host: str = None, 
					 port: int = None, debug: bool = None) -> bool:
		"""Serve the generated Flask-AppBuilder application"""
		project_dir = project_dir or Path.cwd()
		project = APGProject(project_dir)
		
		if not project.config_file.exists():
			print("✗ No APG project found")
			return False
		
		output_dir = project_dir / project.config["output_dir"]
		app_file = output_dir / "app.py"
		
		if not app_file.exists():
			print("✗ No generated application found. Run 'apg build' first.")
			return False
		
		# Use project config or provided values
		serve_host = host or project.config["deployment"]["host"]
		serve_port = port or project.config["deployment"]["port"]
		serve_debug = debug if debug is not None else project.config["deployment"]["debug"]
		
		try:
			print(f"Starting APG application: {project.config['name']}")
			print(f"  Host: {serve_host}")
			print(f"  Port: {serve_port}")
			print(f"  Debug: {serve_debug}")
			print(f"  URL: http://{serve_host}:{serve_port}")
			print("\nPress Ctrl+C to stop")
			
			# Change to output directory and run the app
			original_cwd = os.getcwd()
			os.chdir(output_dir)
			
			try:
				# Set environment variables
				env = os.environ.copy()
				env.update({
					"FLASK_HOST": serve_host,
					"FLASK_PORT": str(serve_port),
					"FLASK_DEBUG": "1" if serve_debug else "0"
				})
				
				# Run the Flask app
				subprocess.run([sys.executable, "app.py"], env=env, check=True)
				return True
				
			finally:
				os.chdir(original_cwd)
				
		except KeyboardInterrupt:
			print("\n✓ Application stopped")
			return True
		except subprocess.CalledProcessError as e:
			print(f"✗ Application failed to start: {e}")
			return False
		except Exception as e:
			print(f"✗ Failed to serve application: {e}")
			return False
	
	def validate_project(self, project_dir: Path = None) -> bool:
		"""Validate APG project without building"""
		project_dir = project_dir or Path.cwd()
		project = APGProject(project_dir)
		
		if not project.config_file.exists():
			print("✗ No APG project found")
			return False
		
		try:
			print(f"Validating APG project: {project.config['name']}")
			
			source_files = project.get_source_files()
			if not source_files:
				print("✗ No APG source files found")
				return False
			
			all_valid = True
			total_errors = 0
			total_warnings = 0
			
			for source_file in source_files:
				print(f"  Checking: {source_file.relative_to(project_dir)}")
				
				# Parse and validate
				parser = APGParser()
				result = parser.parse_file(str(source_file))
				
				if result['errors']:
					print(f"    ✗ {len(result['errors'])} syntax error(s)")
					for error in result['errors'][:2]:
						print(f"      {error}")
					all_valid = False
					total_errors += len(result['errors'])
				else:
					print(f"    ✓ Syntax valid")
				
				# Semantic validation if syntax is valid
				if result['success'] and result['parse_tree']:
					from .compiler.ast_builder import ASTBuilder
					ast_builder = ASTBuilder()
					ast = ast_builder.build_ast(result['parse_tree'], str(source_file))
					
					if ast:
						analyzer = SemanticAnalyzer()
						semantic_result = analyzer.analyze(ast)
						
						if semantic_result['errors']:
							print(f"    ✗ {len(semantic_result['errors'])} semantic error(s)")
							for error in semantic_result['errors'][:2]:
								print(f"      {error}")
							all_valid = False
							total_errors += len(semantic_result['errors'])
						
						if semantic_result['warnings']:
							print(f"    ⚠ {len(semantic_result['warnings'])} warning(s)")
							total_warnings += len(semantic_result['warnings'])
						
						if not semantic_result['errors'] and not semantic_result['warnings']:
							print(f"    ✓ Semantics valid")
			
			# Print summary
			if all_valid:
				print(f"✓ All files valid")
				if total_warnings:
					print(f"  {total_warnings} warning(s) found")
			else:
				print(f"✗ Validation failed: {total_errors} error(s), {total_warnings} warning(s)")
			
			return all_valid
			
		except Exception as e:
			print(f"✗ Validation failed: {e}")
			return False
	
	def list_templates(self) -> bool:
		"""List available project templates"""
		templates = {
			"basic": "Basic APG application with single agent",
			"webapp": "Web application with multiple agents and database",
			"iot": "IoT application with digital twins and sensors",
			"workflow": "Workflow automation application",
			"analytics": "Data analytics and reporting application"
		}
		
		print("Available APG project templates:")
		for name, description in templates.items():
			print(f"  {name:<12} - {description}")
		
		return True
	
	def show_info(self, project_dir: Path = None) -> bool:
		"""Show project information"""
		project_dir = project_dir or Path.cwd()
		project = APGProject(project_dir)
		
		if not project.config_file.exists():
			print("✗ No APG project found")
			return False
		
		config = project.config
		source_files = project.get_source_files()
		output_dir = project_dir / config["output_dir"]
		
		print(f"APG Project Information")
		print(f"=" * 50)
		print(f"Name:        {config['name']}")
		print(f"Version:     {config['version']}")
		print(f"Description: {config['description']}")
		print(f"Author:      {config.get('author', 'Not specified')}")
		print(f"License:     {config.get('license', 'Not specified')}")
		print(f"Target:      {config['target']}")
		print(f"")
		print(f"Directories:")
		print(f"  Project:   {project_dir}")
		print(f"  Source:    {project_dir / config['source_dir']}")
		print(f"  Output:    {output_dir}")
		print(f"")
		print(f"Files:")
		print(f"  Source files: {len(source_files)}")
		for sf in source_files:
			print(f"    - {sf.relative_to(project_dir)}")
		
		if output_dir.exists():
			generated_files = list(output_dir.glob("**/*"))
			generated_files = [f for f in generated_files if f.is_file()]
			print(f"  Generated files: {len(generated_files)}")
		else:
			print(f"  Generated files: 0 (not built)")
		
		print(f"")
		print(f"Configuration:")
		print(f"  Include tests: {config['build']['include_tests']}")
		print(f"  Include docs:  {config['build']['include_docs']}")
		print(f"  Host:         {config['deployment']['host']}")
		print(f"  Port:         {config['deployment']['port']}")
		
		return True
	
	# ========================================
	# Helper Methods
	# ========================================
	
	def _create_sample_app(self, project_dir: Path, project_name: str):
		"""Create a sample APG application"""
		sample_code = f'''module {project_name} version 1.0.0 {{
	description: "Sample APG application";
	author: "APG Developer";
	license: "MIT";
}}

// Sample agent for demonstration
agent HelloAgent {{
	name: str = "Hello Agent";
	message: str = "Hello from APG!";
	counter: int = 0;
	
	greet: (visitor: str) -> str = {{
		counter = counter + 1;
		return "Hello " + visitor + "! Message #" + str(counter);
	}};
	
	get_stats: () -> dict = {{
		return {{
			"name": name,
			"message": message,
			"greetings_sent": counter
		}};
	}};
	
	reset_counter: () -> void = {{
		counter = 0;
	}};
}}

// Sample workflow
workflow SampleWorkflow {{
	name: str = "Sample Workflow";
	steps: list[str] = ["start", "process", "finish"];
	current_step: int = 0;
	
	execute_step: (step_name: str) -> bool = {{
		// TODO: Implement step logic
		return true;
	}};
	
	get_progress: () -> dict = {{
		return {{
			"current_step": current_step,
			"total_steps": len(steps),
			"progress": current_step / len(steps) * 100
		}};
	}};
}}
'''
		
		main_file = project_dir / "src" / "app.apg"
		with open(main_file, 'w') as f:
			f.write(sample_code)
	
	def _create_from_template(self, project_dir: Path, template: str):
		"""Create project from template"""
		# For now, just create the basic template
		# In the future, this could load from template files
		self._create_sample_app(project_dir, project_dir.name)
	
	def _create_gitignore(self, project_dir: Path):
		"""Create .gitignore file"""
		gitignore_content = '''# APG Generated Files
generated/
*.pyc
__pycache__/
*.pyo
*.pyd
.Python
env/
venv/
.env
.venv

# IDE Files
.vscode/
.idea/
*.swp
*.swo

# OS Files
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite
*.sqlite3

# Flask-AppBuilder
app.db
'''
		
		gitignore_file = project_dir / ".gitignore"
		with open(gitignore_file, 'w') as f:
			f.write(gitignore_content)
	
	def _create_readme(self, project_dir: Path, project_name: str):
		"""Create README.md file"""
		readme_content = f'''# {project_name}

APG (Application Programming Generation) project.

## Quick Start

1. **Build the project:**
   ```bash
   apg build
   ```

2. **Install dependencies:**
   ```bash
   cd generated
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   apg serve
   ```

4. **Access the application:**
   Open http://localhost:8080 in your browser

## Project Structure

- `src/` - APG source files
- `generated/` - Generated Flask-AppBuilder application
- `tests/` - Test files
- `docs/` - Documentation
- `apg.json` - Project configuration

## APG Commands

- `apg build` - Build the project
- `apg serve` - Run the application
- `apg validate` - Validate source code
- `apg info` - Show project information

## Generated Application

This project generates a Flask-AppBuilder web application with:

- Interactive dashboards for agents and workflows
- RESTful API endpoints
- Database management interfaces
- User authentication and authorization
- Responsive Bootstrap UI

## Development

Edit the APG source files in the `src/` directory and run `apg build` to regenerate the application.

For more information about APG language features, see the [APG Documentation](https://apg-lang.org/docs).
'''
		
		readme_file = project_dir / "README.md"
		with open(readme_file, 'w') as f:
			f.write(readme_content)


# ========================================
# Main CLI Interface
# ========================================

def create_parser() -> argparse.ArgumentParser:
	"""Create the main argument parser"""
	parser = argparse.ArgumentParser(
		prog='apg',
		description='APG Language CLI - Application Programming Generation',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog='''
Examples:
  apg init myapp                    # Create new APG project
  apg build                         # Build current project
  apg serve                         # Run the generated application
  apg validate                      # Validate source code
  apg info                          # Show project information

For more help on specific commands, use:
  apg <command> --help
		'''
	)
	
	parser.add_argument('--version', action='version', version='APG CLI 1.0.0')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	
	subparsers = parser.add_subparsers(dest='command', help='Available commands')
	
	# Init command
	init_parser = subparsers.add_parser('init', help='Initialize new APG project')
	init_parser.add_argument('name', help='Project name')
	init_parser.add_argument('--target', '-t', default='flask-appbuilder',
							choices=['flask-appbuilder'], help='Target framework')
	init_parser.add_argument('--template', help='Project template')
	
	# Build command
	build_parser = subparsers.add_parser('build', help='Build APG project')
	build_parser.add_argument('--clean', '-c', action='store_true', help='Clean before build')
	build_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	
	# Serve command
	serve_parser = subparsers.add_parser('serve', help='Run generated application')
	serve_parser.add_argument('--host', default=None, help='Host address')
	serve_parser.add_argument('--port', '-p', type=int, default=None, help='Port number')
	serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
	serve_parser.add_argument('--no-debug', dest='debug', action='store_false', help='Disable debug mode')
	
	# Validate command
	validate_parser = subparsers.add_parser('validate', help='Validate APG source code')
	
	# Info command
	info_parser = subparsers.add_parser('info', help='Show project information')
	
	# Templates command
	templates_parser = subparsers.add_parser('templates', help='List available templates')
	
	return parser


def main():
	"""Main CLI entry point"""
	parser = create_parser()
	args = parser.parse_args()
	
	# Configure logging
	log_level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
	logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
	
	# Initialize CLI commands
	cli = APGCLICommands()
	
	# Execute command
	success = False
	
	if args.command == 'init':
		success = cli.init_project(args.name, args.target, args.template)
	
	elif args.command == 'build':
		success = cli.build_project(clean=args.clean, verbose=getattr(args, 'verbose', False))
	
	elif args.command == 'serve':
		success = cli.serve_project(host=args.host, port=args.port, debug=args.debug)
	
	elif args.command == 'validate':
		success = cli.validate_project()
	
	elif args.command == 'info':
		success = cli.show_info()
	
	elif args.command == 'templates':
		success = cli.list_templates()
	
	else:
		parser.print_help()
		success = True
	
	sys.exit(0 if success else 1)


if __name__ == '__main__':
	main()