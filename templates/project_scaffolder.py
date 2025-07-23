#!/usr/bin/env python3
"""
APG Project Scaffolder
=====================

Creates complete APG projects from templates with customization and code generation.
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from datetime import datetime

from .template_types import TemplateType, ProjectConfig
from .template_manager import TemplateManager


class ProjectScaffolder:
	"""Scaffolds complete APG projects from templates"""
	
	def __init__(self, template_manager: Optional[TemplateManager] = None):
		"""Initialize project scaffolder"""
		self.template_manager = template_manager or TemplateManager()
		self.variable_pattern = re.compile(r'\{\{(\w+)\}\}')
	
	def create_project(self, config: ProjectConfig) -> Dict[str, Any]:
		"""Create a complete APG project from template"""
		
		# Validate configuration
		validation_errors = self._validate_config(config)
		if validation_errors:
			return {
				'success': False,
				'errors': validation_errors,
				'project_path': None
			}
		
		# Set up output directory
		if config.output_directory is None:
			config.output_directory = Path.cwd() / config.name
		
		project_path = Path(config.output_directory)
		
		# Check if project already exists
		if project_path.exists() and not config.overwrite_existing:
			return {
				'success': False,
				'errors': [f'Project directory already exists: {project_path}'],
				'project_path': project_path
			}
		
		try:
			# Create project directory
			project_path.mkdir(parents=True, exist_ok=config.overwrite_existing)
			
			# Get template
			template = self.template_manager.get_template(config.template_type)
			if not template:
				return {
					'success': False,
					'errors': [f'Template not found: {config.template_type.value}'],
					'project_path': project_path
				}
			
			# Generate project files
			generated_files = self._generate_project_files(template, config, project_path)
			
			# Create project metadata
			self._create_project_metadata(config, project_path)
			
			# Initialize git repository if requested
			if hasattr(config, 'initialize_git') and config.initialize_git:
				self._initialize_git_repo(project_path)
			
			return {
				'success': True,
				'errors': [],
				'project_path': project_path,
				'generated_files': generated_files,
				'template_used': config.template_type.value
			}
			
		except Exception as e:
			return {
				'success': False,
				'errors': [f'Error creating project: {str(e)}'],
				'project_path': project_path
			}
	
	def _validate_config(self, config: ProjectConfig) -> List[str]:
		"""Validate project configuration"""
		errors = []
		
		# Check required fields
		if not config.name or not config.name.strip():
			errors.append("Project name is required")
		
		if not config.description or not config.description.strip():
			errors.append("Project description is required")
		
		# Validate project name
		if config.name:
			if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', config.name):
				errors.append("Project name must start with letter and contain only letters, numbers, underscores, and hyphens")
		
		# Check template exists
		template = self.template_manager.get_template(config.template_type)
		if not template:
			errors.append(f"Template not found: {config.template_type.value}")
		
		return errors
	
	def _generate_project_files(self, template: Dict[str, Any], config: ProjectConfig, project_path: Path) -> List[str]:
		"""Generate all project files from template"""
		generated_files = []
		
		# Prepare template variables
		template_vars = self._prepare_template_variables(template, config)
		
		# Process template files
		template_files = template.get('template_files', {})
		
		for template_filename, template_content in template_files.items():
			# Remove .template extension for output filename
			output_filename = template_filename.replace('.template', '')
			output_path = project_path / output_filename
			
			# Create subdirectories if needed
			output_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Process template content
			processed_content = self._process_template_content(template_content, template_vars)
			
			# Write file
			with open(output_path, 'w', encoding='utf-8') as f:
				f.write(processed_content)
			
			generated_files.append(str(output_path.relative_to(project_path)))
		
		# Generate additional standard files
		additional_files = self._generate_additional_files(config, project_path)
		generated_files.extend(additional_files)
		
		return generated_files
	
	def _prepare_template_variables(self, template: Dict[str, Any], config: ProjectConfig) -> Dict[str, Any]:
		"""Prepare variables for template substitution"""
		
		# Base variables from config
		variables = {
			'project_name': config.name,
			'project_description': config.description,
			'author': config.author,
			'version': config.version,
			'license': config.license,
			'target_framework': config.target_framework,
			'database_type': config.database_type,
			'python_version': config.python_version,
			'current_date': datetime.now().strftime('%Y-%m-%d'),
			'current_year': datetime.now().year
		}
		
		# Feature flags
		variables.update({
			'enable_authentication': config.enable_authentication,
			'enable_api': config.enable_api,
			'enable_database': config.enable_database,
			'enable_testing': config.enable_testing,
			'enable_docker': config.enable_docker,
			'enable_ai_features': config.enable_ai_features,
			'enable_real_time': config.enable_real_time,
			'use_async': config.use_async,
			'include_examples': config.include_examples,
			'generate_docs': config.generate_docs
		})
		
		# Template-specific variables
		template_vars = template.get('variables', {})
		for var_name, default_value in template_vars.items():
			if var_name not in variables:
				variables[var_name] = default_value
		
		# Special template-specific logic
		if config.template_type == TemplateType.TASK_MANAGEMENT:
			variables.update({
				'enable_assignments': True,
				'enable_notifications': config.enable_real_time,
				'enable_deadlines': True
			})
		elif config.template_type == TemplateType.E_COMMERCE:
			variables.update({
				'enable_payments': True,
				'enable_shipping': True,
				'enable_inventory': True
			})
		elif config.template_type == TemplateType.AI_ASSISTANT:
			variables.update({
				'ai_provider': 'openai',
				'enable_nlp': config.enable_ai_features,
				'knowledge_base': 'local'
			})
		elif config.template_type == TemplateType.MICROSERVICES:
			variables.update({
				'service_count': 3,
				'enable_service_discovery': True,
				'enable_api_gateway': True
			})
		
		return variables
	
	def _process_template_content(self, content: str, variables: Dict[str, Any]) -> str:
		"""Process template content with variable substitution"""
		
		# Simple Jinja2-like template processing
		processed_content = content
		
		# Handle conditional blocks {% if condition %}...{% endif %}
		processed_content = self._process_conditional_blocks(processed_content, variables)
		
		# Handle loops {% for item in list %}...{% endfor %}
		processed_content = self._process_loop_blocks(processed_content, variables)
		
		# Replace variables {{variable_name}}
		for var_name, var_value in variables.items():
			placeholder = f'{{{{{var_name}}}}}'
			if isinstance(var_value, bool):
				# Convert boolean to string for template
				var_value = 'true' if var_value else 'false'
			elif var_value is None:
				var_value = 'null'
			
			processed_content = processed_content.replace(placeholder, str(var_value))
		
		return processed_content
	
	def _process_conditional_blocks(self, content: str, variables: Dict[str, Any]) -> str:
		"""Process conditional blocks in template"""
		
		# Pattern for {% if condition %}...{% endif %}
		if_pattern = re.compile(r'\{\%\s*if\s+(\w+)\s*\%\}(.*?)\{\%\s*endif\s*\%\}', re.DOTALL)
		
		def replace_if_block(match):
			condition_var = match.group(1)
			block_content = match.group(2)
			
			# Evaluate condition
			condition_value = variables.get(condition_var, False)
			if isinstance(condition_value, str):
				condition_value = condition_value.lower() in ['true', '1', 'yes', 'on']
			
			return block_content if condition_value else ''
		
		return if_pattern.sub(replace_if_block, content)
	
	def _process_loop_blocks(self, content: str, variables: Dict[str, Any]) -> str:
		"""Process loop blocks in template"""
		
		# Pattern for {% for item in list %}...{% endfor %}
		for_pattern = re.compile(r'\{\%\s*for\s+(\w+)\s+in\s+(\w+)\s*\%\}(.*?)\{\%\s*endfor\s*\%\}', re.DOTALL)
		
		def replace_for_block(match):
			item_var = match.group(1)
			list_var = match.group(2)
			block_content = match.group(3)
			
			# Get list from variables
			list_value = variables.get(list_var, [])
			if not isinstance(list_value, list):
				return ''
			
			# Process each item
			result = []
			for item in list_value:
				item_content = block_content.replace(f'{{{{{item_var}}}}}', str(item))
				result.append(item_content)
			
			return ''.join(result)
		
		return for_pattern.sub(replace_for_block, content)
	
	def _generate_additional_files(self, config: ProjectConfig, project_path: Path) -> List[str]:
		"""Generate additional standard files"""
		additional_files = []
		
		# Generate .gitignore
		gitignore_content = self._generate_gitignore(config)
		gitignore_path = project_path / '.gitignore'
		with open(gitignore_path, 'w') as f:
			f.write(gitignore_content)
		additional_files.append('.gitignore')
		
		# Generate LICENSE file
		if config.license != 'NONE':
			license_content = self._generate_license(config)
			license_path = project_path / 'LICENSE'
			with open(license_path, 'w') as f:
				f.write(license_content)
			additional_files.append('LICENSE')
		
		# Generate Docker files if enabled
		if config.enable_docker:
			docker_files = self._generate_docker_files(config, project_path)
			additional_files.extend(docker_files)
		
		# Generate test files if enabled
		if config.enable_testing:
			test_files = self._generate_test_files(config, project_path)
			additional_files.extend(test_files)
		
		# Generate project configuration
		apg_config = self._generate_apg_config(config)
		apg_config_path = project_path / 'apg.json'
		with open(apg_config_path, 'w') as f:
			json.dump(apg_config, f, indent=2)
		additional_files.append('apg.json')
		
		return additional_files
	
	def _generate_gitignore(self, config: ProjectConfig) -> str:
		"""Generate .gitignore file"""
		return '''# APG Generated Project .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Flask
instance/
.webassets-cache

# Database
*.db
*.sqlite
*.sqlite3

# APG Generated Files
generated/
.apg_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Environment Variables
.env
.env.local

# Testing
.coverage
.pytest_cache/
htmlcov/

# Documentation
docs/_build/
'''

	def _generate_license(self, config: ProjectConfig) -> str:
		"""Generate LICENSE file"""
		year = datetime.now().year
		
		if config.license.upper() == 'MIT':
			return f'''MIT License

Copyright (c) {year} {config.author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
		
		elif config.license.upper() == 'APACHE':
			return f'''Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright {year} {config.author}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
		
		else:
			return f'''All rights reserved.

Copyright (c) {year} {config.author}
'''
	
	def _generate_docker_files(self, config: ProjectConfig, project_path: Path) -> List[str]:
		"""Generate Docker files"""
		files = []
		
		# Dockerfile
		dockerfile_content = f'''FROM python:{config.python_version}-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create generated code directory
RUN mkdir -p generated

# Compile APG source
RUN apg compile main.apg

EXPOSE 8080

CMD ["python", "app.py"]
'''
		
		dockerfile_path = project_path / 'Dockerfile'
		with open(dockerfile_path, 'w') as f:
			f.write(dockerfile_content)
		files.append('Dockerfile')
		
		# docker-compose.yml
		if config.template_type in [TemplateType.MICROSERVICES, TemplateType.E_COMMERCE]:
			compose_content = f'''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - FLASK_ENV=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/{config.name}
    depends_on:
      - db
    volumes:
      - .:/app

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: {config.name}
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
'''
			
			compose_path = project_path / 'docker-compose.yml'
			with open(compose_path, 'w') as f:
				f.write(compose_content)
			files.append('docker-compose.yml')
		
		return files
	
	def _generate_test_files(self, config: ProjectConfig, project_path: Path) -> List[str]:
		"""Generate test files"""
		files = []
		
		# Create tests directory
		tests_dir = project_path / 'tests'
		tests_dir.mkdir(exist_ok=True)
		
		# Test configuration
		test_config = '''"""
Test Configuration
==================

Configuration for APG application tests.
"""

import os
import tempfile
from pathlib import Path

# Test database (use in-memory SQLite)
TESTING = True
SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Disable authentication for tests
AUTH_TYPE = None

# Test-specific settings
WTF_CSRF_ENABLED = False
SECRET_KEY = 'test-secret-key'
'''
		
		with open(tests_dir / 'test_config.py', 'w') as f:
			f.write(test_config)
		files.append('tests/test_config.py')
		
		# Base test case
		base_test = f'''"""
Base Test Case
==============

Base test case for {config.name} tests.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from compiler.compiler import APGCompiler, CodeGenConfig


class BaseTestCase(unittest.TestCase):
	"""Base test case for APG application tests"""
	
	def setUp(self):
		"""Set up test environment"""
		self.config = CodeGenConfig(
			target_language="flask-appbuilder",
			output_directory="test_output",
			generate_tests=True,
			include_runtime=True
		)
		
		self.compiler = APGCompiler(self.config)
	
	def tearDown(self):
		"""Clean up test environment"""
		pass
	
	def compile_apg_source(self, source: str, module_name: str = "test_module"):
		"""Helper method to compile APG source"""
		result = self.compiler.compile_string(source, module_name)
		self.assertTrue(result.success, f"Compilation failed: {{result.errors}}")
		return result


if __name__ == '__main__':
	unittest.main()
'''
		
		with open(tests_dir / 'base_test.py', 'w') as f:
			f.write(base_test)
		files.append('tests/base_test.py')
		
		# Agent tests
		agent_test = '''"""
Agent Tests
===========

Tests for APG agent functionality.
"""

import unittest
from tests.base_test import BaseTestCase


class TestAgents(BaseTestCase):
	"""Test agent functionality"""
	
	def test_basic_agent_compilation(self):
		"""Test basic agent compiles successfully"""
		apg_source = """
		module test_agent version 1.0.0 {
			description: "Test agent module";
		}
		
		agent TestAgent {
			name: str = "Test Agent";
			counter: int = 0;
			
			increment: () -> int = {
				counter = counter + 1;
				return counter;
			};
		}
		"""
		
		result = self.compile_apg_source(apg_source)
		self.assertIn('app.py', result.generated_files)
	
	def test_agent_methods(self):
		"""Test agent methods are generated correctly"""
		apg_source = """
		module test_methods version 1.0.0 {
			description: "Test agent methods";
		}
		
		agent MethodTestAgent {
			value: int = 0;
			
			set_value: (new_value: int) -> bool = {
				value = new_value;
				return true;
			};
			
			get_value: () -> int = {
				return value;
			};
		}
		"""
		
		result = self.compile_apg_source(apg_source)
		
		# Check that methods are in generated code
		app_content = result.generated_files.get('app.py', '')
		self.assertIn('def set_value_api(self)', app_content)
		self.assertIn('def get_value_api(self)', app_content)


if __name__ == '__main__':
	unittest.main()
'''
		
		with open(tests_dir / 'test_agents.py', 'w') as f:
			f.write(agent_test)
		files.append('tests/test_agents.py')
		
		# Test runner
		test_runner = '''"""
Test Runner
===========

Run all tests for the APG application.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
	"""Run all tests"""
	# Discover and run tests
	loader = unittest.TestLoader()
	start_dir = Path(__file__).parent
	suite = loader.discover(start_dir, pattern='test_*.py')
	
	runner = unittest.TextTestRunner(verbosity=2)
	result = runner.run(suite)
	
	return result.wasSuccessful()


if __name__ == '__main__':
	success = run_all_tests()
	sys.exit(0 if success else 1)
'''
		
		with open(tests_dir / 'run_tests.py', 'w') as f:
			f.write(test_runner)
		files.append('tests/run_tests.py')
		
		return files
	
	def _generate_apg_config(self, config: ProjectConfig) -> Dict[str, Any]:
		"""Generate APG project configuration"""
		return {
			'name': config.name,
			'version': config.version,
			'description': config.description,
			'author': config.author,
			'license': config.license,
			'template': config.template_type.value,
			'target_framework': config.target_framework,
			'python_version': config.python_version,
			'features': {
				'authentication': config.enable_authentication,
				'api': config.enable_api,
				'database': config.enable_database,
				'testing': config.enable_testing,
				'docker': config.enable_docker,
				'ai_features': config.enable_ai_features,
				'real_time': config.enable_real_time,
				'async': config.use_async
			},
			'build': {
				'source_file': 'main.apg',
				'output_directory': 'generated',
				'include_runtime': True,
				'generate_docs': config.generate_docs
			},
			'dependencies': config.custom_dependencies,
			'created_at': datetime.now().isoformat()
		}
	
	def _create_project_metadata(self, config: ProjectConfig, project_path: Path):
		"""Create project metadata files"""
		
		# Create .apg directory for metadata
		apg_dir = project_path / '.apg'
		apg_dir.mkdir(exist_ok=True)
		
		# Store project configuration
		config_dict = asdict(config)
		config_dict['template_type'] = config.template_type.value
		config_dict['output_directory'] = str(config.output_directory)
		config_dict['created_at'] = datetime.now().isoformat()
		
		with open(apg_dir / 'project.json', 'w') as f:
			json.dump(config_dict, f, indent=2)
	
	def _initialize_git_repo(self, project_path: Path):
		"""Initialize git repository"""
		import subprocess
		
		try:
			# Initialize git repo
			subprocess.run(['git', 'init'], cwd=project_path, check=True)
			
			# Add initial commit
			subprocess.run(['git', 'add', '.'], cwd=project_path, check=True)
			subprocess.run(['git', 'commit', '-m', 'Initial commit from APG template'], 
						 cwd=project_path, check=True)
			
		except subprocess.CalledProcessError:
			# Git not available or other error, skip
			pass
	
	def update_project(self, project_path: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
		"""Update existing APG project"""
		
		# Load existing project configuration
		apg_dir = project_path / '.apg'
		config_file = apg_dir / 'project.json'
		
		if not config_file.exists():
			return {
				'success': False,
				'errors': ['Project configuration not found. Not an APG project?']
			}
		
		try:
			with open(config_file, 'r') as f:
				project_config = json.load(f)
			
			# Apply updates
			project_config.update(updates)
			project_config['updated_at'] = datetime.now().isoformat()
			
			# Save updated configuration
			with open(config_file, 'w') as f:
				json.dump(project_config, f, indent=2)
			
			return {
				'success': True,
				'errors': [],
				'updated_config': project_config
			}
			
		except Exception as e:
			return {
				'success': False,
				'errors': [f'Error updating project: {str(e)}']
			}
	
	def get_project_info(self, project_path: Path) -> Optional[Dict[str, Any]]:
		"""Get information about APG project"""
		
		apg_dir = project_path / '.apg'
		config_file = apg_dir / 'project.json'
		
		if not config_file.exists():
			return None
		
		try:
			with open(config_file, 'r') as f:
				return json.load(f)
		except Exception:
			return None