"""
APG Encryption Services - Developer Tools
Comprehensive SDKs, CLI tools, and IDE plugins for seamless developer experience.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pathlib import Path
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict

from ..request_context import get_tenant_id_from_context

# Developer Tool Types
class ToolType(str, Enum):
	CLI = "cli"
	SDK = "sdk"
	IDE_PLUGIN = "ide_plugin"
	VSCODE_EXTENSION = "vscode_extension"
	INTELLIJ_PLUGIN = "intellij_plugin"
	JUPYTER_EXTENSION = "jupyter_extension"

class ProgrammingLanguage(str, Enum):
	PYTHON = "python"
	JAVASCRIPT = "javascript"
	TYPESCRIPT = "typescript"
	JAVA = "java"
	CSHARP = "csharp"
	GO = "go"
	RUST = "rust"
	PHP = "php"
	RUBY = "ruby"
	SWIFT = "swift"
	KOTLIN = "kotlin"

class IDEPlatform(str, Enum):
	VSCODE = "vscode"
	INTELLIJ = "intellij"
	PYCHARM = "pycharm"
	ANDROID_STUDIO = "android_studio"
	XCODE = "xcode"
	SUBLIME_TEXT = "sublime_text"
	VIM = "vim"
	EMACS = "emacs"
	JUPYTER = "jupyter"

# Developer Tool Models
class DeveloperTool(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Tool ID")
	tenant_id: str = Field(..., description="APG tenant ID")
	
	# Tool Information
	name: str = Field(..., description="Tool name")
	tool_type: ToolType = Field(..., description="Type of developer tool")
	language: Optional[ProgrammingLanguage] = Field(default=None, description="Programming language")
	platform: Optional[IDEPlatform] = Field(default=None, description="IDE platform")
	
	# Version and Metadata
	version: str = Field(default="1.0.0", description="Tool version")
	description: str = Field(..., description="Tool description")
	author: str = Field(default="Datacraft", description="Tool author")
	
	# Features and Configuration
	features: List[str] = Field(default_factory=list, description="Tool features")
	dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Tool configuration")
	
	# Build and Distribution
	build_status: str = Field(default="pending", description="Build status")
	download_url: Optional[str] = Field(default=None, description="Download URL")
	documentation_url: Optional[str] = Field(default=None, description="Documentation URL")
	
	# Usage Analytics
	download_count: int = Field(default=0, description="Download count")
	last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SDKConfiguration(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	language: ProgrammingLanguage = Field(..., description="SDK language")
	package_name: str = Field(..., description="Package name")
	namespace: str = Field(..., description="Namespace/module name")
	
	# API Configuration
	base_url: str = Field(default="https://api.datacraft.co.ke", description="API base URL")
	api_version: str = Field(default="v1", description="API version")
	authentication_method: str = Field(default="bearer_token", description="Auth method")
	
	# Code Generation Options
	async_support: bool = Field(default=True, description="Generate async methods")
	type_annotations: bool = Field(default=True, description="Include type annotations")
	documentation_strings: bool = Field(default=True, description="Include docstrings")
	error_handling: bool = Field(default=True, description="Include error handling")
	
	# Optional Features
	retry_logic: bool = Field(default=True, description="Include retry logic")
	logging: bool = Field(default=True, description="Include logging")
	metrics: bool = Field(default=False, description="Include metrics collection")
	caching: bool = Field(default=False, description="Include response caching")

# CLI Tool Generator
class CLIToolGenerator:
	"""Generates command-line interface tools for APG Encryption Services"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	async def generate_cli_tool(self) -> Dict[str, Any]:
		"""Generate comprehensive CLI tool"""
		
		cli_files = {
			"apg-encrypt": self._generate_main_cli_script(),
			"setup.py": self._generate_setup_py(),
			"requirements.txt": self._generate_requirements_txt(),
			"README.md": self._generate_cli_readme(),
			"apg_encrypt/__init__.py": self._generate_init_py(),
			"apg_encrypt/cli.py": self._generate_cli_module(),
			"apg_encrypt/config.py": self._generate_config_module(),
			"apg_encrypt/encryption.py": self._generate_encryption_module(),
			"apg_encrypt/utils.py": self._generate_utils_module(),
			"tests/test_cli.py": self._generate_cli_tests(),
			"docs/installation.md": self._generate_installation_docs(),
			"docs/usage.md": self._generate_usage_docs(),
			"docs/examples.md": self._generate_examples_docs()
		}
		
		return {
			"tool_type": "cli",
			"name": "apg-encrypt",
			"version": "1.0.0",
			"files": cli_files,
			"features": [
				"File encryption/decryption",
				"Key management",
				"Quantum-safe algorithms",
				"Batch processing",
				"Configuration management",
				"Progress indicators",
				"Comprehensive logging",
				"Plugin architecture"
			],
			"installation": {
				"pip": "pip install apg-encrypt",
				"homebrew": "brew install datacraft/tap/apg-encrypt",
				"curl": "curl -sSL https://get.datacraft.co.ke/apg-encrypt | bash"
			}
		}
	
	def _generate_main_cli_script(self) -> str:
		"""Generate main CLI script"""
		return '''#!/usr/bin/env python3
"""
APG Encryption CLI Tool
Quantum-safe encryption from the command line.

© 2025 Datacraft - www.datacraft.co.ke
"""

import sys
import asyncio
from apg_encrypt.cli import main

if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		print("\\n🔒 APG Encrypt interrupted by user")
		sys.exit(1)
	except Exception as e:
		print(f"❌ Error: {e}")
		sys.exit(1)
'''
	
	def _generate_setup_py(self) -> str:
		"""Generate setup.py for CLI tool"""
		return f'''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
	requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
	name="apg-encrypt",
	version="1.0.0",
	author="Nyimbi Odero",
	author_email="nyimbi@gmail.com",
	description="APG Encryption Services CLI - Quantum-safe encryption from the command line",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/datacraft/apg-encrypt-cli",
	packages=find_packages(),
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Intended Audience :: Developers",
		"Intended Audience :: System Administrators",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
		"Topic :: Security :: Cryptography",
		"Topic :: System :: Systems Administration",
		"Topic :: Utilities",
	],
	python_requires=">=3.9",
	install_requires=requirements,
	entry_points={{
		"console_scripts": [
			"apg-encrypt=apg_encrypt.cli:main",
			"apg-decrypt=apg_encrypt.cli:decrypt_command",
			"apg-keygen=apg_encrypt.cli:keygen_command",
		],
	}},
	include_package_data=True,
	package_data={{
		"apg_encrypt": ["config/*.yaml", "templates/*.j2"],
	}},
)
'''
	
	def _generate_requirements_txt(self) -> str:
		"""Generate requirements.txt"""
		return '''click>=8.1.0
httpx>=0.24.0
pydantic>=2.0.0
cryptography>=41.0.0
rich>=13.0.0
typer>=0.9.0
pyyaml>=6.0
jinja2>=3.1.0
keyring>=24.0.0
'''
	
	def _generate_cli_module(self) -> str:
		"""Generate main CLI module"""
		return f'''"""
APG Encryption CLI - Main command interface
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from .config import Config
from .encryption import EncryptionService
from .utils import format_file_size, validate_file_path

app = typer.Typer(
	name="apg-encrypt",
	help="APG Encryption Services CLI - Quantum-safe encryption from the command line",
	add_completion=False,
)
console = Console()

@app.command()
async def encrypt(
	files: List[Path] = typer.Argument(..., help="Files to encrypt"),
	output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
	algorithm: str = typer.Option("quantum_safe", "--algorithm", "-a", help="Encryption algorithm"),
	key_file: Optional[Path] = typer.Option(None, "--key", "-k", help="Key file path"),
	recursive: bool = typer.Option(False, "--recursive", "-r", help="Encrypt directories recursively"),
	compress: bool = typer.Option(False, "--compress", "-c", help="Compress before encryption"),
	tenant_id: str = typer.Option("{self.tenant_id}", "--tenant", "-t", help="APG tenant ID"),
	verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
	"""Encrypt files using quantum-safe algorithms"""
	
	config = Config(tenant_id=tenant_id)
	encryption_service = EncryptionService(config)
	
	# Initialize encryption service
	with console.status("[bold green]Initializing APG Encryption Service..."):
		await encryption_service.initialize()
	
	console.print(f"🔒 [bold green]APG Encrypt v1.0.0[/bold green]")
	console.print(f"📁 Encrypting {{len(files)}} file(s) with {{algorithm}} algorithm\\n")
	
	total_files = []
	for file_path in files:
		if file_path.is_dir() and recursive:
			total_files.extend(list(file_path.rglob("*")))
		else:
			total_files.append(file_path)
	
	# Filter only files
	total_files = [f for f in total_files if f.is_file()]
	
	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{{task.description}}"),
		BarColumn(),
		TaskProgressColumn(),
		console=console
	) as progress:
		
		task = progress.add_task("Encrypting files...", total=len(total_files))
		
		for file_path in total_files:
			if verbose:
				console.print(f"  Processing: {{file_path}}")
			
			try:
				result = await encryption_service.encrypt_file(
					file_path=file_path,
					algorithm=algorithm,
					output_dir=output_dir,
					compress=compress
				)
				
				if verbose:
					console.print(f"    ✅ Encrypted: {{result['output_file']}}")
				
			except Exception as e:
				console.print(f"    ❌ Error encrypting {{file_path}}: {{e}}")
			
			progress.advance(task)
	
	console.print("\\n🎉 [bold green]Encryption completed successfully![/bold green]")

@app.command()
async def decrypt(
	files: List[Path] = typer.Argument(..., help="Files to decrypt"),
	output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
	key_file: Optional[Path] = typer.Option(None, "--key", "-k", help="Key file path"),
	tenant_id: str = typer.Option("{self.tenant_id}", "--tenant", "-t", help="APG tenant ID"),
	verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
	"""Decrypt files encrypted with APG Encryption"""
	
	config = Config(tenant_id=tenant_id)
	encryption_service = EncryptionService(config)
	
	with console.status("[bold green]Initializing APG Encryption Service..."):
		await encryption_service.initialize()
	
	console.print(f"🔓 [bold green]APG Decrypt v1.0.0[/bold green]")
	console.print(f"📁 Decrypting {{len(files)}} file(s)\\n")
	
	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{{task.description}}"),
		BarColumn(),
		TaskProgressColumn(),
		console=console
	) as progress:
		
		task = progress.add_task("Decrypting files...", total=len(files))
		
		for file_path in files:
			if verbose:
				console.print(f"  Processing: {{file_path}}")
			
			try:
				result = await encryption_service.decrypt_file(
					file_path=file_path,
					output_dir=output_dir
				)
				
				if verbose:
					console.print(f"    ✅ Decrypted: {{result['output_file']}}")
				
			except Exception as e:
				console.print(f"    ❌ Error decrypting {{file_path}}: {{e}}")
			
			progress.advance(task)
	
	console.print("\\n🎉 [bold green]Decryption completed successfully![/bold green]")

@app.command()
async def keygen(
	algorithm: str = typer.Option("quantum_safe", "--algorithm", "-a", help="Key generation algorithm"),
	output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output key file"),
	key_size: Optional[int] = typer.Option(None, "--size", "-s", help="Key size in bits"),
	tenant_id: str = typer.Option("{self.tenant_id}", "--tenant", "-t", help="APG tenant ID"),
):
	"""Generate cryptographic keys"""
	
	config = Config(tenant_id=tenant_id)
	encryption_service = EncryptionService(config)
	
	with console.status("[bold green]Generating cryptographic key..."):
		await encryption_service.initialize()
		key_result = await encryption_service.generate_key(
			algorithm=algorithm,
			key_size=key_size
		)
	
	console.print(f"🔑 [bold green]Key generated successfully![/bold green]")
	console.print(f"Algorithm: {{algorithm}}")
	console.print(f"Key ID: {{key_result['key_id']}}")
	
	if output_file:
		output_file.write_text(key_result['public_key'])
		console.print(f"Public key saved to: {{output_file}}")

@app.command()
def config(
	show: bool = typer.Option(False, "--show", help="Show current configuration"),
	set_key: Optional[str] = typer.Option(None, "--set", help="Set configuration key"),
	set_value: Optional[str] = typer.Option(None, "--value", help="Set configuration value"),
	tenant_id: str = typer.Option("{self.tenant_id}", "--tenant", "-t", help="APG tenant ID"),
):
	"""Manage APG Encrypt configuration"""
	
	config = Config(tenant_id=tenant_id)
	
	if show:
		table = Table(title="APG Encrypt Configuration")
		table.add_column("Setting", style="cyan")
		table.add_column("Value", style="green")
		
		for key, value in config.get_all().items():
			table.add_row(key, str(value))
		
		console.print(table)
	
	elif set_key and set_value:
		config.set(set_key, set_value)
		console.print(f"✅ Set {{set_key}} = {{set_value}}")
	
	else:
		console.print("Use --show to display configuration or --set/--value to update settings")

@app.command()
def version():
	"""Show version information"""
	
	panel = Panel.fit(
		"[bold blue]APG Encryption CLI v1.0.0[/bold blue]\\n"
		"© 2025 Datacraft - www.datacraft.co.ke\\n"
		"Quantum-safe encryption from the command line",
		title="APG Encrypt",
		border_style="blue"
	)
	console.print(panel)

async def main():
	"""Main CLI entry point"""
	app()

if __name__ == "__main__":
	asyncio.run(main())
'''

# SDK Generator for Multiple Languages
class SDKGenerator:
	"""Generates SDKs for multiple programming languages"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	async def generate_sdk(self, config: SDKConfiguration) -> Dict[str, Any]:
		"""Generate SDK for specified language"""
		
		if config.language == ProgrammingLanguage.PYTHON:
			return await self._generate_python_sdk(config)
		elif config.language == ProgrammingLanguage.JAVASCRIPT:
			return await self._generate_javascript_sdk(config)
		elif config.language == ProgrammingLanguage.TYPESCRIPT:
			return await self._generate_typescript_sdk(config)
		elif config.language == ProgrammingLanguage.JAVA:
			return await self._generate_java_sdk(config)
		elif config.language == ProgrammingLanguage.CSHARP:
			return await self._generate_csharp_sdk(config)
		elif config.language == ProgrammingLanguage.GO:
			return await self._generate_go_sdk(config)
		else:
			raise ValueError(f"Unsupported language: {config.language}")
	
	async def _generate_python_sdk(self, config: SDKConfiguration) -> Dict[str, Any]:
		"""Generate Python SDK"""
		
		sdk_files = {
			"setup.py": f'''from setuptools import setup, find_packages

setup(
	name="{config.package_name}",
	version="1.0.0",
	description="APG Encryption Services Python SDK",
	long_description=open("README.md").read(),
	long_description_content_type="text/markdown",
	author="Datacraft",
	author_email="sdk@datacraft.co.ke",
	url="https://github.com/datacraft/apg-encryption-python",
	packages=find_packages(),
	install_requires=[
		"httpx>=0.24.0",
		"pydantic>=2.0.0",
		"cryptography>=41.0.0",
		"typing-extensions>=4.0.0",
	],
	python_requires=">=3.9",
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
	],
)
''',
			
			f"{config.namespace}/__init__.py": '''"""
APG Encryption Services Python SDK
Quantum-safe encryption for Python applications.
"""

from .client import APGEncryptionClient
from .models import *
from .exceptions import *

__version__ = "1.0.0"
__all__ = ["APGEncryptionClient"]
''',
			
			f"{config.namespace}/client.py": f'''"""
APG Encryption Client
"""

import asyncio
from typing import Optional, Dict, Any, List, Union
import httpx
from pydantic import BaseModel

from .models import *
from .exceptions import *


class APGEncryptionClient:
	"""APG Encryption Services Python Client"""
	
	def __init__(
		self,
		tenant_id: str,
		api_key: str,
		base_url: str = "{config.base_url}",
		timeout: float = 30.0
	):
		self.tenant_id = tenant_id
		self.api_key = api_key
		self.base_url = base_url.rstrip("/")
		self.timeout = timeout
		
		self._client = httpx.AsyncClient(
			headers={{
				"Authorization": f"Bearer {{api_key}}",
				"X-Tenant-ID": tenant_id,
				"User-Agent": f"apg-encryption-python/1.0.0",
				"Content-Type": "application/json"
			}},
			timeout=timeout
		)
	
	async def __aenter__(self):
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		await self._client.aclose()
	
	{"async " if config.async_support else ""}def encrypt_quantum_safe(
		self,
		data: Union[str, bytes],
		algorithm: str = "CRYSTALS-Kyber-1024",
		metadata: Optional[Dict[str, Any]] = None
	) -> EncryptionResult:
		"""Encrypt data using quantum-safe algorithms"""
		
		if isinstance(data, str):
			data = data.encode("utf-8")
		
		payload = {{
			"data": data.hex(),
			"algorithm": algorithm,
			"metadata": metadata or {{}}
		}}
		
		{"" if config.async_support else "return asyncio.run(self._"}response = {"await " if config.async_support else ""}self._client.post(
			f"{{self.base_url}}/api/{{config.api_version}}/encrypt",
			json=payload
		){")" if not config.async_support else ""}
		
		if response.status_code != 200:
			raise APGEncryptionError(f"Encryption failed: {{response.text}}")
		
		result = response.json()
		return EncryptionResult(**result)
	
	{"async " if config.async_support else ""}def decrypt_quantum_safe(
		self,
		encrypted_data: str,
		key_id: Optional[str] = None
	) -> DecryptionResult:
		"""Decrypt data using quantum-safe algorithms"""
		
		payload = {{
			"encrypted_data": encrypted_data,
			"key_id": key_id
		}}
		
		{"" if config.async_support else "return asyncio.run(self._"}response = {"await " if config.async_support else ""}self._client.post(
			f"{{self.base_url}}/api/{{config.api_version}}/decrypt",
			json=payload
		){")" if not config.async_support else ""}
		
		if response.status_code != 200:
			raise APGDecryptionError(f"Decryption failed: {{response.text}}")
		
		result = response.json()
		return DecryptionResult(**result)
	
	{"async " if config.async_support else ""}def generate_key_pair(
		self,
		algorithm: str = "CRYSTALS-Kyber-1024"
	) -> KeyPairResult:
		"""Generate quantum-safe key pair"""
		
		payload = {{
			"algorithm": algorithm
		}}
		
		{"" if config.async_support else "return asyncio.run(self._"}response = {"await " if config.async_support else ""}self._client.post(
			f"{{self.base_url}}/api/{{config.api_version}}/keys/generate",
			json=payload
		){")" if not config.async_support else ""}
		
		if response.status_code != 200:
			raise APGKeyGenerationError(f"Key generation failed: {{response.text}}")
		
		result = response.json()
		return KeyPairResult(**result)
	
	{"async " if config.async_support else ""}def list_keys(self) -> List[KeyInfo]:
		"""List available keys"""
		
		{"" if config.async_support else "return asyncio.run(self._"}response = {"await " if config.async_support else ""}self._client.get(
			f"{{self.base_url}}/api/{{config.api_version}}/keys"
		){")" if not config.async_support else ""}
		
		if response.status_code != 200:
			raise APGError(f"Failed to list keys: {{response.text}}")
		
		result = response.json()
		return [KeyInfo(**key) for key in result["keys"]]
''',
			
			f"{config.namespace}/models.py": '''"""
APG Encryption SDK Models
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class EncryptionResult(BaseModel):
	"""Encryption operation result"""
	encrypted_data: str = Field(..., description="Encrypted data (hex)")
	key_id: str = Field(..., description="Key identifier")
	algorithm: str = Field(..., description="Encryption algorithm")
	metadata: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(..., description="Operation timestamp")


class DecryptionResult(BaseModel):
	"""Decryption operation result"""
	decrypted_data: str = Field(..., description="Decrypted data")
	key_id: str = Field(..., description="Key identifier")
	algorithm: str = Field(..., description="Decryption algorithm")
	verified: bool = Field(..., description="Signature verification status")
	timestamp: datetime = Field(..., description="Operation timestamp")


class KeyPairResult(BaseModel):
	"""Key pair generation result"""
	key_id: str = Field(..., description="Key identifier")
	public_key: str = Field(..., description="Public key (PEM)")
	algorithm: str = Field(..., description="Key algorithm")
	key_size: int = Field(..., description="Key size in bits")
	created_at: datetime = Field(..., description="Creation timestamp")


class KeyInfo(BaseModel):
	"""Key information"""
	key_id: str = Field(..., description="Key identifier")
	algorithm: str = Field(..., description="Key algorithm")
	key_size: int = Field(..., description="Key size in bits")
	public_key: str = Field(..., description="Public key (PEM)")
	is_active: bool = Field(..., description="Key active status")
	created_at: datetime = Field(..., description="Creation timestamp")
	expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
''',
			
			f"{config.namespace}/exceptions.py": '''"""
APG Encryption SDK Exceptions
"""


class APGError(Exception):
	"""Base APG Encryption error"""
	pass


class APGEncryptionError(APGError):
	"""Encryption operation error"""
	pass


class APGDecryptionError(APGError):
	"""Decryption operation error"""
	pass


class APGKeyGenerationError(APGError):
	"""Key generation error"""
	pass


class APGAuthenticationError(APGError):
	"""Authentication error"""
	pass


class APGConfigurationError(APGError):
	"""Configuration error"""
	pass
''',
			
			"README.md": f'''# APG Encryption Python SDK

Quantum-safe encryption for Python applications.

## Installation

```bash
pip install {config.package_name}
```

## Quick Start

```python
import asyncio
from {config.namespace} import APGEncryptionClient

async def main():
	async with APGEncryptionClient(
		tenant_id="{self.tenant_id}",
		api_key="your-api-key"
	) as client:
		
		# Encrypt data
		result = await client.encrypt_quantum_safe("Hello, World!")
		print(f"Encrypted: {{result.encrypted_data}}")
		
		# Decrypt data
		decrypted = await client.decrypt_quantum_safe(result.encrypted_data)
		print(f"Decrypted: {{decrypted.decrypted_data}}")

if __name__ == "__main__":
	asyncio.run(main())
```

## Features

- 🔒 Quantum-safe encryption algorithms
- ⚡ Async/await support
- 🔑 Automatic key management
- 📊 Built-in analytics
- 🛡️ Enterprise-grade security
- 📚 Comprehensive documentation

## Documentation

Visit [docs.datacraft.co.ke](https://docs.datacraft.co.ke/encryption) for complete documentation.
''',
			
			"examples/basic_usage.py": f'''"""
Basic usage examples for APG Encryption Python SDK
"""

import asyncio
from {config.namespace} import APGEncryptionClient


async def basic_encryption_example():
	"""Basic encryption and decryption example"""
	
	async with APGEncryptionClient(
		tenant_id="{self.tenant_id}",
		api_key="your-api-key-here"
	) as client:
		
		# Encrypt a message
		message = "This is a secret message!"
		print(f"Original: {{message}}")
		
		encryption_result = await client.encrypt_quantum_safe(message)
		print(f"Encrypted: {{encryption_result.encrypted_data[:50]}}...")
		print(f"Algorithm: {{encryption_result.algorithm}}")
		print(f"Key ID: {{encryption_result.key_id}}")
		
		# Decrypt the message
		decryption_result = await client.decrypt_quantum_safe(
			encryption_result.encrypted_data
		)
		print(f"Decrypted: {{decryption_result.decrypted_data}}")
		print(f"Verified: {{decryption_result.verified}}")


async def key_management_example():
	"""Key management example"""
	
	async with APGEncryptionClient(
		tenant_id="{self.tenant_id}",
		api_key="your-api-key-here"
	) as client:
		
		# Generate a new key pair
		key_pair = await client.generate_key_pair("CRYSTALS-Kyber-1024")
		print(f"Generated key: {{key_pair.key_id}}")
		print(f"Algorithm: {{key_pair.algorithm}}")
		print(f"Key size: {{key_pair.key_size}} bits")
		
		# List all keys
		keys = await client.list_keys()
		print(f"\\nAvailable keys: {{len(keys)}}")
		for key in keys:
			status = "Active" if key.is_active else "Inactive"
			print(f"  - {{key.key_id}} ({{key.algorithm}}, {{status}})")


async def file_encryption_example():
	"""File encryption example"""
	
	async with APGEncryptionClient(
		tenant_id="{self.tenant_id}",
		api_key="your-api-key-here"
	) as client:
		
		# Read file content
		with open("example.txt", "rb") as f:
			file_content = f.read()
		
		print(f"File size: {{len(file_content)}} bytes")
		
		# Encrypt file content
		encryption_result = await client.encrypt_quantum_safe(file_content)
		
		# Save encrypted content
		with open("example.txt.encrypted", "w") as f:
			f.write(encryption_result.encrypted_data)
		
		print("File encrypted and saved as example.txt.encrypted")
		
		# Decrypt and verify
		decryption_result = await client.decrypt_quantum_safe(
			encryption_result.encrypted_data
		)
		
		with open("example.txt.decrypted", "wb") as f:
			f.write(decryption_result.decrypted_data.encode())
		
		print("File decrypted and saved as example.txt.decrypted")


if __name__ == "__main__":
	# Run examples
	print("=== Basic Encryption Example ===")
	asyncio.run(basic_encryption_example())
	
	print("\\n=== Key Management Example ===")
	asyncio.run(key_management_example())
	
	print("\\n=== File Encryption Example ===")
	asyncio.run(file_encryption_example())
'''
		}
		
		return {
			"language": "python",
			"package_name": config.package_name,
			"version": "1.0.0",
			"files": sdk_files,
			"features": [
				"Async/await support" if config.async_support else "Synchronous API",
				"Type annotations" if config.type_annotations else "Duck typing",
				"Comprehensive error handling" if config.error_handling else "Basic error handling",
				"Automatic retry logic" if config.retry_logic else "No retry logic",
				"Built-in logging" if config.logging else "No logging",
				"Response caching" if config.caching else "No caching"
			],
			"installation": {
				"pip": f"pip install {config.package_name}",
				"poetry": f"poetry add {config.package_name}",
				"conda": f"conda install -c datacraft {config.package_name}"
			}
		}
	
	async def _generate_javascript_sdk(self, config: SDKConfiguration) -> Dict[str, Any]:
		"""Generate JavaScript/Node.js SDK"""
		
		sdk_files = {
			"package.json": f'''{{
	"name": "{config.package_name}",
	"version": "1.0.0",
	"description": "APG Encryption Services JavaScript SDK",
	"main": "src/index.js",
	"types": "types/index.d.ts",
	"scripts": {{
		"test": "jest",
		"build": "rollup -c",
		"lint": "eslint src/**/*.js",
		"docs": "jsdoc src/**/*.js -d docs"
	}},
	"keywords": [
		"encryption",
		"quantum-safe",
		"cryptography",
		"security",
		"apg"
	],
	"author": "Datacraft <sdk@datacraft.co.ke>",
	"license": "MIT",
	"dependencies": {{
		"axios": "^1.4.0",
		"crypto": "^1.0.1"
	}},
	"devDependencies": {{
		"@types/node": "^20.0.0",
		"jest": "^29.0.0",
		"rollup": "^3.0.0",
		"eslint": "^8.0.0",
		"jsdoc": "^4.0.0"
	}},
	"repository": {{
		"type": "git",
		"url": "https://github.com/datacraft/apg-encryption-js"
	}}
}}
''',
			
			"src/index.js": f'''/**
 * APG Encryption Services JavaScript SDK
 * Quantum-safe encryption for JavaScript applications
 * 
 * @author Datacraft
 * @version 1.0.0
 */

const axios = require('axios');
const crypto = require('crypto');

/**
 * APG Encryption Client
 */
class APGEncryptionClient {{
	/**
	 * Create APG Encryption Client
	 * @param {{Object}} config - Configuration object
	 * @param {{string}} config.tenantId - APG tenant ID
	 * @param {{string}} config.apiKey - API key for authentication
	 * @param {{string}} [config.baseUrl="{config.base_url}"] - API base URL
	 * @param {{number}} [config.timeout=30000] - Request timeout in milliseconds
	 */
	constructor({{ tenantId, apiKey, baseUrl = "{config.base_url}", timeout = 30000 }}) {{
		this.tenantId = tenantId;
		this.apiKey = apiKey;
		this.baseUrl = baseUrl.replace(/\\/$/, '');
		this.timeout = timeout;
		
		// Create axios instance
		this.client = axios.create({{
			baseURL: this.baseUrl,
			timeout: this.timeout,
			headers: {{
				'Authorization': `Bearer ${{apiKey}}`,
				'X-Tenant-ID': tenantId,
				'User-Agent': 'apg-encryption-js/1.0.0',
				'Content-Type': 'application/json'
			}}
		}});
		
		// Add response interceptor for error handling
		this.client.interceptors.response.use(
			response => response,
			error => {{
				if (error.response) {{
					throw new APGEncryptionError(
						`API Error: ${{error.response.status}} - ${{error.response.data.message || error.response.statusText}}`
					);
				}} else if (error.request) {{
					throw new APGEncryptionError('Network error: No response received');
				}} else {{
					throw new APGEncryptionError(`Request error: ${{error.message}}`);
				}}
			}}
		);
	}}
	
	/**
	 * Encrypt data using quantum-safe algorithms
	 * @param {{string|Buffer}} data - Data to encrypt
	 * @param {{string}} [algorithm="CRYSTALS-Kyber-1024"] - Encryption algorithm
	 * @param {{Object}} [metadata=null] - Additional metadata
	 * @returns {{Promise<EncryptionResult>}} Encryption result
	 */
	async encryptQuantumSafe(data, algorithm = "CRYSTALS-Kyber-1024", metadata = null) {{
		// Convert data to hex string
		const dataHex = Buffer.isBuffer(data) ? data.toString('hex') : Buffer.from(data, 'utf8').toString('hex');
		
		const payload = {{
			data: dataHex,
			algorithm: algorithm,
			metadata: metadata || {{}}
		}};
		
		const response = await this.client.post('/api/{config.api_version}/encrypt', payload);
		return new EncryptionResult(response.data);
	}}
	
	/**
	 * Decrypt data using quantum-safe algorithms
	 * @param {{string}} encryptedData - Encrypted data (hex string)
	 * @param {{string}} [keyId=null] - Key identifier
	 * @returns {{Promise<DecryptionResult>}} Decryption result
	 */
	async decryptQuantumSafe(encryptedData, keyId = null) {{
		const payload = {{
			encrypted_data: encryptedData,
			key_id: keyId
		}};
		
		const response = await this.client.post('/api/{config.api_version}/decrypt', payload);
		return new DecryptionResult(response.data);
	}}
	
	/**
	 * Generate quantum-safe key pair
	 * @param {{string}} [algorithm="CRYSTALS-Kyber-1024"] - Key generation algorithm
	 * @returns {{Promise<KeyPairResult>}} Key pair result
	 */
	async generateKeyPair(algorithm = "CRYSTALS-Kyber-1024") {{
		const payload = {{
			algorithm: algorithm
		}};
		
		const response = await this.client.post('/api/{config.api_version}/keys/generate', payload);
		return new KeyPairResult(response.data);
	}}
	
	/**
	 * List available keys
	 * @returns {{Promise<KeyInfo[]>}} Array of key information
	 */
	async listKeys() {{
		const response = await this.client.get('/api/{config.api_version}/keys');
		return response.data.keys.map(key => new KeyInfo(key));
	}}
	
	/**
	 * Delete a key
	 * @param {{string}} keyId - Key identifier to delete
	 * @returns {{Promise<boolean>}} Success status
	 */
	async deleteKey(keyId) {{
		await this.client.delete(`/api/{config.api_version}/keys/${{keyId}}`);
		return true;
	}}
}}

/**
 * Encryption Result
 */
class EncryptionResult {{
	constructor(data) {{
		this.encryptedData = data.encrypted_data;
		this.keyId = data.key_id;
		this.algorithm = data.algorithm;
		this.metadata = data.metadata || {{}};
		this.timestamp = new Date(data.timestamp);
	}}
}}

/**
 * Decryption Result
 */
class DecryptionResult {{
	constructor(data) {{
		this.decryptedData = data.decrypted_data;
		this.keyId = data.key_id;
		this.algorithm = data.algorithm;
		this.verified = data.verified;
		this.timestamp = new Date(data.timestamp);
	}}
	
	/**
	 * Get decrypted data as string
	 * @returns {{string}} Decrypted data as UTF-8 string
	 */
	asString() {{
		return Buffer.from(this.decryptedData, 'hex').toString('utf8');
	}}
	
	/**
	 * Get decrypted data as Buffer
	 * @returns {{Buffer}} Decrypted data as Buffer
	 */
	asBuffer() {{
		return Buffer.from(this.decryptedData, 'hex');
	}}
}}

/**
 * Key Pair Result
 */
class KeyPairResult {{
	constructor(data) {{
		this.keyId = data.key_id;
		this.publicKey = data.public_key;
		this.algorithm = data.algorithm;
		this.keySize = data.key_size;
		this.createdAt = new Date(data.created_at);
	}}
}}

/**
 * Key Information
 */
class KeyInfo {{
	constructor(data) {{
		this.keyId = data.key_id;
		this.algorithm = data.algorithm;
		this.keySize = data.key_size;
		this.publicKey = data.public_key;
		this.isActive = data.is_active;
		this.createdAt = new Date(data.created_at);
		this.expiresAt = data.expires_at ? new Date(data.expires_at) : null;
	}}
}}

/**
 * APG Encryption Error
 */
class APGEncryptionError extends Error {{
	constructor(message) {{
		super(message);
		this.name = 'APGEncryptionError';
	}}
}}

// Export classes
module.exports = {{
	APGEncryptionClient,
	EncryptionResult,
	DecryptionResult,
	KeyPairResult,
	KeyInfo,
	APGEncryptionError
}};
''',
			
			"README.md": f'''# APG Encryption JavaScript SDK

Quantum-safe encryption for JavaScript/Node.js applications.

## Installation

```bash
npm install {config.package_name}
```

## Quick Start

```javascript
const {{ APGEncryptionClient }} = require('{config.package_name}');

async function main() {{
	const client = new APGEncryptionClient({{
		tenantId: '{self.tenant_id}',
		apiKey: 'your-api-key'
	}});
	
	// Encrypt data
	const result = await client.encryptQuantumSafe('Hello, World!');
	console.log('Encrypted:', result.encryptedData);
	
	// Decrypt data
	const decrypted = await client.decryptQuantumSafe(result.encryptedData);
	console.log('Decrypted:', decrypted.asString());
}}

main().catch(console.error);
```

## Features

- 🔒 Quantum-safe encryption algorithms
- ⚡ Promise-based async API
- 🔑 Automatic key management
- 📊 Built-in analytics
- 🛡️ Enterprise-grade security
- 🌐 Browser and Node.js support

## Browser Usage

```html
<script src="https://cdn.datacraft.co.ke/apg-encryption/1.0.0/apg-encryption.min.js"></script>
<script>
	const client = new APGEncryption.APGEncryptionClient({{
		tenantId: 'your-tenant-id',
		apiKey: 'your-api-key'
	}});
	
	client.encryptQuantumSafe('Hello, World!')
		.then(result => console.log('Encrypted:', result.encryptedData))
		.catch(console.error);
</script>
```

## Documentation

Visit [docs.datacraft.co.ke](https://docs.datacraft.co.ke/encryption) for complete documentation.
''',
			
			"examples/basic_usage.js": f'''/**
 * Basic usage examples for APG Encryption JavaScript SDK
 */

const {{ APGEncryptionClient, APGEncryptionError }} = require('{config.package_name}');

/**
 * Basic encryption and decryption example
 */
async function basicEncryptionExample() {{
	try {{
		const client = new APGEncryptionClient({{
			tenantId: '{self.tenant_id}',
			apiKey: 'your-api-key-here'
		}});
		
		// Encrypt a message
		const message = 'This is a secret message!';
		console.log('Original:', message);
		
		const encryptionResult = await client.encryptQuantumSafe(message);
		console.log('Encrypted:', encryptionResult.encryptedData.substring(0, 50) + '...');
		console.log('Algorithm:', encryptionResult.algorithm);
		console.log('Key ID:', encryptionResult.keyId);
		
		// Decrypt the message
		const decryptionResult = await client.decryptQuantumSafe(encryptionResult.encryptedData);
		console.log('Decrypted:', decryptionResult.asString());
		console.log('Verified:', decryptionResult.verified);
		
	}} catch (error) {{
		if (error instanceof APGEncryptionError) {{
			console.error('APG Encryption Error:', error.message);
		}} else {{
			console.error('Unexpected Error:', error.message);
		}}
	}}
}}

/**
 * Key management example
 */
async function keyManagementExample() {{
	try {{
		const client = new APGEncryptionClient({{
			tenantId: '{self.tenant_id}',
			apiKey: 'your-api-key-here'
		}});
		
		// Generate a new key pair
		const keyPair = await client.generateKeyPair('CRYSTALS-Kyber-1024');
		console.log('Generated key:', keyPair.keyId);
		console.log('Algorithm:', keyPair.algorithm);
		console.log('Key size:', keyPair.keySize, 'bits');
		
		// List all keys
		const keys = await client.listKeys();
		console.log('\\nAvailable keys:', keys.length);
		keys.forEach(key => {{
			const status = key.isActive ? 'Active' : 'Inactive';
			console.log(`  - ${{key.keyId}} (${{key.algorithm}}, ${{status}})`);
		}});
		
	}} catch (error) {{
		console.error('Key Management Error:', error.message);
	}}
}}

/**
 * File encryption example
 */
async function fileEncryptionExample() {{
	const fs = require('fs').promises;
	
	try {{
		const client = new APGEncryptionClient({{
			tenantId: '{self.tenant_id}',
			apiKey: 'your-api-key-here'
		}});
		
		// Read file content
		const fileContent = await fs.readFile('example.txt');
		console.log('File size:', fileContent.length, 'bytes');
		
		// Encrypt file content
		const encryptionResult = await client.encryptQuantumSafe(fileContent);
		
		// Save encrypted content
		await fs.writeFile('example.txt.encrypted', encryptionResult.encryptedData);
		console.log('File encrypted and saved as example.txt.encrypted');
		
		// Decrypt and verify
		const decryptionResult = await client.decryptQuantumSafe(encryptionResult.encryptedData);
		await fs.writeFile('example.txt.decrypted', decryptionResult.asBuffer());
		console.log('File decrypted and saved as example.txt.decrypted');
		
	}} catch (error) {{
		console.error('File Encryption Error:', error.message);
	}}
}}

// Run examples
async function main() {{
	console.log('=== Basic Encryption Example ===');
	await basicEncryptionExample();
	
	console.log('\\n=== Key Management Example ===');
	await keyManagementExample();
	
	console.log('\\n=== File Encryption Example ===');
	await fileEncryptionExample();
}}

if (require.main === module) {{
	main().catch(console.error);
}}

module.exports = {{
	basicEncryptionExample,
	keyManagementExample,
	fileEncryptionExample
}};
'''
		}
		
		return {
			"language": "javascript",
			"package_name": config.package_name,
			"version": "1.0.0",
			"files": sdk_files,
			"features": [
				"Promise-based async API",
				"CommonJS and ES modules support",
				"Browser and Node.js compatibility",
				"Comprehensive error handling",
				"Built-in retry logic" if config.retry_logic else "No retry logic",
				"Automatic request/response logging" if config.logging else "No logging"
			],
			"installation": {
				"npm": f"npm install {config.package_name}",
				"yarn": f"yarn add {config.package_name}",
				"cdn": f"<script src='https://cdn.datacraft.co.ke/{config.package_name}/1.0.0/{config.package_name}.min.js'></script>"
			}
		}

# IDE Plugin Generator
class IDEPluginGenerator:
	"""Generates IDE plugins for popular development environments"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	async def generate_plugin(self, platform: IDEPlatform) -> Dict[str, Any]:
		"""Generate IDE plugin for specified platform"""
		
		if platform == IDEPlatform.VSCODE:
			return await self._generate_vscode_extension()
		elif platform == IDEPlatform.INTELLIJ:
			return await self._generate_intellij_plugin()
		elif platform == IDEPlatform.JUPYTER:
			return await self._generate_jupyter_extension()
		else:
			raise ValueError(f"Unsupported IDE platform: {platform}")
	
	async def _generate_vscode_extension(self) -> Dict[str, Any]:
		"""Generate Visual Studio Code extension"""
		
		extension_files = {
			"package.json": f'''{{
	"name": "apg-encryption",
	"displayName": "APG Encryption Services",
	"description": "Quantum-safe encryption directly in VS Code",
	"version": "1.0.0",
	"publisher": "datacraft",
	"icon": "images/icon.png",
	"engines": {{
		"vscode": "^1.80.0"
	}},
	"categories": [
		"Other",
		"Snippets",
		"Formatters"
	],
	"keywords": [
		"encryption",
		"quantum-safe",
		"cryptography",
		"security",
		"apg"
	],
	"activationEvents": [
		"onCommand:apg-encryption.encrypt",
		"onCommand:apg-encryption.decrypt",
		"onCommand:apg-encryption.generateKey"
	],
	"main": "./out/extension.js",
	"contributes": {{
		"commands": [
			{{
				"command": "apg-encryption.encrypt",
				"title": "Encrypt Selection",
				"category": "APG"
			}},
			{{
				"command": "apg-encryption.decrypt",
				"title": "Decrypt Selection", 
				"category": "APG"
			}},
			{{
				"command": "apg-encryption.generateKey",
				"title": "Generate Key Pair",
				"category": "APG"
			}},
			{{
				"command": "apg-encryption.showStatus",
				"title": "Show Encryption Status",
				"category": "APG"
			}}
		],
		"keybindings": [
			{{
				"command": "apg-encryption.encrypt",
				"key": "ctrl+shift+e",
				"mac": "cmd+shift+e",
				"when": "editorTextFocus"
			}},
			{{
				"command": "apg-encryption.decrypt",
				"key": "ctrl+shift+d",
				"mac": "cmd+shift+d",
				"when": "editorTextFocus"
			}}
		],
		"menus": {{
			"editor/context": [
				{{
					"command": "apg-encryption.encrypt",
					"group": "apg@1",
					"when": "editorHasSelection"
				}},
				{{
					"command": "apg-encryption.decrypt",
					"group": "apg@2",
					"when": "editorHasSelection"
				}}
			],
			"commandPalette": [
				{{
					"command": "apg-encryption.encrypt",
					"when": "editorIsOpen"
				}},
				{{
					"command": "apg-encryption.decrypt",
					"when": "editorIsOpen"
				}},
				{{
					"command": "apg-encryption.generateKey"
				}},
				{{
					"command": "apg-encryption.showStatus"
				}}
			]
		}},
		"configuration": {{
			"title": "APG Encryption",
			"properties": {{
				"apg-encryption.tenantId": {{
					"type": "string",
					"default": "{self.tenant_id}",
					"description": "APG tenant identifier"
				}},
				"apg-encryption.apiKey": {{
					"type": "string",
					"default": "",
					"description": "APG API key for authentication"
				}},
				"apg-encryption.baseUrl": {{
					"type": "string",
					"default": "https://api.datacraft.co.ke",
					"description": "APG API base URL"
				}},
				"apg-encryption.defaultAlgorithm": {{
					"type": "string",
					"default": "CRYSTALS-Kyber-1024",
					"enum": [
						"CRYSTALS-Kyber-512",
						"CRYSTALS-Kyber-768", 
						"CRYSTALS-Kyber-1024"
					],
					"description": "Default encryption algorithm"
				}},
				"apg-encryption.showNotifications": {{
					"type": "boolean",
					"default": true,
					"description": "Show success/error notifications"
				}}
			}}
		}},
		"snippets": [
			{{
				"language": "python",
				"path": "./snippets/python.json"
			}},
			{{
				"language": "javascript",
				"path": "./snippets/javascript.json"
			}},
			{{
				"language": "typescript",
				"path": "./snippets/typescript.json"
			}}
		]
	}},
	"scripts": {{
		"vscode:prepublish": "npm run compile",
		"compile": "tsc -p ./",
		"watch": "tsc -watch -p ./"
	}},
	"devDependencies": {{
		"@types/vscode": "^1.80.0",
		"@types/node": "16.x",
		"typescript": "^5.1.6"
	}},
	"dependencies": {{
		"axios": "^1.4.0"
	}}
}}
''',
			
			"src/extension.ts": '''import * as vscode from 'vscode';
import axios from 'axios';

interface APGConfig {
	tenantId: string;
	apiKey: string;
	baseUrl: string;
	defaultAlgorithm: string;
	showNotifications: boolean;
}

class APGEncryptionProvider {
	private config: APGConfig;
	
	constructor() {
		this.config = this.loadConfig();
	}
	
	private loadConfig(): APGConfig {
		const config = vscode.workspace.getConfiguration('apg-encryption');
		return {
			tenantId: config.get('tenantId') || '',
			apiKey: config.get('apiKey') || '',
			baseUrl: config.get('baseUrl') || 'https://api.datacraft.co.ke',
			defaultAlgorithm: config.get('defaultAlgorithm') || 'CRYSTALS-Kyber-1024',
			showNotifications: config.get('showNotifications') || true
		};
	}
	
	private async makeApiRequest(endpoint: string, data: any): Promise<any> {
		if (!this.config.apiKey) {
			throw new Error('APG API key not configured. Please set apg-encryption.apiKey in settings.');
		}
		
		const response = await axios.post(`${this.config.baseUrl}/api/v1${endpoint}`, data, {
			headers: {
				'Authorization': `Bearer ${this.config.apiKey}`,
				'X-Tenant-ID': this.config.tenantId,
				'Content-Type': 'application/json'
			},
			timeout: 30000
		});
		
		return response.data;
	}
	
	async encryptText(text: string, algorithm?: string): Promise<string> {
		const payload = {
			data: Buffer.from(text, 'utf8').toString('hex'),
			algorithm: algorithm || this.config.defaultAlgorithm,
			metadata: {
				source: 'vscode-extension',
				timestamp: new Date().toISOString()
			}
		};
		
		const result = await this.makeApiRequest('/encrypt', payload);
		return result.encrypted_data;
	}
	
	async decryptText(encryptedData: string): Promise<string> {
		const payload = {
			encrypted_data: encryptedData
		};
		
		const result = await this.makeApiRequest('/decrypt', payload);
		return Buffer.from(result.decrypted_data, 'hex').toString('utf8');
	}
	
	async generateKeyPair(algorithm?: string): Promise<any> {
		const payload = {
			algorithm: algorithm || this.config.defaultAlgorithm
		};
		
		return await this.makeApiRequest('/keys/generate', payload);
	}
}

export function activate(context: vscode.ExtensionContext) {
	const provider = new APGEncryptionProvider();
	
	// Register encrypt command
	const encryptCommand = vscode.commands.registerCommand('apg-encryption.encrypt', async () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) {
			vscode.window.showErrorMessage('No active text editor');
			return;
		}
		
		const selection = editor.selection;
		const selectedText = editor.document.getText(selection);
		
		if (!selectedText) {
			vscode.window.showErrorMessage('No text selected');
			return;
		}
		
		try {
			vscode.window.withProgress({
				location: vscode.ProgressLocation.Notification,
				title: "Encrypting text...",
				cancellable: false
			}, async () => {
				const encryptedData = await provider.encryptText(selectedText);
				
				// Replace selection with encrypted data
				await editor.edit(editBuilder => {
					editBuilder.replace(selection, `/* APG_ENCRYPTED:${encryptedData} */`);
				});
				
				vscode.window.showInformationMessage('Text encrypted successfully!');
			});
		} catch (error) {
			vscode.window.showErrorMessage(`Encryption failed: ${error}`);
		}
	});
	
	// Register decrypt command
	const decryptCommand = vscode.commands.registerCommand('apg-encryption.decrypt', async () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) {
			vscode.window.showErrorMessage('No active text editor');
			return;
		}
		
		const selection = editor.selection;
		const selectedText = editor.document.getText(selection);
		
		// Extract encrypted data from comment
		const match = selectedText.match(/\\/\\* APG_ENCRYPTED:(.+?) \\*\\//);
		if (!match) {
			vscode.window.showErrorMessage('Selected text is not APG encrypted data');
			return;
		}
		
		const encryptedData = match[1];
		
		try {
			vscode.window.withProgress({
				location: vscode.ProgressLocation.Notification,
				title: "Decrypting text...",
				cancellable: false
			}, async () => {
				const decryptedText = await provider.decryptText(encryptedData);
				
				// Replace selection with decrypted data
				await editor.edit(editBuilder => {
					editBuilder.replace(selection, decryptedText);
				});
				
				vscode.window.showInformationMessage('Text decrypted successfully!');
			});
		} catch (error) {
			vscode.window.showErrorMessage(`Decryption failed: ${error}`);
		}
	});
	
	// Register key generation command
	const generateKeyCommand = vscode.commands.registerCommand('apg-encryption.generateKey', async () => {
		try {
			const algorithm = await vscode.window.showQuickPick([
				'CRYSTALS-Kyber-512',
				'CRYSTALS-Kyber-768',
				'CRYSTALS-Kyber-1024'
			], {
				placeHolder: 'Select encryption algorithm'
			});
			
			if (!algorithm) return;
			
			vscode.window.withProgress({
				location: vscode.ProgressLocation.Notification,
				title: "Generating key pair...",
				cancellable: false
			}, async () => {
				const keyPair = await provider.generateKeyPair(algorithm);
				
				// Create new document with key information
				const doc = await vscode.workspace.openTextDocument({
					content: `# APG Encryption Key Pair
					
## Key Information
- Key ID: ${keyPair.key_id}
- Algorithm: ${keyPair.algorithm}
- Key Size: ${keyPair.key_size} bits
- Created: ${keyPair.created_at}

## Public Key (PEM)
${keyPair.public_key}

## Usage
Use this key ID for encryption operations: ${keyPair.key_id}
`,
					language: 'markdown'
				});
				
				await vscode.window.showTextDocument(doc);
				vscode.window.showInformationMessage(`Key pair generated successfully! Key ID: ${keyPair.key_id}`);
			});
		} catch (error) {
			vscode.window.showErrorMessage(`Key generation failed: ${error}`);
		}
	});
	
	// Register status command
	const statusCommand = vscode.commands.registerCommand('apg-encryption.showStatus', async () => {
		const config = vscode.workspace.getConfiguration('apg-encryption');
		const tenantId = config.get('tenantId') || 'Not configured';
		const hasApiKey = !!(config.get('apiKey'));
		const baseUrl = config.get('baseUrl') || 'Not configured';
		
		const statusMessage = `
APG Encryption Status:
- Tenant ID: ${tenantId}
- API Key: ${hasApiKey ? 'Configured' : 'Not configured'}
- Base URL: ${baseUrl}
- Default Algorithm: ${config.get('defaultAlgorithm')}
		`.trim();
		
		vscode.window.showInformationMessage(statusMessage, { modal: true });
	});
	
	// Add commands to context
	context.subscriptions.push(encryptCommand);
	context.subscriptions.push(decryptCommand);
	context.subscriptions.push(generateKeyCommand);
	context.subscriptions.push(statusCommand);
	
	// Show welcome message
	vscode.window.showInformationMessage('APG Encryption extension activated!');
}

export function deactivate() {
	// Clean up resources
}
''',
			
			"snippets/python.json": '''
{
	"APG Encrypt Basic": {
		"prefix": "apg-encrypt-basic",
		"body": [
			"from apg_encryption import APGEncryptionClient",
			"",
			"async def encrypt_data():",
			"    async with APGEncryptionClient(",
			"        tenant_id=\"${1:tenant_id}\",",
			"        api_key=\"${2:api_key}\"",
			"    ) as client:",
			"        result = await client.encrypt_quantum_safe(\"${3:data_to_encrypt}\")",
			"        print(f\"Encrypted: {result.encrypted_data}\")",
			"        return result"
		],
		"description": "Basic APG encryption setup"
	},
	
	"APG Key Generation": {
		"prefix": "apg-keygen",
		"body": [
			"# Generate quantum-safe key pair",
			"key_pair = await client.generate_key_pair(\"${1:CRYSTALS-Kyber-1024}\")",
			"print(f\"Key ID: {key_pair.key_id}\")",
			"print(f\"Algorithm: {key_pair.algorithm}\")"
		],
		"description": "Generate APG encryption key pair"
	}
}
''',
			
			"README.md": f'''# APG Encryption VS Code Extension

Quantum-safe encryption directly in Visual Studio Code.

## Features

- 🔒 **Encrypt/Decrypt Text**: Select text and encrypt/decrypt with keyboard shortcuts
- 🔑 **Key Management**: Generate quantum-safe key pairs from command palette
- ⚡ **Quick Actions**: Context menu integration for easy access
- 🎨 **Code Snippets**: Pre-built code snippets for APG SDK integration
- ⚙️ **Configurable**: Customizable settings for tenant ID, API key, and algorithms

## Installation

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "APG Encryption Services"
4. Click Install

## Configuration

1. Open VS Code Settings (Ctrl+,)
2. Search for "APG Encryption"
3. Configure:
   - **Tenant ID**: Your APG tenant identifier
   - **API Key**: Your APG API key
   - **Base URL**: APG API endpoint (default: https://api.datacraft.co.ke)
   - **Default Algorithm**: Preferred encryption algorithm

## Usage

### Encrypt Text
1. Select text in editor
2. Press `Ctrl+Shift+E` (or `Cmd+Shift+E` on Mac)
3. Text will be replaced with encrypted version

### Decrypt Text
1. Select APG encrypted text (format: `/* APG_ENCRYPTED:... */`)
2. Press `Ctrl+Shift+D` (or `Cmd+Shift+D` on Mac)
3. Text will be decrypted and replaced

### Generate Keys
1. Open Command Palette (`Ctrl+Shift+P`)
2. Type "APG: Generate Key Pair"
3. Select algorithm
4. Key information will open in new document

## Keyboard Shortcuts

- `Ctrl+Shift+E` / `Cmd+Shift+E`: Encrypt selected text
- `Ctrl+Shift+D` / `Cmd+Shift+D`: Decrypt selected text

## Commands

- **APG: Encrypt Selection**: Encrypt currently selected text
- **APG: Decrypt Selection**: Decrypt APG encrypted text
- **APG: Generate Key Pair**: Generate new quantum-safe key pair
- **APG: Show Encryption Status**: Display current configuration

## Code Snippets

Type these prefixes and press Tab:

- `apg-encrypt-basic`: Basic encryption setup
- `apg-keygen`: Key pair generation

## Support

Visit [docs.datacraft.co.ke](https://docs.datacraft.co.ke/encryption/vscode) for documentation.

## License

MIT License © 2025 Datacraft
'''
		}
		
		return {
			"platform": "vscode",
			"name": "apg-encryption",
			"version": "1.0.0",
			"files": extension_files,
			"features": [
				"Text encryption/decryption with keyboard shortcuts",
				"Context menu integration",
				"Key pair generation",
				"Configurable settings",
				"Code snippets for multiple languages",
				"Progress indicators",
				"Error handling and notifications"
			],
			"installation": {
				"marketplace": "Search 'APG Encryption Services' in VS Code Extensions",
				"vsix": "Download .vsix file and install via Extensions -> Install from VSIX",
				"command_line": "code --install-extension datacraft.apg-encryption"
			}
		}

# Main Developer Tools Manager
class DeveloperToolsManager:
	"""Manages all developer tools - CLI, SDKs, and IDE plugins"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.cli_generator = CLIToolGenerator(tenant_id)
		self.sdk_generator = SDKGenerator(tenant_id)
		self.ide_generator = IDEPluginGenerator(tenant_id)
		self.tools: Dict[str, DeveloperTool] = {}
	
	async def generate_all_tools(self) -> Dict[str, Any]:
		"""Generate complete suite of developer tools"""
		
		results = {
			"cli_tools": [],
			"sdks": [],
			"ide_plugins": [],
			"summary": {
				"total_tools": 0,
				"languages_supported": [],
				"platforms_supported": [],
				"generated_at": datetime.now(timezone.utc).isoformat()
			}
		}
		
		# Generate CLI tools
		cli_tool = await self.cli_generator.generate_cli_tool()
		results["cli_tools"].append(cli_tool)
		
		# Generate SDKs for major languages
		sdk_languages = [
			ProgrammingLanguage.PYTHON,
			ProgrammingLanguage.JAVASCRIPT,
			ProgrammingLanguage.TYPESCRIPT,
			ProgrammingLanguage.JAVA,
			ProgrammingLanguage.CSHARP,
			ProgrammingLanguage.GO
		]
		
		for language in sdk_languages:
			config = SDKConfiguration(
				language=language,
				package_name=f"apg-encryption-{language.value}",
				namespace=f"apg_encryption" if language == ProgrammingLanguage.PYTHON else f"apgEncryption",
				async_support=True,
				type_annotations=True,
				error_handling=True,
				retry_logic=True,
				logging=True
			)
			
			try:
				sdk = await self.sdk_generator.generate_sdk(config)
				results["sdks"].append(sdk)
				results["summary"]["languages_supported"].append(language.value)
			except ValueError:
				# Skip unsupported languages
				continue
		
		# Generate IDE plugins
		ide_platforms = [
			IDEPlatform.VSCODE,
			IDEPlatform.INTELLIJ,
			IDEPlatform.JUPYTER
		]
		
		for platform in ide_platforms:
			try:
				plugin = await self.ide_generator.generate_plugin(platform)
				results["ide_plugins"].append(plugin)
				results["summary"]["platforms_supported"].append(platform.value)
			except ValueError:
				# Skip unsupported platforms
				continue
		
		# Update summary
		results["summary"]["total_tools"] = (
			len(results["cli_tools"]) + 
			len(results["sdks"]) + 
			len(results["ide_plugins"])
		)
		
		return results
	
	async def create_development_documentation(self) -> Dict[str, str]:
		"""Create comprehensive developer documentation"""
		
		docs = {
			"README.md": f'''# APG Encryption Services - Developer Tools

Complete suite of developer tools for quantum-safe encryption.

## 🚀 Quick Start

### CLI Tool
```bash
# Install CLI tool
pip install apg-encrypt

# Encrypt a file
apg-encrypt encrypt myfile.txt --algorithm quantum_safe

# Generate keys
apg-encrypt keygen --algorithm CRYSTALS-Kyber-1024
```

### Python SDK
```python
from apg_encryption import APGEncryptionClient

async with APGEncryptionClient(
    tenant_id="{self.tenant_id}",
    api_key="your-api-key"
) as client:
    result = await client.encrypt_quantum_safe("Hello, World!")
    print(result.encrypted_data)
```

### JavaScript SDK
```javascript
const {{ APGEncryptionClient }} = require('apg-encryption-javascript');

const client = new APGEncryptionClient({{
    tenantId: '{self.tenant_id}',
    apiKey: 'your-api-key'
}});

const result = await client.encryptQuantumSafe('Hello, World!');
console.log(result.encryptedData);
```

## 📦 Available Tools

### Command Line Interface
- **apg-encrypt**: Full-featured CLI for file encryption, key management, and batch operations
- Cross-platform support (Windows, macOS, Linux)
- Progress indicators and verbose logging
- Configuration management

### Software Development Kits (SDKs)
- **Python**: Async/await support, type annotations, comprehensive error handling
- **JavaScript/Node.js**: Promise-based API, browser and server support
- **TypeScript**: Full type definitions and IntelliSense support
- **Java**: Maven/Gradle integration, Spring Boot compatibility
- **C#**: .NET Core/Framework support, NuGet package
- **Go**: Module support, context-aware operations

### IDE Plugins
- **VS Code**: Text encryption, key management, code snippets
- **IntelliJ IDEA**: Full IDE integration with quantum-safe encryption
- **Jupyter**: Notebook cell encryption for sensitive data science work

## 🔒 Security Features

All tools implement:
- NIST post-quantum cryptography standards
- Hardware security module integration
- Zero-knowledge architecture
- Autonomous key lifecycle management
- Compliance with GDPR, HIPAA, PCI DSS

## 📚 Documentation

- [API Reference](https://docs.datacraft.co.ke/encryption/api)
- [SDK Documentation](https://docs.datacraft.co.ke/encryption/sdks)
- [CLI Documentation](https://docs.datacraft.co.ke/encryption/cli)
- [IDE Plugin Guide](https://docs.datacraft.co.ke/encryption/plugins)

## 🛠️ Installation

### Package Managers
```bash
# Python
pip install apg-encrypt

# JavaScript
npm install apg-encryption-javascript

# Go
go get github.com/datacraft/apg-encryption-go

# Java (Maven)
<dependency>
    <groupId>co.datacraft</groupId>
    <artifactId>apg-encryption</artifactId>
    <version>1.0.0</version>
</dependency>
```

### IDE Extensions
- **VS Code**: Search "APG Encryption Services" in Extensions Marketplace
- **IntelliJ**: Install from JetBrains Plugin Repository
- **Jupyter**: `pip install apg-jupyter-extension`

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Developer     │    │   APG Platform   │    │   Quantum-Safe  │
│   Tools         │────│   Services       │────│   Cryptography  │
│                 │    │                  │    │                 │
├─ CLI Tool       │    ├─ API Gateway     │    ├─ CRYSTALS-Kyber │
├─ SDKs           │    ├─ Key Management  │    ├─ CRYSTALS-Dil.  │
├─ IDE Plugins    │    ├─ Policy Engine   │    ├─ FALCON         │
└─ Mobile Apps    │    └─ Analytics       │    └─ SPHINCS+       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License © 2025 Datacraft

## 📞 Support

- **Documentation**: https://docs.datacraft.co.ke
- **Issues**: https://github.com/datacraft/apg-encryption/issues  
- **Email**: support@datacraft.co.ke
''',
			
			"DEVELOPER_GUIDE.md": '''# APG Encryption - Developer Guide

Comprehensive guide for developers using APG Encryption Services.

## 🎯 Overview

APG Encryption Services provides quantum-safe cryptography through multiple developer-friendly interfaces:

- **Command Line Tools**: For DevOps and system administrators
- **SDKs**: For application developers in multiple languages
- **IDE Plugins**: For seamless development workflow integration
- **Mobile SDKs**: For iOS and Android applications

## 🏛️ Architecture Principles

### Quantum-Safe First
All tools implement NIST-standardized post-quantum cryptography:
- CRYSTALS-Kyber for key encapsulation
- CRYSTALS-Dilithium for digital signatures
- FALCON for compact signatures
- SPHINCS+ for stateless signatures

### Zero-Knowledge Architecture
- Never expose plaintext during operations
- Threshold cryptography for distributed trust
- Privacy-preserving computation capabilities

### Autonomous Management
- AI-powered key lifecycle management
- Predictive threat response
- Automated compliance adherence

## 🛠️ Development Workflow

### 1. Environment Setup

#### API Configuration
```bash
export APG_TENANT_ID="your-tenant-id"
export APG_API_KEY="your-api-key"
export APG_BASE_URL="https://api.datacraft.co.ke"
```

#### CLI Installation
```bash
# Install APG CLI
pip install apg-encrypt

# Verify installation
apg-encrypt version

# Configure CLI
apg-encrypt config --set tenant_id your-tenant-id
apg-encrypt config --set api_key your-api-key
```

### 2. Key Management

#### Generate Quantum-Safe Keys
```bash
# Generate key pair
apg-encrypt keygen --algorithm CRYSTALS-Kyber-1024 --output my-key.pem

# List available keys
apg-encrypt keys list

# Export public key
apg-encrypt keys export KEY_ID --public --output public-key.pem
```

#### Programmatic Key Management
```python
async with APGEncryptionClient(tenant_id="...", api_key="...") as client:
    # Generate key pair
    key_pair = await client.generate_key_pair("CRYSTALS-Kyber-1024")
    
    # Store key ID for future operations
    key_id = key_pair.key_id
    
    # Use key for encryption
    result = await client.encrypt_quantum_safe(
        data="sensitive data",
        key_id=key_id
    )
```

### 3. Data Encryption Patterns

#### File Encryption
```python
import asyncio
from pathlib import Path
from apg_encryption import APGEncryptionClient

async def encrypt_files(file_paths: list[Path], output_dir: Path):
    async with APGEncryptionClient(...) as client:
        for file_path in file_paths:
            # Read file
            data = file_path.read_bytes()
            
            # Encrypt
            result = await client.encrypt_quantum_safe(data)
            
            # Save encrypted file
            encrypted_file = output_dir / f"{file_path.name}.encrypted"
            encrypted_file.write_text(result.encrypted_data)
            
            print(f"✅ Encrypted: {file_path} → {encrypted_file}")
```

#### Database Field Encryption
```python
class SecureUserModel:
    def __init__(self, client: APGEncryptionClient):
        self.client = client
    
    async def save_user(self, user_data: dict):
        # Encrypt sensitive fields
        encrypted_ssn = await self.client.encrypt_quantum_safe(user_data["ssn"])
        encrypted_email = await self.client.encrypt_quantum_safe(user_data["email"])
        
        # Save to database with encrypted fields
        await self.db.users.insert({
            "name": user_data["name"],  # Plain text
            "ssn_encrypted": encrypted_ssn.encrypted_data,
            "email_encrypted": encrypted_email.encrypted_data,
            "key_id": encrypted_ssn.key_id
        })
```

#### API Response Encryption
```python
from fastapi import FastAPI
from apg_encryption import APGEncryptionClient

app = FastAPI()
client = APGEncryptionClient(...)

@app.get("/sensitive-data/{user_id}")
async def get_sensitive_data(user_id: str):
    # Get sensitive data
    data = await get_user_sensitive_data(user_id)
    
    # Encrypt response
    encrypted_response = await client.encrypt_quantum_safe(
        json.dumps(data)
    )
    
    return {
        "encrypted_data": encrypted_response.encrypted_data,
        "key_id": encrypted_response.key_id,
        "algorithm": encrypted_response.algorithm
    }
```

### 4. Integration Patterns

#### Microservices Architecture
```python
# Service A - Encrypt data before sending to Service B
async def send_encrypted_data(data: dict):
    encrypted = await encryption_client.encrypt_quantum_safe(
        json.dumps(data)
    )
    
    response = await httpx.post(
        "http://service-b/api/data",
        json={
            "encrypted_payload": encrypted.encrypted_data,
            "key_id": encrypted.key_id
        }
    )
    return response

# Service B - Decrypt received data
async def receive_encrypted_data(request: dict):
    decrypted = await encryption_client.decrypt_quantum_safe(
        request["encrypted_payload"],
        key_id=request["key_id"]
    )
    
    data = json.loads(decrypted.decrypted_data)
    return await process_data(data)
```

#### Event-Driven Architecture
```python
import asyncio
from apg_encryption import APGEncryptionClient

class EncryptedEventBus:
    def __init__(self, encryption_client: APGEncryptionClient):
        self.client = encryption_client
        self.subscribers = {}
    
    async def publish(self, event_type: str, data: dict):
        # Encrypt event data
        encrypted_event = await self.client.encrypt_quantum_safe(
            json.dumps(data)
        )
        
        # Publish encrypted event
        event = {
            "type": event_type,
            "encrypted_data": encrypted_event.encrypted_data,
            "key_id": encrypted_event.key_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._notify_subscribers(event_type, event)
    
    async def subscribe(self, event_type: str, handler):
        # Handler receives encrypted event and must decrypt
        async def encrypted_handler(event):
            decrypted_data = await self.client.decrypt_quantum_safe(
                event["encrypted_data"],
                key_id=event["key_id"]
            )
            
            original_data = json.loads(decrypted_data.decrypted_data)
            await handler(original_data)
        
        self.subscribers[event_type] = encrypted_handler
```

## 🔐 Security Best Practices

### 1. Key Management
- **Rotate Keys Regularly**: Use automated key rotation
- **Separate Encryption Keys**: Different keys for different data types
- **Hardware Security**: Use hardware security modules when available
- **Key Escrow**: Implement proper key backup and recovery

### 2. Data Handling
- **Encrypt at Rest**: All sensitive data should be encrypted when stored
- **Encrypt in Transit**: Use TLS + application-level encryption
- **Memory Safety**: Clear sensitive data from memory after use
- **Logging**: Never log decrypted sensitive data

### 3. Access Control
- **Principle of Least Privilege**: Grant minimal necessary permissions
- **API Key Management**: Rotate API keys regularly
- **Audit Logging**: Log all encryption/decryption operations
- **Multi-Factor Authentication**: Require MFA for key operations

## 🧪 Testing Strategies

### Unit Testing
```python
import pytest
from apg_encryption import APGEncryptionClient
from apg_encryption.exceptions import APGEncryptionError

@pytest.mark.asyncio
async def test_encrypt_decrypt_round_trip():
    async with APGEncryptionClient(
        tenant_id="test",
        api_key="test-key"
    ) as client:
        # Test data
        original_data = "This is test data"
        
        # Encrypt
        encrypted = await client.encrypt_quantum_safe(original_data)
        assert encrypted.encrypted_data != original_data
        assert encrypted.key_id is not None
        
        # Decrypt
        decrypted = await client.decrypt_quantum_safe(
            encrypted.encrypted_data,
            key_id=encrypted.key_id
        )
        
        assert decrypted.decrypted_data == original_data
        assert decrypted.verified is True

@pytest.mark.asyncio
async def test_invalid_key_handling():
    async with APGEncryptionClient(...) as client:
        with pytest.raises(APGEncryptionError):
            await client.decrypt_quantum_safe(
                "invalid_encrypted_data",
                key_id="invalid_key_id"
            )
```

### Integration Testing
```python
@pytest.mark.integration
async def test_cross_service_encryption():
    # Test encryption in Service A and decryption in Service B
    service_a_client = APGEncryptionClient(tenant_id="service-a", ...)
    service_b_client = APGEncryptionClient(tenant_id="service-b", ...)
    
    # Service A encrypts data
    data = {"user_id": 123, "sensitive_info": "secret"}
    encrypted = await service_a_client.encrypt_quantum_safe(
        json.dumps(data)
    )
    
    # Service B decrypts data (using same tenant)
    decrypted = await service_b_client.decrypt_quantum_safe(
        encrypted.encrypted_data,
        key_id=encrypted.key_id
    )
    
    recovered_data = json.loads(decrypted.decrypted_data)
    assert recovered_data == data
```

### Load Testing
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def load_test_encryption(num_operations: int = 1000):
    async with APGEncryptionClient(...) as client:
        start_time = time.time()
        
        # Create tasks for concurrent encryption
        tasks = []
        for i in range(num_operations):
            task = client.encrypt_quantum_safe(f"Test data {i}")
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Encrypted {num_operations} items in {duration:.2f} seconds")
        print(f"Average: {duration/num_operations*1000:.2f} ms per operation")
        print(f"Throughput: {num_operations/duration:.2f} operations/second")
        
        return results
```

## 📊 Performance Optimization

### Connection Pooling
```python
import httpx
from apg_encryption import APGEncryptionClient

# Use connection pooling for better performance
limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
timeout = httpx.Timeout(30.0)

async with APGEncryptionClient(
    tenant_id="...",
    api_key="...",
    http_limits=limits,
    http_timeout=timeout
) as client:
    # Client will reuse connections
    pass
```

### Batch Operations
```python
async def batch_encrypt(client: APGEncryptionClient, data_items: list[str]):
    # Process items in batches for better performance
    batch_size = 10
    results = []
    
    for i in range(0, len(data_items), batch_size):
        batch = data_items[i:i + batch_size]
        
        # Encrypt batch items concurrently
        tasks = [client.encrypt_quantum_safe(item) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        
        results.extend(batch_results)
        
        # Optional: Add small delay between batches to avoid rate limiting
        await asyncio.sleep(0.1)
    
    return results
```

### Caching Strategies
```python
import functools
from datetime import datetime, timedelta

class CachedEncryptionClient:
    def __init__(self, client: APGEncryptionClient):
        self.client = client
        self.key_cache = {}  # key_id -> (public_key, expires_at)
    
    @functools.lru_cache(maxsize=1000)
    async def get_cached_key(self, key_id: str):
        # Cache key information to avoid repeated API calls
        if key_id in self.key_cache:
            key_info, expires_at = self.key_cache[key_id]
            if datetime.now() < expires_at:
                return key_info
        
        # Fetch from API and cache
        key_info = await self.client.get_key_info(key_id)
        expires_at = datetime.now() + timedelta(hours=1)
        self.key_cache[key_id] = (key_info, expires_at)
        
        return key_info
```

## 🚨 Error Handling

### Comprehensive Error Handling
```python
from apg_encryption.exceptions import (
    APGEncryptionError,
    APGDecryptionError,
    APGKeyGenerationError,
    APGAuthenticationError,
    APGConfigurationError
)

async def robust_encryption_operation(client, data):
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            result = await client.encrypt_quantum_safe(data)
            return result
            
        except APGAuthenticationError as e:
            # Authentication errors are not retryable
            logger.error(f"Authentication failed: {e}")
            raise
            
        except APGConfigurationError as e:
            # Configuration errors are not retryable
            logger.error(f"Configuration error: {e}")
            raise
            
        except APGEncryptionError as e:
            # Encryption errors might be retryable
            if attempt < max_retries - 1:
                logger.warning(f"Encryption attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                logger.error(f"All encryption attempts failed: {e}")
                raise
                
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error during encryption: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                raise
```

## 📈 Monitoring and Analytics

### Operation Metrics
```python
import time
import logging
from functools import wraps

def monitor_encryption_operations(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        operation_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log success metrics
            logging.info(f"{operation_name}_success", extra={
                "duration_ms": duration * 1000,
                "data_size": len(args[1]) if len(args) > 1 else 0,
                "algorithm": getattr(result, 'algorithm', 'unknown')
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error metrics
            logging.error(f"{operation_name}_error", extra={
                "duration_ms": duration * 1000,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise
    
    return wrapper

# Usage
@monitor_encryption_operations
async def encrypt_user_data(client, user_data):
    return await client.encrypt_quantum_safe(json.dumps(user_data))
```

## 🎯 Next Steps

1. **Choose Your Tools**: Select CLI, SDK, or IDE plugin based on your needs
2. **Set Up Environment**: Configure API keys and tenant ID
3. **Start Small**: Begin with simple encryption/decryption operations
4. **Scale Up**: Implement in production with proper error handling and monitoring
5. **Optimize**: Use caching, connection pooling, and batch operations for performance

For more detailed examples and advanced use cases, visit our [documentation](https://docs.datacraft.co.ke/encryption).
''',
			
			"API_REFERENCE.md": '''# APG Encryption API Reference

Complete API documentation for APG Encryption Services.

## Base URL
```
https://api.datacraft.co.ke/api/v1
```

## Authentication

All requests must include:
- `Authorization: Bearer <api_key>`
- `X-Tenant-ID: <tenant_id>`

## Endpoints

### Encryption Operations

#### POST /encrypt
Encrypt data using quantum-safe algorithms.

**Request:**
```json
{
  "data": "48656c6c6f2c20576f726c6421",  // hex-encoded data
  "algorithm": "CRYSTALS-Kyber-1024",
  "metadata": {
    "source": "api",
    "tags": ["sensitive", "user-data"]
  }
}
```

**Response:**
```json
{
  "encrypted_data": "a1b2c3d4...",
  "key_id": "01234567-89ab-cdef-0123-456789abcdef",
  "algorithm": "CRYSTALS-Kyber-1024",
  "metadata": {...},
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### POST /decrypt
Decrypt previously encrypted data.

**Request:**
```json
{
  "encrypted_data": "a1b2c3d4...",
  "key_id": "01234567-89ab-cdef-0123-456789abcdef"
}
```

**Response:**
```json
{
  "decrypted_data": "48656c6c6f2c20576f726c6421",
  "key_id": "01234567-89ab-cdef-0123-456789abcdef",
  "algorithm": "CRYSTALS-Kyber-1024",
  "verified": true,
  "timestamp": "2025-01-15T10:35:00Z"
}
```

### Key Management

#### POST /keys/generate
Generate new quantum-safe key pair.

**Request:**
```json
{
  "algorithm": "CRYSTALS-Kyber-1024",
  "metadata": {
    "purpose": "user-data-encryption",
    "expires_in": 86400
  }
}
```

**Response:**
```json
{
  "key_id": "01234567-89ab-cdef-0123-456789abcdef",
  "public_key": "-----BEGIN PUBLIC KEY-----\\n...\\n-----END PUBLIC KEY-----",
  "algorithm": "CRYSTALS-Kyber-1024",
  "key_size": 1568,
  "created_at": "2025-01-15T10:30:00Z",
  "expires_at": "2025-01-16T10:30:00Z"
}
```

#### GET /keys
List available keys.

**Response:**
```json
{
  "keys": [
    {
      "key_id": "01234567-89ab-cdef-0123-456789abcdef",
      "algorithm": "CRYSTALS-Kyber-1024",
      "key_size": 1568,
      "public_key": "-----BEGIN PUBLIC KEY-----\\n...\\n-----END PUBLIC KEY-----",
      "is_active": true,
      "created_at": "2025-01-15T10:30:00Z",
      "expires_at": null
    }
  ]
}
```

#### DELETE /keys/{key_id}
Delete a key pair.

**Response:**
```json
{
  "success": true,
  "key_id": "01234567-89ab-cdef-0123-456789abcdef",
  "deleted_at": "2025-01-15T10:40:00Z"
}
```

## Error Responses

All errors follow this format:
```json
{
  "error": {
    "code": "ENCRYPTION_FAILED",
    "message": "Encryption operation failed due to invalid algorithm",
    "details": {
      "algorithm": "invalid-algorithm",
      "supported_algorithms": ["CRYSTALS-Kyber-512", "CRYSTALS-Kyber-768", "CRYSTALS-Kyber-1024"]
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req-01234567-89ab-cdef"
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `AUTHENTICATION_FAILED` | Invalid API key or tenant ID |
| `AUTHORIZATION_DENIED` | Insufficient permissions |
| `ENCRYPTION_FAILED` | Encryption operation failed |
| `DECRYPTION_FAILED` | Decryption operation failed |
| `KEY_NOT_FOUND` | Specified key does not exist |
| `KEY_GENERATION_FAILED` | Key generation operation failed |
| `INVALID_ALGORITHM` | Unsupported encryption algorithm |
| `INVALID_DATA` | Invalid input data format |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Server-side error |

## Rate Limits

| Endpoint | Rate Limit |
|----------|------------|
| `/encrypt` | 1000 requests/minute |
| `/decrypt` | 1000 requests/minute |
| `/keys/generate` | 100 requests/minute |
| `/keys` (list) | 500 requests/minute |
| `/keys/{id}` (delete) | 100 requests/minute |

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248300
```

## Supported Algorithms

### Key Encapsulation Mechanisms (KEMs)
- `CRYSTALS-Kyber-512` - NIST Level 1 security
- `CRYSTALS-Kyber-768` - NIST Level 3 security
- `CRYSTALS-Kyber-1024` - NIST Level 5 security (recommended)

### Digital Signatures
- `CRYSTALS-Dilithium2` - NIST Level 1 security
- `CRYSTALS-Dilithium3` - NIST Level 3 security
- `CRYSTALS-Dilithium5` - NIST Level 5 security
- `FALCON-512` - NIST Level 1 security, compact signatures
- `FALCON-1024` - NIST Level 5 security, compact signatures
- `SPHINCS+-128s` - Stateless, conservative security
- `SPHINCS+-256s` - Stateless, high security

## SDKs and Examples

See language-specific documentation:
- [Python SDK](./python-sdk.md)
- [JavaScript SDK](./javascript-sdk.md)
- [Java SDK](./java-sdk.md)
- [C# SDK](./csharp-sdk.md)
- [Go SDK](./go-sdk.md)
'''
		}
		
		return docs
	
	async def package_tools_for_distribution(self, tools_data: Dict[str, Any]) -> Dict[str, str]:
		"""Package all tools for distribution"""
		
		packages = {}
		
		# Create CLI tool package
		if tools_data.get("cli_tools"):
			cli_tool = tools_data["cli_tools"][0]
			cli_package = await self._create_package("cli", cli_tool)
			packages["apg-encrypt-cli.tar.gz"] = cli_package
		
		# Create SDK packages
		for sdk in tools_data.get("sdks", []):
			package_name = f"apg-encryption-{sdk['language']}.tar.gz"
			sdk_package = await self._create_package("sdk", sdk)
			packages[package_name] = sdk_package
		
		# Create IDE plugin packages
		for plugin in tools_data.get("ide_plugins", []):
			package_name = f"apg-encryption-{plugin['platform']}.zip"
			plugin_package = await self._create_package("plugin", plugin)
			packages[package_name] = plugin_package
		
		return packages
	
	async def _create_package(self, tool_type: str, tool_data: Dict[str, Any]) -> str:
		"""Create distribution package for a tool"""
		
		# Create temporary directory
		with tempfile.TemporaryDirectory() as temp_dir:
			temp_path = Path(temp_dir)
			
			# Write all files to temporary directory
			for file_path, content in tool_data.get("files", {}).items():
				full_path = temp_path / file_path
				full_path.parent.mkdir(parents=True, exist_ok=True)
				full_path.write_text(content)
			
			# Create package
			if tool_type in ["cli", "sdk"]:
				package_path = temp_path / f"{tool_data.get('name', 'package')}.tar.gz"
				subprocess.run([
					"tar", "czf", str(package_path), 
					"-C", str(temp_path), "."
				], check=True)
			else:  # plugin
				package_path = temp_path / f"{tool_data.get('name', 'package')}.zip"
				with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
					for file_path in temp_path.rglob("*"):
						if file_path.is_file() and file_path != package_path:
							arcname = file_path.relative_to(temp_path)
							zf.write(file_path, arcname)
			
			# Read package content
			return package_path.read_text() if package_path.suffix == ".txt" else "Binary package created"

# Initialize developer tools manager for immediate use
developer_tools_manager = DeveloperToolsManager(get_tenant_id_from_context())
