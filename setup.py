#!/usr/bin/env python3
"""
APG Language Setup Script
=========================

Setup script for installing the APG language compiler and CLI tools.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from package
version = "1.0.0"

# Define package requirements
install_requires = [
    "antlr4-python3-runtime>=4.13.0",
    "Flask>=2.3.0",
    "Flask-AppBuilder>=4.3.0",
    "Flask-SQLAlchemy>=3.0.0",
    "SQLAlchemy>=2.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "python-dateutil>=2.8.0",
    "Jinja2>=3.1.0",
    "watchdog>=3.0.0",
    "psutil>=5.9.0",
]

# Development dependencies
dev_requires = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
]

# Documentation dependencies
doc_requires = [
    "Sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
]

setup(
    name="apg-language",
    version=version,
    description="Application Programming Generation (APG) Language Compiler and Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="APG Language Team",
    author_email="team@apg-lang.org",
    url="https://github.com/apg-lang/apg",
    project_urls={
        "Documentation": "https://apg-lang.org/docs",
        "Source": "https://github.com/apg-lang/apg",
        "Tracker": "https://github.com/apg-lang/apg/issues",
    },
    
    # Package configuration
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": doc_requires,
        "language-server": [
            "pygls>=1.0.0",
            "lsprotocol>=2023.0.0"
        ],
        "ai": [
            "openai>=1.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "numpy>=1.24.0"
        ],
        "all": dev_requires + doc_requires,
    },
    
    # Entry points for CLI tools
    entry_points={
        "console_scripts": [
            "apg=cli.main:cli",
            "apg-compile=cli.compile_command:compile_cmd",
            "apg-run=cli.run_command:run",
            "apg-create=cli.create_project:create",
            "apg-validate=cli.validate_command:validate",
            "apg-language-server=language_server.server:main",
        ],
    },
    
    # Package data and resources
    package_data={
        "apg": [
            "spec/*.g4",
            "spec/*.apg",
            "templates/templates/**/*.template",
            "templates/templates/**/*.json",
            "vscode-extension/**/*",
            "language_server/**/*.py",
            "compiler/templates/*",
            "cli/**/*.py",
            "docs/**/*",
        ],
    },
    include_package_data=True,
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Code Generators", 
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    
    # Keywords for discoverability
    keywords=[
        "apg", "application-generation", "code-generation", "dsl", 
        "flask-appbuilder", "web-development", "automation", "agents",
        "workflows", "digital-twins", "iot", "compiler", "language"
    ],
    
    # Minimum requirements
    zip_safe=False,
    
    # Command line interface
    scripts=[],
)