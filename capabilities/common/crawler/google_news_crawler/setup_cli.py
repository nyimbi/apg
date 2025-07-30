#!/usr/bin/env python
"""
CLI Setup for Enhanced Google News Crawler
==========================================

Setuptools configuration for installing the Google News Crawler CLI.

Installation:
    pip install -e .  # Install in development mode
    
Usage after installation:
    gnews-crawler --help
    gnews-crawler search "Ethiopia conflict" --crawlee
    gnews-crawler monitor --query "Horn of Africa crisis" --interval 300

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
def read_requirements():
    requirements_file = Path(__file__).parent / 'requirements.txt'
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'aiohttp>=3.8.0',
        'asyncpg>=0.27.0',
        'feedparser>=6.0.10',
        'beautifulsoup4>=4.11.0',
        'lxml>=4.9.0',
        'python-dateutil>=2.8.2',
        'PyYAML>=6.0'
    ]

# Read long description
def read_long_description():
    readme_file = Path(__file__).parent / 'README.md'
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "Enhanced Google News Crawler with Crawlee integration and CLI interface"

setup(
    name="google-news-crawler-cli",
    version="1.0.0",
    description="Enterprise-grade Google News crawler with CLI interface",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Nyimbi Odero",
    author_email="nyimbi@datacraft.co.ke",
    company="Datacraft",
    url="https://github.com/datacraft/lindela",
    license="MIT",
    
    packages=find_packages(),
    include_package_data=True,
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        'crawlee': [
            'crawlee>=0.1.0',
            'playwright>=1.30.0',
            'trafilatura>=1.6.0',
            'newspaper3k>=0.2.8',
            'readability-lxml>=0.8.1'
        ],
        'dev': [
            'pytest>=7.2.0',
            'pytest-asyncio>=0.20.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0'
        ],
        'ml': [
            'pandas>=1.5.0',
            'textblob>=0.17.0',
            'scikit-learn>=1.2.0',
            'numpy>=1.24.0'
        ]
    },
    
    entry_points={
        'console_scripts': [
            'gnews-crawler=cli.main:cli_entry_point',
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Text Processing :: General",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    keywords=[
        "news", "crawler", "google-news", "rss", "scraping", 
        "journalism", "crawlee", "cli", "conflict-monitoring"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/datacraft/lindela/issues",
        "Source": "https://github.com/datacraft/lindela",
        "Documentation": "https://github.com/datacraft/lindela/blob/main/README.md",
    },
)