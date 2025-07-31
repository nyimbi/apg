#!/usr/bin/env python3
"""
Setup script for Enhanced YouTube Crawler Package
==================================================

Installation script for the comprehensive YouTube content crawler
with advanced filtering, metadata extraction, and integration capabilities.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're in the right directory
PACKAGE_DIR = Path(__file__).parent
sys.path.insert(0, str(PACKAGE_DIR))

# Import package information
try:
    from youtube_crawler import __version__, __author__, __email__, __company__, PACKAGE_INFO
except ImportError:
    # Fallback if package can't be imported
    __version__ = "1.0.0"
    __author__ = "Nyimbi Odero"
    __email__ = "nyimbi@datacraft.co.ke"
    __company__ = "Datacraft"
    PACKAGE_INFO = {}

# Read README file
def read_readme():
    """Read README file content."""
    readme_path = PACKAGE_DIR / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Enhanced YouTube Crawler Package"

# Read requirements
def read_requirements(filename="requirements.txt"):
    """Read requirements from file."""
    req_path = PACKAGE_DIR / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []

# Package configuration
PACKAGE_NAME = "enhanced-youtube-crawler"
PACKAGE_DESCRIPTION = "Enterprise-grade YouTube content crawler with advanced filtering and metadata extraction"
LONG_DESCRIPTION = read_readme()
KEYWORDS = [
    "youtube", "crawler", "scraper", "video", "metadata", "api", "social-media",
    "data-extraction", "content-analysis", "web-scraping", "async", "python"
]

# Requirements
INSTALL_REQUIRES = [
    "aiohttp>=3.8.0",
    "asyncpg>=0.27.0",
    "google-api-python-client>=2.70.0",
    "yt-dlp>=2023.1.6",
    "beautifulsoup4>=4.11.0",
    "lxml>=4.9.0",
    "requests>=2.28.0",
    "pandas>=1.5.0",
    "numpy>=1.24.0",
    "python-dateutil>=2.8.0",
    "PyYAML>=6.0",
    "python-decouple>=3.6",
    "ratelimit>=2.2.0",
    "backoff>=2.2.0"
]

EXTRAS_REQUIRE = {
    "full": [
        "textblob>=0.17.0",
        "langdetect>=1.0.9",
        "scikit-learn>=1.2.0",
        "pydantic>=1.10.0",
        "marshmallow>=3.19.0",
        "Pillow>=9.4.0",
        "opencv-python>=4.7.0",
        "ffmpeg-python>=0.2.0",
        "SQLAlchemy>=1.4.0",
        "alembic>=1.9.0",
        "redis>=4.5.0",
        "python-memcached>=1.59",
        "structlog>=22.3.0",
        "sentry-sdk>=1.14.0",
        "httpx>=0.23.0",
        "tqdm>=4.64.0",
        "click>=8.1.0",
        "cryptography>=39.0.0",
        "python-jose>=3.3.0",
        "python-snappy>=0.6.0",
        "lz4>=4.0.0",
        "pytz>=2022.7"
    ],
    "dev": [
        "pytest>=7.2.0",
        "pytest-asyncio>=0.20.0",
        "pytest-mock>=3.10.0",
        "pytest-cov>=4.0.0",
        "factory-boy>=3.2.0",
        "black>=22.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0"
    ],
    "docs": [
        "sphinx>=6.1.0",
        "sphinx-rtd-theme>=1.2.0",
        "sphinxcontrib-asyncio>=0.3.0"
    ],
    "performance": [
        "uvloop>=0.17.0",
        "orjson>=3.8.0",
        "cython>=0.29.0"
    ],
    "monitoring": [
        "prometheus-client>=0.16.0",
        "grafana-api>=1.0.3",
        "elasticsearch>=8.6.0"
    ]
}

# Add 'all' extra that includes everything
EXTRAS_REQUIRE["all"] = list(set(
    dep for extra_deps in EXTRAS_REQUIRE.values() for dep in extra_deps
))

# Development requirements
DEV_REQUIRES = read_requirements("requirements-dev.txt") if (PACKAGE_DIR / "requirements-dev.txt").exists() else []

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Internet :: WWW/HTTP",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Multimedia :: Video",
    "Topic :: Text Processing :: General",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Framework :: AsyncIO",
    "Typing :: Typed"
]

# Entry points
ENTRY_POINTS = {
    "console_scripts": [
        "youtube-crawler=youtube_crawler.cli:main",
        "ytcrawl=youtube_crawler.cli:main"
    ]
}

# Package data
PACKAGE_DATA = {
    "youtube_crawler": [
        "config/*.yml",
        "config/*.yaml",
        "config/*.json",
        "templates/*.html",
        "templates/*.txt",
        "schemas/*.json",
        "data/*.json",
        "data/*.csv"
    ]
}

# Setup configuration
setup(
    # Basic package information
    name=PACKAGE_NAME,
    version=__version__,
    description=PACKAGE_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",

    # Author information
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,

    # URLs
    url="https://github.com/datacraft-ke/youtube-crawler",
    download_url=f"https://github.com/datacraft-ke/youtube-crawler/archive/v{__version__}.tar.gz",
    project_urls={
        "Bug Reports": "https://github.com/datacraft-ke/youtube-crawler/issues",
        "Source": "https://github.com/datacraft-ke/youtube-crawler",
        "Documentation": "https://youtube-crawler.readthedocs.io/",
        "Company": "https://www.datacraft.co.ke"
    },

    # Package discovery
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    package_data=PACKAGE_DATA,
    include_package_data=True,

    # Requirements
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # Entry points
    entry_points=ENTRY_POINTS,

    # Metadata
    license="MIT",
    keywords=" ".join(KEYWORDS),
    classifiers=CLASSIFIERS,

    # Options
    zip_safe=False,
    platforms=["any"],

    # Additional metadata
    provides=["youtube_crawler"],
    obsoletes=[],

    # Test suite
    test_suite="tests",
    tests_require=DEV_REQUIRES,

    # Options for different build systems
    options={
        "build_ext": {
            "inplace": True
        },
        "bdist_wheel": {
            "universal": False
        }
    }
)

# Post-installation message
def post_install():
    """Display post-installation message."""
    print("\n" + "="*60)
    print("Enhanced YouTube Crawler Installation Complete!")
    print("="*60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__} ({__company__})")
    print("\nQuick Start:")
    print("1. Set your YouTube API key:")
    print("   export YOUTUBE_API_KEY='your_api_key_here'")
    print("\n2. Basic usage:")
    print("   from youtube_crawler import create_enhanced_youtube_client")
    print("   client = await create_enhanced_youtube_client()")
    print("   result = await client.crawl_video('dQw4w9WgXcQ')")
    print("\n3. Run examples:")
    print("   python -m youtube_crawler.examples.basic_usage")
    print("\n4. Check documentation:")
    print("   https://youtube-crawler.readthedocs.io/")
    print("\nFor support: nyimbi@datacraft.co.ke")
    print("="*60)

if __name__ == "__main__":
    # Run setup
    setup()

    # Display post-installation message
    if "install" in sys.argv:
        post_install()
