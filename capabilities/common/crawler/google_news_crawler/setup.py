#!/usr/bin/env python
"""
Setup Script for Enhanced Google News Crawler
=============================================

This setup script provides easy installation and configuration for the
Enhanced Google News Crawler package.

Features:
- Dependency installation
- Database setup
- Configuration file generation
- Environment validation
- Development setup
- CLI installation

Usage:
    python setup.py install          # Install dependencies
    python setup.py init             # Initialize configuration
    python setup.py test             # Run tests
    python setup.py dev              # Development setup
    pip install -e .                 # Install CLI (gnews-crawler command)

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Package information
PACKAGE_NAME = "enhanced-google-news-crawler"
PACKAGE_VERSION = "1.0.0"
PACKAGE_DESCRIPTION = "Enterprise-grade Google News crawler with advanced features"
AUTHOR = "Nyimbi Odero"
AUTHOR_EMAIL = "nyimbi@datacraft.co.ke"
COMPANY = "Datacraft"
LICENSE = "MIT"

# Minimum Python version
MIN_PYTHON_VERSION = (3, 8)

# Required system packages (for different OS)
SYSTEM_PACKAGES = {
    'ubuntu': ['postgresql-client', 'libpq-dev', 'python3-dev'],
    'centos': ['postgresql-devel', 'python3-devel'],
    'macos': ['postgresql'],
    'windows': []
}

# Configuration templates
CONFIG_TEMPLATES = {
    'development': {
        'environment': 'development',
        'debug': True,
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'lindela_dev',
            'username': 'postgres',
            'password': 'password'
        },
        'logging': {
            'level': 'DEBUG',
            'enable_console': True,
            'enable_file': True
        },
        'performance': {
            'max_concurrent_requests': 5,
            'enable_caching': True
        }
    },
    'production': {
        'environment': 'production',
        'debug': False,
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'lindela',
            'username': 'lindela_user',
            'password': 'change_me'
        },
        'logging': {
            'level': 'INFO',
            'enable_console': False,
            'enable_file': True
        },
        'performance': {
            'max_concurrent_requests': 20,
            'enable_caching': True,
            'enable_compression': True
        },
        'monitoring': {
            'enabled': True,
            'prometheus_port': 8000
        }
    },
    'testing': {
        'environment': 'testing',
        'debug': True,
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'lindela_test',
            'username': 'postgres',
            'password': 'test'
        },
        'cache': {
            'type': 'memory'
        },
        'performance': {
            'max_concurrent_requests': 2,
            'enable_caching': False
        }
    }
}


class SetupManager:
    """Manages setup and configuration for the Google News Crawler."""

    def __init__(self):
        self.package_root = Path(__file__).parent
        self.project_root = self.package_root.parent.parent.parent.parent
        self.config_dir = self.package_root / 'config'
        self.requirements_file = self.package_root / 'requirements.txt'

    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        current_version = sys.version_info[:2]
        if current_version < MIN_PYTHON_VERSION:
            logger.error(
                f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required. "
                f"Current version: {current_version[0]}.{current_version[1]}"
            )
            return False
        logger.info(f"Python version check passed: {current_version[0]}.{current_version[1]}")
        return True

    def detect_os(self) -> str:
        """Detect operating system."""
        import platform
        system = platform.system().lower()

        if system == 'linux':
            # Try to detect Linux distribution
            try:
                with open('/etc/os-release', 'r') as f:
                    content = f.read().lower()
                    if 'ubuntu' in content or 'debian' in content:
                        return 'ubuntu'
                    elif 'centos' in content or 'rhel' in content or 'fedora' in content:
                        return 'centos'
            except FileNotFoundError:
                pass
            return 'linux'
        elif system == 'darwin':
            return 'macos'
        elif system == 'windows':
            return 'windows'
        else:
            return 'unknown'

    def install_system_dependencies(self) -> bool:
        """Install system-level dependencies."""
        os_type = self.detect_os()
        packages = SYSTEM_PACKAGES.get(os_type, [])

        if not packages:
            logger.info(f"No system dependencies required for {os_type}")
            return True

        logger.info(f"Installing system dependencies for {os_type}: {packages}")

        try:
            if os_type == 'ubuntu':
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y'] + packages, check=True)
            elif os_type == 'centos':
                subprocess.run(['sudo', 'yum', 'install', '-y'] + packages, check=True)
            elif os_type == 'macos':
                # Assume Homebrew is installed
                subprocess.run(['brew', 'install'] + packages, check=True)

            logger.info("System dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install system dependencies: {e}")
            logger.warning("Please install system dependencies manually")
            return False

    def install_python_dependencies(self, dev: bool = False) -> bool:
        """Install Python dependencies."""
        logger.info("Installing Python dependencies...")

        try:
            # Upgrade pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)

            # Install requirements
            if self.requirements_file.exists():
                cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)]
                subprocess.run(cmd, check=True)
            else:
                # Install core dependencies manually
                core_deps = [
                    'aiohttp>=3.8.0',
                    'asyncpg>=0.27.0',
                    'feedparser>=6.0.10',
                    'beautifulsoup4>=4.11.0',
                    'lxml>=4.9.0',
                    'python-dateutil>=2.8.2',
                    'PyYAML>=6.0'
                ]
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + core_deps, check=True)

            # Install development dependencies if requested
            if dev:
                dev_deps = [
                    'pytest>=7.2.0',
                    'pytest-asyncio>=0.20.0',
                    'pytest-cov>=4.0.0',
                    'black>=23.0.0',
                    'flake8>=6.0.0',
                    'isort>=5.12.0'
                ]
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + dev_deps, check=True)

            logger.info("Python dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False

    def create_config_directory(self) -> bool:
        """Create configuration directory."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Configuration directory created: {self.config_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create config directory: {e}")
            return False

    def generate_config_file(self, environment: str = 'development') -> bool:
        """Generate configuration file for specified environment."""
        if environment not in CONFIG_TEMPLATES:
            logger.error(f"Unknown environment: {environment}")
            return False

        config = CONFIG_TEMPLATES[environment].copy()
        config_file = self.config_dir / f'{environment}.yaml'

        try:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration file created: {config_file}")

            # Create .env file for environment variables
            env_file = self.config_dir / f'.env.{environment}'
            env_content = f"""# Environment variables for {environment}
LINDELA_ENVIRONMENT={environment}
LINDELA_DB_HOST={config['database']['host']}
LINDELA_DB_PORT={config['database']['port']}
LINDELA_DB_NAME={config['database']['database']}
LINDELA_DB_USER={config['database']['username']}
LINDELA_DB_PASSWORD={config['database']['password']}
LINDELA_DEBUG={str(config['debug']).lower()}
LINDELA_LOG_LEVEL={config['logging']['level']}
"""

            with open(env_file, 'w') as f:
                f.write(env_content)

            logger.info(f"Environment file created: {env_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to create configuration files: {e}")
            return False

    def check_database_connection(self, config: Dict[str, Any]) -> bool:
        """Check database connection."""
        try:
            import asyncpg
            import asyncio

            async def test_connection():
                try:
                    conn = await asyncpg.connect(
                        host=config['host'],
                        port=config['port'],
                        database=config['database'],
                        user=config['username'],
                        password=config['password']
                    )
                    await conn.execute('SELECT 1')
                    await conn.close()
                    return True
                except Exception as e:
                    logger.error(f"Database connection failed: {e}")
                    return False

            return asyncio.run(test_connection())

        except ImportError:
            logger.warning("asyncpg not available - skipping database check")
            return True
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False

    def run_tests(self) -> bool:
        """Run test suite."""
        logger.info("Running test suite...")

        try:
            test_dir = self.package_root / 'tests'
            if not test_dir.exists():
                logger.warning("Test directory not found")
                return True

            cmd = [sys.executable, '-m', 'pytest', str(test_dir), '-v']
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("All tests passed successfully")
                return True
            else:
                logger.error("Some tests failed")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run tests: {e}")
            return False

    def create_sample_scripts(self) -> bool:
        """Create sample usage scripts."""
        scripts_dir = self.package_root / 'scripts'
        scripts_dir.mkdir(exist_ok=True)

        # Basic usage script
        basic_script = scripts_dir / 'basic_usage.py'
        basic_content = '''#!/usr/bin/env python
"""
Basic usage example for Google News Crawler
"""
import asyncio
from lindela.packages_enhanced.crawlers.google_news_crawler import create_basic_gnews_client

async def main():
    # Mock database manager for demonstration
    class MockDBManager:
        async def initialize(self): pass
        async def close(self): pass
        async def article_exists(self, url): return False
        async def insert_article(self, data): return f"mock_{hash(data.get('url', ''))}"

    db_manager = MockDBManager()
    client = await create_basic_gnews_client(db_manager)

    try:
        # Search for news
        print("Searching for AI news...")
        # results = await client.search_news("artificial intelligence", max_results=5)
        print("Search completed (mock)")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
'''

        try:
            with open(basic_script, 'w') as f:
                f.write(basic_content)
            basic_script.chmod(0o755)

            logger.info(f"Sample script created: {basic_script}")
            return True

        except Exception as e:
            logger.error(f"Failed to create sample scripts: {e}")
            return False

    def print_setup_summary(self, environment: str = 'development'):
        """Print setup summary and next steps."""
        print("\n" + "="*60)
        print("🎉 Google News Crawler Setup Complete!")
        print("="*60)
        print(f"📦 Package: {PACKAGE_NAME} v{PACKAGE_VERSION}")
        print(f"🏢 Company: {COMPANY}")
        print(f"👨‍💻 Author: {AUTHOR} ({AUTHOR_EMAIL})")
        print(f"🌍 Environment: {environment}")
        print()
        print("📁 Files created:")
        print(f"   - Configuration: {self.config_dir}/{environment}.yaml")
        print(f"   - Environment: {self.config_dir}/.env.{environment}")
        print(f"   - Scripts: {self.package_root}/scripts/")
        print()
        print("🚀 Next steps:")
        print("   1. Review and update configuration files")
        print("   2. Set up your PostgreSQL database")
        print("   3. Run the example scripts to test functionality")
        print("   4. Check the README.md for detailed documentation")
        print()
        print("💡 Quick start:")
        print("   cd scripts && python basic_usage.py")
        print()
        print("🔧 Troubleshooting:")
        print("   - Check logs in the console output")
        print("   - Verify database connectivity")
        print("   - Ensure all dependencies are installed")
        print("="*60)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Enhanced Google News Crawler")
    parser.add_argument('command', choices=['install', 'init', 'test', 'dev'],
                       help='Setup command to run')
    parser.add_argument('--environment', '-e', default='development',
                       choices=['development', 'production', 'testing'],
                       help='Environment to configure')
    parser.add_argument('--skip-system', action='store_true',
                       help='Skip system dependency installation')
    parser.add_argument('--skip-db-check', action='store_true',
                       help='Skip database connectivity check')

    args = parser.parse_args()

    setup = SetupManager()

    # Check Python version
    if not setup.check_python_version():
        sys.exit(1)

    success = True

    if args.command == 'install':
        logger.info("🚀 Starting installation process...")

        # Install system dependencies
        if not args.skip_system:
            setup.install_system_dependencies()

        # Install Python dependencies
        if not setup.install_python_dependencies():
            success = False

        logger.info("📦 Installation completed")

    elif args.command == 'init':
        logger.info("🔧 Initializing configuration...")

        # Create config directory
        if not setup.create_config_directory():
            success = False

        # Generate configuration files
        if not setup.generate_config_file(args.environment):
            success = False

        # Create sample scripts
        if not setup.create_sample_scripts():
            success = False

        # Check database connection (if not skipped)
        if not args.skip_db_check:
            config = CONFIG_TEMPLATES[args.environment]['database']
            if not setup.check_database_connection(config):
                logger.warning("Database connection check failed")

        setup.print_setup_summary(args.environment)

    elif args.command == 'test':
        logger.info("🧪 Running test suite...")

        if not setup.run_tests():
            success = False

    elif args.command == 'dev':
        logger.info("🛠 Setting up development environment...")

        # Install with dev dependencies
        if not setup.install_python_dependencies(dev=True):
            success = False

        # Initialize with development config
        if not setup.create_config_directory():
            success = False

        if not setup.generate_config_file('development'):
            success = False

        if not setup.create_sample_scripts():
            success = False

        # Run tests
        setup.run_tests()

        setup.print_setup_summary('development')

    if success:
        logger.info("✅ Setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Setup completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
