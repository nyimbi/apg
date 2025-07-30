#!/usr/bin/env python3
"""
YouTube Crawler CLI Setup Script
================================

Setup script for installing and configuring the Enhanced YouTube Crawler CLI.
Handles dependencies, creates executables, and validates the installation.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
import argparse
import json
import tempfile

class YouTubeCLISetup:
    """Setup and installation manager for YouTube Crawler CLI"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.cli_script = self.project_root / "cli.py"
        self.executable_script = self.project_root / "youtube_crawler_cli"
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3.8):
            print("❌ Error: Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            return False
        
        print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
        return True
    
    def install_dependencies(self, enhanced=True, download_support=True):
        """Install required dependencies"""
        print("📦 Installing dependencies...")
        
        # Core dependencies
        core_deps = [
            "aiohttp>=3.8.0",
            "asyncpg>=0.27.0",
            "google-api-python-client>=2.0.0",
            "google-auth>=2.0.0",
            "google-auth-oauthlib>=0.5.0",
            "google-auth-httplib2>=0.1.0",
            "pyyaml>=6.0",
            "click>=8.0.0",
        ]
        
        # Enhanced CLI dependencies
        enhanced_deps = [
            "rich>=13.0.0",
            "tabulate>=0.9.0",
            "prompt-toolkit>=3.0.0",
        ]
        
        # Download support dependencies
        download_deps = [
            "yt-dlp>=2023.1.6",
            "ffmpeg-python>=0.2.0",
        ]
        
        # Performance optimization dependencies
        performance_deps = [
            "ujson>=5.0.0",
            "cachetools>=5.0.0",
            "redis>=4.0.0",
            "psutil>=5.9.0",
        ]
        
        # Data processing dependencies
        data_deps = [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "openpyxl>=3.0.0",  # For Excel export
        ]
        
        # Analysis dependencies
        analysis_deps = [
            "beautifulsoup4>=4.11.0",
            "textblob>=0.17.0",
            "nltk>=3.8",
        ]
        
        # Image processing dependencies
        image_deps = [
            "Pillow>=9.0.0",
            "opencv-python>=4.6.0",
        ]
        
        dependencies = core_deps[:]
        if enhanced:
            dependencies.extend(enhanced_deps)
            dependencies.extend(data_deps)
        if download_support:
            dependencies.extend(download_deps)
            dependencies.extend(performance_deps)
            dependencies.extend(analysis_deps)
            dependencies.extend(image_deps)
        
        failed_deps = []
        for dep in dependencies:
            try:
                print(f"Installing {dep}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"✅ {dep}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Warning: Failed to install {dep}")
                failed_deps.append(dep)
                if dep in core_deps:
                    print(f"❌ Critical dependency failed: {dep}")
                    return False
        
        if failed_deps:
            print(f"\n⚠️  Some optional dependencies failed to install:")
            for dep in failed_deps:
                print(f"   - {dep}")
            print("You can install them manually later if needed.")
        
        print("✅ Dependencies installation complete")
        return True
    
    def check_external_dependencies(self):
        """Check for external dependencies like ffmpeg"""
        print("🔍 Checking external dependencies...")
        
        # Check for ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("✅ ffmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  ffmpeg not found. Video processing may be limited.")
            print("   Install ffmpeg: https://ffmpeg.org/download.html")
        
        # Check for git (for version tracking)
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            print("✅ git is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  git not found. Version tracking disabled.")
        
        return True
    
    def create_executable(self):
        """Create executable CLI script"""
        if not self.cli_script.exists():
            print(f"❌ CLI script not found: {self.cli_script}")
            return False
        
        # Make CLI script executable
        os.chmod(self.cli_script, 0o755)
        
        # Create wrapper script if it doesn't exist
        if not self.executable_script.exists():
            print("Creating executable wrapper...")
            with open(self.executable_script, 'w') as f:
                f.write(f'''#!/usr/bin/env python3
"""
YouTube Crawler CLI Entry Point
===============================

Executable entry point for the Enhanced YouTube Crawler CLI tool.
"""

import sys
import os
from pathlib import Path

# Add the youtube crawler package to Python path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir))

# Import and run CLI
if __name__ == "__main__":
    try:
        from cli import main
        import asyncio
        
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, just call main
            main()
        except RuntimeError:
            # No event loop, create one
            asyncio.run(main())
            
    except ImportError as e:
        print(f"Error: Failed to import CLI module: {{e}}")
        print("Make sure you're running from the youtube_crawler package directory")
        print("and that all dependencies are installed.")
        print("Run: python setup_youtube_cli.py --help")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {{e}}")
        sys.exit(1)
''')
            
            os.chmod(self.executable_script, 0o755)
        
        print(f"✅ Executable created: {self.executable_script}")
        return True
    
    def create_configuration_examples(self):
        """Create configuration examples directory"""
        configs_dir = self.project_root / "configs"
        
        if configs_dir.exists():
            print(f"✅ Configuration examples ready: {configs_dir}")
            
            # List available configs
            config_files = list(configs_dir.glob("*.yaml"))
            if config_files:
                print("Available configuration examples:")
                for config_file in config_files:
                    print(f"  - {config_file.name}")
            return True
        else:
            print(f"⚠️  Configuration examples directory not found: {configs_dir}")
            return False
    
    def setup_environment_template(self):
        """Create environment template file"""
        env_template = self.project_root / ".env.template"
        
        template_content = """# YouTube Crawler Environment Configuration
# =========================================
# Copy this file to .env and fill in your values

# YouTube Data API Configuration
YOUTUBE_API_KEY=your_youtube_api_key_here

# Database Configuration (Optional)
DATABASE_URL=postgresql://user:password@localhost:5432/youtube_db

# Redis Cache Configuration (Optional)
REDIS_URL=redis://localhost:6379/0

# Performance Settings
MAX_CONCURRENT_REQUESTS=20
MAX_CONCURRENT_DOWNLOADS=5
RATE_LIMIT_PER_MINUTE=60

# Download Settings
DEFAULT_DOWNLOAD_FORMAT=best[height<=720]
DEFAULT_OUTPUT_DIR=./downloads
MAX_DOWNLOAD_SIZE_MB=500

# Monitoring Settings
HEALTH_CHECK_INTERVAL=300
ENABLE_PERFORMANCE_MONITORING=true

# Logging Settings
LOG_LEVEL=INFO
VERBOSE_LOGGING=false

# Advanced Features
ENABLE_CACHING=true
CACHE_TTL=3600
ENABLE_MEMORY_MONITORING=true

# Webhook Configuration (Optional)
WEBHOOK_URL=
SLACK_WEBHOOK=
DISCORD_WEBHOOK=

# Email Notifications (Optional)
SMTP_SERVER=
SMTP_PORT=587
EMAIL_USERNAME=
EMAIL_PASSWORD=
FROM_EMAIL=
TO_EMAILS=
"""
        
        try:
            with open(env_template, 'w') as f:
                f.write(template_content)
            print(f"✅ Environment template created: {env_template}")
            print("   Copy .env.template to .env and configure your settings")
            return True
        except Exception as e:
            print(f"⚠️  Failed to create environment template: {e}")
            return False
    
    def create_symlink(self, target_dir="/usr/local/bin"):
        """Create system-wide symlink for the CLI"""
        try:
            target_path = Path(target_dir) / "youtube-crawler"
            
            if target_path.exists():
                print(f"Removing existing symlink: {target_path}")
                target_path.unlink()
            
            if not Path(target_dir).exists():
                print(f"Target directory does not exist: {target_dir}")
                return False
            
            target_path.symlink_to(self.executable_script.absolute())
            print(f"✅ Symlink created: {target_path} -> {self.executable_script}")
            return True
            
        except PermissionError:
            print(f"⚠️  Permission denied creating symlink in {target_dir}")
            print("Try running with sudo or specify a different directory")
            return False
        except Exception as e:
            print(f"❌ Failed to create symlink: {e}")
            return False
    
    def validate_installation(self):
        """Validate that the CLI is working correctly"""
        print("🔍 Validating installation...")
        
        try:
            # Test basic CLI functionality
            result = subprocess.run([
                sys.executable, str(self.cli_script), "--help"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "YouTube Crawler CLI" in result.stdout:
                print("✅ CLI help system working")
            else:
                print(f"⚠️  CLI help system issues: {result.stderr}")
                return False
            
            # Test health check if available
            result = subprocess.run([
                sys.executable, str(self.cli_script), "health"
            ], capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                print("✅ Health check passed")
            else:
                print(f"⚠️  Health check issues detected")
                print(f"   This is normal if dependencies are missing")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("⚠️  Validation timed out - this may indicate installation issues")
            return False
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def create_usage_examples(self):
        """Create usage examples directory"""
        examples_dir = self.project_root / "examples"
        
        if not examples_dir.exists():
            examples_dir.mkdir()
        
        # Create CLI usage examples script
        examples_script = examples_dir / "cli_usage_examples.sh"
        
        examples_content = '''#!/bin/bash
# YouTube Crawler CLI Usage Examples
# ==================================

echo "🎬 YouTube Crawler CLI Usage Examples"
echo "====================================="

# Basic search
echo "1. Basic video search:"
echo "python cli.py search 'artificial intelligence' --max-results 20"
echo ""

# Advanced search with filters
echo "2. Advanced search with date filter:"
echo "python cli.py search 'machine learning' --max-results 50 --published-after 2025-01-01 --order date"
echo ""

# Download videos
echo "3. Download specific videos:"
echo "python cli.py download VIDEO_ID1 VIDEO_ID2 --audio-only --subtitles"
echo ""

# Interactive mode
echo "4. Interactive exploration mode:"
echo "python cli.py interactive"
echo ""

# Health check
echo "5. Check system health:"
echo "python cli.py health"
echo ""

# Using configuration file
echo "6. Search with configuration file:"
echo "python cli.py search 'AI trends' --config configs/example_search_config.yaml"
echo ""

# Batch download from file
echo "7. Batch download from file:"
echo "python cli.py download --input-file video_ids.txt --concurrent 3"
echo ""

# Monitor channels (requires separate monitor command)
echo "8. Monitor YouTube channels:"
echo "python cli.py monitor --config configs/example_monitoring_config.yaml"
echo ""

# Export to different formats
echo "9. Export search results:"
echo "python cli.py search 'technology news' --format csv --output results.csv"
echo ""

# Advanced download with specific format
echo "10. Download with specific quality:"
echo "python cli.py download VIDEO_ID --format 'best[height<=1080]' --subtitles --thumbnails"
'''
        
        try:
            with open(examples_script, 'w') as f:
                f.write(examples_content)
            os.chmod(examples_script, 0o755)
            print(f"✅ Usage examples created: {examples_script}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to create usage examples: {e}")
            return False
    
    def run_setup(self, install_deps=True, enhanced=True, download_support=True, 
                  create_symlink=False, symlink_dir="/usr/local/bin"):
        """Run complete setup process"""
        print("🎬 Enhanced YouTube Crawler CLI Setup")
        print("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Install dependencies
        if install_deps:
            if not self.install_dependencies(enhanced=enhanced, download_support=download_support):
                print("❌ Failed to install critical dependencies")
                return False
        
        # Check external dependencies
        self.check_external_dependencies()
        
        # Create executable
        if not self.create_executable():
            return False
        
        # Setup configuration examples
        self.create_configuration_examples()
        
        # Create environment template
        self.setup_environment_template()
        
        # Create usage examples
        self.create_usage_examples()
        
        # Create symlink if requested
        if create_symlink:
            self.create_symlink(symlink_dir)
        
        # Validate installation
        if not self.validate_installation():
            print("⚠️  Installation completed with warnings")
        else:
            print("✅ Installation completed successfully")
        
        # Print usage information
        self.print_usage_info(create_symlink)
        
        return True
    
    def print_usage_info(self, has_symlink=False):
        """Print usage information after setup"""
        print("\n" + "=" * 60)
        print("🎉 Setup Complete!")
        print("=" * 60)
        
        print("\n📋 Usage Examples:")
        
        if has_symlink:
            print("  youtube-crawler search 'AI trends' --max-results 50")
            print("  youtube-crawler download VIDEO_ID --audio-only")
            print("  youtube-crawler interactive")
        
        print(f"  python {self.cli_script} search 'AI trends' --max-results 50")
        print(f"  {self.executable_script} download VIDEO_ID --audio-only")
        print(f"  {self.executable_script} interactive")
        
        print("\n⚙️  Configuration:")
        print(f"  Configuration examples: {self.project_root}/configs/")
        print(f"  Environment template: {self.project_root}/.env.template")
        print(f"  Usage examples: {self.project_root}/examples/")
        
        print("\n🔧 Basic Commands:")
        print("  # Search for videos")
        print(f"  python {self.cli_script} search 'machine learning' --max-results 20")
        print("")
        print("  # Download video")
        print(f"  python {self.cli_script} download VIDEO_ID --format 'best[height<=720]'")
        print("")
        print("  # Interactive mode")
        print(f"  python {self.cli_script} interactive")
        print("")
        print("  # Health check")
        print(f"  python {self.cli_script} health")
        print("")
        print("  # Use configuration file")
        print(f"  python {self.cli_script} search 'AI' --config configs/example_search_config.yaml")
        
        print("\n📚 Get Help:")
        print(f"  python {self.cli_script} --help")
        print(f"  python {self.cli_script} search --help")
        print(f"  python {self.cli_script} download --help")
        
        print("\n🔑 API Key Setup:")
        print("  1. Get YouTube Data API key: https://console.developers.google.com/")
        print("  2. Set environment variable: export YOUTUBE_API_KEY=your_key")
        print("  3. Or add to .env file: YOUTUBE_API_KEY=your_key")
        
        print("\n🎯 Next Steps:")
        print("  1. Copy .env.template to .env and configure your settings")
        print("  2. Get a YouTube Data API key (optional but recommended)")
        print("  3. Run your first search to test the installation")
        print("  4. Explore the interactive mode for hands-on learning")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Enhanced YouTube Crawler CLI Setup")
    parser.add_argument(
        "--no-deps", 
        action="store_true", 
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--basic", 
        action="store_true", 
        help="Install only basic dependencies (no enhanced features)"
    )
    parser.add_argument(
        "--no-download-support", 
        action="store_true", 
        help="Skip download support dependencies (yt-dlp, etc.)"
    )
    parser.add_argument(
        "--symlink", 
        action="store_true", 
        help="Create system-wide symlink"
    )
    parser.add_argument(
        "--symlink-dir", 
        default="/usr/local/bin",
        help="Directory for symlink (default: /usr/local/bin)"
    )
    parser.add_argument(
        "--check-deps-only",
        action="store_true",
        help="Only check dependencies, don't install"
    )
    
    args = parser.parse_args()
    
    setup = YouTubeCLISetup()
    
    if args.check_deps_only:
        # Just check dependencies
        setup.check_python_version()
        setup.check_external_dependencies()
        return
    
    success = setup.run_setup(
        install_deps=not args.no_deps,
        enhanced=not args.basic,
        download_support=not args.no_download_support,
        create_symlink=args.symlink,
        symlink_dir=args.symlink_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()