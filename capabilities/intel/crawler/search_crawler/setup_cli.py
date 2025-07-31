#!/usr/bin/env python3
"""
Search Crawler CLI Setup Script
===============================

Setup script for installing and configuring the Search Crawler CLI.
Handles dependencies, creates symlinks, and validates the installation.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
import argparse


class CLISetup:
    """Setup and installation manager for Search Crawler CLI."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.cli_script = self.project_root / "cli.py"
        self.executable_script = self.project_root / "search_crawler_cli"
        
    def check_python_version(self):
        """Check if Python version is compatible."""
        if sys.version_info < (3, 8):
            print("❌ Error: Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            return False
        
        print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
        return True
    
    def install_dependencies(self, enhanced=True):
        """Install required dependencies."""
        print("📦 Installing dependencies...")
        
        # Core dependencies
        core_deps = [
            "aiohttp>=3.8.0",
            "beautifulsoup4>=4.9.0",
            "lxml>=4.6.0",
            "requests>=2.25.0",
            "httpx>=0.23.0"
        ]
        
        # Enhanced CLI dependencies
        enhanced_deps = [
            "rich>=12.0.0",
            "click>=8.0.0",
            "pyyaml>=5.4.0"
        ]
        
        # Optional content extraction dependencies
        optional_deps = [
            "trafilatura>=1.2.0",
            "newspaper3k>=0.2.8",
            "readability-lxml>=0.8.0"
        ]
        
        dependencies = core_deps[:]
        if enhanced:
            dependencies.extend(enhanced_deps)
            dependencies.extend(optional_deps)
        
        for dep in dependencies:
            try:
                print(f"Installing {dep}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"✅ {dep}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Warning: Failed to install {dep}")
        
        print("✅ Dependencies installation complete")
    
    def create_executable(self):
        """Create executable CLI script."""
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
import sys
from pathlib import Path

# Add search crawler to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from cli import main
    main()
''')
            
            os.chmod(self.executable_script, 0o755)
        
        print(f"✅ Executable created: {self.executable_script}")
        return True
    
    def create_symlink(self, target_dir="/usr/local/bin"):
        """Create system-wide symlink for the CLI."""
        try:
            target_path = Path(target_dir) / "search-crawler"
            
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
        """Validate that the CLI is working correctly."""
        print("🔍 Validating installation...")
        
        try:
            # Test basic CLI functionality
            result = subprocess.run([
                sys.executable, str(self.cli_script), "--version"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ CLI version check passed")
            else:
                print(f"⚠️  CLI version check failed: {result.stderr}")
            
            # Test health check
            result = subprocess.run([
                sys.executable, str(self.cli_script), "--health-check"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                print("✅ Health check passed")
            else:
                print(f"⚠️  Health check issues detected")
            
            # Test help
            result = subprocess.run([
                sys.executable, str(self.cli_script), "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "Search Crawler CLI" in result.stdout:
                print("✅ Help system working")
            else:
                print("⚠️  Help system issues")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def setup_examples(self):
        """Setup example configurations and scripts."""
        examples_dir = self.project_root / "examples"
        configs_dir = self.project_root / "configs"
        
        if examples_dir.exists():
            print(f"✅ Examples directory ready: {examples_dir}")
        
        if configs_dir.exists():
            print(f"✅ Configurations directory ready: {configs_dir}")
            
            # List example configs
            config_files = list(configs_dir.glob("*.yaml"))
            if config_files:
                print("Available example configurations:")
                for config_file in config_files:
                    print(f"  - {config_file.name}")
        
        return True
    
    def run_setup(self, install_deps=True, enhanced=True, create_symlink=False, symlink_dir="/usr/local/bin"):
        """Run complete setup process."""
        print("🚀 Search Crawler CLI Setup")
        print("=" * 40)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Install dependencies
        if install_deps:
            self.install_dependencies(enhanced=enhanced)
        
        # Create executable
        if not self.create_executable():
            return False
        
        # Create symlink if requested
        if create_symlink:
            self.create_symlink(symlink_dir)
        
        # Setup examples
        self.setup_examples()
        
        # Validate installation
        if not self.validate_installation():
            print("⚠️  Installation completed with warnings")
        else:
            print("✅ Installation completed successfully")
        
        # Print usage information
        self.print_usage_info(create_symlink)
        
        return True
    
    def print_usage_info(self, has_symlink=False):
        """Print usage information after setup."""
        print("\n" + "=" * 50)
        print("🎉 Setup Complete!")
        print("=" * 50)
        
        print("\nUsage:")
        
        if has_symlink:
            print("  search-crawler \"your query\" --mode conflict")
            print("  search-crawler --health-check")
            print("  search-crawler --interactive")
        
        print(f"  python {self.cli_script} \"your query\" --mode conflict")
        print(f"  {self.executable_script} --health-check")
        print(f"  {self.executable_script} --interactive")
        
        print("\nExamples:")
        print(f"  {self.project_root}/examples/cli_usage_examples.sh")
        
        print("\nConfiguration files:")
        print(f"  {self.project_root}/configs/")
        
        print("\nDocumentation:")
        print(f"  {self.project_root}/CLI_DOCUMENTATION.md")
        
        print("\nGet help:")
        print(f"  python {self.cli_script} --help")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Search Crawler CLI Setup")
    parser.add_argument(
        "--no-deps", 
        action="store_true", 
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--basic", 
        action="store_true", 
        help="Install only basic dependencies (no enhanced CLI features)"
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
    
    args = parser.parse_args()
    
    setup = CLISetup()
    
    success = setup.run_setup(
        install_deps=not args.no_deps,
        enhanced=not args.basic,
        create_symlink=args.symlink,
        symlink_dir=args.symlink_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()