"""
Gen Crawler CLI Examples
========================

Examples demonstrating CLI usage for various scenarios.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import subprocess
import sys
from pathlib import Path

# Example CLI commands for different use cases
CLI_EXAMPLES = {
    "basic_crawl": {
        "description": "Basic website crawl with markdown export",
        "command": [
            "gen-crawler", "crawl", "https://httpbin.org",
            "--output", "./basic_results",
            "--format", "markdown",
            "--max-pages", "10",
            "--verbose"
        ]
    },
    
    "news_site_crawl": {
        "description": "News site crawl with conflict monitoring",
        "command": [
            "gen-crawler", "crawl", "https://example.com",
            "--output", "./news_results",
            "--format", "json",
            "--max-pages", "100",
            "--include-patterns", "article,news,story,breaking",
            "--exclude-patterns", "tag,category,archive,login",
            "--conflict-keywords", "war,violence,crisis,protest,attack",
            "--crawl-delay", "3.0",
            "--max-concurrent", "3"
        ]
    },
    
    "research_crawl": {
        "description": "Research-focused crawl with high quality filtering",
        "command": [
            "gen-crawler", "crawl", "https://example.com",
            "--output", "./research_results",
            "--format", "markdown",
            "--max-pages", "500",
            "--min-content-length", "500",
            "--include-patterns", "research,paper,study,analysis,report",
            "--extraction-method", "trafilatura",
            "--save-raw-html",
            "--verbose", "--verbose"
        ]
    },
    
    "multi_site_crawl": {
        "description": "Multiple sites crawl with comprehensive export",
        "command": [
            "gen-crawler", "crawl", 
            "https://httpbin.org", "https://example.com",
            "--output", "./multi_site_results",
            "--format", "json",
            "--max-pages", "50",
            "--compress",
            "--enable-database",
            "--database-url", "sqlite:///crawl_results.db"
        ]
    },
    
    "config_creation": {
        "description": "Create configuration files for different scenarios",
        "commands": [
            {
                "name": "news_config",
                "command": [
                    "gen-crawler", "config", "--create",
                    "--template", "news",
                    "--output", "./news_config.json"
                ]
            },
            {
                "name": "research_config", 
                "command": [
                    "gen-crawler", "config", "--create",
                    "--template", "research",
                    "--output", "./research_config.json"
                ]
            },
            {
                "name": "monitoring_config",
                "command": [
                    "gen-crawler", "config", "--create",
                    "--template", "monitoring", 
                    "--output", "./monitoring_config.json"
                ]
            }
        ]
    },
    
    "config_usage": {
        "description": "Use existing configuration file",
        "command": [
            "gen-crawler", "crawl", "https://example.com",
            "--config", "./news_config.json",
            "--output", "./config_based_results",
            "--format", "markdown"
        ]
    },
    
    "export_examples": {
        "description": "Export existing crawl data to different formats",
        "commands": [
            {
                "name": "markdown_export",
                "command": [
                    "gen-crawler", "export", "./crawl_results.json",
                    "--format", "markdown",
                    "--output", "./markdown_export",
                    "--organize-by", "site"
                ]
            },
            {
                "name": "html_export",
                "command": [
                    "gen-crawler", "export", "./crawl_results.json",
                    "--format", "html",
                    "--output", "./html_export",
                    "--filter-quality", "0.7"
                ]
            },
            {
                "name": "csv_export",
                "command": [
                    "gen-crawler", "export", "./crawl_results.json",
                    "--format", "csv",
                    "--output", "./csv_export",
                    "--filter-type", "article"
                ]
            }
        ]
    },
    
    "analysis_examples": {
        "description": "Analyze existing crawl data",
        "command": [
            "gen-crawler", "analyze", "./crawl_results.json",
            "--output", "./analysis_report.json",
            "--conflict-analysis",
            "--quality-threshold", "0.6"
        ]
    },
    
    "stealth_crawl": {
        "description": "Stealth crawling with proxy support",
        "command": [
            "gen-crawler", "crawl", "https://example.com",
            "--output", "./stealth_results",
            "--format", "json",
            "--random-user-agents",
            "--crawl-delay", "5.0",
            "--max-concurrent", "2",
            "--proxy-list", "./proxies.txt",
            "--ignore-robots-txt"
        ]
    },
    
    "performance_optimized": {
        "description": "Performance optimized crawl",
        "command": [
            "gen-crawler", "crawl", "https://example.com",
            "--output", "./performance_results",
            "--format", "json",
            "--max-pages", "1000",
            "--max-concurrent", "10",
            "--crawl-delay", "1.0",
            "--request-timeout", "15",
            "--memory-limit", "2048",
            "--disable-image-extraction",
            "--compression-enabled"
        ]
    }
}

def run_example(example_name: str, dry_run: bool = True):
    """
    Run a CLI example.
    
    Args:
        example_name: Name of the example to run
        dry_run: If True, just print the command without executing
    """
    
    if example_name not in CLI_EXAMPLES:
        print(f"Unknown example: {example_name}")
        print(f"Available examples: {list(CLI_EXAMPLES.keys())}")
        return
    
    example = CLI_EXAMPLES[example_name]
    print(f"\n📋 {example['description']}")
    print("=" * 60)
    
    if 'command' in example:
        # Single command
        command = example['command']
        print(f"Command: {' '.join(command)}")
        
        if not dry_run:
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                print(f"\nReturn code: {result.returncode}")
                if result.stdout:
                    print(f"Output:\n{result.stdout}")
                if result.stderr:
                    print(f"Errors:\n{result.stderr}")
            except Exception as e:
                print(f"Error running command: {e}")
    
    elif 'commands' in example:
        # Multiple commands
        for cmd_info in example['commands']:
            name = cmd_info['name']
            command = cmd_info['command']
            print(f"\n{name}: {' '.join(command)}")
            
            if not dry_run:
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    print(f"Return code: {result.returncode}")
                    if result.returncode != 0 and result.stderr:
                        print(f"Errors: {result.stderr}")
                except Exception as e:
                    print(f"Error running {name}: {e}")

def print_all_examples():
    """Print all available CLI examples."""
    
    print("🎯 Gen Crawler CLI Examples")
    print("=" * 50)
    
    for name, example in CLI_EXAMPLES.items():
        print(f"\n📌 {name}")
        print(f"   {example['description']}")
        
        if 'command' in example:
            command_str = ' '.join(example['command'])
            if len(command_str) > 80:
                # Split long commands for readability
                parts = []
                current_part = ""
                for part in example['command']:
                    if len(current_part + " " + part) > 80:
                        if current_part:
                            parts.append(current_part + " \\")
                            current_part = "  " + part
                        else:
                            parts.append(part)
                    else:
                        current_part += " " + part if current_part else part
                if current_part:
                    parts.append(current_part)
                print(f"   {chr(10).join(parts)}")
            else:
                print(f"   {command_str}")
        
        elif 'commands' in example:
            print(f"   Multiple commands:")
            for cmd_info in example['commands']:
                print(f"   - {cmd_info['name']}: {' '.join(cmd_info['command'][:3])}...")

def create_example_scripts():
    """Create shell scripts for each example."""
    
    scripts_dir = Path("./cli_example_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    print(f"📝 Creating example scripts in {scripts_dir}")
    
    for name, example in CLI_EXAMPLES.items():
        script_path = scripts_dir / f"{name}.sh"
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# {example['description']}\n\n")
            
            if 'command' in example:
                command_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in example['command'])
                f.write(f"{command_str}\n")
            
            elif 'commands' in example:
                for cmd_info in example['commands']:
                    f.write(f"# {cmd_info['name']}\n")
                    command_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd_info['command'])
                    f.write(f"{command_str}\n\n")
        
        # Make script executable
        script_path.chmod(0o755)
        print(f"   ✅ {script_path}")

def create_sample_configs():
    """Create sample configuration files."""
    
    configs_dir = Path("./sample_configs")
    configs_dir.mkdir(exist_ok=True)
    
    print(f"⚙️ Creating sample configurations in {configs_dir}")
    
    # Run config creation examples
    if 'config_creation' in CLI_EXAMPLES:
        for cmd_info in CLI_EXAMPLES['config_creation']['commands']:
            command = cmd_info['command'].copy()
            # Update output path to use our configs directory
            if '--output' in command:
                output_index = command.index('--output') + 1
                if output_index < len(command):
                    filename = Path(command[output_index]).name
                    command[output_index] = str(configs_dir / filename)
            
            print(f"   Creating {cmd_info['name']}...")
            try:
                subprocess.run(command, check=True, capture_output=True)
                print(f"   ✅ {cmd_info['name']} created")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to create {cmd_info['name']}: {e}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

def main():
    """Main function for CLI examples."""
    
    if len(sys.argv) < 2:
        print("Usage: python cli_examples.py <command> [example_name]")
        print("\nCommands:")
        print("  list          - List all examples")
        print("  show <name>   - Show specific example")
        print("  run <name>    - Run specific example (dry run)")
        print("  execute <name>- Execute specific example")
        print("  scripts       - Create shell scripts for all examples")
        print("  configs       - Create sample configuration files")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        print_all_examples()
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("Please specify example name")
            return
        run_example(sys.argv[2], dry_run=True)
    
    elif command == "run":
        if len(sys.argv) < 3:
            print("Please specify example name")
            return
        run_example(sys.argv[2], dry_run=True)
    
    elif command == "execute":
        if len(sys.argv) < 3:
            print("Please specify example name")
            return
        print("⚠️ This will execute the actual command. Continue? (y/N): ", end="")
        if input().lower() == 'y':
            run_example(sys.argv[2], dry_run=False)
        else:
            print("Cancelled.")
    
    elif command == "scripts":
        create_example_scripts()
    
    elif command == "configs":
        create_sample_configs()
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()