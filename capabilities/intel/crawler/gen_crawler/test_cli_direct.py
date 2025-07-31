#!/usr/bin/env python3
"""
Direct CLI Testing Script
=========================

Test the CLI components directly to verify they work correctly.
"""

import sys
import os
from pathlib import Path

# Add the gen_crawler package directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_cli_components():
    """Test CLI components directly."""
    print("🧪 Testing CLI Components Directly")
    print("=" * 50)
    
    try:
        print("\n📋 Testing CLI Parser...")
        from cli.main import create_cli_parser
        parser = create_cli_parser()
        print("   ✅ CLI parser created successfully")
        
        # Test help generation
        help_text = parser.format_help()
        print(f"   ✅ Help text generated ({len(help_text)} characters)")
        
    except Exception as e:
        print(f"   ❌ CLI parser test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n🛠️ Testing CLI Utils...")
        from cli.utils import validate_urls, setup_logging
        
        # Test URL validation
        valid_urls, invalid_urls = validate_urls([
            'https://example.com',
            'invalid-url',
            'http://test.org'
        ])
        print(f"   ✅ URL validation: {len(valid_urls)} valid, {len(invalid_urls)} invalid")
        
        # Test logging setup
        log_level = setup_logging(verbose=1, quiet=False, log_file=None)
        print(f"   ✅ Logging setup successful (level: {log_level})")
        
    except Exception as e:
        print(f"   ❌ CLI utils test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n📤 Testing CLI Exporters...")
        from cli.exporters import MarkdownExporter, JSONExporter
        
        # Test exporter creation
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            md_exporter = MarkdownExporter(output_dir=temp_dir)
            print("   ✅ Markdown exporter created")
            
            json_exporter = JSONExporter(output_dir=temp_dir)
            print("   ✅ JSON exporter created")
        
    except Exception as e:
        print(f"   ❌ CLI exporters test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n⚙️ Testing CLI Commands Import...")
        from cli.commands import crawl_command, analyze_command, config_command, export_command
        print("   ✅ All CLI commands imported successfully")
        
    except Exception as e:
        print(f"   ❌ CLI commands test failed: {e}")
        import traceback
        traceback.print_exc()

def test_argument_parsing():
    """Test argument parsing with various scenarios."""
    print("\n🎯 Testing Argument Parsing")
    print("=" * 50)
    
    try:
        from cli.main import create_cli_parser
        parser = create_cli_parser()
        
        # Test basic crawl command
        args = parser.parse_args([
            'crawl', 'https://example.com',
            '--max-pages', '10',
            '--format', 'json',
            '--output', './test_output'
        ])
        print(f"   ✅ Basic crawl: {args.command}, {args.max_pages} pages, {args.format} format")
        
        # Test config command
        args = parser.parse_args([
            'config', '--create',
            '--template', 'news',
            '--output', './config.json'
        ])
        print(f"   ✅ Config creation: {args.template} template")
        
        # Test analyze command
        args = parser.parse_args([
            'analyze', './data.json',
            '--format', 'json',
            '--quality-threshold', '0.7'
        ])
        print(f"   ✅ Analysis: {args.input}, threshold {args.quality_threshold}")
        
    except Exception as e:
        print(f"   ❌ Argument parsing test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_cli_components()
    test_argument_parsing()
    print("\n🎉 CLI testing completed!")