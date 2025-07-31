"""
CLI Tests
=========

Test suite for gen_crawler CLI interface including command parsing,
argument validation, and command execution.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import unittest
import tempfile
import json
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from io import StringIO
import sys

try:
    from ..cli.main import create_cli_parser, build_config_from_args, main_cli
    from ..cli.commands import crawl_command, analyze_command, config_command, export_command
    from ..cli.utils import setup_logging, validate_urls, format_results
    from ..cli.exporters import MarkdownExporter, JSONExporter, CSVExporter
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

if not CLI_AVAILABLE:
    import pytest
    pytest.skip("CLI components not available", allow_module_level=True)

class TestCLIParser(unittest.TestCase):
    """Test CLI argument parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_cli_parser()
    
    def test_parser_creation(self):
        """Test basic parser creation."""
        self.assertIsInstance(self.parser, argparse.ArgumentParser)
        
        # Test help output
        with patch('sys.exit'):
            with patch('sys.stderr', new=StringIO()) as fake_stderr:
                try:
                    self.parser.parse_args(['--help'])
                except SystemExit:
                    pass
                
                help_output = fake_stderr.getvalue()
                # Should contain main commands
                self.assertIn('crawl', help_output)
                self.assertIn('analyze', help_output)
                self.assertIn('config', help_output)
                self.assertIn('export', help_output)
    
    def test_crawl_command_parsing(self):
        """Test crawl command argument parsing."""
        # Basic crawl command
        args = self.parser.parse_args([
            'crawl', 'https://example.com',
            '--output', './results',
            '--format', 'markdown'
        ])
        
        self.assertEqual(args.command, 'crawl')
        self.assertEqual(args.urls, ['https://example.com'])
        self.assertEqual(args.output, './results')
        self.assertEqual(args.format, 'markdown')
    
    def test_crawl_advanced_options(self):
        """Test advanced crawl options parsing."""
        args = self.parser.parse_args([
            'crawl', 'https://example.com',
            '--max-pages', '100',
            '--max-concurrent', '5',
            '--crawl-delay', '2.5',
            '--include-patterns', 'article,news,story',
            '--exclude-patterns', 'tag,archive',
            '--conflict-keywords', 'war,violence,crisis',
            '--user-agent', 'Custom Agent',
            '--enable-database',
            '--database-url', 'postgresql://user:pass@localhost/db'
        ])
        
        self.assertEqual(args.max_pages, 100)
        self.assertEqual(args.max_concurrent, 5)
        self.assertEqual(args.crawl_delay, 2.5)
        self.assertEqual(args.include_patterns, 'article,news,story')
        self.assertEqual(args.exclude_patterns, 'tag,archive')
        self.assertEqual(args.conflict_keywords, 'war,violence,crisis')
        self.assertEqual(args.user_agent, 'Custom Agent')
        self.assertTrue(args.enable_database)
        self.assertEqual(args.database_url, 'postgresql://user:pass@localhost/db')
    
    def test_config_command_parsing(self):
        """Test config command parsing."""
        # Create config
        args = self.parser.parse_args([
            'config', '--create',
            '--template', 'news',
            '--output', './config.json'
        ])
        
        self.assertEqual(args.command, 'config')
        self.assertTrue(args.create)
        self.assertEqual(args.template, 'news')
        self.assertEqual(args.output, './config.json')
        
        # Validate config
        args = self.parser.parse_args([
            'config', '--validate', './config.json'
        ])
        
        self.assertEqual(args.validate, './config.json')
    
    def test_export_command_parsing(self):
        """Test export command parsing."""
        args = self.parser.parse_args([
            'export', './results.json',
            '--format', 'markdown',
            '--output', './markdown/',
            '--filter-quality', '0.7',
            '--organize-by', 'site'
        ])
        
        self.assertEqual(args.command, 'export')
        self.assertEqual(args.input, './results.json')
        self.assertEqual(args.format, 'markdown')
        self.assertEqual(args.output, './markdown/')
        self.assertEqual(args.filter_quality, 0.7)
        self.assertEqual(args.organize_by, 'site')
    
    def test_analyze_command_parsing(self):
        """Test analyze command parsing."""
        args = self.parser.parse_args([
            'analyze', './results.json',
            '--output', './analysis.json',
            '--conflict-analysis',
            '--quality-threshold', '0.6'
        ])
        
        self.assertEqual(args.command, 'analyze')
        self.assertEqual(args.input, './results.json')
        self.assertEqual(args.output, './analysis.json')
        self.assertTrue(args.conflict_analysis)
        self.assertEqual(args.quality_threshold, 0.6)
    
    def test_global_options(self):
        """Test global CLI options."""
        args = self.parser.parse_args([
            'crawl', 'https://example.com',
            '--verbose', '--verbose',
            '--log-file', './crawl.log'
        ])
        
        self.assertEqual(args.verbose, 2)
        self.assertEqual(args.log_file, './crawl.log')
        
        # Test quiet option
        args = self.parser.parse_args([
            'crawl', 'https://example.com',
            '--quiet'
        ])
        
        self.assertTrue(args.quiet)

class TestConfigBuilding(unittest.TestCase):
    """Test configuration building from CLI arguments."""
    
    def test_build_basic_config(self):
        """Test building basic configuration."""
        parser = create_cli_parser()
        args = parser.parse_args([
            'crawl', 'https://example.com',
            '--max-pages', '100',
            '--max-concurrent', '3',
            '--crawl-delay', '1.5'
        ])
        
        config = build_config_from_args(args)
        
        self.assertEqual(config['performance']['max_pages_per_site'], 100)
        self.assertEqual(config['performance']['max_concurrent'], 3)
        self.assertEqual(config['performance']['crawl_delay'], 1.5)
    
    def test_build_content_filter_config(self):
        """Test building content filter configuration."""
        parser = create_cli_parser()
        args = parser.parse_args([
            'crawl', 'https://example.com',
            '--min-content-length', '200',
            '--include-patterns', 'article,news',
            '--exclude-patterns', 'tag,login',
            '--exclude-extensions', '.pdf,.doc'
        ])
        
        config = build_config_from_args(args)
        
        self.assertEqual(config['content_filters']['min_content_length'], 200)
        self.assertEqual(config['content_filters']['include_patterns'], ['article', 'news'])
        self.assertEqual(config['content_filters']['exclude_patterns'], ['tag', 'login'])
        self.assertEqual(config['content_filters']['exclude_extensions'], ['.pdf', '.doc'])
    
    def test_build_database_config(self):
        """Test building database configuration."""
        parser = create_cli_parser()
        args = parser.parse_args([
            'crawl', 'https://example.com',
            '--enable-database',
            '--database-url', 'postgresql://user:pass@localhost/db',
            '--database-table-prefix', 'test_'
        ])
        
        config = build_config_from_args(args)
        
        self.assertTrue(config['database']['enable_database'])
        self.assertEqual(config['database']['connection_string'], 'postgresql://user:pass@localhost/db')
        self.assertEqual(config['database']['table_prefix'], 'test_')
    
    def test_build_stealth_config(self):
        """Test building stealth configuration."""
        parser = create_cli_parser()
        args = parser.parse_args([
            'crawl', 'https://example.com',
            '--user-agent', 'Custom Agent',
            '--random-user-agents',
            '--ignore-robots-txt'
        ])
        
        config = build_config_from_args(args)
        
        self.assertEqual(config['stealth']['user_agent'], 'Custom Agent')
        self.assertTrue(config['stealth']['random_user_agents'])
        self.assertFalse(config['stealth']['respect_robots_txt'])

class TestCLIUtils(unittest.TestCase):
    """Test CLI utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup with different verbosity levels."""
        # Test quiet mode
        log_level = setup_logging(verbose=0, quiet=True, log_file=None)
        self.assertEqual(log_level, 40)  # ERROR level
        
        # Test normal mode
        log_level = setup_logging(verbose=0, quiet=False, log_file=None)
        self.assertEqual(log_level, 20)  # INFO level
        
        # Test verbose mode
        log_level = setup_logging(verbose=1, quiet=False, log_file=None)
        self.assertEqual(log_level, 10)  # DEBUG level
        
        # Test very verbose mode
        log_level = setup_logging(verbose=2, quiet=False, log_file=None)
        self.assertEqual(log_level, 10)  # DEBUG level
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / 'test.log'
            
            setup_logging(verbose=1, quiet=False, log_file=str(log_file))
            
            # File should be created (though may be empty)
            self.assertTrue(log_file.exists())
    
    def test_validate_urls(self):
        """Test URL validation."""
        test_urls = [
            'https://example.com',
            'http://test.org',
            'example.com',  # Should be converted to https://
            'invalid-url',
            'ftp://example.com',  # Invalid protocol
            '',  # Empty
            '   ',  # Whitespace only
        ]
        
        valid_urls, invalid_urls = validate_urls(test_urls)
        
        # Should have valid URLs
        self.assertIn('https://example.com', valid_urls)
        self.assertIn('http://test.org', valid_urls)
        self.assertIn('https://example.com', valid_urls)  # Converted
        
        # Should have invalid URLs
        self.assertIn('invalid-url', invalid_urls)
        self.assertIn('ftp://example.com', invalid_urls)
    
    def test_format_results_summary(self):
        """Test result formatting in summary mode."""
        mock_results = [
            {
                'base_url': 'https://example.com',
                'total_pages': 10,
                'successful_pages': 8,
                'failed_pages': 2
            },
            {
                'base_url': 'https://test.org',
                'total_pages': 5,
                'successful_pages': 5,
                'failed_pages': 0
            }
        ]
        
        formatted = format_results(mock_results, 'summary')
        
        self.assertIn('CRAWL RESULTS SUMMARY', formatted)
        self.assertIn('Sites crawled: 2', formatted)
        self.assertIn('Total pages: 15', formatted)
        self.assertIn('Successful pages: 13', formatted)
        self.assertIn('example.com', formatted)
        self.assertIn('test.org', formatted)
    
    def test_format_results_detailed(self):
        """Test result formatting in detailed mode."""
        mock_results = [
            {
                'base_url': 'https://example.com',
                'total_pages': 3,
                'successful_pages': 2,
                'failed_pages': 1,
                'success_rate': 66.7,
                'total_time': 10.5,
                'pages': [
                    {
                        'title': 'Article 1',
                        'url': 'https://example.com/article1',
                        'word_count': 500,
                        'content_type': 'article'
                    }
                ]
            }
        ]
        
        formatted = format_results(mock_results, 'detailed')
        
        self.assertIn('SITE 1: https://example.com', formatted)
        self.assertIn('Total pages: 3', formatted)
        self.assertIn('Success rate: 66.7%', formatted)
        self.assertIn('Article 1', formatted)
    
    def test_format_results_json(self):
        """Test result formatting in JSON mode."""
        mock_results = [{'test': 'data'}]
        
        formatted = format_results(mock_results, 'json')
        
        # Should be valid JSON
        parsed = json.loads(formatted)
        self.assertEqual(parsed, mock_results)

class TestCrawlCommand(unittest.TestCase):
    """Test crawl command implementation."""
    
    @patch('gen_crawler.cli.commands.GenCrawler')
    @patch('gen_crawler.cli.commands.create_gen_config')
    def test_crawl_command_execution(self, mock_create_config, mock_crawler_class):
        """Test crawl command execution."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.get_crawler_config.return_value = {'test': 'config'}
        mock_create_config.return_value = mock_config
        
        mock_crawler = MagicMock()
        mock_crawler.initialize = AsyncMock()
        mock_crawler.crawl_site = AsyncMock()
        mock_crawler.cleanup = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        
        # Mock site result
        from ..core.gen_crawler import GenSiteResult
        mock_site_result = GenSiteResult(
            base_url="https://example.com",
            total_pages=1,
            successful_pages=1,
            success_rate=100.0
        )
        mock_crawler.crawl_site.return_value = mock_site_result
        
        # Create test arguments
        parser = create_cli_parser()
        args = parser.parse_args([
            'crawl', 'https://example.com',
            '--output', './test_results',
            '--format', 'json'
        ])
        
        # Mock export function
        with patch('gen_crawler.cli.commands._export_crawl_results') as mock_export:
            mock_export.return_value = AsyncMock()
            
            # Run command
            import asyncio
            asyncio.run(crawl_command(args))
            
            # Verify calls
            mock_crawler.initialize.assert_called_once()
            mock_crawler.crawl_site.assert_called_once_with('https://example.com')
            mock_crawler.cleanup.assert_called_once()

class TestExportCommand(unittest.TestCase):
    """Test export command implementation."""
    
    def test_export_command_with_mock_data(self):
        """Test export command with mock data."""
        # Create test data
        test_data = [{
            'base_url': 'https://example.com',
            'pages': [{
                'url': 'https://example.com/article',
                'title': 'Test Article',
                'content': 'Test content',
                'success': True,
                'quality_score': 0.8
            }]
        }]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = Path(temp_dir) / 'input.json'
            with open(input_file, 'w') as f:
                json.dump(test_data, f)
            
            output_dir = Path(temp_dir) / 'output'
            
            # Create test arguments
            parser = create_cli_parser()
            args = parser.parse_args([
                'export', str(input_file),
                '--output', str(output_dir),
                '--format', 'json'
            ])
            
            # Mock the exporter
            with patch('gen_crawler.cli.commands.JSONExporter') as mock_exporter_class:
                mock_exporter = MagicMock()
                mock_exporter.export_results = AsyncMock()
                mock_exporter_class.return_value = mock_exporter
                
                # Run command
                import asyncio
                asyncio.run(export_command(args))
                
                # Verify exporter was called
                mock_exporter_class.assert_called_once()
                mock_exporter.export_results.assert_called_once()

class TestAnalyzeCommand(unittest.TestCase):
    """Test analyze command implementation."""
    
    def test_analyze_command_with_test_data(self):
        """Test analyze command with test data."""
        # Create test data
        test_data = [{
            'base_url': 'https://example.com',
            'pages': [
                {
                    'url': 'https://example.com/article1',
                    'title': 'Article 1',
                    'content_type': 'article',
                    'quality_score': 0.8,
                    'word_count': 500,
                    'success': True
                },
                {
                    'url': 'https://example.com/article2',
                    'title': 'Article 2',
                    'content_type': 'article',
                    'quality_score': 0.6,
                    'word_count': 300,
                    'success': True
                }
            ]
        }]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = Path(temp_dir) / 'input.json'
            with open(input_file, 'w') as f:
                json.dump(test_data, f)
            
            output_file = Path(temp_dir) / 'analysis.json'
            
            # Create test arguments
            parser = create_cli_parser()
            args = parser.parse_args([
                'analyze', str(input_file),
                '--output', str(output_file),
                '--quality-threshold', '0.7'
            ])
            
            # Run command
            import asyncio
            asyncio.run(analyze_command(args))
            
            # Verify output file
            self.assertTrue(output_file.exists())
            
            # Load and verify analysis
            with open(output_file, 'r') as f:
                analysis = json.load(f)
            
            self.assertIn('summary', analysis)
            self.assertIn('sites_analyzed', analysis['summary'])
            self.assertEqual(analysis['summary']['total_pages'], 2)

class TestConfigCommand(unittest.TestCase):
    """Test config command implementation."""
    
    def test_config_create_command(self):
        """Test config creation command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / 'config.json'
            
            # Create test arguments
            parser = create_cli_parser()
            args = parser.parse_args([
                'config', '--create',
                '--template', 'news',
                '--output', str(output_file)
            ])
            
            # Mock the config creation
            with patch('gen_crawler.cli.commands.create_gen_config') as mock_create_config:
                mock_config = MagicMock()
                mock_config.save_to_file = MagicMock()
                mock_create_config.return_value = mock_config
                
                # Run command
                import asyncio
                asyncio.run(config_command(args))
                
                # Verify config was created and saved
                mock_create_config.assert_called_once()
                mock_config.save_to_file.assert_called_once_with(output_file)
    
    def test_config_validate_command(self):
        """Test config validation command."""
        # Create test config file
        test_config = {
            'performance': {
                'max_pages_per_site': 100,
                'max_concurrent': 5
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(test_config, f)
            
            # Create test arguments
            parser = create_cli_parser()
            args = parser.parse_args([
                'config', '--validate', str(config_file)
            ])
            
            # Mock the config validation
            with patch('gen_crawler.cli.commands.create_gen_config') as mock_create_config:
                mock_config = MagicMock()
                mock_create_config.return_value = mock_config
                
                # Run command (should not raise exception)
                import asyncio
                asyncio.run(config_command(args))
                
                # Verify config was loaded
                mock_create_config.assert_called_once_with(config_file=config_file)

class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration scenarios."""
    
    def test_full_cli_workflow(self):
        """Test complete CLI workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Create configuration
            config_file = temp_path / 'test_config.json'
            
            parser = create_cli_parser()
            config_args = parser.parse_args([
                'config', '--create',
                '--template', 'basic',
                '--output', str(config_file)
            ])
            
            with patch('gen_crawler.cli.commands.create_gen_config') as mock_create_config:
                mock_config = MagicMock()
                mock_config.save_to_file = MagicMock()
                mock_create_config.return_value = mock_config
                
                import asyncio
                asyncio.run(config_command(config_args))
                
                mock_config.save_to_file.assert_called_once()
            
            # Step 2: Test with invalid URLs
            test_urls = ['https://example.com', 'invalid-url', '']
            valid_urls, invalid_urls = validate_urls(test_urls)
            
            self.assertEqual(len(valid_urls), 1)
            self.assertEqual(len(invalid_urls), 1)  # 'invalid-url'
            
            # Step 3: Test result formatting
            mock_results = [{
                'base_url': 'https://example.com',
                'total_pages': 5,
                'successful_pages': 4,
                'failed_pages': 1
            }]
            
            summary = format_results(mock_results, 'summary')
            self.assertIn('Sites crawled: 1', summary)
            
            detailed = format_results(mock_results, 'detailed')
            self.assertIn('SITE 1:', detailed)
            
            json_output = format_results(mock_results, 'json')
            parsed = json.loads(json_output)
            self.assertEqual(parsed, mock_results)

if __name__ == '__main__':
    unittest.main(verbosity=2)