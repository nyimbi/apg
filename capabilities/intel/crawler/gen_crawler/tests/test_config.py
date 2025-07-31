"""
Configuration Tests
==================

Test suite for gen_crawler configuration system including
settings validation, file handling, and environment integration.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch

try:
    from ..config.gen_config import (
        GenCrawlerConfig, GenCrawlerSettings, ContentFilterConfig,
        DatabaseConfig, PerformanceConfig, AdaptiveConfig, StealthConfig,
        create_gen_config, get_default_gen_config, load_gen_config_from_file
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

if not CONFIG_AVAILABLE:
    import pytest
    pytest.skip("Configuration components not available", allow_module_level=True)

class TestContentFilterConfig(unittest.TestCase):
    """Test ContentFilterConfig data class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ContentFilterConfig()
        
        self.assertEqual(config.min_content_length, 100)
        self.assertIn('.pdf', config.exclude_extensions)
        self.assertIn('article', config.include_patterns)
        self.assertIn('tag', config.exclude_patterns)
        self.assertEqual(config.max_content_length, 1000000)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ContentFilterConfig(
            min_content_length=200,
            exclude_extensions=['.pdf', '.doc'],
            include_patterns=['news', 'story'],
            exclude_patterns=['admin', 'login']
        )
        
        self.assertEqual(config.min_content_length, 200)
        self.assertEqual(config.exclude_extensions, ['.pdf', '.doc'])
        self.assertEqual(config.include_patterns, ['news', 'story'])
        self.assertEqual(config.exclude_patterns, ['admin', 'login'])

class TestDatabaseConfig(unittest.TestCase):
    """Test DatabaseConfig data class."""
    
    def test_default_values(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        
        self.assertFalse(config.enable_database)
        self.assertIsNone(config.connection_string)
        self.assertEqual(config.pool_size, 10)
        self.assertEqual(config.table_prefix, "gen_crawler_")
    
    def test_custom_values(self):
        """Test custom database configuration."""
        config = DatabaseConfig(
            enable_database=True,
            connection_string="postgresql://user:pass@localhost/db",
            pool_size=20,
            table_prefix="test_"
        )
        
        self.assertTrue(config.enable_database)
        self.assertEqual(config.connection_string, "postgresql://user:pass@localhost/db")
        self.assertEqual(config.pool_size, 20)
        self.assertEqual(config.table_prefix, "test_")

class TestPerformanceConfig(unittest.TestCase):
    """Test PerformanceConfig data class."""
    
    def test_default_values(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        
        self.assertEqual(config.max_pages_per_site, 500)
        self.assertEqual(config.max_concurrent, 5)
        self.assertEqual(config.request_timeout, 30)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.crawl_delay, 2.0)
        self.assertEqual(config.max_depth, 10)
    
    def test_custom_values(self):
        """Test custom performance configuration."""
        config = PerformanceConfig(
            max_pages_per_site=1000,
            max_concurrent=10,
            crawl_delay=1.0,
            memory_limit_mb=2048
        )
        
        self.assertEqual(config.max_pages_per_site, 1000)
        self.assertEqual(config.max_concurrent, 10)
        self.assertEqual(config.crawl_delay, 1.0)
        self.assertEqual(config.memory_limit_mb, 2048)

class TestAdaptiveConfig(unittest.TestCase):
    """Test AdaptiveConfig data class."""
    
    def test_default_values(self):
        """Test default adaptive configuration."""
        config = AdaptiveConfig()
        
        self.assertTrue(config.enable_adaptive_crawling)
        self.assertEqual(config.strategy_switching_threshold, 0.8)
        self.assertTrue(config.performance_monitoring)
        self.assertTrue(config.auto_optimize)
    
    def test_custom_values(self):
        """Test custom adaptive configuration."""
        config = AdaptiveConfig(
            enable_adaptive_crawling=False,
            strategy_switching_threshold=0.9,
            performance_monitoring=False
        )
        
        self.assertFalse(config.enable_adaptive_crawling)
        self.assertEqual(config.strategy_switching_threshold, 0.9)
        self.assertFalse(config.performance_monitoring)

class TestStealthConfig(unittest.TestCase):
    """Test StealthConfig data class."""
    
    def test_default_values(self):
        """Test default stealth configuration."""
        config = StealthConfig()
        
        self.assertTrue(config.enable_stealth)
        self.assertEqual(config.user_agent, 'GenCrawler/1.0 (+https://datacraft.co.ke)')
        self.assertFalse(config.random_user_agents)
        self.assertFalse(config.enable_proxy_rotation)
        self.assertTrue(config.respect_robots_txt)
    
    def test_custom_values(self):
        """Test custom stealth configuration."""
        config = StealthConfig(
            enable_stealth=False,
            user_agent="Custom Agent",
            random_user_agents=True,
            proxy_list=['proxy1', 'proxy2']
        )
        
        self.assertFalse(config.enable_stealth)
        self.assertEqual(config.user_agent, "Custom Agent")
        self.assertTrue(config.random_user_agents)
        self.assertEqual(config.proxy_list, ['proxy1', 'proxy2'])

class TestGenCrawlerSettings(unittest.TestCase):
    """Test GenCrawlerSettings composite configuration."""
    
    def test_default_settings(self):
        """Test default settings creation."""
        settings = GenCrawlerSettings()
        
        self.assertIsInstance(settings.content_filters, ContentFilterConfig)
        self.assertIsInstance(settings.database, DatabaseConfig)
        self.assertIsInstance(settings.performance, PerformanceConfig)
        self.assertIsInstance(settings.adaptive, AdaptiveConfig)
        self.assertIsInstance(settings.stealth, StealthConfig)
        
        self.assertTrue(settings.enable_content_analysis)
        self.assertTrue(settings.enable_image_extraction)
        self.assertFalse(settings.save_raw_html)
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        settings = GenCrawlerSettings()
        settings_dict = settings.to_dict()
        
        self.assertIsInstance(settings_dict, dict)
        self.assertIn('content_filters', settings_dict)
        self.assertIn('database', settings_dict)
        self.assertIn('performance', settings_dict)
        self.assertIn('adaptive', settings_dict)
        self.assertIn('stealth', settings_dict)
        
        # Test nested structure
        self.assertIn('min_content_length', settings_dict['content_filters'])
        self.assertIn('enable_database', settings_dict['database'])
        self.assertIn('max_pages_per_site', settings_dict['performance'])
    
    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        test_dict = {
            'content_filters': {
                'min_content_length': 200,
                'include_patterns': ['test', 'example']
            },
            'performance': {
                'max_pages_per_site': 1000,
                'max_concurrent': 10
            },
            'enable_content_analysis': False
        }
        
        settings = GenCrawlerSettings.from_dict(test_dict)
        
        self.assertEqual(settings.content_filters.min_content_length, 200)
        self.assertEqual(settings.content_filters.include_patterns, ['test', 'example'])
        self.assertEqual(settings.performance.max_pages_per_site, 1000)
        self.assertEqual(settings.performance.max_concurrent, 10)
        self.assertFalse(settings.enable_content_analysis)
    
    def test_partial_dict_creation(self):
        """Test creation from partial dictionary."""
        partial_dict = {
            'performance': {
                'max_pages_per_site': 100
            }
        }
        
        settings = GenCrawlerSettings.from_dict(partial_dict)
        
        # Updated value
        self.assertEqual(settings.performance.max_pages_per_site, 100)
        
        # Default values should remain
        self.assertEqual(settings.performance.max_concurrent, 5)
        self.assertEqual(settings.content_filters.min_content_length, 100)

class TestGenCrawlerConfig(unittest.TestCase):
    """Test GenCrawlerConfig management class."""
    
    def test_default_config_creation(self):
        """Test creation with default settings."""
        config = GenCrawlerConfig()
        
        self.assertIsInstance(config.settings, GenCrawlerSettings)
        self.assertIsNone(config.config_file)
    
    def test_config_with_settings(self):
        """Test creation with custom settings."""
        custom_settings = GenCrawlerSettings()
        custom_settings.performance.max_pages_per_site = 1000
        
        config = GenCrawlerConfig(settings=custom_settings)
        
        self.assertEqual(config.settings.performance.max_pages_per_site, 1000)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration should pass
        config = GenCrawlerConfig()
        # No exception should be raised
        
        # Invalid configuration should raise ValueError
        invalid_settings = GenCrawlerSettings()
        invalid_settings.performance.max_concurrent = -1
        
        with self.assertRaises(ValueError):
            GenCrawlerConfig(settings=invalid_settings)
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        settings = GenCrawlerSettings()
        
        # Test max_content_length <= min_content_length
        settings.content_filters.min_content_length = 1000
        settings.content_filters.max_content_length = 500
        
        with self.assertRaises(ValueError):
            GenCrawlerConfig(settings=settings)
    
    def test_get_crawler_config(self):
        """Test crawler configuration generation."""
        config = GenCrawlerConfig()
        crawler_config = config.get_crawler_config()
        
        self.assertIsInstance(crawler_config, dict)
        self.assertIn('max_pages_per_site', crawler_config)
        self.assertIn('max_concurrent', crawler_config)
        self.assertIn('content_filters', crawler_config)
        self.assertIn('database_config', crawler_config)
    
    def test_get_adaptive_config(self):
        """Test adaptive configuration generation."""
        config = GenCrawlerConfig()
        adaptive_config = config.get_adaptive_config()
        
        self.assertIsInstance(adaptive_config, dict)
        self.assertIn('enable_adaptive_crawling', adaptive_config)
        self.assertIn('strategy_switching_threshold', adaptive_config)
        self.assertIn('performance_monitoring', adaptive_config)

class TestConfigFileHandling(unittest.TestCase):
    """Test configuration file loading and saving."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config_data = {
            'performance': {
                'max_pages_per_site': 100,
                'max_concurrent': 3
            },
            'content_filters': {
                'min_content_length': 150,
                'include_patterns': ['test', 'sample']
            },
            'enable_content_analysis': False
        }
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.json'
            
            # Create and save configuration
            settings = GenCrawlerSettings.from_dict(self.test_config_data)
            config = GenCrawlerConfig(settings=settings)
            config.save_to_file(config_path)
            
            # Verify file exists
            self.assertTrue(config_path.exists())
            
            # Load and verify configuration
            loaded_config = GenCrawlerConfig(config_file=config_path)
            
            self.assertEqual(
                loaded_config.settings.performance.max_pages_per_site, 100
            )
            self.assertEqual(
                loaded_config.settings.performance.max_concurrent, 3
            )
            self.assertEqual(
                loaded_config.settings.content_filters.min_content_length, 150
            )
            self.assertFalse(loaded_config.settings.enable_content_analysis)
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / 'nonexistent.json'
            
            # Should fall back to defaults without error
            config = GenCrawlerConfig(config_file=nonexistent_path)
            
            # Should have default values
            self.assertEqual(config.settings.performance.max_pages_per_site, 500)
    
    def test_load_invalid_config(self):
        """Test loading invalid JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_config_path = Path(temp_dir) / 'invalid.json'
            
            # Write invalid JSON
            with open(invalid_config_path, 'w') as f:
                f.write('{ invalid json }')
            
            # Should fall back to defaults
            config = GenCrawlerConfig(config_file=invalid_config_path)
            
            # Should have default values
            self.assertEqual(config.settings.performance.max_pages_per_site, 500)
    
    def test_factory_function_with_file(self):
        """Test factory function with config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'factory_test.json'
            
            # Save test configuration
            with open(config_path, 'w') as f:
                json.dump(self.test_config_data, f)
            
            # Use factory function
            config = load_gen_config_from_file(config_path)
            
            self.assertEqual(config.settings.performance.max_pages_per_site, 100)

class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable integration."""
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            'GEN_CRAWLER_MAX_PAGES': '1000',
            'GEN_CRAWLER_MAX_CONCURRENT': '8',
            'GEN_CRAWLER_CRAWL_DELAY': '1.5',
            'GEN_CRAWLER_USER_AGENT': 'Test Agent',
            'GEN_CRAWLER_ENABLE_DATABASE': 'true',
            'GEN_CRAWLER_DB_CONNECTION': 'postgresql://test:test@localhost/test'
        }
        
        with patch.dict(os.environ, env_vars):
            config = create_gen_config(load_from_env=True)
            
            self.assertEqual(config.settings.performance.max_pages_per_site, 1000)
            self.assertEqual(config.settings.performance.max_concurrent, 8)
            self.assertEqual(config.settings.performance.crawl_delay, 1.5)
            self.assertEqual(config.settings.stealth.user_agent, 'Test Agent')
            self.assertTrue(config.settings.database.enable_database)
            self.assertEqual(
                config.settings.database.connection_string,
                'postgresql://test:test@localhost/test'
            )
    
    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        env_vars = {
            'GEN_CRAWLER_MAX_PAGES': 'invalid_number',
            'GEN_CRAWLER_CRAWL_DELAY': 'not_a_float',
            'GEN_CRAWLER_ENABLE_DATABASE': 'maybe'  # not true/false
        }
        
        with patch.dict(os.environ, env_vars):
            # Should not raise exception, should use defaults
            config = create_gen_config(load_from_env=True)
            
            # Should have default values due to conversion errors
            self.assertEqual(config.settings.performance.max_pages_per_site, 500)
            self.assertEqual(config.settings.performance.crawl_delay, 2.0)

class TestFactoryFunctions(unittest.TestCase):
    """Test configuration factory functions."""
    
    def test_create_gen_config_defaults(self):
        """Test create_gen_config with defaults."""
        config = create_gen_config()
        
        self.assertIsInstance(config, GenCrawlerConfig)
        self.assertEqual(config.settings.performance.max_pages_per_site, 500)
    
    def test_create_gen_config_with_settings(self):
        """Test create_gen_config with custom settings."""
        custom_settings = GenCrawlerSettings()
        custom_settings.performance.max_pages_per_site = 1000
        
        config = create_gen_config(settings=custom_settings)
        
        self.assertEqual(config.settings.performance.max_pages_per_site, 1000)
    
    def test_get_default_gen_config(self):
        """Test get_default_gen_config function."""
        default_config = get_default_gen_config()
        
        self.assertIsInstance(default_config, dict)
        self.assertEqual(default_config['max_pages_per_site'], 500)
        self.assertEqual(default_config['max_concurrent'], 5)
        self.assertIn('content_filters', default_config)
    
    def test_create_gen_config_without_env_loading(self):
        """Test create_gen_config without environment loading."""
        env_vars = {
            'GEN_CRAWLER_MAX_PAGES': '9999'
        }
        
        with patch.dict(os.environ, env_vars):
            config = create_gen_config(load_from_env=False)
            
            # Should have default value, not environment value
            self.assertEqual(config.settings.performance.max_pages_per_site, 500)

class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration system."""
    
    def test_full_configuration_workflow(self):
        """Test complete configuration workflow."""
        # Create custom configuration
        settings = GenCrawlerSettings()
        settings.performance.max_pages_per_site = 200
        settings.content_filters.include_patterns = ['news', 'article']
        settings.database.enable_database = True
        settings.database.connection_string = 'sqlite:///test.db'
        
        config = GenCrawlerConfig(settings=settings)
        
        # Test crawler configuration generation
        crawler_config = config.get_crawler_config()
        self.assertEqual(crawler_config['max_pages_per_site'], 200)
        self.assertTrue(crawler_config['enable_database'])
        
        # Test adaptive configuration generation
        adaptive_config = config.get_adaptive_config()
        self.assertTrue(adaptive_config['enable_adaptive_crawling'])
        
        # Test file save/load roundtrip
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'integration_test.json'
            
            config.save_to_file(config_path)
            loaded_config = GenCrawlerConfig(config_file=config_path)
            
            self.assertEqual(
                loaded_config.settings.performance.max_pages_per_site, 200
            )
            self.assertEqual(
                loaded_config.settings.content_filters.include_patterns,
                ['news', 'article']
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)