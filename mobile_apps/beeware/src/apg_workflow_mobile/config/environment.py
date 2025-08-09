"""
Environment detection and configuration

© 2025 Datacraft. All rights reserved.
"""

import os
from enum import Enum
from typing import Optional


class Environment(str, Enum):
	"""Application environment enumeration"""
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"
	TESTING = "testing"


def get_environment() -> Environment:
	"""Detect current environment from environment variables"""
	env_name = os.getenv("APG_ENVIRONMENT", "development").lower()
	
	# Map common environment variable values
	env_mapping = {
		"dev": Environment.DEVELOPMENT,
		"development": Environment.DEVELOPMENT,
		"local": Environment.DEVELOPMENT,
		"stage": Environment.STAGING,
		"staging": Environment.STAGING,
		"prod": Environment.PRODUCTION,
		"production": Environment.PRODUCTION,
		"test": Environment.TESTING,
		"testing": Environment.TESTING,
	}
	
	return env_mapping.get(env_name, Environment.DEVELOPMENT)


def is_development() -> bool:
	"""Check if running in development environment"""
	return get_environment() == Environment.DEVELOPMENT


def is_production() -> bool:
	"""Check if running in production environment"""
	return get_environment() == Environment.PRODUCTION


def is_testing() -> bool:
	"""Check if running in testing environment"""
	return get_environment() == Environment.TESTING


def get_config_file_path() -> Optional[str]:
	"""Get path to environment-specific config file"""
	env = get_environment()
	
	# Check for explicit config file path
	explicit_path = os.getenv("APG_CONFIG_FILE")
	if explicit_path and os.path.exists(explicit_path):
		return explicit_path
	
	# Look for environment-specific config files
	config_dir = os.path.join(os.path.dirname(__file__), "files")
	config_files = {
		Environment.DEVELOPMENT: f"{config_dir}/development.yml",
		Environment.STAGING: f"{config_dir}/staging.yml", 
		Environment.PRODUCTION: f"{config_dir}/production.yml",
		Environment.TESTING: f"{config_dir}/testing.yml",
	}
	
	config_file = config_files.get(env)
	if config_file and os.path.exists(config_file):
		return config_file
	
	return None