#!/usr/bin/env python3
"""
Development setup script for APG Real-Time Collaboration

This script sets up a development environment with SQLite for quick testing.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Ensure we can import from the current directory
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, get_development_config
from database import DatabaseManager, create_initial_schema_migration

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def setup_development_environment():
	"""Set up development environment with SQLite database"""
	print("🚀 Setting up APG Real-Time Collaboration development environment...")
	
	try:
		# Set up development environment
		os.environ.update({
			'ENVIRONMENT': 'development',
			'DATABASE_URL': 'sqlite:///rtc_dev.db',
			'REDIS_URL': 'redis://localhost:6379/0',
			'DEBUG': 'true',
			'API_RELOAD': 'true',
			'LOG_LEVEL': 'DEBUG'
		})
		
		# Get development configuration
		config = get_development_config()
		print(f"✅ Configuration loaded for environment: {config.environment}")
		
		# Initialize database manager
		db_manager = DatabaseManager()
		
		# Initialize database
		print("🗃️  Initializing database...")
		await db_manager.initialize()
		print("✅ Database initialized successfully")
		
		# Create sample data
		print("📊 Creating sample data...")
		db_manager.create_sample_data()
		print("✅ Sample data created")
		
		print("\n🎉 Development environment setup complete!")
		print("\nNext steps:")
		print("1. Start the API server: python app.py serve")
		print("2. Or run development mode: python app.py dev-setup")
		print("3. Open http://localhost:8000 to view the API")
		print("4. Open http://localhost:8000/docs for API documentation")
		print("5. Run tests: python -m pytest tests/test_basic_functionality.py -v")
		
	except Exception as e:
		print(f"❌ Development setup failed: {e}")
		logger.exception("Development setup error")
		sys.exit(1)

def run_basic_tests():
	"""Run basic functionality tests"""
	print("🧪 Running basic functionality tests...")
	
	try:
		import subprocess
		result = subprocess.run([
			sys.executable, '-m', 'pytest', 
			'tests/test_basic_functionality.py', 
			'-v'
		], capture_output=True, text=True)
		
		print(result.stdout)
		if result.stderr:
			print("Errors:")
			print(result.stderr)
		
		if result.returncode == 0:
			print("✅ All basic tests passed!")
		else:
			print("❌ Some tests failed")
			return False
			
	except Exception as e:
		print(f"❌ Test execution failed: {e}")
		return False
	
	return True

def validate_imports():
	"""Validate that all required modules can be imported"""
	print("📦 Validating imports...")
	
	required_modules = [
		'fastapi',
		'uvicorn', 
		'sqlalchemy',
		'pydantic',
		'redis',
		'asyncpg'
	]
	
	missing_modules = []
	for module in required_modules:
		try:
			__import__(module)
			print(f"✅ {module}")
		except ImportError:
			print(f"❌ {module} - MISSING")
			missing_modules.append(module)
	
	if missing_modules:
		print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
		print("Install with: pip install " + " ".join(missing_modules))
		return False
	
	print("✅ All required modules available")
	return True

async def test_database_operations():
	"""Test basic database operations"""
	print("🗃️  Testing database operations...")
	
	try:
		# Set up SQLite for testing
		os.environ['DATABASE_URL'] = 'sqlite:///test_rtc.db'
		
		from database import test_database_connection, initialize_database
		
		# Initialize database
		await initialize_database()
		print("✅ Database initialization")
		
		# Test connection
		if await test_database_connection():
			print("✅ Database connection test")
		else:
			print("❌ Database connection failed")
			return False
		
		# Clean up test database
		test_db = Path("test_rtc.db")
		if test_db.exists():
			test_db.unlink()
		
		return True
		
	except Exception as e:
		print(f"❌ Database test failed: {e}")
		return False

def main():
	"""Main setup function"""
	print("=" * 60)
	print("APG Real-Time Collaboration - Development Setup")
	print("=" * 60)
	
	# Validate imports first
	if not validate_imports():
		sys.exit(1)
	
	# Test database operations
	if not asyncio.run(test_database_operations()):
		sys.exit(1)
	
	# Run basic tests
	if not run_basic_tests():
		print("⚠️  Basic tests failed, but continuing with setup...")
	
	# Set up development environment
	asyncio.run(setup_development_environment())

if __name__ == "__main__":
	main()