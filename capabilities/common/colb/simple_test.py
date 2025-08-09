#!/usr/bin/env python3
"""
Simple test script for APG Real-Time Collaboration

Tests basic functionality without external dependencies.
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
	"""Test basic Python imports"""
	print("📦 Testing basic imports...")
	
	try:
		import uuid
		import json
		import asyncio
		from datetime import datetime
		print("✅ Standard library imports working")
		return True
	except Exception as e:
		print(f"❌ Standard library import error: {e}")
		return False

def test_apg_stubs():
	"""Test APG integration stubs"""
	print("🔌 Testing APG stubs...")
	
	try:
		from apg_stubs import auth_service, ai_service, notification_service
		print("✅ APG stubs imported successfully")
		return True
	except Exception as e:
		print(f"❌ APG stubs import error: {e}")
		return False

def test_config_module():
	"""Test configuration module"""
	print("⚙️  Testing configuration...")
	
	try:
		# Test basic config loading
		os.environ.update({
			'ENVIRONMENT': 'development',
			'DATABASE_URL': 'sqlite:///test.db',
			'DEBUG': 'true'
		})
		
		from config import get_config, get_development_config
		config = get_development_config()
		
		assert config.environment == 'development'
		assert config.debug == True
		assert 'sqlite' in config.database.url
		
		print("✅ Configuration module working")
		return True
		
	except Exception as e:
		print(f"❌ Configuration error: {e}")
		return False

def test_basic_data_structures():
	"""Test basic data structure creation"""
	print("📊 Testing data structures...")
	
	try:
		# Test session data
		session_data = {
			"session_id": str(__import__('uuid').uuid4()),
			"session_name": "Test Session",
			"tenant_id": "test_tenant",
			"created_at": datetime.now().isoformat(),
			"is_active": True,
			"participants": []
		}
		
		# Test JSON serialization
		json_str = json.dumps(session_data)
		parsed_data = json.loads(json_str)
		
		assert parsed_data["session_name"] == "Test Session"
		assert parsed_data["is_active"] == True
		
		print("✅ Data structures working")
		return True
		
	except Exception as e:
		print(f"❌ Data structure error: {e}")
		return False

async def test_async_functionality():
	"""Test async functionality"""
	print("⚡ Testing async functionality...")
	
	try:
		# Test basic async operation
		await asyncio.sleep(0.1)
		
		# Test APG stubs async calls
		from apg_stubs import auth_service, ai_service
		
		# Test auth stub
		token_result = await auth_service.validate_token("test_token")
		assert token_result["valid"] == True
		
		# Test AI stub
		suggestions = await ai_service.suggest_participants({"page_url": "/test"})
		assert len(suggestions) > 0
		
		print("✅ Async functionality working")
		return True
		
	except Exception as e:
		print(f"❌ Async functionality error: {e}")
		return False

def test_database_models():
	"""Test database model definitions"""
	print("🗃️  Testing database models...")
	
	try:
		# Test basic SQLAlchemy functionality
		from sqlalchemy import Column, String, Integer, Boolean, create_engine
		from sqlalchemy.ext.declarative import declarative_base
		
		Base = declarative_base()
		
		class TestModel(Base):
			__tablename__ = 'test_model'
			id = Column(String, primary_key=True)
			name = Column(String(100))
			is_active = Column(Boolean, default=True)
		
		# Test in-memory SQLite
		engine = create_engine('sqlite:///:memory:')
		Base.metadata.create_all(engine)
		
		print("✅ Database models working")
		return True
		
	except Exception as e:
		print(f"❌ Database model error: {e}")
		return False

def run_basic_tests():
	"""Run basic functionality tests"""
	print("🧪 Running basic tests...")
	
	try:
		# Import and run the existing test
		from tests.test_basic_functionality import TestBasicFunctionality
		
		test_instance = TestBasicFunctionality()
		
		# Run individual tests
		test_instance.test_uuid_generation()
		test_instance.test_datetime_operations()
		test_instance.test_json_serialization()
		
		print("✅ Basic functionality tests passed")
		return True
		
	except Exception as e:
		print(f"❌ Basic tests error: {e}")
		return False

def create_development_database():
	"""Create a simple SQLite database for development"""
	print("🗃️  Creating development database...")
	
	try:
		import sqlite3
		
		# Create SQLite database
		db_path = "rtc_development.db"
		
		with sqlite3.connect(db_path) as conn:
			cursor = conn.cursor()
			
			# Create a simple sessions table
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS rtc_sessions (
					session_id TEXT PRIMARY KEY,
					session_name TEXT NOT NULL,
					tenant_id TEXT NOT NULL,
					owner_user_id TEXT NOT NULL,
					is_active BOOLEAN DEFAULT 1,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				)
			''')
			
			# Insert sample data
			cursor.execute('''
				INSERT OR IGNORE INTO rtc_sessions 
				(session_id, session_name, tenant_id, owner_user_id, is_active)
				VALUES (?, ?, ?, ?, ?)
			''', ('dev-session-001', 'Development Session', 'dev_tenant', 'dev_user', 1))
			
			conn.commit()
		
		print(f"✅ Development database created: {db_path}")
		return True
		
	except Exception as e:
		print(f"❌ Database creation error: {e}")
		return False

async def main():
	"""Main test function"""
	print("=" * 60)
	print("APG Real-Time Collaboration - Simple Functionality Test")
	print("=" * 60)
	
	all_tests_passed = True
	
	# Run tests
	tests = [
		test_basic_imports,
		test_apg_stubs,
		test_config_module,
		test_basic_data_structures,
		test_database_models,
		run_basic_tests,
		create_development_database
	]
	
	for test_func in tests:
		try:
			if asyncio.iscoroutinefunction(test_func):
				result = await test_func()
			else:
				result = test_func()
			
			if not result:
				all_tests_passed = False
		except Exception as e:
			print(f"❌ Test {test_func.__name__} failed: {e}")
			all_tests_passed = False
	
	# Test async functionality
	try:
		await test_async_functionality()
	except Exception as e:
		print(f"❌ Async test failed: {e}")
		all_tests_passed = False
	
	print("\n" + "=" * 60)
	if all_tests_passed:
		print("🎉 All tests passed! APG RTC basic functionality is working.")
		print("\nNext steps:")
		print("1. The development database has been created")
		print("2. Basic functionality is validated")
		print("3. APG integration stubs are working")
		print("4. Configuration system is operational")
		print("\nYou can now:")
		print("- Start developing more advanced features")
		print("- Test the WebSocket functionality")
		print("- Integrate with actual APG services")
	else:
		print("❌ Some tests failed. Please check the errors above.")
		return False
	
	return True

if __name__ == "__main__":
	success = asyncio.run(main())
	sys.exit(0 if success else 1)