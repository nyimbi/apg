"""
APG Multi-language Localization - Test Package

Comprehensive test suite for internationalization and localization services,
ensuring reliability, performance, and accuracy across all supported languages.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import asyncio
import pytest
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import aioredis

# Test configuration
TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_localization"
TEST_REDIS_URL = "redis://localhost:6379/1"

# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
	"""Create a test database session."""
	engine = create_async_engine(TEST_DATABASE_URL, echo=False)
	async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
	
	async with async_session() as session:
		yield session
		await session.rollback()

@pytest.fixture
async def redis_client() -> AsyncGenerator[aioredis.Redis, None]:
	"""Create a test Redis client."""
	redis = aioredis.from_url(TEST_REDIS_URL)
	
	# Clear test database
	await redis.flushdb()
	
	yield redis
	
	# Cleanup
	await redis.flushdb()
	await redis.close()

# Test constants
SAMPLE_LANGUAGES = [
	{"code": "en", "name": "English", "native_name": "English", "script": "Latn"},
	{"code": "es", "name": "Spanish", "native_name": "Español", "script": "Latn"},
	{"code": "fr", "name": "French", "native_name": "Français", "script": "Latn"},
	{"code": "de", "name": "German", "native_name": "Deutsch", "script": "Latn"},
	{"code": "zh", "name": "Chinese", "native_name": "中文", "script": "Hans"},
	{"code": "ja", "name": "Japanese", "native_name": "日本語", "script": "Jpan"},
	{"code": "ar", "name": "Arabic", "native_name": "العربية", "script": "Arab"},
	{"code": "he", "name": "Hebrew", "native_name": "עברית", "script": "Hebr"},
	{"code": "hi", "name": "Hindi", "native_name": "हिन्दी", "script": "Deva"},
	{"code": "ru", "name": "Russian", "native_name": "Русский", "script": "Cyrl"}
]

SAMPLE_TRANSLATION_KEYS = [
	{"key": "common.buttons.save", "source_text": "Save"},
	{"key": "common.buttons.cancel", "source_text": "Cancel"},
	{"key": "common.buttons.delete", "source_text": "Delete"},
	{"key": "user.profile.title", "source_text": "User Profile"},
	{"key": "user.settings.notifications", "source_text": "Notification Settings"},
	{"key": "error.validation.required", "source_text": "This field is required"},
	{"key": "error.server.internal", "source_text": "An internal server error occurred"},
	{"key": "welcome.message", "source_text": "Welcome to our platform, {username}!"},
	{"key": "email.subject.welcome", "source_text": "Welcome to {platform_name}"},
	{"key": "notification.task.completed", "source_text": "Task '{task_name}' has been completed"}
]

# Test categories
TEST_CATEGORIES = {
	"unit": "Unit tests for individual components",
	"integration": "Integration tests for service interactions", 
	"performance": "Performance and load testing",
	"security": "Security and validation testing",
	"i18n": "Internationalization specific tests",
	"api": "API endpoint testing",
	"ui": "User interface testing"
}

__all__ = [
	"event_loop",
	"db_session", 
	"redis_client",
	"SAMPLE_LANGUAGES",
	"SAMPLE_TRANSLATION_KEYS",
	"TEST_CATEGORIES"
]