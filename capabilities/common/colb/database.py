"""
Database Setup and Migration System for APG Real-Time Collaboration

Handles database initialization, migrations, and connection management.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import (
	create_engine, MetaData, Table, Column, String, Integer, 
	Boolean, DateTime, Text, JSON, inspect, text
)
from sqlalchemy.pool import NullPool

try:
	from .config import get_config
except ImportError:
	from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()

# Global variables for database connections
async_engine = None
async_session_factory = None
sync_engine = None
sync_session_factory = None


def get_database_url(async_mode: bool = True) -> str:
	"""Get database URL for async or sync mode"""
	config = get_config()
	url = config.database.url
	
	if async_mode:
		# Convert postgresql:// to postgresql+asyncpg://
		if url.startswith('postgresql://'):
			url = url.replace('postgresql://', 'postgresql+asyncpg://')
		elif url.startswith('sqlite://'):
			url = url.replace('sqlite://', 'sqlite+aiosqlite://')
	else:
		# Ensure sync URL format
		if url.startswith('postgresql+asyncpg://'):
			url = url.replace('postgresql+asyncpg://', 'postgresql://')
		elif url.startswith('sqlite+aiosqlite://'):
			url = url.replace('sqlite+aiosqlite://', 'sqlite://')
	
	return url


def create_sync_engine():
	"""Create synchronous database engine"""
	config = get_config()
	url = get_database_url(async_mode=False)
	
	return create_engine(
		url,
		pool_size=config.database.pool_size,
		max_overflow=config.database.max_overflow,
		pool_timeout=config.database.pool_timeout,
		echo=config.database.echo,
		poolclass=NullPool if 'sqlite' in url else None
	)


def create_async_engine_instance():
	"""Create asynchronous database engine"""
	config = get_config()
	url = get_database_url(async_mode=True)
	
	return create_async_engine(
		url,
		pool_size=config.database.pool_size,
		max_overflow=config.database.max_overflow,
		pool_timeout=config.database.pool_timeout,
		echo=config.database.echo,
		poolclass=NullPool if 'sqlite' in url else None
	)


async def init_database():
	"""Initialize database connections"""
	global async_engine, async_session_factory, sync_engine, sync_session_factory
	
	try:
		# Create engines
		async_engine = create_async_engine_instance()
		sync_engine = create_sync_engine()
		
		# Create session factories
		async_session_factory = async_sessionmaker(
			async_engine,
			class_=AsyncSession,
			expire_on_commit=False
		)
		
		sync_session_factory = sessionmaker(
			sync_engine,
			expire_on_commit=False
		)
		
		logger.info("Database connections initialized successfully")
		
	except Exception as e:
		logger.error(f"Failed to initialize database: {e}")
		raise


async def get_async_session() -> AsyncSession:
	"""Get async database session"""
	if async_session_factory is None:
		await init_database()
	
	return async_session_factory()


def get_sync_session():
	"""Get sync database session"""
	global sync_engine, sync_session_factory
	
	if sync_session_factory is None:
		# Initialize sync components only
		sync_engine = create_sync_engine()
		sync_session_factory = sessionmaker(sync_engine, expire_on_commit=False)
	
	return sync_session_factory()


async def close_database():
	"""Close database connections"""
	global async_engine, sync_engine
	
	if async_engine:
		await async_engine.dispose()
		logger.info("Async database engine disposed")
	
	if sync_engine:
		sync_engine.dispose()
		logger.info("Sync database engine disposed")


class DatabaseMigration:
	"""Database migration system"""
	
	def __init__(self):
		self.migrations_dir = Path(__file__).parent / "migrations"
		self.migrations_dir.mkdir(exist_ok=True)
		
		# Create migrations table if it doesn't exist
		self._ensure_migrations_table()
	
	def _ensure_migrations_table(self):
		"""Ensure migrations tracking table exists"""
		engine = create_sync_engine()
		
		with engine.connect() as conn:
			conn.execute(text("""
				CREATE TABLE IF NOT EXISTS rtc_migrations (
					id SERIAL PRIMARY KEY,
					name VARCHAR(255) NOT NULL UNIQUE,
					applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				)
			"""))
			conn.commit()
	
	def create_migration(self, name: str, up_sql: str, down_sql: str = "") -> str:
		"""Create a new migration file"""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = f"{timestamp}_{name}.sql"
		filepath = self.migrations_dir / filename
		
		migration_content = f"""-- Migration: {name}
-- Created: {datetime.now().isoformat()}

-- UP
{up_sql}

-- DOWN (for rollback)
{down_sql}
"""
		
		with open(filepath, 'w') as f:
			f.write(migration_content)
		
		logger.info(f"Created migration: {filename}")
		return str(filepath)
	
	def get_pending_migrations(self) -> List[str]:
		"""Get list of pending migrations"""
		engine = create_sync_engine()
		
		with engine.connect() as conn:
			# Get applied migrations
			result = conn.execute(text("SELECT name FROM rtc_migrations ORDER BY name"))
			applied = {row[0] for row in result}
		
		# Get all migration files
		migration_files = sorted([
			f.stem for f in self.migrations_dir.glob("*.sql")
		])
		
		# Return pending migrations
		return [m for m in migration_files if m not in applied]
	
	def apply_migration(self, migration_name: str) -> bool:
		"""Apply a single migration"""
		migration_file = self.migrations_dir / f"{migration_name}.sql"
		
		if not migration_file.exists():
			logger.error(f"Migration file not found: {migration_file}")
			return False
		
		try:
			with open(migration_file, 'r') as f:
				content = f.read()
			
			# Extract UP section
			up_section = self._extract_section(content, "UP")
			
			if not up_section.strip():
				logger.warning(f"No UP section found in migration: {migration_name}")
				return False
			
			engine = create_sync_engine()
			
			with engine.connect() as conn:
				# Execute migration
				conn.execute(text(up_section))
				
				# Mark as applied
				conn.execute(text(
					"INSERT INTO rtc_migrations (name) VALUES (:name)"
				), {"name": migration_name})
				
				conn.commit()
			
			logger.info(f"Applied migration: {migration_name}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to apply migration {migration_name}: {e}")
			return False
	
	def rollback_migration(self, migration_name: str) -> bool:
		"""Rollback a migration"""
		migration_file = self.migrations_dir / f"{migration_name}.sql"
		
		if not migration_file.exists():
			logger.error(f"Migration file not found: {migration_file}")
			return False
		
		try:
			with open(migration_file, 'r') as f:
				content = f.read()
			
			# Extract DOWN section
			down_section = self._extract_section(content, "DOWN")
			
			if not down_section.strip():
				logger.warning(f"No DOWN section found in migration: {migration_name}")
				return False
			
			engine = create_sync_engine()
			
			with engine.connect() as conn:
				# Execute rollback
				conn.execute(text(down_section))
				
				# Remove from applied migrations
				conn.execute(text(
					"DELETE FROM rtc_migrations WHERE name = :name"
				), {"name": migration_name})
				
				conn.commit()
			
			logger.info(f"Rolled back migration: {migration_name}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to rollback migration {migration_name}: {e}")
			return False
	
	def _extract_section(self, content: str, section: str) -> str:
		"""Extract UP or DOWN section from migration content"""
		lines = content.split('\n')
		section_start = f"-- {section}"
		
		start_idx = None
		end_idx = None
		
		for i, line in enumerate(lines):
			if line.strip().startswith(section_start):
				start_idx = i + 1
			elif start_idx is not None and line.strip().startswith("-- ") and "DOWN" in line:
				end_idx = i
				break
		
		if start_idx is None:
			return ""
		
		if end_idx is None:
			end_idx = len(lines)
		
		return '\n'.join(lines[start_idx:end_idx])
	
	def migrate(self) -> bool:
		"""Apply all pending migrations"""
		pending = self.get_pending_migrations()
		
		if not pending:
			logger.info("No pending migrations")
			return True
		
		logger.info(f"Applying {len(pending)} pending migrations")
		
		success = True
		for migration in pending:
			if not self.apply_migration(migration):
				success = False
				break
		
		return success


def create_initial_schema_migration():
	"""Create the initial schema migration"""
	migration = DatabaseMigration()
	
	up_sql = """
-- Create rtc_sessions table
CREATE TABLE IF NOT EXISTS rtc_sessions (
	session_id VARCHAR(36) PRIMARY KEY,
	tenant_id VARCHAR(100) NOT NULL,
	session_name VARCHAR(200) NOT NULL,
	session_type VARCHAR(50) NOT NULL DEFAULT 'page_collaboration',
	digital_twin_id VARCHAR(200),
	owner_user_id VARCHAR(100) NOT NULL,
	is_active BOOLEAN DEFAULT true,
	max_participants INTEGER DEFAULT 10,
	current_participant_count INTEGER DEFAULT 0,
	participant_user_ids TEXT DEFAULT '[]',
	collaboration_mode VARCHAR(50) DEFAULT 'open',
	require_approval BOOLEAN DEFAULT false,
	scheduled_start TIMESTAMP,
	scheduled_end TIMESTAMP,
	actual_start TIMESTAMP,
	actual_end TIMESTAMP,
	duration_minutes REAL,
	recording_enabled BOOLEAN DEFAULT true,
	voice_chat_enabled BOOLEAN DEFAULT false,
	video_chat_enabled BOOLEAN DEFAULT false,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rtc_participants table
CREATE TABLE IF NOT EXISTS rtc_participants (
	participant_id VARCHAR(36) PRIMARY KEY,
	session_id VARCHAR(36) NOT NULL,
	user_id VARCHAR(100) NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	display_name VARCHAR(200),
	role VARCHAR(50) DEFAULT 'viewer',
	joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	left_at TIMESTAMP,
	is_active BOOLEAN DEFAULT true,
	can_edit BOOLEAN DEFAULT false,
	can_annotate BOOLEAN DEFAULT true,
	can_chat BOOLEAN DEFAULT true,
	can_share_screen BOOLEAN DEFAULT false,
	last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	presence_status VARCHAR(50) DEFAULT 'active',
	cursor_position TEXT,
	FOREIGN KEY (session_id) REFERENCES rtc_sessions(session_id) ON DELETE CASCADE
);

-- Create rtc_video_calls table
CREATE TABLE IF NOT EXISTS rtc_video_calls (
	call_id VARCHAR(36) PRIMARY KEY,
	session_id VARCHAR(36) NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	call_name VARCHAR(200) NOT NULL,
	call_type VARCHAR(50) DEFAULT 'video',
	status VARCHAR(50) DEFAULT 'scheduled',
	meeting_id VARCHAR(100),
	host_user_id VARCHAR(100) NOT NULL,
	current_participants INTEGER DEFAULT 0,
	max_participants INTEGER DEFAULT 100,
	scheduled_start TIMESTAMP,
	started_at TIMESTAMP,
	ended_at TIMESTAMP,
	duration_minutes REAL,
	video_quality VARCHAR(20) DEFAULT 'hd',
	audio_quality VARCHAR(20) DEFAULT 'high',
	enable_recording BOOLEAN DEFAULT false,
	waiting_room_enabled BOOLEAN DEFAULT true,
	end_to_end_encryption BOOLEAN DEFAULT true,
	breakout_rooms_enabled BOOLEAN DEFAULT false,
	polls_enabled BOOLEAN DEFAULT true,
	whiteboard_enabled BOOLEAN DEFAULT true,
	screen_sharing_enabled BOOLEAN DEFAULT true,
	chat_enabled BOOLEAN DEFAULT true,
	teams_meeting_url TEXT,
	teams_meeting_id VARCHAR(100),
	zoom_meeting_id VARCHAR(100),
	meet_url TEXT,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY (session_id) REFERENCES rtc_sessions(session_id) ON DELETE CASCADE
);

-- Create rtc_page_collaboration table
CREATE TABLE IF NOT EXISTS rtc_page_collaboration (
	page_collab_id VARCHAR(36) PRIMARY KEY,
	tenant_id VARCHAR(100) NOT NULL,
	page_url VARCHAR(500) NOT NULL,
	page_title VARCHAR(200),
	page_type VARCHAR(100),
	blueprint_name VARCHAR(100),
	view_name VARCHAR(100),
	is_active BOOLEAN DEFAULT true,
	current_users TEXT DEFAULT '[]',
	total_collaboration_sessions INTEGER DEFAULT 0,
	total_form_delegations INTEGER DEFAULT 0,
	total_assistance_requests INTEGER DEFAULT 0,
	average_users_per_session REAL DEFAULT 0.0,
	delegated_fields TEXT DEFAULT '{}',
	assistance_requests TEXT DEFAULT '[]',
	first_collaboration TIMESTAMP,
	last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rtc_third_party_integrations table
CREATE TABLE IF NOT EXISTS rtc_third_party_integrations (
	integration_id VARCHAR(36) PRIMARY KEY,
	tenant_id VARCHAR(100) NOT NULL,
	platform VARCHAR(50) NOT NULL,
	platform_name VARCHAR(100) NOT NULL,
	integration_type VARCHAR(50) DEFAULT 'api',
	status VARCHAR(50) DEFAULT 'active',
	api_key VARCHAR(500),
	api_secret VARCHAR(500),
	webhook_url VARCHAR(500),
	last_sync TIMESTAMP,
	sync_frequency_minutes INTEGER DEFAULT 60,
	total_meetings_synced INTEGER DEFAULT 0,
	total_api_calls INTEGER DEFAULT 0,
	monthly_api_limit INTEGER,
	current_month_usage INTEGER DEFAULT 0,
	sync_meetings BOOLEAN DEFAULT true,
	sync_participants BOOLEAN DEFAULT true,
	sync_recordings BOOLEAN DEFAULT true,
	auto_create_meetings BOOLEAN DEFAULT false,
	teams_tenant_id VARCHAR(100),
	teams_application_id VARCHAR(100),
	zoom_account_id VARCHAR(100),
	google_workspace_domain VARCHAR(200),
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_rtc_sessions_tenant_active ON rtc_sessions(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_rtc_sessions_owner ON rtc_sessions(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_rtc_participants_session ON rtc_participants(session_id);
CREATE INDEX IF NOT EXISTS idx_rtc_participants_user ON rtc_participants(user_id);
CREATE INDEX IF NOT EXISTS idx_rtc_video_calls_session ON rtc_video_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_rtc_video_calls_host ON rtc_video_calls(host_user_id);
CREATE INDEX IF NOT EXISTS idx_rtc_page_collaboration_url ON rtc_page_collaboration(page_url);
CREATE INDEX IF NOT EXISTS idx_rtc_page_collaboration_tenant ON rtc_page_collaboration(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rtc_integrations_tenant ON rtc_third_party_integrations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rtc_integrations_platform ON rtc_third_party_integrations(platform);
"""
	
	down_sql = """
-- Drop tables in reverse order
DROP TABLE IF EXISTS rtc_third_party_integrations;
DROP TABLE IF EXISTS rtc_page_collaboration;
DROP TABLE IF EXISTS rtc_video_calls;
DROP TABLE IF EXISTS rtc_participants;
DROP TABLE IF EXISTS rtc_sessions;
"""
	
	migration.create_migration("initial_schema", up_sql, down_sql)
	return migration


async def test_database_connection() -> bool:
	"""Test database connection"""
	try:
		async with get_async_session() as session:
			result = await session.execute(text("SELECT 1"))
			result.scalar()
		
		logger.info("Database connection test successful")
		return True
		
	except Exception as e:
		logger.error(f"Database connection test failed: {e}")
		return False


# Database utilities
class DatabaseManager:
	"""Database management utilities"""
	
	def __init__(self):
		self.migration = DatabaseMigration()
	
	async def initialize(self):
		"""Initialize database and run migrations"""
		logger.info("Initializing database...")
		
		# Initialize connections
		await init_database()
		
		# Test connection
		if not await test_database_connection():
			raise RuntimeError("Database connection failed")
		
		# Run migrations
		if not self.migration.migrate():
			raise RuntimeError("Database migration failed")
		
		logger.info("Database initialization complete")
	
	async def reset_database(self):
		"""Reset database (for development/testing)"""
		logger.warning("Resetting database - this will delete all data!")
		
		engine = create_sync_engine()
		
		with engine.connect() as conn:
			# Drop all tables
			conn.execute(text("""
				DROP TABLE IF EXISTS rtc_third_party_integrations CASCADE;
				DROP TABLE IF EXISTS rtc_page_collaboration CASCADE;
				DROP TABLE IF EXISTS rtc_video_calls CASCADE;
				DROP TABLE IF EXISTS rtc_participants CASCADE;
				DROP TABLE IF EXISTS rtc_sessions CASCADE;
				DROP TABLE IF EXISTS rtc_migrations CASCADE;
			"""))
			conn.commit()
		
		# Recreate schema
		migration = create_initial_schema_migration()
		migration.migrate()
		
		logger.info("Database reset complete")
	
	def create_sample_data(self):
		"""Create sample data for development"""
		logger.info("Creating sample data...")
		
		engine = create_sync_engine()
		
		with engine.connect() as conn:
			# Sample session
			conn.execute(text("""
				INSERT INTO rtc_sessions (
					session_id, tenant_id, session_name, session_type,
					owner_user_id, is_active, max_participants
				) VALUES (
					'sample-session-001', 'tenant123', 'Sample Budget Session', 
					'page_collaboration', 'user123', true, 10
				) ON CONFLICT (session_id) DO NOTHING;
			"""))
			
			# Sample page collaboration
			conn.execute(text("""
				INSERT INTO rtc_page_collaboration (
					page_collab_id, tenant_id, page_url, page_title, page_type,
					blueprint_name, view_name, is_active
				) VALUES (
					'page-collab-001', 'tenant123', '/admin/users/list', 
					'User Management', 'list_view', 'admin', 'users', true
				) ON CONFLICT (page_collab_id) DO NOTHING;
			"""))
			
			conn.commit()
		
		logger.info("Sample data created")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
async def initialize_database():
	"""Initialize database system"""
	await db_manager.initialize()


async def reset_database():
	"""Reset database (development only)"""
	await db_manager.reset_database()


def create_sample_data():
	"""Create sample data"""
	db_manager.create_sample_data()