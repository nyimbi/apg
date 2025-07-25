#!/usr/bin/env python3
"""
PostgreSQL Database Initialization for APG Capabilities
=======================================================

Comprehensive database setup script for all APG capability blueprints.
Creates all necessary tables, indexes, and constraints for PostgreSQL.
"""

import os
import sys
import asyncpg
import asyncio
from typing import Dict, List, Any
from pathlib import Path

# Import all schema definitions
sys.path.append(str(Path(__file__).parent.parent))

from blueprints.base import POSTGRESQL_SCHEMAS
from blueprints.enhanced_computer_vision_blueprint import ENHANCED_COMPUTER_VISION_SCHEMAS
from blueprints.enhanced_iot_blueprint import ENHANCED_IOT_SCHEMAS
from blueprints.audio_processing_blueprint import AUDIO_PROCESSING_SCHEMAS
from blueprints.digital_twin_blueprint import DIGITAL_TWIN_SCHEMAS

class PostgreSQLInitializer:
	"""Initialize PostgreSQL database with all APG capability schemas"""
	
	def __init__(self, connection_string: str = None):
		"""
		Initialize database connection
		
		Args:
			connection_string: PostgreSQL connection string
				Format: postgresql://user:password@host:port/database
				If None, will use environment variables or defaults
		"""
		self.connection_string = connection_string or self._build_connection_string()
		self.connection = None
		
		# Collect all schemas from blueprints
		self.all_schemas = {
			**POSTGRESQL_SCHEMAS,
			**ENHANCED_COMPUTER_VISION_SCHEMAS,
			**ENHANCED_IOT_SCHEMAS,
			**AUDIO_PROCESSING_SCHEMAS,
			**DIGITAL_TWIN_SCHEMAS
		}
		
		print(f"Found {len(self.all_schemas)} schema definitions to create")
	
	def _build_connection_string(self) -> str:
		"""Build connection string from environment variables or defaults"""
		host = os.getenv('POSTGRES_HOST', 'localhost')
		port = os.getenv('POSTGRES_PORT', '5432')
		database = os.getenv('POSTGRES_DB', 'apg_capabilities')
		user = os.getenv('POSTGRES_USER', 'postgres')
		password = os.getenv('POSTGRES_PASSWORD', 'postgres')
		
		return f"postgresql://{user}:{password}@{host}:{port}/{database}"
	
	async def connect(self):
		"""Establish database connection"""
		try:
			self.connection = await asyncpg.connect(self.connection_string)
			print(f"✅ Connected to PostgreSQL database")
		except Exception as e:
			print(f"❌ Failed to connect to database: {e}")
			raise
	
	async def disconnect(self):
		"""Close database connection"""
		if self.connection:
			await self.connection.close()
			print("✅ Database connection closed")
	
	async def create_extensions(self):
		"""Create necessary PostgreSQL extensions"""
		extensions = [
			'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
			'CREATE EXTENSION IF NOT EXISTS "pgcrypto";',
			'CREATE EXTENSION IF NOT EXISTS "btree_gin";',
			'CREATE EXTENSION IF NOT EXISTS "btree_gist";'
		]
		
		print("Creating PostgreSQL extensions...")
		for extension in extensions:
			try:
				await self.connection.execute(extension)
				print(f"  ✅ {extension}")
			except Exception as e:
				print(f"  ⚠️  {extension} - {e}")
	
	async def create_base_functions(self):
		"""Create base utility functions"""
		functions = [
			"""
			CREATE OR REPLACE FUNCTION update_updated_at_column()
			RETURNS TRIGGER AS $$
			BEGIN
				NEW.updated_at = NOW();
				RETURN NEW;
			END;
			$$ language 'plpgsql';
			""",
			
			"""
			CREATE OR REPLACE FUNCTION generate_short_id()
			RETURNS TEXT AS $$
			BEGIN
				RETURN encode(gen_random_bytes(6), 'base64')::text;
			END;
			$$ language 'plpgsql';
			""",
			
			"""
			CREATE OR REPLACE FUNCTION jsonb_array_length_safe(jsonb_data JSONB)
			RETURNS INTEGER AS $$
			BEGIN
				IF jsonb_data IS NULL OR jsonb_typeof(jsonb_data) != 'array' THEN
					RETURN 0;
				END IF;
				RETURN jsonb_array_length(jsonb_data);
			END;
			$$ language 'plpgsql';
			"""
		]
		
		print("Creating utility functions...")
		for function in functions:
			try:
				await self.connection.execute(function)
				print(f"  ✅ Function created")
			except Exception as e:
				print(f"  ❌ Function creation failed: {e}")
	
	async def create_schema(self, schema_name: str, schema_sql: str):
		"""Create a single schema with error handling"""
		try:
			print(f"Creating schema: {schema_name}")
			
			# Split and execute each statement
			statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
			
			for statement in statements:
				if statement:
					await self.connection.execute(statement + ';')
			
			print(f"  ✅ {schema_name} schema created successfully")
			return True
			
		except Exception as e:
			print(f"  ❌ Failed to create {schema_name}: {e}")
			return False
	
	async def create_all_schemas(self):
		"""Create all schemas in dependency order"""
		print(f"\nCreating {len(self.all_schemas)} database schemas...")
		
		# Order schemas by dependencies (base tables first)
		ordered_schemas = [
			# Base infrastructure
			'operation_logs',
			'system_metrics', 
			'capability_configurations',
			
			# Project-level tables
			'cv_projects',
			'iot_projects',
			'audio_projects',
			'dt_projects',
			
			# Device and model registries
			'iot_gateways',
			'iot_devices',
			'cv_person_registry',
			'dt_templates',
			
			# Core processing tables
			'cv_image_jobs',
			'cv_video_jobs',
			'audio_recordings',
			'audio_transcriptions',
			'audio_synthesis_jobs',
			'digital_twins',
			
			# Detail and analysis tables
			'cv_face_embeddings',
			'iot_audio_recordings',
			'iot_anomaly_detections',
			'audio_segments',
			'audio_annotations',
			'dt_telemetry_data',
			'dt_simulation_jobs',
			'dt_events'
		]
		
		created_count = 0
		failed_schemas = []
		
		# Create schemas in order
		for schema_name in ordered_schemas:
			if schema_name in self.all_schemas:
				success = await self.create_schema(schema_name, self.all_schemas[schema_name])
				if success:
					created_count += 1
				else:
					failed_schemas.append(schema_name)
		
		# Create any remaining schemas not in the ordered list
		for schema_name, schema_sql in self.all_schemas.items():
			if schema_name not in ordered_schemas:
				success = await self.create_schema(schema_name, schema_sql)
				if success:
					created_count += 1
				else:
					failed_schemas.append(schema_name)
		
		print(f"\nSchema creation summary:")
		print(f"  ✅ {created_count} schemas created successfully")
		if failed_schemas:
			print(f"  ❌ {len(failed_schemas)} schemas failed: {', '.join(failed_schemas)}")
		
		return len(failed_schemas) == 0
	
	async def create_triggers(self):
		"""Create database triggers for automatic timestamp updates"""
		
		# Tables that need updated_at triggers
		tables_with_timestamps = [
			'operation_logs', 'system_metrics', 'capability_configurations',
			'cv_projects', 'cv_image_jobs', 'cv_video_jobs', 'cv_face_embeddings',
			'cv_person_registry', 'iot_projects', 'iot_devices', 'iot_gateways',
			'iot_audio_recordings', 'iot_anomaly_detections',
			'audio_projects', 'audio_recordings', 'audio_transcriptions', 
			'audio_synthesis_jobs', 'audio_segments', 'audio_annotations',
			'dt_projects', 'digital_twins', 'dt_telemetry_data', 'dt_simulation_jobs',
			'dt_events', 'dt_templates'
		]
		
		print("Creating database triggers...")
		
		for table in tables_with_timestamps:
			trigger_sql = f"""
			CREATE TRIGGER update_{table}_updated_at
				BEFORE UPDATE ON {table}
				FOR EACH ROW
				EXECUTE FUNCTION update_updated_at_column();
			"""
			
			try:
				await self.connection.execute(trigger_sql)
				print(f"  ✅ Trigger created for {table}")
			except Exception as e:
				if "already exists" not in str(e):
					print(f"  ⚠️  Trigger for {table}: {e}")
	
	async def create_sample_data(self):
		"""Insert sample data for testing"""
		print("Creating sample data...")
		
		sample_data_sql = [
			# Sample CV project
			"""
			INSERT INTO cv_projects (name, description, project_type, status)
			VALUES ('Security Monitoring', 'AI-powered security camera analysis', 'security', 'active')
			ON CONFLICT DO NOTHING;
			""",
			
			# Sample IoT project
			"""
			INSERT INTO iot_projects (name, description, project_type, deployment_environment)
			VALUES ('Smart Building', 'IoT sensors for building automation', 'smart_home', 'indoor')
			ON CONFLICT DO NOTHING;
			""",
			
			# Sample audio project
			"""
			INSERT INTO audio_projects (name, description, project_type, language)
			VALUES ('Meeting Transcripts', 'Automated meeting transcription system', 'transcription', 'en')
			ON CONFLICT DO NOTHING;
			""",
			
			# Sample person for face recognition
			"""
			INSERT INTO cv_person_registry (person_id, name, description, is_active)
			VALUES ('admin_001', 'System Administrator', 'Default admin user for testing', true)
			ON CONFLICT DO NOTHING;
			""",
			
			# Sample digital twin project
			"""
			INSERT INTO dt_projects (name, description, project_type, industry)
			VALUES ('Smart Factory', 'Digital twin of manufacturing facility', 'industrial', 'manufacturing')
			ON CONFLICT DO NOTHING;
			""",
			
			# Sample digital twin template
			"""
			INSERT INTO dt_templates (template_name, display_name, description, category, twin_type)
			VALUES ('industrial_machine', 'Industrial Machine', 'Template for industrial machinery digital twins', 'machinery', 'asset')
			ON CONFLICT (template_name) DO NOTHING;
			"""
		]
		
		for sql in sample_data_sql:
			try:
				await self.connection.execute(sql)
				print(f"  ✅ Sample data inserted")
			except Exception as e:
				print(f"  ⚠️  Sample data insertion: {e}")
	
	async def verify_installation(self):
		"""Verify that all tables were created correctly"""
		print("Verifying database installation...")
		
		# Check that all expected tables exist
		table_check_sql = """
		SELECT table_name 
		FROM information_schema.tables 
		WHERE table_schema = 'public' 
		AND table_type = 'BASE TABLE'
		ORDER BY table_name;
		"""
		
		try:
			result = await self.connection.fetch(table_check_sql)
			tables = [row['table_name'] for row in result]
			
			print(f"  ✅ Found {len(tables)} tables in database:")
			for table in tables:
				print(f"    • {table}")
			
			# Check for essential tables
			essential_tables = [
				'operation_logs', 'cv_projects', 'cv_image_jobs', 
				'iot_projects', 'iot_devices', 'audio_projects', 'audio_recordings',
				'dt_projects', 'digital_twins', 'dt_simulation_jobs'
			]
			
			missing_tables = [table for table in essential_tables if table not in tables]
			if missing_tables:
				print(f"  ❌ Missing essential tables: {', '.join(missing_tables)}")
				return False
			else:
				print(f"  ✅ All essential tables present")
				return True
				
		except Exception as e:
			print(f"  ❌ Verification failed: {e}")
			return False
	
	async def get_database_stats(self):
		"""Get database statistics"""
		try:
			stats_sql = """
			SELECT 
				schemaname,
				tablename,
				n_tup_ins as inserts,
				n_tup_upd as updates,
				n_tup_del as deletes
			FROM pg_stat_user_tables
			ORDER BY tablename;
			"""
			
			result = await self.connection.fetch(stats_sql)
			
			print("\nDatabase Statistics:")
			print(f"{'Table':<30} {'Inserts':<10} {'Updates':<10} {'Deletes':<10}")
			print("-" * 70)
			
			for row in result:
				print(f"{row['tablename']:<30} {row['inserts']:<10} {row['updates']:<10} {row['deletes']:<10}")
				
		except Exception as e:
			print(f"Could not retrieve database stats: {e}")
	
	async def initialize_database(self, create_sample_data: bool = True):
		"""Complete database initialization process"""
		print("🚀 Starting APG PostgreSQL Database Initialization")
		print("=" * 60)
		
		try:
			# Connect to database
			await self.connect()
			
			# Create extensions
			await self.create_extensions()
			
			# Create utility functions
			await self.create_base_functions()
			
			# Create all schemas
			success = await self.create_all_schemas()
			if not success:
				print("❌ Schema creation failed - aborting")
				return False
			
			# Create triggers
			await self.create_triggers()
			
			# Insert sample data if requested
			if create_sample_data:
				await self.create_sample_data()
			
			# Verify installation
			verified = await self.verify_installation()
			
			# Show database stats
			await self.get_database_stats()
			
			if verified:
				print("\n🎉 Database initialization completed successfully!")
				print("✅ APG capabilities database is ready for use")
				return True
			else:
				print("\n❌ Database initialization completed with errors")
				return False
				
		except Exception as e:
			print(f"\n💥 Database initialization failed: {e}")
			return False
		
		finally:
			await self.disconnect()

async def main():
	"""Main initialization function"""
	import argparse
	
	parser = argparse.ArgumentParser(description='Initialize APG PostgreSQL Database')
	parser.add_argument('--connection-string', '-c', 
						help='PostgreSQL connection string')
	parser.add_argument('--no-sample-data', action='store_true',
						help='Skip creating sample data')
	parser.add_argument('--verify-only', action='store_true',
						help='Only verify existing installation')
	
	args = parser.parse_args()
	
	# Initialize database
	initializer = PostgreSQLInitializer(args.connection_string)
	
	if args.verify_only:
		await initializer.connect()
		verified = await initializer.verify_installation()
		await initializer.get_database_stats()
		await initializer.disconnect()
		sys.exit(0 if verified else 1)
	else:
		success = await initializer.initialize_database(
			create_sample_data=not args.no_sample_data
		)
		sys.exit(0 if success else 1)

if __name__ == "__main__":
	asyncio.run(main())