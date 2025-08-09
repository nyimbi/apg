#!/usr/bin/env python3
"""
APG Customer Relationship Management - Migration Runner

Command-line migration runner for managing CRM database schema changes
with comprehensive migration management, rollback capabilities, and status reporting.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add migrations directory to path
sys.path.append(str(Path(__file__).parent))

from migrations.migration_manager import MigrationManager, MigrationError


class MigrationRunner:
	"""Command-line migration runner"""
	
	def __init__(self):
		"""Initialize migration runner"""
		self.config = self._load_config()
		self.manager = MigrationManager(self.config["database"])
	
	def _load_config(self) -> Dict[str, Any]:
		"""Load configuration from environment or default values"""
		import os
		
		# Default configuration
		config = {
			"database": {
				"host": os.getenv("CRM_DB_HOST", "localhost"),
				"port": int(os.getenv("CRM_DB_PORT", "5432")),
				"database": os.getenv("CRM_DB_NAME", "crm_db"),
				"user": os.getenv("CRM_DB_USER", "crm_user"),
				"password": os.getenv("CRM_DB_PASSWORD", "crm_password")
			}
		}
		
		# Try to load from config file
		config_file = Path(__file__).parent / "config.json"
		if config_file.exists():
			try:
				with open(config_file) as f:
					file_config = json.load(f)
					config.update(file_config)
			except Exception as e:
				logger.warning(f"Failed to load config file: {e}")
		
		return config
	
	async def migrate_to_latest(self):
		"""Migrate to the latest schema version"""
		try:
			print("🚀 Migrating CRM database to latest version...")
			
			await self.manager.initialize()
			result = await self.manager.migrate_to_latest()
			
			if result["status"] == "up_to_date":
				print("✅ Database is already at the latest version")
			elif result["status"] == "completed":
				print(f"🎉 Successfully executed {len(result['executed_migrations'])} migrations")
				print(f"⏱️ Migration completed in {result['duration_seconds']:.2f} seconds")
				
				if result["executed_migrations"]:
					print("\n📋 Executed migrations:")
					for migration_id in result["executed_migrations"]:
						print(f"  ✅ {migration_id}")
			else:
				print(f"❌ Migration failed: {result.get('status', 'unknown error')}")
				if result.get("failed_migrations"):
					print("Failed migrations:")
					for migration_id in result["failed_migrations"]:
						print(f"  ❌ {migration_id}")
				return 1
			
		except Exception as e:
			print(f"❌ Migration failed: {str(e)}")
			logger.error("Migration failed", exc_info=True)
			return 1
		finally:
			await self.manager.shutdown()
		
		return 0
	
	async def migrate(self, migration_ids: list):
		"""Migrate specific migrations"""
		try:
			print(f"🚀 Executing migrations: {', '.join(migration_ids)}")
			
			await self.manager.initialize()
			result = await self.manager.migrate(migration_ids)
			
			if result["status"] == "completed":
				print(f"🎉 Successfully executed {len(result['executed_migrations'])} migrations")
				print(f"⏱️ Migration completed in {result['duration_seconds']:.2f} seconds")
				
				if result["executed_migrations"]:
					print("\n📋 Executed migrations:")
					for migration_id in result["executed_migrations"]:
						print(f"  ✅ {migration_id}")
			else:
				print(f"❌ Migration failed: {result.get('status', 'unknown error')}")
				if result.get("failed_migrations"):
					print("Failed migrations:")
					for migration_id in result["failed_migrations"]:
						print(f"  ❌ {migration_id}")
				return 1
			
		except Exception as e:
			print(f"❌ Migration failed: {str(e)}")
			logger.error("Migration failed", exc_info=True)
			return 1
		finally:
			await self.manager.shutdown()
		
		return 0
	
	async def rollback(self, target_migration_id: str = None):
		"""Rollback migrations"""
		try:
			if target_migration_id:
				print(f"🔄 Rolling back to migration: {target_migration_id}")
			else:
				print("🔄 Rolling back last migration...")
			
			await self.manager.initialize()
			result = await self.manager.rollback(target_migration_id)
			
			if result["status"] == "no_rollback_needed":
				print("✅ No migrations to rollback")
			elif result["status"] == "completed":
				print(f"🎉 Successfully rolled back {len(result['rolled_back_migrations'])} migrations")
				print(f"⏱️ Rollback completed in {result['duration_seconds']:.2f} seconds")
				
				if result["rolled_back_migrations"]:
					print("\n📋 Rolled back migrations:")
					for migration_id in result["rolled_back_migrations"]:
						print(f"  🔄 {migration_id}")
			else:
				print(f"❌ Rollback failed: {result.get('status', 'unknown error')}")
				return 1
			
		except Exception as e:
			print(f"❌ Rollback failed: {str(e)}")
			logger.error("Rollback failed", exc_info=True)
			return 1
		finally:
			await self.manager.shutdown()
		
		return 0
	
	async def status(self):
		"""Show migration status"""
		try:
			print("📊 CRM Database Migration Status")
			print("=" * 50)
			
			await self.manager.initialize()
			status = await self.manager.get_migration_status()
			
			print(f"Database Version: {status['database_version'] or 'None'}")
			print(f"Total Migrations: {status['total_migrations']}")
			print(f"Applied Migrations: {status['applied_migrations']}")
			print(f"Pending Migrations: {status['pending_migrations']}")
			print(f"Status: {status['status']}")
			
			if status.get('last_migration_at'):
				print(f"Last Migration: {status['last_migration_at']}")
			
			if status['migrations']:
				print("\n📋 Migration Details:")
				print("-" * 80)
				print(f"{'ID':<30} {'Version':<10} {'Status':<10} {'Description'}")
				print("-" * 80)
				
				for migration in status['migrations']:
					status_icon = "✅" if migration['is_applied'] else "⏳"
					status_text = "Applied" if migration['is_applied'] else "Pending"
					
					print(f"{status_icon} {migration['migration_id']:<28} {migration['version']:<10} {status_text:<10} {migration['description']}")
			
		except Exception as e:
			print(f"❌ Failed to get migration status: {str(e)}")
			logger.error("Status check failed", exc_info=True)
			return 1
		finally:
			await self.manager.shutdown()
		
		return 0
	
	async def validate(self):
		"""Validate database schema"""
		try:
			print("🔍 Validating CRM database schema...")
			
			await self.manager.initialize()
			validation = await self.manager.validate_schema()
			
			print(f"Schema Status: {validation['status']}")
			
			if validation['errors']:
				print("\n❌ Validation Errors:")
				for error in validation['errors']:
					print(f"  • {error}")
			
			if validation['warnings']:
				print("\n⚠️ Validation Warnings:")
				for warning in validation['warnings']:
					print(f"  • {warning}")
			
			if validation['migration_validations']:
				print("\n📋 Migration Validations:")
				for migration_id, result in validation['migration_validations'].items():
					status_icon = "✅" if result['valid'] else "❌"
					print(f"{status_icon} {migration_id}: {result['migration_name']}")
			
			if validation['status'] == 'valid':
				print("\n🎉 Schema validation passed!")
				return 0
			else:
				print(f"\n❌ Schema validation failed: {validation['status']}")
				return 1
			
		except Exception as e:
			print(f"❌ Schema validation failed: {str(e)}")
			logger.error("Schema validation failed", exc_info=True)
			return 1
		finally:
			await self.manager.shutdown()
	
	async def health_check(self):
		"""Check migration system health"""
		try:
			print("🏥 CRM Migration System Health Check")
			print("=" * 50)
			
			await self.manager.initialize()
			health = await self.manager.health_check()
			
			status_icon = "✅" if health['status'] == 'healthy' else "❌"
			print(f"Overall Status: {status_icon} {health['status']}")
			print(f"Initialized: {'✅' if health['initialized'] else '❌'}")
			print(f"Database Connected: {'✅' if health['database_connected'] else '❌'}")
			print(f"Migration Table: {'✅' if health['migration_table_exists'] else '❌'}")
			print(f"Applied Migrations: {health['applied_migrations']}")
			print(f"Available Migrations: {health['available_migrations']}")
			print(f"Migration Records: {health['migration_table_rows']}")
			print(f"Check Time: {health['timestamp']}")
			
			if health.get('error'):
				print(f"\n❌ Error: {health['error']}")
				return 1
			
			return 0 if health['status'] == 'healthy' else 1
			
		except Exception as e:
			print(f"❌ Health check failed: {str(e)}")
			logger.error("Health check failed", exc_info=True)
			return 1
		finally:
			if hasattr(self, 'manager') and self.manager._initialized:
				await self.manager.shutdown()


async def main():
	"""Main entry point"""
	parser = argparse.ArgumentParser(
		description="CRM Database Migration Manager",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python migrate.py migrate          # Migrate to latest version
  python migrate.py migrate 001      # Migrate specific version
  python migrate.py rollback         # Rollback last migration
  python migrate.py rollback 001     # Rollback to specific version
  python migrate.py status           # Show migration status
  python migrate.py validate         # Validate schema
  python migrate.py health           # Health check
		"""
	)
	
	subparsers = parser.add_subparsers(dest='command', help='Available commands')
	
	# Migrate command
	migrate_parser = subparsers.add_parser('migrate', help='Apply migrations')
	migrate_parser.add_argument('migrations', nargs='*', help='Specific migration IDs to apply')
	
	# Rollback command
	rollback_parser = subparsers.add_parser('rollback', help='Rollback migrations')
	rollback_parser.add_argument('target', nargs='?', help='Target migration ID to rollback to')
	
	# Status command
	subparsers.add_parser('status', help='Show migration status')
	
	# Validate command
	subparsers.add_parser('validate', help='Validate database schema')
	
	# Health command
	subparsers.add_parser('health', help='Check system health')
	
	args = parser.parse_args()
	
	if not args.command:
		parser.print_help()
		return 1
	
	runner = MigrationRunner()
	
	try:
		if args.command == 'migrate':
			if args.migrations:
				return await runner.migrate(args.migrations)
			else:
				return await runner.migrate_to_latest()
		
		elif args.command == 'rollback':
			return await runner.rollback(args.target)
		
		elif args.command == 'status':
			return await runner.status()
		
		elif args.command == 'validate':
			return await runner.validate()
		
		elif args.command == 'health':
			return await runner.health_check()
		
		else:
			print(f"Unknown command: {args.command}")
			return 1
			
	except KeyboardInterrupt:
		print("\n⚠️ Operation cancelled by user")
		return 1
	except Exception as e:
		print(f"❌ Unexpected error: {str(e)}")
		logger.error("Unexpected error", exc_info=True)
		return 1


if __name__ == "__main__":
	sys.exit(asyncio.run(main()))