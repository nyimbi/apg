"""
APG Customer Relationship Management - Database Migrations

Revolutionary database migration system providing comprehensive schema management,
versioning, rollback capabilities, and multi-tenant support for the CRM capability.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

__version__ = "1.0.0"

from .migration_manager import MigrationManager
from .base_migration import BaseMigration, MigrationDirection

__all__ = [
	"MigrationManager",
	"BaseMigration", 
	"MigrationDirection"
]