"""
APG Customer Relationship Management - Contact Relationships Migration

Database migration to create contact relationships table and supporting structures
for mapping and analyzing relationships between contacts.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from .base_migration import BaseMigration, MigrationDirection


logger = logging.getLogger(__name__)


class ContactRelationshipsMigration(BaseMigration):
	"""Migration for contact relationships functionality"""
	
	def _get_migration_id(self) -> str:
		return "003_contact_relationships"
	
	def _get_version(self) -> str:
		return "003"
	
	def _get_description(self) -> str:
		return "Create contact relationships table and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating contact relationships table...")
			
			# Create relationship type enum
			await connection.execute("""
				CREATE TYPE crm_relationship_type AS ENUM (
					'colleague', 'manager', 'subordinate', 'partner', 'vendor',
					'customer', 'referrer', 'referred', 'family', 'friend',
					'mentor', 'mentee', 'competitor', 'collaborator', 'influencer',
					'decision_maker', 'gatekeeper', 'champion', 'detractor', 'neutral'
				)
			""")
			
			# Create relationship strength enum
			await connection.execute("""
				CREATE TYPE crm_relationship_strength AS ENUM (
					'weak', 'moderate', 'strong', 'very_strong'
				)
			""")
			
			# Create relationship status enum
			await connection.execute("""
				CREATE TYPE crm_relationship_status AS ENUM (
					'active', 'inactive', 'terminated', 'pending'
				)
			""")
			
			# Create contact relationships table
			await connection.execute("""
				CREATE TABLE crm_contact_relationships (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					from_contact_id TEXT NOT NULL,
					to_contact_id TEXT NOT NULL,
					relationship_type crm_relationship_type NOT NULL,
					relationship_strength crm_relationship_strength DEFAULT 'moderate',
					relationship_status crm_relationship_status DEFAULT 'active',
					is_mutual BOOLEAN DEFAULT FALSE,
					confidence_score DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
					notes TEXT,
					tags TEXT[] DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					
					-- Source information
					source TEXT,
					source_confidence DECIMAL(3,2) DEFAULT 0.5 CHECK (source_confidence >= 0.0 AND source_confidence <= 1.0),
					verified_at TIMESTAMP WITH TIME ZONE,
					verified_by TEXT,
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Foreign key constraints
					FOREIGN KEY (from_contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE,
					FOREIGN KEY (to_contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_different_contacts CHECK (from_contact_id != to_contact_id),
					CONSTRAINT check_notes_length CHECK (char_length(notes) <= 2000)
				)
			""")
			
			# Create indexes for performance
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_tenant 
				ON crm_contact_relationships(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_from_contact 
				ON crm_contact_relationships(from_contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_to_contact 
				ON crm_contact_relationships(to_contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_type 
				ON crm_contact_relationships(relationship_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_strength 
				ON crm_contact_relationships(relationship_strength)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_status 
				ON crm_contact_relationships(relationship_status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_created_at 
				ON crm_contact_relationships(created_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_confidence 
				ON crm_contact_relationships(confidence_score)
			""")
			
			# Create composite indexes for common queries
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_tenant_from 
				ON crm_contact_relationships(tenant_id, from_contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_tenant_to 
				ON crm_contact_relationships(tenant_id, to_contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_from_to 
				ON crm_contact_relationships(from_contact_id, to_contact_id)
			""")
			
			# Create unique constraint to prevent duplicate relationships
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_contact_relationships_unique 
				ON crm_contact_relationships(tenant_id, from_contact_id, to_contact_id)
			""")
			
			# Create GIN index for metadata and tags
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_metadata 
				ON crm_contact_relationships USING GIN (metadata)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_relationships_tags 
				ON crm_contact_relationships USING GIN (tags)
			""")
			
			# Create trigger for updating updated_at timestamp
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_contact_relationships_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_contact_relationships_updated_at
					BEFORE UPDATE ON crm_contact_relationships
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_contact_relationships_updated_at()
			""")
			
			# Create view for relationship analytics
			await connection.execute("""
				CREATE VIEW crm_contact_relationship_stats AS
				SELECT 
					c.id as contact_id,
					c.tenant_id,
					c.first_name,
					c.last_name,
					c.email,
					c.company,
					
					-- Outgoing relationships
					(SELECT COUNT(*) FROM crm_contact_relationships r 
					 WHERE r.from_contact_id = c.id AND r.tenant_id = c.tenant_id) as outgoing_relationships,
					
					-- Incoming relationships
					(SELECT COUNT(*) FROM crm_contact_relationships r 
					 WHERE r.to_contact_id = c.id AND r.tenant_id = c.tenant_id) as incoming_relationships,
					
					-- Total relationships
					(SELECT COUNT(*) FROM crm_contact_relationships r 
					 WHERE (r.from_contact_id = c.id OR r.to_contact_id = c.id) 
					 AND r.tenant_id = c.tenant_id) as total_relationships,
					
					-- Strong relationships
					(SELECT COUNT(*) FROM crm_contact_relationships r 
					 WHERE (r.from_contact_id = c.id OR r.to_contact_id = c.id) 
					 AND r.tenant_id = c.tenant_id 
					 AND r.relationship_strength IN ('strong', 'very_strong')) as strong_relationships,
					
					-- Average confidence score
					(SELECT AVG(r.confidence_score) FROM crm_contact_relationships r 
					 WHERE (r.from_contact_id = c.id OR r.to_contact_id = c.id) 
					 AND r.tenant_id = c.tenant_id) as avg_confidence_score
					
				FROM crm_contacts c
			""")
			
			# Create function for relationship path finding
			await connection.execute("""
				CREATE OR REPLACE FUNCTION find_relationship_path(
					start_contact_id TEXT,
					end_contact_id TEXT,
					max_depth INTEGER DEFAULT 3,
					tenant_filter TEXT DEFAULT NULL
				)
				RETURNS TABLE(
					path_length INTEGER,
					path_contacts TEXT[],
					path_relationships TEXT[]
				) AS $$
				BEGIN
					-- This is a placeholder for future graph traversal implementation
					-- For now, return empty result
					RETURN;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Contact relationships table created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create contact relationships table: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back contact relationships migration...")
			
			# Drop view
			await connection.execute("DROP VIEW IF EXISTS crm_contact_relationship_stats CASCADE")
			
			# Drop function
			await connection.execute("DROP FUNCTION IF EXISTS find_relationship_path CASCADE")
			
			# Drop trigger and function
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_contact_relationships_updated_at ON crm_contact_relationships")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_contact_relationships_updated_at CASCADE")
			
			# Drop table (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_contact_relationships CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_relationship_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_relationship_strength CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_relationship_type CASCADE")
			
			logger.info("✅ Contact relationships migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback contact relationships migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if table exists
			table_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.tables 
					WHERE table_name = 'crm_contact_relationships'
				)
			""")
			
			if not table_exists:
				return False
			
			# Check if enums exist
			enum_exists = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN ('crm_relationship_type', 'crm_relationship_strength', 'crm_relationship_status')
			""")
			
			if enum_exists != 3:
				return False
			
			# Check if view exists
			view_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.views 
					WHERE table_name = 'crm_contact_relationship_stats'
				)
			""")
			
			if not view_exists:
				return False
			
			# Check if indexes exist (check a few key ones)
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename = 'crm_contact_relationships'
				AND indexname IN (
					'idx_crm_contact_relationships_tenant',
					'idx_crm_contact_relationships_from_contact',
					'idx_crm_contact_relationships_unique'
				)
			""")
			
			if index_count < 3:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False