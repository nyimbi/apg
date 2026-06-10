#!/usr/bin/env python3
"""
APG Metadata Management - Database Management
Advanced multi-tenant database operations with PostgreSQL + Neo4j integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
from sqlalchemy import create_engine, text, Index, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from neo4j import AsyncGraphDatabase, Driver as Neo4jDriver
import redis.asyncio as aioredis

from .models import (
	Base, MetaAsset, MetaAssetVersion, MetaLineage, MetaClassification,
	MetaQualityAssessment, MetaGovernancePolicy, MetaUserActivity,
	MetaComment, MetaBookmark, MetaSearchHistory
)


@dataclass
class DatabaseConfig:
	"""Database configuration parameters"""
	postgres_url: str
	neo4j_url: str
	neo4j_username: str
	neo4j_password: str
	redis_url: str
	pool_size: int = 20
	max_overflow: int = 30
	pool_timeout: int = 30
	pool_recycle: int = 3600
	echo: bool = False
	enable_query_cache: bool = True
	cache_ttl: int = 300


class DatabaseHealthStatus:
	"""Database health monitoring"""
	
	def __init__(self):
		self.postgres_healthy: bool = False
		self.neo4j_healthy: bool = False
		self.redis_healthy: bool = False
		self.last_check: datetime = datetime.utcnow()
		self.postgres_latency: float = 0.0
		self.neo4j_latency: float = 0.0
		self.redis_latency: float = 0.0
		self.error_messages: List[str] = []
	
	@property
	def is_healthy(self) -> bool:
		"""Overall health status"""
		return self.postgres_healthy and self.neo4j_healthy and self.redis_healthy
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for API responses"""
		return {
			"overall_healthy": self.is_healthy,
			"postgres": {
				"healthy": self.postgres_healthy,
				"latency_ms": round(self.postgres_latency * 1000, 2)
			},
			"neo4j": {
				"healthy": self.neo4j_healthy,
				"latency_ms": round(self.neo4j_latency * 1000, 2)
			},
			"redis": {
				"healthy": self.redis_healthy,
				"latency_ms": round(self.redis_latency * 1000, 2)
			},
			"last_check": self.last_check.isoformat(),
			"errors": self.error_messages
		}


class MetaDatabaseManager:
	"""Advanced metadata database manager with multi-tenant isolation and graph capabilities"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = self._build_config(config or {})
		self.health_status = DatabaseHealthStatus()
		
		# PostgreSQL for structured metadata
		self.postgres_engine = create_async_engine(
			self.config.postgres_url,
			poolclass=QueuePool,
			pool_size=self.config.pool_size,
			max_overflow=self.config.max_overflow,
			pool_timeout=self.config.pool_timeout,
			pool_recycle=self.config.pool_recycle,
			echo=self.config.echo,
			connect_args={
				"server_settings": {
					"application_name": "apg_metadata",
					"jit": "off"
				}
			}
		)
		
		# Session factory
		self.async_session_factory = async_sessionmaker(
			self.postgres_engine,
			class_=AsyncSession,
			expire_on_commit=False,
			autoflush=False,
			autocommit=False
		)
		
		# Neo4j for complex lineage relationships
		self.neo4j_driver: Optional[Neo4jDriver] = None
		
		# Redis for caching and session management  
		self.redis_client: Optional[aioredis.Redis] = None
		
		# Multi-tenancy support
		self.tenant_schemas: Dict[str, str] = {}
		self.initialized = False
	
	def _build_config(self, config_dict: Dict[str, Any]) -> DatabaseConfig:
		"""Build database configuration from dictionary and environment"""
		return DatabaseConfig(
			postgres_url=config_dict.get('postgres_url') or os.getenv(
				'META_POSTGRES_URL', 
				'postgresql+asyncpg://postgres:postgres@localhost:5432/apg_meta'
			),
			neo4j_url=config_dict.get('neo4j_url') or os.getenv(
				'META_NEO4J_URL',
				'bolt://localhost:7687'
			),
			neo4j_username=config_dict.get('neo4j_username') or os.getenv(
				'META_NEO4J_USERNAME',
				'neo4j'
			),
			neo4j_password=config_dict.get('neo4j_password') or os.getenv(
				'META_NEO4J_PASSWORD',
				'password'
			),
			redis_url=config_dict.get('redis_url') or os.getenv(
				'META_REDIS_URL',
				'redis://localhost:6379/2'
			),
			pool_size=config_dict.get('pool_size', 20),
			max_overflow=config_dict.get('max_overflow', 30),
			echo=config_dict.get('debug', False)
		)
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize all database connections and schemas"""
		if self.initialized:
			return {"status": "already_initialized"}
		
		try:
			# Initialize PostgreSQL
			await self._init_postgres()
			
			# Initialize Neo4j
			await self._init_neo4j()
			
			# Initialize Redis
			await self._init_redis()
			
			# Run initial health check
			await self.health_check()
			
			self.initialized = True
			
			return {
				"status": "initialized",
				"postgres": "connected",
				"neo4j": "connected",
				"redis": "connected",
				"health": self.health_status.to_dict()
			}
			
		except Exception as e:
			await self._log_error(f"Database initialization failed: {str(e)}")
			raise
	
	async def _init_postgres(self):
		"""Initialize PostgreSQL connection and schemas"""
		try:
			# Test connection
			async with self.postgres_engine.begin() as conn:
				await conn.execute(text("SELECT 1"))
			
			# Create database schema if not exists
			await self.create_schema()
			
			# Set up query performance monitoring
			await self._setup_postgres_monitoring()
			
		except Exception as e:
			await self._log_error(f"PostgreSQL initialization failed: {str(e)}")
			raise
	
	async def _init_neo4j(self):
		"""Initialize Neo4j connection for lineage graphs"""
		try:
			self.neo4j_driver = AsyncGraphDatabase.driver(
				self.config.neo4j_url,
				auth=(self.config.neo4j_username, self.config.neo4j_password),
				max_connection_lifetime=3600,
				max_connection_pool_size=50,
				connection_acquisition_timeout=60
			)
			
			# Test connection
			await self.neo4j_driver.verify_connectivity()
			
			# Create lineage graph schema
			await self._create_neo4j_schema()
			
		except Exception as e:
			await self._log_error(f"Neo4j initialization failed: {str(e)}")
			raise
	
	async def _init_redis(self):
		"""Initialize Redis connection for caching"""
		try:
			self.redis_client = aioredis.from_url(
				self.config.redis_url,
				encoding="utf-8",
				decode_responses=True,
				health_check_interval=30,
				socket_keepalive=True,
				socket_keepalive_options={}
			)
			
			# Test connection
			await self.redis_client.ping()
			
		except Exception as e:
			await self._log_error(f"Redis initialization failed: {str(e)}")
			raise
	
	async def create_schema(self):
		"""Create database schema and indexes"""
		try:
			async with self.postgres_engine.begin() as conn:
				# Create all tables
				await conn.run_sync(Base.metadata.create_all)
				
				# Create additional performance indexes
				await self._create_performance_indexes(conn)
				
				# Create full-text search indexes
				await self._create_fulltext_indexes(conn)
				
				# Set up row-level security for multi-tenancy
				await self._setup_row_level_security(conn)
				
		except Exception as e:
			await self._log_error(f"Schema creation failed: {str(e)}")
			raise
	
	async def _create_performance_indexes(self, conn):
		"""Create additional performance indexes"""
		indexes = [
			# Composite indexes for common queries
			"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meta_assets_search "
			"ON meta_assets USING GIN ((name || ' ' || COALESCE(description, '')) gin_trgm_ops)",
			
			# Partial indexes for active records
			"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meta_assets_active "
			"ON meta_assets (tenant_id, asset_type) WHERE status = 'active' AND is_deleted = false",
			
			# Quality score range queries
			"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meta_assets_quality_range "
			"ON meta_assets (tenant_id, quality_score DESC) WHERE quality_score IS NOT NULL",
			
			# Lineage path queries
			"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meta_lineage_path "
			"ON meta_lineage (tenant_id, source_asset_id, target_asset_id) WHERE is_active = true",
			
			# Classification queries
			"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meta_classifications_asset_type "
			"ON meta_classifications (asset_id, classification_type) WHERE status = 'approved'",
			
			# User activity analytics
			"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meta_activities_user_time "
			"ON meta_user_activities (user_id, timestamp DESC)",
			
			# Search performance
			"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meta_search_queries "
			"ON meta_search_history (tenant_id, timestamp DESC)"
		]
		
		for index_sql in indexes:
			try:
				await conn.execute(text(index_sql))
			except Exception as e:
				await self._log_error(f"Index creation failed: {index_sql} - {str(e)}")
	
	async def _create_fulltext_indexes(self, conn):
		"""Create full-text search indexes using PostgreSQL trigram extension"""
		try:
			# Enable extensions
			await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
			await conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
			await conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gin"))
			
			# Create full-text search configuration
			search_config_sql = """
			CREATE TEXT SEARCH CONFIGURATION IF NOT EXISTS meta_search (COPY = pg_catalog.english);
			ALTER TEXT SEARCH CONFIGURATION meta_search
			ALTER MAPPING FOR hword, hword_part, word WITH unaccent, simple;
			"""
			await conn.execute(text(search_config_sql))
			
		except Exception as e:
			await self._log_error(f"Full-text index creation failed: {str(e)}")
	
	async def _setup_row_level_security(self, conn):
		"""Set up row-level security for multi-tenant isolation"""
		try:
			# Enable RLS on all tenant-aware tables
			tables_with_rls = [
				'meta_assets', 'meta_asset_versions', 'meta_lineage', 
				'meta_classifications', 'meta_quality_assessments',
				'meta_governance_policies', 'meta_user_activities',
				'meta_comments', 'meta_bookmarks', 'meta_search_history'
			]
			
			for table in tables_with_rls:
				rls_sql = f"""
				ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
				
				CREATE POLICY IF NOT EXISTS {table}_tenant_isolation ON {table}
				FOR ALL TO PUBLIC
				USING (tenant_id = current_setting('meta.current_tenant_id', true));
				"""
				await conn.execute(text(rls_sql))
				
		except Exception as e:
			await self._log_error(f"RLS setup failed: {str(e)}")
	
	async def _create_neo4j_schema(self):
		"""Create Neo4j schema for lineage graphs"""
		try:
			async with self.neo4j_driver.session() as session:
				# Create constraints and indexes
				constraints = [
					"CREATE CONSTRAINT asset_id_unique IF NOT EXISTS FOR (a:Asset) REQUIRE a.asset_id IS UNIQUE",
					"CREATE CONSTRAINT tenant_asset_unique IF NOT EXISTS FOR (a:Asset) REQUIRE (a.tenant_id, a.asset_id) IS UNIQUE",
					"CREATE INDEX asset_tenant_type IF NOT EXISTS FOR (a:Asset) ON (a.tenant_id, a.asset_type)",
					"CREATE INDEX asset_name IF NOT EXISTS FOR (a:Asset) ON a.name",
					"CREATE INDEX lineage_tenant IF NOT EXISTS FOR ()-[r:LINEAGE]-() ON r.tenant_id"
				]
				
				for constraint in constraints:
					try:
						await session.run(constraint)
					except Exception as e:
						# Constraint might already exist
						await self._log_error(f"Neo4j constraint warning: {str(e)}")
				
		except Exception as e:
			await self._log_error(f"Neo4j schema creation failed: {str(e)}")
			raise
	
	async def _setup_postgres_monitoring(self):
		"""Set up PostgreSQL performance monitoring"""
		try:
			async with self.postgres_engine.begin() as conn:
				# Enable query statistics
				await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
				
				# Create monitoring views if needed
				monitoring_view = """
				CREATE OR REPLACE VIEW meta_query_stats AS
				SELECT 
					query,
					calls,
					total_time,
					mean_time,
					rows
				FROM pg_stat_statements 
				WHERE query LIKE '%meta_%'
				ORDER BY total_time DESC
				LIMIT 50;
				"""
				await conn.execute(text(monitoring_view))
				
		except Exception as e:
			await self._log_error(f"PostgreSQL monitoring setup failed: {str(e)}")
	
	@asynccontextmanager
	async def get_session(self, tenant_id: str = None) -> AsyncGenerator[AsyncSession, None]:
		"""Get database session with optional tenant context"""
		async with self.async_session_factory() as session:
			try:
				# Set tenant context for RLS
				if tenant_id:
					await session.execute(text(f"SET meta.current_tenant_id = '{tenant_id}'"))
				
				yield session
				await session.commit()
				
			except Exception:
				await session.rollback()
				raise
			finally:
				# Reset tenant context
				if tenant_id:
					try:
						await session.execute(text("RESET meta.current_tenant_id"))
					except Exception:
						pass
	
	@asynccontextmanager
	async def get_neo4j_session(self) -> AsyncGenerator:
		"""Get Neo4j session for lineage operations"""
		if not self.neo4j_driver:
			raise RuntimeError("Neo4j not initialized")
		
		async with self.neo4j_driver.session() as session:
			yield session
	
	async def execute_tenant_query(self, 
								  query: str, 
								  params: Dict[str, Any] = None,
								  tenant_id: str = None) -> Any:
		"""Execute query with tenant isolation"""
		async with self.get_session(tenant_id) as session:
			result = await session.execute(text(query), params or {})
			return result
	
	async def cache_get(self, key: str) -> Optional[str]:
		"""Get value from Redis cache"""
		if not self.redis_client:
			return None
		
		try:
			return await self.redis_client.get(key)
		except Exception as e:
			await self._log_error(f"Cache get failed: {str(e)}")
			return None
	
	async def cache_set(self, key: str, value: str, ttl: int = None) -> bool:
		"""Set value in Redis cache with optional TTL"""
		if not self.redis_client:
			return False
		
		try:
			ttl = ttl or self.config.cache_ttl
			return await self.redis_client.setex(key, ttl, value)
		except Exception as e:
			await self._log_error(f"Cache set failed: {str(e)}")
			return False
	
	async def cache_delete(self, key: str) -> bool:
		"""Delete value from Redis cache"""
		if not self.redis_client:
			return False
		
		try:
			return await self.redis_client.delete(key) > 0
		except Exception as e:
			await self._log_error(f"Cache delete failed: {str(e)}")
			return False
	
	async def cache_invalidate_pattern(self, pattern: str) -> int:
		"""Invalidate all cache keys matching pattern"""
		if not self.redis_client:
			return 0
		
		try:
			keys = await self.redis_client.keys(pattern)
			if keys:
				return await self.redis_client.delete(*keys)
			return 0
		except Exception as e:
			await self._log_error(f"Cache pattern invalidation failed: {str(e)}")
			return 0
	
	async def health_check(self) -> DatabaseHealthStatus:
		"""Comprehensive health check for all database components"""
		status = DatabaseHealthStatus()
		status.error_messages = []
		
		# Check PostgreSQL
		try:
			start_time = asyncio.get_event_loop().time()
			async with self.postgres_engine.begin() as conn:
				await conn.execute(text("SELECT 1"))
			status.postgres_latency = asyncio.get_event_loop().time() - start_time
			status.postgres_healthy = True
		except Exception as e:
			status.postgres_healthy = False
			status.error_messages.append(f"PostgreSQL: {str(e)}")
		
		# Check Neo4j
		try:
			start_time = asyncio.get_event_loop().time()
			if self.neo4j_driver:
				await self.neo4j_driver.verify_connectivity()
				status.neo4j_latency = asyncio.get_event_loop().time() - start_time
				status.neo4j_healthy = True
			else:
				status.error_messages.append("Neo4j: Not initialized")
		except Exception as e:
			status.neo4j_healthy = False
			status.error_messages.append(f"Neo4j: {str(e)}")
		
		# Check Redis
		try:
			start_time = asyncio.get_event_loop().time()
			if self.redis_client:
				await self.redis_client.ping()
				status.redis_latency = asyncio.get_event_loop().time() - start_time
				status.redis_healthy = True
			else:
				status.error_messages.append("Redis: Not initialized")
		except Exception as e:
			status.redis_healthy = False
			status.error_messages.append(f"Redis: {str(e)}")
		
		status.last_check = datetime.utcnow()
		self.health_status = status
		return status
	
	async def get_database_stats(self) -> Dict[str, Any]:
		"""Get comprehensive database statistics"""
		stats = {
			"postgres": {},
			"neo4j": {},
			"redis": {}
		}
		
		try:
			# PostgreSQL stats
			async with self.get_session() as session:
				# Table sizes
				size_query = """
				SELECT 
					schemaname,
					tablename,
					pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
					pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size_pretty
				FROM pg_tables 
				WHERE tablename LIKE 'meta_%'
				ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
				"""
				result = await session.execute(text(size_query))
				stats["postgres"]["table_sizes"] = [dict(row) for row in result]
				
				# Connection stats
				conn_query = """
				SELECT 
					state,
					count(*) as count
				FROM pg_stat_activity 
				WHERE datname = current_database()
				GROUP BY state;
				"""
				result = await session.execute(text(conn_query))
				stats["postgres"]["connections"] = [dict(row) for row in result]
		
		except Exception as e:
			stats["postgres"]["error"] = str(e)
		
		try:
			# Neo4j stats
			if self.neo4j_driver:
				async with self.neo4j_driver.session() as session:
					# Node and relationship counts
					counts_query = """
					CALL apoc.meta.stats() 
					YIELD labels, relTypesCount
					RETURN labels, relTypesCount
					"""
					try:
						result = await session.run(counts_query)
						record = await result.single()
						if record:
							stats["neo4j"]["labels"] = record["labels"]
							stats["neo4j"]["relationships"] = record["relTypesCount"]
					except Exception:
						# Fallback if APOC is not available
						result = await session.run("MATCH (n) RETURN count(n) as node_count")
						record = await result.single()
						stats["neo4j"]["node_count"] = record["node_count"] if record else 0
		
		except Exception as e:
			stats["neo4j"]["error"] = str(e)
		
		try:
			# Redis stats
			if self.redis_client:
				redis_info = await self.redis_client.info()
				stats["redis"] = {
					"connected_clients": redis_info.get("connected_clients", 0),
					"used_memory_human": redis_info.get("used_memory_human", "0B"),
					"keyspace_hits": redis_info.get("keyspace_hits", 0),
					"keyspace_misses": redis_info.get("keyspace_misses", 0),
					"hit_rate": 0.0
				}
				
				# Calculate hit rate
				hits = stats["redis"]["keyspace_hits"]
				misses = stats["redis"]["keyspace_misses"]
				if hits + misses > 0:
					stats["redis"]["hit_rate"] = hits / (hits + misses) * 100
		
		except Exception as e:
			stats["redis"]["error"] = str(e)
		
		return stats
	
	async def create_lineage_graph(self, 
								  source_asset_id: str,
								  target_asset_id: str,
								  relationship_type: str,
								  tenant_id: str,
								  metadata: Dict[str, Any] = None) -> bool:
		"""Create lineage relationship in Neo4j graph"""
		if not self.neo4j_driver:
			return False
		
		try:
			async with self.neo4j_driver.session() as session:
				# Create or merge assets as nodes
				create_nodes_query = """
				MERGE (source:Asset {asset_id: $source_id, tenant_id: $tenant_id})
				MERGE (target:Asset {asset_id: $target_id, tenant_id: $tenant_id})
				MERGE (source)-[r:LINEAGE {
					type: $rel_type,
					tenant_id: $tenant_id,
					created_at: datetime(),
					metadata: $metadata
				}]->(target)
				RETURN r
				"""
				
				await session.run(
					create_nodes_query,
					source_id=source_asset_id,
					target_id=target_asset_id,
					tenant_id=tenant_id,
					rel_type=relationship_type,
					metadata=metadata or {}
				)
				
				return True
				
		except Exception as e:
			await self._log_error(f"Lineage graph creation failed: {str(e)}")
			return False
	
	async def get_lineage_path(self,
							  asset_id: str,
							  tenant_id: str,
							  direction: str = "both",
							  max_depth: int = 10) -> List[Dict[str, Any]]:
		"""Get lineage path for an asset from Neo4j"""
		if not self.neo4j_driver:
			return []
		
		try:
			async with self.neo4j_driver.session() as session:
				if direction == "upstream":
					query = """
					MATCH path = (target:Asset {asset_id: $asset_id, tenant_id: $tenant_id})
					           <-[:LINEAGE*1..$max_depth]-(source:Asset {tenant_id: $tenant_id})
					RETURN path
					"""
				elif direction == "downstream":
					query = """
					MATCH path = (source:Asset {asset_id: $asset_id, tenant_id: $tenant_id})
					           -[:LINEAGE*1..$max_depth]->(target:Asset {tenant_id: $tenant_id})
					RETURN path
					"""
				else:  # both
					query = """
					MATCH path = (n:Asset {tenant_id: $tenant_id})
					           -[:LINEAGE*1..$max_depth]-(asset:Asset {asset_id: $asset_id, tenant_id: $tenant_id})
					RETURN path
					"""
				
				result = await session.run(
					query,
					asset_id=asset_id,
					tenant_id=tenant_id,
					max_depth=max_depth
				)
				
				paths = []
				async for record in result:
					path_data = record["path"]
					paths.append({
						"nodes": [dict(node) for node in path_data.nodes],
						"relationships": [dict(rel) for rel in path_data.relationships]
					})
				
				return paths
				
		except Exception as e:
			await self._log_error(f"Lineage path query failed: {str(e)}")
			return []
	
	async def optimize_performance(self) -> Dict[str, Any]:
		"""Run performance optimization tasks"""
		results = {
			"postgres_vacuum": False,
			"postgres_analyze": False,
			"neo4j_cleanup": False,
			"redis_cleanup": False
		}
		
		try:
			# PostgreSQL maintenance
			async with self.postgres_engine.begin() as conn:
				# Run VACUUM ANALYZE on metadata tables
				tables = [
					'meta_assets', 'meta_lineage', 'meta_classifications',
					'meta_quality_assessments', 'meta_user_activities'
				]
				
				for table in tables:
					await conn.execute(text(f"VACUUM ANALYZE {table}"))
				
				results["postgres_vacuum"] = True
				results["postgres_analyze"] = True
		
		except Exception as e:
			await self._log_error(f"PostgreSQL optimization failed: {str(e)}")
		
		try:
			# Neo4j cleanup
			if self.neo4j_driver:
				async with self.neo4j_driver.session() as session:
					# Remove orphaned nodes
					cleanup_query = """
					MATCH (n:Asset)
					WHERE NOT EXISTS((n)-[:LINEAGE]-())
					AND n.created_at < datetime() - duration('P30D')
					DELETE n
					"""
					await session.run(cleanup_query)
					results["neo4j_cleanup"] = True
		
		except Exception as e:
			await self._log_error(f"Neo4j cleanup failed: {str(e)}")
		
		try:
			# Redis cleanup
			if self.redis_client:
				# Remove expired keys
				await self.redis_client.flushdb()  # In production, be more selective
				results["redis_cleanup"] = True
		
		except Exception as e:
			await self._log_error(f"Redis cleanup failed: {str(e)}")
		
		return results
	
	async def _log_error(self, message: str):
		"""Log error message with timestamp"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META DB ERROR: {message}")
	
	async def close(self):
		"""Close all database connections"""
		try:
			if self.postgres_engine:
				await self.postgres_engine.dispose()
			
			if self.neo4j_driver:
				await self.neo4j_driver.close()
			
			if self.redis_client:
				await self.redis_client.close()
			
			self.initialized = False
			
		except Exception as e:
			await self._log_error(f"Database cleanup failed: {str(e)}")


# Utility functions for common database operations

async def create_database_manager(config: Dict[str, Any] = None) -> MetaDatabaseManager:
	"""Factory function to create and initialize database manager"""
	db_manager = MetaDatabaseManager(config)
	await db_manager.initialize()
	return db_manager


async def migrate_database(db_manager: MetaDatabaseManager) -> Dict[str, Any]:
	"""Run database migrations"""
	try:
		await db_manager.create_schema()
		return {
			"status": "success",
			"message": "Database migration completed",
			"timestamp": datetime.utcnow().isoformat()
		}
	except Exception as e:
		return {
			"status": "error",
			"message": f"Migration failed: {str(e)}",
			"timestamp": datetime.utcnow().isoformat()
		}