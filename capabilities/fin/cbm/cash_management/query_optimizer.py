#!/usr/bin/env python3
"""APG Cash Management - Advanced Query Optimizer

Intelligent SQL query optimization with automatic indexing,
query plan analysis, and performance tuning.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import time
import hashlib
import json
import statistics
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from contextlib import asynccontextmanager

import asyncpg
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str
import sqlparse
from sqlparse.sql import Statement, IdentifierList, Identifier, Function
from sqlparse.tokens import Keyword, DML

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(str, Enum):
	"""SQL query types."""
	SELECT = "SELECT"
	INSERT = "INSERT"
	UPDATE = "UPDATE"
	DELETE = "DELETE"
	CREATE = "CREATE"
	ALTER = "ALTER"
	DROP = "DROP"

class OptimizationStrategy(str, Enum):
	"""Query optimization strategies."""
	INDEX_OPTIMIZATION = "index_optimization"
	QUERY_REWRITING = "query_rewriting"
	PARTITION_PRUNING = "partition_pruning"
	JOIN_OPTIMIZATION = "join_optimization"
	SUBQUERY_OPTIMIZATION = "subquery_optimization"
	PARALLEL_EXECUTION = "parallel_execution"

@dataclass
class QueryPlan:
	"""Query execution plan analysis."""
	query_hash: str
	plan_json: Dict[str, Any]
	total_cost: float
	execution_time_ms: float
	rows_returned: int
	index_usage: List[str]
	table_scans: List[str]
	join_methods: List[str]
	optimization_opportunities: List[str] = field(default_factory=list)

@dataclass
class QueryStats:
	"""Query performance statistics."""
	query_hash: str
	execution_count: int
	total_time_ms: float
	average_time_ms: float
	min_time_ms: float
	max_time_ms: float
	rows_examined: int
	rows_returned: int
	cache_hits: int
	last_executed: datetime
	optimization_applied: bool = False

class QueryPattern(BaseModel):
	"""Query pattern for optimization."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	pattern_id: str = Field(default_factory=uuid7str)
	query_type: QueryType
	tables_involved: List[str]
	columns_accessed: List[str]
	where_conditions: List[str]
	join_conditions: List[str]
	order_by_columns: List[str]
	frequency: int = 0
	avg_execution_time_ms: float = 0.0
	optimization_potential: float = 0.0

class IndexRecommendation(BaseModel):
	"""Index recommendation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	recommendation_id: str = Field(default_factory=uuid7str)
	table_name: str
	columns: List[str]
	index_type: str = "btree"
	estimated_benefit: float
	estimated_cost: float
	queries_benefited: List[str]
	priority: str = "medium"  # low, medium, high, critical
	created_at: datetime = Field(default_factory=datetime.now)

class QueryOptimizer:
	"""Advanced SQL query optimization engine."""
	
	def __init__(
		self,
		tenant_id: str,
		db_pool: asyncpg.Pool,
		enable_auto_indexing: bool = True,
		enable_query_rewriting: bool = True
	):
		self.tenant_id = tenant_id
		self.db_pool = db_pool
		self.enable_auto_indexing = enable_auto_indexing
		self.enable_query_rewriting = enable_query_rewriting
		
		# Query tracking
		self.query_stats: Dict[str, QueryStats] = {}
		self.query_patterns: Dict[str, QueryPattern] = {}
		self.query_plans: Dict[str, QueryPlan] = {}
		
		# Optimization tracking
		self.index_recommendations: List[IndexRecommendation] = []
		self.applied_optimizations: Dict[str, List[str]] = {}
		
		# Performance monitoring
		self.slow_query_threshold_ms = 1000.0
		self.optimization_interval_hours = 24
		self.last_optimization = datetime.now()
		
		logger.info(f"Initialized QueryOptimizer for tenant {tenant_id}")
	
	async def analyze_query(
		self,
		sql: str,
		execution_time_ms: Optional[float] = None,
		rows_returned: Optional[int] = None
	) -> str:
		"""Analyze and optimize a SQL query."""
		query_hash = self._hash_query(sql)
		
		try:
			# Parse query
			parsed_query = self._parse_sql(sql)
			query_type = self._identify_query_type(parsed_query)
			
			# Extract query pattern
			pattern = await self._extract_query_pattern(parsed_query, query_type)
			
			# Get execution plan
			plan = await self._get_execution_plan(sql)
			
			# Update statistics
			await self._update_query_stats(
				query_hash, execution_time_ms, rows_returned
			)
			
			# Store query plan
			if plan:
				self.query_plans[query_hash] = plan
			
			# Check for optimization opportunities
			if execution_time_ms and execution_time_ms > self.slow_query_threshold_ms:
				await self._identify_optimization_opportunities(query_hash, sql, plan)
			
			# Apply real-time optimizations
			optimized_sql = await self._apply_query_optimizations(sql, pattern)
			
			return optimized_sql
			
		except Exception as e:
			logger.error(f"Query analysis failed for hash {query_hash}: {e}")
			return sql  # Return original query on error
	
	def _hash_query(self, sql: str) -> str:
		"""Generate hash for SQL query normalization."""
		# Normalize query for consistent hashing
		normalized = self._normalize_query(sql)
		return hashlib.sha256(normalized.encode()).hexdigest()[:16]
	
	def _normalize_query(self, sql: str) -> str:
		"""Normalize SQL query for pattern matching."""
		# Remove extra whitespace and comments
		sql = re.sub(r'\s+', ' ', sql.strip())
		sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
		sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
		
		# Normalize string literals and numbers
		sql = re.sub(r"'[^']*'", "'?'", sql)
		sql = re.sub(r'\b\d+\b', '?', sql)
		
		# Normalize IN clauses
		sql = re.sub(r'IN\s*\([^)]+\)', 'IN (?)', sql, flags=re.IGNORECASE)
		
		return sql.upper()
	
	def _parse_sql(self, sql: str) -> Statement:
		"""Parse SQL using sqlparse."""
		try:
			parsed = sqlparse.parse(sql)
			return parsed[0] if parsed else None
		except Exception as e:
			logger.warning(f"Failed to parse SQL: {e}")
			return None
	
	def _identify_query_type(self, parsed_query: Statement) -> QueryType:
		"""Identify the type of SQL query."""
		if not parsed_query:
			return QueryType.SELECT
		
		for token in parsed_query.tokens:
			if token.ttype is DML:
				return QueryType(token.value.upper())
		
		# Check for DDL statements
		sql_upper = str(parsed_query).upper().strip()
		if sql_upper.startswith('CREATE'):
			return QueryType.CREATE
		elif sql_upper.startswith('ALTER'):
			return QueryType.ALTER
		elif sql_upper.startswith('DROP'):
			return QueryType.DROP
		
		return QueryType.SELECT
	
	async def _extract_query_pattern(
		self, 
		parsed_query: Statement, 
		query_type: QueryType
	) -> QueryPattern:
		"""Extract query pattern for optimization analysis."""
		if not parsed_query:
			return QueryPattern(
				query_type=query_type,
				tables_involved=[],
				columns_accessed=[],
				where_conditions=[],
				join_conditions=[],
				order_by_columns=[]
			)
		
		# Extract tables, columns, conditions
		tables = self._extract_tables(parsed_query)
		columns = self._extract_columns(parsed_query)
		where_conditions = self._extract_where_conditions(parsed_query)
		joins = self._extract_join_conditions(parsed_query)
		order_by = self._extract_order_by(parsed_query)
		
		return QueryPattern(
			query_type=query_type,
			tables_involved=tables,
			columns_accessed=columns,
			where_conditions=where_conditions,
			join_conditions=joins,
			order_by_columns=order_by
		)
	
	def _extract_tables(self, parsed_query: Statement) -> List[str]:
		"""Extract table names from parsed query."""
		tables = []
		
		def extract_from_token(token):
			if hasattr(token, 'tokens'):
				for subtoken in token.tokens:
					extract_from_token(subtoken)
			elif token.ttype is None and isinstance(token, Identifier):
				tables.append(str(token))
			elif token.ttype is None and isinstance(token, IdentifierList):
				for identifier in token.get_identifiers():
					tables.append(str(identifier))
		
		# Look for FROM and JOIN clauses
		in_from_clause = False
		for token in parsed_query.flatten():
			if token.ttype is Keyword and token.value.upper() in ('FROM', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'):
				in_from_clause = True
			elif token.ttype is Keyword and token.value.upper() in ('WHERE', 'GROUP', 'ORDER', 'HAVING'):
				in_from_clause = False
			elif in_from_clause and token.ttype is None and not token.is_whitespace:
				# Clean table name
				table_name = str(token).strip('(),')
				if table_name and not table_name.upper() in ('ON', 'AS'):
					tables.append(table_name)
		
		return list(set(tables))  # Remove duplicates
	
	def _extract_columns(self, parsed_query: Statement) -> List[str]:
		"""Extract column names from parsed query."""
		columns = []
		sql_str = str(parsed_query).upper()
		
		# Simple extraction for SELECT columns
		if 'SELECT' in sql_str:
			select_part = sql_str.split('FROM')[0].replace('SELECT', '').strip()
			if select_part != '*':
				# Basic column extraction
				column_parts = [c.strip() for c in select_part.split(',')]
				for part in column_parts:
					# Remove aliases and functions
					col = part.split(' AS ')[0].split(' ')[0]
					if '.' in col:
						col = col.split('.')[1]  # Remove table prefix
					columns.append(col)
		
		return columns
	
	def _extract_where_conditions(self, parsed_query: Statement) -> List[str]:
		"""Extract WHERE conditions from parsed query."""
		conditions = []
		sql_str = str(parsed_query).upper()
		
		if 'WHERE' in sql_str:
			where_part = sql_str.split('WHERE')[1]
			# Split on logical operators
			where_part = where_part.split('GROUP BY')[0].split('ORDER BY')[0].split('HAVING')[0]
			
			# Simple condition extraction
			condition_parts = re.split(r'\s+AND\s+|\s+OR\s+', where_part)
			for condition in condition_parts:
				condition = condition.strip()
				if condition:
					conditions.append(condition)
		
		return conditions
	
	def _extract_join_conditions(self, parsed_query: Statement) -> List[str]:
		"""Extract JOIN conditions from parsed query."""
		joins = []
		sql_str = str(parsed_query).upper()
		
		# Find JOIN clauses
		join_pattern = r'(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN)\s+(\w+)\s+ON\s+([^WHERE|GROUP|ORDER|HAVING]+)'
		matches = re.findall(join_pattern, sql_str)
		
		for match in matches:
			join_type, table, condition = match
			joins.append(f"{join_type} {table} ON {condition.strip()}")
		
		return joins
	
	def _extract_order_by(self, parsed_query: Statement) -> List[str]:
		"""Extract ORDER BY columns from parsed query."""
		order_by_columns = []
		sql_str = str(parsed_query).upper()
		
		if 'ORDER BY' in sql_str:
			order_part = sql_str.split('ORDER BY')[1]
			order_part = order_part.split('LIMIT')[0].split('OFFSET')[0].strip()
			
			# Extract columns
			columns = [c.strip() for c in order_part.split(',')]
			for col in columns:
				# Remove ASC/DESC
				col_clean = col.replace(' ASC', '').replace(' DESC', '').strip()
				if '.' in col_clean:
					col_clean = col_clean.split('.')[1]  # Remove table prefix
				order_by_columns.append(col_clean)
		
		return order_by_columns
	
	async def _get_execution_plan(self, sql: str) -> Optional[QueryPlan]:
		"""Get query execution plan from PostgreSQL."""
		try:
			async with self.db_pool.acquire() as conn:
				# Get JSON execution plan
				explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {sql}"
				result = await conn.fetchval(explain_sql)
				
				if result and len(result) > 0:
					plan_data = result[0]['Plan']
					
					return QueryPlan(
						query_hash=self._hash_query(sql),
						plan_json=plan_data,
						total_cost=plan_data.get('Total Cost', 0),
						execution_time_ms=result[0].get('Execution Time', 0),
						rows_returned=plan_data.get('Actual Rows', 0),
						index_usage=self._extract_index_usage(plan_data),
						table_scans=self._extract_table_scans(plan_data),
						join_methods=self._extract_join_methods(plan_data)
					)
		
		except Exception as e:
			logger.warning(f"Could not get execution plan: {e}")
			return None
	
	def _extract_index_usage(self, plan_data: Dict) -> List[str]:
		"""Extract index usage from execution plan."""
		indexes = []
		
		def traverse_plan(node):
			node_type = node.get('Node Type', '')
			if 'Index' in node_type:
				index_name = node.get('Index Name', '')
				if index_name:
					indexes.append(index_name)
			
			# Traverse child plans
			plans = node.get('Plans', [])
			for child_plan in plans:
				traverse_plan(child_plan)
		
		traverse_plan(plan_data)
		return indexes
	
	def _extract_table_scans(self, plan_data: Dict) -> List[str]:
		"""Extract table scans from execution plan."""
		scans = []
		
		def traverse_plan(node):
			node_type = node.get('Node Type', '')
			if node_type == 'Seq Scan':
				table_name = node.get('Relation Name', '')
				if table_name:
					scans.append(table_name)
			
			# Traverse child plans
			plans = node.get('Plans', [])
			for child_plan in plans:
				traverse_plan(child_plan)
		
		traverse_plan(plan_data)
		return scans
	
	def _extract_join_methods(self, plan_data: Dict) -> List[str]:
		"""Extract join methods from execution plan."""
		joins = []
		
		def traverse_plan(node):
			node_type = node.get('Node Type', '')
			if 'Join' in node_type:
				joins.append(node_type)
			
			# Traverse child plans
			plans = node.get('Plans', [])
			for child_plan in plans:
				traverse_plan(child_plan)
		
		traverse_plan(plan_data)
		return joins
	
	async def _update_query_stats(
		self,
		query_hash: str,
		execution_time_ms: Optional[float],
		rows_returned: Optional[int]
	) -> None:
		"""Update query performance statistics."""
		if query_hash not in self.query_stats:
			self.query_stats[query_hash] = QueryStats(
				query_hash=query_hash,
				execution_count=0,
				total_time_ms=0.0,
				average_time_ms=0.0,
				min_time_ms=float('inf'),
				max_time_ms=0.0,
				rows_examined=0,
				rows_returned=0,
				cache_hits=0,
				last_executed=datetime.now()
			)
		
		stats = self.query_stats[query_hash]
		stats.execution_count += 1
		stats.last_executed = datetime.now()
		
		if execution_time_ms is not None:
			stats.total_time_ms += execution_time_ms
			stats.average_time_ms = stats.total_time_ms / stats.execution_count
			stats.min_time_ms = min(stats.min_time_ms, execution_time_ms)
			stats.max_time_ms = max(stats.max_time_ms, execution_time_ms)
		
		if rows_returned is not None:
			stats.rows_returned += rows_returned
	
	async def _identify_optimization_opportunities(
		self,
		query_hash: str,
		sql: str,
		plan: Optional[QueryPlan]
	) -> None:
		"""Identify optimization opportunities for slow queries."""
		opportunities = []
		
		if not plan:
			return
		
		# Check for table scans
		if plan.table_scans:
			opportunities.append("Consider adding indexes for table scans")
			await self._suggest_indexes_for_scans(sql, plan.table_scans)
		
		# Check for expensive joins
		if plan.join_methods and plan.total_cost > 1000:
			opportunities.append("Consider optimizing join conditions or adding indexes")
		
		# Check for high cost queries
		if plan.total_cost > 10000:
			opportunities.append("Query has high execution cost - consider rewriting")
		
		# Check for large result sets
		if plan.rows_returned > 10000:
			opportunities.append("Consider adding LIMIT or improving WHERE conditions")
		
		plan.optimization_opportunities.extend(opportunities)
		
		# Generate specific recommendations
		await self._generate_optimization_recommendations(query_hash, sql, plan)
	
	async def _suggest_indexes_for_scans(self, sql: str, table_scans: List[str]) -> None:
		"""Suggest indexes for table scans."""
		parsed_query = self._parse_sql(sql)
		pattern = await self._extract_query_pattern(parsed_query, QueryType.SELECT)
		
		for table in table_scans:
			# Find columns used in WHERE conditions for this table
			relevant_columns = []
			
			for condition in pattern.where_conditions:
				# Simple column extraction from conditions
				condition_parts = condition.split()
				for part in condition_parts:
					if '.' in part and part.startswith(table + '.'):
						column = part.split('.')[1]
						relevant_columns.append(column)
					elif not '.' in part and part not in ('AND', 'OR', '=', '>', '<', 'IN', 'LIKE'):
						# Assume it's a column if no table prefix
						relevant_columns.append(part)
			
			# Add ORDER BY columns
			relevant_columns.extend(pattern.order_by_columns)
			
			if relevant_columns:
				recommendation = IndexRecommendation(
					table_name=table,
					columns=list(set(relevant_columns)),
					estimated_benefit=0.7,  # Rough estimate
					estimated_cost=0.1,
					queries_benefited=[self._hash_query(sql)],
					priority="high"
				)
				
				# Check if recommendation already exists
				existing = next(
					(r for r in self.index_recommendations 
					 if r.table_name == table and set(r.columns) == set(relevant_columns)),
					None
				)
				
				if not existing:
					self.index_recommendations.append(recommendation)
					logger.info(f"Generated index recommendation for {table}({', '.join(relevant_columns)})")
	
	async def _generate_optimization_recommendations(
		self,
		query_hash: str,
		sql: str,
		plan: QueryPlan
	) -> None:
		"""Generate specific optimization recommendations."""
		# This could be expanded with more sophisticated analysis
		if plan.total_cost > 1000 and len(plan.table_scans) > 0:
			await self._suggest_indexes_for_scans(sql, plan.table_scans)
	
	async def _apply_query_optimizations(
		self,
		sql: str,
		pattern: QueryPattern
	) -> str:
		"""Apply real-time query optimizations."""
		if not self.enable_query_rewriting:
			return sql
		
		optimized_sql = sql
		
		# Apply various optimization techniques
		optimized_sql = self._optimize_subqueries(optimized_sql)
		optimized_sql = self._optimize_joins(optimized_sql, pattern)
		optimized_sql = await self._add_query_hints(optimized_sql, pattern)
		
		return optimized_sql
	
	def _optimize_subqueries(self, sql: str) -> str:
		"""Optimize subqueries to use JOINs where beneficial."""
		# Convert EXISTS subqueries to JOINs
		sql = re.sub(
			r'EXISTS\s*\(\s*SELECT\s+1\s+FROM\s+(\w+)\s+WHERE\s+([^)]+)\)',
			r'INNER JOIN \1 ON \2',
			sql,
			flags=re.IGNORECASE
		)
		
		return sql
	
	def _optimize_joins(self, sql: str, pattern: QueryPattern) -> str:
		"""Optimize JOIN operations."""
		# This is a simplified example - real optimization would be much more complex
		if len(pattern.tables_involved) > 3:
			# For complex queries, suggest using materialized views or temp tables
			pass
		
		return sql
	
	async def _add_query_hints(self, sql: str, pattern: QueryPattern) -> str:
		"""Add PostgreSQL query hints for optimization."""
		hints = []
		
		# Add parallel execution hint for large table scans
		if len(pattern.tables_involved) > 1:
			hints.append("/*+ Parallel(4) */")
		
		# Add index hints if we have recommendations
		for recommendation in self.index_recommendations:
			if recommendation.table_name in pattern.tables_involved:
				hints.append(f"/*+ IndexScan({recommendation.table_name}) */")
		
		if hints:
			hint_comment = " ".join(hints)
			sql = f"{hint_comment}\n{sql}"
		
		return sql
	
	async def apply_index_recommendations(
		self,
		max_recommendations: int = 5,
		min_priority: str = "medium"
	) -> List[str]:
		"""Apply index recommendations."""
		if not self.enable_auto_indexing:
			return []
		
		# Sort recommendations by priority and benefit
		priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
		
		sorted_recommendations = sorted(
			self.index_recommendations,
			key=lambda r: (priority_order.get(r.priority, 0), r.estimated_benefit),
			reverse=True
		)
		
		applied_indexes = []
		min_priority_value = priority_order.get(min_priority, 0)
		
		for recommendation in sorted_recommendations[:max_recommendations]:
			if priority_order.get(recommendation.priority, 0) >= min_priority_value:
				try:
					index_sql = await self._generate_index_sql(recommendation)
					
					async with self.db_pool.acquire() as conn:
						await conn.execute(index_sql)
						applied_indexes.append(index_sql)
						logger.info(f"Applied index: {index_sql}")
				
				except Exception as e:
					logger.error(f"Failed to create index: {e}")
		
		return applied_indexes
	
	async def _generate_index_sql(self, recommendation: IndexRecommendation) -> str:
		"""Generate SQL for creating recommended index."""
		columns_str = ", ".join(recommendation.columns)
		index_name = f"idx_{recommendation.table_name}_{'_'.join(recommendation.columns)}"
		
		# Ensure index name is not too long (PostgreSQL limit is 63 characters)
		if len(index_name) > 60:
			index_name = f"idx_{recommendation.table_name}_{recommendation.recommendation_id[:8]}"
		
		sql = f"CREATE INDEX CONCURRENTLY {index_name} ON {recommendation.table_name} ({columns_str})"
		
		# Add index type if not default
		if recommendation.index_type != "btree":
			sql = sql.replace(f"ON {recommendation.table_name}", 
							  f"ON {recommendation.table_name} USING {recommendation.index_type}")
		
		return sql
	
	async def analyze_performance_trends(self) -> Dict[str, Any]:
		"""Analyze query performance trends."""
		if not self.query_stats:
			return {"status": "No query statistics available"}
		
		# Find slowest queries
		slowest_queries = sorted(
			self.query_stats.values(),
			key=lambda s: s.average_time_ms,
			reverse=True
		)[:10]
		
		# Find most frequent queries
		most_frequent = sorted(
			self.query_stats.values(),
			key=lambda s: s.execution_count,
			reverse=True
		)[:10]
		
		# Calculate overall statistics
		total_queries = sum(s.execution_count for s in self.query_stats.values())
		total_time = sum(s.total_time_ms for s in self.query_stats.values())
		avg_time = total_time / total_queries if total_queries > 0 else 0
		
		return {
			"overall_statistics": {
				"total_unique_queries": len(self.query_stats),
				"total_executions": total_queries,
				"total_time_ms": total_time,
				"average_time_ms": avg_time
			},
			"slowest_queries": [
				{
					"query_hash": s.query_hash,
					"average_time_ms": s.average_time_ms,
					"execution_count": s.execution_count,
					"total_time_ms": s.total_time_ms
				}
				for s in slowest_queries
			],
			"most_frequent_queries": [
				{
					"query_hash": s.query_hash,
					"execution_count": s.execution_count,
					"average_time_ms": s.average_time_ms
				}
				for s in most_frequent
			],
			"index_recommendations": [
				{
					"table_name": r.table_name,
					"columns": r.columns,
					"priority": r.priority,
					"estimated_benefit": r.estimated_benefit
				}
				for r in self.index_recommendations
			]
		}
	
	async def get_optimization_report(self) -> Dict[str, Any]:
		"""Generate comprehensive optimization report."""
		performance_analysis = await self.analyze_performance_trends()
		
		# Analyze query patterns
		pattern_analysis = self._analyze_query_patterns()
		
		# Generate recommendations
		recommendations = await self._generate_comprehensive_recommendations()
		
		return {
			"performance_analysis": performance_analysis,
			"pattern_analysis": pattern_analysis,
			"optimization_recommendations": recommendations,
			"system_status": {
				"auto_indexing_enabled": self.enable_auto_indexing,
				"query_rewriting_enabled": self.enable_query_rewriting,
				"slow_query_threshold_ms": self.slow_query_threshold_ms,
				"last_optimization": self.last_optimization.isoformat()
			}
		}
	
	def _analyze_query_patterns(self) -> Dict[str, Any]:
		"""Analyze query patterns for optimization insights."""
		if not self.query_patterns:
			return {"status": "No query patterns available"}
		
		# Analyze table access patterns
		table_access = {}
		for pattern in self.query_patterns.values():
			for table in pattern.tables_involved:
				if table not in table_access:
					table_access[table] = {
						"access_count": 0,
						"query_types": set(),
						"columns_accessed": set()
					}
				table_access[table]["access_count"] += pattern.frequency
				table_access[table]["query_types"].add(pattern.query_type.value)
				table_access[table]["columns_accessed"].update(pattern.columns_accessed)
		
		# Convert sets to lists for JSON serialization
		for table_data in table_access.values():
			table_data["query_types"] = list(table_data["query_types"])
			table_data["columns_accessed"] = list(table_data["columns_accessed"])
		
		return {
			"table_access_patterns": table_access,
			"total_patterns": len(self.query_patterns),
			"query_type_distribution": self._get_query_type_distribution()
		}
	
	def _get_query_type_distribution(self) -> Dict[str, int]:
		"""Get distribution of query types."""
		distribution = {}
		for pattern in self.query_patterns.values():
			query_type = pattern.query_type.value
			distribution[query_type] = distribution.get(query_type, 0) + pattern.frequency
		return distribution
	
	async def _generate_comprehensive_recommendations(self) -> List[Dict[str, Any]]:
		"""Generate comprehensive optimization recommendations."""
		recommendations = []
		
		# Index recommendations
		for rec in self.index_recommendations:
			recommendations.append({
				"type": "index_creation",
				"priority": rec.priority,
				"description": f"Create index on {rec.table_name}({', '.join(rec.columns)})",
				"estimated_benefit": rec.estimated_benefit,
				"sql": await self._generate_index_sql(rec)
			})
		
		# Query rewriting recommendations
		slow_queries = [
			s for s in self.query_stats.values() 
			if s.average_time_ms > self.slow_query_threshold_ms
		]
		
		for query_stat in slow_queries[:5]:  # Top 5 slow queries
			recommendations.append({
				"type": "query_optimization",
				"priority": "high" if query_stat.average_time_ms > 5000 else "medium",
				"description": f"Optimize slow query (avg: {query_stat.average_time_ms:.2f}ms)",
				"query_hash": query_stat.query_hash,
				"execution_count": query_stat.execution_count
			})
		
		# Partition recommendations for large tables
		if hasattr(self, 'large_tables'):  # This would be populated by table analysis
			for table in getattr(self, 'large_tables', []):
				recommendations.append({
					"type": "partitioning",
					"priority": "medium",
					"description": f"Consider partitioning large table {table}",
					"table_name": table
				})
		
		return recommendations
	
	async def run_optimization_cycle(self) -> Dict[str, Any]:
		"""Run a complete optimization cycle."""
		logger.info("Starting query optimization cycle")
		
		try:
			# Apply index recommendations
			applied_indexes = await self.apply_index_recommendations()
			
			# Update last optimization time
			self.last_optimization = datetime.now()
			
			# Generate report
			report = await self.get_optimization_report()
			
			result = {
				"success": True,
				"applied_optimizations": len(applied_indexes),
				"index_sql": applied_indexes,
				"optimization_time": self.last_optimization.isoformat(),
				"report": report
			}
			
			logger.info(f"Optimization cycle completed: {len(applied_indexes)} indexes applied")
			return result
			
		except Exception as e:
			logger.error(f"Optimization cycle failed: {e}")
			return {
				"success": False,
				"error": str(e),
				"optimization_time": datetime.now().isoformat()
			}

# Query optimization decorator
def optimize_query(optimizer: QueryOptimizer):
	"""Decorator to automatically optimize queries."""
	def decorator(func):
		async def wrapper(*args, **kwargs):
			# Extract SQL from function or arguments
			sql = None
			if hasattr(func, '__sql__'):
				sql = func.__sql__
			elif args and isinstance(args[0], str):
				sql = args[0]
			
			if sql:
				start_time = time.time()
				
				# Optimize query
				optimized_sql = await optimizer.analyze_query(sql)
				
				# Replace original SQL with optimized version
				if isinstance(args[0], str):
					args = (optimized_sql,) + args[1:]
				
				# Execute function
				result = await func(*args, **kwargs)
				
				# Record execution time
				execution_time = (time.time() - start_time) * 1000
				await optimizer.analyze_query(
					optimized_sql, 
					execution_time_ms=execution_time,
					rows_returned=len(result) if isinstance(result, list) else 1
				)
				
				return result
			else:
				return await func(*args, **kwargs)
		
		return wrapper
	return decorator

if __name__ == "__main__":
	async def main():
		# Example usage would require a real database connection
		print("Query optimizer initialized")
		print("This module provides advanced SQL query optimization capabilities")
	
	asyncio.run(main())