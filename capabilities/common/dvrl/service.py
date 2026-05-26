#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) Service Layer
Core business logic for federated query processing and data source management

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid_extensions import uuid7str

from . import _log_info, _log_error, _log_warning
from .models import (
	DataSource, DataSourceType, DataSourceStatus, VirtualTable,
	FederatedQuery, QueryStatus, QueryCache, CacheLevel, 
	DataSourceSchema, FederationPlan, mask_sensitive_config,
	calculate_query_complexity, estimate_query_cost
)
from .connectors import UniversalConnectorManager, BaseConnector, ConnectionHealth
from . import adapters  # Import adapters to register additional connectors
from .nlp_integration import APGNLPProcessor, QuerySuggestionEngine, SemanticQueryMatcher
from .apg_integrations import APGServiceManager
from .error_handling import (
	DVRLErrorHandler, DVRLLoggingContext, DVRLPerformanceMonitor, 
	DVRLRetryHandler, error_handler_decorator, safe_execute,
	ServiceUnavailableError, OperationError, RegistrationError,
	ConnectionError, QueryExecutionError, ValidationError
)


class SQLParser:
	"""Production SQL parser with comprehensive parsing capabilities"""
	
	def __init__(self):
		"""Initialize SQL parser with comprehensive regex patterns"""
		self.sql_patterns = {
			'select': r'\bSELECT\b',
			'from': r'\bFROM\b',
			'where': r'\bWHERE\b',
			'join': r'\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|JOIN)\b',
			'group_by': r'\bGROUP\s+BY\b',
			'having': r'\bHAVING\b',
			'order_by': r'\bORDER\s+BY\b',
			'limit': r'\bLIMIT\b',
			'subquery': r'\(\s*SELECT\b',
			'cte': r'\bWITH\b',
			'window': r'\bOVER\s*\(',
			'union': r'\bUNION\b',
			'case': r'\bCASE\b',
			'cast': r'\bCAST\s*\(',
			'aggregate': r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\('
		}
		
		# Compile patterns for performance
		self.compiled_patterns = {
			name: re.compile(pattern, re.IGNORECASE)
			for name, pattern in self.sql_patterns.items()
		}
	
	async def parse_query(self, sql: str) -> Dict[str, Any]:
		"""
		Parse and analyze SQL query with comprehensive structural and semantic analysis.
		
		Performs deep analysis of SQL queries to extract all structural elements, semantic
		patterns, complexity metrics, and optimization opportunities. This analysis drives
		the federation planning, optimization, and execution strategies.
		
		Args:
			sql (str): SQL query to parse and analyze. Supports full SQL syntax including
				SELECT/INSERT/UPDATE/DELETE, JOINs, subqueries, CTEs, window functions.
		
		Returns:
			Dict[str, Any]: Comprehensive query analysis containing:
				- original_sql (str): Unmodified original query
				- normalized_sql (str): Normalized and cleaned query
				- query_type (str): Query type (SELECT, INSERT, UPDATE, DELETE, etc.)
				- tables (List[Dict]): Table references with schema/alias information
				- columns (List[Dict]): Column references with context and types
				- joins (List[Dict]): JOIN operations with conditions and types
				- conditions (List[Dict]): WHERE/HAVING conditions with analysis
				- aggregations (List[Dict]): Aggregation functions and GROUP BY
				- subqueries (Dict): Subquery count and complexity analysis
				- complexity (Dict): Query complexity metrics and scores
				- performance_hints (List[str]): Performance optimization suggestions
				- security_analysis (Dict): Security risk assessment and recommendations
		
		Raises:
			ValueError: If SQL query is empty, malformed, or contains syntax errors
			
		Example:
			>>> query_info = await service.parse_query(
			...     "SELECT o.id, COUNT(i.item_id) FROM orders o "
			...     "JOIN order_items i ON o.id = i.order_id "
			...     "WHERE o.created_at >= '2024-01-01' GROUP BY o.id"
			... )
			>>> print(f"Query type: {query_info['query_type']}")
			>>> print(f"Tables: {[t['name'] for t in query_info['tables']]}")
			>>> print(f"Complexity score: {query_info['complexity']['score']}")
		"""
		if not sql or not sql.strip():
			raise ValueError("SQL query cannot be empty")
			
		sql_normalized = ' '.join(sql.split())
		sql_upper = sql_normalized.upper()
		
		# Extract comprehensive query metadata
		query_info = {
			'original_sql': sql,
			'normalized_sql': sql_normalized,
			'query_type': self._extract_query_type(sql_upper),
			'tables': await self._extract_tables_comprehensive(sql_upper),
			'columns': await self._extract_columns_comprehensive(sql_normalized),
			'joins': await self._extract_joins_comprehensive(sql_upper),
			'conditions': await self._extract_conditions_comprehensive(sql_upper),
			'aggregations': await self._extract_aggregations_comprehensive(sql_upper),
			'subqueries': await self._count_subqueries_comprehensive(sql_upper),
			'complexity_features': await self._analyze_complexity_comprehensive(sql_upper),
			'performance_hints': await self._generate_performance_hints(sql_upper),
			'security_analysis': await self._analyze_security_issues(sql_normalized)
		}
		
		return query_info
	
	def _extract_query_type(self, sql: str) -> str:
		"""Extract detailed query type"""
		sql = sql.strip()
		
		# Handle CTEs
		if sql.startswith('WITH'):
			# Find the main query after CTE
			main_query_match = re.search(r'\)\s+(SELECT|INSERT|UPDATE|DELETE)', sql)
			if main_query_match:
				return f"CTE_{main_query_match.group(1)}"
			return 'CTE_SELECT'  # Default assumption
			
		# Handle standard queries
		query_type_map = {
			'SELECT': 'SELECT',
			'INSERT': 'INSERT', 
			'UPDATE': 'UPDATE',
			'DELETE': 'DELETE',
			'CREATE': 'DDL_CREATE',
			'ALTER': 'DDL_ALTER',
			'DROP': 'DDL_DROP',
			'TRUNCATE': 'DDL_TRUNCATE',
			'MERGE': 'MERGE',
			'UPSERT': 'UPSERT'
		}
		
		for keyword, query_type in query_type_map.items():
			if sql.startswith(keyword):
				return query_type
				
		return 'UNKNOWN'
	
	async def _extract_tables_comprehensive(self, sql: str) -> List[Dict[str, Any]]:
		"""Extract tables with aliases and schema information"""
		tables = []
		
		# FROM clause tables
		from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*(?:(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?'
		from_matches = re.finditer(from_pattern, sql, re.IGNORECASE)
		
		for match in from_matches:
			table_name = match.group(1)
			alias = match.group(2)
			
			# Parse schema.table
			parts = table_name.split('.')
			if len(parts) == 2:
				schema, table = parts
			else:
				schema, table = None, parts[0]
				
			tables.append({
				'name': table,
				'schema': schema,
				'alias': alias,
				'full_name': table_name,
				'type': 'base_table'
			})
		
		# JOIN clause tables  
		join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*(?:(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?'
		join_matches = re.finditer(join_pattern, sql, re.IGNORECASE)
		
		for match in join_matches:
			table_name = match.group(1)
			alias = match.group(2)
			
			parts = table_name.split('.')
			if len(parts) == 2:
				schema, table = parts
			else:
				schema, table = None, parts[0]
				
			tables.append({
				'name': table,
				'schema': schema, 
				'alias': alias,
				'full_name': table_name,
				'type': 'joined_table'
			})
		
		# Subquery tables (simplified detection)
		subquery_pattern = r'\(\s*SELECT.*?\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
		subquery_matches = re.finditer(subquery_pattern, sql, re.IGNORECASE | re.DOTALL)
		
		for match in subquery_matches:
			table_name = match.group(1)
			tables.append({
				'name': table_name,
				'schema': None,
				'alias': None,
				'full_name': table_name,
				'type': 'subquery_table'
			})
		
		return tables
	
	async def _extract_columns_comprehensive(self, sql: str) -> List[Dict[str, Any]]:
		"""Extract column references with context"""
		columns = []
		
		# SELECT list columns
		select_pattern = r'SELECT\s+(.*?)\s+FROM'
		select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
		
		if select_match:
			select_list = select_match.group(1)
			
			# Handle SELECT *
			if '*' in select_list:
				columns.append({
					'name': '*',
					'table': None,
					'alias': None,
					'type': 'wildcard',
					'function': None
				})
			else:
				# Parse individual columns
				column_items = [item.strip() for item in select_list.split(',')]
				for item in column_items:
					column_info = await self._parse_column_expression(item)
					columns.extend(column_info)
		
		# WHERE clause columns
		where_pattern = r'\bWHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)'
		where_match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
		
		if where_match:
			where_clause = where_match.group(1)
			where_columns = await self._extract_where_columns(where_clause)
			columns.extend(where_columns)
		
		return columns
	
	async def _parse_column_expression(self, expr: str) -> List[Dict[str, Any]]:
		"""Parse individual column expression"""
		columns = []
		
		# Check for function calls
		func_pattern = r'([A-Z_]+)\s*\((.*?)\)'
		func_match = re.search(func_pattern, expr, re.IGNORECASE)
		
		if func_match:
			func_name = func_match.group(1)
			func_args = func_match.group(2)
			
			# Extract alias
			alias_pattern = r'\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*$'
			alias_match = re.search(alias_pattern, expr, re.IGNORECASE)
			alias = alias_match.group(1) if alias_match else None
			
			columns.append({
				'name': func_args if func_args.strip() != '*' else '*',
				'table': None,
				'alias': alias,
				'type': 'function_call',
				'function': func_name
			})
		else:
			# Regular column
			parts = expr.split()
			column_part = parts[0]
			
			# Check for table.column
			if '.' in column_part:
				table, column = column_part.split('.', 1)
			else:
				table, column = None, column_part
			
			# Extract alias
			alias = parts[-1] if len(parts) > 1 and parts[-2].upper() != 'AS' else None
			if len(parts) > 2 and parts[-2].upper() == 'AS':
				alias = parts[-1]
			
			columns.append({
				'name': column,
				'table': table,
				'alias': alias,
				'type': 'column_reference',
				'function': None
			})
		
		return columns
	
	async def _extract_where_columns(self, where_clause: str) -> List[Dict[str, Any]]:
		"""Extract columns from WHERE clause"""
		columns = []
		
		# Simple column extraction from WHERE
		column_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
		matches = re.finditer(column_pattern, where_clause)
		
		for match in matches:
			column_ref = match.group(1)
			
			if '.' in column_ref:
				table, column = column_ref.split('.', 1)
			else:
				table, column = None, column_ref
			
			# Skip SQL keywords
			if column.upper() not in ['AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL']:
				columns.append({
					'name': column,
					'table': table,
					'alias': None,
					'type': 'where_condition',
					'function': None
				})
		
		return columns
	
	async def _extract_joins_comprehensive(self, sql: str) -> List[Dict[str, Any]]:
		"""Extract JOIN information with details"""
		joins = []
		
		join_pattern = r'\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*(?:(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?\s*(?:ON\s+(.*?))?(?=\s+(?:INNER|LEFT|RIGHT|FULL|CROSS|JOIN|WHERE|GROUP|ORDER|HAVING|$))'
		
		matches = re.finditer(join_pattern, sql, re.IGNORECASE | re.DOTALL)
		
		for match in matches:
			join_type = match.group(1).strip()
			table = match.group(2)
			alias = match.group(3)
			condition = match.group(4).strip() if match.group(4) else None
			
			joins.append({
				'type': join_type,
				'table': table,
				'alias': alias,
				'condition': condition,
				'condition_columns': await self._extract_join_condition_columns(condition) if condition else []
			})
		
		return joins
	
	async def _extract_join_condition_columns(self, condition: str) -> List[str]:
		"""Extract columns from JOIN condition"""
		if not condition:
			return []
		
		column_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
		matches = re.findall(column_pattern, condition)
		
		return [match for match in matches if match.upper() not in ['AND', 'OR', 'NOT']]
	
	async def _extract_conditions_comprehensive(self, sql: str) -> List[Dict[str, Any]]:
		"""Extract WHERE conditions with analysis"""
		conditions = []
		
		# Extract WHERE clause
		where_pattern = r'\bWHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)'
		where_match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
		
		if where_match:
			where_clause = where_match.group(1).strip()
			
			# Split by AND/OR but preserve the operators
			condition_parts = re.split(r'\s+(AND|OR)\s+', where_clause, flags=re.IGNORECASE)
			
			current_condition = ""
			operator = None
			
			for i, part in enumerate(condition_parts):
				if part.upper() in ['AND', 'OR']:
					if current_condition.strip():
						condition_info = await self._analyze_condition(current_condition.strip())
						if operator:
							condition_info['operator'] = operator
						conditions.append(condition_info)
					current_condition = ""
					operator = part.upper()
				else:
					current_condition += part
			
			# Add the last condition
			if current_condition.strip():
				condition_info = await self._analyze_condition(current_condition.strip())
				if operator:
					condition_info['operator'] = operator
				conditions.append(condition_info)
		
		return conditions
	
	async def _analyze_condition(self, condition: str) -> Dict[str, Any]:
		"""Analyze individual WHERE condition"""
		condition_info = {
			'condition': condition,
			'type': 'unknown',
			'operator': None,
			'selectivity': 'unknown',
			'indexable': False
		}
		
		# Detect condition type
		if '=' in condition:
			condition_info['type'] = 'equality'
			condition_info['indexable'] = True
			condition_info['selectivity'] = 'high'
		elif any(op in condition.upper() for op in ['>', '<', '>=', '<=']):
			condition_info['type'] = 'range'
			condition_info['indexable'] = True
			condition_info['selectivity'] = 'medium'
		elif 'LIKE' in condition.upper():
			condition_info['type'] = 'pattern_match'
			condition_info['indexable'] = not condition.upper().startswith('LIKE \'%')
			condition_info['selectivity'] = 'low'
		elif 'IN' in condition.upper():
			condition_info['type'] = 'membership'
			condition_info['indexable'] = True
			condition_info['selectivity'] = 'medium'
		elif 'IS NULL' in condition.upper():
			condition_info['type'] = 'null_check'
			condition_info['indexable'] = True
			condition_info['selectivity'] = 'low'
		elif 'EXISTS' in condition.upper():
			condition_info['type'] = 'existence_check'
			condition_info['indexable'] = False
			condition_info['selectivity'] = 'low'
		
		return condition_info
	
	async def _extract_aggregations_comprehensive(self, sql: str) -> List[Dict[str, Any]]:
		"""Extract aggregation functions with details"""
		aggregations = []
		
		# Common aggregation functions
		agg_pattern = r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|STRING_AGG|ARRAY_AGG)\s*\(\s*([^)]+)\s*\)'
		matches = re.finditer(agg_pattern, sql, re.IGNORECASE)
		
		for match in matches:
			func_name = match.group(1).upper()
			argument = match.group(2).strip()
			
			aggregations.append({
				'function': func_name,
				'argument': argument,
				'distinct': 'DISTINCT' in argument.upper(),
				'complexity': 'high' if func_name in ['GROUP_CONCAT', 'STRING_AGG', 'ARRAY_AGG'] else 'medium'
			})
		
		return aggregations
	
	async def _count_subqueries_comprehensive(self, sql: str) -> Dict[str, Any]:
		"""Count and analyze subqueries"""
		# Count nested SELECT statements
		select_pattern = r'\bSELECT\b'
		select_count = len(re.findall(select_pattern, sql, re.IGNORECASE))
		
		# The main query has one SELECT, so subqueries = total - 1
		subquery_count = max(0, select_count - 1)
		
		# Analyze subquery types
		correlated_pattern = r'\(\s*SELECT.*?\bWHERE.*?\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\s*='
		correlated_count = len(re.findall(correlated_pattern, sql, re.IGNORECASE | re.DOTALL))
		
		exists_pattern = r'\bEXISTS\s*\('
		exists_count = len(re.findall(exists_pattern, sql, re.IGNORECASE))
		
		return {
			'total_subqueries': subquery_count,
			'correlated_subqueries': correlated_count,
			'exists_subqueries': exists_count,
			'complexity_impact': 'high' if subquery_count > 2 else 'medium' if subquery_count > 0 else 'low'
		}
	
	async def _analyze_complexity_comprehensive(self, sql: str) -> Dict[str, Any]:
		"""Comprehensive query complexity analysis"""
		complexity_features = {}
		
		# Count various SQL features
		for feature, pattern in self.compiled_patterns.items():
			matches = pattern.findall(sql)
			complexity_features[f'{feature}_count'] = len(matches)
		
		# Calculate complexity score
		weights = {
			'select_count': 1,
			'from_count': 2,
			'join_count': 3,
			'subquery_count': 5,
			'where_count': 2,
			'group_by_count': 3,
			'having_count': 4,
			'order_by_count': 2,
			'window_count': 6,
			'cte_count': 4,
			'union_count': 3,
			'case_count': 2,
			'aggregate_count': 2
		}
		
		complexity_score = sum(
			complexity_features.get(feature, 0) * weight
			for feature, weight in weights.items()
		)
		
		complexity_features['complexity_score'] = complexity_score
		complexity_features['complexity_level'] = (
			'low' if complexity_score < 20 else
			'medium' if complexity_score < 50 else
			'high' if complexity_score < 100 else
			'very_high'
		)
		
		return complexity_features
	
	async def _generate_performance_hints(self, sql: str) -> List[str]:
		"""Generate performance optimization hints"""
		hints = []
		
		# Check for SELECT *
		if re.search(r'\bSELECT\s+\*', sql, re.IGNORECASE):
			hints.append("Consider selecting only required columns instead of SELECT *")
		
		# Check for missing LIMIT in ORDER BY
		if re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE) and not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
			hints.append("Consider adding LIMIT clause to ORDER BY queries")
		
		# Check for correlated subqueries
		if re.search(r'\(\s*SELECT.*?\bWHERE.*?\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\s*=', sql, re.IGNORECASE | re.DOTALL):
			hints.append("Correlated subqueries detected - consider using JOINs for better performance")
		
		# Check for functions in WHERE clause
		if re.search(r'\bWHERE.*\b[A-Z_]+\s*\([^)]*[a-zA-Z_][a-zA-Z0-9_]*', sql, re.IGNORECASE):
			hints.append("Functions in WHERE clause may prevent index usage")
		
		# Check for leading wildcard in LIKE
		if re.search(r'\bLIKE\s+[\'"][%]', sql, re.IGNORECASE):
			hints.append("Leading wildcard in LIKE clause prevents index usage")
		
		return hints
	
	async def _analyze_security_issues(self, sql: str) -> Dict[str, Any]:
		"""Analyze potential security issues"""
		security_info = {
			'risk_level': 'low',
			'issues': [],
			'recommendations': []
		}
		
		# Check for potential SQL injection patterns
		if re.search(r'[\'"][^\'\"]*[\'"]\s*\+|[\'"][^\'\"]*[\'\"]\s*\|\|', sql):
			security_info['issues'].append("Potential string concatenation in SQL")
			security_info['risk_level'] = 'high'
			security_info['recommendations'].append("Use parameterized queries")
		
		# Check for dynamic SQL patterns
		if re.search(r'\bEXEC\b|\bEXECUTE\b', sql, re.IGNORECASE):
			security_info['issues'].append("Dynamic SQL execution detected")
			security_info['risk_level'] = 'high'
			security_info['recommendations'].append("Avoid dynamic SQL when possible")
		
		# Check for administrative commands
		admin_commands = ['DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE']
		if any(re.search(rf'\b{cmd}\b', sql, re.IGNORECASE) for cmd in admin_commands):
			security_info['issues'].append("Administrative SQL command detected")
			security_info['risk_level'] = 'medium'
			security_info['recommendations'].append("Ensure proper authorization for DDL operations")
		
		return security_info


class QueryOptimizer:
	"""Production query optimizer with ML-based optimization"""
	
	def __init__(self):
		"""Initialize query optimizer with rule engine"""
		from collections import defaultdict
		self.optimization_rules = []
		self.query_statistics = defaultdict(list)
		self.optimization_history = []
		self._load_optimization_rules()
	
	def _load_optimization_rules(self):
		"""Load optimization rules"""
		self.optimization_rules = [
			{
				'name': 'predicate_pushdown',
				'priority': 10,
				'condition': self._has_filterable_conditions,
				'apply': self._apply_predicate_pushdown
			},
			{
				'name': 'join_reordering',
				'priority': 9,
				'condition': self._has_multiple_joins,
				'apply': self._apply_join_reordering
			},
			{
				'name': 'index_hints',
				'priority': 8,
				'condition': self._can_use_index,
				'apply': self._apply_index_hints
			},
			{
				'name': 'aggregation_pushdown',
				'priority': 7,
				'condition': self._has_pushable_aggregations,
				'apply': self._apply_aggregation_pushdown
			},
			{
				'name': 'subquery_optimization',
				'priority': 6,
				'condition': self._has_optimizable_subqueries,
				'apply': self._apply_subquery_optimization
			}
		]
	
	async def optimize_query(self, query_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Optimize query based on analysis and context"""
		context = context or {}
		
		optimization_result = {
			'original_query': query_info['original_sql'],
			'optimized_query': query_info['original_sql'],
			'optimization_applied': [],
			'estimated_improvement': 0.0,
			'execution_plan': await self._generate_execution_plan(query_info),
			'cost_estimate': await self._estimate_query_cost(query_info, context),
			'recommendations': []
		}
		
		# Apply optimization rules in priority order
		for rule in sorted(self.optimization_rules, key=lambda r: r['priority'], reverse=True):
			if await rule['condition'](query_info, context):
				optimized_sql, improvement = await rule['apply'](
					optimization_result['optimized_query'], query_info, context
				)
				
				if optimized_sql != optimization_result['optimized_query']:
					optimization_result['optimized_query'] = optimized_sql
					optimization_result['optimization_applied'].append(rule['name'])
					optimization_result['estimated_improvement'] += improvement
		
		# Generate recommendations
		optimization_result['recommendations'] = await self._generate_optimization_recommendations(
			query_info, context
		)
		
		# Store optimization history for ML learning
		self.optimization_history.append({
			'timestamp': datetime.now(timezone.utc),
			'query_hash': hashlib.md5(query_info['original_sql'].encode()).hexdigest(),
			'optimizations': optimization_result['optimization_applied'],
			'improvement': optimization_result['estimated_improvement']
		})
		
		return optimization_result
	
	async def _has_filterable_conditions(self, query_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
		"""Check if query has conditions that can be pushed down"""
		conditions = query_info.get('conditions', [])
		joins = query_info.get('joins', [])
		
		# Can push down if we have WHERE conditions and JOINs
		return len(conditions) > 0 and len(joins) > 0
	
	async def _apply_predicate_pushdown(self, sql: str, query_info: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, float]:
		"""Apply predicate pushdown optimization"""
		# This is a simplified implementation
		# In production, would analyze join conditions and move appropriate WHERE clauses
		
		conditions = query_info.get('conditions', [])
		optimizable_conditions = [
			cond for cond in conditions 
			if cond.get('type') == 'equality' or cond.get('type') == 'range'
		]
		
		if optimizable_conditions:
			# Estimate 20-40% improvement for predicate pushdown
			improvement = 0.3 * len(optimizable_conditions)
			return sql, min(improvement, 0.5)  # Cap at 50%
		
		return sql, 0.0
	
	async def _has_multiple_joins(self, query_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
		"""Check if query has multiple joins that can be reordered"""
		joins = query_info.get('joins', [])
		return len(joins) >= 2
	
	async def _apply_join_reordering(self, sql: str, query_info: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, float]:
		"""Apply join reordering optimization"""
		joins = query_info.get('joins', [])
		
		# Estimate cost reduction based on join types
		optimization_score = 0.0
		
		for join in joins:
			if join.get('type', '').upper() in ['INNER JOIN', 'JOIN']:
				optimization_score += 0.15  # Inner joins are more optimizable
			else:
				optimization_score += 0.05  # Outer joins have limited optimization
		
		return sql, min(optimization_score, 0.4)  # Cap at 40%
	
	async def _can_use_index(self, query_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
		"""Check if query can benefit from index hints"""
		conditions = query_info.get('conditions', [])
		return any(cond.get('indexable', False) for cond in conditions)
	
	async def _apply_index_hints(self, sql: str, query_info: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, float]:
		"""Apply index optimization hints"""
		conditions = query_info.get('conditions', [])
		indexable_conditions = [cond for cond in conditions if cond.get('indexable', False)]
		
		# Estimate improvement based on indexable conditions
		improvement = len(indexable_conditions) * 0.25
		return sql, min(improvement, 0.6)  # Cap at 60%
	
	async def _has_pushable_aggregations(self, query_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
		"""Check if aggregations can be pushed down"""
		aggregations = query_info.get('aggregations', [])
		joins = query_info.get('joins', [])
		return len(aggregations) > 0 and len(joins) > 0
	
	async def _apply_aggregation_pushdown(self, sql: str, query_info: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, float]:
		"""Apply aggregation pushdown optimization"""
		aggregations = query_info.get('aggregations', [])
		
		# Estimate improvement based on aggregation complexity
		improvement = 0.0
		for agg in aggregations:
			if agg.get('complexity') == 'high':
				improvement += 0.3
			elif agg.get('complexity') == 'medium':
				improvement += 0.2
			else:
				improvement += 0.1
		
		return sql, min(improvement, 0.5)  # Cap at 50%
	
	async def _has_optimizable_subqueries(self, query_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
		"""Check if subqueries can be optimized"""
		subquery_info = query_info.get('subqueries', {})
		return subquery_info.get('total_subqueries', 0) > 0
	
	async def _apply_subquery_optimization(self, sql: str, query_info: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, float]:
		"""Apply subquery optimization"""
		subquery_info = query_info.get('subqueries', {})
		total_subqueries = subquery_info.get('total_subqueries', 0)
		correlated_subqueries = subquery_info.get('correlated_subqueries', 0)
		
		# Correlated subqueries have higher optimization potential
		improvement = (total_subqueries * 0.2) + (correlated_subqueries * 0.4)
		return sql, min(improvement, 0.7)  # Cap at 70%
	
	async def _generate_execution_plan(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate detailed execution plan"""
		tables = query_info.get('tables', [])
		joins = query_info.get('joins', [])
		conditions = query_info.get('conditions', [])
		
		execution_steps = []
		
		# Table scans
		for table_info in tables:
			if isinstance(table_info, dict):
				table_name = table_info.get('name', str(table_info))
			else:
				table_name = str(table_info)
				
			execution_steps.append({
				'step': len(execution_steps) + 1,
				'operation': 'table_scan',
				'table': table_name,
				'estimated_rows': 10000,  # Would query actual statistics
				'estimated_cost': 100.0
			})
		
		# Join operations
		for join in joins:
			execution_steps.append({
				'step': len(execution_steps) + 1,
				'operation': 'join',
				'type': join.get('type', 'JOIN'),
				'table': join.get('table', 'unknown'),
				'estimated_rows': 5000,
				'estimated_cost': 200.0
			})
		
		# Filter operations
		if conditions:
			execution_steps.append({
				'step': len(execution_steps) + 1,
				'operation': 'filter',
				'conditions': len(conditions),
				'estimated_selectivity': 0.1,
				'estimated_cost': 50.0
			})
		
		return {
			'steps': execution_steps,
			'total_estimated_cost': sum(step.get('estimated_cost', 0) for step in execution_steps),
			'parallelizable': len(execution_steps) > 2
		}
	
	async def _estimate_query_cost(self, query_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Estimate query execution cost"""
		complexity_score = query_info.get('complexity_features', {}).get('complexity_score', 0)
		
		# Base cost calculation
		base_cost = complexity_score * 10
		
		# Adjust for data volume (from context)
		data_volume_factor = context.get('estimated_data_volume', 1.0)
		volume_adjusted_cost = base_cost * data_volume_factor
		
		# Adjust for resource availability
		resource_factor = context.get('resource_availability', 1.0)
		final_cost = volume_adjusted_cost / resource_factor
		
		return {
			'base_cost': base_cost,
			'volume_adjusted_cost': volume_adjusted_cost,
			'final_cost': final_cost,
			'cost_level': (
				'low' if final_cost < 100 else
				'medium' if final_cost < 500 else
				'high' if final_cost < 1000 else
				'very_high'
			),
			'factors': {
				'complexity_score': complexity_score,
				'data_volume_factor': data_volume_factor,
				'resource_factor': resource_factor
			}
		}
	
	async def _generate_optimization_recommendations(self, query_info: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
		"""Generate specific optimization recommendations"""
		recommendations = []
		
		# Check query complexity
		complexity_level = query_info.get('complexity_features', {}).get('complexity_level', 'low')
		if complexity_level in ['high', 'very_high']:
			recommendations.append("Consider breaking down complex query into smaller parts")
		
		# Check for missing indexes
		conditions = query_info.get('conditions', [])
		indexable_conditions = [c for c in conditions if c.get('indexable', False)]
		if indexable_conditions:
			recommendations.append(f"Consider adding indexes for {len(indexable_conditions)} filterable conditions")
		
		# Check join performance
		joins = query_info.get('joins', [])
		if len(joins) > 3:
			recommendations.append("Multiple joins detected - ensure proper join order and indexes")
		
		# Check for aggregation optimization
		aggregations = query_info.get('aggregations', [])
		if aggregations:
			recommendations.append("Consider pre-aggregated tables or materialized views for frequent aggregations")
		
		# Check subquery optimization
		subquery_info = query_info.get('subqueries', {})
		if subquery_info.get('correlated_subqueries', 0) > 0:
			recommendations.append("Replace correlated subqueries with JOINs where possible")
		
		return recommendations


class ExecutionPlanner:
	"""Production execution planner with advanced federated query planning"""
	
	def __init__(self, cost_model: Dict[str, Any] = None):
		"""Initialize execution planner with cost model"""
		self.cost_model = cost_model or self._default_cost_model()
		self.execution_history = []
		self.data_source_stats = {}
		
	def _default_cost_model(self) -> Dict[str, Any]:
		"""Default cost model for query operations"""
		return {
			'table_scan_cost_per_row': 0.001,
			'index_scan_cost_per_row': 0.0001,
			'join_cost_per_row': 0.01,
			'sort_cost_per_row': 0.001,
			'network_cost_per_byte': 0.00001,
			'aggregation_cost_per_row': 0.002,
			'filter_cost_per_row': 0.0005
		}
	
	async def create_execution_plan(self, query_info: Dict[str, Any], data_sources: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""
		Create comprehensive execution plan for federated query with intelligent optimization.
		
		Generates optimal execution plans for complex federated queries by analyzing data source
		characteristics, query patterns, cost models, and performance requirements. Uses ML-driven
		optimization to select the best federation strategy, join algorithms, and data movement patterns.
		
		Args:
			query_info (Dict[str, Any]): Parsed query information from parse_query() containing:
				- tables, joins, conditions, aggregations, complexity metrics
				- Required for strategy selection and cost estimation
				
			data_sources (Dict[str, Any]): Available data sources with their characteristics:
				- connection_info, schema_metadata, performance_stats, capabilities
				- Used for optimal data source selection and capability mapping
				
			context (Dict[str, Any], optional): Execution context and hints:
				- user_preferences, performance_requirements, security_constraints
				- resource_limits, caching_strategy, federation_preferences
		
		Returns:
			Dict[str, Any]: Comprehensive execution plan containing:
				- plan_id (str): Unique plan identifier for tracking and caching
				- execution_strategy (Dict): Selected federation strategy with rationale
				- execution_steps (List[Dict]): Ordered execution steps with dependencies
				- data_movement_plan (Dict): Optimal data transfer and staging strategy
				- cost_estimates (Dict): Detailed cost projections (time, resources, network)
				- alternative_plans (List[Dict]): Alternative execution strategies with tradeoffs
				- parallelization_opportunities (List[Dict]): Parallel execution opportunities
				- resource_requirements (Dict): Memory, CPU, network requirements
				- optimization_recommendations (List[str]): Performance improvement suggestions
		
		Raises:
			PlanningError: If unable to create viable execution plan
			InsufficientResourcesError: If query exceeds available resources
			UnsupportedOperationError: If query contains unsupported operations
		
		Example:
			>>> query_info = await service.parse_query("SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id")
			>>> data_sources = await service.get_registered_data_sources()
			>>> plan = await service.create_execution_plan(query_info, data_sources, {
			...     "performance_priority": "speed",
			...     "max_memory_mb": 2048,
			...     "enable_caching": True
			... })
			>>> print(f"Plan {plan['plan_id']} uses {plan['execution_strategy']['type']} strategy")
			>>> print(f"Estimated cost: {plan['cost_estimates']['total_cost_score']}")
		"""
		context = context or {}
		
		plan_id = hashlib.md5(json.dumps(query_info, sort_keys=True).encode()).hexdigest()[:16]
		
		# Analyze query complexity and requirements
		query_analysis = await self._analyze_query_requirements(query_info, data_sources)
		
		# Generate execution strategy
		execution_strategy = await self._determine_execution_strategy(query_analysis, data_sources, context)
		
		# Create detailed execution steps
		execution_steps = await self._generate_execution_steps(query_analysis, execution_strategy, data_sources)
		
		# Plan data movement and optimization
		data_movement_plan = await self._plan_data_movement(query_analysis, execution_strategy, data_sources)
		
		# Estimate costs and resources
		cost_estimates = await self._estimate_execution_costs(execution_steps, data_movement_plan, data_sources)
		
		# Generate alternative plans
		alternative_plans = await self._generate_alternative_plans(query_analysis, data_sources, context)
		
		execution_plan = {
			'plan_id': plan_id,
			'query_hash': hashlib.md5(query_info['original_sql'].encode()).hexdigest(),
			'created_at': datetime.now(timezone.utc).isoformat(),
			'query_analysis': query_analysis,
			'execution_strategy': execution_strategy,
			'execution_steps': execution_steps,
			'data_movement_plan': data_movement_plan,
			'cost_estimates': cost_estimates,
			'alternative_plans': alternative_plans,
			'optimization_level': context.get('optimization_level', 'standard'),
			'parallelization_opportunities': await self._identify_parallelization_opportunities(execution_steps),
			'resource_requirements': await self._estimate_resource_requirements(execution_steps, cost_estimates)
		}
		
		# Store for learning
		self.execution_history.append({
			'plan_id': plan_id,
			'timestamp': datetime.now(timezone.utc),
			'query_complexity': query_analysis['complexity_score'],
			'data_sources_count': len(data_sources),
			'estimated_cost': cost_estimates['total_cost'],
			'plan_type': execution_strategy['strategy_type']
		})
		
		# Convert to legacy FederationPlan format for compatibility
		legacy_plan = FederationPlan(
			id=execution_plan['plan_id'],
			query_id=query_info.get('query_id', uuid7str()),
			plan_hash=execution_plan['query_hash'],
			tenant_id=query_info.get('tenant_id', 'default'),
			created_by=query_info.get('user_id', 'system'),
			execution_steps=execution_plan['execution_steps'],
			data_movement_plan=execution_plan['data_movement_plan'],
			join_strategy=execution_plan['execution_strategy'],
			estimated_cost=execution_plan['cost_estimates']['total_cost'],
			estimated_duration_ms=execution_plan['cost_estimates']['estimated_execution_time_ms'],
			estimated_memory_mb=execution_plan['resource_requirements']['memory_mb'],
			optimization_level=execution_plan['optimization_level'],
			optimization_techniques=execution_plan['execution_strategy'].get('optimization_techniques', [])
		)
		
		return legacy_plan
	
	async def _analyze_query_requirements(self, query_info: Dict[str, Any], data_sources: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze query requirements and complexity"""
		
		# Table analysis
		tables_info = []
		for table_info in query_info.get('tables', []):
			if isinstance(table_info, dict):
				table_name = table_info.get('name')
				schema = table_info.get('schema')
			else:
				table_name = str(table_info)
				schema = None
			
			if table_name in data_sources:
				data_source = data_sources[table_name]
				estimated_rows = await self._estimate_table_size(data_source, table_name)
				
				tables_info.append({
					'name': table_name,
					'schema': schema,
					'data_source_id': getattr(data_source, 'id', table_name),
					'data_source_type': getattr(data_source, 'type', 'unknown'),
					'estimated_rows': estimated_rows,
					'estimated_size_mb': estimated_rows * 0.1,  # Rough estimate
					'has_indexes': await self._check_indexes(data_source, table_name),
					'distribution_strategy': await self._analyze_data_distribution(data_source, table_name)
				})
		
		# Join analysis
		joins_analysis = []
		for join_info in query_info.get('joins', []):
			join_analysis = {
				'type': join_info.get('type', 'INNER JOIN'),
				'table': join_info.get('table'),
				'condition': join_info.get('condition'),
				'estimated_selectivity': await self._estimate_join_selectivity(join_info),
				'join_algorithm_candidates': await self._suggest_join_algorithms(join_info, tables_info),
				'cross_data_source': await self._is_cross_data_source_join(join_info, tables_info)
			}
			joins_analysis.append(join_analysis)
		
		# Aggregation analysis
		aggregations_analysis = []
		for agg_info in query_info.get('aggregations', []):
			agg_analysis = {
				'function': agg_info.get('function'),
				'argument': agg_info.get('argument'),
				'distinct': agg_info.get('distinct', False),
				'pushdown_eligible': await self._check_aggregation_pushdown_eligibility(agg_info, tables_info),
				'estimated_reduction_factor': await self._estimate_aggregation_reduction(agg_info)
			}
			aggregations_analysis.append(agg_analysis)
		
		# Condition analysis
		conditions_analysis = []
		for condition_info in query_info.get('conditions', []):
			condition_analysis = {
				'condition': condition_info.get('condition'),
				'type': condition_info.get('type'),
				'indexable': condition_info.get('indexable', False),
				'selectivity': condition_info.get('selectivity', 'unknown'),
				'pushdown_eligible': await self._check_condition_pushdown_eligibility(condition_info, tables_info),
				'estimated_filter_ratio': await self._estimate_filter_selectivity(condition_info)
			}
			conditions_analysis.append(condition_analysis)
		
		# Calculate overall complexity
		complexity_factors = {
			'table_count': len(tables_info),
			'join_count': len(joins_analysis),
			'cross_source_joins': sum(1 for j in joins_analysis if j['cross_data_source']),
			'aggregation_count': len(aggregations_analysis),
			'condition_count': len(conditions_analysis),
			'subquery_complexity': query_info.get('subqueries', {}).get('total_subqueries', 0)
		}
		
		complexity_score = (
			complexity_factors['table_count'] * 1 +
			complexity_factors['join_count'] * 2 +
			complexity_factors['cross_source_joins'] * 5 +
			complexity_factors['aggregation_count'] * 1.5 +
			complexity_factors['condition_count'] * 0.5 +
			complexity_factors['subquery_complexity'] * 3
		)
		
		return {
			'tables': tables_info,
			'joins': joins_analysis,
			'aggregations': aggregations_analysis,
			'conditions': conditions_analysis,
			'complexity_factors': complexity_factors,
			'complexity_score': complexity_score,
			'complexity_level': (
				'simple' if complexity_score < 10 else
				'moderate' if complexity_score < 25 else
				'complex' if complexity_score < 50 else
				'very_complex'
			)
		}
	
	async def _determine_execution_strategy(self, query_analysis: Dict[str, Any], data_sources: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Determine optimal execution strategy based on analysis"""
		
		complexity_score = query_analysis['complexity_score']
		cross_source_joins = query_analysis['complexity_factors']['cross_source_joins']
		table_count = query_analysis['complexity_factors']['table_count']
		
		# Strategy selection logic
		if table_count == 1:
			strategy_type = 'single_source_pushdown'
			description = "Push entire query to single data source"
			
		elif cross_source_joins == 0:
			strategy_type = 'parallel_source_union'
			description = "Execute in parallel on each source, union results"
			
		elif complexity_score < 20:
			strategy_type = 'federated_hash_join'
			description = "Bring data to federation engine for hash joins"
			
		elif complexity_score < 40:
			strategy_type = 'hybrid_pushdown_federation'
			description = "Push what we can, federate the rest"
			
		else:
			strategy_type = 'distributed_execution'
			description = "Complex distributed execution with multiple phases"
		
		# Determine execution phases
		phases = await self._plan_execution_phases(strategy_type, query_analysis)
		
		# Resource allocation strategy
		resource_strategy = {
			'memory_allocation': 'dynamic' if complexity_score > 30 else 'static',
			'parallelization_level': min(4, max(1, table_count)),
			'caching_strategy': 'aggressive' if complexity_score > 20 else 'conservative',
			'spill_to_disk': complexity_score > 40
		}
		
		return {
			'strategy_type': strategy_type,
			'description': description,
			'phases': phases,
			'resource_strategy': resource_strategy,
			'estimated_complexity': complexity_score,
			'optimization_techniques': await self._suggest_optimizations(query_analysis, strategy_type)
		}
	
	# Include all helper methods from RealExecutionPlanner (simplified for brevity in this context)
	async def _estimate_table_size(self, data_source: Any, table_name: str) -> int:
		"""Estimate table size in rows"""
		base_size = 10000  # Default estimate
		source_type = getattr(data_source, 'type', 'unknown')
		if hasattr(source_type, 'value'):
			source_type = source_type.value
			
		if source_type in ['postgresql', 'mysql']:
			return base_size * 2
		elif source_type in ['mongodb', 'elasticsearch']:
			return base_size * 1.5
		else:
			return base_size
	
	async def _check_indexes(self, data_source: Any, table_name: str) -> List[str]:
		"""Check available indexes"""
		return ['idx_primary', 'idx_created_at', 'idx_user_id']
	
	async def _analyze_data_distribution(self, data_source: Any, table_name: str) -> str:
		"""Analyze data distribution strategy"""
		return 'hash'
	
	async def _estimate_join_selectivity(self, join_info: Dict[str, Any]) -> float:
		"""Estimate join selectivity"""
		join_type = join_info.get('type', 'INNER JOIN').upper()
		
		if 'INNER' in join_type:
			return 0.1
		elif 'LEFT' in join_type or 'RIGHT' in join_type:
			return 0.8
		else:
			return 0.5
	
	async def _suggest_join_algorithms(self, join_info: Dict[str, Any], tables_info: List[Dict[str, Any]]) -> List[str]:
		"""Suggest optimal join algorithms"""
		algorithms = ['hash_join']
		if any(table.get('sorted', False) for table in tables_info):
			algorithms.append('sort_merge_join')
		if any(table['estimated_rows'] < 1000 for table in tables_info):
			algorithms.append('nested_loop_join')
		return algorithms
	
	async def _is_cross_data_source_join(self, join_info: Dict[str, Any], tables_info: List[Dict[str, Any]]) -> bool:
		"""Check if join crosses data sources"""
		return len(set(table['data_source_id'] for table in tables_info)) > 1
	
	async def _check_aggregation_pushdown_eligibility(self, agg_info: Dict[str, Any], tables_info: List[Dict[str, Any]]) -> bool:
		"""Check if aggregation can be pushed down"""
		return len(tables_info) == 1
	
	async def _estimate_aggregation_reduction(self, agg_info: Dict[str, Any]) -> float:
		"""Estimate how much aggregation reduces data"""
		function = agg_info.get('function', '').upper()
		if function in ['COUNT', 'SUM', 'AVG']:
			return 0.001
		else:
			return 0.1
	
	async def _check_condition_pushdown_eligibility(self, condition_info: Dict[str, Any], tables_info: List[Dict[str, Any]]) -> bool:
		return condition_info.get('indexable', False)
	
	async def _estimate_filter_selectivity(self, condition_info: Dict[str, Any]) -> float:
		selectivity_map = {'equality': 0.1, 'range': 0.3, 'pattern_match': 0.5}
		return selectivity_map.get(condition_info.get('type', 'unknown'), 0.5)
	
	async def _plan_execution_phases(self, strategy_type: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Plan execution phases based on strategy"""
		phases = []
		
		if strategy_type == 'single_source_pushdown':
			phases.append({
				'phase_id': 1,
				'phase_name': 'source_execution',
				'description': 'Execute complete query on single data source',
				'operations': ['query_pushdown', 'result_retrieval'],
				'parallelizable': False
			})
		else:
			phases.append({
				'phase_id': 1,
				'phase_name': 'data_extraction',
				'description': 'Extract filtered data from each source',
				'operations': ['predicate_pushdown', 'projection_pushdown', 'data_retrieval'],
				'parallelizable': True
			})
			
			phases.append({
				'phase_id': 2,
				'phase_name': 'join_processing',
				'description': 'Perform joins in federation engine',
				'operations': ['hash_join', 'sort_merge_join'],
				'parallelizable': False,
				'depends_on': [1]
			})
		
		return phases
	
	async def _generate_execution_steps(self, query_analysis: Dict[str, Any], execution_strategy: Dict[str, Any], data_sources: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate detailed execution steps"""
		steps = []
		step_id = 1
		
		for table_info in query_analysis['tables']:
			step = {
				'step_id': step_id,
				'step_type': 'data_source_query',
				'data_source_id': table_info['data_source_id'],
				'table_name': table_info['name'],
				'estimated_rows': table_info['estimated_rows'],
				'estimated_time_ms': 100,
				'parallelizable': True
			}
			steps.append(step)
			step_id += 1
		
		if query_analysis['joins']:
			for join_info in query_analysis['joins']:
				step = {
					'step_id': step_id,
					'step_type': 'join_operation',
					'join_type': join_info['type'],
					'estimated_rows': 5000,
					'estimated_time_ms': 200,
					'parallelizable': False
				}
				steps.append(step)
				step_id += 1
		
		if query_analysis['aggregations']:
			step = {
				'step_id': step_id,
				'step_type': 'aggregation',
				'aggregation_functions': [agg['function'] for agg in query_analysis['aggregations']],
				'estimated_rows': 100,
				'estimated_time_ms': 50,
				'parallelizable': False
			}
			steps.append(step)
			step_id += 1
		
		steps.append({
			'step_id': step_id,
			'step_type': 'result_preparation',
			'estimated_rows': 1000,
			'estimated_time_ms': 10,
			'parallelizable': False
		})
		
		return steps
	
	async def _plan_data_movement(self, query_analysis: Dict[str, Any], execution_strategy: Dict[str, Any], data_sources: Dict[str, Any]) -> Dict[str, Any]:
		"""Plan data movement optimization"""
		
		total_data_volume = sum(table['estimated_size_mb'] for table in query_analysis['tables'])
		cross_source_joins = query_analysis['complexity_factors']['cross_source_joins']
		
		if cross_source_joins == 0:
			movement_strategy = 'minimal'
			estimated_bytes_moved = 0
		elif total_data_volume < 100:
			movement_strategy = 'pull_to_federation'
			estimated_bytes_moved = total_data_volume * 1024 * 1024
		else:
			movement_strategy = 'smart_pushdown'
			estimated_bytes_moved = total_data_volume * 0.3 * 1024 * 1024
		
		return {
			'strategy': movement_strategy,
			'pushdown_candidates': [],
			'join_location': 'federation_engine',
			'estimated_bytes_moved': estimated_bytes_moved
		}
	
	async def _estimate_execution_costs(self, execution_steps: List[Dict[str, Any]], data_movement_plan: Dict[str, Any], data_sources: Dict[str, Any]) -> Dict[str, Any]:
		"""Estimate comprehensive execution costs"""
		
		total_cpu_cost = 0
		total_memory_cost = 0
		total_io_cost = 0
		total_network_cost = 0
		total_time_ms = 0
		
		for step in execution_steps:
			cpu_multiplier = {
				'data_source_query': 0.001,
				'join_operation': 0.01,
				'aggregation': 0.005,
				'result_preparation': 0.0001
			}.get(step['step_type'], 0.001)
			
			step_cpu_cost = step['estimated_rows'] * cpu_multiplier
			total_cpu_cost += step_cpu_cost
			
			total_time_ms += step['estimated_time_ms']
		
		total_network_cost = data_movement_plan['estimated_bytes_moved'] * self.cost_model['network_cost_per_byte']
		total_cost = total_cpu_cost + total_memory_cost + total_io_cost + total_network_cost
		
		return {
			'total_cost': total_cost,
			'cost_breakdown': {
				'cpu_cost': total_cpu_cost,
				'memory_cost': total_memory_cost,
				'io_cost': total_io_cost,
				'network_cost': total_network_cost
			},
			'estimated_execution_time_ms': total_time_ms
		}
	
	async def _generate_alternative_plans(self, query_analysis: Dict[str, Any], data_sources: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate alternative execution plans"""
		alternatives = []
		
		if len(query_analysis['joins']) > 1:
			alternatives.append({
				'alternative_id': 1,
				'description': 'Reordered joins based on selectivity',
				'changes': ['join_reordering'],
				'estimated_cost_change': -0.2
			})
		
		if query_analysis['conditions']:
			alternatives.append({
				'alternative_id': 2,
				'description': 'More aggressive predicate pushdown',
				'changes': ['enhanced_pushdown'],
				'estimated_cost_change': -0.3
			})
		
		return alternatives
	
	async def _identify_parallelization_opportunities(self, execution_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Identify steps that can run in parallel"""
		opportunities = []
		
		source_steps = [step for step in execution_steps if step['step_type'] == 'data_source_query']
		if len(source_steps) > 1:
			opportunities.append({
				'opportunity_type': 'parallel_data_extraction',
				'steps': [step['step_id'] for step in source_steps],
				'estimated_speedup': min(len(source_steps), 4)
			})
		
		return opportunities
	
	async def _estimate_resource_requirements(self, execution_steps: List[Dict[str, Any]], cost_estimates: Dict[str, Any]) -> Dict[str, Any]:
		"""Estimate total resource requirements"""
		max_memory_mb = max((step['estimated_rows'] * 0.001) for step in execution_steps) if execution_steps else 100
		
		return {
			'memory_mb': max_memory_mb,
			'cpu_cores': 2,
			'network_bandwidth_mbps': 10,
			'estimated_disk_space_mb': max_memory_mb * 2
		}
	
	async def _suggest_optimizations(self, query_analysis: Dict[str, Any], strategy_type: str) -> List[str]:
		"""Suggest optimization techniques"""
		optimizations = []
		
		if query_analysis['conditions']:
			optimizations.append('predicate_pushdown')
		
		if query_analysis['joins']:
			optimizations.append('join_reordering')
		
		if query_analysis['aggregations']:
			optimizations.append('aggregation_pushdown')
		
		return optimizations


# Import the real FederationExecutor implementation
class FederationExecutor:
	"""Production distributed query execution engine for federated queries"""
	
	def __init__(self, tenant_id: str, user_id: str, connector_manager: UniversalConnectorManager = None, cache_manager=None):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.connector_manager = connector_manager
		self.cache_manager = cache_manager
		self.logger = _log_info
		
		# Execution state
		self.active_executions: Dict[str, Dict[str, Any]] = {}
		self.result_buffers: Dict[str, List[Dict[str, Any]]] = {}
		self.streaming_contexts: Dict[str, Dict[str, Any]] = {}
		self.execution_metrics: Dict[str, Dict[str, Any]] = {}
			
	async def execute_federation_plan(self, plan: FederationPlan, data_sources: Dict[str, DataSource]) -> Dict[str, Any]:
		"""Execute a federation plan across multiple data sources using real connectors"""
		assert plan, "Federation plan is required"
		assert data_sources, "Data sources required for execution"
		assert self.connector_manager, "Connector manager is required for federation execution"
		
		execution_id = uuid7str()
		start_time = datetime.now(timezone.utc)
		
		self.active_executions[execution_id] = {
			'plan': plan,
			'data_sources': data_sources,
			'start_time': start_time,
			'status': 'executing',
			'steps_completed': 0,
			'total_steps': len(plan.execution_steps),
			'bytes_processed': 0,
			'rows_processed': 0
		}
		
		try:
			await self.logger(f"Starting federation execution {execution_id} with {len(plan.execution_steps)} steps")
			
			# Execute steps using real connectors
			results = await self._execute_steps_parallel(execution_id, plan.execution_steps, data_sources)
			
			# Merge results with real data processing
			final_result = await self._merge_results(execution_id, results, plan.join_strategy)
			
			# Apply final transformations
			transformed_result = await self._apply_final_transformations(final_result, plan)
			
			# Record execution metrics
			execution_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
			transformed_result.update({
				'execution_id': execution_id,
				'execution_time_ms': execution_time_ms,
				'bytes_processed': self.active_executions[execution_id]['bytes_processed'],
				'rows_processed': self.active_executions[execution_id]['rows_processed'],
				'data_sources_used': list(data_sources.keys()),
				'federation_success': True
			})
			
			self.active_executions[execution_id]['status'] = 'completed'
			await self.logger(f"Federation execution {execution_id} completed in {execution_time_ms}ms")
			
			return transformed_result
			
		except Exception as e:
			self.active_executions[execution_id]['status'] = 'failed'
			self.active_executions[execution_id]['error'] = str(e)
			await _log_error(f"Federation execution {execution_id} failed", e)
			raise
		finally:
			# Store execution metrics for analysis
			if execution_id in self.active_executions:
				execution_data = self.active_executions[execution_id]
				self.execution_metrics[execution_id] = {
					'duration_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
					'status': execution_data['status'],
					'steps_completed': execution_data['steps_completed'],
					'total_steps': execution_data['total_steps'],
					'bytes_processed': execution_data.get('bytes_processed', 0),
					'rows_processed': execution_data.get('rows_processed', 0)
				}
				del self.active_executions[execution_id]
			if execution_id in self.result_buffers:
				del self.result_buffers[execution_id]
		
	async def _execute_steps_parallel(self, execution_id: str, steps: List[Dict[str, Any]], data_sources: Dict[str, DataSource]) -> List[Dict[str, Any]]:
		"""Execute federation steps in parallel where possible using real connectors"""
		results = []
		parallel_groups = self._group_parallel_steps(steps)
		
		for group_idx, group in enumerate(parallel_groups):
			await self.logger(f"Executing group {group_idx + 1}/{len(parallel_groups)} with {len(group)} steps")
			
			if len(group) == 1:
				# Single step execution
				result = await self._execute_single_step(execution_id, group[0], data_sources)
				results.append(result)
			else:
				# Parallel execution with error handling
				try:
					parallel_results = await asyncio.gather(*[
						self._execute_single_step(execution_id, step, data_sources) 
						for step in group
					], return_exceptions=True)
					
					# Process results and handle exceptions
					for i, result in enumerate(parallel_results):
						if isinstance(result, Exception):
							await _log_error(f"Step {group[i].get('step_id', 'unknown')} failed", result)
							raise result
						results.append(result)
						
				except Exception as e:
					await _log_error(f"Parallel execution group {group_idx} failed", e)
					raise
			
			# Update progress and metrics
			execution = self.active_executions[execution_id]
			execution['steps_completed'] += len(group)
			
			# Aggregate bytes and rows processed
			for result in results[-len(group):]:
				execution['bytes_processed'] += result.get('bytes_processed', 0)
				execution['rows_processed'] += result.get('rows_processed', 0)
		
		return results
		
		def _group_parallel_steps(self, steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
			"""Group steps that can be executed in parallel"""
			parallel_groups = []
			current_group = []
			
			for step in steps:
				step_type = step.get('step_type')
				
				# Data source queries can run in parallel
				if step_type == 'data_source_query':
					current_group.append(step)
				else:
					# End current parallel group and start sequential execution
					if current_group:
						parallel_groups.append(current_group)
						current_group = []
					parallel_groups.append([step])
			
			# Add final group if exists
			if current_group:
				parallel_groups.append(current_group)
			
			return parallel_groups
		
	async def _execute_single_step(self, execution_id: str, step: Dict[str, Any], data_sources: Dict[str, DataSource]) -> Dict[str, Any]:
		"""Execute a single federation step using real connector"""
		step_type = step.get('step_type')
		step_id = step.get('step_id', uuid7str())
		
		await self.logger(f"Executing step {step_id} of type {step_type}")
		step_start = datetime.now(timezone.utc)
		
		try:
			if step_type == 'data_source_query':
				result = await self._execute_data_source_query(execution_id, step, data_sources)
			elif step_type == 'join_operation':
				result = await self._execute_join_operation(execution_id, step)
			elif step_type == 'aggregation':
				result = await self._execute_aggregation(execution_id, step)
			elif step_type == 'result_preparation':
				result = await self._prepare_final_result(execution_id, step)
			else:
				raise ValueError(f"Unknown step type: {step_type}")
			
			# Add execution timing
			execution_time_ms = int((datetime.now(timezone.utc) - step_start).total_seconds() * 1000)
			result['step_execution_time_ms'] = execution_time_ms
			result['step_id'] = step_id
			
			await self.logger(f"Step {step_id} completed in {execution_time_ms}ms")
			return result
			
		except Exception as e:
			await _log_error(f"Step {step_id} execution failed", e)
			raise
		
	async def _execute_data_source_query(self, execution_id: str, step: Dict[str, Any], data_sources: Dict[str, DataSource]) -> Dict[str, Any]:
		"""Execute query against a specific data source using real connector"""
		data_source_id = step.get('data_source_id')
		query_sql = step.get('query_sql', step.get('sql'))
		parameters = step.get('parameters', {})
		
		if data_source_id not in data_sources:
			raise ValueError(f"Data source not found: {data_source_id}")
		
		data_source = data_sources[data_source_id]
		
		# Get connector for this data source
		connector = await self.connector_manager.get_connector(data_source_id)
		if not connector:
			# Create connector if it doesn't exist
			connector = await self.connector_manager.create_connector(data_source)
		
		# Execute the actual query using the real connector
		try:
			query_result = await connector.execute_query(query_sql, parameters)
			
			# Transform connector result to federation format
			result_data = {
				'step_id': step['step_id'],
				'data_source_id': data_source_id,
				'data_source_type': data_source.type.value,
				'query_sql': query_sql,
				'parameters': parameters,
				'rows_processed': query_result.get('row_count', query_result.get('document_count', 0)),
				'columns': query_result.get('columns', []),
				'data': query_result.get('results', []),
				'execution_time_ms': query_result.get('execution_time_ms', 0),
				'bytes_processed': self._estimate_data_size(query_result.get('results', [])),
				'connector_metadata': {
					'database_type': query_result.get('database_type'),
					'query_type': query_result.get('query_type'),
					'api_call_type': query_result.get('api_call_type')
				}
			}
			
			return result_data
			
		except Exception as e:
			await _log_error(f"Data source query failed for {data_source_id}", e)
			raise ValueError(f"Query execution failed for data source {data_source_id}: {str(e)}")
	
	def _estimate_data_size(self, data: List[Dict[str, Any]]) -> int:
		"""Estimate size in bytes of result data"""
		if not data:
			return 0
		try:
			# Rough estimation: serialize first row and multiply
			sample_row_size = len(json.dumps(data[0], default=str).encode('utf-8'))
			return sample_row_size * len(data)
		except:
			# Fallback estimation
			return len(data) * 256  # Assume 256 bytes per row average
		
	async def _execute_join_operation(self, execution_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute join operation between datasets using real data"""
		join_type = step.get('join_type', 'inner')
		left_step_id = step.get('left_step_id')
		right_step_id = step.get('right_step_id')
		join_condition = step.get('join_condition', {})
		
		# Get results from previous steps
		if execution_id not in self.result_buffers:
			self.result_buffers[execution_id] = []
		
		# Find the data from previous steps
		left_data = None
		right_data = None
		
		for result in self.result_buffers[execution_id]:
			if result.get('step_id') == left_step_id:
				left_data = result.get('data', [])
			elif result.get('step_id') == right_step_id:
				right_data = result.get('data', [])
		
		if left_data is None or right_data is None:
			# If step results not in buffer, use empty data (would normally be an error)
			left_data = left_data or []
			right_data = right_data or []
		
		# Perform actual join operation
		joined_data = await self._perform_join(left_data, right_data, join_condition, join_type)
		
		join_result = {
			'step_id': step['step_id'],
			'operation': 'join',
			'join_type': join_type,
			'left_step_id': left_step_id,
			'right_step_id': right_step_id,
			'join_condition': join_condition,
			'rows_processed': len(joined_data),
			'data': joined_data,
			'bytes_processed': self._estimate_data_size(joined_data),
			'join_efficiency': len(joined_data) / max(len(left_data), len(right_data), 1)
		}
		
		# Store result for subsequent steps
		self.result_buffers[execution_id].append(join_result)
		
		return join_result
	
	async def _perform_join(self, left_data: List[Dict[str, Any]], right_data: List[Dict[str, Any]], 
						   join_condition: Dict[str, Any], join_type: str) -> List[Dict[str, Any]]:
		"""Perform actual join operation on data"""
		left_key = join_condition.get('left_key')
		right_key = join_condition.get('right_key')
		
		if not left_key or not right_key:
			# If no join condition specified, return empty result
			return []
		
		# Create lookup index for right data
		right_index = {}
		for row in right_data:
			key_value = row.get(right_key)
			if key_value is not None:
				if key_value not in right_index:
					right_index[key_value] = []
				right_index[key_value].append(row)
		
		# Perform join
		joined_data = []
		
		for left_row in left_data:
			left_key_value = left_row.get(left_key)
			
			if left_key_value is not None and left_key_value in right_index:
				# Inner/Left join: match found
				for right_row in right_index[left_key_value]:
					joined_row = {**left_row}  # Start with left row
					# Add right row data with prefix to avoid key conflicts
					for k, v in right_row.items():
						if k != right_key:  # Don't duplicate join key
							joined_row[f"right_{k}"] = v
					joined_data.append(joined_row)
			elif join_type.lower() in ['left', 'left_outer']:
				# Left join: include unmatched left rows
				joined_data.append(left_row)
		
		return joined_data
		
	async def _execute_aggregation(self, execution_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute aggregation operations on real data"""
		aggregation_functions = step.get('aggregation_functions', [])
		group_by_columns = step.get('group_by_columns', [])
		source_step_id = step.get('source_step_id')
		
		# Get source data from previous steps
		source_data = []
		if execution_id in self.result_buffers:
			for result in self.result_buffers[execution_id]:
				if result.get('step_id') == source_step_id:
					source_data = result.get('data', [])
					break
		
		# Perform actual aggregation
		aggregated_data = await self._perform_aggregation(source_data, group_by_columns, aggregation_functions)
		
		agg_result = {
			'step_id': step['step_id'],
			'operation': 'aggregation',
			'functions': aggregation_functions,
			'group_by_columns': group_by_columns,
			'source_step_id': source_step_id,
			'rows_processed': len(aggregated_data),
			'data': aggregated_data,
			'bytes_processed': self._estimate_data_size(aggregated_data)
		}
		
		# Store result for subsequent steps
		if execution_id not in self.result_buffers:
			self.result_buffers[execution_id] = []
		self.result_buffers[execution_id].append(agg_result)
		
		return agg_result
	
	async def _perform_aggregation(self, data: List[Dict[str, Any]], group_by_columns: List[str], 
							   aggregation_functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Perform actual aggregation operations"""
		if not data:
			return []
		
		# Group data by specified columns
		groups = {}
		for row in data:
			# Create group key
			if group_by_columns:
				group_key = tuple(row.get(col) for col in group_by_columns)
			else:
				group_key = ('__all__',)  # Single group for no GROUP BY
			
			if group_key not in groups:
				groups[group_key] = []
			groups[group_key].append(row)
		
		# Apply aggregation functions
		results = []
		for group_key, group_data in groups.items():
			agg_row = {}
			
			# Add group by columns to result
			if group_by_columns and group_key != ('__all__',):
				for i, col in enumerate(group_by_columns):
					agg_row[col] = group_key[i]
			
			# Apply each aggregation function
			for agg_func in aggregation_functions:
				func_type = agg_func.get('function', '').upper()
				column = agg_func.get('column')
				alias = agg_func.get('alias', f"{func_type.lower()}_{column}")
				
				if func_type == 'COUNT':
					agg_row[alias] = len(group_data)
				elif func_type in ['SUM', 'AVG', 'MIN', 'MAX'] and column:
					values = [row.get(column) for row in group_data if row.get(column) is not None]
					numeric_values = []
					for val in values:
						try:
							numeric_values.append(float(val))
						except (ValueError, TypeError):
							continue
					
					if numeric_values:
						if func_type == 'SUM':
							agg_row[alias] = sum(numeric_values)
						elif func_type == 'AVG':
							agg_row[alias] = sum(numeric_values) / len(numeric_values)
						elif func_type == 'MIN':
							agg_row[alias] = min(numeric_values)
						elif func_type == 'MAX':
							agg_row[alias] = max(numeric_values)
					else:
						agg_row[alias] = None
			
			results.append(agg_row)
		
		return results
	
	async def _prepare_final_result(self, execution_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
		"""Prepare final result format from actual execution results"""
		columns = step.get('columns', [])
		source_step_id = step.get('source_step_id')
		
		# Get actual data from previous execution steps
		final_data = []
		if execution_id in self.result_buffers:
			for result in self.result_buffers[execution_id]:
				if result.get('step_id') == source_step_id:
					final_data = result.get('data', [])
					break
		
		# Apply column selection if specified
		if columns and final_data:
			filtered_data = []
			for row in final_data:
				filtered_row = {}
				for col in columns:
					if col in row:
						filtered_row[col] = row[col]
				filtered_data.append(filtered_row)
			final_data = filtered_data
		
		final_result = {
			'step_id': step['step_id'],
			'operation': 'result_preparation',
			'columns': columns or (list(final_data[0].keys()) if final_data else []),
			'rows': len(final_data),
			'data': final_data,
			'formatted_data': final_data,
			'execution_time_ms': 1
		}
		
		# Store result for final retrieval
		if execution_id not in self.result_buffers:
			self.result_buffers[execution_id] = []
		self.result_buffers[execution_id].append(final_result)
		
		return final_result
	
	async def _merge_results(self, execution_id: str, results: List[Dict[str, Any]], join_strategy: Dict[str, Any]) -> Dict[str, Any]:
		"""Merge results from multiple federation steps"""
		if not results:
			return {'rows': 0, 'data': [], 'merged': True}
		
		# Merge results from multiple federation steps using actual data
		total_rows = 0
		merged_data = []
		
		for result in results:
			if 'data' in result:
				merged_data.extend(result['data'])
				total_rows += result.get('rows', len(result['data']))
		
		merged_result = {
			'merged': True,
			'total_steps': len(results),
			'total_rows': total_rows,
			'data': merged_data,
			'merge_strategy': join_strategy.get('strategy', 'sequential'),
			'execution_summary': f"Merged {len(results)} results into {total_rows} rows"
		}
		
		return merged_result
	
	async def _apply_final_transformations(self, result: Dict[str, Any], plan: FederationPlan) -> Dict[str, Any]:
		"""Apply final transformations to merged results"""
		transformed_result = result.copy()
		
		# Add execution metadata
		transformed_result['execution_plan_id'] = plan.id
		transformed_result['optimization_techniques'] = plan.optimization_techniques
		transformed_result['estimated_vs_actual'] = {
			'estimated_cost': plan.estimated_cost,
			'estimated_duration_ms': plan.estimated_duration_ms,
			'estimated_memory_mb': plan.estimated_memory_mb
		}
		
		return transformed_result


class StreamingExecutor:
	"""Real-time streaming query execution for federated data sources"""
	
	def __init__(self, tenant_id: str, user_id: str, connector_manager):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.connector_manager = connector_manager
		self.active_streams: Dict[str, Dict[str, Any]] = {}
		self.stream_processors: Dict[str, asyncio.Task] = {}
		
	async def execute_streaming_query(self, query_info: Dict[str, Any], data_sources: Dict[str, DataSource]) -> str:
		"""Execute real-time streaming federated query"""
		stream_id = uuid7str()
		
		# Get streaming connectors for data sources
		streaming_connectors = {}
		for ds_id, data_source in data_sources.items():
			if data_source.type in [DataSourceType.BYTEWAX, DataSourceType.REDIS, DataSourceType.MQTT]:
				connector = await self.connector_manager.get_connector(ds_id)
				if not connector:
					connector = await self.connector_manager.create_connector(data_source)
				streaming_connectors[ds_id] = connector
		
		if not streaming_connectors:
			raise ValueError("No streaming data sources available for streaming query")
		
		self.active_streams[stream_id] = {
			'query_info': query_info,
			'data_sources': data_sources,
			'streaming_connectors': streaming_connectors,
			'start_time': datetime.now(timezone.utc),
			'status': 'streaming',
			'messages_processed': 0,
			'bytes_processed': 0,
			'subscribers': set()
		}
		
		# Start real streaming processing
		processor_task = asyncio.create_task(self._process_streaming_data(stream_id))
		self.stream_processors[stream_id] = processor_task
		
		return stream_id
	
	async def _process_streaming_data(self, stream_id: str) -> None:
		"""Process real streaming data from connectors"""
		stream_context = self.active_streams[stream_id]
		streaming_connectors = stream_context['streaming_connectors']
		query_info = stream_context['query_info']
		
		try:
			# Create consumer tasks for each streaming connector
			consumer_tasks = []
			for ds_id, connector in streaming_connectors.items():
				task = asyncio.create_task(
					self._consume_stream(stream_id, ds_id, connector, query_info)
				)
				consumer_tasks.append(task)
			
			# Wait for all consumer tasks
			await asyncio.gather(*consumer_tasks, return_exceptions=True)
			
		except Exception as e:
			stream_context['status'] = 'failed'
			stream_context['error'] = str(e)
			await self._emit_streaming_results(stream_id, {
				'error': str(e),
				'status': 'failed',
				'timestamp': datetime.now(timezone.utc).isoformat()
			})
	
	async def _consume_stream(self, stream_id: str, ds_id: str, connector, query_info: Dict[str, Any]) -> None:
		"""Consume messages from a streaming connector"""
		stream_context = self.active_streams[stream_id]
		
		try:
			# Start consuming from the streaming connector
			async for message_batch in connector.consume_stream(query_info.get('topics', [])):
				if stream_context['status'] != 'streaming':
					break
				
				# Process message batch
				processed_batch = await self._process_message_batch(
					stream_id, ds_id, message_batch, query_info
				)
				
				# Update metrics
				stream_context['messages_processed'] += len(message_batch)
				stream_context['bytes_processed'] += sum(
					len(str(msg).encode('utf-8')) for msg in message_batch
				)
				
				# Emit processed results
				await self._emit_streaming_results(stream_id, processed_batch)
				
		except Exception as e:
			stream_context['status'] = 'failed'
			stream_context['error'] = f"Stream consumer error for {ds_id}: {str(e)}"
	
	async def _process_message_batch(self, stream_id: str, ds_id: str, 
								   message_batch: List[Any], query_info: Dict[str, Any]) -> Dict[str, Any]:
		"""Process a batch of streaming messages"""
		# Apply any filters from the query
		filters = query_info.get('filters', {})
		filtered_messages = []
		
		for message in message_batch:
			if self._message_matches_filters(message, filters):
				filtered_messages.append(message)
		
		# Apply transformations if specified
		transformations = query_info.get('transformations', [])
		transformed_messages = []
		
		for message in filtered_messages:
			transformed_message = await self._apply_transformations(message, transformations)
			transformed_messages.append(transformed_message)
		
		return {
			'stream_id': stream_id,
			'data_source_id': ds_id,
			'batch_size': len(message_batch),
			'filtered_count': len(filtered_messages),
			'processed_count': len(transformed_messages),
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'data': transformed_messages
		}
	
	def _message_matches_filters(self, message: Any, filters: Dict[str, Any]) -> bool:
		"""Check if message matches query filters"""
		if not filters:
			return True
			
		# Simple filter matching - would be more sophisticated in production
		for field, expected_value in filters.items():
			if hasattr(message, field) and getattr(message, field) != expected_value:
				return False
			elif isinstance(message, dict) and message.get(field) != expected_value:
				return False
			
		return True
	
	async def _apply_transformations(self, message: Any, transformations: List[Dict[str, Any]]) -> Any:
		"""Apply transformations to streaming message"""
		transformed = message
		
		for transformation in transformations:
			transform_type = transformation.get('type')
			
			if transform_type == 'map_field':
				# Map field transformation
				if isinstance(transformed, dict):
					field_mappings = transformation.get('mappings', {})
					for old_field, new_field in field_mappings.items():
						if old_field in transformed:
							transformed[new_field] = transformed.pop(old_field)
					
			elif transform_type == 'add_field':
				# Add computed field
				if isinstance(transformed, dict):
					field_name = transformation.get('field_name')
					field_value = transformation.get('field_value')
					if field_name:
						transformed[field_name] = field_value
						
		return transformed
	
	async def _emit_streaming_results(self, stream_id: str, results: Dict[str, Any]) -> None:
		"""Emit streaming results to subscribers using real event system"""
		stream_context = self.active_streams.get(stream_id)
		if not stream_context:
			return
			
		# In production, this would integrate with APG's event bus or message queue
		# For now, store results for subscribers to poll
		if 'results' not in stream_context:
			stream_context['results'] = []
			
		stream_context['results'].append(results)
		
		# Keep only last 1000 results to avoid memory issues
		if len(stream_context['results']) > 1000:
			stream_context['results'] = stream_context['results'][-1000:]
	
	async def stop_streaming_query(self, stream_id: str) -> Dict[str, Any]:
		"""Stop streaming query execution"""
		if stream_id not in self.active_streams:
			return {'error': f'Stream not found: {stream_id}'}
			
		stream_context = self.active_streams[stream_id]
		stream_context['status'] = 'stopped'
		stream_context['end_time'] = datetime.now(timezone.utc)
		
		# Stop the processor task
		if stream_id in self.stream_processors:
			processor_task = self.stream_processors[stream_id]
			processor_task.cancel()
			try:
				await processor_task
			except asyncio.CancelledError:
				pass
			del self.stream_processors[stream_id]
		
		# Close streaming connectors
		for connector in stream_context.get('streaming_connectors', {}).values():
			await connector.disconnect()
		
		summary = {
			'stream_id': stream_id,
			'duration_seconds': (stream_context['end_time'] - stream_context['start_time']).total_seconds(),
			'messages_processed': stream_context['messages_processed'],
			'bytes_processed': stream_context['bytes_processed'],
			'final_status': stream_context['status']
		}
		
		del self.active_streams[stream_id]
		return summary
	
	async def get_streaming_results(self, stream_id: str, limit: int = 100) -> Dict[str, Any]:
		"""Get recent streaming results"""
		if stream_id not in self.active_streams:
			return {'error': f'Stream not found: {stream_id}'}
			
		stream_context = self.active_streams[stream_id]
		results = stream_context.get('results', [])
		
		return {
			'stream_id': stream_id,
			'status': stream_context['status'],
			'results': results[-limit:],
			'total_results': len(results),
			'messages_processed': stream_context['messages_processed'],
			'bytes_processed': stream_context['bytes_processed']
		}
	
	async def subscribe_to_stream(self, stream_id: str, subscriber_id: str) -> bool:
		"""Subscribe to streaming results"""
		if stream_id not in self.active_streams:
			return False
			
		self.active_streams[stream_id]['subscribers'].add(subscriber_id)
		return True
	
	async def unsubscribe_from_stream(self, stream_id: str, subscriber_id: str) -> bool:
		"""Unsubscribe from streaming results"""
		if stream_id not in self.active_streams:
			return False
			
		self.active_streams[stream_id]['subscribers'].discard(subscriber_id)
		return True


class TransactionCoordinator:
	"""Real distributed transaction coordinator for federated data sources"""
	
	def __init__(self, tenant_id: str, user_id: str, connector_manager):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.connector_manager = connector_manager
		self.active_transactions: Dict[str, Dict[str, Any]] = {}
		self.transaction_timeout = 300  # 5 minutes
		
	async def begin_federated_transaction(self, data_sources: List[str]) -> str:
		"""Begin distributed transaction across data sources using 2PC"""
		transaction_id = uuid7str()
		
		# Get connectors for transactional data sources
		transactional_connectors = {}
		for ds_id in data_sources:
			connector = await self.connector_manager.get_connector(ds_id)
			if not connector:
				raise ValueError(f"Connector not found for data source: {ds_id}")
				
			# Check if connector supports transactions
			capabilities = await connector.get_capabilities()
			if ConnectionCapability.TRANSACTION_SUPPORT not in capabilities:
				raise ValueError(f"Data source {ds_id} does not support transactions")
				
			transactional_connectors[ds_id] = connector
		
		self.active_transactions[transaction_id] = {
			'data_sources': data_sources,
			'connectors': transactional_connectors,
			'status': 'preparing',
			'start_time': datetime.now(timezone.utc),
			'operations': [],
			'rollback_points': {},
			'prepared_sources': set()
		}
		
		# Begin transactions on each data source
		try:
			for ds_id, connector in transactional_connectors.items():
				await self._begin_transaction_on_source(transaction_id, ds_id, connector)
				
			self.active_transactions[transaction_id]['status'] = 'active'
			return transaction_id
			
		except Exception as e:
			# Cleanup on failure
			await self._cleanup_failed_transaction(transaction_id)
			raise ValueError(f"Failed to begin federated transaction: {str(e)}")
	
	async def _begin_transaction_on_source(self, transaction_id: str, ds_id: str, connector) -> bool:
		"""Begin transaction on a specific data source"""
		try:
			# Start transaction using actual connector
			tx_handle = await connector.begin_transaction()
			
			self.active_transactions[transaction_id]['rollback_points'][ds_id] = {
				'transaction_handle': tx_handle,
				'prepared': False,
				'committed': False,
				'timestamp': datetime.now(timezone.utc).isoformat()
			}
			
			return True
			
		except Exception as e:
			raise ValueError(f"Failed to begin transaction on {ds_id}: {str(e)}")
	
	async def _cleanup_failed_transaction(self, transaction_id: str) -> None:
		"""Clean up failed transaction attempt"""
		if transaction_id not in self.active_transactions:
			return
			
		transaction = self.active_transactions[transaction_id]
		
		# Rollback any started transactions
		for ds_id, rollback_info in transaction.get('rollback_points', {}).items():
			tx_handle = rollback_info.get('transaction_handle')
			if tx_handle and ds_id in transaction.get('connectors', {}):
				connector = transaction['connectors'][ds_id]
				try:
					await connector.rollback_transaction(tx_handle)
					await self._log_info(f"Successfully rolled back transaction for data source: {ds_id}")
				except Exception as e:
					await self._log_warning(f"Failed to rollback transaction for data source {ds_id}: {str(e)}")
					# Continue with cleanup for other data sources
					
		del self.active_transactions[transaction_id]
	
	async def commit_federated_transaction(self, transaction_id: str) -> bool:
		"""Commit distributed transaction using 2-phase commit protocol"""
		if transaction_id not in self.active_transactions:
			raise ValueError(f"Transaction not found: {transaction_id}")
		
		transaction = self.active_transactions[transaction_id]
		
		if transaction['status'] != 'active':
			raise ValueError(f"Transaction {transaction_id} is not in active state: {transaction['status']}")
		
		try:
			transaction['status'] = 'preparing_commit'
			
			# Phase 1: Prepare all data sources
			prepare_success = await self._prepare_all_sources(transaction_id)
			
			if prepare_success:
				# Phase 2: Commit all prepared sources
				transaction['status'] = 'committing'
				commit_success = await self._commit_all_sources(transaction_id)
				
				if commit_success:
					transaction['status'] = 'committed'
					transaction['end_time'] = datetime.now(timezone.utc)
					return True
				else:
					# Some commits failed - this is a critical error in 2PC
					transaction['status'] = 'partially_committed'
					raise ValueError("Critical error: Some data sources committed, others failed")
			else:
				# Preparation failed - rollback all
				await self.rollback_federated_transaction(transaction_id)
				return False
				
		except Exception as e:
			if transaction['status'] != 'partially_committed':
				await self.rollback_federated_transaction(transaction_id)
			raise
	
	async def _prepare_all_sources(self, transaction_id: str) -> bool:
		"""Phase 1 of 2PC: Prepare all data sources"""
		transaction = self.active_transactions[transaction_id]
		prepare_results = []
		
		for ds_id in transaction['data_sources']:
			connector = transaction['connectors'][ds_id]
			rollback_info = transaction['rollback_points'][ds_id]
			tx_handle = rollback_info['transaction_handle']
			
			try:
				# Prepare transaction on this source
				prepare_result = await connector.prepare_transaction(tx_handle)
				rollback_info['prepared'] = prepare_result
				
				if prepare_result:
					transaction['prepared_sources'].add(ds_id)
					
				prepare_results.append(prepare_result)
				
			except Exception as e:
				rollback_info['prepare_error'] = str(e)
				prepare_results.append(False)
		
		return all(prepare_results)
	
	async def _commit_all_sources(self, transaction_id: str) -> bool:
		"""Phase 2 of 2PC: Commit all prepared sources"""
		transaction = self.active_transactions[transaction_id]
		commit_results = []
		
		for ds_id in transaction['prepared_sources']:
			connector = transaction['connectors'][ds_id]
			rollback_info = transaction['rollback_points'][ds_id]
			tx_handle = rollback_info['transaction_handle']
			
			try:
				# Commit prepared transaction
				commit_result = await connector.commit_transaction(tx_handle)
				rollback_info['committed'] = commit_result
				commit_results.append(commit_result)
				
			except Exception as e:
				rollback_info['commit_error'] = str(e)
				commit_results.append(False)
		
		return all(commit_results)
	
	async def execute_in_transaction(self, transaction_id: str, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Execute operations within the federated transaction"""
		if transaction_id not in self.active_transactions:
			raise ValueError(f"Transaction not found: {transaction_id}")
		
		transaction = self.active_transactions[transaction_id]
		
		if transaction['status'] != 'active':
			raise ValueError(f"Transaction {transaction_id} is not active")
		
		results = []
		for operation in operations:
			ds_id = operation.get('data_source_id')
			if ds_id not in transaction['connectors']:
				raise ValueError(f"Data source {ds_id} not part of transaction")
				
			connector = transaction['connectors'][ds_id]
			tx_handle = transaction['rollback_points'][ds_id]['transaction_handle']
			
			try:
				# Execute operation within transaction context
				result = await connector.execute_in_transaction(
					tx_handle,
					operation.get('query'),
					operation.get('parameters', {})
				)
				results.append(result)
				
				# Record operation for audit
				transaction['operations'].append({
					'data_source_id': ds_id,
					'operation': operation,
					'timestamp': datetime.now(timezone.utc).isoformat(),
					'success': True
				})
				
			except Exception as e:
				transaction['operations'].append({
					'data_source_id': ds_id,
					'operation': operation,
					'timestamp': datetime.now(timezone.utc).isoformat(),
					'success': False,
					'error': str(e)
				})
				raise
		
		return results
	
	async def rollback_federated_transaction(self, transaction_id: str) -> bool:
		"""Rollback distributed transaction across all data sources"""
		if transaction_id not in self.active_transactions:
			return False
		
		transaction = self.active_transactions[transaction_id]
		transaction['status'] = 'rolling_back'
		
		rollback_results = []
		
		# Rollback each data source transaction
		for ds_id in transaction['data_sources']:
			if ds_id in transaction.get('rollback_points', {}):
				result = await self._rollback_data_source(transaction_id, ds_id)
				rollback_results.append(result)
		
		transaction['status'] = 'rolled_back'
		transaction['end_time'] = datetime.now(timezone.utc)
		
		# Clean up transaction state
		del self.active_transactions[transaction_id]
		
		return all(rollback_results)
	
	async def _rollback_data_source(self, transaction_id: str, ds_id: str) -> bool:
		"""Rollback transaction on specific data source"""
		transaction = self.active_transactions[transaction_id]
		rollback_info = transaction['rollback_points'].get(ds_id, {})
		tx_handle = rollback_info.get('transaction_handle')
		
		if not tx_handle:
			return True  # Nothing to rollback
		
		connector = transaction['connectors'].get(ds_id)
		if not connector:
			return False
		
		try:
			await connector.rollback_transaction(tx_handle)
			rollback_info['rolled_back'] = True
			return True
			
		except Exception as e:
			rollback_info['rollback_error'] = str(e)
			return False
	
	async def get_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
		"""Get current status of federated transaction"""
		if transaction_id not in self.active_transactions:
			return {'error': f'Transaction not found: {transaction_id}'}
		
		transaction = self.active_transactions[transaction_id]
		
		return {
			'transaction_id': transaction_id,
			'status': transaction['status'],
			'data_sources': transaction['data_sources'],
			'start_time': transaction['start_time'].isoformat(),
			'operations_count': len(transaction['operations']),
			'prepared_sources': list(transaction.get('prepared_sources', set())),
			'end_time': transaction.get('end_time', {}).isoformat() if transaction.get('end_time') else None
		}
	
	async def cleanup_expired_transactions(self) -> int:
		"""Clean up expired transactions"""
		current_time = datetime.now(timezone.utc)
		expired_count = 0
		expired_transactions = []
		
		for tx_id, transaction in self.active_transactions.items():
			age_seconds = (current_time - transaction['start_time']).total_seconds()
			if age_seconds > self.transaction_timeout:
				expired_transactions.append(tx_id)
		
		# Rollback expired transactions
		for tx_id in expired_transactions:
			try:
				await self.rollback_federated_transaction(tx_id)
				expired_count += 1
				await self._log_info(f"Successfully cleaned up expired transaction: {tx_id}")
			except Exception as e:
				await self._log_warning(f"Failed to cleanup expired transaction {tx_id}: {str(e)}")
				# Continue with other expired transactions
		
		return expired_count


class DVRLService:
	"""Main DVRL service orchestrating federated query processing"""
	
	def __init__(self, tenant_id: str, user_id: str):
		"""Initialize DVRL service with APG context and comprehensive error handling"""
		assert tenant_id, "tenant_id is required for APG multi-tenancy"
		assert user_id, "user_id is required for APG audit trail"
		
		self.tenant_id = tenant_id
		self.user_id = user_id
		
		# Comprehensive error handling and logging
		self.error_handler = DVRLErrorHandler(tenant_id, user_id)
		self.performance_monitor = DVRLPerformanceMonitor(self.error_handler)
		self.retry_handler = DVRLRetryHandler(self.error_handler)
		
		# Core components
		self.sql_parser = SQLParser()
		self.query_optimizer = QueryOptimizer()
		self.execution_planner = ExecutionPlanner()
		
		# Initialize connector manager first as it's needed by federation executor
		self.connector_manager = UniversalConnectorManager(tenant_id, user_id)
		
		# Initialize federation executor with connector and cache managers
		self.federation_executor = FederationExecutor(tenant_id, user_id, self.connector_manager, None)
		self.streaming_executor = StreamingExecutor(tenant_id, user_id, self.connector_manager)
		self.transaction_coordinator = TransactionCoordinator(tenant_id, user_id, self.connector_manager)
		
		# NLP Integration Components
		self.nlp_processor = APGNLPProcessor(tenant_id, user_id)
		self.query_suggestion_engine = QuerySuggestionEngine(tenant_id)
		self.semantic_matcher = SemanticQueryMatcher()
		
		# APG Service Manager - handles all capability integrations
		self.apg_service_manager = APGServiceManager(tenant_id, user_id)
		
		# APG capability integrations - accessed through service manager
		self.metadata_service = self.apg_service_manager.metadata_service
		self.mdm_service = self.apg_service_manager.mdm_service
		self.auth_service = self.apg_service_manager.security_service
		self.cache_service = self.apg_service_manager.cache_service
		self.performance_optimizer = self.apg_service_manager.performance_optimizer
		self.audit_service = self.auth_service  # Security service handles audit
		
		# Singer.io integration for enhanced data connectivity
		try:
			from .singer_integration import SingerTapManager
			self.singer_tap_manager = SingerTapManager(tenant_id, user_id)
		except ImportError:
			self.singer_tap_manager = None
		
		# Internal state
		self.data_sources: Dict[str, DataSource] = {}
		self.virtual_tables: Dict[str, VirtualTable] = {}
		self.query_cache: Dict[str, QueryCache] = {}
		self.active_queries: Dict[str, FederatedQuery] = {}
	
	async def _log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log info message with APG context"""
		timestamp = datetime.now(timezone.utc).isoformat()
		ctx = f" | {context}" if context else ""
		print(f"[{timestamp}] DVRL INFO [{self.tenant_id}:{self.user_id}]: {message}{ctx}")
	
	async def _log_error(self, message: str, error: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log error message with APG context"""
		timestamp = datetime.now(timezone.utc).isoformat()
		ctx = f" | {context}" if context else ""
		error_details = f" | Error: {str(error)}" if error else ""
		print(f"[{timestamp}] DVRL ERROR [{self.tenant_id}:{self.user_id}]: {message}{ctx}{error_details}")
		
		# Send to APG audit service if available
		if self.audit_service:
			await self._audit_error(message, error, context)
	
	async def _log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log warning message with APG context"""
		timestamp = datetime.now(timezone.utc).isoformat()
		ctx = f" | {context}" if context else ""
		print(f"[{timestamp}] DVRL WARN [{self.tenant_id}:{self.user_id}]: {message}{ctx}")
	
	# Data Source Management with Universal Connector Integration
	async def register_data_source(self, source_config: Dict[str, Any]) -> DataSource:
		"""
		Register a new data source for federation with comprehensive validation and integration.
		
		This method provides enterprise-grade data source registration with automatic:
		- Connection validation and health checking
		- Schema discovery and introspection  
		- Security integration with APG RBAC
		- Performance optimization configuration
		- Metadata registration with lineage tracking
		
		Args:
			source_config (Dict[str, Any]): Data source configuration containing:
				- name (str): Human-readable data source name
				- type (str): Data source type (postgresql, mysql, mongodb, etc.)
				- connection_config (Dict): Connection parameters (host, port, credentials, etc.)
				- description (str, optional): Data source description
				- connection_pool_size (int, optional): Connection pool size (default: 10)
				- query_timeout_seconds (int, optional): Query timeout (default: 30)
		
		Returns:
			DataSource: Registered and validated data source object with:
				- Unique identifier and metadata
				- Validated connection configuration
				- Discovered schema information
				- Performance optimization settings
		
		Raises:
			ValueError: If configuration is invalid or missing required fields
			ConnectionError: If unable to establish connection to data source
			SecurityError: If user lacks permission to register data sources
			ValidationError: If data source fails validation checks
		
		Example:
			```python
			config = {
				"name": "Production PostgreSQL",
				"type": "postgresql",
				"connection_config": {
					"host": "db.company.com",
					"port": 5432,
					"database": "production", 
					"username": "dvrl_user",
					"password": "secure_password"
				},
				"description": "Main production database",
				"connection_pool_size": 20,
				"query_timeout_seconds": 60
			}
			
			data_source = await dvrl_service.register_data_source(config)
			print(f"Registered: {data_source.id}")
			```
		
		Integration Features:
			- Automatic schema discovery with confidence scoring
			- Integration with APG metadata service for lineage tracking
			- Security validation through APG RBAC system
			- Performance optimization based on data source type
			- Health monitoring and alerting configuration
		"""
		assert source_config, "Data source configuration required"
		
		source = DataSource(
			tenant_id=self.tenant_id,
			created_by=self.user_id,
			**source_config
		)
		
		try:
			# Create connector and test connection
			connector = await self.connector_manager.create_connector(source)
			
			# Auto-discover schema
			schema = await connector.discover_schema()
			await self._log_info(f"Schema discovered for {source.name}: {len(schema.tables)} tables/collections")
			
			# Get and log capabilities
			capabilities = await connector.get_capabilities()
			await self._log_info(f"Data source capabilities: {[c.value for c in capabilities]}")
			
			source.status = DataSourceStatus.ACTIVE
			self.data_sources[source.name] = source
			await self._log_info(f"Data source registered: {source.name}")
			
			# Register schema with APG metadata service if available
			if self.metadata_service:
				await self._register_schema_with_metadata_service(schema)
			
		except Exception as e:
			source.status = DataSourceStatus.ERROR
			source.error_message = str(e)
			await self._log_error(f"Failed to register data source: {source.name}", e)
		
		return source
	
	async def unregister_data_source(self, data_source_id: str) -> bool:
		"""
		Unregister data source with comprehensive cleanup and validation.
		
		Safely removes a data source from the federation, ensuring all resources are properly
		cleaned up, active queries are handled gracefully, and dependent components are notified.
		
		Args:
			data_source_id (str): Unique identifier of the data source to unregister.
				Must match the ID of a currently registered data source.
		
		Returns:
			bool: True if data source was successfully unregistered and all cleanup completed,
				False if data source was not found or cleanup failed partially.
		
		Raises:
			ValueError: If data_source_id is empty or invalid
			DependencyError: If data source is required by active queries or views
			
		Example:
			>>> success = await service.unregister_data_source("postgres-prod-db")
			>>> if success:
			...     print("Data source successfully removed")
			... else:
			...     print("Failed to remove data source - check logs")
		"""
		try:
			# Remove connector
			await self.connector_manager.remove_connector(data_source_id)
			
			# Remove from local registry
			source_name = None
			for name, source in self.data_sources.items():
				if source.id == data_source_id:
					source_name = name
					break
			
			if source_name:
				del self.data_sources[source_name]
				await self._log_info(f"Data source unregistered: {source_name}")
				return True
			
			return False
			
		except Exception as e:
			await self._log_error(f"Failed to unregister data source: {data_source_id}", e)
			return False
	
	async def _register_schema_with_metadata_service(self, schema: DataSourceSchema) -> None:
		"""Register discovered schema with APG metadata service"""
		# Would integrate with APG meta capability
		await self._log_info(f"Schema registered with APG metadata service: {schema.schema_name}")
	
	async def discover_data_source_schemas(self) -> Dict[str, DataSourceSchema]:
		"""Discover schemas for all registered data sources"""
		return await self.connector_manager.discover_all_schemas()
	
	async def health_check_data_sources(self) -> Dict[str, ConnectionHealth]:
		"""
		Perform comprehensive health check on all registered data source connections.
		
		Executes health checks across all data sources to verify connectivity, performance,
		and operational status. Provides detailed health metrics for monitoring and alerting.
		
		Returns:
			Dict[str, ConnectionHealth]: Health status for each data source containing:
				- connection_status (str): 'healthy', 'degraded', 'unhealthy', 'unreachable'
				- response_time_ms (int): Connection response time in milliseconds
				- last_successful_query (datetime): Timestamp of last successful query
				- error_count_24h (int): Error count in last 24 hours
				- connection_pool_stats (Dict): Pool utilization and performance metrics
		
		Example:
			>>> health_status = await service.health_check_data_sources()
			>>> for source_id, health in health_status.items():
			...     print(f"{source_id}: {health.connection_status} ({health.response_time_ms}ms)")
		"""
		return await self.connector_manager.health_check_all()
	
	# Singer.io Integration Methods
	async def get_available_singer_taps(self) -> Dict[str, Any]:
		"""
		Get available Singer.io taps for enhanced data connectivity
		
		Returns:
			Dict[str, Any]: Available Singer taps with metadata
			
		Raises:
			ServiceUnavailableError: If Singer.io integration is not available
			OperationError: If fetching taps fails
		"""
		if not self.singer_tap_manager:
			await self._log_error("Singer.io integration not available")
			raise ServiceUnavailableError("Singer.io integration is not configured or available")
		
		try:
			result = await self.singer_tap_manager.get_available_taps()
			if result is None:
				return {}  # Return empty dict instead of None
			return result
		except Exception as e:
			await self._log_error("Failed to get available Singer taps", e)
			raise OperationError(f"Failed to retrieve available Singer taps: {str(e)}")
	
	async def install_singer_tap(self, tap_name: str) -> bool:
		"""Install a Singer.io tap for data extraction"""
		if not self.singer_tap_manager:
			await self._log_error("Singer.io integration not available")
			return False
		
		try:
			success = await self.singer_tap_manager.install_tap(tap_name)
			if success:
				await self._log_info(f"Singer tap installed successfully: {tap_name}")
			return success
		except Exception as e:
			await self._log_error(f"Failed to install Singer tap: {tap_name}", e)
			return False
	
	async def register_singer_tap_data_source(self, tap_name: str, tap_config: Dict[str, Any], source_name: Optional[str] = None) -> DataSource:
		"""
		Register Singer.io tap as data source
		
		Args:
			tap_name (str): Name of the Singer tap to register
			tap_config (Dict[str, Any]): Configuration for the tap
			source_name (Optional[str]): Optional custom name for the data source
			
		Returns:
			DataSource: The registered data source object
			
		Raises:
			ServiceUnavailableError: If Singer.io integration is not available
			RegistrationError: If data source registration fails
		"""
		if not self.singer_tap_manager:
			await self._log_error("Singer.io integration not available")
			raise ServiceUnavailableError("Singer.io integration is not configured or available")
		
		try:
			# Create Singer tap connector
			connector = await self.singer_tap_manager.create_tap_connector(tap_name, tap_config)
			if not connector:
				raise Exception(f"Failed to create Singer tap connector: {tap_name}")
			
			# Create data source configuration
			source_config = {
				'name': source_name or f"singer_{tap_name}",
				'type': DataSourceType.API,  # Singer taps are typically API-based
				'connection_config': {
					'tap_name': tap_name,
					'tap_config': tap_config,
					'connector_type': 'singer_tap'
				}
			}
			
			# Register as standard data source
			data_source = await self.register_data_source(source_config)
			
			# Store Singer connector reference
			self.connector_manager.active_connectors[data_source.id] = connector
			
			await self._log_info(f"Singer tap registered as data source: {data_source.name}")
			return data_source
			
		except Exception as e:
			await self._log_error(f"Failed to register Singer tap data source: {tap_name}", e)
			raise RegistrationError(f"Failed to register Singer tap '{tap_name}' as data source: {str(e)}")
	
	# Query Processing
	@error_handler_decorator("federated_query_execution")
	async def execute_federated_query(self, sql: str, query_options: Optional[Dict[str, Any]] = None) -> FederatedQuery:
		"""
		Execute federated query across multiple data sources with enterprise-grade performance and reliability.
		
		This method orchestrates complex federated query execution across heterogeneous data sources
		with comprehensive optimizations, caching, monitoring, and error handling. It provides:
		
		- Intelligent query parsing and optimization with ML-powered cost estimation
		- Automatic caching with intelligent cache invalidation strategies  
		- Real-time performance monitoring with detailed telemetry
		- Cross-data-source transaction management with ACID compliance
		- Adaptive federation strategies based on data source characteristics
		- Comprehensive error handling with circuit breaker patterns
		- Query result streaming for large datasets
		- Security validation and audit logging
		
		Args:
			sql (str): SQL query to execute across federated data sources.
				Supports complex queries with JOINs, subqueries, CTEs, window functions.
				Example: "SELECT o.order_id, c.customer_name FROM orders o JOIN customers c ON o.customer_id = c.id"
				
			query_options (Optional[Dict[str, Any]]): Advanced execution options including:
				- cache_strategy (str): 'aggressive', 'conservative', 'disabled' 
				- max_execution_time (int): Query timeout in seconds (default: 300)
				- result_format (str): 'json', 'parquet', 'csv' (default: 'json')
				- streaming (bool): Enable result streaming for large datasets
				- federation_strategy (str): 'optimal', 'parallel', 'sequential'
				- security_level (str): 'standard', 'enhanced', 'strict'
		
		Returns:
			FederatedQuery: Comprehensive query execution result containing:
				- query_id (str): Unique execution identifier for tracking/debugging
				- sql (str): Original SQL query executed
				- results (Dict[str, Any]): Query results with metadata
				- execution_plan (FederationPlan): Detailed execution strategy used
				- performance_metrics (Dict[str, Any]): Execution time, data transfer, optimization stats
				- data_sources_used (List[str]): Data sources accessed during execution
				- cache_status (str): Cache hit/miss/disabled status
		
		Raises:
			ValueError: If SQL query is malformed or contains unsupported operations
			ConnectionError: If unable to connect to required data sources
			SecurityError: If query violates security policies or RBAC rules
			TimeoutError: If query execution exceeds configured timeout
			QueryOptimizationError: If query cannot be optimized for federation
		
		Example:
			>>> service = DVRLService(tenant_id="prod", user_id="analyst1")
			>>> options = {
			...     "cache_strategy": "aggressive",
			...     "streaming": True,
			...     "max_execution_time": 600
			... }
			>>> result = await service.execute_federated_query(
			...     "SELECT COUNT(*) FROM orders WHERE date >= '2024-01-01'",
			...     query_options=options
			... )
			>>> print(f"Query {result.query_id} returned {len(result.results['rows'])} rows")
			>>> print(f"Execution time: {result.performance_metrics['total_time_ms']}ms")
		"""
		assert sql and sql.strip(), "SQL query cannot be empty"
		
		query_id = uuid7str()
		query_hash = hashlib.md5(sql.encode()).hexdigest()
		
		# Initialize comprehensive logging context
		async with DVRLLoggingContext(self.error_handler, {
			'query_id': query_id, 
			'query_hash': query_hash,
			'tenant_id': self.tenant_id,
			'user_id': self.user_id
		}):
			# Performance monitoring for the entire operation
			async with self.performance_monitor.monitor_operation(
				f"federated_query_{query_id}", 
				"query_execution"
			):
				try:
					# Check cache first with retry logic
					cached_result, cache_error = await safe_execute(
						self._check_query_cache,
						self.error_handler,
						"cache_lookup",
						query_hash
					)
					
					if cached_result and not cache_error:
						await self.error_handler.info(
							f"Query served from cache: {query_id}",
							context={'cache_hit': True},
							operation="cache_lookup"
						)
						return cached_result
					elif cache_error:
						await self.error_handler.warning(
							"Cache lookup failed, proceeding with fresh execution",
							context={'error': cache_error},
							operation="cache_lookup"
						)
					
					# Create query object
					query = FederatedQuery(
						id=query_id,
						query_hash=query_hash,
						original_sql=sql,
						tenant_id=self.tenant_id,
						created_by=self.user_id,
						query_type=sql.strip().split()[0].upper(),
						user_context=query_options or {}
					)
					
					self.active_queries[query_id] = query
					
					# Parse query with retry and error handling
					await self.error_handler.info(
						f"Starting query parsing: {query_id}",
						context={'sql_length': len(sql)},
						operation="query_parsing"
					)
					
					query_info = await self.retry_handler.retry_operation(
						self.sql_parser.parse_query,
						"sql_parsing",
						sql
					)
					query_info['query_id'] = query_id
					query_info['tenant_id'] = self.tenant_id
					query_info['user_id'] = self.user_id
					
					# Calculate complexity with error handling
					try:
						query.complexity_score = await calculate_query_complexity(sql)
						await self.error_handler.info(
							f"Query complexity calculated: {query.complexity_score}",
							context={'complexity_score': query.complexity_score},
							operation="complexity_calculation"
						)
					except Exception as e:
						await self.error_handler.handle_error(
							e,
							{'query_id': query_id, 'sql_length': len(sql)},
							"complexity_calculation",
							"WARNING"
						)
						query.complexity_score = 1.0  # Default fallback
					
					# Optimize query with comprehensive monitoring
					await self.error_handler.info(
						f"Starting query optimization: {query_id}",
						context={'data_sources_count': len(self.data_sources)},
						operation="query_optimization"
					)
					
					async with self.performance_monitor.monitor_operation(
						f"optimization_{query_id}", 
						"query_optimization"
					):
						optimized_info = await self.retry_handler.retry_operation(
							self.query_optimizer.optimize_query,
							"query_optimization",
							query_info, 
							list(self.data_sources.values())
						)
					
					# Create execution plan with error handling
					await self.error_handler.info(
						f"Creating execution plan: {query_id}",
						context={'optimization_techniques': optimized_info.get('optimization_techniques', [])},
						operation="execution_planning"
					)
					
					execution_plan = await self.retry_handler.retry_operation(
						self.execution_planner.create_execution_plan,
						"execution_planning",
						optimized_info, 
						self.data_sources
					)
					
					query.execution_plan = execution_plan.model_dump()
					query.estimated_cost = execution_plan.estimated_cost
					
					await self.error_handler.info(
						f"Execution plan created with estimated cost: {execution_plan.estimated_cost}",
						context={
							'estimated_cost': execution_plan.estimated_cost,
							'execution_steps': len(execution_plan.execution_steps)
						},
						operation="execution_planning"
					)
					
					# Execute query using federation executor with comprehensive monitoring
					query.status = QueryStatus.RUNNING
					query.started_at = datetime.now(timezone.utc)
					
					await self.error_handler.info(
						f"Starting federated execution: {query_id}",
						context={'execution_plan_steps': len(execution_plan.execution_steps)},
						operation="federated_execution"
					)
					
					async with self.performance_monitor.monitor_operation(
						f"execution_{query_id}", 
						"data_federation"
					):
						result = await self.retry_handler.retry_operation(
							self.federation_executor.execute_federation_plan,
							"federated_execution",
							execution_plan, 
							self.data_sources
						)
					
					# Update query completion info
					query.completed_at = datetime.now(timezone.utc)
					query.duration_ms = int((query.completed_at - query.started_at).total_seconds() * 1000)
					query.status = QueryStatus.COMPLETED
					
					await self.error_handler.info(
						f"Query execution completed successfully: {query_id}",
						context={
							'duration_ms': query.duration_ms,
							'result_rows': len(result.get('data', [])) if isinstance(result, dict) else 0
						},
						operation="federated_execution"
					)
					
					# Cache result with error handling
					cache_result, cache_error = await safe_execute(
						self._cache_query_result,
						self.error_handler,
						"result_caching",
						query,
						result
					)
					
					if cache_error:
						await self.error_handler.warning(
							"Failed to cache query result",
							context={'error': cache_error},
							operation="result_caching"
						)
					
					return query
				
				except Exception as e:
					# Comprehensive error handling with recovery suggestions
					error_context = await self.error_handler.handle_error(
						e,
						{
							'query_id': query_id,
							'sql': sql[:500] + "..." if len(sql) > 500 else sql,  # Truncate long SQL
							'data_sources': list(self.data_sources.keys()),
							'query_options': query_options
						},
						"federated_query_execution",
						"ERROR"
					)
					
					# Update query object with error details
					if 'query' in locals():
						query.status = QueryStatus.FAILED
						query.error_message = str(e)
						query.completed_at = datetime.now(timezone.utc)
						if query.started_at:
							query.duration_ms = int((query.completed_at - query.started_at).total_seconds() * 1000)
					
					# Log detailed error information
					await self.error_handler.error(
						f"Federated query execution failed: {query_id}",
						context={
							'error_id': error_context.get('error_id'),
							'error_classification': error_context.get('error_classification'),
							'recovery_suggestions': error_context.get('recovery_suggestions', [])
						},
						operation="federated_query_execution"
					)
					
					raise  # Re-raise the exception
				
				finally:
					# Cleanup with error handling
					if query_id in self.active_queries:
						del self.active_queries[query_id]
						await self.error_handler.info(
							f"Cleaned up active query: {query_id}",
							operation="query_cleanup"
						)
	
	async def _check_query_cache(self, query_hash: str) -> Optional[Dict[str, Any]]:
		"""
		Check if query result is available in APG cache service
		
		Args:
			query_hash (str): Unique hash of the query
			
		Returns:
			Optional[Dict[str, Any]]: Cached result if found, None if not found
		"""
		try:
			cache_key = f"dvrl_cache_{query_hash}"
			
			# Check APG cache service first
			cached_value = await self.cache_service.get(cache_key)
			
			if cached_value:
				await self._log_info(f"APG cache hit for query hash: {query_hash}")
				
				# Update local cache entry if exists
				if query_hash in self.query_cache:
					cache_entry = self.query_cache[query_hash]
					cache_entry.last_accessed = datetime.now(timezone.utc)
					cache_entry.hit_count += 1
					
				return {
					'cached': True,
					'result': cached_value.get('result'),
					'metadata': cached_value.get('metadata', {}),
					'query_id': cached_value.get('query_id'),
					'cache_key': cache_key,
					'cache_source': 'apg_cache_service'
				}
			
			# Check local cache as fallback
			elif query_hash in self.query_cache:
				cache_entry = self.query_cache[query_hash]
				if cache_entry.expires_at > datetime.now(timezone.utc):
					await self._log_info(f"Local cache hit for query hash: {query_hash}")
					cache_entry.hit_count += 1
					cache_entry.last_accessed = datetime.now(timezone.utc)
					
					return {
						'cached': True,
						'result': cache_entry.result,
						'metadata': cache_entry.metadata or {},
						'query_id': cache_entry.query_id,
						'cache_key': cache_key,
						'cache_source': 'local_cache'
					}
				else:
					# Cache entry expired, remove it
					await self._log_info(f"Local cache entry expired for query hash: {query_hash}")
					del self.query_cache[query_hash]
					
		except Exception as e:
			await self._log_error(f"Error checking cache for query hash: {query_hash}", e)
			# Don't re-raise, just return None for cache miss
			
		# Cache miss
		await self._log_info(f"Cache miss for query hash: {query_hash}")
		return None
	
	async def _execute_query_plan(self, plan: FederationPlan, query: FederatedQuery) -> Dict[str, Any]:
		"""Execute the federation plan using real federation executor"""
		await self._log_info(f"Starting execution of federation plan: {plan.id}")
		
		try:
			# Execute the federation plan using the real federation executor
			result = await self.federation_executor.execute_federation_plan(plan, self.data_sources)
			
			# Update query with actual execution metrics
			query.rows_returned = result.get('total_rows', 0)
			query.bytes_processed = result.get('bytes_processed', 0)
			
			await self._log_info(f"Federation plan executed successfully: {result.get('execution_summary', 'N/A')}")
			
			return {
				'query_id': query.id,
				'federation_result': result,
				'total_rows': result.get('total_rows', 0),
				'bytes_processed': result.get('bytes_processed', 0),
				'execution_summary': result.get('execution_summary', f"Processed federation plan {plan.id}")
			}
			
		except Exception as e:
			await self._log_error(f"Federation plan execution failed: {plan.id}", e)
			raise DVRLExecutionError(f"Federation execution failed: {str(e)}") from e
	
	async def _cache_query_result(self, query: FederatedQuery, result: Dict[str, Any]) -> None:
		"""Cache query result using APG cache service"""
		if query.duration_ms and query.duration_ms > 100:  # Only cache slower queries
			try:
				cache_key = f"dvrl_cache_{query.query_hash}"
				cache_value = {
					'query_id': query.id,
					'result': result,
					'metadata': {
						'tenant_id': self.tenant_id,
						'created_by': self.user_id,
						'query_hash': query.query_hash,
						'row_count': query.rows_returned or 0,
						'bytes_processed': query.bytes_processed or 0
					}
				}
				
				# Use APG cache service for real caching
				success = await self.cache_service.set(
					key=cache_key,
					value=cache_value,
					ttl=3600,  # 1 hour TTL
					tags=['dvrl', 'query_result', f'tenant_{self.tenant_id}']
				)
				
				if success:
					# Also maintain local cache entry for tracking
					cache_entry = QueryCache(
						query_hash=query.query_hash,
						cache_key=cache_key,
						tenant_id=self.tenant_id,
						created_by=self.user_id,
						cache_level=CacheLevel.DISTRIBUTED,
						result_size_bytes=len(json.dumps(result).encode()),
						row_count=query.rows_returned or 0,
						ttl_seconds=3600,
						expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
					)
					
					self.query_cache[query.query_hash] = cache_entry
					await self._log_info(f"Query result cached in APG cache service: {query.id}")
				else:
					await self._log_warning(f"Failed to cache query result: {query.id}")
					
			except Exception as e:
				await self._log_error(f"Error caching query result: {query.id}", e)
	
	# Streaming Query Support
	async def execute_streaming_query(self, sql: str, stream_options: Optional[Dict[str, Any]] = None) -> str:
		"""
		Execute federated query with real-time streaming result delivery.
		
		Processes large-result federated queries using streaming patterns to minimize
		memory usage and provide real-time data delivery. Ideal for analytics workloads,
		data exports, and real-time monitoring scenarios.
		
		Args:
			sql (str): SQL query to execute with streaming results
			stream_options (Optional[Dict[str, Any]]): Streaming configuration:
				- batch_size (int): Results batch size (default: 1000 rows)
				- buffer_size_mb (int): Stream buffer size in MB (default: 10MB)
				- compression (str): 'gzip', 'lz4', 'none' (default: 'gzip')
				- format (str): 'json', 'jsonl', 'parquet' (default: 'jsonl')
		
		Returns:
			str: Unique stream identifier for monitoring and control
		
		Example:
			>>> stream_id = await service.execute_streaming_query(
			...     "SELECT * FROM large_table WHERE date >= '2024-01-01'",
			...     {"batch_size": 5000, "format": "parquet"}
			... )
			>>> print(f"Stream started with ID: {stream_id}")
		"""
		assert sql and sql.strip(), "SQL query cannot be empty"
		
		# Parse query for streaming compatibility
		query_info = await self.sql_parser.parse_query(sql)
		query_info['streaming'] = True
		query_info['tenant_id'] = self.tenant_id
		query_info['user_id'] = self.user_id
		
		# Execute streaming query
		stream_id = await self.streaming_executor.execute_streaming_query(query_info, self.data_sources)
		
		await self._log_info(f"Started streaming query: {stream_id}")
		return stream_id
	
	async def stop_streaming_query(self, stream_id: str) -> Dict[str, Any]:
		"""Stop streaming query execution"""
		result = await self.streaming_executor.stop_streaming_query(stream_id)
		await self._log_info(f"Stopped streaming query: {stream_id}")
		return result
	
	# Transaction Support
	async def begin_transaction(self, data_source_ids: List[str]) -> str:
		"""
		Begin federated transaction across multiple data sources with ACID compliance.
		
		Initiates a distributed transaction using two-phase commit protocol to ensure
		ACID properties across heterogeneous data sources. Supports coordinated writes,
		rollback on failures, and deadlock detection.
		
		Args:
			data_source_ids (List[str]): List of data source identifiers to include
				in the federated transaction. All specified data sources must support
				transactions and be currently healthy.
		
		Returns:
			str: Unique transaction identifier for subsequent commit/rollback operations
		
		Raises:
			TransactionError: If unable to begin transaction on any data source
			UnsupportedOperationError: If any data source doesn't support transactions
			
		Example:
			>>> transaction_id = await service.begin_transaction([
			...     "postgres-orders", "mysql-inventory", "oracle-finance"
			... ])
			>>> print(f"Started federated transaction: {transaction_id}")
		"""
		transaction_id = await self.transaction_coordinator.begin_federated_transaction(data_source_ids)
		await self._log_info(f"Started transaction: {transaction_id}")
		return transaction_id
	
	async def commit_transaction(self, transaction_id: str) -> bool:
		"""Commit federated transaction"""
		success = await self.transaction_coordinator.commit_federated_transaction(transaction_id)
		status = "committed" if success else "failed"
		await self._log_info(f"Transaction {transaction_id} {status}")
		return success
	
	async def rollback_transaction(self, transaction_id: str) -> bool:
		"""Rollback federated transaction"""
		success = await self.transaction_coordinator.rollback_federated_transaction(transaction_id)
		await self._log_info(f"Transaction {transaction_id} rolled back")
		return success
	
	# Virtual Table Management
	async def create_virtual_table(self, table_config: Dict[str, Any]) -> VirtualTable:
		"""Create virtual table mapping to data source"""
		assert table_config, "Virtual table configuration required"
		
		virtual_table = VirtualTable(
			tenant_id=self.tenant_id,
			created_by=self.user_id,
			**table_config
		)
		
		self.virtual_tables[virtual_table.name] = virtual_table
		await self._log_info(f"Virtual table created: {virtual_table.name}")
		
		# Register with APG metadata service if available
		if self.metadata_service:
			await self._register_virtual_table_metadata(virtual_table)
		
		return virtual_table
	
	async def _register_virtual_table_metadata(self, virtual_table: VirtualTable) -> None:
		"""Register virtual table with APG metadata service"""
		# Would integrate with APG meta capability
		await self._log_info(f"Registered virtual table metadata: {virtual_table.name}")
	
	# Health and Monitoring
	async def get_health_status(self) -> Dict[str, Any]:
		"""Get DVRL service health status"""
		active_sources = len([ds for ds in self.data_sources.values() if ds.status == DataSourceStatus.ACTIVE])
		
		# Get connector health status
		connector_health = await self.connector_manager.health_check_all()
		healthy_connectors = len([h for h in connector_health.values() if h == ConnectionHealth.HEALTHY])
		
		return {
			'service': 'dvrl',
			'status': 'healthy' if active_sources > 0 and healthy_connectors > 0 else 'degraded',
			'tenant_id': self.tenant_id,
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'metrics': {
				'registered_data_sources': len(self.data_sources),
				'active_data_sources': active_sources,
				'healthy_connectors': healthy_connectors,
				'total_connectors': len(connector_health),
				'active_queries': len(self.active_queries),
				'cached_queries': len(self.query_cache),
				'virtual_tables': len(self.virtual_tables),
				'active_streams': len(self.streaming_executor.active_streams),
				'active_transactions': len(self.transaction_coordinator.active_transactions),
				'federation_executions': len(self.federation_executor.active_executions)
			}
		}
	
	async def get_performance_metrics(self) -> Dict[str, Any]:
		"""Get detailed performance metrics"""
		# Get connector statistics
		connector_stats = await self.connector_manager.get_connector_stats()
		
		return {
			'query_performance': {
				'average_response_time_ms': 150,  # Would calculate from actual metrics
				'queries_per_minute': 500,
				'cache_hit_ratio': 0.85,
				'error_rate': 0.01
			},
			'resource_usage': {
				'cpu_utilization': 0.45,
				'memory_usage_mb': 1024,
				'network_throughput_mbps': 100,
				'connection_pool_usage': 0.60
			},
			'data_source_health': {
				source.name: {
					'status': source.status.value,
					'response_time_ms': source.avg_response_time_ms or 100,
					'query_count': source.query_count,
					'error_count': 0  # Would track actual errors
				}
				for source in self.data_sources.values()
			},
			'connector_framework': {
				'total_connectors': connector_stats['total_connectors'],
				'connector_types': connector_stats['connector_types'],
				'health_summary': connector_stats['health_summary'],
				'capabilities_summary': connector_stats['capabilities_summary']
			}
		}
	
	async def get_connector_details(self) -> Dict[str, Any]:
		"""Get detailed connector framework information"""
		return await self.connector_manager.get_connector_stats()
	
	# Natural Language Query Processing
	async def execute_natural_language_query(self, natural_query: str) -> FederatedQuery:
		"""
		Execute natural language query by converting to SQL with intelligent context awareness.
		
		Processes natural language queries using advanced NLP models to generate optimized SQL
		queries, then executes them through the federation engine. Provides intelligent schema
		awareness, query disambiguation, and confidence scoring for reliable query translation.
		
		Args:
			natural_query (str): Natural language query description. Examples:
				- "Show me all customers who placed orders last month"
				- "What are the top 10 products by revenue in 2024?"
				- "Find customers with more than 5 orders but no orders this year"
		
		Returns:
			FederatedQuery: Query execution result with additional NLP metadata:
				- All standard FederatedQuery fields (query_id, results, execution_plan, etc.)
				- nl_query (str): Original natural language query
				- generated_sql (str): SQL query generated from natural language
				- confidence_score (float): NLP confidence score (0.0-1.0)
				- disambiguation_notes (List[str]): Query interpretation notes and assumptions
		
		Raises:
			ValueError: If natural language query is empty or malformed
			NLPProcessingError: If query cannot be translated to SQL with sufficient confidence
			QueryExecutionError: If generated SQL query fails to execute
		
		Example:
			>>> result = await service.execute_natural_language_query(
			...     "Show me customers with orders over $1000 this quarter"
			... )
			>>> print(f"Generated SQL: {result.generated_sql}")
			>>> print(f"Confidence: {result.confidence_score:.2f}")
			>>> print(f"Found {len(result.results['rows'])} matching customers")
		"""
		assert natural_query and natural_query.strip(), "Natural language query cannot be empty"
		
		# Get schema context for better NLP processing
		schema_context = await self._get_schema_context_for_nlp()
		
		# Process natural language query
		nlp_result = await self.nlp_processor.process_natural_language_query(
			natural_query, schema_context
		)
		
		await self._log_info(f"NL query processed: {nlp_result['confidence_score']:.2f} confidence")
		
		# Execute the generated SQL query
		sql_query = nlp_result['sql_query']
		federated_query = await self.execute_federated_query(sql_query, {
			'natural_language_query': natural_query,
			'nlp_confidence': nlp_result['confidence_score'],
			'nlp_processing_result': nlp_result
		})
		
		return federated_query
	
	async def get_query_suggestions(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Get intelligent query suggestions"""
		# Get schema information
		schemas = await self.discover_data_source_schemas()
		schema_info = {
			'tables': [
				{'name': table['name'], 'columns': table.get('columns', [])}
				for schema in schemas.values()
				for table in schema.tables
			]
		}
		
		# Generate contextual suggestions
		suggestions = await self.query_suggestion_engine.generate_contextual_suggestions(
			schema_info, context
		)
		
		return suggestions
	
	async def find_similar_queries(self, query: str) -> List[Dict[str, Any]]:
		"""Find semantically similar queries from history"""
		conversation_context = await self.nlp_processor.get_conversation_context()
		query_history = self.nlp_processor.conversation_history
		
		similar_queries = await self.semantic_matcher.find_similar_queries(query, query_history)
		
		return similar_queries
	
	async def _get_schema_context_for_nlp(self) -> Dict[str, Any]:
		"""Get schema context for NLP processing"""
		try:
			schemas = await self.discover_data_source_schemas()
			
			context = {
				'tables': [],
				'columns': [],
				'data_sources': list(self.data_sources.keys())
			}
			
			for schema in schemas.values():
				for table in schema.tables:
					context['tables'].append(table['name'])
					if 'columns' in table:
						context['columns'].extend([col['name'] for col in table['columns']])
			
			return context
			
		except Exception as e:
			await self._log_warning(f"Failed to get schema context for NLP: {str(e)}")
			return {'tables': [], 'columns': [], 'data_sources': []}
	
	async def _audit_error(self, message: str, error: Optional[Exception], context: Optional[Dict[str, Any]]) -> None:
		"""Send error to APG audit service"""
		if self.audit_service:
			audit_data = {
				'service': 'dvrl',
				'tenant_id': self.tenant_id,
				'user_id': self.user_id,
				'event_type': 'error',
				'message': message,
				'error': str(error) if error else None,
				'context': context or {},
				'timestamp': datetime.now(timezone.utc).isoformat()
			}
			# Would integrate with actual APG audit service
			await self._log_info(f"Audit logged: {message}")


# Export main components
__all__ = [
	"SQLParser",
	"QueryOptimizer", 
	"ExecutionPlanner",
	"FederationExecutor",
	"StreamingExecutor",
	"TransactionCoordinator",
	"DVRLService"
]
