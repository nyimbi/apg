"""
Comprehensive tests for Data Lineage Engine
Tests graph algorithms, sensitive data detection, and visualization data generation

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
from unittest.mock import Mock, patch
import networkx as nx

from ...lineage_engine import (
    DataLineageEngine, LineageNodeInfo, LineageEdgeInfo,
    LineageRelationshipType, SensitivityLevel
)
from ...sqlalchemy_models import (
    CnConnection, CnDataFlow, CnLineageNode, CnLineageEdge,
    ConnectionType, LineageNodeType
)


class TestDataLineageEngine:
	"""Test DataLineageEngine functionality"""

	def test_initialization(self, lineage_engine):
		"""Test lineage engine initializes properly"""
		assert lineage_engine.graph is not None
		assert isinstance(lineage_engine.graph, nx.DiGraph)
		assert len(lineage_engine.node_cache) == 0
		assert len(lineage_engine.edge_cache) == 0
		assert len(lineage_engine.sensitive_patterns) > 0

	def test_generate_node_id(self, lineage_engine):
		"""Test node ID generation"""
		# Test with field
		node_id = lineage_engine._generate_node_id('conn1', 'public', 'users', 'email')
		assert node_id == 'conn1.public.users.email'

		# Test without field
		table_id = lineage_engine._generate_node_id('conn1', 'public', 'users')
		assert table_id == 'conn1.public.users'

	def test_classify_sensitivity_pii_patterns(self, lineage_engine):
		"""Test PII sensitivity classification"""
		test_cases = [
			('user_email', None, None, SensitivityLevel.PII),
			('first_name', None, None, SensitivityLevel.PII),
			('ssn_number', None, None, SensitivityLevel.PII),
			('phone_mobile', None, None, SensitivityLevel.PII),
			('regular_field', None, None, SensitivityLevel.INTERNAL),
		]

		for field_name, data_type, sample_data, expected_level in test_cases:
			sensitivity, classification = lineage_engine._classify_sensitivity(
				field_name, data_type, sample_data
			)
			assert sensitivity == expected_level

	def test_classify_sensitivity_sample_data(self, lineage_engine):
		"""Test sensitivity classification using sample data"""
		# Email detection
		email_samples = ['user@example.com', 'admin@test.com', 'test@domain.org']
		sensitivity, classification = lineage_engine._classify_sensitivity(
			'contact', 'varchar', email_samples
		)
		assert sensitivity == SensitivityLevel.PII
		assert 'Email format detected' in classification

		# Phone detection
		phone_samples = ['555-123-4567', '(555) 234-5678', '+1-555-345-6789']
		sensitivity, classification = lineage_engine._classify_sensitivity(
			'contact_info', 'varchar', phone_samples
		)
		assert sensitivity == SensitivityLevel.PII
		assert 'Phone number format' in classification

		# SSN detection
		ssn_samples = ['123-45-6789', '987654321', '555-44-3333']
		sensitivity, classification = lineage_engine._classify_sensitivity(
			'identifier', 'varchar', ssn_samples
		)
		assert sensitivity == SensitivityLevel.PII
		assert 'SSN format' in classification

	async def test_discover_connection_schema(self, lineage_engine, sample_connection):
		"""Test connection schema discovery"""
		result = await lineage_engine.discover_connection_schema(sample_connection)

		assert 'connection_id' in result
		assert result['connection_id'] == sample_connection.id
		assert result['nodes_created'] >= 1  # At least connection node

		# Check connection node was created
		conn_node_id = f"conn_{sample_connection.id}"
		assert conn_node_id in lineage_engine.node_cache

		conn_node = lineage_engine.node_cache[conn_node_id]
		assert conn_node.name == sample_connection.name
		assert conn_node.node_type == LineageNodeType.CONNECTION

	async def test_discover_singer_schema(self, lineage_engine, sample_connection):
		"""Test Singer.io schema discovery"""
		# Test the internal Singer discovery method
		result = await lineage_engine._discover_singer_schema(sample_connection)

		assert 'tables_discovered' in result
		assert 'fields_discovered' in result
		assert 'sensitive_fields' in result
		assert 'nodes_created' in result

		# Should discover mock catalog tables
		assert result['tables_discovered'] == 2  # users and orders
		assert result['fields_discovered'] > 0
		assert result['sensitive_fields'] > 0  # Email field should be detected

		# Check nodes were created in cache
		assert len(lineage_engine.node_cache) > 0

		# Check for table nodes
		table_nodes = [
			node for node in lineage_engine.node_cache.values()
			if node.node_type == LineageNodeType.TABLE
		]
		assert len(table_nodes) == 2

		# Check for field nodes
		field_nodes = [
			node for node in lineage_engine.node_cache.values()
			if node.node_type == LineageNodeType.FIELD
		]
		assert len(field_nodes) > 0

		# Check sensitive fields
		sensitive_fields = [
			node for node in field_nodes
			if node.sensitivity == SensitivityLevel.PII
		]
		assert len(sensitive_fields) > 0

	async def test_track_flow_lineage(self, lineage_engine, sample_flow):
		"""Test flow lineage tracking"""
		result = await lineage_engine.track_flow_lineage(sample_flow)

		assert 'flow_id' in result
		assert result['flow_id'] == sample_flow.id
		assert 'relationships_created' in result
		assert 'transformations_tracked' in result

		# Check flow node was created
		flow_node_id = f"flow_{sample_flow.id}"
		assert flow_node_id in lineage_engine.node_cache

		flow_node = lineage_engine.node_cache[flow_node_id]
		assert flow_node.name == sample_flow.name
		assert flow_node.node_type == LineageNodeType.FLOW

	async def test_create_or_update_node(self, lineage_engine, db_session):
		"""Test node creation and updates"""
		lineage_engine.db_session = db_session

		node_info = LineageNodeInfo(
			id='test_node_1',
			name='Test Node',
			node_type=LineageNodeType.TABLE,
			connection_id='conn_123',
			schema_name='public',
			table_name='test_table',
			sensitivity=SensitivityLevel.INTERNAL,
			metadata={'test': 'data'}
		)

		await lineage_engine._create_or_update_node(node_info)

		# Check cache
		assert 'test_node_1' in lineage_engine.node_cache
		assert lineage_engine.node_cache['test_node_1'].name == 'Test Node'

		# Check graph
		assert lineage_engine.graph.has_node('test_node_1')

		# Check database
		db_node = db_session.query(CnLineageNode).filter(
			CnLineageNode.id == 'test_node_1'
		).first()
		assert db_node is not None
		assert db_node.name == 'Test Node'
		assert db_node.table_name == 'test_table'

	async def test_create_relationship(self, lineage_engine, db_session):
		"""Test relationship creation"""
		lineage_engine.db_session = db_session

		# Create source and target nodes first
		source_node = LineageNodeInfo(
			id='source_node',
			name='Source',
			node_type=LineageNodeType.TABLE
		)
		target_node = LineageNodeInfo(
			id='target_node',
			name='Target',
			node_type=LineageNodeType.TABLE
		)

		await lineage_engine._create_or_update_node(source_node)
		await lineage_engine._create_or_update_node(target_node)

		# Create relationship
		await lineage_engine._create_relationship(
			'source_node',
			'target_node',
			LineageRelationshipType.DERIVES_FROM,
			transformation_logic='SELECT * FROM source',
			confidence_score=0.95,
			metadata={'test': 'relationship'}
		)

		# Check cache
		edge_id = 'source_node__target_node__derives_from'
		assert edge_id in lineage_engine.edge_cache

		edge_info = lineage_engine.edge_cache[edge_id]
		assert edge_info.relationship_type == LineageRelationshipType.DERIVES_FROM
		assert edge_info.confidence_score == 0.95

		# Check graph
		assert lineage_engine.graph.has_edge('source_node', 'target_node')

		# Check database
		db_edge = db_session.query(CnLineageEdge).filter(
			CnLineageEdge.id == edge_id
		).first()
		assert db_edge is not None
		assert db_edge.relationship_type == 'derives_from'
		assert db_edge.confidence_score == 0.95

	def test_get_lineage_visualization_full(self, lineage_engine):
		"""Test full lineage visualization data generation"""
		# Add test nodes and edges
		test_nodes = [
			LineageNodeInfo(
				id='conn_1', name='Database', node_type=LineageNodeType.CONNECTION,
				sensitivity=SensitivityLevel.INTERNAL
			),
			LineageNodeInfo(
				id='table_1', name='users', node_type=LineageNodeType.TABLE,
				sensitivity=SensitivityLevel.INTERNAL
			),
			LineageNodeInfo(
				id='field_1', name='email', node_type=LineageNodeType.FIELD,
				sensitivity=SensitivityLevel.PII, pii_classification='email'
			)
		]

		test_edges = [
			LineageEdgeInfo(
				id='edge_1', source_node_id='conn_1', target_node_id='table_1',
				relationship_type=LineageRelationshipType.CONTAINS
			),
			LineageEdgeInfo(
				id='edge_2', source_node_id='table_1', target_node_id='field_1',
				relationship_type=LineageRelationshipType.CONTAINS
			)
		]

		# Add to caches
		for node in test_nodes:
			lineage_engine.node_cache[node.id] = node
		for edge in test_edges:
			lineage_engine.edge_cache[edge.id] = edge
			lineage_engine.graph.add_edge(edge.source_node_id, edge.target_node_id)

		# Get visualization data
		viz_data = lineage_engine.get_lineage_visualization()

		assert 'nodes' in viz_data
		assert 'edges' in viz_data
		assert 'summary' in viz_data

		assert len(viz_data['nodes']) == 3
		assert len(viz_data['edges']) == 2

		# Check summary
		summary = viz_data['summary']
		assert summary['total_nodes'] == 3
		assert summary['total_edges'] == 2
		assert summary['sensitive_entities'] == 1

		# Check node format
		email_node = next(node for node in viz_data['nodes'] if node['id'] == 'field_1')
		assert email_node['label'] == 'email'
		assert email_node['type'] == 'field'
		assert email_node['metadata']['sensitive'] is True
		assert email_node['metadata']['pii'] is True

	def test_get_upstream_lineage(self, lineage_engine):
		"""Test upstream lineage traversal"""
		# Create test graph: conn -> table -> field
		lineage_engine.graph.add_edge('conn_1', 'table_1')
		lineage_engine.graph.add_edge('table_1', 'field_1')

		# Add node info to cache
		lineage_engine.node_cache['conn_1'] = LineageNodeInfo(
			id='conn_1', name='Connection', node_type=LineageNodeType.CONNECTION
		)
		lineage_engine.node_cache['table_1'] = LineageNodeInfo(
			id='table_1', name='Table', node_type=LineageNodeType.TABLE
		)
		lineage_engine.node_cache['field_1'] = LineageNodeInfo(
			id='field_1', name='Field', node_type=LineageNodeType.FIELD
		)

		# Add edge info
		lineage_engine.edge_cache['conn_1__table_1__contains'] = LineageEdgeInfo(
			id='conn_1__table_1__contains',
			source_node_id='conn_1',
			target_node_id='table_1',
			relationship_type=LineageRelationshipType.CONTAINS
		)

		# Get upstream lineage from field
		nodes, edges = lineage_engine._get_upstream_lineage('field_1', max_depth=10)

		assert len(nodes) == 3  # field_1, table_1, conn_1
		assert len(edges) >= 1

		# Should include the field itself and its upstream dependencies
		node_ids = {node.id for node in nodes}
		assert 'field_1' in node_ids
		assert 'table_1' in node_ids
		assert 'conn_1' in node_ids

	def test_get_downstream_lineage(self, lineage_engine):
		"""Test downstream lineage traversal"""
		# Create test graph: conn -> table -> field
		lineage_engine.graph.add_edge('conn_1', 'table_1')
		lineage_engine.graph.add_edge('table_1', 'field_1')

		# Add node info to cache
		lineage_engine.node_cache['conn_1'] = LineageNodeInfo(
			id='conn_1', name='Connection', node_type=LineageNodeType.CONNECTION
		)
		lineage_engine.node_cache['table_1'] = LineageNodeInfo(
			id='table_1', name='Table', node_type=LineageNodeType.TABLE
		)
		lineage_engine.node_cache['field_1'] = LineageNodeInfo(
			id='field_1', name='Field', node_type=LineageNodeType.FIELD
		)

		# Get downstream lineage from connection
		nodes, edges = lineage_engine._get_downstream_lineage('conn_1', max_depth=10)

		assert len(nodes) == 3  # conn_1, table_1, field_1

		# Should include the connection itself and its downstream dependencies
		node_ids = {node.id for node in nodes}
		assert 'conn_1' in node_ids
		assert 'table_1' in node_ids
		assert 'field_1' in node_ids

	def test_get_impact_analysis(self, lineage_engine):
		"""Test impact analysis (both upstream and downstream)"""
		# Create more complex graph: source -> table -> field -> target_table
		edges = [
			('source_conn', 'table_1'),
			('table_1', 'field_1'),
			('field_1', 'target_table')
		]

		for source, target in edges:
			lineage_engine.graph.add_edge(source, target)

		# Add node info to cache
		for node_id in ['source_conn', 'table_1', 'field_1', 'target_table']:
			lineage_engine.node_cache[node_id] = LineageNodeInfo(
				id=node_id, name=node_id.title(), node_type=LineageNodeType.TABLE
			)

		# Get impact analysis for central table
		nodes, edges = lineage_engine._get_impact_analysis('table_1', max_depth=10)

		# Should include upstream (source_conn) and downstream (field_1, target_table)
		node_ids = {node.id for node in nodes}
		assert 'source_conn' in node_ids  # Upstream
		assert 'table_1' in node_ids      # Center
		assert 'field_1' in node_ids      # Downstream
		assert 'target_table' in node_ids # Further downstream

	async def test_load_lineage_from_database(self, lineage_engine, sample_lineage_nodes, sample_lineage_edges, db_session):
		"""Test loading existing lineage from database"""
		lineage_engine.db_session = db_session

		# Clear caches to simulate fresh start
		lineage_engine.node_cache.clear()
		lineage_engine.edge_cache.clear()
		lineage_engine.graph.clear()

		# Load from database
		await lineage_engine.load_lineage_from_database()

		# Check nodes were loaded
		assert len(lineage_engine.node_cache) == len(sample_lineage_nodes)
		assert len(lineage_engine.edge_cache) == len(sample_lineage_edges)

		# Check graph was populated
		assert lineage_engine.graph.number_of_nodes() == len(sample_lineage_nodes)
		assert lineage_engine.graph.number_of_edges() == len(sample_lineage_edges)

		# Check sensitive data was preserved
		sensitive_nodes = [
			node for node in lineage_engine.node_cache.values()
			if node.sensitivity == SensitivityLevel.PII
		]
		assert len(sensitive_nodes) > 0

	def test_count_node_types(self, lineage_engine):
		"""Test node type counting"""
		test_nodes = [
			LineageNodeInfo(id='conn_1', name='Connection', node_type=LineageNodeType.CONNECTION),
			LineageNodeInfo(id='table_1', name='Table1', node_type=LineageNodeType.TABLE),
			LineageNodeInfo(id='table_2', name='Table2', node_type=LineageNodeType.TABLE),
			LineageNodeInfo(id='field_1', name='Field1', node_type=LineageNodeType.FIELD),
		]

		counts = lineage_engine._count_node_types(test_nodes)

		assert counts['connection'] == 1
		assert counts['table'] == 2
		assert counts['field'] == 1


class TestLineageTransformationTracking:
	"""Test transformation lineage tracking"""

	async def test_track_filter_transformation(self, lineage_engine, sample_flow):
		"""Test filter transformation tracking"""
		# Set transformation config with filter
		sample_flow.transformation_config = {
			'transformations': [
				{
					'type': 'filter',
					'conditions': [
						{'field': 'status', 'operator': 'equals', 'value': 'active'},
						{'field': 'created_at', 'operator': 'gte', 'value': '2024-01-01'}
					]
				}
			]
		}

		flow_node_id = f"flow_{sample_flow.id}"

		await lineage_engine._track_transformations(sample_flow, flow_node_id)

		# Check that filter relationships were created
		filter_edges = [
			edge for edge in lineage_engine.edge_cache.values()
			if edge.relationship_type == LineageRelationshipType.FILTERS_FROM
		]

		assert len(filter_edges) == 2  # Two filter conditions

		# Check transformation logic was recorded
		assert any('status' in edge.transformation_logic for edge in filter_edges)
		assert any('created_at' in edge.transformation_logic for edge in filter_edges)

	async def test_track_aggregate_transformation(self, lineage_engine, sample_flow):
		"""Test aggregation transformation tracking"""
		sample_flow.transformation_config = {
			'transformations': [
				{
					'type': 'aggregate',
					'group_by': ['region', 'product_category'],
					'aggregations': {
						'total_sales': {'field': 'amount', 'function': 'sum'},
						'avg_price': {'field': 'price', 'function': 'avg'},
						'order_count': {'field': 'id', 'function': 'count'}
					}
				}
			]
		}

		flow_node_id = f"flow_{sample_flow.id}"

		await lineage_engine._track_transformations(sample_flow, flow_node_id)

		# Check aggregate relationships
		aggregate_edges = [
			edge for edge in lineage_engine.edge_cache.values()
			if edge.relationship_type == LineageRelationshipType.AGGREGATES
		]

		# Should have edges for group by fields + aggregation fields
		assert len(aggregate_edges) >= 5  # 2 group by + 3 aggregations

		# Check transformation logic
		sum_edge = next(
			edge for edge in aggregate_edges
			if 'sum(amount)' in edge.transformation_logic
		)
		assert 'total_sales' in sum_edge.transformation_logic

	async def test_track_join_transformation(self, lineage_engine, sample_flow):
		"""Test join transformation tracking"""
		sample_flow.transformation_config = {
			'transformations': [
				{
					'type': 'join',
					'join_type': 'left',
					'left_on': 'user_id',
					'right_on': 'id'
				}
			]
		}

		flow_node_id = f"flow_{sample_flow.id}"

		await lineage_engine._track_transformations(sample_flow, flow_node_id)

		# Check join relationships
		join_edges = [
			edge for edge in lineage_engine.edge_cache.values()
			if edge.relationship_type == LineageRelationshipType.JOINS_WITH
		]

		assert len(join_edges) == 1

		join_edge = join_edges[0]
		assert 'LEFT JOIN' in join_edge.transformation_logic
		assert 'user_id = id' in join_edge.transformation_logic

	async def test_track_field_mappings(self, lineage_engine, sample_flow):
		"""Test field mapping tracking"""
		flow_node_id = f"flow_{sample_flow.id}"

		await lineage_engine._track_field_mappings(sample_flow, flow_node_id)

		# Check mapping relationships were created
		mapping_edges = [
			edge for edge in lineage_engine.edge_cache.values()
			if edge.relationship_type == LineageRelationshipType.MAPS_TO
		]

		# Should have one edge per field mapping
		assert len(mapping_edges) == len(sample_flow.field_mappings)

		# Check specific mapping
		email_mapping = next(
			edge for edge in mapping_edges
			if 'users.email' in edge.transformation_logic
		)
		assert 'customers.email_address' in email_mapping.transformation_logic


class TestLineageVisualization:
	"""Test lineage visualization data generation"""

	def test_visualization_node_format(self, lineage_engine):
		"""Test visualization node data format"""
		test_node = LineageNodeInfo(
			id='test_field',
			name='user_email',
			node_type=LineageNodeType.FIELD,
			connection_id='conn_123',
			schema_name='public',
			table_name='users',
			field_name='email',
			sensitivity=SensitivityLevel.PII,
			pii_classification='email_address',
			metadata={'data_type': 'varchar', 'nullable': False}
		)

		lineage_engine.node_cache['test_field'] = test_node

		viz_data = lineage_engine.get_lineage_visualization()

		node = viz_data['nodes'][0]
		assert node['id'] == 'test_field'
		assert node['label'] == 'user_email'
		assert node['type'] == 'field'
		assert node['metadata']['sensitive'] is True
		assert node['metadata']['pii'] is True
		assert node['metadata']['connection_id'] == 'conn_123'
		assert node['metadata']['table_name'] == 'users'
		assert node['metadata']['data_type'] == 'varchar'

	def test_visualization_edge_format(self, lineage_engine):
		"""Test visualization edge data format"""
		test_edge = LineageEdgeInfo(
			id='test_edge',
			source_node_id='source',
			target_node_id='target',
			relationship_type=LineageRelationshipType.TRANSFORMS_TO,
			transformation_logic='UPPER(name) AS display_name',
			confidence_score=0.92,
			metadata={'flow_name': 'User ETL', 'execution_time': 45.2}
		)

		lineage_engine.edge_cache['test_edge'] = test_edge

		viz_data = lineage_engine.get_lineage_visualization()

		edge = viz_data['edges'][0]
		assert edge['id'] == 'test_edge'
		assert edge['source'] == 'source'
		assert edge['target'] == 'target'
		assert edge['type'] == 'transforms_to'
		assert edge['metadata']['transformation_logic'] == 'UPPER(name) AS display_name'
		assert edge['metadata']['confidence_score'] == 0.92
		assert edge['metadata']['flow_name'] == 'User ETL'

	def test_visualization_summary_statistics(self, lineage_engine):
		"""Test visualization summary statistics"""
		# Add mixed nodes
		nodes = [
			LineageNodeInfo(id='conn1', name='DB', node_type=LineageNodeType.CONNECTION),
			LineageNodeInfo(id='table1', name='Users', node_type=LineageNodeType.TABLE),
			LineageNodeInfo(id='field1', name='Email', node_type=LineageNodeType.FIELD,
						   sensitivity=SensitivityLevel.PII),
			LineageNodeInfo(id='field2', name='ID', node_type=LineageNodeType.FIELD),
		]

		edges = [
			LineageEdgeInfo(id='e1', source_node_id='conn1', target_node_id='table1',
						   relationship_type=LineageRelationshipType.CONTAINS),
			LineageEdgeInfo(id='e2', source_node_id='table1', target_node_id='field1',
						   relationship_type=LineageRelationshipType.CONTAINS),
		]

		for node in nodes:
			lineage_engine.node_cache[node.id] = node
		for edge in edges:
			lineage_engine.edge_cache[edge.id] = edge

		viz_data = lineage_engine.get_lineage_visualization()

		summary = viz_data['summary']
		assert summary['total_nodes'] == 4
		assert summary['total_edges'] == 2
		assert summary['sensitive_entities'] == 1

		node_types = summary['node_types']
		assert node_types['connection'] == 1
		assert node_types['table'] == 1
		assert node_types['field'] == 2


# Performance tests
class TestLineageEnginePerformance:
	"""Performance tests for lineage engine"""

	async def test_large_graph_performance(self, lineage_engine):
		"""Test performance with large lineage graph"""
		import time

		# Create large number of nodes
		num_connections = 10
		num_tables_per_conn = 50
		num_fields_per_table = 20

		start_time = time.time()

		for conn_i in range(num_connections):
			conn_id = f'conn_{conn_i}'
			conn_node = LineageNodeInfo(
				id=conn_id,
				name=f'Connection {conn_i}',
				node_type=LineageNodeType.CONNECTION
			)
			await lineage_engine._create_or_update_node(conn_node)

			for table_i in range(num_tables_per_conn):
				table_id = f'{conn_id}_table_{table_i}'
				table_node = LineageNodeInfo(
					id=table_id,
					name=f'Table {table_i}',
					node_type=LineageNodeType.TABLE
				)
				await lineage_engine._create_or_update_node(table_node)

				# Connect table to connection
				await lineage_engine._create_relationship(
					conn_id, table_id, LineageRelationshipType.CONTAINS
				)

				for field_i in range(num_fields_per_table):
					field_id = f'{table_id}_field_{field_i}'
					field_node = LineageNodeInfo(
						id=field_id,
						name=f'Field {field_i}',
						node_type=LineageNodeType.FIELD
					)
					await lineage_engine._create_or_update_node(field_node)

					# Connect field to table
					await lineage_engine._create_relationship(
						table_id, field_id, LineageRelationshipType.CONTAINS
					)

		creation_time = time.time() - start_time

		# Test visualization generation performance
		viz_start = time.time()
		viz_data = lineage_engine.get_lineage_visualization()
		viz_time = time.time() - viz_start

		total_nodes = num_connections * (1 + num_tables_per_conn * (1 + num_fields_per_table))
		total_edges = num_connections * (num_tables_per_conn + num_tables_per_conn * num_fields_per_table)

		assert len(viz_data['nodes']) == total_nodes
		assert len(viz_data['edges']) == total_edges

		# Performance assertions (adjust based on expected performance)
		assert creation_time < 30.0  # Should create large graph in under 30 seconds
		assert viz_time < 5.0        # Should generate visualization in under 5 seconds

		print(f"Created {total_nodes} nodes and {total_edges} edges in {creation_time:.2f}s")
		print(f"Generated visualization in {viz_time:.2f}s")