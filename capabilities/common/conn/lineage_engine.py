"""
APG Data Lineage Engine
Advanced data lineage tracking and visualization engine

Provides comprehensive data lineage capabilities:
- Automatic schema discovery and lineage detection
- Field-level lineage tracking
- Sensitive data classification
- Impact analysis and change propagation
- Graph-based lineage storage and querying

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from collections import defaultdict, deque
try:
    import networkx as nx
except ImportError:
    class _CompatDiGraph:
        """Small directed graph fallback used when networkx is unavailable."""

        def __init__(self):
            self._nodes: dict[str, dict[str, Any]] = {}
            self._successors: dict[str, set[str]] = defaultdict(set)
            self._predecessors: dict[str, set[str]] = defaultdict(set)
            self._edges: dict[tuple[str, str], dict[str, Any]] = {}

        def add_node(self, node_id: str, **attrs: Any) -> None:
            self._nodes[node_id] = attrs

        def add_edge(self, source_id: str, target_id: str, **attrs: Any) -> None:
            self._nodes.setdefault(source_id, {})
            self._nodes.setdefault(target_id, {})
            self._successors[source_id].add(target_id)
            self._predecessors[target_id].add(source_id)
            self._edges[(source_id, target_id)] = attrs

        def has_node(self, node_id: str) -> bool:
            return node_id in self._nodes

        def predecessors(self, node_id: str):
            return iter(self._predecessors.get(node_id, set()))

        def successors(self, node_id: str):
            return iter(self._successors.get(node_id, set()))

    class _NetworkXCompat:
        DiGraph = _CompatDiGraph

    nx = _NetworkXCompat()

from .sqlalchemy_models import (
    CnConnection, CnDataFlow, CnLineageNode, CnLineageEdge,
    LineageNodeType, ConnectionStatus, ConnectionType
)


class LineageRelationshipType(str, Enum):
    """Types of lineage relationships"""
    CONTAINS = "contains"           # Table contains field
    DERIVES_FROM = "derives_from"   # Target derives from source
    TRANSFORMS_TO = "transforms_to" # Field transforms to field
    MAPS_TO = "maps_to"             # Direct field mapping
    AGGREGATES = "aggregates"       # Aggregation relationship
    JOINS_WITH = "joins_with"       # Join relationship
    FILTERS_FROM = "filters_from"   # Filtering relationship


class SensitivityLevel(str, Enum):
    """Data sensitivity classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Industry


@dataclass
class LineageNodeInfo:
    """Information about a lineage node"""
    id: str
    name: str
    node_type: LineageNodeType
    connection_id: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    field_name: Optional[str] = None
    data_type: Optional[str] = None
    sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC
    pii_classification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdgeInfo:
    """Information about a lineage relationship"""
    id: str
    source_node_id: str
    target_node_id: str
    relationship_type: LineageRelationshipType
    transformation_logic: Optional[str] = None
    flow_id: Optional[str] = None
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLineageEngine:
    """
    Core data lineage engine for tracking and analyzing data relationships
    """

    def __init__(self, db_session=None):
        self.db_session = db_session
        self.graph = nx.DiGraph()  # Directed graph for lineage
        self.node_cache: Dict[str, LineageNodeInfo] = {}
        self.edge_cache: Dict[str, LineageEdgeInfo] = {}

        # Pattern recognition for sensitive data
        self.sensitive_patterns = {
            SensitivityLevel.PII: [
                r'.*email.*', r'.*ssn.*', r'.*social.*security.*',
                r'.*phone.*', r'.*mobile.*', r'.*address.*',
                r'.*first.*name.*', r'.*last.*name.*', r'.*full.*name.*',
                r'.*birth.*date.*', r'.*dob.*'
            ],
            SensitivityLevel.PCI: [
                r'.*credit.*card.*', r'.*card.*number.*', r'.*cvv.*',
                r'.*expir.*', r'.*payment.*'
            ],
            SensitivityLevel.PHI: [
                r'.*medical.*', r'.*health.*', r'.*diagnosis.*',
                r'.*treatment.*', r'.*medication.*', r'.*patient.*'
            ]
        }

    def _generate_node_id(self, connection_id: str, schema: str, table: str, field: str = None) -> str:
        """Generate unique node ID"""
        parts = [connection_id, schema, table]
        if field:
            parts.append(field)
        return ".".join(parts)

    def _classify_sensitivity(self, field_name: str, data_type: str = None,
                            sample_data: List[Any] = None) -> Tuple[SensitivityLevel, Optional[str]]:
        """Classify field sensitivity based on name, type, and sample data"""

        field_lower = field_name.lower()

        # Check against patterns
        for sensitivity, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                if re.match(pattern, field_lower):
                    return sensitivity, f"Pattern match: {pattern}"

        # Check data type
        if data_type:
            type_lower = data_type.lower()
            if 'uuid' in type_lower or 'id' in field_lower:
                # IDs might be sensitive depending on context
                if any(word in field_lower for word in ['customer', 'user', 'person', 'patient']):
                    return SensitivityLevel.PII, "ID field for person"

        # Analyze sample data if available
        if sample_data:
            for sample in sample_data[:10]:  # Check first 10 samples
                if sample and isinstance(sample, str):
                    # Check for email patterns
                    if '@' in sample and '.' in sample:
                        return SensitivityLevel.PII, "Email format detected"

                    # Check for phone patterns
                    if re.match(r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$', sample):
                        return SensitivityLevel.PII, "Phone number format"

                    # Check for SSN patterns
                    if re.match(r'^\d{3}-?\d{2}-?\d{4}$', sample):
                        return SensitivityLevel.PII, "SSN format"

        return SensitivityLevel.INTERNAL, None

    async def discover_connection_schema(self, connection: CnConnection) -> Dict[str, Any]:
        """
        Discover schema and create lineage nodes for a connection
        Uses Singer.io discovery when available
        """

        schema_info = {
            'connection_id': connection.id,
            'nodes_created': 0,
            'tables_discovered': 0,
            'fields_discovered': 0,
            'sensitive_fields': 0
        }

        try:
            # Create connection node
            conn_node = LineageNodeInfo(
                id=f"conn_{connection.id}",
                name=connection.name,
                node_type=LineageNodeType.CONNECTION,
                connection_id=str(connection.id),
                metadata={
                    'connection_type': connection.connection_type.value,
                    'singer_tap': connection.singer_tap,
                    'description': connection.description
                }
            )
            await self._create_or_update_node(conn_node)
            schema_info['nodes_created'] += 1

            # If Singer tap is configured, use discovery
            if connection.singer_tap and connection.tap_config:
                schema_info.update(await self._discover_singer_schema(connection))
            else:
                # Fallback to basic schema detection
                schema_info.update(await self._discover_basic_schema(connection))

        except Exception as e:
            print(f"Error discovering schema for {connection.name}: {e}")

        return schema_info

    async def _discover_singer_schema(self, connection: CnConnection) -> Dict[str, Any]:
        """Discover schema using Singer.io catalog"""

        # This would integrate with actual Singer.io discovery
        # For now, we'll simulate the structure

        discovery_info = {
            'tables_discovered': 0,
            'fields_discovered': 0,
            'sensitive_fields': 0,
            'nodes_created': 0
        }

        # Simulated catalog structure
        mock_catalog = {
            'streams': [
                {
                    'tap_stream_id': 'users',
                    'schema': {
                        'properties': {
                            'id': {'type': 'integer'},
                            'email': {'type': 'string'},
                            'first_name': {'type': 'string'},
                            'last_name': {'type': 'string'},
                            'phone': {'type': 'string'},
                            'created_at': {'type': 'string', 'format': 'date-time'}
                        }
                    },
                    'metadata': [
                        {'breadcrumb': [], 'metadata': {'table-key-properties': ['id']}}
                    ]
                },
                {
                    'tap_stream_id': 'orders',
                    'schema': {
                        'properties': {
                            'id': {'type': 'integer'},
                            'user_id': {'type': 'integer'},
                            'total_amount': {'type': 'number'},
                            'created_at': {'type': 'string', 'format': 'date-time'},
                            'status': {'type': 'string'}
                        }
                    }
                }
            ]
        }

        # Process catalog
        for stream in mock_catalog['streams']:
            table_name = stream['tap_stream_id']
            schema = stream.get('schema', {})
            properties = schema.get('properties', {})

            # Create table node
            table_node_id = self._generate_node_id(str(connection.id), 'public', table_name)
            table_node = LineageNodeInfo(
                id=table_node_id,
                name=table_name,
                node_type=LineageNodeType.TABLE,
                connection_id=str(connection.id),
                schema_name='public',
                table_name=table_name,
                metadata={
                    'stream_metadata': stream.get('metadata', []),
                    'field_count': len(properties)
                }
            )
            await self._create_or_update_node(table_node)
            discovery_info['nodes_created'] += 1
            discovery_info['tables_discovered'] += 1

            # Create relationship: connection contains table
            await self._create_relationship(
                f"conn_{connection.id}",
                table_node_id,
                LineageRelationshipType.CONTAINS,
                metadata={'discovered_via': 'singer_catalog'}
            )

            # Create field nodes
            for field_name, field_schema in properties.items():
                field_node_id = self._generate_node_id(
                    str(connection.id), 'public', table_name, field_name
                )

                # Classify sensitivity
                sensitivity, classification = self._classify_sensitivity(
                    field_name, field_schema.get('type')
                )

                field_node = LineageNodeInfo(
                    id=field_node_id,
                    name=f"{table_name}.{field_name}",
                    node_type=LineageNodeType.FIELD,
                    connection_id=str(connection.id),
                    schema_name='public',
                    table_name=table_name,
                    field_name=field_name,
                    data_type=field_schema.get('type'),
                    sensitivity=sensitivity,
                    pii_classification=classification,
                    metadata={
                        'singer_schema': field_schema,
                        'format': field_schema.get('format')
                    }
                )

                await self._create_or_update_node(field_node)
                discovery_info['nodes_created'] += 1
                discovery_info['fields_discovered'] += 1

                if sensitivity != SensitivityLevel.PUBLIC:
                    discovery_info['sensitive_fields'] += 1

                # Create relationship: table contains field
                await self._create_relationship(
                    table_node_id,
                    field_node_id,
                    LineageRelationshipType.CONTAINS,
                    metadata={'field_type': field_schema.get('type')}
                )

        return discovery_info

    async def _discover_basic_schema(self, connection: CnConnection) -> Dict[str, Any]:
        """Basic schema discovery fallback"""

        # This would implement basic schema detection based on connection type
        # For now, return empty discovery
        return {
            'tables_discovered': 0,
            'fields_discovered': 0,
            'sensitive_fields': 0,
            'nodes_created': 0
        }

    async def track_flow_lineage(self, flow: CnDataFlow) -> Dict[str, Any]:
        """Track lineage created by a data flow execution"""

        lineage_info = {
            'flow_id': flow.id,
            'relationships_created': 0,
            'transformations_tracked': 0
        }

        try:
            # Create flow node
            flow_node_id = f"flow_{flow.id}"
            flow_node = LineageNodeInfo(
                id=flow_node_id,
                name=flow.name,
                node_type=LineageNodeType.FLOW,
                metadata={
                    'description': flow.description,
                    'source_connection_id': str(flow.source_connection_id),
                    'target_connection_id': str(flow.target_connection_id),
                    'transformation_config': flow.transformation_config,
                    'field_mappings': flow.field_mappings
                }
            )
            await self._create_or_update_node(flow_node)

            # Track field mappings
            if flow.field_mappings:
                await self._track_field_mappings(flow, flow_node_id)
                lineage_info['relationships_created'] += len(flow.field_mappings)

            # Track transformations
            if flow.transformation_config:
                await self._track_transformations(flow, flow_node_id)
                lineage_info['transformations_tracked'] += 1

        except Exception as e:
            print(f"Error tracking flow lineage for {flow.name}: {e}")

        return lineage_info

    async def _track_field_mappings(self, flow: CnDataFlow, flow_node_id: str):
        """Track field-level mappings in a flow"""

        source_conn = str(flow.source_connection_id)
        target_conn = str(flow.target_connection_id)

        for source_field, target_field in flow.field_mappings.items():
            # Find or create source and target field nodes
            # This is simplified - would need actual schema discovery

            source_parts = source_field.split('.')
            target_parts = target_field.split('.')

            if len(source_parts) >= 2:
                source_table = source_parts[0]
                source_field_name = source_parts[1]
                source_node_id = self._generate_node_id(
                    source_conn, 'public', source_table, source_field_name
                )

            if len(target_parts) >= 2:
                target_table = target_parts[0]
                target_field_name = target_parts[1]
                target_node_id = self._generate_node_id(
                    target_conn, 'public', target_table, target_field_name
                )

                # Create mapping relationship
                await self._create_relationship(
                    source_node_id,
                    target_node_id,
                    LineageRelationshipType.MAPS_TO,
                    flow_id=str(flow.id),
                    transformation_logic=f"Direct mapping: {source_field} -> {target_field}",
                    metadata={
                        'mapping_type': 'direct',
                        'flow_name': flow.name
                    }
                )

    async def _track_transformations(self, flow: CnDataFlow, flow_node_id: str):
        """Track transformation logic in lineage"""

        transformation_config = flow.transformation_config

        for transform in transformation_config.get('transformations', []):
            transform_type = transform.get('type')

            if transform_type == 'filter':
                # Track filtering relationships
                await self._track_filter_transformation(flow, transform, flow_node_id)

            elif transform_type == 'aggregate':
                # Track aggregation relationships
                await self._track_aggregate_transformation(flow, transform, flow_node_id)

            elif transform_type == 'join':
                # Track join relationships
                await self._track_join_transformation(flow, transform, flow_node_id)

    async def _track_filter_transformation(self, flow: CnDataFlow, transform: Dict, flow_node_id: str):
        """Track filtering transformation lineage"""

        conditions = transform.get('conditions', [])

        for condition in conditions:
            field = condition.get('field')
            if field:
                # Create filter relationship
                source_node_id = self._generate_node_id(
                    str(flow.source_connection_id), 'public', 'unknown', field
                )

                await self._create_relationship(
                    source_node_id,
                    flow_node_id,
                    LineageRelationshipType.FILTERS_FROM,
                    flow_id=str(flow.id),
                    transformation_logic=json.dumps(condition),
                    metadata={
                        'transform_type': 'filter',
                        'condition': condition
                    }
                )

    async def _track_aggregate_transformation(self, flow: CnDataFlow, transform: Dict, flow_node_id: str):
        """Track aggregation transformation lineage"""

        group_by = transform.get('group_by', [])
        aggregations = transform.get('aggregations', {})

        # Track group by fields
        for field in group_by:
            source_node_id = self._generate_node_id(
                str(flow.source_connection_id), 'public', 'unknown', field
            )

            await self._create_relationship(
                source_node_id,
                flow_node_id,
                LineageRelationshipType.AGGREGATES,
                flow_id=str(flow.id),
                transformation_logic=f"GROUP BY {field}",
                metadata={
                    'transform_type': 'group_by',
                    'field': field
                }
            )

        # Track aggregated fields
        for result_field, agg_config in aggregations.items():
            source_field = agg_config.get('field')
            function = agg_config.get('function')

            if source_field:
                source_node_id = self._generate_node_id(
                    str(flow.source_connection_id), 'public', 'unknown', source_field
                )

                target_node_id = self._generate_node_id(
                    str(flow.target_connection_id), 'public', 'unknown', result_field
                )

                await self._create_relationship(
                    source_node_id,
                    target_node_id,
                    LineageRelationshipType.AGGREGATES,
                    flow_id=str(flow.id),
                    transformation_logic=f"{function}({source_field}) AS {result_field}",
                    metadata={
                        'transform_type': 'aggregate',
                        'function': function,
                        'source_field': source_field,
                        'result_field': result_field
                    }
                )

    async def _track_join_transformation(self, flow: CnDataFlow, transform: Dict, flow_node_id: str):
        """Track join transformation lineage"""

        join_type = transform.get('join_type', 'inner')
        left_on = transform.get('left_on')
        right_on = transform.get('right_on')

        if left_on and right_on:
            left_node_id = self._generate_node_id(
                str(flow.source_connection_id), 'public', 'left_table', left_on
            )

            right_node_id = self._generate_node_id(
                str(flow.source_connection_id), 'public', 'right_table', right_on
            )

            await self._create_relationship(
                left_node_id,
                right_node_id,
                LineageRelationshipType.JOINS_WITH,
                flow_id=str(flow.id),
                transformation_logic=f"{join_type.upper()} JOIN ON {left_on} = {right_on}",
                metadata={
                    'transform_type': 'join',
                    'join_type': join_type,
                    'left_field': left_on,
                    'right_field': right_on
                }
            )

    async def _create_or_update_node(self, node_info: LineageNodeInfo):
        """Create or update a lineage node in database and cache"""

        # Update cache
        self.node_cache[node_info.id] = node_info

        # Add to graph
        self.graph.add_node(node_info.id, **node_info.__dict__)

        # Update database if session available
        if self.db_session:
            try:
                existing = self.db_session.query(CnLineageNode).filter(
                    CnLineageNode.id == node_info.id
                ).first()

                if existing:
                    # Update existing node
                    existing.name = node_info.name
                    existing.node_type = node_info.node_type
                    existing.connection_id = node_info.connection_id
                    existing.schema_name = node_info.schema_name
                    existing.table_name = node_info.table_name
                    existing.field_name = node_info.field_name
                    existing.sensitive = (node_info.sensitivity != SensitivityLevel.PUBLIC)
                    existing.pii_classification = node_info.pii_classification
                    existing.meta_data = node_info.metadata
                    existing.properties = node_info.properties
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new node
                    new_node = CnLineageNode(
                        id=node_info.id,
                        tenant_id='default',
                        name=node_info.name,
                        node_type=node_info.node_type,
                        connection_id=node_info.connection_id,
                        schema_name=node_info.schema_name,
                        table_name=node_info.table_name,
                        field_name=node_info.field_name,
                        sensitive=(node_info.sensitivity != SensitivityLevel.PUBLIC),
                        pii_classification=node_info.pii_classification,
                        meta_data=node_info.metadata,
                        properties=node_info.properties
                    )
                    self.db_session.add(new_node)

                self.db_session.commit()

            except Exception as e:
                print(f"Error persisting node {node_info.id}: {e}")
                if self.db_session:
                    self.db_session.rollback()

    async def _create_relationship(self, source_id: str, target_id: str,
                                 relationship_type: LineageRelationshipType,
                                 flow_id: Optional[str] = None,
                                 transformation_logic: Optional[str] = None,
                                 confidence_score: float = 1.0,
                                 metadata: Optional[Dict] = None):
        """Create a lineage relationship"""

        edge_id = f"{source_id}__{target_id}__{relationship_type.value}"

        edge_info = LineageEdgeInfo(
            id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_type=relationship_type,
            transformation_logic=transformation_logic,
            flow_id=flow_id,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )

        # Update cache
        self.edge_cache[edge_id] = edge_info

        # Add to graph
        self.graph.add_edge(
            source_id, target_id,
            relationship_type=relationship_type.value,
            transformation_logic=transformation_logic,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )

        # Update database if session available
        if self.db_session:
            try:
                existing = self.db_session.query(CnLineageEdge).filter(
                    CnLineageEdge.id == edge_id
                ).first()

                if existing:
                    # Update existing edge
                    existing.relationship_type = relationship_type.value
                    existing.transformation_logic = transformation_logic
                    existing.flow_id = flow_id
                    existing.confidence_score = confidence_score
                    existing.meta_data = metadata or {}
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new edge
                    new_edge = CnLineageEdge(
                        id=edge_id,
                        tenant_id='default',
                        source_node_id=source_id,
                        target_node_id=target_id,
                        relationship_type=relationship_type.value,
                        transformation_logic=transformation_logic,
                        flow_id=flow_id,
                        confidence_score=confidence_score,
                        meta_data=metadata or {}
                    )
                    self.db_session.add(new_edge)

                self.db_session.commit()

            except Exception as e:
                print(f"Error persisting edge {edge_id}: {e}")
                if self.db_session:
                    self.db_session.rollback()

    def get_lineage_visualization(self,
                                node_id: Optional[str] = None,
                                visualization_type: str = 'full',
                                max_depth: int = 10) -> Dict[str, Any]:
        """Get lineage data for visualization"""

        if visualization_type == 'upstream' and node_id:
            nodes, edges = self._get_upstream_lineage(node_id, max_depth)
        elif visualization_type == 'downstream' and node_id:
            nodes, edges = self._get_downstream_lineage(node_id, max_depth)
        elif visualization_type == 'impact' and node_id:
            nodes, edges = self._get_impact_analysis(node_id, max_depth)
        else:
            # Full visualization
            nodes = list(self.node_cache.values())
            edges = list(self.edge_cache.values())

        # Convert to visualization format
        viz_nodes = []
        for node in nodes:
            viz_nodes.append({
                'id': node.id,
                'label': node.name,
                'type': node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                'metadata': {
                    'sensitive': node.sensitivity != SensitivityLevel.PUBLIC if hasattr(node, 'sensitivity') else False,
                    'pii': bool(node.pii_classification) if hasattr(node, 'pii_classification') else False,
                    'connection_id': node.connection_id,
                    'schema_name': node.schema_name,
                    'table_name': node.table_name,
                    'field_name': node.field_name,
                    **node.metadata
                }
            })

        viz_edges = []
        for edge in edges:
            viz_edges.append({
                'id': edge.id,
                'source': edge.source_node_id,
                'target': edge.target_node_id,
                'type': edge.relationship_type.value if hasattr(edge.relationship_type, 'value') else str(edge.relationship_type),
                'metadata': {
                    'transformation_logic': edge.transformation_logic,
                    'confidence_score': edge.confidence_score,
                    'flow_id': edge.flow_id,
                    **edge.metadata
                }
            })

        # Calculate summary statistics
        sensitive_levels = {
            SensitivityLevel.CONFIDENTIAL,
            SensitivityLevel.RESTRICTED,
            SensitivityLevel.PII,
            SensitivityLevel.PHI,
            SensitivityLevel.PCI,
        }
        sensitive_count = sum(
            1 for node in nodes
            if getattr(node, 'sensitivity', SensitivityLevel.PUBLIC) in sensitive_levels
        )

        return {
            'nodes': viz_nodes,
            'edges': viz_edges,
            'summary': {
                'total_nodes': len(viz_nodes),
                'total_edges': len(viz_edges),
                'sensitive_entities': sensitive_count,
                'node_types': self._count_node_types(nodes)
            }
        }

    def _get_upstream_lineage(self, node_id: str, max_depth: int) -> Tuple[List, List]:
        """Get upstream dependencies of a node"""

        upstream_nodes = set()
        upstream_edges = set()

        # BFS to find upstream nodes
        queue = deque([(node_id, 0)])
        visited = {node_id}

        while queue and len(queue) > 0:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Find predecessors in graph
            if self.graph.has_node(current_id):
                predecessors = list(self.graph.predecessors(current_id))

                for pred_id in predecessors:
                    if pred_id not in visited:
                        visited.add(pred_id)
                        queue.append((pred_id, depth + 1))
                        upstream_nodes.add(pred_id)

                    # Add edge
                    edge_key = f"{pred_id}__{current_id}"
                    for edge_id, edge_info in self.edge_cache.items():
                        if edge_info.source_node_id == pred_id and edge_info.target_node_id == current_id:
                            upstream_edges.add(edge_id)
                            break

        # Include the starting node
        upstream_nodes.add(node_id)

        # Get node info
        nodes = [self.node_cache[nid] for nid in upstream_nodes if nid in self.node_cache]
        edges = [self.edge_cache[eid] for eid in upstream_edges if eid in self.edge_cache]

        return nodes, edges

    def _get_downstream_lineage(self, node_id: str, max_depth: int) -> Tuple[List, List]:
        """Get downstream dependencies of a node"""

        downstream_nodes = set()
        downstream_edges = set()

        # BFS to find downstream nodes
        queue = deque([(node_id, 0)])
        visited = {node_id}

        while queue and len(queue) > 0:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Find successors in graph
            if self.graph.has_node(current_id):
                successors = list(self.graph.successors(current_id))

                for succ_id in successors:
                    if succ_id not in visited:
                        visited.add(succ_id)
                        queue.append((succ_id, depth + 1))
                        downstream_nodes.add(succ_id)

                    # Add edge
                    for edge_id, edge_info in self.edge_cache.items():
                        if edge_info.source_node_id == current_id and edge_info.target_node_id == succ_id:
                            downstream_edges.add(edge_id)
                            break

        # Include the starting node
        downstream_nodes.add(node_id)

        # Get node info
        nodes = [self.node_cache[nid] for nid in downstream_nodes if nid in self.node_cache]
        edges = [self.edge_cache[eid] for eid in downstream_edges if eid in self.edge_cache]

        return nodes, edges

    def _get_impact_analysis(self, node_id: str, max_depth: int) -> Tuple[List, List]:
        """Get impact analysis - both upstream and downstream"""

        # Get both upstream and downstream
        up_nodes, up_edges = self._get_upstream_lineage(node_id, max_depth)
        down_nodes, down_edges = self._get_downstream_lineage(node_id, max_depth)

        # Combine and deduplicate
        all_nodes = {node.id: node for node in up_nodes + down_nodes}
        all_edges = {edge.id: edge for edge in up_edges + down_edges}

        return list(all_nodes.values()), list(all_edges.values())

    def _count_node_types(self, nodes: List) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = defaultdict(int)

        for node in nodes:
            node_type = node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type)
            type_counts[node_type] += 1

        return dict(type_counts)

    async def load_lineage_from_database(self):
        """Load existing lineage from database into memory"""

        if not self.db_session:
            return

        try:
            # Load nodes
            nodes = self.db_session.query(CnLineageNode).all()
            for node in nodes:
                node_info = LineageNodeInfo(
                    id=node.id,
                    name=node.name,
                    node_type=node.node_type,
                    connection_id=node.connection_id,
                    schema_name=node.schema_name,
                    table_name=node.table_name,
                    field_name=node.field_name,
                    sensitivity=SensitivityLevel.PII if node.sensitive else SensitivityLevel.PUBLIC,
                    pii_classification=node.pii_classification,
                    metadata=node.meta_data or {},
                    properties=node.properties or {}
                )
                self.node_cache[node.id] = node_info
                self.graph.add_node(node.id, **node_info.__dict__)

            # Load edges
            edges = self.db_session.query(CnLineageEdge).all()
            for edge in edges:
                edge_info = LineageEdgeInfo(
                    id=edge.id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    relationship_type=LineageRelationshipType(edge.relationship_type),
                    transformation_logic=edge.transformation_logic,
                    flow_id=edge.flow_id,
                    confidence_score=edge.confidence_score or 1.0,
                    metadata=edge.meta_data or {}
                )
                self.edge_cache[edge.id] = edge_info
                self.graph.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    **edge_info.__dict__
                )

        except Exception as e:
            print(f"Error loading lineage from database: {e}")


# Global lineage engine instance
lineage_engine = DataLineageEngine()
