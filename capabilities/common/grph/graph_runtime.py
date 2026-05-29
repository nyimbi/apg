"""Domain graph algorithms for the GRPH capability runtime."""

from __future__ import annotations

from collections import deque

from .models import GraphEdge, GraphNode, RelationshipClassification


class GraphTraversalPlanner:
	"""Execute deterministic outbound graph traversals over in-memory edge sets."""

	def traverse(
		self,
		tenant_id: str,
		start_node_id: str,
		max_depth: int,
		edges: list[GraphEdge],
	) -> tuple[list[str], list[str]]:
		seen_nodes = {start_node_id}
		seen_edges: set[str] = set()
		queue: deque[tuple[str, int]] = deque([(start_node_id, 0)])
		while queue:
			node_id, depth = queue.popleft()
			if depth >= max_depth:
				continue
			for edge in edges:
				if edge.tenant_id != tenant_id or edge.from_node_id != node_id:
					continue
				seen_edges.add(edge.id)
				if edge.to_node_id not in seen_nodes:
					seen_nodes.add(edge.to_node_id)
					queue.append((edge.to_node_id, depth + 1))
		return sorted(seen_nodes), sorted(seen_edges)


class GraphQualityInspector:
	"""Compute graph quality metrics used by the GRPH quality console."""

	def inspect(
		self,
		nodes: list[GraphNode],
		edges: list[GraphEdge],
	) -> dict[str, int]:
		connected = {edge.from_node_id for edge in edges} | {edge.to_node_id for edge in edges}
		return {
			"orphan_node_count": len([node for node in nodes if node.id not in connected]),
			"missing_owner_count": len([node for node in nodes if not node.owner_id])
			+ len([edge for edge in edges if not edge.owner_id]),
			"restricted_edge_count": len([
				edge for edge in edges
				if edge.classification == RelationshipClassification.RESTRICTED
			]),
		}
