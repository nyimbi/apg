"""
Federated Identity Mesh - Decentralized Identity Federation System

Revolutionary decentralized identity federation that enables seamless authentication
across multiple identity providers without centralized authority, using distributed
consensus protocols and cryptographic verification.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import base64
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import aiohttp
import networkx as nx
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrustLevel(Enum):
	"""Trust levels in the federated mesh"""
	UNKNOWN = "unknown"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERIFIED = "verified"


class FederationProtocol(Enum):
	"""Supported federation protocols"""
	SAML2 = "saml2"
	OIDC = "oidc"
	OAUTH2 = "oauth2"
	APG_MESH = "apg_mesh"
	CUSTOM = "custom"


class MeshNodeStatus(Enum):
	"""Status of mesh nodes"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	COMPROMISED = "compromised"
	MAINTENANCE = "maintenance"


@dataclass
class CryptographicProof:
	"""Cryptographic proof for mesh operations"""
	signature: str
	public_key: str
	timestamp: datetime
	nonce: str
	algorithm: str = "RSA-SHA256"


@dataclass
class TrustMetrics:
	"""Trust metrics for mesh nodes"""
	successful_authentications: int = 0
	failed_authentications: int = 0
	average_response_time: float = 0.0
	security_incidents: int = 0
	uptime_percentage: float = 100.0
	consensus_participation: float = 0.0
	peer_endorsements: int = 0
	reputation_score: float = 0.0


class IdentityAssertion(BaseModel):
	"""Identity assertion in the mesh"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	issuer_node_id: str
	subject_id: str
	attributes: Dict[str, Any]
	trust_level: TrustLevel
	expiry: datetime
	proof: Dict[str, Any]
	chain_of_trust: List[str]
	created_at: datetime = Field(default_factory=datetime.utcnow)


class MeshNode(BaseModel):
	"""Node in the federated identity mesh"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	domain: str
	public_key: str
	private_key_hash: str  # Hash of private key for verification
	supported_protocols: List[FederationProtocol]
	trust_level: TrustLevel
	status: MeshNodeStatus
	endpoints: Dict[str, str]
	capabilities: List[str]
	trust_metrics: Dict[str, Any] = Field(default_factory=dict)
	peer_connections: List[str] = Field(default_factory=list)
	last_heartbeat: Optional[datetime] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)


class ConsensusProposal(BaseModel):
	"""Proposal for mesh consensus"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	proposer_id: str
	proposal_type: str
	content: Dict[str, Any]
	voting_deadline: datetime
	required_consensus: float = 0.67  # 2/3 majority
	votes: Dict[str, bool] = Field(default_factory=dict)  # node_id -> vote
	signatures: Dict[str, str] = Field(default_factory=dict)
	status: str = "pending"
	created_at: datetime = Field(default_factory=datetime.utcnow)


class FederatedIdentityMesh:
	"""
	Decentralized identity federation mesh that enables authentication
	across multiple identity providers without central authority
	"""
	
	def __init__(self):
		self.nodes: Dict[str, MeshNode] = {}
		self.graph = nx.Graph()
		self.trust_scores: Dict[str, float] = {}
		self.active_proposals: Dict[str, ConsensusProposal] = {}
		self.assertion_cache: Dict[str, IdentityAssertion] = {}
		self.consensus_history: List[Dict[str, Any]] = []
		self.mesh_metrics = {
			"total_nodes": 0,
			"active_nodes": 0,
			"trust_relationships": 0,
			"successful_authentications": 0,
			"consensus_decisions": 0
		}
	
	def _log_mesh_operation(self, operation: str, details: Dict[str, Any]) -> None:
		"""Log mesh operations"""
		logger.info(f"Mesh Operation: {operation}")
		for key, value in details.items():
			logger.info(f"  {key}: {value}")
	
	async def register_node(
		self,
		node: MeshNode,
		endorsing_nodes: Optional[List[str]] = None
	) -> bool:
		"""Register a new node in the mesh"""
		assert isinstance(node, MeshNode), "Invalid node type"
		assert node.id not in self.nodes, "Node already registered"
		
		try:
			# Verify node credentials
			if not await self._verify_node_credentials(node):
				self._log_mesh_operation("node_registration_failed", {
					"node_id": node.id,
					"reason": "credential_verification_failed"
				})
				return False
			
			# Check endorsements for trust establishment
			initial_trust_level = TrustLevel.UNKNOWN
			if endorsing_nodes:
				trust_score = await self._calculate_endorsement_trust(endorsing_nodes)
				if trust_score >= 0.8:
					initial_trust_level = TrustLevel.HIGH
				elif trust_score >= 0.6:
					initial_trust_level = TrustLevel.MEDIUM
				else:
					initial_trust_level = TrustLevel.LOW
			
			node.trust_level = initial_trust_level
			
			# Add to mesh
			self.nodes[node.id] = node
			self.graph.add_node(node.id, **node.model_dump())
			self.trust_scores[node.id] = self._trust_level_to_score(initial_trust_level)
			
			# Create initial trust relationships
			if endorsing_nodes:
				for endorser_id in endorsing_nodes:
					if endorser_id in self.nodes:
						self.graph.add_edge(node.id, endorser_id, trust_weight=0.5)
			
			# Update metrics
			self.mesh_metrics["total_nodes"] = len(self.nodes)
			self.mesh_metrics["active_nodes"] = len([
				n for n in self.nodes.values() 
				if n.status == MeshNodeStatus.ACTIVE
			])
			
			self._log_mesh_operation("node_registered", {
				"node_id": node.id,
				"domain": node.domain,
				"trust_level": initial_trust_level.value,
				"endorsers": len(endorsing_nodes) if endorsing_nodes else 0
			})
			
			return True
			
		except Exception as e:
			self._log_mesh_operation("node_registration_error", {
				"node_id": node.id,
				"error": str(e)
			})
			return False
	
	async def authenticate_user(
		self,
		user_id: str,
		source_node_id: str,
		target_node_id: str,
		context: Dict[str, Any]
	) -> Optional[IdentityAssertion]:
		"""Authenticate user across mesh nodes"""
		assert user_id, "User ID required"
		assert source_node_id in self.nodes, "Source node not found"
		assert target_node_id in self.nodes, "Target node not found"
		
		try:
			source_node = self.nodes[source_node_id]
			target_node = self.nodes[target_node_id]
			
			# Check if nodes can communicate
			trust_path = await self._find_trust_path(source_node_id, target_node_id)
			if not trust_path:
				self._log_mesh_operation("authentication_failed", {
					"user_id": user_id,
					"reason": "no_trust_path",
					"source": source_node_id,
					"target": target_node_id
				})
				return None
			
			# Calculate trust score for this authentication
			path_trust_score = await self._calculate_path_trust(trust_path)
			if path_trust_score < 0.5:
				self._log_mesh_operation("authentication_failed", {
					"user_id": user_id,
					"reason": "insufficient_trust",
					"path_trust": path_trust_score
				})
				return None
			
			# Create identity assertion
			assertion = IdentityAssertion(
				issuer_node_id=source_node_id,
				subject_id=user_id,
				attributes=context.get("attributes", {}),
				trust_level=self._score_to_trust_level(path_trust_score),
				expiry=datetime.utcnow() + timedelta(hours=8),
				proof=await self._create_assertion_proof(source_node, user_id, context),
				chain_of_trust=trust_path
			)
			
			# Cache assertion
			self.assertion_cache[assertion.id] = assertion
			
			# Update metrics
			self.mesh_metrics["successful_authentications"] += 1
			
			# Update node trust metrics
			await self._update_node_trust_metrics(source_node_id, True)
			
			self._log_mesh_operation("user_authenticated", {
				"user_id": user_id,
				"assertion_id": assertion.id,
				"trust_level": assertion.trust_level.value,
				"trust_path_length": len(trust_path)
			})
			
			return assertion
			
		except Exception as e:
			self._log_mesh_operation("authentication_error", {
				"user_id": user_id,
				"error": str(e)
			})
			await self._update_node_trust_metrics(source_node_id, False)
			return None
	
	async def establish_trust_relationship(
		self,
		node_a_id: str,
		node_b_id: str,
		trust_weight: float = 0.5
	) -> bool:
		"""Establish trust relationship between nodes"""
		assert node_a_id in self.nodes, "Node A not found"
		assert node_b_id in self.nodes, "Node B not found"
		assert 0.0 <= trust_weight <= 1.0, "Trust weight must be between 0 and 1"
		
		try:
			# Verify mutual agreement through consensus
			proposal = ConsensusProposal(
				proposer_id=node_a_id,
				proposal_type="trust_establishment",
				content={
					"node_a": node_a_id,
					"node_b": node_b_id,
					"trust_weight": trust_weight
				},
				voting_deadline=datetime.utcnow() + timedelta(hours=24)
			)
			
			# Get endorsements from neighboring nodes
			neighbors_a = list(self.graph.neighbors(node_a_id))
			neighbors_b = list(self.graph.neighbors(node_b_id))
			voting_nodes = set(neighbors_a + neighbors_b + [node_a_id, node_b_id])
			
			consensus_reached = await self._execute_consensus(proposal, list(voting_nodes))
			
			if consensus_reached:
				# Establish trust relationship
				self.graph.add_edge(node_a_id, node_b_id, trust_weight=trust_weight)
				
				# Update trust scores
				await self._recalculate_trust_scores()
				
				# Update metrics
				self.mesh_metrics["trust_relationships"] = self.graph.number_of_edges()
				
				self._log_mesh_operation("trust_established", {
					"node_a": node_a_id,
					"node_b": node_b_id,
					"trust_weight": trust_weight
				})
				
				return True
			
			return False
			
		except Exception as e:
			self._log_mesh_operation("trust_establishment_error", {
				"node_a": node_a_id,
				"node_b": node_b_id,
				"error": str(e)
			})
			return False
	
	async def revoke_trust_relationship(
		self,
		node_a_id: str,
		node_b_id: str,
		reason: str = "manual_revocation"
	) -> bool:
		"""Revoke trust relationship between nodes"""
		assert node_a_id in self.nodes, "Node A not found"
		assert node_b_id in self.nodes, "Node B not found"
		
		try:
			if self.graph.has_edge(node_a_id, node_b_id):
				# Create revocation proposal
				proposal = ConsensusProposal(
					proposer_id=node_a_id,
					proposal_type="trust_revocation",
					content={
						"node_a": node_a_id,
						"node_b": node_b_id,
						"reason": reason
					},
					voting_deadline=datetime.utcnow() + timedelta(hours=12)
				)
				
				# Quick consensus for revocation (security priority)
				neighbors = list(set(self.graph.neighbors(node_a_id) + 
								   self.graph.neighbors(node_b_id)))
				
				consensus_reached = await self._execute_consensus(proposal, neighbors)
				
				if consensus_reached:
					self.graph.remove_edge(node_a_id, node_b_id)
					await self._recalculate_trust_scores()
					
					self._log_mesh_operation("trust_revoked", {
						"node_a": node_a_id,
						"node_b": node_b_id,
						"reason": reason
					})
					
					return True
			
			return False
			
		except Exception as e:
			self._log_mesh_operation("trust_revocation_error", {
				"node_a": node_a_id,
				"node_b": node_b_id,
				"error": str(e)
			})
			return False
	
	async def detect_mesh_anomalies(self) -> Dict[str, Any]:
		"""Detect anomalies in the mesh topology and behavior"""
		try:
			anomalies = {
				"isolated_nodes": [],
				"suspicious_patterns": [],
				"trust_inconsistencies": [],
				"performance_outliers": []
			}
			
			# Detect isolated nodes
			for node_id in self.nodes:
				if self.graph.degree(node_id) == 0:
					anomalies["isolated_nodes"].append(node_id)
			
			# Detect suspicious authentication patterns
			for node_id, node in self.nodes.items():
				metrics = TrustMetrics(**node.trust_metrics)
				
				# High failure rate
				total_auths = metrics.successful_authentications + metrics.failed_authentications
				if total_auths > 100:
					failure_rate = metrics.failed_authentications / total_auths
					if failure_rate > 0.3:
						anomalies["suspicious_patterns"].append({
							"node_id": node_id,
							"issue": "high_failure_rate",
							"rate": failure_rate
						})
				
				# Unusually high response time
				if metrics.average_response_time > 5000:  # 5 seconds
					anomalies["performance_outliers"].append({
						"node_id": node_id,
						"issue": "high_response_time",
						"time": metrics.average_response_time
					})
			
			# Detect trust inconsistencies
			for edge in self.graph.edges():
				node_a, node_b = edge
				trust_a = self.trust_scores.get(node_a, 0)
				trust_b = self.trust_scores.get(node_b, 0)
				
				# Large trust disparity
				if abs(trust_a - trust_b) > 0.5:
					anomalies["trust_inconsistencies"].append({
						"nodes": [node_a, node_b],
						"trust_disparity": abs(trust_a - trust_b)
					})
			
			self._log_mesh_operation("anomaly_detection_completed", {
				"isolated_nodes": len(anomalies["isolated_nodes"]),
				"suspicious_patterns": len(anomalies["suspicious_patterns"]),
				"trust_issues": len(anomalies["trust_inconsistencies"]),
				"performance_issues": len(anomalies["performance_outliers"])
			})
			
			return anomalies
			
		except Exception as e:
			self._log_mesh_operation("anomaly_detection_error", {"error": str(e)})
			return {}
	
	async def optimize_mesh_topology(self) -> Dict[str, Any]:
		"""Optimize mesh topology for better performance and trust distribution"""
		try:
			optimization_results = {
				"new_trust_relationships": [],
				"recommended_removals": [],
				"topology_improvements": {}
			}
			
			# Calculate centrality measures
			centrality = nx.degree_centrality(self.graph)
			betweenness = nx.betweenness_centrality(self.graph)
			
			# Identify highly central nodes (potential bottlenecks)
			high_centrality_nodes = [
				node for node, cent in centrality.items() 
				if cent > 0.7
			]
			
			# Recommend new connections for isolated or poorly connected nodes
			for node_id in self.nodes:
				if centrality.get(node_id, 0) < 0.2:  # Poorly connected
					# Find best candidates for connection
					candidates = await self._find_connection_candidates(node_id)
					optimization_results["new_trust_relationships"].extend(candidates)
			
			# Calculate network metrics
			optimization_results["topology_improvements"] = {
				"network_density": nx.density(self.graph),
				"average_clustering": nx.average_clustering(self.graph),
				"network_diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else "disconnected",
				"high_centrality_nodes": high_centrality_nodes
			}
			
			self._log_mesh_operation("mesh_optimization_completed", {
				"new_relationships": len(optimization_results["new_trust_relationships"]),
				"network_density": optimization_results["topology_improvements"]["network_density"]
			})
			
			return optimization_results
			
		except Exception as e:
			self._log_mesh_operation("mesh_optimization_error", {"error": str(e)})
			return {}
	
	# Helper methods
	
	async def _verify_node_credentials(self, node: MeshNode) -> bool:
		"""Verify node credentials"""
		try:
			# Verify public key format
			RSA.import_key(node.public_key)
			
			# Verify domain ownership (simplified)
			if not node.domain or len(node.domain) < 3:
				return False
			
			# Verify endpoints are accessible (simplified check)
			for endpoint_type, url in node.endpoints.items():
				if not url.startswith(('http://', 'https://')):
					return False
			
			return True
			
		except Exception:
			return False
	
	async def _calculate_endorsement_trust(self, endorsing_nodes: List[str]) -> float:
		"""Calculate trust score based on endorsements"""
		if not endorsing_nodes:
			return 0.0
		
		total_trust = 0.0
		valid_endorsers = 0
		
		for endorser_id in endorsing_nodes:
			if endorser_id in self.trust_scores:
				total_trust += self.trust_scores[endorser_id]
				valid_endorsers += 1
		
		return total_trust / valid_endorsers if valid_endorsers > 0 else 0.0
	
	def _trust_level_to_score(self, trust_level: TrustLevel) -> float:
		"""Convert trust level to numeric score"""
		mapping = {
			TrustLevel.UNKNOWN: 0.0,
			TrustLevel.LOW: 0.25,
			TrustLevel.MEDIUM: 0.5,
			TrustLevel.HIGH: 0.75,
			TrustLevel.VERIFIED: 1.0
		}
		return mapping.get(trust_level, 0.0)
	
	def _score_to_trust_level(self, score: float) -> TrustLevel:
		"""Convert numeric score to trust level"""
		if score >= 0.9:
			return TrustLevel.VERIFIED
		elif score >= 0.7:
			return TrustLevel.HIGH
		elif score >= 0.4:
			return TrustLevel.MEDIUM
		elif score >= 0.1:
			return TrustLevel.LOW
		else:
			return TrustLevel.UNKNOWN
	
	async def _find_trust_path(self, source: str, target: str) -> Optional[List[str]]:
		"""Find trust path between nodes"""
		try:
			if source == target:
				return [source]
			
			if not nx.has_path(self.graph, source, target):
				return None
			
			# Find shortest path weighted by trust
			path = nx.shortest_path(
				self.graph, 
				source, 
				target,
				weight=lambda u, v, d: 1.0 - d.get('trust_weight', 0.5)
			)
			
			return path
			
		except Exception:
			return None
	
	async def _calculate_path_trust(self, path: List[str]) -> float:
		"""Calculate trust score for a path"""
		if len(path) <= 1:
			return 1.0
		
		total_trust = 1.0
		
		for i in range(len(path) - 1):
			edge_data = self.graph.get_edge_data(path[i], path[i + 1])
			trust_weight = edge_data.get('trust_weight', 0.5) if edge_data else 0.1
			total_trust *= trust_weight
		
		return total_trust
	
	async def _create_assertion_proof(
		self, 
		node: MeshNode, 
		user_id: str, 
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Create cryptographic proof for assertion"""
		timestamp = datetime.utcnow().isoformat()
		nonce = uuid7str()
		
		# Create signature data
		signature_data = f"{node.id}:{user_id}:{timestamp}:{nonce}"
		
		# In production, this would use the node's private key
		signature = base64.b64encode(
			hmac.new(
				node.private_key_hash.encode(),
				signature_data.encode(),
				hashlib.sha256
			).digest()
		).decode()
		
		return {
			"signature": signature,
			"timestamp": timestamp,
			"nonce": nonce,
			"algorithm": "HMAC-SHA256"
		}
	
	async def _update_node_trust_metrics(self, node_id: str, success: bool) -> None:
		"""Update node trust metrics"""
		if node_id not in self.nodes:
			return
		
		node = self.nodes[node_id]
		metrics = TrustMetrics(**node.trust_metrics)
		
		if success:
			metrics.successful_authentications += 1
		else:
			metrics.failed_authentications += 1
		
		# Update reputation score
		total = metrics.successful_authentications + metrics.failed_authentications
		if total > 0:
			success_rate = metrics.successful_authentications / total
			metrics.reputation_score = success_rate * 0.8 + metrics.uptime_percentage * 0.2
		
		node.trust_metrics = metrics.__dict__
		
		# Update global trust score
		self.trust_scores[node_id] = metrics.reputation_score
	
	async def _execute_consensus(
		self, 
		proposal: ConsensusProposal, 
		voting_nodes: List[str]
	) -> bool:
		"""Execute consensus mechanism"""
		try:
			# Simplified consensus - in production would use proper distributed consensus
			positive_votes = 0
			total_votes = 0
			
			for node_id in voting_nodes:
				if node_id in self.nodes:
					node = self.nodes[node_id]
					# Simulate voting based on trust level and reputation
					trust_score = self.trust_scores.get(node_id, 0.5)
					
					# Higher trust nodes more likely to vote positively for legitimate proposals
					vote = trust_score > 0.6
					proposal.votes[node_id] = vote
					
					if vote:
						positive_votes += 1
					total_votes += 1
			
			# Check if consensus reached
			if total_votes > 0:
				consensus_ratio = positive_votes / total_votes
				consensus_reached = consensus_ratio >= proposal.required_consensus
				
				proposal.status = "approved" if consensus_reached else "rejected"
				
				# Record in history
				self.consensus_history.append({
					"proposal_id": proposal.id,
					"type": proposal.proposal_type,
					"status": proposal.status,
					"votes": total_votes,
					"positive_votes": positive_votes,
					"consensus_ratio": consensus_ratio,
					"timestamp": datetime.utcnow()
				})
				
				self.mesh_metrics["consensus_decisions"] += 1
				
				return consensus_reached
			
			return False
			
		except Exception as e:
			self._log_mesh_operation("consensus_error", {
				"proposal_id": proposal.id,
				"error": str(e)
			})
			return False
	
	async def _recalculate_trust_scores(self) -> None:
		"""Recalculate trust scores based on network topology"""
		try:
			# Use PageRank algorithm to calculate trust scores
			pagerank_scores = nx.pagerank(self.graph, weight='trust_weight')
			
			# Combine with existing reputation scores
			for node_id, pagerank_score in pagerank_scores.items():
				if node_id in self.nodes:
					node = self.nodes[node_id]
					metrics = TrustMetrics(**node.trust_metrics)
					
					# Weighted combination of PageRank and reputation
					combined_score = (
						pagerank_score * 0.4 +
						metrics.reputation_score * 0.6
					)
					
					self.trust_scores[node_id] = combined_score
			
		except Exception as e:
			self._log_mesh_operation("trust_recalculation_error", {"error": str(e)})
	
	async def _find_connection_candidates(self, node_id: str) -> List[Dict[str, Any]]:
		"""Find candidates for new trust relationships"""
		candidates = []
		
		if node_id not in self.nodes:
			return candidates
		
		node = self.nodes[node_id]
		current_connections = set(self.graph.neighbors(node_id))
		
		# Find nodes with similar capabilities or high trust
		for other_id, other_node in self.nodes.items():
			if other_id == node_id or other_id in current_connections:
				continue
			
			# Calculate compatibility score
			common_protocols = set(node.supported_protocols) & set(other_node.supported_protocols)
			protocol_score = len(common_protocols) / max(len(node.supported_protocols), 1)
			
			trust_score = self.trust_scores.get(other_id, 0)
			
			# Combined compatibility score
			compatibility = protocol_score * 0.6 + trust_score * 0.4
			
			if compatibility > 0.5:
				candidates.append({
					"node_id": other_id,
					"compatibility_score": compatibility,
					"common_protocols": len(common_protocols),
					"trust_score": trust_score
				})
		
		# Sort by compatibility score
		candidates.sort(key=lambda x: x["compatibility_score"], reverse=True)
		return candidates[:5]  # Top 5 candidates


# Usage example and testing functions

async def create_sample_mesh() -> FederatedIdentityMesh:
	"""Create a sample federated identity mesh for testing"""
	mesh = FederatedIdentityMesh()
	
	# Create sample nodes
	nodes = []
	for i in range(5):
		key = RSA.generate(2048)
		public_key = key.publickey().export_key().decode()
		private_key_hash = hashlib.sha256(key.export_key()).hexdigest()
		
		node = MeshNode(
			name=f"Node_{i}",
			domain=f"node{i}.example.com",
			public_key=public_key,
			private_key_hash=private_key_hash,
			supported_protocols=[FederationProtocol.OIDC, FederationProtocol.APG_MESH],
			trust_level=TrustLevel.MEDIUM,
			status=MeshNodeStatus.ACTIVE,
			endpoints={
				"auth": f"https://node{i}.example.com/auth",
				"federation": f"https://node{i}.example.com/federation"
			},
			capabilities=["authentication", "authorization", "federation"],
			trust_metrics=TrustMetrics(
				successful_authentications=100 + i * 50,
				failed_authentications=i * 5,
				average_response_time=200 + i * 100,
				uptime_percentage=98.0 + i * 0.4,
				reputation_score=0.8 + i * 0.04
			).__dict__
		)
		nodes.append(node)
	
	# Register nodes
	for i, node in enumerate(nodes):
		endorsers = [nodes[j].id for j in range(max(0, i-1), min(len(nodes), i+2)) if j != i]
		await mesh.register_node(node, endorsers)
	
	# Establish some trust relationships
	for i in range(len(nodes) - 1):
		await mesh.establish_trust_relationship(
			nodes[i].id, 
			nodes[i + 1].id, 
			0.7 + i * 0.05
		)
	
	return mesh


async def demo_federated_authentication():
	"""Demonstrate federated authentication capabilities"""
	print("=== Federated Identity Mesh Demo ===")
	
	# Create mesh
	mesh = await create_sample_mesh()
	
	print(f"Created mesh with {len(mesh.nodes)} nodes")
	print(f"Trust relationships: {mesh.graph.number_of_edges()}")
	
	# Test authentication
	node_ids = list(mesh.nodes.keys())
	if len(node_ids) >= 2:
		source_node = node_ids[0]
		target_node = node_ids[-1]
		
		assertion = await mesh.authenticate_user(
			user_id="user123",
			source_node_id=source_node,
			target_node_id=target_node,
			context={
				"attributes": {
					"role": "admin",
					"department": "engineering"
				},
				"authentication_method": "mfa"
			}
		)
		
		if assertion:
			print(f"Authentication successful!")
			print(f"Assertion ID: {assertion.id}")
			print(f"Trust level: {assertion.trust_level.value}")
			print(f"Trust chain length: {len(assertion.chain_of_trust)}")
		else:
			print("Authentication failed")
	
	# Test anomaly detection
	anomalies = await mesh.detect_mesh_anomalies()
	print(f"Detected anomalies: {sum(len(v) if isinstance(v, list) else 0 for v in anomalies.values())}")
	
	# Test optimization
	optimization = await mesh.optimize_mesh_topology()
	print(f"Network density: {optimization.get('topology_improvements', {}).get('network_density', 'N/A')}")
	
	print("=== Demo Complete ===")


if __name__ == "__main__":
	asyncio.run(demo_federated_authentication())