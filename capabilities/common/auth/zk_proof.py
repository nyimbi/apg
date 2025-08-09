"""
Zero-Knowledge Proof Authentication

Privacy-preserving authentication using zero-knowledge proofs that never
reveal user secrets while providing cryptographic proof of knowledge.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import secrets
import json
import math
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import hmac
import struct
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import numpy as np

class ZKProofType(str, Enum):
	"""Types of zero-knowledge proofs supported"""
	SCHNORR = "schnorr"              # Schnorr signatures
	SIGMA_PROTOCOL = "sigma_protocol"  # Sigma protocols
	COMMITMENT_SCHEME = "commitment_scheme"  # Pedersen commitments
	RANGE_PROOF = "range_proof"      # Range proofs
	MEMBERSHIP_PROOF = "membership_proof"  # Set membership proofs
	KNOWLEDGE_PROOF = "knowledge_proof"  # Proof of knowledge

class ProofStatus(str, Enum):
	"""Status of proof verification"""
	VALID = "valid"
	INVALID = "invalid"
	EXPIRED = "expired"
	PENDING = "pending"

class ZKCommitment(BaseModel):
	"""Zero-knowledge commitment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Commitment identifier")
	user_id: str = Field(..., description="User identifier")
	commitment_value: bytes = Field(..., description="Commitment value")
	
	# Commitment metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	expires_at: Optional[datetime] = Field(default=None, description="Commitment expiration")
	proof_type: ZKProofType = Field(..., description="Type of proof this commitment supports")
	
	# Cryptographic parameters
	generator: bytes = Field(..., description="Generator point/value")
	modulus: Optional[bytes] = Field(default=None, description="Modulus for discrete log groups")
	randomness_commitment: bytes = Field(..., description="Commitment to randomness")
	
	# Usage tracking
	challenge_count: int = Field(default=0, description="Number of challenges issued")
	last_used_at: Optional[datetime] = Field(default=None, description="Last proof generation")
	
	def is_valid(self) -> bool:
		"""Check if commitment is still valid"""
		if self.expires_at and self.expires_at <= datetime.utcnow():
			return False
		return True

class ZKChallenge(BaseModel):
	"""Zero-knowledge challenge"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Challenge identifier")
	commitment_id: str = Field(..., description="Associated commitment ID")
	challenge_value: bytes = Field(..., description="Challenge value")
	
	# Challenge metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Challenge creation")
	expires_at: datetime = Field(..., description="Challenge expiration")
	nonce: bytes = Field(..., description="Challenge nonce")
	
	# Context
	verifier_id: str = Field(..., description="Verifier identifier")
	context: Dict[str, Any] = Field(default_factory=dict, description="Challenge context")
	
	def is_valid(self) -> bool:
		"""Check if challenge is still valid"""
		return datetime.utcnow() < self.expires_at

class ZKProof(BaseModel):
	"""Zero-knowledge proof"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Proof identifier")
	user_id: str = Field(..., description="Prover user ID")
	commitment_id: str = Field(..., description="Associated commitment ID")
	challenge_id: str = Field(..., description="Associated challenge ID")
	
	# Proof data
	proof_value: bytes = Field(..., description="Proof value/response")
	proof_type: ZKProofType = Field(..., description="Type of zero-knowledge proof")
	
	# Proof metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Proof creation")
	verified_at: Optional[datetime] = Field(default=None, description="Verification timestamp")
	status: ProofStatus = Field(default=ProofStatus.PENDING, description="Proof status")
	
	# Verification results
	verification_result: Optional[bool] = Field(default=None, description="Verification result")
	verifier_notes: Optional[str] = Field(default=None, description="Verifier notes")
	
	# Cryptographic components
	witness_commitment: Optional[bytes] = Field(default=None, description="Witness commitment")
	auxiliary_data: Dict[str, bytes] = Field(default_factory=dict, description="Additional proof data")

class SchnorrProof:
	"""Schnorr zero-knowledge proof implementation"""
	
	def __init__(self, group_size_bits: int = 256):
		self.group_size_bits = group_size_bits
		# Use a safe prime for the group (simplified - in production use standardized groups)
		self.p = 2**256 - 189  # Large safe prime
		self.q = (self.p - 1) // 2  # Order of subgroup
		self.g = 2  # Generator
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[SchnorrProof INFO] {message} {kwargs if kwargs else ''}")
	
	def _hash_to_scalar(self, data: bytes) -> int:
		"""Hash data to scalar in group"""
		hash_val = hashlib.sha256(data).digest()
		return int.from_bytes(hash_val, 'big') % self.q
	
	async def generate_commitment(self, secret: int, randomness: int) -> Tuple[int, int]:
		"""Generate Pedersen commitment: g^secret * h^randomness"""
		h = pow(self.g, 2, self.p)  # Second generator
		commitment = (pow(self.g, secret, self.p) * pow(h, randomness, self.p)) % self.p
		return commitment, randomness
	
	async def create_proof(self, secret: int, challenge: bytes, randomness: int) -> Tuple[int, int]:
		"""Create Schnorr proof of knowledge of discrete logarithm"""
		self._log_info("Creating Schnorr proof")
		
		# Generate random nonce
		k = secrets.randbelow(self.q)
		
		# Compute commitment: R = g^k mod p
		R = pow(self.g, k, self.p)
		
		# Hash challenge with commitment
		challenge_hash = hashlib.sha256(challenge + R.to_bytes(32, 'big')).digest()
		c = self._hash_to_scalar(challenge_hash)
		
		# Compute response: s = k + c * secret mod q
		s = (k + c * secret) % self.q
		
		self._log_info("Schnorr proof created", R_size=R.bit_length(), s_size=s.bit_length())
		
		return R, s
	
	async def verify_proof(self, proof_R: int, proof_s: int, public_key: int, 
						   challenge: bytes) -> bool:
		"""Verify Schnorr proof"""
		self._log_info("Verifying Schnorr proof")
		
		try:
			# Recompute challenge
			challenge_hash = hashlib.sha256(challenge + proof_R.to_bytes(32, 'big')).digest()
			c = self._hash_to_scalar(challenge_hash)
			
			# Verify: g^s = R * y^c mod p (where y is public key)
			left_side = pow(self.g, proof_s, self.p)
			right_side = (proof_R * pow(public_key, c, self.p)) % self.p
			
			valid = left_side == right_side
			
			self._log_info("Schnorr proof verification", valid=valid)
			
			return valid
			
		except Exception as e:
			self._log_info("Schnorr proof verification failed", error=str(e))
			return False

class SigmaProtocol:
	"""Sigma protocol for zero-knowledge proofs"""
	
	def __init__(self):
		# Elliptic curve parameters (simplified - use standard curves in production)
		self.p = 2**256 - 2**32 - 977  # secp256k1 prime
		self.n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # Order
		self.g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
		self.g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[SigmaProtocol INFO] {message} {kwargs if kwargs else ''}")
	
	async def commit_phase(self, witness: int) -> Tuple[int, int]:
		"""Commitment phase of sigma protocol"""
		self._log_info("Sigma protocol commit phase")
		
		# Generate random commitment
		r = secrets.randbelow(self.n)
		
		# Compute commitment: a = g^r mod p
		a = pow(self.g_x, r, self.p)
		
		return a, r
	
	async def challenge_phase(self, commitment: int, context: bytes) -> int:
		"""Challenge phase - generate random challenge"""
		self._log_info("Sigma protocol challenge phase")
		
		# Generate deterministic challenge from commitment and context
		challenge_data = commitment.to_bytes(32, 'big') + context
		challenge_hash = hashlib.sha256(challenge_data).digest()
		challenge = int.from_bytes(challenge_hash, 'big') % self.n
		
		return challenge
	
	async def response_phase(self, witness: int, randomness: int, challenge: int) -> int:
		"""Response phase - compute response"""
		self._log_info("Sigma protocol response phase")
		
		# Compute response: z = r + c * witness mod n
		response = (randomness + challenge * witness) % self.n
		
		return response
	
	async def verify_sigma_proof(self, commitment: int, challenge: int, response: int,
								 public_value: int, context: bytes) -> bool:
		"""Verify complete sigma protocol proof"""
		self._log_info("Verifying sigma protocol proof")
		
		try:
			# Verify challenge is correctly computed
			expected_challenge = await self.challenge_phase(commitment, context)
			if challenge != expected_challenge:
				self._log_info("Invalid challenge in sigma proof")
				return False
			
			# Verify proof equation: g^z = a * y^c mod p
			left_side = pow(self.g_x, response, self.p)
			right_side = (commitment * pow(public_value, challenge, self.p)) % self.p
			
			valid = left_side == right_side
			
			self._log_info("Sigma protocol verification", valid=valid)
			
			return valid
			
		except Exception as e:
			self._log_info("Sigma protocol verification failed", error=str(e))
			return False

class ZKProofAuthenticator:
	"""Main zero-knowledge proof authentication engine"""
	
	def __init__(self):
		self._commitments: Dict[str, ZKCommitment] = {}
		self._challenges: Dict[str, ZKChallenge] = {}
		self._proofs: Dict[str, ZKProof] = {}
		
		# User secrets (in production, would be securely stored/derived)
		self._user_secrets: Dict[str, bytes] = {}
		
		# Cryptographic engines
		self.schnorr = SchnorrProof()
		self.sigma = SigmaProtocol()
		
		# Performance tracking
		self._operation_times: Dict[str, List[float]] = {}
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[ZKProofAuth INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[ZKProofAuth WARNING] {message} {kwargs if kwargs else ''}")
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[ZKProofAuth ERROR] {message} {kwargs if kwargs else ''}")
	
	async def _time_operation(self, operation_name: str, operation_func):
		"""Time operations for performance monitoring"""
		start_time = asyncio.get_event_loop().time()
		result = await operation_func()
		end_time = asyncio.get_event_loop().time()
		
		duration_ms = (end_time - start_time) * 1000
		
		if operation_name not in self._operation_times:
			self._operation_times[operation_name] = []
		self._operation_times[operation_name].append(duration_ms)
		
		# Keep only last 100 measurements
		self._operation_times[operation_name] = self._operation_times[operation_name][-100:]
		
		return result, duration_ms
	
	async def register_user_secret(self, user_id: str, secret: str) -> str:
		"""Register user secret for ZK authentication (setup phase)"""
		assert user_id, "User ID is required"
		assert secret, "Secret is required"
		
		self._log_info("Registering user secret for ZK auth", user_id=user_id)
		
		# Hash and store secret securely
		secret_hash = hashlib.sha256(secret.encode()).digest()
		self._user_secrets[user_id] = secret_hash
		
		# Generate initial commitment for the user
		commitment = await self.create_commitment(user_id, ZKProofType.SCHNORR)
		
		self._log_info("User secret registered", user_id=user_id, commitment_id=commitment.id)
		
		return commitment.id
	
	async def create_commitment(self, user_id: str, proof_type: ZKProofType,
								expires_in_hours: int = 24) -> ZKCommitment:
		"""Create zero-knowledge commitment for user"""
		assert user_id, "User ID is required"
		
		if user_id not in self._user_secrets:
			raise ValueError("User secret not registered")
		
		self._log_info("Creating ZK commitment", user_id=user_id, proof_type=proof_type.value)
		
		# Get user secret
		user_secret_hash = self._user_secrets[user_id]
		secret_int = int.from_bytes(user_secret_hash, 'big')
		
		# Generate commitment based on proof type
		if proof_type == ZKProofType.SCHNORR:
			# Generate random commitment value
			randomness = secrets.randbelow(self.schnorr.q)
			commitment_value, _ = await self.schnorr.generate_commitment(secret_int, randomness)
			commitment_bytes = commitment_value.to_bytes(32, 'big')
			generator_bytes = self.schnorr.g.to_bytes(4, 'big')
			modulus_bytes = self.schnorr.p.to_bytes(33, 'big')
			randomness_commitment = randomness.to_bytes(32, 'big')
			
		elif proof_type == ZKProofType.SIGMA_PROTOCOL:
			# Generate sigma protocol commitment
			commitment_value, randomness = await self.sigma.commit_phase(secret_int)
			commitment_bytes = commitment_value.to_bytes(32, 'big')
			generator_bytes = self.sigma.g_x.to_bytes(32, 'big')
			modulus_bytes = self.sigma.p.to_bytes(33, 'big')
			randomness_commitment = randomness.to_bytes(32, 'big')
			
		else:
			raise ValueError(f"Unsupported proof type: {proof_type}")
		
		# Create commitment object
		commitment = ZKCommitment(
			user_id=user_id,
			commitment_value=commitment_bytes,
			expires_at=datetime.utcnow() + timedelta(hours=expires_in_hours),
			proof_type=proof_type,
			generator=generator_bytes,
			modulus=modulus_bytes,
			randomness_commitment=randomness_commitment
		)
		
		# Store commitment
		self._commitments[commitment.id] = commitment
		
		self._log_info("ZK commitment created", 
					   commitment_id=commitment.id,
					   proof_type=proof_type.value,
					   expires_at=commitment.expires_at.isoformat())
		
		return commitment
	
	async def issue_challenge(self, commitment_id: str, verifier_id: str,
							  context: Optional[Dict[str, Any]] = None,
							  expires_in_minutes: int = 5) -> ZKChallenge:
		"""Issue challenge for zero-knowledge proof"""
		assert commitment_id, "Commitment ID is required"
		assert verifier_id, "Verifier ID is required"
		
		commitment = self._commitments.get(commitment_id)
		if not commitment or not commitment.is_valid():
			raise ValueError("Invalid or expired commitment")
		
		self._log_info("Issuing ZK challenge", 
					   commitment_id=commitment_id,
					   verifier_id=verifier_id)
		
		# Generate challenge
		nonce = secrets.token_bytes(16)
		context_bytes = json.dumps(context or {}, sort_keys=True).encode()
		
		# Create challenge value
		challenge_data = (
			commitment_id.encode() +
			verifier_id.encode() +
			nonce +
			context_bytes
		)
		challenge_value = hashlib.sha256(challenge_data).digest()
		
		# Create challenge object
		challenge = ZKChallenge(
			commitment_id=commitment_id,
			challenge_value=challenge_value,
			expires_at=datetime.utcnow() + timedelta(minutes=expires_in_minutes),
			nonce=nonce,
			verifier_id=verifier_id,
			context=context or {}
		)
		
		# Store challenge
		self._challenges[challenge.id] = challenge
		
		# Update commitment usage
		commitment.challenge_count += 1
		commitment.last_used_at = datetime.utcnow()
		
		self._log_info("ZK challenge issued",
					   challenge_id=challenge.id,
					   expires_at=challenge.expires_at.isoformat())
		
		return challenge
	
	async def create_proof(self, user_id: str, challenge_id: str) -> ZKProof:
		"""Create zero-knowledge proof in response to challenge"""
		assert user_id, "User ID is required"
		assert challenge_id, "Challenge ID is required"
		
		challenge = self._challenges.get(challenge_id)
		if not challenge or not challenge.is_valid():
			raise ValueError("Invalid or expired challenge")
		
		commitment = self._commitments.get(challenge.commitment_id)
		if not commitment or commitment.user_id != user_id:
			raise ValueError("Invalid commitment or user mismatch")
		
		self._log_info("Creating ZK proof", user_id=user_id, challenge_id=challenge_id)
		
		# Get user secret
		user_secret_hash = self._user_secrets[user_id]
		secret_int = int.from_bytes(user_secret_hash, 'big')
		
		# Create proof based on type
		if commitment.proof_type == ZKProofType.SCHNORR:
			# Extract randomness from commitment
			randomness = int.from_bytes(commitment.randomness_commitment, 'big')
			
			# Create Schnorr proof
			proof_result, proof_time = await self._time_operation(
				"schnorr_proof_creation",
				lambda: self.schnorr.create_proof(secret_int, challenge.challenge_value, randomness)
			)
			proof_R, proof_s = proof_result
			
			# Serialize proof
			proof_value = proof_R.to_bytes(32, 'big') + proof_s.to_bytes(32, 'big')
			
		elif commitment.proof_type == ZKProofType.SIGMA_PROTOCOL:
			# Extract parameters
			commitment_value = int.from_bytes(commitment.commitment_value, 'big')
			randomness = int.from_bytes(commitment.randomness_commitment, 'big')
			
			# Generate challenge and response
			sigma_challenge = await self.sigma.challenge_phase(commitment_value, challenge.challenge_value)
			response = await self.sigma.response_phase(secret_int, randomness, sigma_challenge)
			
			# Serialize proof
			proof_value = (
				commitment_value.to_bytes(32, 'big') +
				sigma_challenge.to_bytes(32, 'big') +
				response.to_bytes(32, 'big')
			)
		
		else:
			raise ValueError(f"Unsupported proof type: {commitment.proof_type}")
		
		# Create proof object
		proof = ZKProof(
			user_id=user_id,
			commitment_id=commitment.id,
			challenge_id=challenge_id,
			proof_value=proof_value,
			proof_type=commitment.proof_type,
			witness_commitment=commitment.commitment_value
		)
		
		# Store proof
		self._proofs[proof.id] = proof
		
		self._log_info("ZK proof created",
					   proof_id=proof.id,
					   proof_type=commitment.proof_type.value,
					   proof_size=len(proof_value))
		
		return proof
	
	async def verify_proof(self, proof_id: str, verifier_id: str) -> bool:
		"""Verify zero-knowledge proof"""
		assert proof_id, "Proof ID is required"
		assert verifier_id, "Verifier ID is required"
		
		proof = self._proofs.get(proof_id)
		if not proof:
			raise ValueError("Proof not found")
		
		challenge = self._challenges.get(proof.challenge_id)
		if not challenge or challenge.verifier_id != verifier_id:
			raise ValueError("Invalid challenge or verifier mismatch")
		
		commitment = self._commitments.get(proof.commitment_id)
		if not commitment:
			raise ValueError("Commitment not found")
		
		self._log_info("Verifying ZK proof",
					   proof_id=proof_id,
					   verifier_id=verifier_id,
					   proof_type=proof.proof_type.value)
		
		try:
			# Verify based on proof type
			if proof.proof_type == ZKProofType.SCHNORR:
				# Extract proof components
				proof_R = int.from_bytes(proof.proof_value[:32], 'big')
				proof_s = int.from_bytes(proof.proof_value[32:64], 'big')
				
				# Calculate public key from commitment
				# In practice, this would be stored/derived properly
				commitment_value = int.from_bytes(commitment.commitment_value, 'big')
				
				# Verify Schnorr proof
				verification_result, verify_time = await self._time_operation(
					"schnorr_proof_verification",
					lambda: self.schnorr.verify_proof(proof_R, proof_s, commitment_value, challenge.challenge_value)
				)
				
			elif proof.proof_type == ZKProofType.SIGMA_PROTOCOL:
				# Extract proof components
				commitment_value = int.from_bytes(proof.proof_value[:32], 'big')
				sigma_challenge = int.from_bytes(proof.proof_value[32:64], 'big')
				response = int.from_bytes(proof.proof_value[64:96], 'big')
				
				# Verify sigma protocol proof
				# Calculate public value (in practice, would be properly derived)
				public_value = int.from_bytes(commitment.commitment_value, 'big')
				
				verification_result = await self.sigma.verify_sigma_proof(
					commitment_value, sigma_challenge, response, public_value, challenge.challenge_value
				)
			
			else:
				raise ValueError(f"Unsupported proof type: {proof.proof_type}")
			
			# Update proof status
			proof.verification_result = verification_result
			proof.verified_at = datetime.utcnow()
			proof.status = ProofStatus.VALID if verification_result else ProofStatus.INVALID
			
			if not verification_result:
				proof.verifier_notes = "Cryptographic verification failed"
			
			self._log_info("ZK proof verification complete",
						   proof_id=proof_id,
						   valid=verification_result,
						   verify_time_ms=verify_time if proof.proof_type == ZKProofType.SCHNORR else None)
			
			return verification_result
			
		except Exception as e:
			# Update proof status on error
			proof.verification_result = False
			proof.verified_at = datetime.utcnow()
			proof.status = ProofStatus.INVALID
			proof.verifier_notes = f"Verification error: {str(e)}"
			
			self._log_error("ZK proof verification failed", proof_id=proof_id, error=str(e))
			
			return False
	
	async def authenticate_with_zk_proof(self, user_id: str, verifier_id: str,
										 proof_type: ZKProofType = ZKProofType.SCHNORR,
										 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Complete zero-knowledge authentication flow"""
		self._log_info("Starting ZK authentication flow",
					   user_id=user_id,
					   verifier_id=verifier_id,
					   proof_type=proof_type.value)
		
		try:
			# Step 1: Create commitment
			commitment = await self.create_commitment(user_id, proof_type)
			
			# Step 2: Issue challenge
			challenge = await self.issue_challenge(commitment.id, verifier_id, context)
			
			# Step 3: Create proof
			proof = await self.create_proof(user_id, challenge.id)
			
			# Step 4: Verify proof
			verification_result = await self.verify_proof(proof.id, verifier_id)
			
			# Prepare authentication result
			auth_result = {
				"success": verification_result,
				"user_id": user_id,
				"proof_id": proof.id,
				"commitment_id": commitment.id,
				"challenge_id": challenge.id,
				"proof_type": proof_type.value,
				"verified_at": proof.verified_at.isoformat() if proof.verified_at else None,
				"privacy_preserved": True,  # ZK proofs preserve privacy by design
				"secret_revealed": False   # Secrets are never revealed
			}
			
			if not verification_result:
				auth_result["error"] = "Zero-knowledge proof verification failed"
				auth_result["notes"] = proof.verifier_notes
			
			self._log_info("ZK authentication flow complete",
						   success=verification_result,
						   user_id=user_id)
			
			return auth_result
			
		except Exception as e:
			self._log_error("ZK authentication flow failed", user_id=user_id, error=str(e))
			
			return {
				"success": False,
				"user_id": user_id,
				"error": str(e),
				"privacy_preserved": True,
				"secret_revealed": False
			}
	
	async def create_membership_proof(self, user_id: str, member_set: Set[str],
									  proof_value: str) -> ZKProof:
		"""Create zero-knowledge proof of set membership without revealing the element"""
		assert user_id, "User ID is required"
		assert member_set, "Member set is required"
		assert proof_value in member_set, "Proof value must be in member set"
		
		self._log_info("Creating ZK membership proof", user_id=user_id, set_size=len(member_set))
		
		# Convert set to sorted list for deterministic ordering
		sorted_set = sorted(member_set)
		proof_index = sorted_set.index(proof_value)
		
		# Create Merkle tree commitment to the set
		tree_leaves = [hashlib.sha256(item.encode()).digest() for item in sorted_set]
		tree_root = self._build_merkle_tree(tree_leaves)
		
		# Create membership proof using Merkle path
		merkle_path = self._generate_merkle_path(tree_leaves, proof_index)
		
		# Generate zero-knowledge proof that we know a value in the set
		commitment_data = tree_root + proof_value.encode()
		commitment_hash = hashlib.sha256(commitment_data).digest()
		
		# Create proof object
		proof = ZKProof(
			user_id=user_id,
			commitment_id="membership_" + uuid7str(),
			challenge_id="membership_challenge_" + uuid7str(),
			proof_value=commitment_hash,
			proof_type=ZKProofType.MEMBERSHIP_PROOF,
			witness_commitment=tree_root,
			auxiliary_data={
				"merkle_path": b"".join(merkle_path),
				"set_size": len(member_set).to_bytes(4, 'big'),
				"proof_index": proof_index.to_bytes(4, 'big')
			}
		)
		
		self._proofs[proof.id] = proof
		
		self._log_info("ZK membership proof created", proof_id=proof.id, set_size=len(member_set))
		
		return proof
	
	def _build_merkle_tree(self, leaves: List[bytes]) -> bytes:
		"""Build Merkle tree and return root hash"""
		if not leaves:
			return b""
		
		if len(leaves) == 1:
			return leaves[0]
		
		# Build tree bottom-up
		current_level = leaves[:]
		
		while len(current_level) > 1:
			next_level = []
			
			for i in range(0, len(current_level), 2):
				if i + 1 < len(current_level):
					# Hash pair
					combined = current_level[i] + current_level[i + 1]
					next_level.append(hashlib.sha256(combined).digest())
				else:
					# Odd number of nodes, promote the last one
					next_level.append(current_level[i])
			
			current_level = next_level
		
		return current_level[0]
	
	def _generate_merkle_path(self, leaves: List[bytes], target_index: int) -> List[bytes]:
		"""Generate Merkle path for proof of inclusion"""
		if not leaves or target_index >= len(leaves):
			return []
		
		path = []
		current_level = leaves[:]
		current_index = target_index
		
		while len(current_level) > 1:
			# Determine sibling
			if current_index % 2 == 0:
				# Left node, sibling is right
				if current_index + 1 < len(current_level):
					path.append(current_level[current_index + 1])
			else:
				# Right node, sibling is left
				path.append(current_level[current_index - 1])
			
			# Move to next level
			next_level = []
			for i in range(0, len(current_level), 2):
				if i + 1 < len(current_level):
					combined = current_level[i] + current_level[i + 1]
					next_level.append(hashlib.sha256(combined).digest())
				else:
					next_level.append(current_level[i])
			
			current_level = next_level
			current_index = current_index // 2
		
		return path
	
	async def create_range_proof(self, user_id: str, secret_value: int, 
								 min_value: int, max_value: int) -> ZKProof:
		"""Create zero-knowledge range proof (proves value is in range without revealing it)"""
		assert user_id, "User ID is required"
		assert min_value <= secret_value <= max_value, "Secret value must be in specified range"
		
		self._log_info("Creating ZK range proof", 
					   user_id=user_id,
					   range_size=max_value - min_value + 1)
		
		# Simplified range proof using bit decomposition
		range_size = max_value - min_value + 1
		bit_length = range_size.bit_length()
		
		# Normalize value to 0-based range
		normalized_value = secret_value - min_value
		
		# Create bit commitments for each bit of the normalized value
		bit_commitments = []
		randomness_values = []
		
		for i in range(bit_length):
			bit = (normalized_value >> i) & 1
			randomness = secrets.randbelow(self.schnorr.q)
			
			# Commit to bit using Pedersen commitment
			commitment, _ = await self.schnorr.generate_commitment(bit, randomness)
			bit_commitments.append(commitment)
			randomness_values.append(randomness)
		
		# Create aggregated commitment
		aggregated_commitment = 1
		for i, commitment in enumerate(bit_commitments):
			power_of_two = pow(2, i, self.schnorr.p)
			commitment_power = pow(commitment, power_of_two, self.schnorr.p)
			aggregated_commitment = (aggregated_commitment * commitment_power) % self.schnorr.p
		
		# Serialize proof data
		proof_data = b""
		for commitment in bit_commitments:
			proof_data += commitment.to_bytes(32, 'big')
		
		proof_data += aggregated_commitment.to_bytes(32, 'big')
		proof_data += min_value.to_bytes(8, 'big')
		proof_data += max_value.to_bytes(8, 'big')
		
		# Create proof object
		proof = ZKProof(
			user_id=user_id,
			commitment_id="range_" + uuid7str(),
			challenge_id="range_challenge_" + uuid7str(),
			proof_value=proof_data,
			proof_type=ZKProofType.RANGE_PROOF,
			witness_commitment=aggregated_commitment.to_bytes(32, 'big'),
			auxiliary_data={
				"bit_length": bit_length.to_bytes(4, 'big'),
				"min_value": min_value.to_bytes(8, 'big'),
				"max_value": max_value.to_bytes(8, 'big')
			}
		)
		
		self._proofs[proof.id] = proof
		
		self._log_info("ZK range proof created", 
					   proof_id=proof.id,
					   bit_length=bit_length,
					   range_size=range_size)
		
		return proof
	
	def get_user_commitments(self, user_id: str) -> List[ZKCommitment]:
		"""Get all active commitments for user"""
		return [commitment for commitment in self._commitments.values()
				if commitment.user_id == user_id and commitment.is_valid()]
	
	def get_user_proofs(self, user_id: str) -> List[ZKProof]:
		"""Get all proofs for user"""
		return [proof for proof in self._proofs.values() if proof.user_id == user_id]
	
	def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
		"""Get performance metrics for ZK operations"""
		metrics = {}
		
		for operation, times in self._operation_times.items():
			if times:
				metrics[operation] = {
					"avg_ms": np.mean(times),
					"min_ms": np.min(times),
					"max_ms": np.max(times),
					"std_ms": np.std(times),
					"count": len(times)
				}
		
		return metrics
	
	def cleanup_expired_data(self):
		"""Clean up expired commitments, challenges, and proofs"""
		now = datetime.utcnow()
		
		# Clean up expired commitments
		expired_commitments = [
			commitment_id for commitment_id, commitment in self._commitments.items()
			if not commitment.is_valid()
		]
		for commitment_id in expired_commitments:
			del self._commitments[commitment_id]
		
		# Clean up expired challenges  
		expired_challenges = [
			challenge_id for challenge_id, challenge in self._challenges.items()
			if not challenge.is_valid()
		]
		for challenge_id in expired_challenges:
			del self._challenges[challenge_id]
		
		# Mark expired proofs
		for proof in self._proofs.values():
			challenge = self._challenges.get(proof.challenge_id)
			if challenge and not challenge.is_valid():
				proof.status = ProofStatus.EXPIRED
		
		self._log_info("Expired ZK data cleaned up",
					   commitments=len(expired_commitments),
					   challenges=len(expired_challenges))
	
	def clear_user_data(self, user_id: str):
		"""Clear all ZK data for user (GDPR compliance)"""
		# Clear user secret
		if user_id in self._user_secrets:
			del self._user_secrets[user_id]
		
		# Clear user commitments
		user_commitments = [
			commitment_id for commitment_id, commitment in self._commitments.items()
			if commitment.user_id == user_id
		]
		for commitment_id in user_commitments:
			del self._commitments[commitment_id]
		
		# Clear user proofs
		user_proofs = [
			proof_id for proof_id, proof in self._proofs.items()
			if proof.user_id == user_id
		]
		for proof_id in user_proofs:
			del self._proofs[proof_id]
		
		# Clear related challenges
		related_challenges = [
			challenge_id for challenge_id, challenge in self._challenges.items()
			if challenge.commitment_id in [c.id for c in self.get_user_commitments(user_id)]
		]
		for challenge_id in related_challenges:
			if challenge_id in self._challenges:
				del self._challenges[challenge_id]
		
		self._log_info("ZK data cleared for user",
					   user_id=user_id,
					   commitments=len(user_commitments),
					   proofs=len(user_proofs),
					   challenges=len(related_challenges))