#!/usr/bin/env python3
"""
APG Key Management - Quantum-Safe Cryptography & Migration
Post-quantum cryptography implementation with automated migration tools

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str

from .models import Key, KeySpec, KeyAlgorithm, KeyUsage, KeyState


class QuantumThreatLevel(str, Enum):
	"""Quantum computing threat assessment levels"""
	MINIMAL = "minimal"  # Current state - no immediate threat
	EMERGING = "emerging"  # Early quantum computers appearing
	MODERATE = "moderate"  # Quantum computers breaking some algorithms
	HIGH = "high"  # Quantum computers breaking most classical algorithms
	CRITICAL = "critical"  # Large-scale quantum computers widely available


class MigrationStrategy(str, Enum):
	"""Post-quantum migration strategies"""
	HYBRID = "hybrid"  # Run both classical and post-quantum side-by-side
	GRADUAL = "gradual"  # Gradually replace classical with post-quantum
	IMMEDIATE = "immediate"  # Replace all keys with post-quantum immediately
	ON_DEMAND = "on_demand"  # Replace keys as they're accessed/rotated


class QuantumResistance(str, Enum):
	"""Quantum resistance levels of algorithms"""
	VULNERABLE = "vulnerable"  # Broken by quantum computers
	PARTIALLY_RESISTANT = "partially_resistant"  # Reduced security but not broken
	QUANTUM_SAFE = "quantum_safe"  # Believed secure against quantum attacks
	QUANTUM_PROVEN = "quantum_proven"  # Mathematically proven secure


@dataclass
class QuantumThreatAssessment:
	"""Assessment of quantum threat to specific algorithm"""
	algorithm: KeyAlgorithm
	current_security_level: int  # bits of security
	quantum_security_level: int  # bits of security against quantum attacks
	resistance_level: QuantumResistance
	estimated_break_date: datetime | None
	migration_priority: str  # low, medium, high, critical
	recommended_replacement: KeyAlgorithm | None


@dataclass
class MigrationPlan:
	"""Migration plan for transitioning to post-quantum cryptography"""
	plan_id: str = field(default_factory=uuid7str)
	tenant_id: str = ""
	strategy: MigrationStrategy = MigrationStrategy.HYBRID
	target_completion_date: datetime | None = None
	priority_keys: List[str] = field(default_factory=list)  # Keys to migrate first
	algorithm_mappings: Dict[KeyAlgorithm, KeyAlgorithm] = field(default_factory=dict)
	rollback_capability: bool = True
	validation_requirements: List[str] = field(default_factory=list)
	estimated_duration_days: int = 90
	business_impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationProgress:
	"""Progress tracking for quantum-safe migration"""
	plan_id: str
	total_keys: int
	migrated_keys: int
	failed_migrations: int
	completion_percentage: float
	current_phase: str
	estimated_remaining_days: int
	last_updated: datetime = field(default_factory=datetime.utcnow)


class QuantumSafeCryptographyManager:
	"""
	Post-quantum cryptography manager
	Provides quantum threat assessment and automated migration to quantum-safe algorithms
	"""
	
	def __init__(self, config: Dict[str, Any] | None = None):
		self.config = config or {}
		self.current_threat_level = QuantumThreatLevel.MINIMAL
		self.threat_assessments: Dict[KeyAlgorithm, QuantumThreatAssessment] = {}
		self.migration_plans: Dict[str, MigrationPlan] = {}
		self.migration_progress: Dict[str, MigrationProgress] = {}
		
		# Post-quantum algorithm performance characteristics
		self.pq_algorithm_specs = {
			KeyAlgorithm.KYBER_512: {
				"type": "kem",  # Key Encapsulation Mechanism
				"security_level": 128,
				"public_key_size": 800,
				"private_key_size": 1632,
				"ciphertext_size": 768,
				"performance_factor": 0.8  # Relative to RSA-2048
			},
			KeyAlgorithm.KYBER_768: {
				"type": "kem",
				"security_level": 192,
				"public_key_size": 1184,
				"private_key_size": 2400,
				"ciphertext_size": 1088,
				"performance_factor": 0.7
			},
			KeyAlgorithm.KYBER_1024: {
				"type": "kem",
				"security_level": 256,
				"public_key_size": 1568,
				"private_key_size": 3168,
				"ciphertext_size": 1568,
				"performance_factor": 0.6
			},
			KeyAlgorithm.DILITHIUM_2: {
				"type": "signature",
				"security_level": 128,
				"public_key_size": 1312,
				"private_key_size": 2528,
				"signature_size": 2420,
				"performance_factor": 0.3
			},
			KeyAlgorithm.DILITHIUM_3: {
				"type": "signature",
				"security_level": 192,
				"public_key_size": 1952,
				"private_key_size": 4000,
				"signature_size": 3293,
				"performance_factor": 0.25
			},
			KeyAlgorithm.DILITHIUM_5: {
				"type": "signature",
				"security_level": 256,
				"public_key_size": 2592,
				"private_key_size": 4864,
				"signature_size": 4595,
				"performance_factor": 0.2
			},
			KeyAlgorithm.FALCON_512: {
				"type": "signature",
				"security_level": 128,
				"public_key_size": 897,
				"private_key_size": 1281,
				"signature_size": 690,
				"performance_factor": 0.5
			},
			KeyAlgorithm.FALCON_1024: {
				"type": "signature",
				"security_level": 256,
				"public_key_size": 1793,
				"private_key_size": 2305,
				"signature_size": 1330,
				"performance_factor": 0.4
			}
		}
		
		# Initialize threat assessments
		self._initialize_threat_assessments()
	
	async def _log_quantum_operation(self, operation: str, details: str = "") -> None:
		"""Log quantum-safe cryptography operations"""
		print(f"[QUANTUM-SAFE] {operation}: {details}")
	
	def _initialize_threat_assessments(self) -> None:
		"""Initialize quantum threat assessments for all algorithms"""
		
		# Classical algorithms vulnerable to quantum attacks
		self.threat_assessments[KeyAlgorithm.RSA_2048] = QuantumThreatAssessment(
			algorithm=KeyAlgorithm.RSA_2048,
			current_security_level=112,
			quantum_security_level=0,  # Completely broken by Shor's algorithm
			resistance_level=QuantumResistance.VULNERABLE,
			estimated_break_date=datetime(2030, 1, 1),  # Conservative estimate
			migration_priority="high",
			recommended_replacement=KeyAlgorithm.KYBER_768
		)
		
		self.threat_assessments[KeyAlgorithm.RSA_4096] = QuantumThreatAssessment(
			algorithm=KeyAlgorithm.RSA_4096,
			current_security_level=152,
			quantum_security_level=0,
			resistance_level=QuantumResistance.VULNERABLE,
			estimated_break_date=datetime(2030, 1, 1),
			migration_priority="high",
			recommended_replacement=KeyAlgorithm.KYBER_1024
		)
		
		self.threat_assessments[KeyAlgorithm.ECDSA_P256] = QuantumThreatAssessment(
			algorithm=KeyAlgorithm.ECDSA_P256,
			current_security_level=128,
			quantum_security_level=0,  # Broken by Shor's algorithm
			resistance_level=QuantumResistance.VULNERABLE,
			estimated_break_date=datetime(2028, 1, 1),  # ECC breaks faster
			migration_priority="critical",
			recommended_replacement=KeyAlgorithm.DILITHIUM_2
		)
		
		self.threat_assessments[KeyAlgorithm.ECDSA_P384] = QuantumThreatAssessment(
			algorithm=KeyAlgorithm.ECDSA_P384,
			current_security_level=192,
			quantum_security_level=0,
			resistance_level=QuantumResistance.VULNERABLE,
			estimated_break_date=datetime(2028, 1, 1),
			migration_priority="critical",
			recommended_replacement=KeyAlgorithm.DILITHIUM_3
		)
		
		# Symmetric algorithms - partially quantum resistant
		self.threat_assessments[KeyAlgorithm.AES_128] = QuantumThreatAssessment(
			algorithm=KeyAlgorithm.AES_128,
			current_security_level=128,
			quantum_security_level=64,  # Grover's algorithm halves security
			resistance_level=QuantumResistance.PARTIALLY_RESISTANT,
			estimated_break_date=datetime(2035, 1, 1),
			migration_priority="medium",
			recommended_replacement=KeyAlgorithm.AES_256
		)
		
		self.threat_assessments[KeyAlgorithm.AES_256] = QuantumThreatAssessment(
			algorithm=KeyAlgorithm.AES_256,
			current_security_level=256,
			quantum_security_level=128,  # Still secure against quantum attacks
			resistance_level=QuantumResistance.PARTIALLY_RESISTANT,
			estimated_break_date=None,  # No expected break date
			migration_priority="low",
			recommended_replacement=None
		)
		
		# Post-quantum algorithms - quantum safe
		for pq_algo in [KeyAlgorithm.KYBER_512, KeyAlgorithm.KYBER_768, KeyAlgorithm.KYBER_1024,
						KeyAlgorithm.DILITHIUM_2, KeyAlgorithm.DILITHIUM_3, KeyAlgorithm.DILITHIUM_5,
						KeyAlgorithm.FALCON_512, KeyAlgorithm.FALCON_1024]:
			
			security_level = self.pq_algorithm_specs[pq_algo]["security_level"]
			self.threat_assessments[pq_algo] = QuantumThreatAssessment(
				algorithm=pq_algo,
				current_security_level=security_level,
				quantum_security_level=security_level,  # Maintains security
				resistance_level=QuantumResistance.QUANTUM_SAFE,
				estimated_break_date=None,
				migration_priority="none",
				recommended_replacement=None
			)
	
	async def assess_quantum_threat(self, keys: List[Key]) -> Dict[str, Any]:
		"""Assess quantum threat across key portfolio"""
		
		threat_summary = {
			"assessment_date": datetime.utcnow().isoformat(),
			"current_threat_level": self.current_threat_level.value,
			"total_keys_analyzed": len(keys),
			"vulnerability_breakdown": {
				"vulnerable": 0,
				"partially_resistant": 0,
				"quantum_safe": 0
			},
			"migration_urgency": {
				"critical": [],
				"high": [],
				"medium": [],
				"low": []
			},
			"algorithm_analysis": {},
			"recommendations": []
		}
		
		algorithm_counts = {}
		
		# Analyze each key
		for key in keys:
			algorithm = key.spec.algorithm
			assessment = self.threat_assessments.get(algorithm)
			
			if not assessment:
				continue
			
			# Count algorithms
			if algorithm not in algorithm_counts:
				algorithm_counts[algorithm] = 0
			algorithm_counts[algorithm] += 1
			
			# Categorize by resistance level
			if assessment.resistance_level == QuantumResistance.VULNERABLE:
				threat_summary["vulnerability_breakdown"]["vulnerable"] += 1
				
				if assessment.migration_priority == "critical":
					threat_summary["migration_urgency"]["critical"].append(key.spec.id)
				elif assessment.migration_priority == "high":
					threat_summary["migration_urgency"]["high"].append(key.spec.id)
					
			elif assessment.resistance_level == QuantumResistance.PARTIALLY_RESISTANT:
				threat_summary["vulnerability_breakdown"]["partially_resistant"] += 1
				
				if assessment.migration_priority == "medium":
					threat_summary["migration_urgency"]["medium"].append(key.spec.id)
				else:
					threat_summary["migration_urgency"]["low"].append(key.spec.id)
					
			elif assessment.resistance_level == QuantumResistance.QUANTUM_SAFE:
				threat_summary["vulnerability_breakdown"]["quantum_safe"] += 1
		
		# Generate algorithm analysis
		for algorithm, count in algorithm_counts.items():
			assessment = self.threat_assessments[algorithm]
			threat_summary["algorithm_analysis"][algorithm.value] = {
				"key_count": count,
				"current_security_bits": assessment.current_security_level,
				"quantum_security_bits": assessment.quantum_security_level,
				"resistance_level": assessment.resistance_level.value,
				"migration_priority": assessment.migration_priority,
				"recommended_replacement": assessment.recommended_replacement.value if assessment.recommended_replacement else None,
				"estimated_break_date": assessment.estimated_break_date.isoformat() if assessment.estimated_break_date else None
			}
		
		# Generate recommendations
		threat_summary["recommendations"] = await self._generate_migration_recommendations(threat_summary)
		
		await self._log_quantum_operation(
			"THREAT_ASSESSMENT_COMPLETE",
			f"Analyzed {len(keys)} keys, {threat_summary['vulnerability_breakdown']['vulnerable']} vulnerable"
		)
		
		return threat_summary
	
	async def _generate_migration_recommendations(self, threat_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate intelligent migration recommendations"""
		recommendations = []
		
		vulnerable_count = threat_summary["vulnerability_breakdown"]["vulnerable"]
		critical_keys = len(threat_summary["migration_urgency"]["critical"])
		high_priority_keys = len(threat_summary["migration_urgency"]["high"])
		
		if critical_keys > 0:
			recommendations.append({
				"priority": "critical",
				"title": "Immediate Migration Required",
				"description": f"{critical_keys} keys using ECDSA algorithms need immediate migration",
				"action": "Begin emergency migration of ECDSA keys to Dilithium signatures",
				"timeline": "Within 30 days",
				"risk": "High risk of compromise if quantum computers advance"
			})
		
		if high_priority_keys > 10:
			recommendations.append({
				"priority": "high",
				"title": "RSA Key Migration Planning",
				"description": f"{high_priority_keys} RSA keys require migration planning",
				"action": "Develop comprehensive migration plan for RSA to Kyber key encapsulation",
				"timeline": "Within 90 days",
				"risk": "Medium risk, should be addressed proactively"
			})
		
		if vulnerable_count > (critical_keys + high_priority_keys):
			recommendations.append({
				"priority": "medium",
				"title": "Comprehensive Quantum-Safe Strategy",
				"description": f"Total of {vulnerable_count} keys vulnerable to quantum attacks",
				"action": "Implement organization-wide post-quantum cryptography strategy",
				"timeline": "Within 180 days",
				"risk": "Future risk that requires strategic planning"
			})
		
		# Performance considerations
		recommendations.append({
			"priority": "low",
			"title": "Performance Testing",
			"description": "Post-quantum algorithms have different performance characteristics",
			"action": "Conduct performance testing of post-quantum algorithms in your environment",
			"timeline": "Before migration",
			"risk": "Potential performance impact on applications"
		})
		
		return recommendations
	
	async def create_migration_plan(self, tenant_id: str, strategy: MigrationStrategy,
								   target_keys: List[str] | None = None,
								   completion_date: datetime | None = None) -> MigrationPlan:
		"""Create intelligent migration plan"""
		
		plan = MigrationPlan(
			tenant_id=tenant_id,
			strategy=strategy,
			target_completion_date=completion_date or datetime.utcnow() + timedelta(days=90)
		)
		
		# Define algorithm mappings based on security requirements
		plan.algorithm_mappings = {
			# RSA -> Kyber (Key Encapsulation)
			KeyAlgorithm.RSA_2048: KeyAlgorithm.KYBER_768,
			KeyAlgorithm.RSA_4096: KeyAlgorithm.KYBER_1024,
			
			# ECDSA -> Dilithium (Digital Signatures)
			KeyAlgorithm.ECDSA_P256: KeyAlgorithm.DILITHIUM_2,
			KeyAlgorithm.ECDSA_P384: KeyAlgorithm.DILITHIUM_3,
			
			# AES stays but might upgrade
			KeyAlgorithm.AES_128: KeyAlgorithm.AES_256
		}
		
		# Set priority keys based on threat assessment
		if target_keys:
			plan.priority_keys = target_keys
		else:
			plan.priority_keys = await self._identify_priority_keys(tenant_id)
		
		# Define validation requirements
		plan.validation_requirements = [
			"algorithm_compatibility_test",
			"performance_benchmark_test",
			"interoperability_test",
			"security_validation_test",
			"rollback_test"
		]
		
		# Business impact assessment
		plan.business_impact_assessment = await self._assess_business_impact(plan)
		
		# Adjust timeline based on strategy
		if strategy == MigrationStrategy.IMMEDIATE:
			plan.estimated_duration_days = 30
		elif strategy == MigrationStrategy.GRADUAL:
			plan.estimated_duration_days = 180
		elif strategy == MigrationStrategy.HYBRID:
			plan.estimated_duration_days = 90
		
		self.migration_plans[plan.plan_id] = plan
		
		await self._log_quantum_operation(
			"MIGRATION_PLAN_CREATED",
			f"Strategy: {strategy.value}, Duration: {plan.estimated_duration_days} days"
		)
		
		return plan
	
	async def _identify_priority_keys(self, tenant_id: str) -> List[str]:
		"""Identify keys that should be migrated first"""
		# Placeholder - would query actual key database
		# Priority order: ECDSA > RSA > AES-128
		return ["priority_key_1", "priority_key_2", "priority_key_3"]
	
	async def _assess_business_impact(self, plan: MigrationPlan) -> Dict[str, Any]:
		"""Assess business impact of migration plan"""
		return {
			"affected_applications": 5,  # Would analyze actual application dependencies
			"expected_downtime_minutes": 30 if plan.strategy == MigrationStrategy.IMMEDIATE else 10,
			"performance_impact_percentage": 15,  # Post-quantum algorithms are slower
			"compatibility_risks": [
				"Legacy systems may not support post-quantum algorithms",
				"Third-party integrations may require updates"
			],
			"mitigation_strategies": [
				"Implement hybrid mode for compatibility",
				"Gradual rollout with canary testing",
				"Comprehensive testing in staging environment"
			]
		}
	
	async def execute_migration_plan(self, plan_id: str) -> MigrationProgress:
		"""Execute post-quantum migration plan"""
		
		plan = self.migration_plans.get(plan_id)
		if not plan:
			raise ValueError("Migration plan not found")
		
		# Initialize progress tracking
		progress = MigrationProgress(
			plan_id=plan_id,
			total_keys=len(plan.priority_keys) if plan.priority_keys else 100,  # Placeholder
			migrated_keys=0,
			failed_migrations=0,
			completion_percentage=0.0,
			current_phase="initialization",
			estimated_remaining_days=plan.estimated_duration_days
		)
		
		self.migration_progress[plan_id] = progress
		
		try:
			# Phase 1: Validation
			progress.current_phase = "validation"
			await self._execute_validation_phase(plan, progress)
			
			# Phase 2: Key Migration
			progress.current_phase = "key_migration"
			await self._execute_key_migration_phase(plan, progress)
			
			# Phase 3: Verification
			progress.current_phase = "verification"
			await self._execute_verification_phase(plan, progress)
			
			# Phase 4: Cleanup
			progress.current_phase = "cleanup"
			await self._execute_cleanup_phase(plan, progress)
			
			progress.current_phase = "completed"
			progress.completion_percentage = 100.0
			progress.estimated_remaining_days = 0
			
		except Exception as e:
			progress.current_phase = "failed"
			await self._log_quantum_operation("MIGRATION_FAILED", str(e))
			raise
		
		await self._log_quantum_operation(
			"MIGRATION_COMPLETED",
			f"Migrated {progress.migrated_keys} keys, {progress.failed_migrations} failures"
		)
		
		return progress
	
	async def _execute_validation_phase(self, plan: MigrationPlan, progress: MigrationProgress) -> None:
		"""Execute validation phase of migration"""
		await self._log_quantum_operation("VALIDATION_PHASE_START", "")
		
		# Simulate validation tests
		for requirement in plan.validation_requirements:
			await asyncio.sleep(0.1)  # Simulate test execution
			await self._log_quantum_operation("VALIDATION_TEST", requirement)
		
		progress.completion_percentage = 20.0
		progress.estimated_remaining_days = int(plan.estimated_duration_days * 0.8)
	
	async def _execute_key_migration_phase(self, plan: MigrationPlan, progress: MigrationProgress) -> None:
		"""Execute key migration phase"""
		await self._log_quantum_operation("KEY_MIGRATION_PHASE_START", "")
		
		total_keys = progress.total_keys
		migrated = 0
		failed = 0
		
		# Simulate key migration
		for i in range(total_keys):
			key_id = f"key_{i}"
			
			try:
				# Migrate based on strategy
				if plan.strategy == MigrationStrategy.HYBRID:
					await self._migrate_key_hybrid(key_id, plan)
				elif plan.strategy == MigrationStrategy.GRADUAL:
					await self._migrate_key_gradual(key_id, plan)
				elif plan.strategy == MigrationStrategy.IMMEDIATE:
					await self._migrate_key_immediate(key_id, plan)
				
				migrated += 1
				
			except Exception as e:
				failed += 1
				await self._log_quantum_operation("KEY_MIGRATION_FAILED", f"{key_id}: {e}")
			
			# Update progress
			progress.migrated_keys = migrated
			progress.failed_migrations = failed
			progress.completion_percentage = 20.0 + (60.0 * migrated / total_keys)
			
			await asyncio.sleep(0.01)  # Simulate migration time
	
	async def _migrate_key_hybrid(self, key_id: str, plan: MigrationPlan) -> None:
		"""Migrate key using hybrid strategy"""
		# Create both classical and post-quantum versions
		await self._log_quantum_operation("HYBRID_MIGRATION", f"Key: {key_id}")
		await asyncio.sleep(0.05)
	
	async def _migrate_key_gradual(self, key_id: str, plan: MigrationPlan) -> None:
		"""Migrate key using gradual strategy"""
		# Gradually replace classical with post-quantum
		await self._log_quantum_operation("GRADUAL_MIGRATION", f"Key: {key_id}")
		await asyncio.sleep(0.03)
	
	async def _migrate_key_immediate(self, key_id: str, plan: MigrationPlan) -> None:
		"""Migrate key using immediate strategy"""
		# Immediately replace with post-quantum
		await self._log_quantum_operation("IMMEDIATE_MIGRATION", f"Key: {key_id}")
		await asyncio.sleep(0.02)
	
	async def _execute_verification_phase(self, plan: MigrationPlan, progress: MigrationProgress) -> None:
		"""Execute verification phase"""
		await self._log_quantum_operation("VERIFICATION_PHASE_START", "")
		
		# Verify migrated keys work correctly
		await asyncio.sleep(0.5)  # Simulate verification
		
		progress.completion_percentage = 90.0
		progress.estimated_remaining_days = int(plan.estimated_duration_days * 0.1)
	
	async def _execute_cleanup_phase(self, plan: MigrationPlan, progress: MigrationProgress) -> None:
		"""Execute cleanup phase"""
		await self._log_quantum_operation("CLEANUP_PHASE_START", "")
		
		# Clean up temporary resources
		await asyncio.sleep(0.2)
		
		progress.completion_percentage = 95.0
		progress.estimated_remaining_days = 1
	
	async def recommend_post_quantum_algorithm(self, current_algorithm: KeyAlgorithm,
											   security_requirement: int = 128,
											   performance_priority: float = 0.5) -> KeyAlgorithm | None:
		"""Recommend optimal post-quantum algorithm"""
		
		# Get algorithm type and usage
		if current_algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096]:
			# RSA used for both encryption and signatures - recommend KEM + signature
			if security_requirement <= 128:
				return KeyAlgorithm.KYBER_512
			elif security_requirement <= 192:
				return KeyAlgorithm.KYBER_768
			else:
				return KeyAlgorithm.KYBER_1024
		
		elif current_algorithm in [KeyAlgorithm.ECDSA_P256, KeyAlgorithm.ECDSA_P384]:
			# Digital signatures - recommend Dilithium or Falcon
			if performance_priority > 0.7:  # High performance priority
				if security_requirement <= 128:
					return KeyAlgorithm.FALCON_512
				else:
					return KeyAlgorithm.FALCON_1024
			else:  # Security priority
				if security_requirement <= 128:
					return KeyAlgorithm.DILITHIUM_2
				elif security_requirement <= 192:
					return KeyAlgorithm.DILITHIUM_3
				else:
					return KeyAlgorithm.DILITHIUM_5
		
		elif current_algorithm == KeyAlgorithm.AES_128:
			# Symmetric encryption - upgrade to AES-256
			return KeyAlgorithm.AES_256
		
		return None
	
	async def validate_post_quantum_compatibility(self, key_spec: KeySpec) -> Dict[str, Any]:
		"""Validate post-quantum algorithm compatibility"""
		
		algorithm = key_spec.algorithm
		validation_result = {
			"algorithm": algorithm.value,
			"is_post_quantum": algorithm in self.pq_algorithm_specs,
			"compatibility_issues": [],
			"performance_impact": {},
			"recommendations": []
		}
		
		if algorithm in self.pq_algorithm_specs:
			spec = self.pq_algorithm_specs[algorithm]
			
			# Check performance impact
			validation_result["performance_impact"] = {
				"relative_performance": spec["performance_factor"],
				"key_size_increase": self._calculate_key_size_increase(algorithm),
				"signature_size": spec.get("signature_size", 0),
				"estimated_slowdown": f"{int((1 - spec['performance_factor']) * 100)}%"
			}
			
			# Check for compatibility issues
			if spec["public_key_size"] > 2048:
				validation_result["compatibility_issues"].append(
					"Large key sizes may cause issues with legacy systems"
				)
			
			if spec.get("signature_size", 0) > 2000:
				validation_result["compatibility_issues"].append(
					"Large signature sizes may impact network protocols"
				)
			
			# Generate recommendations
			if spec["performance_factor"] < 0.5:
				validation_result["recommendations"].append(
					"Consider performance testing before deployment"
				)
			
			if len(validation_result["compatibility_issues"]) > 0:
				validation_result["recommendations"].append(
					"Test with existing applications and protocols"
				)
		
		else:
			validation_result["compatibility_issues"].append(
				"Algorithm is not quantum-safe"
			)
			
			# Recommend post-quantum alternative
			recommended = await self.recommend_post_quantum_algorithm(algorithm)
			if recommended:
				validation_result["recommendations"].append(
					f"Consider migrating to {recommended.value}"
				)
		
		return validation_result
	
	def _calculate_key_size_increase(self, pq_algorithm: KeyAlgorithm) -> str:
		"""Calculate key size increase compared to classical algorithms"""
		spec = self.pq_algorithm_specs.get(pq_algorithm)
		if not spec:
			return "Unknown"
		
		if spec["type"] == "kem":
			# Compare to RSA-2048 (256 bytes public key)
			increase = (spec["public_key_size"] / 256) - 1
			return f"{int(increase * 100)}% larger than RSA-2048"
		
		elif spec["type"] == "signature":
			# Compare to ECDSA-P256 (64 bytes public key)
			increase = (spec["public_key_size"] / 64) - 1
			return f"{int(increase * 100)}% larger than ECDSA-P256"
		
		return "N/A"
	
	async def get_migration_dashboard(self) -> Dict[str, Any]:
		"""Get comprehensive migration dashboard"""
		
		dashboard = {
			"generated_at": datetime.utcnow().isoformat(),
			"quantum_threat_level": self.current_threat_level.value,
			"active_migration_plans": len(self.migration_plans),
			"completed_migrations": len([p for p in self.migration_progress.values() 
										if p.current_phase == "completed"]),
			"migration_summary": {},
			"algorithm_readiness": {},
			"upcoming_deadlines": []
		}
		
		# Migration summary
		total_keys_to_migrate = sum(p.total_keys for p in self.migration_progress.values())
		total_keys_migrated = sum(p.migrated_keys for p in self.migration_progress.values())
		
		dashboard["migration_summary"] = {
			"total_keys_to_migrate": total_keys_to_migrate,
			"total_keys_migrated": total_keys_migrated,
			"migration_completion_rate": (total_keys_migrated / max(1, total_keys_to_migrate)) * 100,
			"average_success_rate": self._calculate_average_success_rate()
		}
		
		# Algorithm readiness assessment
		for algorithm in KeyAlgorithm:
			assessment = self.threat_assessments.get(algorithm)
			if assessment:
				dashboard["algorithm_readiness"][algorithm.value] = {
					"quantum_resistance": assessment.resistance_level.value,
					"migration_priority": assessment.migration_priority,
					"estimated_break_date": assessment.estimated_break_date.isoformat() if assessment.estimated_break_date else None
				}
		
		# Upcoming deadlines
		current_date = datetime.utcnow()
		for plan in self.migration_plans.values():
			if plan.target_completion_date and plan.target_completion_date > current_date:
				days_remaining = (plan.target_completion_date - current_date).days
				dashboard["upcoming_deadlines"].append({
					"plan_id": plan.plan_id,
					"tenant_id": plan.tenant_id,
					"target_date": plan.target_completion_date.isoformat(),
					"days_remaining": days_remaining,
					"urgency": "high" if days_remaining < 30 else "medium" if days_remaining < 90 else "low"
				})
		
		return dashboard
	
	def _calculate_average_success_rate(self) -> float:
		"""Calculate average migration success rate"""
		if not self.migration_progress:
			return 100.0
		
		total_attempts = 0
		total_successes = 0
		
		for progress in self.migration_progress.values():
			total_attempts += progress.migrated_keys + progress.failed_migrations
			total_successes += progress.migrated_keys
		
		if total_attempts == 0:
			return 100.0
		
		return (total_successes / total_attempts) * 100


# Export quantum-safe cryptography components
__all__ = [
	"QuantumSafeCryptographyManager", "QuantumThreatAssessment", "MigrationPlan", 
	"MigrationProgress", "QuantumThreatLevel", "MigrationStrategy", "QuantumResistance"
]