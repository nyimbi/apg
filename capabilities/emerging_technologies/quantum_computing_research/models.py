"""
Quantum Computing Models

Database models for quantum computing integration with digital twins,
quantum simulations, optimization algorithms, and hybrid quantum-classical computing.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class QCQuantumCircuit(Model, AuditMixin, BaseMixin):
	"""
	Quantum circuit definition and configuration.
	
	Represents quantum circuits used for digital twin simulations,
	optimization problems, and quantum machine learning tasks.
	"""
	__tablename__ = 'qc_quantum_circuit'
	
	# Identity
	circuit_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Circuit Information
	circuit_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	circuit_type = Column(String(50), nullable=False, index=True)  # optimization, simulation, ml, cryptography
	algorithm_type = Column(String(50), nullable=False, index=True)  # qaoa, vqe, qsvm, qnn, grover, shor
	
	# Circuit Configuration
	num_qubits = Column(Integer, nullable=False)
	circuit_depth = Column(Integer, default=0)
	gate_count = Column(Integer, default=0)
	circuit_definition = Column(JSON, default=dict)  # Complete circuit description
	gate_sequence = Column(JSON, default=list)  # Ordered list of gates
	
	# Parameters
	parameters = Column(JSON, default=dict)  # Circuit parameters (angles, etc.)
	parameter_bounds = Column(JSON, default=dict)  # Parameter optimization bounds
	classical_bits = Column(Integer, default=0)  # Number of classical measurement bits
	
	# Quantum Properties
	entanglement_measure = Column(Float, nullable=True)  # Entanglement quantification
	coherence_requirements = Column(JSON, default=dict)  # Coherence time requirements
	noise_tolerance = Column(Float, default=0.01)  # Noise tolerance level
	
	# Hardware Requirements
	target_hardware = Column(String(50), default='simulator')  # ibm_quantum, google_quantum_ai, etc.
	qubit_connectivity = Column(JSON, default=list)  # Required qubit connections
	gate_set_requirements = Column(JSON, default=list)  # Required quantum gates
	
	# Performance Metrics
	theoretical_complexity = Column(String(100), nullable=True)  # Big-O complexity
	estimated_runtime_ms = Column(Float, nullable=True)
	quantum_advantage_potential = Column(Float, nullable=True)  # 0-1 score
	success_probability = Column(Float, nullable=True)
	
	# Validation and Testing
	is_validated = Column(Boolean, default=False)
	validation_results = Column(JSON, default=dict)
	test_results = Column(JSON, default=list)
	benchmark_scores = Column(JSON, default=dict)
	
	# Usage Tracking
	execution_count = Column(Integer, default=0)
	total_runtime_ms = Column(Float, default=0.0)
	success_rate = Column(Float, nullable=True)
	last_executed = Column(DateTime, nullable=True)
	
	# Version Control
	version = Column(String(20), default='1.0.0')
	parent_circuit_id = Column(String(36), nullable=True)  # For circuit evolution
	optimization_generation = Column(Integer, default=0)
	
	# Access Control
	created_by = Column(String(36), nullable=False, index=True)
	visibility = Column(String(20), default='private')  # private, shared, public
	shared_with = Column(JSON, default=list)
	
	# Relationships
	executions = relationship("QCQuantumExecution", back_populates="circuit")
	optimizations = relationship("QCQuantumOptimization", back_populates="circuit")
	
	def __repr__(self):
		return f"<QCQuantumCircuit {self.circuit_name}>"
	
	def calculate_circuit_metrics(self) -> Dict[str, Any]:
		"""Calculate circuit complexity and performance metrics"""
		if not self.gate_sequence:
			return {}
		
		# Count gate types
		gate_counts = {}
		for gate in self.gate_sequence:
			gate_type = gate.get('type', 'unknown')
			gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
		
		# Calculate depth (simplified)
		self.circuit_depth = len(self.gate_sequence)  # Simplified depth calculation
		self.gate_count = sum(gate_counts.values())
		
		return {
			'depth': self.circuit_depth,
			'gate_count': self.gate_count,
			'gate_distribution': gate_counts,
			'two_qubit_gates': gate_counts.get('CNOT', 0) + gate_counts.get('TOFFOLI', 0),
			'single_qubit_gates': self.gate_count - (gate_counts.get('CNOT', 0) + gate_counts.get('TOFFOLI', 0))
		}
	
	def estimate_quantum_advantage(self) -> float:
		"""Estimate potential quantum advantage"""
		# Simplified quantum advantage estimation
		if self.algorithm_type in ['shor', 'grover']:
			return 0.9  # High advantage for known quantum algorithms
		elif self.algorithm_type in ['qaoa', 'vqe']:
			return 0.7  # Moderate advantage for NISQ algorithms
		elif self.num_qubits > 10:
			return 0.6  # Higher qubit count suggests potential advantage
		else:
			return 0.3  # Lower advantage for small circuits
	
	def is_executable_on_hardware(self, hardware_name: str) -> bool:
		"""Check if circuit can be executed on specific hardware"""
		if hardware_name == 'simulator':
			return True
		
		# Check hardware-specific constraints
		hardware_limits = {
			'ibm_quantum': {'max_qubits': 127, 'max_depth': 1000},
			'google_quantum_ai': {'max_qubits': 70, 'max_depth': 500},
			'rigetti': {'max_qubits': 80, 'max_depth': 800}
		}
		
		limits = hardware_limits.get(hardware_name, {})
		if not limits:
			return False
		
		return (
			self.num_qubits <= limits.get('max_qubits', 0) and
			self.circuit_depth <= limits.get('max_depth', 0)
		)


class QCQuantumExecution(Model, AuditMixin, BaseMixin):
	"""
	Quantum circuit execution tracking and results.
	
	Tracks individual executions of quantum circuits with detailed
	performance metrics, results, and hardware utilization.
	"""
	__tablename__ = 'qc_quantum_execution'
	
	# Identity
	execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	circuit_id = Column(String(36), ForeignKey('qc_quantum_circuit.circuit_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Execution Context
	execution_name = Column(String(200), nullable=True)
	purpose = Column(String(100), nullable=True)  # optimization, simulation, analysis
	triggered_by = Column(String(36), nullable=True, index=True)
	execution_environment = Column(String(50), default='simulator')
	
	# Hardware Configuration
	quantum_hardware = Column(String(50), nullable=False)  # Target hardware platform
	hardware_topology = Column(JSON, default=dict)  # Hardware connectivity graph
	qubit_mapping = Column(JSON, default=dict)  # Logical to physical qubit mapping
	calibration_data = Column(JSON, default=dict)  # Hardware calibration info
	
	# Execution Parameters
	num_shots = Column(Integer, default=1024)  # Number of circuit executions
	optimization_level = Column(Integer, default=1)  # Circuit optimization level
	noise_model = Column(JSON, default=dict)  # Noise model configuration
	error_mitigation = Column(JSON, default=dict)  # Error mitigation techniques
	
	# Timing and Performance
	queued_at = Column(DateTime, nullable=True)
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True)
	queue_time_ms = Column(Float, nullable=True)
	execution_time_ms = Column(Float, nullable=True)
	total_time_ms = Column(Float, nullable=True)
	
	# Results
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
	measurement_results = Column(JSON, default=dict)  # Raw measurement outcomes
	state_probabilities = Column(JSON, default=dict)  # Quantum state probabilities
	expectation_values = Column(JSON, default=dict)  # Observable expectation values
	
	# Quality Metrics
	fidelity_estimate = Column(Float, nullable=True)  # Execution fidelity
	error_rate = Column(Float, nullable=True)  # Estimated error rate
	noise_impact_score = Column(Float, nullable=True)  # Noise impact assessment
	
	# Resource Usage
	quantum_volume = Column(Integer, nullable=True)  # Quantum volume achieved
	gate_errors = Column(JSON, default=dict)  # Per-gate error rates
	readout_errors = Column(JSON, default=dict)  # Measurement readout errors
	coherence_utilization = Column(Float, nullable=True)  # Coherence time utilization
	
	# Cost and Billing
	execution_cost = Column(Float, nullable=True)  # Execution cost
	hardware_time_seconds = Column(Float, nullable=True)  # Actual hardware time used
	priority_level = Column(Integer, default=1)  # Execution priority
	
	# Error Handling
	error_occurred = Column(Boolean, default=False)
	error_type = Column(String(100), nullable=True)
	error_message = Column(Text, nullable=True)
	error_details = Column(JSON, default=dict)
	retry_count = Column(Integer, default=0)
	
	# Post-processing
	post_processing_applied = Column(JSON, default=list)  # Applied post-processing
	classical_processing_time_ms = Column(Float, nullable=True)
	final_results = Column(JSON, default=dict)  # Final processed results
	
	# Relationships
	circuit = relationship("QCQuantumCircuit", back_populates="executions")
	
	def __repr__(self):
		return f"<QCQuantumExecution {self.circuit.circuit_name} - {self.status}>"
	
	def calculate_duration(self) -> Optional[float]:
		"""Calculate total execution duration"""
		if self.started_at and self.completed_at:
			duration = self.completed_at - self.started_at
			self.total_time_ms = duration.total_seconds() * 1000
			return self.total_time_ms
		return None
	
	def calculate_success_rate(self) -> float:
		"""Calculate execution success rate based on fidelity"""
		if self.fidelity_estimate is not None:
			return self.fidelity_estimate * 100
		return 0.0
	
	def estimate_quantum_advantage(self) -> float:
		"""Estimate quantum advantage achieved in this execution"""
		if not self.measurement_results:
			return 0.0
		
		# Simple heuristic based on result quality and execution time
		base_advantage = 0.5
		
		if self.fidelity_estimate and self.fidelity_estimate > 0.9:
			base_advantage += 0.3
		
		if self.execution_time_ms and self.execution_time_ms < 1000:  # Fast execution
			base_advantage += 0.2
		
		return min(1.0, base_advantage)
	
	def get_performance_summary(self) -> Dict[str, Any]:
		"""Get execution performance summary"""
		return {
			'execution_time_ms': self.execution_time_ms,
			'queue_time_ms': self.queue_time_ms,
			'success_rate': self.calculate_success_rate(),
			'fidelity': self.fidelity_estimate,
			'error_rate': self.error_rate,
			'quantum_advantage': self.estimate_quantum_advantage(),
			'hardware_utilization': {
				'quantum_volume': self.quantum_volume,
				'coherence_utilization': self.coherence_utilization,
				'hardware_time_seconds': self.hardware_time_seconds
			},
			'cost_metrics': {
				'execution_cost': self.execution_cost,
				'cost_per_shot': self.execution_cost / self.num_shots if self.execution_cost and self.num_shots else 0
			}
		}


class QCQuantumOptimization(Model, AuditMixin, BaseMixin):
	"""
	Quantum optimization problem and solution tracking.
	
	Manages quantum optimization problems for digital twins including
	resource allocation, scheduling, and energy optimization.
	"""
	__tablename__ = 'qc_quantum_optimization'
	
	# Identity
	optimization_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	circuit_id = Column(String(36), ForeignKey('qc_quantum_circuit.circuit_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Problem Definition
	problem_name = Column(String(200), nullable=False, index=True)
	problem_type = Column(String(50), nullable=False, index=True)  # resource_allocation, scheduling, energy, routing
	problem_description = Column(Text, nullable=True)
	problem_size = Column(Integer, nullable=True)  # Problem complexity measure
	
	# Problem Configuration
	objective_function = Column(JSON, default=dict)  # Objective function definition
	constraints = Column(JSON, default=list)  # Problem constraints
	variables = Column(JSON, default=dict)  # Optimization variables
	parameters = Column(JSON, default=dict)  # Problem-specific parameters
	
	# Algorithm Configuration
	algorithm_type = Column(String(50), nullable=False)  # qaoa, vqe, quantum_annealing
	num_layers = Column(Integer, default=2)  # For variational algorithms
	optimizer_type = Column(String(50), default='cobyla')  # Classical optimizer
	max_iterations = Column(Integer, default=100)
	convergence_threshold = Column(Float, default=1e-6)
	
	# Execution Status
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed
	progress_percentage = Column(Float, default=0.0)
	current_iteration = Column(Integer, default=0)
	
	# Results
	optimal_solution = Column(JSON, default=dict)  # Best solution found
	optimal_value = Column(Float, nullable=True)  # Best objective value
	solution_quality = Column(Float, nullable=True)  # Solution quality score (0-1)
	convergence_achieved = Column(Boolean, default=False)
	
	# Performance Metrics
	total_iterations = Column(Integer, default=0)
	total_function_evaluations = Column(Integer, default=0)
	classical_runtime_ms = Column(Float, nullable=True)
	quantum_runtime_ms = Column(Float, nullable=True)
	total_runtime_ms = Column(Float, nullable=True)
	
	# Quantum Advantage Analysis
	quantum_speedup_factor = Column(Float, nullable=True)  # Compared to classical
	quantum_advantage_score = Column(Float, nullable=True)  # Overall advantage metric
	classical_comparison_time = Column(Float, nullable=True)  # Classical solution time
	
	# Optimization History
	parameter_evolution = Column(JSON, default=list)  # Parameter values over iterations
	objective_evolution = Column(JSON, default=list)  # Objective values over iterations
	convergence_data = Column(JSON, default=dict)  # Convergence analysis
	
	# Solution Validation
	solution_verified = Column(Boolean, default=False)
	verification_results = Column(JSON, default=dict)
	constraint_satisfaction = Column(JSON, default=dict)  # Constraint satisfaction status
	
	# Error Analysis
	optimization_errors = Column(JSON, default=list)  # Errors during optimization
	parameter_sensitivity = Column(JSON, default=dict)  # Parameter sensitivity analysis
	noise_impact_analysis = Column(JSON, default=dict)  # Impact of quantum noise
	
	# Business Impact
	cost_savings_estimate = Column(Float, nullable=True)  # Estimated cost savings
	efficiency_improvement = Column(Float, nullable=True)  # Efficiency improvement %
	roi_estimate = Column(Float, nullable=True)  # Return on investment estimate
	
	# Relationships
	circuit = relationship("QCQuantumCircuit", back_populates="optimizations")
	
	def __repr__(self):
		return f"<QCQuantumOptimization {self.problem_name}>"
	
	def calculate_optimization_metrics(self) -> Dict[str, Any]:
		"""Calculate optimization performance metrics"""
		success_rate = 1.0 if self.convergence_achieved else 0.0
		
		# Calculate improvement rate
		if self.objective_evolution and len(self.objective_evolution) > 1:
			initial_value = self.objective_evolution[0]
			final_value = self.objective_evolution[-1]
			improvement_rate = abs((final_value - initial_value) / initial_value * 100) if initial_value != 0 else 0
		else:
			improvement_rate = 0.0
		
		return {
			'success_rate': success_rate,
			'improvement_rate': improvement_rate,
			'convergence_speed': self.total_iterations / max(1, self.max_iterations),
			'solution_quality': self.solution_quality or 0.0,
			'quantum_advantage': self.quantum_advantage_score or 0.0
		}
	
	def estimate_business_impact(self) -> Dict[str, Any]:
		"""Estimate business impact of optimization"""
		return {
			'cost_savings': self.cost_savings_estimate or 0.0,
			'efficiency_gain': self.efficiency_improvement or 0.0,
			'roi_estimate': self.roi_estimate or 0.0,
			'payback_period_months': 12 / max(0.1, self.roi_estimate or 0.1) if self.roi_estimate else None
		}
	
	def is_solution_valid(self) -> bool:
		"""Check if solution satisfies all constraints"""
		if not self.constraint_satisfaction:
			return False
		
		return all(self.constraint_satisfaction.values())


class QCQuantumTwin(Model, AuditMixin, BaseMixin):
	"""
	Quantum-enhanced digital twin configuration.
	
	Represents digital twins with quantum computing capabilities
	for enhanced simulation and optimization.
	"""
	__tablename__ = 'qc_quantum_twin'
	
	# Identity
	twin_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Twin Information
	twin_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	twin_type = Column(String(50), nullable=False, index=True)  # industrial, molecular, financial, logistics
	industry_sector = Column(String(100), nullable=True)
	
	# Quantum Configuration
	quantum_enabled = Column(Boolean, default=True)
	quantum_hardware_preference = Column(String(50), default='simulator')
	num_qubits_required = Column(Integer, default=4)
	quantum_algorithms_enabled = Column(JSON, default=list)  # Enabled quantum algorithms
	
	# Twin Properties
	classical_model = Column(JSON, default=dict)  # Classical twin model
	quantum_model = Column(JSON, default=dict)  # Quantum enhancement model
	hybrid_configuration = Column(JSON, default=dict)  # Hybrid quantum-classical config
	
	# Performance Requirements
	latency_requirements_ms = Column(Float, nullable=True)
	accuracy_requirements = Column(Float, default=0.95)
	reliability_requirements = Column(Float, default=0.99)
	scalability_requirements = Column(JSON, default=dict)
	
	# Quantum Features
	quantum_simulation_enabled = Column(Boolean, default=True)
	quantum_optimization_enabled = Column(Boolean, default=True)
	quantum_ml_enabled = Column(Boolean, default=False)
	quantum_sensing_enabled = Column(Boolean, default=False)
	
	# Resource Allocation
	quantum_resource_quota = Column(JSON, default=dict)  # Resource limits
	priority_level = Column(Integer, default=1)  # Resource priority
	cost_budget = Column(Float, nullable=True)  # Budget for quantum resources
	
	# Performance Metrics
	quantum_advantage_achieved = Column(Float, default=0.0)  # Measured quantum advantage
	simulation_accuracy = Column(Float, nullable=True)
	optimization_performance = Column(Float, nullable=True)
	total_quantum_time_used_ms = Column(Float, default=0.0)
	
	# Status and Health
	status = Column(String(20), default='active', index=True)  # active, inactive, maintenance, error
	health_score = Column(Float, default=1.0)  # 0-1 health score
	last_quantum_execution = Column(DateTime, nullable=True)
	
	# Monitoring
	monitoring_enabled = Column(Boolean, default=True)
	alert_thresholds = Column(JSON, default=dict)
	performance_baselines = Column(JSON, default=dict)
	
	# Integration
	connected_systems = Column(JSON, default=list)  # Connected external systems
	data_sources = Column(JSON, default=list)  # Data source configurations
	api_endpoints = Column(JSON, default=dict)  # API endpoint configurations
	
	def __repr__(self):
		return f"<QCQuantumTwin {self.twin_name}>"
	
	def calculate_quantum_readiness(self) -> float:
		"""Calculate quantum readiness score"""
		readiness_factors = {
			'quantum_enabled': 0.3 if self.quantum_enabled else 0.0,
			'algorithm_coverage': len(self.quantum_algorithms_enabled) / 8 * 0.2,  # Assume 8 max algorithms
			'resource_availability': 0.2 if self.quantum_resource_quota else 0.0,
			'performance_track_record': min(1.0, self.quantum_advantage_achieved) * 0.3
		}
		
		return sum(readiness_factors.values())
	
	def get_quantum_capabilities(self) -> Dict[str, Any]:
		"""Get quantum capabilities summary"""
		return {
			'quantum_enabled': self.quantum_enabled,
			'supported_algorithms': self.quantum_algorithms_enabled,
			'hardware_platforms': [self.quantum_hardware_preference],
			'qubit_requirements': self.num_qubits_required,
			'feature_matrix': {
				'simulation': self.quantum_simulation_enabled,
				'optimization': self.quantum_optimization_enabled,
				'machine_learning': self.quantum_ml_enabled,
				'sensing': self.quantum_sensing_enabled
			}
		}
	
	def estimate_quantum_benefit(self) -> Dict[str, Any]:
		"""Estimate potential quantum computing benefits"""
		return {
			'speedup_potential': self.quantum_advantage_achieved or 1.0,
			'accuracy_improvement': (self.simulation_accuracy or 0.95) - 0.85,  # Compare to classical baseline
			'cost_efficiency': self.cost_budget / max(1, self.total_quantum_time_used_ms) if self.total_quantum_time_used_ms else 0,
			'scalability_factor': len(self.scalability_requirements) * 0.2
		}


class QCHardwareStatus(Model, AuditMixin, BaseMixin):
	"""
	Quantum hardware status and availability tracking.
	
	Monitors quantum hardware platforms and their availability
	for digital twin computations.
	"""
	__tablename__ = 'qc_hardware_status'
	
	# Identity
	hardware_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	
	# Hardware Information
	hardware_name = Column(String(200), nullable=False, index=True)
	provider = Column(String(100), nullable=False, index=True)  # IBM, Google, Rigetti, etc.
	hardware_type = Column(String(50), nullable=False)  # superconducting, trapped_ion, photonic
	location = Column(String(100), nullable=True)
	
	# Specifications
	num_qubits = Column(Integer, nullable=False)
	quantum_volume = Column(Integer, nullable=True)
	gate_set = Column(JSON, default=list)  # Available quantum gates
	connectivity_graph = Column(JSON, default=dict)  # Qubit connectivity
	
	# Performance Characteristics
	coherence_time_us = Column(Float, nullable=True)  # T2 coherence time
	gate_fidelity = Column(Float, nullable=True)  # Average gate fidelity
	readout_fidelity = Column(Float, nullable=True)  # Measurement fidelity
	error_rates = Column(JSON, default=dict)  # Various error rates
	
	# Availability
	status = Column(String(20), default='available', index=True)  # available, busy, maintenance, offline
	availability_percentage = Column(Float, default=100.0)
	queue_length = Column(Integer, default=0)
	estimated_wait_time_minutes = Column(Float, nullable=True)
	
	# Usage Statistics
	total_executions = Column(Integer, default=0)
	successful_executions = Column(Integer, default=0)
	average_execution_time_ms = Column(Float, nullable=True)
	utilization_percentage = Column(Float, default=0.0)
	
	# Cost Information
	cost_per_shot = Column(Float, nullable=True)
	cost_per_second = Column(Float, nullable=True)
	cost_model = Column(JSON, default=dict)  # Detailed cost structure
	
	# Monitoring
	last_calibration = Column(DateTime, nullable=True)
	next_maintenance = Column(DateTime, nullable=True)
	health_metrics = Column(JSON, default=dict)
	alert_status = Column(String(20), default='normal')  # normal, warning, critical
	
	# API Information
	api_endpoint = Column(String(500), nullable=True)
	authentication_required = Column(Boolean, default=True)
	rate_limits = Column(JSON, default=dict)
	supported_features = Column(JSON, default=list)
	
	def __repr__(self):
		return f"<QCHardwareStatus {self.hardware_name}>"
	
	def calculate_performance_score(self) -> float:
		"""Calculate overall hardware performance score"""
		scores = []
		
		# Gate fidelity score
		if self.gate_fidelity:
			scores.append(self.gate_fidelity)
		
		# Availability score
		scores.append(self.availability_percentage / 100)
		
		# Success rate score
		if self.total_executions > 0:
			success_rate = self.successful_executions / self.total_executions
			scores.append(success_rate)
		
		# Coherence score (normalized)
		if self.coherence_time_us:
			coherence_score = min(1.0, self.coherence_time_us / 100)  # Normalize to 100us max
			scores.append(coherence_score)
		
		return sum(scores) / len(scores) if scores else 0.0
	
	def is_suitable_for_circuit(self, circuit: QCQuantumCircuit) -> bool:
		"""Check if hardware is suitable for given circuit"""
		# Check qubit count
		if circuit.num_qubits > self.num_qubits:
			return False
		
		# Check gate set compatibility
		required_gates = set(gate.get('type', '') for gate in circuit.gate_sequence)
		available_gates = set(self.gate_set)
		if not required_gates.issubset(available_gates):
			return False
		
		# Check availability
		if self.status != 'available':
			return False
		
		return True
	
	def estimate_execution_cost(self, circuit: QCQuantumCircuit, num_shots: int = 1024) -> float:
		"""Estimate cost for executing circuit"""
		if not self.cost_per_shot and not self.cost_per_second:
			return 0.0
		
		cost = 0.0
		
		if self.cost_per_shot:
			cost += self.cost_per_shot * num_shots
		
		if self.cost_per_second and circuit.estimated_runtime_ms:
			runtime_seconds = circuit.estimated_runtime_ms / 1000
			cost += self.cost_per_second * runtime_seconds
		
		return cost