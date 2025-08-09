"""
Neuromorphic Authentication Processor - Brain-Inspired Authentication Computing

Revolutionary neuromorphic computing system that mimics brain neural networks
for ultra-fast (sub-millisecond) authentication decisions with adaptive learning
and real-time pattern recognition capabilities.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import random
from collections import defaultdict, deque
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuronType(Enum):
	"""Types of neurons in the neuromorphic processor"""
	INPUT = "input"
	HIDDEN = "hidden"
	OUTPUT = "output"
	MEMORY = "memory"
	INHIBITORY = "inhibitory"


class SpikeType(Enum):
	"""Types of neural spikes"""
	EXCITATORY = "excitatory"
	INHIBITORY = "inhibitory"
	MODULATORY = "modulatory"


class LearningRule(Enum):
	"""Neuromorphic learning rules"""
	STDP = "spike_time_dependent_plasticity"
	HEBBIAN = "hebbian"
	HOMEOSTATIC = "homeostatic"
	REINFORCEMENT = "reinforcement"


class AuthenticationDecision(Enum):
	"""Authentication decisions from neuromorphic processor"""
	ALLOW = "allow"
	DENY = "deny"
	CHALLENGE = "challenge"
	MONITOR = "monitor"


@dataclass
class Spike:
	"""Neural spike event"""
	neuron_id: str
	timestamp: float  # in milliseconds
	amplitude: float
	spike_type: SpikeType
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Synapse:
	"""Synaptic connection between neurons"""
	pre_neuron_id: str
	post_neuron_id: str
	weight: float
	delay: float  # in milliseconds
	plasticity: float = 1.0
	last_update: float = 0.0


class SpikingNeuron(BaseModel):
	"""Individual spiking neuron in the neuromorphic network"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	neuron_type: NeuronType
	threshold: float = 1.0
	resting_potential: float = 0.0
	membrane_potential: float = 0.0
	refractory_period: float = 2.0  # milliseconds
	last_spike_time: float = 0.0
	leak_rate: float = 0.1
	adaptation_rate: float = 0.01
	inhibition_strength: float = 0.5
	spike_count: int = 0
	input_history: List[float] = Field(default_factory=list)
	output_spikes: List[Spike] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class NeuromorphicLayer(BaseModel):
	"""Layer in the neuromorphic network"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	neurons: List[SpikingNeuron] = Field(default_factory=list)
	synapses: List[Synapse] = Field(default_factory=list)
	layer_type: str
	activation_pattern: List[float] = Field(default_factory=list)
	learning_rate: float = 0.001
	connectivity_density: float = 0.1


class AuthenticationContext(BaseModel):
	"""Context for neuromorphic authentication decision"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str
	session_id: str
	behavioral_features: List[float]
	biometric_features: List[float]
	contextual_features: List[float]
	risk_indicators: List[float]
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	processing_time: float = 0.0
	confidence_score: float = 0.0
	decision_path: List[str] = Field(default_factory=list)


class NeuromorphicProcessor:
	"""
	Neuromorphic authentication processor that uses spiking neural networks
	for ultra-fast authentication decisions with adaptive learning
	"""
	
	def __init__(self, config: Optional[Dict[str, Any]] = None):
		self.config = config or {}
		self.layers: List[NeuromorphicLayer] = []
		self.global_synapses: Dict[str, Synapse] = {}
		self.spike_queue: deque = deque()
		self.current_time: float = 0.0
		self.decision_history: List[Dict[str, Any]] = []
		self.learning_enabled = True
		self.adaptation_history: List[Dict[str, Any]] = []
		
		# Performance metrics
		self.metrics = {
			"total_decisions": 0,
			"correct_decisions": 0,
			"average_processing_time": 0.0,
			"spike_frequency": 0.0,
			"learning_iterations": 0,
			"adaptation_events": 0,
			"false_positives": 0,
			"false_negatives": 0
		}
		
		# Initialize default architecture
		self._initialize_default_architecture()
	
	def _log_neuro_operation(self, operation: str, details: Dict[str, Any]) -> None:
		"""Log neuromorphic operations"""
		logger.info(f"Neuro Operation: {operation}")
		for key, value in details.items():
			logger.info(f"  {key}: {value}")
	
	def _initialize_default_architecture(self) -> None:
		"""Initialize default neuromorphic architecture for authentication"""
		try:
			# Input layer for authentication features
			input_layer = NeuromorphicLayer(
				name="input_layer",
				layer_type="input"
			)
			
			# Create input neurons for different feature types
			feature_groups = [
				("behavioral", 20),  # Behavioral biometrics
				("biometric", 15),   # Traditional biometrics
				("contextual", 10),  # Context features
				("risk", 8)         # Risk indicators
			]
			
			input_neurons = []
			for group_name, count in feature_groups:
				for i in range(count):
					neuron = SpikingNeuron(
						neuron_type=NeuronType.INPUT,
						threshold=0.5,
						metadata={"feature_group": group_name, "feature_index": i}
					)
					input_neurons.append(neuron)
			
			input_layer.neurons = input_neurons
			
			# Hidden processing layers
			hidden_layer_1 = NeuromorphicLayer(
				name="pattern_detection",
				layer_type="hidden"
			)
			
			# Pattern detection neurons
			pattern_neurons = []
			for i in range(64):  # 64 pattern detection neurons
				neuron = SpikingNeuron(
					neuron_type=NeuronType.HIDDEN,
					threshold=1.2,
					adaptation_rate=0.02,
					metadata={"layer": "pattern_detection", "neuron_index": i}
				)
				pattern_neurons.append(neuron)
			
			hidden_layer_1.neurons = pattern_neurons
			
			# Memory layer for temporal patterns
			memory_layer = NeuromorphicLayer(
				name="temporal_memory",
				layer_type="memory"
			)
			
			memory_neurons = []
			for i in range(32):  # 32 memory neurons
				neuron = SpikingNeuron(
					neuron_type=NeuronType.MEMORY,
					threshold=0.8,
					leak_rate=0.05,  # Slower leak for memory
					metadata={"layer": "memory", "neuron_index": i}
				)
				memory_neurons.append(neuron)
			
			memory_layer.neurons = memory_neurons
			
			# Decision layer
			decision_layer = NeuromorphicLayer(
				name="decision_output",
				layer_type="output"
			)
			
			# Decision neurons (one for each decision type)
			decision_neurons = []
			for decision_type in AuthenticationDecision:
				neuron = SpikingNeuron(
					neuron_type=NeuronType.OUTPUT,
					threshold=1.5,
					metadata={"decision_type": decision_type.value}
				)
				decision_neurons.append(neuron)
			
			decision_layer.neurons = decision_neurons
			
			# Inhibitory control layer
			inhibitory_layer = NeuromorphicLayer(
				name="inhibitory_control",
				layer_type="inhibitory"
			)
			
			inhibitory_neurons = []
			for i in range(16):  # 16 inhibitory neurons
				neuron = SpikingNeuron(
					neuron_type=NeuronType.INHIBITORY,
					threshold=0.9,
					inhibition_strength=0.8,
					metadata={"layer": "inhibitory", "neuron_index": i}
				)
				inhibitory_neurons.append(neuron)
			
			inhibitory_layer.neurons = inhibitory_neurons
			
			# Add layers to processor
			self.layers = [
				input_layer,
				hidden_layer_1,
				memory_layer,
				decision_layer,
				inhibitory_layer
			]
			
			# Create synaptic connections
			self._create_synaptic_connections()
			
			self._log_neuro_operation("architecture_initialized", {
				"layers": len(self.layers),
				"total_neurons": sum(len(layer.neurons) for layer in self.layers),
				"total_synapses": len(self.global_synapses)
			})
			
		except Exception as e:
			self._log_neuro_operation("architecture_initialization_error", {"error": str(e)})
	
	def _create_synaptic_connections(self) -> None:
		"""Create synaptic connections between layers"""
		try:
			synapse_count = 0
			
			# Connect input to hidden layer
			input_neurons = self.layers[0].neurons  # input_layer
			hidden_neurons = self.layers[1].neurons  # pattern_detection
			
			for input_neuron in input_neurons:
				for hidden_neuron in hidden_neurons:
					if random.random() < 0.3:  # 30% connectivity
						weight = random.gauss(0.5, 0.2)  # Random weight with Gaussian distribution
						synapse = Synapse(
							pre_neuron_id=input_neuron.id,
							post_neuron_id=hidden_neuron.id,
							weight=weight,
							delay=random.uniform(0.5, 2.0),  # 0.5-2ms delay
							plasticity=1.0
						)
						self.global_synapses[f"{input_neuron.id}->{hidden_neuron.id}"] = synapse
						synapse_count += 1
			
			# Connect hidden to memory layer
			memory_neurons = self.layers[2].neurons  # temporal_memory
			
			for hidden_neuron in hidden_neurons:
				for memory_neuron in memory_neurons:
					if random.random() < 0.4:  # 40% connectivity to memory
						weight = random.gauss(0.6, 0.15)
						synapse = Synapse(
							pre_neuron_id=hidden_neuron.id,
							post_neuron_id=memory_neuron.id,
							weight=weight,
							delay=random.uniform(0.3, 1.5),
							plasticity=1.2  # Higher plasticity for memory connections
						)
						self.global_synapses[f"{hidden_neuron.id}->{memory_neuron.id}"] = synapse
						synapse_count += 1
			
			# Connect hidden and memory to decision layer
			decision_neurons = self.layers[3].neurons  # decision_output
			
			for source_neuron in hidden_neurons + memory_neurons:
				for decision_neuron in decision_neurons:
					if random.random() < 0.6:  # 60% connectivity to decision
						weight = random.gauss(0.7, 0.1)
						synapse = Synapse(
							pre_neuron_id=source_neuron.id,
							post_neuron_id=decision_neuron.id,
							weight=weight,
							delay=random.uniform(0.2, 1.0),
							plasticity=0.8
						)
						self.global_synapses[f"{source_neuron.id}->{decision_neuron.id}"] = synapse
						synapse_count += 1
			
			# Connect inhibitory neurons to all layers
			inhibitory_neurons = self.layers[4].neurons  # inhibitory_control
			
			for inhibitory_neuron in inhibitory_neurons:
				for layer in self.layers[:-1]:  # All except inhibitory layer itself
					for target_neuron in layer.neurons:
						if random.random() < 0.1:  # 10% inhibitory connectivity
							weight = -random.uniform(0.3, 0.8)  # Negative weights for inhibition
							synapse = Synapse(
								pre_neuron_id=inhibitory_neuron.id,
								post_neuron_id=target_neuron.id,
								weight=weight,
								delay=random.uniform(0.1, 0.5),
								plasticity=0.5
							)
							self.global_synapses[f"{inhibitory_neuron.id}->{target_neuron.id}"] = synapse
							synapse_count += 1
			
			self._log_neuro_operation("synapses_created", {"count": synapse_count})
			
		except Exception as e:
			self._log_neuro_operation("synapse_creation_error", {"error": str(e)})
	
	async def process_authentication(
		self, 
		context: AuthenticationContext
	) -> Tuple[AuthenticationDecision, float, Dict[str, Any]]:
		"""Process authentication request through neuromorphic network"""
		start_time = asyncio.get_event_loop().time() * 1000  # Convert to milliseconds
		self.current_time = start_time
		
		try:
			# Convert features to spikes
			await self._encode_features_to_spikes(context)
			
			# Run neural simulation
			decision_spikes = await self._simulate_network(duration_ms=10.0)
			
			# Decode decision from output spikes
			decision, confidence = await self._decode_decision(decision_spikes)
			
			# Calculate processing time
			end_time = asyncio.get_event_loop().time() * 1000
			processing_time = end_time - start_time
			
			# Create decision metadata
			decision_metadata = {
				"processing_time_ms": processing_time,
				"spike_count": len(decision_spikes),
				"confidence_score": confidence,
				"network_state": await self._get_network_state(),
				"feature_importance": await self._analyze_feature_importance(context)
			}
			
			# Update metrics
			self.metrics["total_decisions"] += 1
			self.metrics["average_processing_time"] = (
				(self.metrics["average_processing_time"] * (self.metrics["total_decisions"] - 1) + 
				 processing_time) / self.metrics["total_decisions"]
			)
			
			# Record decision
			self.decision_history.append({
				"timestamp": datetime.utcnow(),
				"user_id": context.user_id,
				"decision": decision.value,
				"confidence": confidence,
				"processing_time": processing_time,
				"context_hash": hash(str(context.behavioral_features + context.biometric_features))
			})
			
			# Trigger learning if enabled
			if self.learning_enabled:
				await self._trigger_learning(context, decision, confidence)
			
			self._log_neuro_operation("authentication_processed", {
				"user_id": context.user_id,
				"decision": decision.value,
				"confidence": confidence,
				"processing_time_ms": processing_time,
				"spike_count": len(decision_spikes)
			})
			
			return decision, confidence, decision_metadata
			
		except Exception as e:
			self._log_neuro_operation("authentication_processing_error", {
				"user_id": context.user_id,
				"error": str(e)
			})
			return AuthenticationDecision.DENY, 0.0, {"error": str(e)}
	
	async def _encode_features_to_spikes(self, context: AuthenticationContext) -> None:
		"""Encode authentication features into neural spikes"""
		try:
			input_layer = self.layers[0]  # input_layer
			feature_groups = [
				("behavioral", context.behavioral_features),
				("biometric", context.biometric_features),
				("contextual", context.contextual_features),
				("risk", context.risk_indicators)
			]
			
			neuron_index = 0
			for group_name, features in feature_groups:
				for i, feature_value in enumerate(features):
					if neuron_index < len(input_layer.neurons):
						neuron = input_layer.neurons[neuron_index]
						
						# Convert feature value to spike rate (Poisson encoding)
						spike_rate = max(0, min(100, feature_value * 100))  # 0-100 Hz
						spike_probability = spike_rate / 1000.0  # Per millisecond
						
						# Generate spikes for this feature
						for t in range(int(self.current_time), int(self.current_time + 10)):
							if random.random() < spike_probability:
								spike = Spike(
									neuron_id=neuron.id,
									timestamp=t,
									amplitude=feature_value,
									spike_type=SpikeType.EXCITATORY,
									metadata={"feature_group": group_name, "feature_value": feature_value}
								)
								self.spike_queue.append(spike)
								neuron.output_spikes.append(spike)
						
						neuron_index += 1
			
			self._log_neuro_operation("features_encoded", {
				"total_features": sum(len(features) for _, features in feature_groups),
				"spikes_generated": len(self.spike_queue)
			})
			
		except Exception as e:
			self._log_neuro_operation("feature_encoding_error", {"error": str(e)})
	
	async def _simulate_network(self, duration_ms: float = 10.0) -> List[Spike]:
		"""Simulate the neuromorphic network for given duration"""
		output_spikes = []
		
		try:
			time_step = 0.1  # 0.1 ms time steps
			steps = int(duration_ms / time_step)
			
			for step in range(steps):
				current_time = self.current_time + step * time_step
				
				# Process input spikes
				while self.spike_queue and self.spike_queue[0].timestamp <= current_time:
					spike = self.spike_queue.popleft()
					await self._propagate_spike(spike, current_time)
				
				# Update all neurons
				for layer in self.layers:
					for neuron in layer.neurons:
						spike = await self._update_neuron(neuron, current_time)
						if spike:
							if neuron.neuron_type == NeuronType.OUTPUT:
								output_spikes.append(spike)
							else:
								# Propagate to connected neurons
								await self._propagate_spike(spike, current_time)
			
			self.metrics["spike_frequency"] = len(output_spikes) / duration_ms * 1000  # Hz
			
			return output_spikes
			
		except Exception as e:
			self._log_neuro_operation("network_simulation_error", {"error": str(e)})
			return []
	
	async def _update_neuron(self, neuron: SpikingNeuron, current_time: float) -> Optional[Spike]:
		"""Update individual neuron state"""
		try:
			# Check refractory period
			if current_time - neuron.last_spike_time < neuron.refractory_period:
				return None
			
			# Apply membrane leak
			neuron.membrane_potential *= (1 - neuron.leak_rate * 0.1)
			
			# Check for threshold crossing
			if neuron.membrane_potential >= neuron.threshold:
				# Generate spike
				spike = Spike(
					neuron_id=neuron.id,
					timestamp=current_time,
					amplitude=neuron.membrane_potential,
					spike_type=SpikeType.EXCITATORY if neuron.neuron_type != NeuronType.INHIBITORY else SpikeType.INHIBITORY
				)
				
				# Reset neuron
				neuron.membrane_potential = neuron.resting_potential
				neuron.last_spike_time = current_time
				neuron.spike_count += 1
				
				# Adaptation
				neuron.threshold += neuron.adaptation_rate
				
				neuron.output_spikes.append(spike)
				return spike
			
			return None
			
		except Exception as e:
			self._log_neuro_operation("neuron_update_error", {"error": str(e)})
			return None
	
	async def _propagate_spike(self, spike: Spike, current_time: float) -> None:
		"""Propagate spike through synaptic connections"""
		try:
			# Find all synapses from this neuron
			for synapse_key, synapse in self.global_synapses.items():
				if synapse.pre_neuron_id == spike.neuron_id:
					# Calculate arrival time
					arrival_time = current_time + synapse.delay
					
					# Find post-synaptic neuron
					post_neuron = await self._find_neuron_by_id(synapse.post_neuron_id)
					if post_neuron:
						# Apply synaptic input
						input_current = spike.amplitude * synapse.weight
						
						# Add to membrane potential (with delay)
						if arrival_time <= self.current_time + 10:  # Within simulation window
							post_neuron.membrane_potential += input_current
							post_neuron.input_history.append(input_current)
							
							# Keep history limited
							if len(post_neuron.input_history) > 100:
								post_neuron.input_history.pop(0)
		
		except Exception as e:
			self._log_neuro_operation("spike_propagation_error", {"error": str(e)})
	
	async def _find_neuron_by_id(self, neuron_id: str) -> Optional[SpikingNeuron]:
		"""Find neuron by ID across all layers"""
		for layer in self.layers:
			for neuron in layer.neurons:
				if neuron.id == neuron_id:
					return neuron
		return None
	
	async def _decode_decision(self, output_spikes: List[Spike]) -> Tuple[AuthenticationDecision, float]:
		"""Decode authentication decision from output spikes"""
		try:
			decision_layer = self.layers[3]  # decision_output layer
			decision_counts = defaultdict(int)
			decision_amplitudes = defaultdict(list)
			
			# Count spikes per decision type
			for spike in output_spikes:
				for neuron in decision_layer.neurons:
					if neuron.id == spike.neuron_id:
						decision_type = neuron.metadata.get("decision_type")
						if decision_type:
							decision_counts[decision_type] += 1
							decision_amplitudes[decision_type].append(spike.amplitude)
			
			if not decision_counts:
				return AuthenticationDecision.DENY, 0.1
			
			# Find decision with highest spike count
			max_decision = max(decision_counts.items(), key=lambda x: x[1])
			decision_type = max_decision[0]
			spike_count = max_decision[1]
			
			# Calculate confidence based on spike count and amplitude
			avg_amplitude = np.mean(decision_amplitudes[decision_type]) if decision_amplitudes[decision_type] else 1.0
			total_spikes = sum(decision_counts.values())
			
			confidence = min(1.0, (spike_count / max(1, total_spikes)) * (avg_amplitude / 2.0))
			
			# Convert to enum
			decision = AuthenticationDecision(decision_type)
			
			return decision, confidence
			
		except Exception as e:
			self._log_neuro_operation("decision_decoding_error", {"error": str(e)})
			return AuthenticationDecision.DENY, 0.0
	
	async def _get_network_state(self) -> Dict[str, Any]:
		"""Get current network state summary"""
		try:
			state = {
				"layer_activations": {},
				"total_neurons": 0,
				"active_neurons": 0,
				"average_membrane_potential": 0.0,
				"synapse_weights_distribution": {}
			}
			
			total_potential = 0.0
			active_count = 0
			
			for layer in self.layers:
				layer_info = {
					"neuron_count": len(layer.neurons),
					"average_potential": 0.0,
					"spike_rate": 0.0
				}
				
				layer_potential = 0.0
				layer_spikes = 0
				
				for neuron in layer.neurons:
					layer_potential += neuron.membrane_potential
					layer_spikes += neuron.spike_count
					total_potential += neuron.membrane_potential
					
					if neuron.membrane_potential > 0.1:
						active_count += 1
				
				if len(layer.neurons) > 0:
					layer_info["average_potential"] = layer_potential / len(layer.neurons)
					layer_info["spike_rate"] = layer_spikes / len(layer.neurons)
				
				state["layer_activations"][layer.name] = layer_info
				state["total_neurons"] += len(layer.neurons)
			
			state["active_neurons"] = active_count
			if state["total_neurons"] > 0:
				state["average_membrane_potential"] = total_potential / state["total_neurons"]
			
			# Analyze synapse weights
			weights = [synapse.weight for synapse in self.global_synapses.values()]
			if weights:
				state["synapse_weights_distribution"] = {
					"mean": np.mean(weights),
					"std": np.std(weights),
					"min": np.min(weights),
					"max": np.max(weights)
				}
			
			return state
			
		except Exception as e:
			self._log_neuro_operation("network_state_error", {"error": str(e)})
			return {}
	
	async def _analyze_feature_importance(self, context: AuthenticationContext) -> Dict[str, float]:
		"""Analyze feature importance based on network activation"""
		try:
			input_layer = self.layers[0]
			importance_scores = {}
			
			feature_groups = [
				("behavioral", context.behavioral_features),
				("biometric", context.biometric_features),
				("contextual", context.contextual_features),
				("risk", context.risk_indicators)
			]
			
			neuron_index = 0
			for group_name, features in feature_groups:
				for i, feature_value in enumerate(features):
					if neuron_index < len(input_layer.neurons):
						neuron = input_layer.neurons[neuron_index]
						
						# Feature importance based on downstream activation
						downstream_activation = 0.0
						for synapse_key, synapse in self.global_synapses.items():
							if synapse.pre_neuron_id == neuron.id:
								post_neuron = await self._find_neuron_by_id(synapse.post_neuron_id)
								if post_neuron:
									downstream_activation += post_neuron.membrane_potential * abs(synapse.weight)
						
						feature_key = f"{group_name}_{i}"
						importance_scores[feature_key] = downstream_activation * feature_value
						
						neuron_index += 1
			
			# Normalize importance scores
			if importance_scores:
				max_importance = max(importance_scores.values())
				if max_importance > 0:
					for key in importance_scores:
						importance_scores[key] /= max_importance
			
			return importance_scores
			
		except Exception as e:
			self._log_neuro_operation("feature_importance_error", {"error": str(e)})
			return {}
	
	async def _trigger_learning(
		self, 
		context: AuthenticationContext, 
		decision: AuthenticationDecision, 
		confidence: float
	) -> None:
		"""Trigger neuromorphic learning based on authentication outcome"""
		try:
			# STDP (Spike-Timing-Dependent Plasticity) learning
			await self._apply_stdp_learning()
			
			# Homeostatic plasticity to maintain network stability
			await self._apply_homeostatic_plasticity()
			
			# Record learning event
			self.adaptation_history.append({
				"timestamp": datetime.utcnow(),
				"user_id": context.user_id,
				"decision": decision.value,
				"confidence": confidence,
				"learning_type": "stdp_homeostatic"
			})
			
			self.metrics["learning_iterations"] += 1
			
		except Exception as e:
			self._log_neuro_operation("learning_error", {"error": str(e)})
	
	async def _apply_stdp_learning(self) -> None:
		"""Apply Spike-Timing-Dependent Plasticity learning"""
		try:
			for synapse_key, synapse in self.global_synapses.items():
				pre_neuron = await self._find_neuron_by_id(synapse.pre_neuron_id)
				post_neuron = await self._find_neuron_by_id(synapse.post_neuron_id)
				
				if pre_neuron and post_neuron and pre_neuron.output_spikes and post_neuron.output_spikes:
					# Find recent spike pairs
					pre_spikes = [s for s in pre_neuron.output_spikes if s.timestamp > self.current_time - 20]
					post_spikes = [s for s in post_neuron.output_spikes if s.timestamp > self.current_time - 20]
					
					weight_change = 0.0
					for pre_spike in pre_spikes:
						for post_spike in post_spikes:
							dt = post_spike.timestamp - pre_spike.timestamp
							
							if abs(dt) < 20:  # 20ms STDP window
								if dt > 0:  # Post after pre - strengthen
									weight_change += 0.01 * math.exp(-abs(dt) / 10.0)
								else:  # Pre after post - weaken
									weight_change -= 0.005 * math.exp(-abs(dt) / 10.0)
					
					# Apply weight change with bounds
					synapse.weight = max(-2.0, min(2.0, synapse.weight + weight_change * synapse.plasticity))
					synapse.last_update = self.current_time
			
		except Exception as e:
			self._log_neuro_operation("stdp_learning_error", {"error": str(e)})
	
	async def _apply_homeostatic_plasticity(self) -> None:
		"""Apply homeostatic plasticity to maintain network stability"""
		try:
			target_activity = 0.1  # Target activity level
			
			for layer in self.layers:
				for neuron in layer.neurons:
					recent_activity = len([s for s in neuron.output_spikes if s.timestamp > self.current_time - 100])
					activity_rate = recent_activity / 100.0  # Activity in last 100ms
					
					# Adjust threshold to maintain target activity
					if activity_rate > target_activity:
						neuron.threshold += 0.01  # Increase threshold if too active
					elif activity_rate < target_activity:
						neuron.threshold = max(0.1, neuron.threshold - 0.01)  # Decrease threshold if too quiet
					
					# Bound threshold
					neuron.threshold = max(0.1, min(3.0, neuron.threshold))
			
		except Exception as e:
			self._log_neuro_operation("homeostatic_plasticity_error", {"error": str(e)})
	
	async def adapt_to_user_behavior(
		self, 
		user_id: str, 
		authentication_history: List[Dict[str, Any]]
	) -> None:
		"""Adapt neuromorphic processor to specific user behavior patterns"""
		try:
			if len(authentication_history) < 10:
				return  # Need sufficient history
			
			# Analyze user-specific patterns
			user_decisions = [h for h in authentication_history if h.get("user_id") == user_id]
			
			if len(user_decisions) < 5:
				return
			
			# Calculate user-specific thresholds based on historical success
			successful_authentications = [d for d in user_decisions if d.get("correct", True)]
			success_rate = len(successful_authentications) / len(user_decisions)
			
			# Adjust decision layer sensitivity based on user history
			decision_layer = self.layers[3]
			
			for neuron in decision_layer.neurons:
				decision_type = neuron.metadata.get("decision_type")
				
				if decision_type == "allow" and success_rate > 0.9:
					# Lower threshold for reliable users
					neuron.threshold *= 0.95
				elif decision_type == "deny" and success_rate < 0.7:
					# Lower threshold for risky users
					neuron.threshold *= 0.9
				elif decision_type == "challenge":
					# Adjust challenge threshold based on user preference
					neuron.threshold *= (1.0 - success_rate * 0.2)
			
			# Record adaptation
			self.metrics["adaptation_events"] += 1
			
			self._log_neuro_operation("user_adaptation_completed", {
				"user_id": user_id,
				"history_size": len(user_decisions),
				"success_rate": success_rate
			})
			
		except Exception as e:
			self._log_neuro_operation("user_adaptation_error", {
				"user_id": user_id,
				"error": str(e)
			})
	
	async def get_performance_metrics(self) -> Dict[str, Any]:
		"""Get detailed performance metrics of the neuromorphic processor"""
		try:
			network_state = await self._get_network_state()
			
			metrics = {
				"processing_metrics": self.metrics.copy(),
				"network_state": network_state,
				"adaptation_stats": {
					"total_adaptations": len(self.adaptation_history),
					"recent_adaptations": len([
						a for a in self.adaptation_history 
						if (datetime.utcnow() - a["timestamp"]).total_seconds() < 3600
					])
				},
				"decision_distribution": {},
				"learning_effectiveness": 0.0
			}
			
			# Analyze decision distribution
			if self.decision_history:
				decision_counts = defaultdict(int)
				for decision in self.decision_history:
					decision_counts[decision["decision"]] += 1
				
				total_decisions = len(self.decision_history)
				for decision_type, count in decision_counts.items():
					metrics["decision_distribution"][decision_type] = count / total_decisions
			
			# Calculate learning effectiveness
			if len(self.decision_history) > 20:
				recent_decisions = self.decision_history[-10:]
				older_decisions = self.decision_history[-20:-10]
				
				recent_avg_confidence = np.mean([d["confidence"] for d in recent_decisions])
				older_avg_confidence = np.mean([d["confidence"] for d in older_decisions])
				
				metrics["learning_effectiveness"] = recent_avg_confidence - older_avg_confidence
			
			return metrics
			
		except Exception as e:
			self._log_neuro_operation("metrics_calculation_error", {"error": str(e)})
			return {}


# Usage example and testing functions

async def create_sample_authentication_context() -> AuthenticationContext:
	"""Create sample authentication context for testing"""
	return AuthenticationContext(
		user_id="user_12345",
		session_id="session_67890",
		behavioral_features=[
			0.85, 0.92, 0.78, 0.88, 0.95, 0.73, 0.82, 0.89, 0.76, 0.91,
			0.84, 0.87, 0.79, 0.93, 0.81, 0.86, 0.77, 0.90, 0.83, 0.94
		],
		biometric_features=[
			0.96, 0.89, 0.94, 0.87, 0.92, 0.85, 0.91, 0.88, 0.95, 0.86,
			0.93, 0.84, 0.90, 0.89, 0.97
		],
		contextual_features=[
			0.75, 0.82, 0.88, 0.79, 0.85, 0.91, 0.77, 0.84, 0.90, 0.81
		],
		risk_indicators=[
			0.12, 0.08, 0.15, 0.06, 0.10, 0.14, 0.09, 0.11
		]
	)


async def demo_neuromorphic_authentication():
	"""Demonstrate neuromorphic authentication capabilities"""
	print("=== Neuromorphic Authentication Processor Demo ===")
	
	# Create processor
	processor = NeuromorphicProcessor()
	
	print(f"Initialized processor with {len(processor.layers)} layers")
	print(f"Total neurons: {sum(len(layer.neurons) for layer in processor.layers)}")
	print(f"Total synapses: {len(processor.global_synapses)}")
	
	# Test authentication
	context = await create_sample_authentication_context()
	
	print(f"\nProcessing authentication for user: {context.user_id}")
	
	decision, confidence, metadata = await processor.process_authentication(context)
	
	print(f"Decision: {decision.value}")
	print(f"Confidence: {confidence:.3f}")
	print(f"Processing time: {metadata.get('processing_time_ms', 0):.2f} ms")
	print(f"Spike count: {metadata.get('spike_count', 0)}")
	
	# Test multiple authentications for adaptation
	print(f"\nTesting adaptive learning with multiple authentications...")
	
	for i in range(5):
		# Vary the context slightly
		context.behavioral_features = [f + random.gauss(0, 0.1) for f in context.behavioral_features]
		context.risk_indicators = [max(0, f + random.gauss(0, 0.05)) for f in context.risk_indicators]
		
		decision, confidence, _ = await processor.process_authentication(context)
		print(f"  Authentication {i+1}: {decision.value} (confidence: {confidence:.3f})")
	
	# Get performance metrics
	metrics = await processor.get_performance_metrics()
	print(f"\nPerformance Metrics:")
	print(f"  Total decisions: {metrics['processing_metrics']['total_decisions']}")
	print(f"  Average processing time: {metrics['processing_metrics']['average_processing_time']:.2f} ms")
	print(f"  Learning iterations: {metrics['processing_metrics']['learning_iterations']}")
	print(f"  Learning effectiveness: {metrics.get('learning_effectiveness', 0):.3f}")
	
	print("=== Demo Complete ===")


if __name__ == "__main__":
	asyncio.run(demo_neuromorphic_authentication())