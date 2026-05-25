"""First-class edge computing capability runtime for APG."""

from __future__ import annotations

import asyncio
import inspect
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Any, Callable

from uuid_extensions import uuid7str


class EdgeNodeType(StrEnum):
	"""Supported edge node categories."""

	GATEWAY = "gateway"
	COMPUTE = "compute"
	STORAGE = "storage"
	HYBRID = "hybrid"


class EdgeTaskPriority(IntEnum):
	"""Priority levels for edge tasks."""

	LOW = 1
	NORMAL = 5
	HIGH = 10
	CRITICAL = 20


@dataclass
class EdgeComputingNode:
	"""An edge node with resource capacity and health metadata."""

	name: str
	node_type: EdgeNodeType
	location: dict[str, float]
	capacity: dict[str, float]
	network_latency_ms: float = 10.0
	reliability_score: float = 0.9
	specialized_capabilities: list[str] = field(default_factory=list)
	id: str = field(default_factory=uuid7str)
	status: str = "active"
	current_load: dict[str, float] = field(default_factory=lambda: {"cpu_utilization": 0.0})
	last_heartbeat: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EdgeTask:
	"""A task that can be scheduled onto an edge node."""

	twin_id: str
	task_type: str
	priority: EdgeTaskPriority
	requirements: dict[str, Any]
	payload: dict[str, Any]
	id: str = field(default_factory=uuid7str)
	status: str = "pending"
	assigned_node: str | None = None
	result: dict[str, Any] | None = None
	created_at: datetime = field(default_factory=datetime.utcnow)
	started_at: datetime | None = None
	completed_at: datetime | None = None
	execution_time_ms: float | None = None


class EdgeStreamProcessor:
	"""Register and execute synchronous or asynchronous stream processors."""

	def __init__(self, buffer_size: int = 1000):
		self.buffer_size = buffer_size
		self.processing_functions: dict[str, Callable[[Any], Any]] = {}
		self.stream_buffers: dict[str, deque[Any]] = {}
		self.processing_stats: dict[str, dict[str, Any]] = {}

	def register_processor(self, stream_name: str, processor: Callable[[Any], Any]) -> None:
		self.processing_functions[stream_name] = processor
		self.stream_buffers[stream_name] = deque(maxlen=self.buffer_size)
		self.processing_stats[stream_name] = {
			"processed": 0,
			"errors": 0,
			"avg_latency_ms": 0.0,
			"last_processed": None,
		}

	async def process_stream_data(self, stream_name: str, data: Any) -> dict[str, Any]:
		start = time.perf_counter()
		stats = self.processing_stats.setdefault(
			stream_name,
			{"processed": 0, "errors": 0, "avg_latency_ms": 0.0, "last_processed": None},
		)
		try:
			processor = self.processing_functions[stream_name]
			self.stream_buffers.setdefault(stream_name, deque(maxlen=self.buffer_size)).append(data)
			result = processor(data)
			if inspect.isawaitable(result):
				result = await result
			latency_ms = (time.perf_counter() - start) * 1000
			stats["processed"] += 1
			stats["avg_latency_ms"] = self._moving_average(stats["avg_latency_ms"], latency_ms, stats["processed"])
			stats["last_processed"] = datetime.utcnow()
			return {"result": result, "processing_time_ms": latency_ms}
		except Exception as exc:
			latency_ms = (time.perf_counter() - start) * 1000
			stats["errors"] += 1
			stats["last_processed"] = datetime.utcnow()
			return {"error": str(exc), "processing_time_ms": latency_ms}

	@staticmethod
	def _moving_average(previous: float, value: float, count: int) -> float:
		if count <= 1:
			return value
		return ((previous * (count - 1)) + value) / count


class EdgeLoadBalancer:
	"""Load prediction and distribution scoring for edge nodes."""

	def __init__(self):
		self.load_history: dict[str, list[float]] = defaultdict(list)

	def predict_node_load(self, node_id: str, current_load: float) -> float:
		history = self.load_history[node_id]
		history.append(current_load)
		if len(history) < 3:
			return current_load
		weights = list(range(1, len(history) + 1))
		return sum(load * weight for load, weight in zip(history, weights)) / sum(weights)

	def calculate_load_distribution_score(self, nodes: list[EdgeComputingNode]) -> float:
		if not nodes:
			return 0.0
		loads = [float(node.current_load.get("cpu_utilization", 0.0)) for node in nodes]
		if len(loads) == 1:
			return 100.0
		spread = statistics.pstdev(loads)
		return max(0.0, 100.0 - spread)


class EdgeComputingCluster:
	"""In-memory edge cluster with resource-aware task scheduling."""

	def __init__(self):
		self.nodes: dict[str, EdgeComputingNode] = {}
		self.tasks: dict[str, EdgeTask] = {}
		self.task_queue: list[str] = []
		self.performance_metrics: dict[str, dict[str, Any]] = {}
		self.stream_processor = EdgeStreamProcessor()
		self.load_balancer = EdgeLoadBalancer()
		self._running = False
		self._next_node_index = 0

	async def add_node(self, node: EdgeComputingNode) -> bool:
		self.nodes[node.id] = node
		self.performance_metrics[node.id] = {
			"tasks_processed": 0,
			"total_latency_ms": 0.0,
			"failures": 0,
		}
		return True

	async def submit_task(self, task: EdgeTask) -> str:
		self.tasks[task.id] = task
		self.task_queue.append(task.id)
		self.task_queue.sort(key=lambda task_id: self.tasks[task_id].priority, reverse=True)
		return task.id

	async def schedule_tasks(self) -> None:
		while self._running:
			if not self.task_queue:
				await asyncio.sleep(0.01)
				continue
			task_id = self.task_queue.pop(0)
			task = self.tasks[task_id]
			node = await self._find_optimal_node(task)
			if node is None:
				task.status = "pending"
				self.task_queue.append(task_id)
				await asyncio.sleep(0.02)
				continue
			await self._execute_task_on_node(task, node)
			await asyncio.sleep(0)

	async def _execute_task_on_node(self, task: EdgeTask, node: EdgeComputingNode) -> None:
		start = time.perf_counter()
		task.status = "executing"
		task.started_at = datetime.utcnow()
		task.assigned_node = node.id
		await asyncio.sleep(min(0.005 + node.network_latency_ms / 100_000, 0.02))
		task.result = {
			"node_id": node.id,
			"task_type": task.task_type,
			"payload_keys": list(task.payload.keys()),
			"processed": True,
		}
		task.status = "completed"
		task.completed_at = datetime.utcnow()
		task.execution_time_ms = (time.perf_counter() - start) * 1000
		metrics = self.performance_metrics[node.id]
		metrics["tasks_processed"] += 1
		metrics["total_latency_ms"] += task.execution_time_ms
		node.current_load["cpu_utilization"] = min(
			95.0,
			float(node.current_load.get("cpu_utilization", 0.0)) + float(task.requirements.get("cpu_cores", 0.5)) * 5,
		)

	async def _find_optimal_node(self, task: EdgeTask) -> EdgeComputingNode | None:
		candidates = [
			node for node in self.nodes.values()
			if node.status == "active" and self._can_node_handle_task(node, task)
		]
		if not candidates:
			return None
		candidates.sort(key=lambda node: self._node_score(node, task), reverse=True)

		if len(candidates) > 1 and task.priority < EdgeTaskPriority.CRITICAL:
			start = self._next_node_index % len(candidates)
			self._next_node_index += 1
			return candidates[start]
		return candidates[0]

	def _can_node_handle_task(self, node: EdgeComputingNode, task: EdgeTask) -> bool:
		if node.status != "active":
			return False
		if float(task.requirements.get("cpu_cores", 0.0)) > float(node.capacity.get("cpu_cores", 0.0)):
			return False
		memory_required_mb = float(task.requirements.get("memory_mb", 0.0))
		if memory_required_mb > float(node.capacity.get("memory_gb", 0.0)) * 1024:
			return False
		if task.requirements.get("gpu_required") and float(node.capacity.get("gpu_cores", 0.0)) <= 0:
			return False
		max_latency = task.requirements.get("max_latency_ms")
		if max_latency is not None and node.network_latency_ms > float(max_latency) * 5:
			return False
		return True

	def _node_score(self, node: EdgeComputingNode, task: EdgeTask) -> float:
		load = float(node.current_load.get("cpu_utilization", 0.0))
		latency_limit = float(task.requirements.get("max_latency_ms", 100.0))
		latency_score = max(0.0, latency_limit + 10.0 - node.network_latency_ms)
		reliability_score = node.reliability_score * 100.0
		capability_bonus = 20.0 if task.task_type in node.specialized_capabilities else 0.0
		return reliability_score + latency_score + capability_bonus - load

	async def _monitor_node_heartbeats(self) -> None:
		while self._running:
			cutoff = datetime.utcnow() - timedelta(seconds=60)
			for node in self.nodes.values():
				if node.last_heartbeat < cutoff:
					node.status = "inactive"
			await asyncio.sleep(0.05)

	def get_cluster_status(self) -> dict[str, Any]:
		completed = [task for task in self.tasks.values() if task.status == "completed"]
		total_latency = sum((task.execution_time_ms or 0.0) for task in completed)
		total_processed = len(completed)
		return {
			"nodes": {
				"total": len(self.nodes),
				"active": sum(1 for node in self.nodes.values() if node.status == "active"),
			},
			"tasks": {
				"total": len(self.tasks),
				"completed": total_processed,
				"pending": sum(1 for task in self.tasks.values() if task.status == "pending"),
			},
			"performance": {
				"cluster": {
					"total_tasks_processed": total_processed,
					"avg_task_latency_ms": total_latency / total_processed if total_processed else 0.0,
				},
				"nodes": self.performance_metrics,
			},
			"queue_size": len(self.task_queue),
			"timestamp": datetime.utcnow().isoformat(),
		}

	async def stop_cluster(self) -> None:
		self._running = False
		await asyncio.sleep(0)


class EdgeEnabledDigitalTwin:
	"""Digital twin facade backed by an edge cluster."""

	def __init__(self, twin_id: str):
		self.twin_id = twin_id
		self.edge_cluster = EdgeComputingCluster()
		self.real_time_processors: dict[str, Callable[[Any], Any]] = {}

	async def deploy_to_edge(self, nodes: list[EdgeComputingNode]) -> None:
		for node in nodes:
			await self.edge_cluster.add_node(node)
		self.edge_cluster._running = True

	async def register_real_time_processor(self, stream_name: str, processor: Callable[[Any], Any]) -> None:
		self.real_time_processors[stream_name] = processor
		self.edge_cluster.stream_processor.register_processor(stream_name, processor)

	async def process_real_time_data(
		self,
		stream_name: str,
		data: Any,
		max_latency_ms: float = 50.0,
	) -> dict[str, Any]:
		start = time.perf_counter()
		result = await self.edge_cluster.stream_processor.process_stream_data(stream_name, data)
		latency_ms = (time.perf_counter() - start) * 1000
		network_latency = min(
			(node.network_latency_ms for node in self.edge_cluster.nodes.values()),
			default=0.0,
		)
		if latency_ms + network_latency > max_latency_ms:
			raise TimeoutError(f"Edge processing exceeded {max_latency_ms}ms")
		result["total_latency_ms"] = latency_ms + network_latency
		return result

	def get_edge_performance_metrics(self) -> dict[str, Any]:
		return {
			"twin_id": self.twin_id,
			"edge_cluster": self.edge_cluster.get_cluster_status(),
			"real_time_processors": list(self.real_time_processors),
			"total_edge_tasks": len(self.edge_cluster.tasks),
		}


__all__ = [
	"EdgeComputingCluster",
	"EdgeComputingNode",
	"EdgeEnabledDigitalTwin",
	"EdgeLoadBalancer",
	"EdgeNodeType",
	"EdgeStreamProcessor",
	"EdgeTask",
	"EdgeTaskPriority",
]
