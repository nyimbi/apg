import pytest

from capabilities.composition.gateway.topology_3d_engine import Topology3DEngine


@pytest.mark.asyncio
async def test_topology_3d_engine_generates_scene_from_services_and_connections() -> None:
	engine = Topology3DEngine()

	scene = await engine.generate_3d_scene({
		"services": ["orders", "billing", "inventory"],
		"connections": [
			{"source": "orders", "target": "billing", "strength": 0.8},
			{"source": "orders", "target": "inventory", "latency": 23.0},
		],
		"metrics": {
			"orders": {"traffic_volume": 120.0, "cpu_usage": 64.0},
			"billing": {"error_rate": 1.2},
		},
	})

	assert len(scene["nodes"]) == 3
	assert len(scene["edges"]) == 2
	assert scene["topology_summary"]["node_count"] == 3
	assert scene["nodes"][0]["position"].keys() == {"x", "y", "z"}
	assert scene["edges"][0]["metrics"]["traffic_flow"] == 80.0


@pytest.mark.asyncio
async def test_topology_3d_engine_marks_vr_scene_optimized() -> None:
	engine = Topology3DEngine()
	scene = await engine.generate_3d_scene({
		"services": ["gateway", "api"],
		"connections": [{"source": "gateway", "target": "api"}],
	})

	vr_scene = await engine.optimize_for_vr(scene)

	assert vr_scene["optimized"] is True
	assert vr_scene["vr_options"]["webxr_enabled"] is True
	assert vr_scene["optimization_summary"] == {
		"node_count": 2,
		"edge_count": 1,
		"instanced_edges": False,
	}
