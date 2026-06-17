"""
APG Configuration Management - Edge Computing & IoT Integration
Production edge computing configuration management with intelligent orchestration.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from uuid_extensions import uuid7str

from .models import CMResource, ConfigurationDSL


class EdgeDeviceType(str, Enum):
    """Edge device types"""
    INDUSTRIAL_IOT = "industrial_iot"
    SMART_CITY = "smart_city"
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"
    RETAIL_EDGE = "retail_edge"
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    AGRICULTURE = "agriculture"
    TELECOMMUNICATIONS = "telecommunications"


class EdgeConnectivity(str, Enum):
    """Edge connectivity types"""
    FIBER = "fiber"
    CELLULAR_5G = "cellular_5g"
    CELLULAR_4G = "cellular_4g"
    SATELLITE = "satellite"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    LORA = "lora"
    NFC = "nfc"


class EdgeComputeCapability(str, Enum):
    """Edge compute capability levels"""
    MICRO = "micro"          # <1 CPU, <1GB RAM
    NANO = "nano"            # 1-2 CPU, 1-4GB RAM
    SMALL = "small"          # 2-4 CPU, 4-16GB RAM
    MEDIUM = "medium"        # 4-8 CPU, 16-64GB RAM
    LARGE = "large"          # 8+ CPU, 64GB+ RAM
    GPU_ACCELERATED = "gpu_accelerated"
    FPGA_ACCELERATED = "fpga_accelerated"


class EdgeDeploymentStrategy(str, Enum):
    """Edge deployment strategies"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"
    GEOGRAPHIC_ROLLOUT = "geographic_rollout"
    DEVICE_TYPE_ROLLOUT = "device_type_rollout"


class EdgeDevice(BaseModel):
    """Edge device model"""
    
    id: str = Field(default_factory=uuid7str)
    name: str = Field(..., min_length=3, max_length=100)
    device_type: EdgeDeviceType
    location: Dict[str, Any] = Field(..., description="Geographic location and metadata")
    hardware_specs: Dict[str, Any] = Field(..., description="Hardware specifications")
    connectivity: List[EdgeConnectivity] = Field(..., min_length=1)
    compute_capability: EdgeComputeCapability
    current_config_version: Optional[str] = None
    health_status: str = Field(default="unknown")
    last_heartbeat: Optional[datetime] = None
    configuration_state: str = Field(default="unmanaged")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('location')
    def validate_location(cls, v):
        required_fields = ['latitude', 'longitude', 'timezone']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Location must contain {field}")
        return v


class EdgeCluster(BaseModel):
    """Edge cluster model for grouped device management"""
    
    id: str = Field(default_factory=uuid7str)
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    devices: List[str] = Field(default_factory=list, description="Device IDs in cluster")
    geographic_region: Dict[str, Any] = Field(..., description="Geographic boundary")
    cluster_type: str = Field(..., description="Cluster purpose/type")
    load_balancing_strategy: str = Field(default="round_robin")
    failover_configuration: Dict[str, Any] = Field(default_factory=dict)
    health_score: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EdgeConfiguration(BaseModel):
    """Edge-specific configuration model"""
    
    id: str = Field(default_factory=uuid7str)
    name: str = Field(..., min_length=3, max_length=100)
    target_devices: List[str] = Field(..., min_length=1, description="Target device IDs")
    target_clusters: List[str] = Field(default_factory=list, description="Target cluster IDs")
    configuration_spec: Dict[str, Any] = Field(..., description="Edge-specific configuration")
    resource_constraints: Dict[str, Any] = Field(default_factory=dict)
    network_policies: Dict[str, Any] = Field(default_factory=dict)
    security_policies: Dict[str, Any] = Field(default_factory=dict)
    deployment_strategy: EdgeDeploymentStrategy = Field(default=EdgeDeploymentStrategy.ROLLING_UPDATE)
    rollback_configuration: Dict[str, Any] = Field(default_factory=dict)
    monitoring_configuration: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0")


class EdgeDeploymentExecution(BaseModel):
    """Edge deployment execution tracking"""
    
    id: str = Field(default_factory=uuid7str)
    configuration_id: str
    strategy: EdgeDeploymentStrategy
    target_devices: List[str]
    target_clusters: List[str]
    status: str = Field(default="pending")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    successful_devices: List[str] = Field(default_factory=list)
    failed_devices: List[str] = Field(default_factory=list)
    rollback_triggered: bool = Field(default=False)
    health_checks: List[Dict[str, Any]] = Field(default_factory=list)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)


class EdgeComputingManager:
    """Production edge computing configuration manager"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.devices: Dict[str, EdgeDevice] = {}
        self.clusters: Dict[str, EdgeCluster] = {}
        self.configurations: Dict[str, EdgeConfiguration] = {}
        self.deployments: Dict[str, EdgeDeploymentExecution] = {}
        self.device_heartbeats: Dict[str, datetime] = {}
        
        # Edge-specific optimization settings
        self.edge_settings = {
            "heartbeat_interval_seconds": 30,
            "max_configuration_size_mb": 50,
            "deployment_timeout_minutes": 30,
            "health_check_interval_seconds": 60,
            "connectivity_timeout_seconds": 10,
            "batch_deployment_size": 100,
            "geographic_optimization": True,
            "bandwidth_optimization": True
        }
    
    async def register_edge_device(
        self,
        device_config: Dict[str, Any]
    ) -> str:
        """Register new edge device with the platform"""
        
        # Validate device configuration
        device = EdgeDevice(**device_config)
        
        # Perform device capability assessment
        capability_score = await self._assess_device_capabilities(device)
        
        # Initialize device monitoring
        device.configuration_state = "registered"
        device.health_status = "initializing"
        device.last_heartbeat = datetime.now()
        
        # Store device
        self.devices[device.id] = device
        self.device_heartbeats[device.id] = datetime.now()
        
        # Auto-discover nearby devices for clustering
        await self._discover_device_clusters(device.id)
        
        # Initialize device-specific monitoring
        await self._initialize_device_monitoring(device.id)
        
        return device.id
    
    async def create_edge_cluster(
        self,
        cluster_config: Dict[str, Any]
    ) -> str:
        """Create edge device cluster for coordinated management"""
        
        cluster = EdgeCluster(**cluster_config)
        
        # Validate all devices exist and are compatible
        for device_id in cluster.devices:
            if device_id not in self.devices:
                raise ValueError(f"Device {device_id} not found")
        
        # Optimize cluster configuration based on geographic proximity
        if self.edge_settings["geographic_optimization"]:
            await self._optimize_cluster_geography(cluster)
        
        # Configure cluster networking and load balancing
        await self._configure_cluster_networking(cluster)
        
        # Initialize cluster health monitoring
        await self._initialize_cluster_monitoring(cluster)
        
        self.clusters[cluster.id] = cluster
        
        return cluster.id
    
    async def create_edge_configuration(
        self,
        config_data: Dict[str, Any]
    ) -> str:
        """Create edge-specific configuration with optimization"""
        
        edge_config = EdgeConfiguration(**config_data)
        
        # Validate target devices and clusters
        await self._validate_edge_targets(edge_config)
        
        # Optimize configuration for edge constraints
        optimized_spec = await self._optimize_edge_configuration(edge_config)
        edge_config.configuration_spec = optimized_spec
        
        # Apply bandwidth optimization if enabled
        if self.edge_settings["bandwidth_optimization"]:
            await self._optimize_configuration_bandwidth(edge_config)
        
        # Generate resource constraints automatically
        edge_config.resource_constraints = await self._generate_resource_constraints(edge_config)
        
        # Create security policies for edge deployment
        edge_config.security_policies = await self._generate_edge_security_policies(edge_config)
        
        self.configurations[edge_config.id] = edge_config
        
        return edge_config.id
    
    async def deploy_edge_configuration(
        self,
        configuration_id: str,
        deployment_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy configuration to edge devices with advanced orchestration"""
        
        if configuration_id not in self.configurations:
            raise ValueError(f"Configuration {configuration_id} not found")
        
        config = self.configurations[configuration_id]
        options = deployment_options or {}
        
        # Create deployment execution
        deployment = EdgeDeploymentExecution(
            configuration_id=configuration_id,
            strategy=config.deployment_strategy,
            target_devices=config.target_devices.copy(),
            target_clusters=config.target_clusters.copy()
        )
        
        # Expand cluster targets to individual devices
        await self._expand_cluster_targets(deployment)
        
        # Optimize deployment order based on geography and connectivity
        deployment_order = await self._optimize_deployment_order(deployment)
        
        # Start deployment execution
        deployment.status = "running"
        deployment.started_at = datetime.now()
        
        self.deployments[deployment.id] = deployment
        
        # Execute deployment strategy
        if deployment.strategy == EdgeDeploymentStrategy.ROLLING_UPDATE:
            await self._execute_rolling_deployment(deployment, deployment_order)
        elif deployment.strategy == EdgeDeploymentStrategy.BLUE_GREEN:
            await self._execute_blue_green_deployment(deployment, deployment_order)
        elif deployment.strategy == EdgeDeploymentStrategy.CANARY:
            await self._execute_canary_deployment(deployment, deployment_order)
        elif deployment.strategy == EdgeDeploymentStrategy.GEOGRAPHIC_ROLLOUT:
            await self._execute_geographic_deployment(deployment, deployment_order)
        
        return deployment.id
    
    async def monitor_edge_health(self) -> Dict[str, Any]:
        """Comprehensive edge infrastructure health monitoring"""
        
        current_time = datetime.now()
        health_data = {
            "timestamp": current_time.isoformat(),
            "total_devices": len(self.devices),
            "total_clusters": len(self.clusters),
            "active_deployments": len([d for d in self.deployments.values() if d.status == "running"]),
            "device_health": {},
            "cluster_health": {},
            "connectivity_status": {},
            "performance_metrics": {}
        }
        
        # Analyze device health
        healthy_devices = 0
        for device_id, device in self.devices.items():
            last_heartbeat = self.device_heartbeats.get(device_id)
            if last_heartbeat:
                time_since_heartbeat = (current_time - last_heartbeat).total_seconds()
                if time_since_heartbeat < self.edge_settings["heartbeat_interval_seconds"] * 2:
                    device_status = "healthy"
                    healthy_devices += 1
                elif time_since_heartbeat < self.edge_settings["heartbeat_interval_seconds"] * 5:
                    device_status = "degraded"
                else:
                    device_status = "unhealthy"
            else:
                device_status = "unknown"
            
            health_data["device_health"][device_id] = {
                "status": device_status,
                "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
                "device_type": device.device_type.value,
                "location": device.location
            }
        
        # Calculate cluster health
        for cluster_id, cluster in self.clusters.items():
            cluster_healthy_devices = 0
            for device_id in cluster.devices:
                if health_data["device_health"].get(device_id, {}).get("status") == "healthy":
                    cluster_healthy_devices += 1
            
            cluster_health_percentage = (cluster_healthy_devices / len(cluster.devices) * 100) if cluster.devices else 0
            
            health_data["cluster_health"][cluster_id] = {
                "healthy_devices": cluster_healthy_devices,
                "total_devices": len(cluster.devices),
                "health_percentage": cluster_health_percentage,
                "status": "healthy" if cluster_health_percentage >= 80 else "degraded" if cluster_health_percentage >= 50 else "critical"
            }
        
        # Overall health metrics
        overall_health_percentage = (healthy_devices / len(self.devices) * 100) if self.devices else 100
        
        health_data["performance_metrics"] = {
            "overall_health_percentage": overall_health_percentage,
            "healthy_device_count": healthy_devices,
            "total_device_count": len(self.devices),
            "average_cluster_health": sum(c["health_percentage"] for c in health_data["cluster_health"].values()) / len(self.clusters) if self.clusters else 100
        }
        
        return health_data
    
    async def get_edge_analytics(self) -> Dict[str, Any]:
        """Advanced edge computing analytics and insights"""
        
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "device_analytics": {},
            "deployment_analytics": {},
            "geographic_analytics": {},
            "performance_analytics": {},
            "predictive_insights": {}
        }
        
        # Device type distribution
        device_types = {}
        compute_capabilities = {}
        connectivity_types = {}
        
        for device in self.devices.values():
            # Device type analysis
            device_type = device.device_type.value
            device_types[device_type] = device_types.get(device_type, 0) + 1
            
            # Compute capability analysis
            capability = device.compute_capability.value
            compute_capabilities[capability] = compute_capabilities.get(capability, 0) + 1
            
            # Connectivity analysis
            for conn in device.connectivity:
                conn_type = conn.value
                connectivity_types[conn_type] = connectivity_types.get(conn_type, 0) + 1
        
        analytics["device_analytics"] = {
            "device_type_distribution": device_types,
            "compute_capability_distribution": compute_capabilities,
            "connectivity_distribution": connectivity_types,
            "total_devices": len(self.devices)
        }
        
        # Deployment success rates
        total_deployments = len(self.deployments)
        successful_deployments = len([d for d in self.deployments.values() if d.status == "completed"])
        failed_deployments = len([d for d in self.deployments.values() if d.status == "failed"])
        
        analytics["deployment_analytics"] = {
            "total_deployments": total_deployments,
            "successful_deployments": successful_deployments,
            "failed_deployments": failed_deployments,
            "success_rate": successful_deployments / total_deployments if total_deployments > 0 else 1.0,
            "average_deployment_time": await self._calculate_average_deployment_time()
        }
        
        # Geographic distribution
        geographic_regions = {}
        for device in self.devices.values():
            # Simple geographic clustering based on timezone
            timezone = device.location.get("timezone", "unknown")
            geographic_regions[timezone] = geographic_regions.get(timezone, 0) + 1
        
        analytics["geographic_analytics"] = {
            "regional_distribution": geographic_regions,
            "total_clusters": len(self.clusters),
            "average_devices_per_cluster": len(self.devices) / len(self.clusters) if self.clusters else 0
        }
        
        return analytics
    
    async def _assess_device_capabilities(self, device: EdgeDevice) -> float:
        """Assess device capabilities for optimization"""
        
        score = 0.0
        
        # Compute capability scoring
        capability_scores = {
            EdgeComputeCapability.MICRO: 10,
            EdgeComputeCapability.NANO: 20,
            EdgeComputeCapability.SMALL: 40,
            EdgeComputeCapability.MEDIUM: 60,
            EdgeComputeCapability.LARGE: 80,
            EdgeComputeCapability.GPU_ACCELERATED: 90,
            EdgeComputeCapability.FPGA_ACCELERATED: 95
        }
        
        score += capability_scores.get(device.compute_capability, 0)
        
        # Connectivity scoring
        connectivity_scores = {
            EdgeConnectivity.FIBER: 20,
            EdgeConnectivity.CELLULAR_5G: 18,
            EdgeConnectivity.CELLULAR_4G: 15,
            EdgeConnectivity.WIFI: 12,
            EdgeConnectivity.SATELLITE: 10,
            EdgeConnectivity.BLUETOOTH: 5,
            EdgeConnectivity.ZIGBEE: 5,
            EdgeConnectivity.LORA: 8,
            EdgeConnectivity.NFC: 3
        }
        
        connectivity_score = max([connectivity_scores.get(conn, 0) for conn in device.connectivity])
        score += connectivity_score
        
        return min(score, 100.0)
    
    async def _optimize_edge_configuration(self, config: EdgeConfiguration) -> Dict[str, Any]:
        """Optimize configuration for edge deployment"""
        
        optimized_spec = config.configuration_spec.copy()
        
        # Analyze target devices for optimization
        target_capabilities = []
        for device_id in config.target_devices:
            if device_id in self.devices:
                device = self.devices[device_id]
                target_capabilities.append({
                    "compute": device.compute_capability,
                    "connectivity": device.connectivity,
                    "hardware": device.hardware_specs
                })
        
        # Optimize based on minimum common capabilities
        if target_capabilities:
            # Resource optimization
            if "resources" in optimized_spec:
                await self._optimize_resource_requirements(optimized_spec["resources"], target_capabilities)
            
            # Network optimization
            if "networking" in optimized_spec:
                await self._optimize_network_configuration(optimized_spec["networking"], target_capabilities)
            
            # Storage optimization
            if "storage" in optimized_spec:
                await self._optimize_storage_configuration(optimized_spec["storage"], target_capabilities)
        
        return optimized_spec
    
    async def _execute_rolling_deployment(
        self,
        deployment: EdgeDeploymentExecution,
        deployment_order: List[str]
    ):
        """Execute rolling deployment strategy"""
        
        batch_size = min(self.edge_settings["batch_deployment_size"], len(deployment_order))
        
        for i in range(0, len(deployment_order), batch_size):
            batch_devices = deployment_order[i:i + batch_size]
            
            # Deploy to batch
            batch_results = await self._deploy_to_device_batch(
                deployment.configuration_id,
                batch_devices
            )
            
            # Update deployment status
            for device_id, success in batch_results.items():
                if success:
                    deployment.successful_devices.append(device_id)
                else:
                    deployment.failed_devices.append(device_id)
            
            # Update progress
            deployment.progress_percentage = (len(deployment.successful_devices) + len(deployment.failed_devices)) / len(deployment_order) * 100
            
            # Health check between batches
            if i + batch_size < len(deployment_order):
                await self._perform_deployment_health_check(deployment)
                await asyncio.sleep(5)  # Brief pause between batches
        
        # Finalize deployment
        if len(deployment.failed_devices) == 0:
            deployment.status = "completed"
        elif len(deployment.successful_devices) > len(deployment.failed_devices):
            deployment.status = "partially_completed"
        else:
            deployment.status = "failed"
        
        deployment.completed_at = datetime.now()
    
    async def _deploy_to_device_batch(
        self,
        configuration_id: str,
        device_ids: List[str]
    ) -> Dict[str, bool]:
        """Deploy configuration to batch of devices"""
        
        results = {}
        
        # Simulate deployment to each device
        for device_id in device_ids:
            try:
                # Simulate device-specific deployment
                await asyncio.sleep(0.1)  # Simulate deployment time
                
                # Check device health and connectivity
                if device_id in self.devices:
                    device = self.devices[device_id]
                    last_heartbeat = self.device_heartbeats.get(device_id)
                    
                    if last_heartbeat and (datetime.now() - last_heartbeat).total_seconds() < 120:
                        # Device is healthy and reachable
                        device.current_config_version = configuration_id
                        device.configuration_state = "configured"
                        results[device_id] = True
                    else:
                        # Device unreachable
                        results[device_id] = False
                else:
                    results[device_id] = False
                    
            except Exception:
                results[device_id] = False
        
        return results
    
    async def _calculate_average_deployment_time(self) -> float:
        """Calculate average deployment time"""
        
        completed_deployments = [
            d for d in self.deployments.values()
            if d.status == "completed" and d.started_at and d.completed_at
        ]
        
        if not completed_deployments:
            return 0.0
        
        total_time = sum([
            (d.completed_at - d.started_at).total_seconds()
            for d in completed_deployments
        ])
        
        return total_time / len(completed_deployments)


async def get_edge_computing_manager(tenant_id: str) -> EdgeComputingManager:
    """Get edge computing manager instance"""
    
    manager = EdgeComputingManager(tenant_id)
    
    # Initialize with sample edge infrastructure for testing
    await manager._initialize_sample_edge_infrastructure()
    
    return manager


# EdgeComputingManager method implementations
async def _initialize_sample_edge_infrastructure(self):
    """Initialize sample edge infrastructure"""
    
    # Sample edge devices
    sample_devices = [
        {
            "name": "smart-factory-sensor-01",
            "device_type": EdgeDeviceType.MANUFACTURING,
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "timezone": "America/New_York",
                "address": "New York Manufacturing Plant"
            },
            "hardware_specs": {
                "cpu_cores": 2,
                "memory_gb": 4,
                "storage_gb": 32,
                "sensors": ["temperature", "pressure", "vibration"]
            },
            "connectivity": [EdgeConnectivity.WIFI, EdgeConnectivity.CELLULAR_4G],
            "compute_capability": EdgeComputeCapability.SMALL
        },
        {
            "name": "retail-pos-terminal-01",
            "device_type": EdgeDeviceType.RETAIL_EDGE,
            "location": {
                "latitude": 34.0522,
                "longitude": -118.2437,
                "timezone": "America/Los_Angeles",
                "address": "Los Angeles Retail Store"
            },
            "hardware_specs": {
                "cpu_cores": 4,
                "memory_gb": 8,
                "storage_gb": 128,
                "display": "touchscreen"
            },
            "connectivity": [EdgeConnectivity.FIBER, EdgeConnectivity.WIFI],
            "compute_capability": EdgeComputeCapability.MEDIUM
        }
    ]
    
    for device_config in sample_devices:
        await self.register_edge_device(device_config)


# Attach the method to the class
EdgeComputingManager._initialize_sample_edge_infrastructure = _initialize_sample_edge_infrastructure


# Additional helper methods for EdgeComputingManager
async def _discover_device_clusters(self, device_id: str):
    """Auto-discover potential device clusters"""
    device = self.devices[device_id]
    nearby_devices = []
    for candidate_id, candidate in self.devices.items():
        if candidate_id == device_id:
            continue
        same_timezone = candidate.location.get("timezone") == device.location.get("timezone")
        same_type = candidate.device_type == device.device_type
        if same_timezone or same_type:
            nearby_devices.append(candidate_id)
            discovery = candidate.metadata.setdefault("cluster_discovery", {
                "nearby_devices": [],
                "candidate_count": 0,
                "discovered_at": datetime.now().isoformat()
            })
            if device_id not in discovery["nearby_devices"]:
                discovery["nearby_devices"].append(device_id)
                discovery["candidate_count"] = len(discovery["nearby_devices"])
                discovery["discovered_at"] = datetime.now().isoformat()
    device.metadata["cluster_discovery"] = {
        "nearby_devices": nearby_devices,
        "candidate_count": len(nearby_devices),
        "discovered_at": datetime.now().isoformat()
    }


async def _initialize_device_monitoring(self, device_id: str):
    """Initialize monitoring for edge device"""
    device = self.devices[device_id]
    device.health_status = "healthy"
    device.last_heartbeat = datetime.now()
    self.device_heartbeats[device_id] = device.last_heartbeat
    device.metadata["monitoring"] = {
        "enabled": True,
        "heartbeat_interval_seconds": self.edge_settings["heartbeat_interval_seconds"],
        "health_check_interval_seconds": self.edge_settings["health_check_interval_seconds"],
        "initialized_at": datetime.now().isoformat()
    }


async def _optimize_cluster_geography(self, cluster: EdgeCluster):
    """Optimize cluster for geographic proximity"""
    devices = [self.devices[device_id] for device_id in cluster.devices if device_id in self.devices]
    if not devices:
        cluster.metadata["geographic_optimization"] = {"status": "skipped", "reason": "no devices"}
        return
    average_latitude = sum(float(device.location["latitude"]) for device in devices) / len(devices)
    average_longitude = sum(float(device.location["longitude"]) for device in devices) / len(devices)
    timezones = sorted({device.location.get("timezone", "unknown") for device in devices})
    cluster.geographic_region.update({
        "centroid": {
            "latitude": round(average_latitude, 6),
            "longitude": round(average_longitude, 6)
        },
        "timezones": timezones,
        "device_count": len(devices)
    })
    cluster.metadata["geographic_optimization"] = {
        "status": "optimized",
        "optimized_at": datetime.now().isoformat()
    }


async def _configure_cluster_networking(self, cluster: EdgeCluster):
    """Configure cluster networking and load balancing"""
    connectivity_counts: Dict[str, int] = {}
    for device_id in cluster.devices:
        device = self.devices.get(device_id)
        if not device:
            continue
        for connectivity in device.connectivity:
            connectivity_counts[connectivity.value] = connectivity_counts.get(connectivity.value, 0) + 1
    preferred_connectivity = max(connectivity_counts, key=connectivity_counts.get) if connectivity_counts else "unknown"
    cluster.failover_configuration.update({
        "enabled": True,
        "preferred_connectivity": preferred_connectivity,
        "connectivity_counts": connectivity_counts,
        "minimum_healthy_devices": max(1, len(cluster.devices) // 2)
    })
    if len(cluster.devices) > 10:
        cluster.load_balancing_strategy = "least_connections"


async def _initialize_cluster_monitoring(self, cluster: EdgeCluster):
    """Initialize cluster health monitoring"""
    healthy_devices = 0
    now = datetime.now()
    for device_id in cluster.devices:
        last_heartbeat = self.device_heartbeats.get(device_id)
        if last_heartbeat and (now - last_heartbeat).total_seconds() <= self.edge_settings["heartbeat_interval_seconds"] * 2:
            healthy_devices += 1
    cluster.health_score = (healthy_devices / len(cluster.devices) * 100) if cluster.devices else 0.0
    cluster.metadata["monitoring"] = {
        "enabled": True,
        "healthy_devices": healthy_devices,
        "total_devices": len(cluster.devices),
        "initialized_at": now.isoformat()
    }


async def _validate_edge_targets(self, config: EdgeConfiguration):
    """Validate edge configuration targets"""
    missing_devices = [device_id for device_id in config.target_devices if device_id not in self.devices]
    missing_clusters = [cluster_id for cluster_id in config.target_clusters if cluster_id not in self.clusters]
    if missing_devices:
        raise ValueError(f"Unknown edge target devices: {', '.join(missing_devices)}")
    if missing_clusters:
        raise ValueError(f"Unknown edge target clusters: {', '.join(missing_clusters)}")
    expanded_targets = set(config.target_devices)
    for cluster_id in config.target_clusters:
        expanded_targets.update(self.clusters[cluster_id].devices)
    if not expanded_targets:
        raise ValueError("Edge configuration must target at least one device")
    config.configuration_spec.setdefault("target_summary", {
        "device_count": len(expanded_targets),
        "cluster_count": len(config.target_clusters)
    })


async def _optimize_configuration_bandwidth(self, config: EdgeConfiguration):
    """Optimize configuration for bandwidth constraints"""
    networking = config.configuration_spec.setdefault("networking", {})
    networking.setdefault("compression", "gzip")
    networking.setdefault("delta_sync", True)
    networking.setdefault("max_payload_kb", 512)
    monitoring = config.monitoring_configuration
    monitoring.setdefault("sampling_interval_seconds", 60)
    monitoring.setdefault("batch_metrics", True)


async def _generate_resource_constraints(self, config: EdgeConfiguration) -> Dict[str, Any]:
    """Generate resource constraints automatically"""
    return {
        "max_memory_mb": 1024,
        "max_cpu_percent": 80,
        "max_storage_mb": 100,
        "max_network_bandwidth_kbps": 1000
    }


async def _generate_edge_security_policies(self, config: EdgeConfiguration) -> Dict[str, Any]:
    """Generate edge-specific security policies"""
    return {
        "encryption_required": True,
        "certificate_validation": True,
        "secure_communication": True,
        "device_authentication": True
    }


async def _expand_cluster_targets(self, deployment: EdgeDeploymentExecution):
    """Expand cluster targets to individual devices"""
    ordered_targets = list(deployment.target_devices)
    seen = set(ordered_targets)
    for cluster_id in deployment.target_clusters:
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            raise ValueError(f"Cluster {cluster_id} not found")
        for device_id in cluster.devices:
            if device_id not in seen:
                ordered_targets.append(device_id)
                seen.add(device_id)
    deployment.target_devices = ordered_targets


async def _optimize_deployment_order(self, deployment: EdgeDeploymentExecution) -> List[str]:
    """Optimize deployment order based on geography and connectivity"""
    return deployment.target_devices  # Simple implementation


async def _execute_blue_green_deployment(self, deployment: EdgeDeploymentExecution, deployment_order: List[str]):
    """Execute blue-green deployment strategy"""
    deployment.execution_log.append({
        "phase": "blue_green_prepare",
        "message": "Prepared green configuration slot",
        "timestamp": datetime.now().isoformat()
    })
    batch_results = await self._deploy_to_device_batch(deployment.configuration_id, deployment_order)
    for device_id, success in batch_results.items():
        if success:
            deployment.successful_devices.append(device_id)
        else:
            deployment.failed_devices.append(device_id)
    await self._perform_deployment_health_check(deployment)
    deployment.status = "completed" if not deployment.failed_devices else "failed"
    deployment.progress_percentage = 100.0
    deployment.completed_at = datetime.now()


async def _execute_canary_deployment(self, deployment: EdgeDeploymentExecution, deployment_order: List[str]):
    """Execute canary deployment strategy"""
    canary_size = max(1, len(deployment_order) // 10)
    canary_devices = deployment_order[:canary_size]
    remaining_devices = deployment_order[canary_size:]
    canary_results = await self._deploy_to_device_batch(deployment.configuration_id, canary_devices)
    for device_id, success in canary_results.items():
        if success:
            deployment.successful_devices.append(device_id)
        else:
            deployment.failed_devices.append(device_id)
    await self._perform_deployment_health_check(deployment)
    if deployment.failed_devices:
        deployment.status = "failed"
        deployment.rollback_triggered = True
        deployment.completed_at = datetime.now()
        deployment.progress_percentage = len(deployment.successful_devices) / len(deployment_order) * 100
        return
    remaining_results = await self._deploy_to_device_batch(deployment.configuration_id, remaining_devices)
    for device_id, success in remaining_results.items():
        if success:
            deployment.successful_devices.append(device_id)
        else:
            deployment.failed_devices.append(device_id)
    deployment.status = "completed" if not deployment.failed_devices else "partially_completed"
    deployment.progress_percentage = 100.0
    deployment.completed_at = datetime.now()


async def _execute_geographic_deployment(self, deployment: EdgeDeploymentExecution, deployment_order: List[str]):
    """Execute geographic rollout deployment"""
    by_timezone: Dict[str, List[str]] = {}
    for device_id in deployment_order:
        device = self.devices.get(device_id)
        timezone = device.location.get("timezone", "unknown") if device else "unknown"
        by_timezone.setdefault(timezone, []).append(device_id)
    for timezone in sorted(by_timezone):
        deployment.execution_log.append({
            "phase": "geographic_rollout",
            "timezone": timezone,
            "device_count": len(by_timezone[timezone]),
            "timestamp": datetime.now().isoformat()
        })
        results = await self._deploy_to_device_batch(deployment.configuration_id, by_timezone[timezone])
        for device_id, success in results.items():
            if success:
                deployment.successful_devices.append(device_id)
            else:
                deployment.failed_devices.append(device_id)
        await self._perform_deployment_health_check(deployment)
    deployment.status = "completed" if not deployment.failed_devices else "partially_completed"
    deployment.progress_percentage = 100.0
    deployment.completed_at = datetime.now()


async def _perform_deployment_health_check(self, deployment: EdgeDeploymentExecution):
    """Perform health check during deployment"""
    checked_devices = deployment.successful_devices + deployment.failed_devices
    healthy_devices = 0
    now = datetime.now()
    for device_id in checked_devices:
        last_heartbeat = self.device_heartbeats.get(device_id)
        if last_heartbeat and (now - last_heartbeat).total_seconds() <= 120:
            healthy_devices += 1
    total_checked = len(checked_devices)
    success_rate = healthy_devices / total_checked if total_checked else 1.0
    deployment.health_checks.append({
        "healthy": success_rate >= 0.8,
        "healthy_devices": healthy_devices,
        "checked_devices": total_checked,
        "success_rate": success_rate,
        "timestamp": now.isoformat()
    })
    if total_checked and success_rate < 0.5:
        deployment.rollback_triggered = True


async def _optimize_resource_requirements(self, resources: Dict[str, Any], capabilities: List[Dict[str, Any]]):
    """Optimize resource requirements for edge devices"""
    if not capabilities:
        return
    min_cpu = min(int(item["hardware"].get("cpu_cores", 1)) for item in capabilities)
    min_memory_mb = min(int(float(item["hardware"].get("memory_gb", 1)) * 1024) for item in capabilities)
    min_storage_mb = min(int(float(item["hardware"].get("storage_gb", 1)) * 1024) for item in capabilities)
    resources["cpu_cores"] = min(int(resources.get("cpu_cores", min_cpu)), min_cpu)
    resources["memory_mb"] = min(int(resources.get("memory_mb", min_memory_mb)), min_memory_mb)
    resources["storage_mb"] = min(int(resources.get("storage_mb", min_storage_mb)), min_storage_mb)
    resources["optimized_for_edge"] = True


async def _optimize_network_configuration(self, networking: Dict[str, Any], capabilities: List[Dict[str, Any]]):
    """Optimize network configuration for edge devices"""
    connectivity_counts: Dict[str, int] = {}
    for capability in capabilities:
        for connectivity in capability["connectivity"]:
            connectivity_counts[connectivity.value] = connectivity_counts.get(connectivity.value, 0) + 1
    preferred = max(connectivity_counts, key=connectivity_counts.get) if connectivity_counts else "unknown"
    networking["preferred_connectivity"] = preferred
    networking["retry_policy"] = {
        "max_retries": 3,
        "backoff_seconds": 5
    }
    networking["offline_tolerant"] = preferred in {EdgeConnectivity.SATELLITE.value, EdgeConnectivity.LORA.value, EdgeConnectivity.CELLULAR_4G.value}


async def _optimize_storage_configuration(self, storage: Dict[str, Any], capabilities: List[Dict[str, Any]]):
    """Optimize storage configuration for edge devices"""
    if not capabilities:
        return
    min_storage_gb = min(float(item["hardware"].get("storage_gb", 1)) for item in capabilities)
    requested_cache_mb = int(storage.get("cache_mb", min_storage_gb * 1024 * 0.1))
    storage["cache_mb"] = min(requested_cache_mb, int(min_storage_gb * 1024 * 0.2))
    storage["retention_policy"] = storage.get("retention_policy", "size_bounded")
    storage["local_persistence"] = storage.get("local_persistence", True)


# Attach helper methods to the class
for method_name, method in list(locals().items()):
    if method_name.startswith('_') and callable(method) and method_name != '__init__':
        setattr(EdgeComputingManager, method_name, method)


__all__ = [
    'EdgeDeviceType',
    'EdgeConnectivity', 
    'EdgeComputeCapability',
    'EdgeDeploymentStrategy',
    'EdgeDevice',
    'EdgeCluster',
    'EdgeConfiguration',
    'EdgeDeploymentExecution',
    'EdgeComputingManager',
    'get_edge_computing_manager'
]
