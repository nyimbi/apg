#!/usr/bin/env python3
"""
Create IoT and Sensor Management Capabilities
==============================================

Create comprehensive IoT capabilities for the APG composable template system.
"""

import json
from pathlib import Path
from templates.composable.capability import Capability, CapabilityCategory, CapabilityDependency, CapabilityIntegration

def create_mqtt_capability():
    """Create MQTT messaging capability"""
    return Capability(
        name="MQTT Messaging",
        category=CapabilityCategory.IOT,
        description="MQTT pub/sub messaging for IoT device communication",
        version="1.0.0",
        python_requirements=[
            "paho-mqtt>=1.6.1",
            "json-logging>=1.3.0"
        ],
        features=[
            "MQTT Broker Connection",
            "Topic Subscription Management", 
            "Message Publishing",
            "Device Status Monitoring",
            "Retained Messages",
            "QoS Level Support"
        ],
        compatible_bases=["flask_webapp", "microservice", "real_time", "dashboard"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store device data and message history")
        ],
        integration=CapabilityIntegration(
            models=["MQTTDevice", "MQTTTopic", "MQTTMessage"],
            views=["DeviceView", "TopicView", "MessageView"],
            apis=["mqtt/publish", "mqtt/subscribe", "mqtt/devices"],
            config_additions={
                "MQTT_BROKER_HOST": "localhost",
                "MQTT_BROKER_PORT": 1883,
                "MQTT_KEEPALIVE": 60
            }
        )
    )

def create_device_management_capability():
    """Create IoT device management capability"""
    return Capability(
        name="Device Management",
        category=CapabilityCategory.IOT,
        description="Comprehensive IoT device lifecycle management",
        version="1.0.0",
        python_requirements=[
            "requests>=2.31.0",
            "cryptography>=41.0.0",
            "schedule>=1.2.0"
        ],
        features=[
            "Device Registration",
            "Device Provisioning", 
            "Firmware Updates",
            "Configuration Management",
            "Device Health Monitoring",
            "Remote Control",
            "Device Groups"
        ],
        compatible_bases=["flask_webapp", "dashboard", "microservice"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store device information and configurations"),
            CapabilityDependency("auth/basic_authentication", reason="Secure device management access")
        ],
        integration=CapabilityIntegration(
            models=["IoTDevice", "DeviceType", "DeviceGroup", "FirmwareVersion", "DeviceConfig"],
            views=["DeviceManagementView", "DeviceGroupView", "FirmwareView"],
            apis=["devices/register", "devices/provision", "devices/update", "devices/control"],
            templates=["device_dashboard.html", "device_detail.html"]
        )
    )

def create_sensor_data_capability():
    """Create sensor data processing capability"""
    return Capability(
        name="Sensor Data Processing",
        category=CapabilityCategory.IOT,
        description="Real-time sensor data collection, processing, and analytics",
        version="1.0.0",
        python_requirements=[
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "plotly>=5.15.0"
        ],
        features=[
            "Real-time Data Ingestion",
            "Data Validation",
            "Anomaly Detection",
            "Data Aggregation",
            "Threshold Alerts",
            "Historical Analysis",
            "Data Visualization"
        ],
        compatible_bases=["flask_webapp", "dashboard", "real_time"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store sensor readings and metadata"),
            CapabilityDependency("analytics/basic_analytics", reason="Data visualization and reporting")
        ],
        integration=CapabilityIntegration(
            models=["Sensor", "SensorReading", "SensorType", "Alert", "DataStream"],
            views=["SensorView", "ReadingsView", "AlertsView"],
            apis=["sensors/data", "sensors/alerts", "sensors/analytics"],
            templates=["sensor_dashboard.html", "sensor_charts.html"]
        )
    )

def create_digital_twin_capability():
    """Create digital twin capability"""
    return Capability(
        name="Digital Twin",
        category=CapabilityCategory.IOT,
        description="Digital twin modeling and simulation for IoT systems",
        version="1.0.0",
        python_requirements=[
            "networkx>=3.1",
            "matplotlib>=3.7.0",
            "simpy>=4.0.1",
            "pydantic>=2.0.0"
        ],
        features=[
            "Twin Model Creation",
            "Real-time Synchronization",
            "Predictive Simulation",
            "State Management",
            "Behavior Modeling",
            "Performance Optimization",
            "3D Visualization"
        ],
        compatible_bases=["flask_webapp", "dashboard"],
        dependencies=[
            CapabilityDependency("iot/sensor_data_processing", reason="Real-time data for twin synchronization"),
            CapabilityDependency("data/postgresql_database", reason="Store twin models and state")
        ],
        integration=CapabilityIntegration(
            models=["DigitalTwin", "TwinModel", "TwinState", "Simulation"],
            views=["TwinView", "SimulationView"],
            apis=["twins/create", "twins/simulate", "twins/sync"],
            templates=["twin_dashboard.html", "twin_3d.html"]
        )
    )

def create_edge_computing_capability():
    """Create edge computing capability"""
    return Capability(
        name="Edge Computing",
        category=CapabilityCategory.IOT,
        description="Edge computing orchestration and container management",
        version="1.0.0",
        python_requirements=[
            "docker>=6.1.0",
            "kubernetes>=27.2.0",
            "redis>=4.6.0"
        ],
        features=[
            "Edge Node Management",
            "Container Orchestration",
            "Edge Analytics",
            "Local Data Processing",
            "Offline Capability",
            "Edge-to-Cloud Sync",
            "Resource Monitoring"
        ],
        compatible_bases=["microservice", "api_only"],
        dependencies=[
            CapabilityDependency("data/redis", reason="Edge caching and message queuing", optional=True)
        ],
        integration=CapabilityIntegration(
            models=["EdgeNode", "EdgeContainer", "EdgeJob"],
            apis=["edge/deploy", "edge/monitor", "edge/sync"],
            config_additions={
                "EDGE_REGISTRY_URL": "localhost:5000",
                "KUBERNETES_CONFIG": "/etc/k8s/config"
            }
        )
    )

def create_industrial_protocols_capability():
    """Create industrial protocols capability"""
    return Capability(
        name="Industrial Protocols",
        category=CapabilityCategory.IOT,
        description="Support for industrial communication protocols (Modbus, OPC-UA, etc.)",
        version="1.0.0",
        python_requirements=[
            "pymodbus>=3.4.0",
            "opcua>=0.98.13",
            "can-isotp>=2.0.0",
            "pyserial>=3.5"
        ],
        features=[
            "Modbus TCP/RTU Support",
            "OPC-UA Client/Server",
            "CAN Bus Communication", 
            "Serial Communication",
            "Protocol Translation",
            "Data Mapping",
            "Real-time Polling"
        ],
        compatible_bases=["microservice", "real_time"],
        dependencies=[
            CapabilityDependency("iot/sensor_data_processing", reason="Process industrial sensor data")
        ],
        integration=CapabilityIntegration(
            models=["ProtocolConfig", "IndustrialDevice", "DataMap"],
            apis=["protocols/modbus", "protocols/opcua", "protocols/can"],
            config_additions={
                "MODBUS_PORT": 502,
                "OPCUA_ENDPOINT": "opc.tcp://localhost:4840"
            }
        )
    )

def save_iot_capabilities():
    """Save all IoT capabilities to the filesystem"""
    print("🔧 Creating IoT and Sensor Management Capabilities")
    print("=" * 60)
    
    # Create capabilities
    capabilities = [
        create_mqtt_capability(),
        create_device_management_capability(), 
        create_sensor_data_capability(),
        create_digital_twin_capability(),
        create_edge_computing_capability(),
        create_industrial_protocols_capability()
    ]
    
    # Save each capability
    capabilities_dir = Path(__file__).parent / 'templates' / 'composable' / 'capabilities' / 'iot'
    capabilities_dir.mkdir(parents=True, exist_ok=True)
    
    for capability in capabilities:
        # Create capability directory
        cap_name = capability.name.lower().replace(' ', '_')
        cap_dir = capabilities_dir / cap_name
        cap_dir.mkdir(exist_ok=True)
        
        # Create standard directories
        for subdir in ['models', 'views', 'templates', 'static', 'tests', 'config', 'scripts']:
            (cap_dir / subdir).mkdir(exist_ok=True)
        
        # Save capability.json
        with open(cap_dir / 'capability.json', 'w') as f:
            json.dump(capability.to_dict(), f, indent=2)
        
        # Create integration template
        create_integration_template(cap_dir, capability)
        
        print(f"  ✅ Created {capability.name}")
    
    print(f"\n📁 IoT capabilities saved to: {capabilities_dir}")
    return capabilities

def create_integration_template(cap_dir: Path, capability: Capability):
    """Create integration template for IoT capability"""
    cap_name_snake = capability.name.lower().replace(' ', '_')
    cap_name_class = capability.name.replace(' ', '')
    
    integration_content = f'''"""
{capability.name} Integration
{'=' * (len(capability.name) + 12)}

Integration logic for the {capability.name} capability.
Handles IoT-specific setup and configuration.
"""

import logging
from flask import Blueprint
from flask_appbuilder import BaseView

# Configure logging
log = logging.getLogger(__name__)

# Create capability blueprint
{cap_name_snake}_bp = Blueprint(
    '{cap_name_snake}',
    __name__,
    url_prefix='/iot/{cap_name_snake}',
    template_folder='templates',
    static_folder='static'
)


def integrate_{cap_name_snake}(app, appbuilder, db):
    """
    Integrate {capability.name} capability into the application.
    
    Args:
        app: Flask application instance
        appbuilder: Flask-AppBuilder instance
        db: SQLAlchemy database instance
    """
    try:
        # Register blueprint
        app.register_blueprint({cap_name_snake}_bp)
        
        # Import and register models
        from .models import *  # noqa
        
        # Import and register views
        from .views import *  # noqa
        
        # Apply IoT-specific configuration
        config_additions = {repr(capability.integration.config_additions)}
        for key, value in config_additions.items():
            app.config[key] = value
        
        # Initialize IoT services
        iot_service = {cap_name_class}Service(app, appbuilder, db)
        app.extensions['{cap_name_snake}_service'] = iot_service
        
        log.info(f"Successfully integrated {capability.name} capability")
        
    except Exception as e:
        log.error(f"Failed to integrate {capability.name} capability: {{e}}")
        raise


class {cap_name_class}Service:
    """
    Main service class for {capability.name}.
    
    Handles IoT-specific functionality and business logic.
    """
    
    def __init__(self, app, appbuilder, db):
        self.app = app
        self.appbuilder = appbuilder
        self.db = db
        self.initialize_service()
    
    def initialize_service(self):
        """Initialize IoT service components"""
        log.info(f"Initializing {capability.name} service")
        
        # IoT-specific initialization logic here
        # For example: setup MQTT connections, device discovery, etc.
        pass
    
    def start_monitoring(self):
        """Start IoT monitoring and data collection"""
        pass
    
    def stop_monitoring(self):
        """Stop IoT monitoring"""
        pass
'''
    
    # Save integration template
    with open(cap_dir / 'integration.py.template', 'w') as f:
        f.write(integration_content)
    
    # Create models template for IoT
    models_content = f'''"""
{capability.name} Models
{'=' * (len(capability.name) + 7)}

Database models for {capability.name} capability.
"""

from flask_appbuilder import Model
from flask_appbuilder.models.mixins import AuditMixin
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime


class IoTBaseModel(AuditMixin, Model):
    """Base model for IoT entities"""
    __abstract__ = True
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    active = Column(Boolean, default=True)


# Add IoT-specific models based on capability
{generate_iot_models(capability)}
'''
    
    with open(cap_dir / 'models' / '__init__.py.template', 'w') as f:
        f.write(models_content)

def generate_iot_models(capability: Capability) -> str:
    """Generate IoT-specific models based on capability type"""
    if "MQTT" in capability.name:
        return '''
class MQTTDevice(IoTBaseModel):
    """MQTT IoT device model"""
    __tablename__ = 'mqtt_devices'
    
    id = Column(Integer, primary_key=True)
    device_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    device_type = Column(String(64))
    last_seen = Column(DateTime)
    status = Column(String(32), default='offline')
    
    # MQTT specific fields
    client_id = Column(String(128))
    topics = relationship("MQTTTopic", back_populates="device")


class MQTTTopic(IoTBaseModel):
    """MQTT topic subscription model"""
    __tablename__ = 'mqtt_topics'
    
    id = Column(Integer, primary_key=True)
    topic_name = Column(String(256), nullable=False)
    qos_level = Column(Integer, default=0)
    device_id = Column(Integer, ForeignKey('mqtt_devices.id'))
    
    device = relationship("MQTTDevice", back_populates="topics")


class MQTTMessage(IoTBaseModel):
    """MQTT message history"""
    __tablename__ = 'mqtt_messages'
    
    id = Column(Integer, primary_key=True)
    topic = Column(String(256), nullable=False)
    payload = Column(Text)
    qos = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
'''
    elif "Sensor" in capability.name:
        return '''
class Sensor(IoTBaseModel):
    """IoT sensor model"""
    __tablename__ = 'sensors'
    
    id = Column(Integer, primary_key=True)
    sensor_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    sensor_type = Column(String(64))
    location = Column(String(256))
    unit = Column(String(32))
    min_value = Column(Float)
    max_value = Column(Float)
    
    readings = relationship("SensorReading", back_populates="sensor")


class SensorReading(IoTBaseModel):
    """Sensor data reading"""
    __tablename__ = 'sensor_readings'
    
    id = Column(Integer, primary_key=True)
    sensor_id = Column(Integer, ForeignKey('sensors.id'))
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    quality = Column(String(32), default='good')
    
    sensor = relationship("Sensor", back_populates="readings")


class Alert(IoTBaseModel):
    """Sensor alert model"""
    __tablename__ = 'sensor_alerts'
    
    id = Column(Integer, primary_key=True)
    sensor_id = Column(Integer, ForeignKey('sensors.id'))
    alert_type = Column(String(64))
    message = Column(Text)
    severity = Column(String(32))
    acknowledged = Column(Boolean, default=False)
'''
    elif "Digital Twin" in capability.name:
        return '''
class DigitalTwin(IoTBaseModel):
    """Digital twin model"""
    __tablename__ = 'digital_twins'
    
    id = Column(Integer, primary_key=True)
    twin_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text)
    model_version = Column(String(32))
    last_sync = Column(DateTime)
    
    states = relationship("TwinState", back_populates="twin")


class TwinState(IoTBaseModel):
    """Digital twin state snapshot"""
    __tablename__ = 'twin_states'
    
    id = Column(Integer, primary_key=True)
    twin_id = Column(Integer, ForeignKey('digital_twins.id'))
    state_data = Column(Text)  # JSON data
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    twin = relationship("DigitalTwin", back_populates="states")
'''
    else:
        return '''
# Generic IoT device model
class GenericIoTDevice(IoTBaseModel):
    """Generic IoT device model"""
    __tablename__ = 'iot_devices'
    
    id = Column(Integer, primary_key=True)
    device_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    device_type = Column(String(64))
    status = Column(String(32), default='active')
    metadata = Column(Text)  # JSON metadata
'''

def main():
    """Create all IoT capabilities"""
    try:
        capabilities = save_iot_capabilities()
        
        print(f"\n🎉 Successfully created {len(capabilities)} IoT capabilities!")
        print(f"\n📋 IoT Capabilities Created:")
        for cap in capabilities:
            print(f"   • {cap.name} - {cap.description}")
        
        print(f"\n🚀 These capabilities enable:")
        print(f"   • Industrial IoT device management")
        print(f"   • Real-time sensor data processing")
        print(f"   • MQTT messaging and communication")
        print(f"   • Digital twin modeling and simulation")
        print(f"   • Edge computing orchestration")
        print(f"   • Industrial protocol support")
        
        return True
        
    except Exception as e:
        print(f"💥 Error creating IoT capabilities: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)