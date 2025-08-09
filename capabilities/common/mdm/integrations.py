#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - APG Ecosystem Integration
Event streaming, caching, audit logging, and configuration management integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from uuid_extensions import uuid7str

import aiohttp
import asyncio
import redis.asyncio as redis
from pydantic import BaseModel, Field, ConfigDict

from .models import EntityType, EntityStatus, DataQualityStatus, MDMOperationType
from .database import MDMDatabaseManager


class APGCapability(str, Enum):
    """APG capabilities that MDM integrates with"""
    MESSAGE_QUEUE = "mqeb"          # Event streaming bus
    CACHING = "cach"                # Distributed caching
    AUDIT_LOGGING = "audl"          # Audit logging
    CONFIGURATION = "conf"          # Configuration management
    NOTIFICATION = "ntfy"           # Real-time notifications
    AUTH_RBAC = "auth"              # Authentication and RBAC
    MULTI_TENANT = "mten"           # Multi-tenant management
    ENCRYPTION = "encr"             # Encryption services


@dataclass
class MDMEvent:
    """MDM event for APG ecosystem propagation"""
    event_id: str
    event_type: str
    entity_id: Optional[str]
    entity_type: Optional[str]
    tenant_id: str
    user_id: str
    timestamp: datetime
    event_data: Dict[str, Any]
    source_service: str = "mdm"
    correlation_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'event_data': self.event_data,
            'source_service': self.source_service,
            'correlation_id': self.correlation_id,
            'priority': self.priority
        }


class EventPublisher:
    """APG Message Queue Event Bus integration for publishing MDM events"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mqeb_url = self.config.get('mqeb_url', 'http://localhost:8080')
        self.session = None
        self.event_queue = asyncio.Queue()
        self.publisher_task = None
        self.is_running = False
        
        # Event type routing configuration
        self.event_routing = {
            'entity.created': {'topic': 'mdm.entities', 'priority': 'normal'},
            'entity.updated': {'topic': 'mdm.entities', 'priority': 'normal'},
            'entity.deleted': {'topic': 'mdm.entities', 'priority': 'high'},
            'entity.merged': {'topic': 'mdm.entities', 'priority': 'high'},
            'quality.assessed': {'topic': 'mdm.quality', 'priority': 'low'},
            'quality.degraded': {'topic': 'mdm.quality', 'priority': 'high'},
            'duplicates.detected': {'topic': 'mdm.duplicates', 'priority': 'medium'},
            'golden_record.created': {'topic': 'mdm.golden_records', 'priority': 'high'},
            'golden_record.updated': {'topic': 'mdm.golden_records', 'priority': 'high'},
            'anomaly.detected': {'topic': 'mdm.anomalies', 'priority': 'high'}
        }
    
    async def start(self):
        """Start the event publisher"""
        if self.is_running:
            return
        
        self.session = aiohttp.ClientSession()
        self.is_running = True
        self.publisher_task = asyncio.create_task(self._publisher_worker())
        print("[MDM-Events] Event publisher started")
    
    async def stop(self):
        """Stop the event publisher"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.publisher_task:
            self.publisher_task.cancel()
            try:
                await self.publisher_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        print("[MDM-Events] Event publisher stopped")
    
    async def publish_event(self, event: MDMEvent) -> bool:
        """Publish event to APG message queue"""
        try:
            await self.event_queue.put(event)
            return True
        except Exception as e:
            print(f"[MDM-Events] Error queuing event: {str(e)}")
            return False
    
    async def _publisher_worker(self):
        """Background worker to publish events"""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Publish to APG message queue
                success = await self._publish_to_mqeb(event)
                
                if success:
                    print(f"[MDM-Events] Published event: {event.event_type} for entity {event.entity_id}")
                else:
                    print(f"[MDM-Events] Failed to publish event: {event.event_type}")
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
            except Exception as e:
                print(f"[MDM-Events] Publisher worker error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _publish_to_mqeb(self, event: MDMEvent) -> bool:
        """Publish event to APG Message Queue Event Bus"""
        try:
            if not self.session:
                return False
            
            # Get routing configuration for event type
            routing_config = self.event_routing.get(event.event_type, {
                'topic': 'mdm.general',
                'priority': 'normal'
            })
            
            # Prepare event payload for APG MQEB
            payload = {
                'topic': routing_config['topic'],
                'priority': routing_config['priority'],
                'event': event.to_dict(),
                'headers': {
                    'content-type': 'application/json',
                    'source': 'mdm',
                    'tenant-id': event.tenant_id,
                    'correlation-id': event.correlation_id or event.event_id
                }
            }
            
            # Publish to APG MQEB
            async with self.session.post(
                f"{self.mqeb_url}/api/v1/events/publish",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                
                if response.status == 200:
                    return True
                else:
                    error_text = await response.text()
                    print(f"[MDM-Events] MQEB publish error: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            print(f"[MDM-Events] Error publishing to MQEB: {str(e)}")
            return False


class CacheManager:
    """APG Distributed Caching integration for MDM data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.redis_url = self.config.get('redis_url', 'redis://localhost:6379')
        self.cache_prefix = self.config.get('cache_prefix', 'mdm')
        self.default_ttl = self.config.get('default_ttl', 3600)  # 1 hour
        self.redis_client = None
        
        # Cache configuration for different data types
        self.cache_config = {
            'entities': {'ttl': 1800, 'compress': False},      # 30 minutes
            'quality_scores': {'ttl': 600, 'compress': False}, # 10 minutes
            'duplicate_results': {'ttl': 3600, 'compress': True},  # 1 hour
            'golden_records': {'ttl': 7200, 'compress': False},    # 2 hours
            'search_results': {'ttl': 300, 'compress': True},      # 5 minutes
            'statistics': {'ttl': 1800, 'compress': True}          # 30 minutes
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            print(f"[MDM-Cache] Connected to Redis at {self.redis_url}")
        except Exception as e:
            print(f"[MDM-Cache] Failed to connect to Redis: {str(e)}")
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _get_cache_key(self, cache_type: str, tenant_id: str, key: str) -> str:
        """Generate cache key with proper namespacing"""
        return f"{self.cache_prefix}:{cache_type}:{tenant_id}:{key}"
    
    async def get_entity(self, tenant_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get cached entity data"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key('entities', tenant_id, entity_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
                
        except Exception as e:
            print(f"[MDM-Cache] Error getting entity from cache: {str(e)}")
        
        return None
    
    async def set_entity(self, tenant_id: str, entity_id: str, entity_data: Dict[str, Any]) -> bool:
        """Cache entity data"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._get_cache_key('entities', tenant_id, entity_id)
            config = self.cache_config['entities']
            
            await self.redis_client.setex(
                cache_key,
                config['ttl'],
                json.dumps(entity_data, default=str)
            )
            
            return True
            
        except Exception as e:
            print(f"[MDM-Cache] Error caching entity: {str(e)}")
            return False
    
    async def invalidate_entity(self, tenant_id: str, entity_id: str) -> bool:
        """Invalidate cached entity data"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._get_cache_key('entities', tenant_id, entity_id)
            await self.redis_client.delete(cache_key)
            
            # Also invalidate related caches
            related_patterns = [
                self._get_cache_key('quality_scores', tenant_id, entity_id),
                self._get_cache_key('duplicate_results', tenant_id, f"{entity_id}:*"),
                self._get_cache_key('search_results', tenant_id, '*')
            ]
            
            for pattern in related_patterns:
                if '*' in pattern:
                    # Pattern-based deletion
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    await self.redis_client.delete(pattern)
            
            return True
            
        except Exception as e:
            print(f"[MDM-Cache] Error invalidating entity cache: {str(e)}")
            return False
    
    async def get_quality_score(self, tenant_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get cached quality assessment"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key('quality_scores', tenant_id, entity_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
                
        except Exception as e:
            print(f"[MDM-Cache] Error getting quality score from cache: {str(e)}")
        
        return None
    
    async def set_quality_score(self, tenant_id: str, entity_id: str, 
                              quality_data: Dict[str, Any]) -> bool:
        """Cache quality assessment results"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._get_cache_key('quality_scores', tenant_id, entity_id)
            config = self.cache_config['quality_scores']
            
            await self.redis_client.setex(
                cache_key,
                config['ttl'],
                json.dumps(quality_data, default=str)
            )
            
            return True
            
        except Exception as e:
            print(f"[MDM-Cache] Error caching quality score: {str(e)}")
            return False
    
    async def get_duplicate_results(self, tenant_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get cached duplicate detection results"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key('duplicate_results', tenant_id, entity_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
                
        except Exception as e:
            print(f"[MDM-Cache] Error getting duplicate results from cache: {str(e)}")
        
        return None
    
    async def set_duplicate_results(self, tenant_id: str, entity_id: str,
                                  duplicate_data: Dict[str, Any]) -> bool:
        """Cache duplicate detection results"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._get_cache_key('duplicate_results', tenant_id, entity_id)
            config = self.cache_config['duplicate_results']
            
            # Compress large duplicate result sets
            data_str = json.dumps(duplicate_data, default=str)
            if config['compress'] and len(data_str) > 1000:
                import gzip
                data_str = gzip.compress(data_str.encode()).hex()
                cache_key += ':compressed'
            
            await self.redis_client.setex(cache_key, config['ttl'], data_str)
            return True
            
        except Exception as e:
            print(f"[MDM-Cache] Error caching duplicate results: {str(e)}")
            return False
    
    async def get_search_results(self, tenant_id: str, search_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached search results"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key('search_results', tenant_id, search_hash)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                # Handle compressed data
                if cache_key.endswith(':compressed'):
                    import gzip
                    cached_data = gzip.decompress(bytes.fromhex(cached_data)).decode()
                
                return json.loads(cached_data)
                
        except Exception as e:
            print(f"[MDM-Cache] Error getting search results from cache: {str(e)}")
        
        return None
    
    async def set_search_results(self, tenant_id: str, search_hash: str,
                               search_data: Dict[str, Any]) -> bool:
        """Cache search results"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._get_cache_key('search_results', tenant_id, search_hash)
            config = self.cache_config['search_results']
            
            # Compress search results
            data_str = json.dumps(search_data, default=str)
            if config['compress']:
                import gzip
                data_str = gzip.compress(data_str.encode()).hex()
                cache_key += ':compressed'
            
            await self.redis_client.setex(cache_key, config['ttl'], data_str)
            return True
            
        except Exception as e:
            print(f"[MDM-Cache] Error caching search results: {str(e)}")
            return False
    
    def generate_search_hash(self, search_criteria: Dict[str, Any]) -> str:
        """Generate hash for search criteria to use as cache key"""
        import hashlib
        
        # Sort criteria for consistent hashing
        sorted_criteria = json.dumps(search_criteria, sort_keys=True)
        return hashlib.md5(sorted_criteria.encode()).hexdigest()


class APGAuditLogger:
    """APG Audit Logging integration for comprehensive MDM audit trails"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.audl_url = self.config.get('audl_url', 'http://localhost:8081')
        self.session = None
        self.audit_queue = asyncio.Queue()
        self.audit_task = None
        self.is_running = False
        
        # Audit event categorization
        self.audit_categories = {
            'entity.created': {'category': 'data_creation', 'retention_years': 7},
            'entity.updated': {'category': 'data_modification', 'retention_years': 7},
            'entity.deleted': {'category': 'data_deletion', 'retention_years': 10},
            'entity.merged': {'category': 'data_consolidation', 'retention_years': 10},
            'quality.assessed': {'category': 'quality_monitoring', 'retention_years': 3},
            'duplicate.detected': {'category': 'duplicate_management', 'retention_years': 5},
            'access.granted': {'category': 'access_control', 'retention_years': 5},
            'access.denied': {'category': 'security_event', 'retention_years': 5}
        }
    
    async def start(self):
        """Start the audit logger"""
        if self.is_running:
            return
        
        self.session = aiohttp.ClientSession()
        self.is_running = True
        self.audit_task = asyncio.create_task(self._audit_worker())
        print("[MDM-Audit] APG audit logger started")
    
    async def stop(self):
        """Stop the audit logger"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.audit_task:
            self.audit_task.cancel()
            try:
                await self.audit_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        print("[MDM-Audit] APG audit logger stopped")
    
    async def log_audit_event(self, event_type: str, entity_id: Optional[str],
                            tenant_id: str, user_id: str, event_details: Dict[str, Any],
                            risk_level: str = 'medium') -> bool:
        """Log audit event to APG audit logging system"""
        try:
            audit_event = {
                'event_id': uuid7str(),
                'event_type': event_type,
                'entity_id': entity_id,
                'tenant_id': tenant_id,
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'event_details': event_details,
                'risk_level': risk_level,
                'source_service': 'mdm'
            }
            
            await self.audit_queue.put(audit_event)
            return True
            
        except Exception as e:
            print(f"[MDM-Audit] Error queuing audit event: {str(e)}")
            return False
    
    async def _audit_worker(self):
        """Background worker to send audit events to APG AUDL"""
        while self.is_running:
            try:
                # Get audit event from queue with timeout
                audit_event = await asyncio.wait_for(self.audit_queue.get(), timeout=1.0)
                
                # Send to APG audit logging
                success = await self._send_to_audl(audit_event)
                
                if success:
                    print(f"[MDM-Audit] Logged audit event: {audit_event['event_type']}")
                else:
                    print(f"[MDM-Audit] Failed to log audit event: {audit_event['event_type']}")
                
                # Mark task as done
                self.audit_queue.task_done()
                
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
            except Exception as e:
                print(f"[MDM-Audit] Audit worker error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _send_to_audl(self, audit_event: Dict[str, Any]) -> bool:
        """Send audit event to APG Audit Logging service"""
        try:
            if not self.session:
                return False
            
            # Get audit configuration for event type
            audit_config = self.audit_categories.get(audit_event['event_type'], {
                'category': 'general',
                'retention_years': 7
            })
            
            # Prepare audit payload for APG AUDL
            payload = {
                'event_id': audit_event['event_id'],
                'event_type': audit_event['event_type'],
                'category': audit_config['category'],
                'tenant_id': audit_event['tenant_id'],
                'user_id': audit_event['user_id'],
                'timestamp': audit_event['timestamp'],
                'source_service': audit_event['source_service'],
                'entity_id': audit_event.get('entity_id'),
                'risk_level': audit_event['risk_level'],
                'retention_period': f"{audit_config['retention_years']}Y",
                'event_details': audit_event['event_details'],
                'compliance_tags': ['mdm', 'data_governance', 'gdpr', 'sox']
            }
            
            # Send to APG AUDL
            async with self.session.post(
                f"{self.audl_url}/api/v1/audit/events",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                
                if response.status == 201:
                    return True
                else:
                    error_text = await response.text()
                    print(f"[MDM-Audit] AUDL logging error: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            print(f"[MDM-Audit] Error sending to AUDL: {str(e)}")
            return False


class ConfigurationManager:
    """APG Configuration Management integration for dynamic MDM configuration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.conf_url = self.config.get('conf_url', 'http://localhost:8082')
        self.session = None
        self.config_cache = {}
        self.config_watchers = {}
        
        # Default MDM configuration
        self.default_config = {
            'quality_thresholds': {
                'excellent': 95.0,
                'good': 80.0,
                'fair': 60.0,
                'poor': 40.0
            },
            'matching_thresholds': {
                'exact_match': 100.0,
                'high_confidence': 90.0,
                'medium_confidence': 70.0,
                'minimum_match': 50.0
            },
            'cache_settings': {
                'entity_ttl': 1800,
                'quality_ttl': 600,
                'search_ttl': 300
            },
            'ai_settings': {
                'enable_ai': True,
                'ollama_url': 'http://localhost:11434',
                'confidence_threshold': 0.7
            }
        }
    
    async def initialize(self):
        """Initialize connection to APG Configuration service"""
        try:
            self.session = aiohttp.ClientSession()
            # Load initial configuration
            await self.load_configuration()
            print(f"[MDM-Config] Connected to APG Configuration service at {self.conf_url}")
        except Exception as e:
            print(f"[MDM-Config] Failed to connect to Configuration service: {str(e)}")
    
    async def close(self):
        """Close configuration manager"""
        if self.session:
            await self.session.close()
    
    async def load_configuration(self) -> Dict[str, Any]:
        """Load MDM configuration from APG Configuration service"""
        try:
            if not self.session:
                return self.default_config
            
            async with self.session.get(
                f"{self.conf_url}/api/v1/config/mdm",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                
                if response.status == 200:
                    config_data = await response.json()
                    self.config_cache = {**self.default_config, **config_data.get('config', {})}
                    print("[MDM-Config] Configuration loaded from APG CONF")
                else:
                    print(f"[MDM-Config] Using default configuration (CONF service unavailable: {response.status})")
                    self.config_cache = self.default_config
                    
        except Exception as e:
            print(f"[MDM-Config] Error loading configuration: {str(e)}")
            self.config_cache = self.default_config
        
        return self.config_cache
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        try:
            # Navigate nested config using dot notation (e.g., 'quality_thresholds.excellent')
            value = self.config_cache
            for part in key.split('.'):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
            
        except Exception as e:
            print(f"[MDM-Config] Error getting config key '{key}': {str(e)}")
            return default
    
    async def update_config(self, key: str, value: Any, tenant_id: Optional[str] = None) -> bool:
        """Update configuration value in APG Configuration service"""
        try:
            if not self.session:
                return False
            
            payload = {
                'key': key,
                'value': value,
                'tenant_id': tenant_id,
                'updated_by': 'mdm_service',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            async with self.session.put(
                f"{self.conf_url}/api/v1/config/mdm/{key}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                
                if response.status == 200:
                    # Update local cache
                    await self.load_configuration()
                    print(f"[MDM-Config] Updated configuration key: {key}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"[MDM-Config] Config update error: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            print(f"[MDM-Config] Error updating configuration: {str(e)}")
            return False
    
    async def watch_config_changes(self, callback: Callable[[str, Any], None]) -> bool:
        """Watch for configuration changes from APG Configuration service"""
        try:
            # In a real implementation, this would establish a WebSocket connection
            # or use Server-Sent Events to receive configuration updates
            print("[MDM-Config] Configuration change watching enabled")
            return True
        except Exception as e:
            print(f"[MDM-Config] Error setting up config watch: {str(e)}")
            return False


class APGIntegrationManager:
    """Main integration manager for all APG ecosystem services"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize integration components
        self.event_publisher = EventPublisher(config)
        self.cache_manager = CacheManager(config)
        self.audit_logger = APGAuditLogger(config)
        self.config_manager = ConfigurationManager(config)
        
        self.is_initialized = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all APG integrations"""
        initialization_results = {
            'event_publisher': False,
            'cache_manager': False,
            'audit_logger': False,
            'config_manager': False
        }
        
        try:
            # Initialize event publisher
            await self.event_publisher.start()
            initialization_results['event_publisher'] = True
            
            # Initialize cache manager
            await self.cache_manager.initialize()
            initialization_results['cache_manager'] = self.cache_manager.redis_client is not None
            
            # Initialize audit logger
            await self.audit_logger.start()
            initialization_results['audit_logger'] = True
            
            # Initialize configuration manager
            await self.config_manager.initialize()
            initialization_results['config_manager'] = True
            
            self.is_initialized = True
            
        except Exception as e:
            print(f"[MDM-Integration] Error during initialization: {str(e)}")
        
        return {
            'status': 'success' if self.is_initialized else 'partial',
            'components': initialization_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown all APG integrations"""
        try:
            await self.event_publisher.stop()
            await self.cache_manager.close()
            await self.audit_logger.stop()
            await self.config_manager.close()
            
            self.is_initialized = False
            print("[MDM-Integration] All APG integrations shut down")
            
        except Exception as e:
            print(f"[MDM-Integration] Error during shutdown: {str(e)}")
    
    async def publish_entity_event(self, event_type: str, entity_id: str, entity_type: str,
                                 tenant_id: str, user_id: str, event_data: Dict[str, Any],
                                 correlation_id: str = None) -> bool:
        """Publish entity-related event to APG ecosystem"""
        try:
            # Create MDM event
            event = MDMEvent(
                event_id=uuid7str(),
                event_type=event_type,
                entity_id=entity_id,
                entity_type=entity_type,
                tenant_id=tenant_id,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                event_data=event_data,
                correlation_id=correlation_id
            )
            
            # Publish to event bus
            published = await self.event_publisher.publish_event(event)
            
            # Log audit event
            risk_level = 'high' if event_type in ['entity.deleted', 'entity.merged'] else 'medium'
            await self.audit_logger.log_audit_event(
                event_type, entity_id, tenant_id, user_id, event_data, risk_level
            )
            
            return published
            
        except Exception as e:
            print(f"[MDM-Integration] Error publishing entity event: {str(e)}")
            return False
    
    async def get_cached_entity(self, tenant_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity from cache with fallback"""
        return await self.cache_manager.get_entity(tenant_id, entity_id)
    
    async def cache_entity(self, tenant_id: str, entity_id: str, entity_data: Dict[str, Any]) -> bool:
        """Cache entity data"""
        return await self.cache_manager.set_entity(tenant_id, entity_id, entity_data)
    
    async def invalidate_entity_cache(self, tenant_id: str, entity_id: str) -> bool:
        """Invalidate entity cache"""
        return await self.cache_manager.invalidate_entity(tenant_id, entity_id)
    
    async def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from APG Configuration service"""
        return await self.config_manager.get_config(key, default)
    
    async def update_config_value(self, key: str, value: Any, tenant_id: str = None) -> bool:
        """Update configuration value in APG Configuration service"""
        return await self.config_manager.update_config(key, value, tenant_id)


# Export integration classes
__all__ = [
    'APGCapability', 'MDMEvent', 'EventPublisher', 'CacheManager', 
    'APGAuditLogger', 'ConfigurationManager', 'APGIntegrationManager'
]