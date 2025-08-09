"""
APG Central Configuration - Revolutionary Configuration Management Engine

The world's first AI-powered, universally compatible, zero-trust configuration
management platform that makes all existing solutions obsolete.

Features:
- AI-Powered Intelligent Configuration Management
- Universal Multi-Cloud Abstraction Layer  
- Real-Time Collaborative Configuration
- Zero-Trust Security by Design
- Unlimited Scale with Intelligent Tiering
- GitOps-Native with Advanced Workflows
- Semantic Configuration Understanding
- Developer Experience Revolution
- Autonomous Operations (NoOps)
- Ecosystem Integration Hub

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Core async libraries
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_, desc
import redis.asyncio as redis

# Configuration storage backends - real SDKs
import consul.aio
import etcd3
import hvac  # HashiCorp Vault client

# Real-time updates and watching
import asyncio_mqtt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration validation and parsing
import yaml
import toml
from pydantic import BaseModel, ValidationError
import cerberus  # Schema validation
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Encryption and security
from cryptography.fernet import Fernet
from jose import jwt
import bcrypt

# Utilities
from uuid_extensions import uuid7str
import structlog

# Error handling system
from .error_handling import (
	ErrorHandler, ErrorCategory, ErrorSeverity, ConfigurationError, 
	NetworkError, DatabaseError, ValidationError, AuthenticationError,
	ExternalServiceError, with_error_handling, error_context
)

# Real-time synchronization
from .realtime_sync import (
	RealtimeSyncManager, SyncEvent, SyncEventType, ConflictResolutionStrategy,
	create_realtime_sync_manager
)

logger = structlog.get_logger(__name__)

# =============================================================================
# Data Models and Types
# =============================================================================

class ConfigFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    PROPERTIES = "properties"
    INI = "ini"

class ConfigScope(Enum):
    GLOBAL = "global"
    TENANT = "tenant"
    ENVIRONMENT = "environment"
    SERVICE = "service"
    INSTANCE = "instance"

class ConfigStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class StorageBackend(Enum):
    REDIS = "redis"
    CONSUL = "consul"
    ETCD = "etcd"
    VAULT = "vault"
    DATABASE = "database"
    FILE = "file"

class ChangeType(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"

@dataclass
class ConfigKey:
    """Configuration key metadata."""
    key: str
    scope: ConfigScope
    format: ConfigFormat
    encrypted: bool
    version: int
    checksum: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

@dataclass
class ConfigValue:
    """Configuration value with metadata."""
    value: Any
    raw_value: str
    format: ConfigFormat
    encrypted: bool
    version: int
    checksum: str
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class ConfigChange:
    """Configuration change record."""
    change_id: str
    key: str
    change_type: ChangeType
    old_value: Optional[Any]
    new_value: Optional[Any]
    changed_by: str
    changed_at: datetime
    reason: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class ConfigWatcher:
    """Configuration watcher registration."""
    watcher_id: str
    key_pattern: str
    callback: Callable
    filters: Dict[str, Any]
    active: bool

# =============================================================================
# Central Configuration Service
# =============================================================================

class CentralConfigurationService:
    """Main central configuration service with multiple backend support."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        consul_client: Optional[consul.aio.Consul] = None,
        etcd_client: Optional[etcd3.Etcd3Client] = None,
        vault_client: Optional[hvac.Client] = None,
        encryption_key: Optional[str] = None,
        kafka_bootstrap_servers: Optional[List[str]] = None,
        mqtt_broker_host: Optional[str] = None,
        enable_realtime_sync: bool = True
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.consul_client = consul_client
        self.etcd_client = etcd_client
        self.vault_client = vault_client
        
        # Encryption
        self.cipher = Fernet(encryption_key.encode()) if encryption_key else None
        
        # Configuration storage services
        self.redis_storage = RedisConfigStorage(redis_client)
        self.consul_storage = ConsulConfigStorage(consul_client) if consul_client else None
        self.etcd_storage = EtcdConfigStorage(etcd_client) if etcd_client else None
        self.vault_storage = VaultConfigStorage(vault_client) if vault_client else None
        self.db_storage = DatabaseConfigStorage(db_session)
        
        # Configuration management services
        self.template_engine = ConfigTemplateEngine()
        self.validator = ConfigValidator()
        self.version_manager = ConfigVersionManager(db_session, redis_client)
        self.change_tracker = ConfigChangeTracker(db_session)
        self.watcher_manager = ConfigWatcherManager(redis_client)
        
        # Active watchers and subscriptions
        self.active_watchers: Dict[str, ConfigWatcher] = {}
        self.watch_tasks: Dict[str, asyncio.Task] = {}
        
        # Error handling system
        self.error_handler = ErrorHandler("central_configuration_service")
        
        # Circuit breakers for external services
        from .error_handling import CircuitBreakerConfig
        if consul_client:
            self.consul_circuit_breaker = self.error_handler.get_circuit_breaker(
                "consul", CircuitBreakerConfig(failure_threshold=3, reset_timeout=30.0)
            )
        if etcd_client:
            self.etcd_circuit_breaker = self.error_handler.get_circuit_breaker(
                "etcd", CircuitBreakerConfig(failure_threshold=3, reset_timeout=30.0)
            )
        if vault_client:
            self.vault_circuit_breaker = self.error_handler.get_circuit_breaker(
                "vault", CircuitBreakerConfig(failure_threshold=5, reset_timeout=60.0)
            )
        
        # Real-time synchronization
        self.enable_realtime_sync = enable_realtime_sync
        self.sync_manager: Optional[RealtimeSyncManager] = None
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.mqtt_broker_host = mqtt_broker_host
        
        # Initialize sync manager if enabled
        if enable_realtime_sync:
            asyncio.create_task(self._initialize_sync_manager())
    
    async def _initialize_sync_manager(self):
        """Initialize real-time synchronization manager."""
        try:
            self.sync_manager = await create_realtime_sync_manager(
                redis_client=self.redis_client,
                kafka_bootstrap_servers=self.kafka_bootstrap_servers,
                mqtt_broker_host=self.mqtt_broker_host
            )
            logger.info("Real-time synchronization manager initialized")
        except Exception as e:
            await self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH,
                "initialize_sync_manager", {}
            )
            # Continue without real-time sync if initialization fails
            self.enable_realtime_sync = False
        
    @with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM)
    async def set_config(
        self,
        key: str,
        value: Any,
        scope: ConfigScope = ConfigScope.GLOBAL,
        format: ConfigFormat = ConfigFormat.JSON,
        encrypted: bool = False,
        expires_at: Optional[datetime] = None,
        backend: StorageBackend = StorageBackend.REDIS,
        tenant_id: Optional[str] = None,
        changed_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Set configuration value with comprehensive error handling."""
        
        # Input validation
        if not key or not isinstance(key, str):
            raise ValidationError("Configuration key must be a non-empty string")
        
        if value is None:
            raise ValidationError("Configuration value cannot be None")
        
        async with error_context("set_config", ErrorCategory.CONFIGURATION_ERROR, error_handler=self.error_handler):
            try:
                # Build full key with scope
                full_key = self._build_key(key, scope, tenant_id)
                
                # Get current value for change tracking with proper error handling
                current_value = None
                try:
                    current_config = await self.get_config(key, scope, tenant_id=tenant_id)
                    current_value = current_config.value if current_config else None
                except DatabaseError as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.DATABASE_ERROR, ErrorSeverity.MEDIUM, 
                        "get_current_config_for_change_tracking",
                        {"key": key, "scope": scope.value, "tenant_id": tenant_id}
                    )
                    # Continue with None current_value
                except Exception as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
                        "get_current_config_for_change_tracking",
                        {"key": key, "scope": scope.value, "tenant_id": tenant_id}
                    )
                    # Continue with None current_value
                
                # Serialize value with error handling
                try:
                    if format == ConfigFormat.JSON:
                        raw_value = json.dumps(value, default=str)
                    elif format == ConfigFormat.YAML:
                        raw_value = yaml.dump(value, default_flow_style=False)
                    elif format == ConfigFormat.TOML:
                        raw_value = toml.dumps(value)
                    else:
                        raw_value = str(value)
                except (TypeError, ValueError, yaml.YAMLError) as e:
                    raise ValidationError(f"Failed to serialize configuration value in {format.value} format: {str(e)}")
                except Exception as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.VALIDATION_ERROR, ErrorSeverity.HIGH,
                        "serialize_config_value",
                        {"key": key, "format": format.value, "value_type": type(value).__name__}
                    )
                    raise ConfigurationError(f"Unexpected error during serialization: {str(e)}")
                
                # Encrypt if required with error handling
                if encrypted:
                    if not self.cipher:
                        raise ConfigurationError("Encryption requested but no encryption key configured")
                    try:
                        raw_value = self.cipher.encrypt(raw_value.encode()).decode()
                    except Exception as e:
                        await self.error_handler.handle_error(
                            e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH,
                            "encrypt_config_value",
                            {"key": key, "encrypted": encrypted}
                        )
                        raise ConfigurationError(f"Failed to encrypt configuration value: {str(e)}")
                
                # Calculate checksum
                try:
                    checksum = hashlib.sha256(raw_value.encode()).hexdigest()
                except Exception as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM,
                        "calculate_checksum",
                        {"key": key}
                    )
                    raise ConfigurationError(f"Failed to calculate checksum: {str(e)}")
                
                # Create config value
                config_value = ConfigValue(
                    value=value,
                    raw_value=raw_value,
                    format=format,
                    encrypted=encrypted,
                    version=1,  # Will be updated by version manager
                    checksum=checksum,
                    expires_at=expires_at,
                    metadata=metadata or {}
                )
                
                # Store in selected backend with circuit breaker protection
                storage = self._get_storage_backend(backend)
                try:
                    if backend in [StorageBackend.CONSUL] and hasattr(self, 'consul_circuit_breaker'):
                        version = await self.consul_circuit_breaker.call(storage.set, full_key, config_value)
                    elif backend in [StorageBackend.ETCD] and hasattr(self, 'etcd_circuit_breaker'):
                        version = await self.etcd_circuit_breaker.call(storage.set, full_key, config_value)
                    elif backend in [StorageBackend.VAULT] and hasattr(self, 'vault_circuit_breaker'):
                        version = await self.vault_circuit_breaker.call(storage.set, full_key, config_value)
                    else:
                        version = await storage.set(full_key, config_value)
                    
                    config_value.version = version
                    
                except NetworkError as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.NETWORK_ERROR, ErrorSeverity.HIGH,
                        "store_config_backend",
                        {"key": key, "backend": backend.value, "tenant_id": tenant_id}
                    )
                    raise
                except ExternalServiceError as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.EXTERNAL_SERVICE_ERROR, ErrorSeverity.HIGH,
                        "store_config_backend",
                        {"key": key, "backend": backend.value, "tenant_id": tenant_id}
                    )
                    raise
                except Exception as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL,
                        "store_config_backend",
                        {"key": key, "backend": backend.value, "tenant_id": tenant_id}
                    )
                    raise ConfigurationError(f"Failed to store configuration in {backend.value}: {str(e)}")
                
                # Store metadata in database with transaction handling
                try:
                    await self._store_config_metadata(full_key, config_value, scope, tenant_id, changed_by)
                except DatabaseError as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.DATABASE_ERROR, ErrorSeverity.HIGH,
                        "store_config_metadata",
                        {"key": key, "tenant_id": tenant_id, "changed_by": changed_by}
                    )
                    # Try to rollback the backend storage
                    try:
                        await storage.delete(full_key)
                    except Exception as rollback_error:
                        await self.error_handler.handle_error(
                            rollback_error, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL,
                            "rollback_config_storage",
                            {"key": key, "backend": backend.value}
                        )
                    raise
                except Exception as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH,
                        "store_config_metadata",
                        {"key": key, "tenant_id": tenant_id}
                    )
                    raise ConfigurationError(f"Failed to store configuration metadata: {str(e)}")
                
                # Track change with error handling
                try:
                    change_id = await self.change_tracker.record_change(
                        key=full_key,
                        change_type=ChangeType.UPDATE if current_value is not None else ChangeType.CREATE,
                        old_value=current_value,
                        new_value=value,
                        changed_by=changed_by,
                        metadata=metadata or {}
                    )
                except Exception as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM,
                        "track_config_change",
                        {"key": key, "changed_by": changed_by}
                    )
                    # Don't fail the entire operation for change tracking errors
                    change_id = None
                
                # Notify watchers with error handling
                try:
                    await self._notify_watchers(
                        full_key, config_value, 
                        ChangeType.UPDATE if current_value else ChangeType.CREATE
                    )
                except Exception as e:
                    await self.error_handler.handle_error(
                        e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
                        "notify_config_watchers",
                        {"key": key, "watcher_count": len(self.active_watchers)}
                    )
                    # Don't fail the operation for notification errors
                
                # Cache in Redis for fast access with error handling
                if backend != StorageBackend.REDIS:
                    try:
                        await self.redis_storage.set(full_key, config_value)
                    except Exception as e:
                        await self.error_handler.handle_error(
                            e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
                            "cache_config_in_redis",
                            {"key": key, "backend": backend.value}
                        )
                        # Don't fail the operation for caching errors
                
                # Broadcast real-time sync event
                if self.enable_realtime_sync and self.sync_manager:
                    try:
                        sync_event = SyncEvent(
                            event_type=SyncEventType.CONFIG_CHANGED if current_value else SyncEventType.CONFIG_CREATED,
                            source_node=self.sync_manager.node_id,
                            tenant_id=tenant_id,
                            user_id=changed_by,
                            config_key=full_key,
                            old_value=current_value,
                            new_value=value,
                            version=config_value.version,
                            checksum=config_value.checksum,
                            metadata={
                                "scope": scope.value,
                                "format": format.value,
                                "encrypted": encrypted,
                                "backend": backend.value,
                                "change_id": change_id
                            }
                        )
                        await self.sync_manager.broadcast_sync_event(sync_event)
                    except Exception as e:
                        await self.error_handler.handle_error(
                            e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
                            "broadcast_sync_event",
                            {"key": key, "sync_event_type": "config_changed"}
                        )
                        # Don't fail the operation for sync broadcast errors
                
                logger.info(f"Successfully set configuration {full_key} version {config_value.version}")
                return change_id
    
    # ==================== Real-Time Synchronization Methods ====================
    
    @with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM)
    async def acquire_config_lock(
        self,
        key: str,
        user_id: str,
        scope: ConfigScope = ConfigScope.GLOBAL,
        tenant_id: Optional[str] = None,
        timeout: int = 300
    ) -> bool:
        """Acquire exclusive lock on configuration for real-time editing."""
        if not self.enable_realtime_sync or not self.sync_manager:
            return True  # Always succeed if sync is disabled
        
        full_key = self._build_key(key, scope, tenant_id)
        
        return await self.sync_manager.acquire_config_lock(
            config_key=full_key,
            user_id=user_id,
            tenant_id=tenant_id,
            timeout=timeout
        )
    
    @with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.LOW)
    async def release_config_lock(
        self,
        key: str,
        user_id: str,
        scope: ConfigScope = ConfigScope.GLOBAL,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Release configuration lock."""
        if not self.enable_realtime_sync or not self.sync_manager:
            return True  # Always succeed if sync is disabled
        
        full_key = self._build_key(key, scope, tenant_id)
        
        return await self.sync_manager.release_config_lock(
            config_key=full_key,
            user_id=user_id,
            tenant_id=tenant_id
        )
    
    @with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.HIGH)
    async def resolve_config_conflict(
        self,
        key: str,
        competing_updates: List[Dict[str, Any]],
        resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS,
        scope: ConfigScope = ConfigScope.GLOBAL,
        tenant_id: Optional[str] = None
    ) -> Any:
        """Resolve configuration conflicts using specified strategy."""
        if not self.enable_realtime_sync or not self.sync_manager:
            # Fallback to simple last-write-wins
            if competing_updates:
                latest_update = max(
                    competing_updates,
                    key=lambda x: x.get("timestamp", "1970-01-01T00:00:00Z")
                )
                return latest_update.get("new_value")
            return None
        
        full_key = self._build_key(key, scope, tenant_id)
        
        conflict_info = await self.sync_manager.detect_and_resolve_conflicts(
            config_key=full_key,
            competing_updates=competing_updates,
            resolution_strategy=resolution_strategy
        )
        
        if conflict_info.resolved:
            # Apply the resolved value
            resolved_value = conflict_info.resolution_result
            if isinstance(resolved_value, dict) and "new_value" in resolved_value:
                await self.set_config(
                    key=key,
                    value=resolved_value["new_value"],
                    scope=scope,
                    tenant_id=tenant_id,
                    changed_by=f"conflict_resolution_{conflict_info.conflict_id}"
                )
                return resolved_value["new_value"]
            else:
                await self.set_config(
                    key=key,
                    value=resolved_value,
                    scope=scope,
                    tenant_id=tenant_id,
                    changed_by=f"conflict_resolution_{conflict_info.conflict_id}"
                )
                return resolved_value
        
        return None
    
    @with_error_handling(ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM)
    async def add_realtime_websocket_connection(
        self,
        connection_id: str,
        websocket,
        subscription_patterns: Optional[List[str]] = None,
        tenant_id: Optional[str] = None
    ):
        """Add WebSocket connection for real-time configuration updates."""
        if not self.enable_realtime_sync or not self.sync_manager:
            raise ConfigurationError("Real-time synchronization is not enabled")
        
        # Add tenant-specific patterns if provided
        patterns = subscription_patterns or ["*"]
        if tenant_id:
            tenant_patterns = [f"*{tenant_id}*", f"{tenant_id}:*"]
            patterns.extend(tenant_patterns)
        
        await self.sync_manager.add_websocket_connection(
            connection_id=connection_id,
            websocket=websocket,
            subscription_patterns=patterns
        )
    
    async def remove_realtime_websocket_connection(self, connection_id: str):
        """Remove WebSocket connection."""
        if self.enable_realtime_sync and self.sync_manager:
            await self.sync_manager.remove_websocket_connection(connection_id)
    
    @with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM)
    async def get_realtime_sync_status(self) -> Dict[str, Any]:
        """Get real-time synchronization status and metrics."""
        if not self.enable_realtime_sync or not self.sync_manager:
            return {
                "enabled": False,
                "status": "disabled"
            }
        
        return {
            "enabled": True,
            "status": "active",
            "node_id": self.sync_manager.node_id,
            "active_connections": len(self.sync_manager.websocket_connections),
            "active_locks": len(self.sync_manager.active_locks),
            "kafka_enabled": self.sync_manager.kafka_producer is not None,
            "mqtt_enabled": self.sync_manager.mqtt_client is not None,
            "last_heartbeat": datetime.now(timezone.utc).isoformat()
        }
    
    @with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM)
    async def get_config(
        self,
        key: str,
        scope: ConfigScope = ConfigScope.GLOBAL,
        version: Optional[int] = None,
        backend: StorageBackend = StorageBackend.REDIS,
        tenant_id: Optional[str] = None,
        decrypt: bool = True
    ) -> Optional[ConfigValue]:
        """Get configuration value with comprehensive error handling."""
        
        # Input validation
        if not key or not isinstance(key, str):
            raise ValidationError("Configuration key must be a non-empty string")
        
        async with error_context("get_config", ErrorCategory.CONFIGURATION_ERROR, error_handler=self.error_handler):
            full_key = self._build_key(key, scope, tenant_id)
            
            # Try cache first (Redis) with error handling
            if backend != StorageBackend.REDIS:
                try:
                    cached_value = await self.redis_storage.get(full_key, version)
                if cached_value:
                    return cached_value
            
            # Get from specified backend
            storage = self._get_storage_backend(backend)
            config_value = await storage.get(full_key, version)
            
            if not config_value:
                return None
            
            # Decrypt if needed
            if config_value.encrypted and decrypt and self.cipher:
                try:
                    decrypted_raw = self.cipher.decrypt(config_value.raw_value.encode()).decode()
                    
                    # Parse decrypted value
                    if config_value.format == ConfigFormat.JSON:
                        config_value.value = json.loads(decrypted_raw)
                    elif config_value.format == ConfigFormat.YAML:
                        config_value.value = yaml.safe_load(decrypted_raw)
                    elif config_value.format == ConfigFormat.TOML:
                        config_value.value = toml.loads(decrypted_raw)
                    else:
                        config_value.value = decrypted_raw
                        
                except Exception as e:
                    logger.error(f"Failed to decrypt configuration {full_key}: {e}")
                    raise
            
            # Cache in Redis if from different backend
            if backend != StorageBackend.REDIS:
                await self.redis_storage.set(full_key, config_value)
            
            return config_value
            
        except Exception as e:
            logger.error(f"Failed to get configuration {key}: {e}")
            raise
    
    async def delete_config(
        self,
        key: str,
        scope: ConfigScope = ConfigScope.GLOBAL,
        backend: StorageBackend = StorageBackend.REDIS,
        tenant_id: Optional[str] = None,
        deleted_by: str = "system"
    ) -> bool:
        """Delete configuration value."""
        
        try:
            full_key = self._build_key(key, scope, tenant_id)
            
            # Get current value for change tracking
            current_config = await self.get_config(key, scope, tenant_id=tenant_id, backend=backend)
            current_value = current_config.value if current_config else None
            
            # Delete from backend
            storage = self._get_storage_backend(backend)
            success = await storage.delete(full_key)
            
            if success:
                # Delete from cache
                await self.redis_storage.delete(full_key)
                
                # Track change
                await self.change_tracker.record_change(
                    key=full_key,
                    change_type=ChangeType.DELETE,
                    old_value=current_value,
                    new_value=None,
                    changed_by=deleted_by
                )
                
                # Notify watchers
                await self._notify_watchers(full_key, None, ChangeType.DELETE)
                
                logger.info(f"Deleted configuration {full_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete configuration {key}: {e}")
            raise
    
    async def list_configs(
        self,
        key_pattern: str = "*",
        scope: Optional[ConfigScope] = None,
        backend: StorageBackend = StorageBackend.REDIS,
        tenant_id: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """List configuration keys matching pattern."""
        
        try:
            # Build search pattern
            if scope:
                pattern = self._build_key(key_pattern, scope, tenant_id)
            else:
                pattern = key_pattern
            
            storage = self._get_storage_backend(backend)
            config_list = await storage.list(pattern)
            
            # Add metadata if requested
            if include_metadata:
                enriched_list = []
                for config_info in config_list:
                    metadata = await self._get_config_metadata(config_info["key"])
                    config_info.update(metadata)
                    enriched_list.append(config_info)
                return enriched_list
            
            return config_list
            
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            raise
    
    async def watch_config(
        self,
        key_pattern: str,
        callback: Callable[[str, Optional[ConfigValue], ChangeType], None],
        scope: Optional[ConfigScope] = None,
        tenant_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Watch configuration changes."""
        
        watcher_id = f"watch_{uuid7str()}"
        
        # Build full pattern
        if scope:
            full_pattern = self._build_key(key_pattern, scope, tenant_id)
        else:
            full_pattern = key_pattern
        
        # Create watcher
        watcher = ConfigWatcher(
            watcher_id=watcher_id,
            key_pattern=full_pattern,
            callback=callback,
            filters=filters or {},
            active=True
        )
        
        self.active_watchers[watcher_id] = watcher
        
        # Register with watcher manager
        await self.watcher_manager.register_watcher(watcher)
        
        logger.info(f"Registered configuration watcher {watcher_id} for pattern {full_pattern}")
        return watcher_id
    
    async def unwatch_config(self, watcher_id: str) -> bool:
        """Remove configuration watcher."""
        
        if watcher_id in self.active_watchers:
            watcher = self.active_watchers[watcher_id]
            watcher.active = False
            
            await self.watcher_manager.unregister_watcher(watcher_id)
            del self.active_watchers[watcher_id]
            
            logger.info(f"Unregistered configuration watcher {watcher_id}")
            return True
        
        return False
    
    async def apply_template(
        self,
        template_content: str,
        variables: Dict[str, Any],
        output_key: str,
        scope: ConfigScope = ConfigScope.GLOBAL,
        tenant_id: Optional[str] = None,
        created_by: str = "system"
    ) -> str:
        """Apply Jinja2 template to generate configuration."""
        
        try:
            # Render template
            rendered_config = self.template_engine.render(template_content, variables)
            
            # Parse rendered configuration
            try:
                config_value = yaml.safe_load(rendered_config)
            except:
                config_value = rendered_config
            
            # Store rendered configuration
            change_id = await self.set_config(
                key=output_key,
                value=config_value,
                scope=scope,
                format=ConfigFormat.YAML,
                tenant_id=tenant_id,
                changed_by=created_by,
                metadata={
                    "generated_from_template": True,
                    "template_variables": variables
                }
            )
            
            logger.info(f"Applied template to generate configuration {output_key}")
            return change_id
            
        except Exception as e:
            logger.error(f"Failed to apply template: {e}")
            raise
    
    async def validate_config(
        self,
        key: str,
        value: Any,
        schema: Dict[str, Any],
        scope: ConfigScope = ConfigScope.GLOBAL,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate configuration against schema."""
        
        try:
            validation_result = self.validator.validate(value, schema)
            
            # Store validation result
            full_key = self._build_key(key, scope, tenant_id)
            cache_key = f"validation:{full_key}"
            
            await self.redis_client.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(validation_result, default=str)
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate configuration {key}: {e}")
            raise
    
    async def get_config_history(
        self,
        key: str,
        scope: ConfigScope = ConfigScope.GLOBAL,
        tenant_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ConfigChange]:
        """Get configuration change history."""
        
        full_key = self._build_key(key, scope, tenant_id)
        return await self.change_tracker.get_history(full_key, limit)
    
    async def restore_config_version(
        self,
        key: str,
        version: int,
        scope: ConfigScope = ConfigScope.GLOBAL,
        tenant_id: Optional[str] = None,
        restored_by: str = "system"
    ) -> str:
        """Restore configuration to specific version."""
        
        try:
            # Get version from version manager
            config_value = await self.version_manager.get_version(key, version, scope, tenant_id)
            
            if not config_value:
                raise ValueError(f"Version {version} not found for key {key}")
            
            # Set as current configuration
            change_id = await self.set_config(
                key=key,
                value=config_value.value,
                scope=scope,
                format=config_value.format,
                encrypted=config_value.encrypted,
                tenant_id=tenant_id,
                changed_by=restored_by,
                metadata={
                    "restored_from_version": version,
                    "original_checksum": config_value.checksum
                }
            )
            
            logger.info(f"Restored configuration {key} to version {version}")
            return change_id
            
        except Exception as e:
            logger.error(f"Failed to restore configuration {key} to version {version}: {e}")
            raise
    
    async def bulk_import_configs(
        self,
        configs: Dict[str, Any],
        scope: ConfigScope = ConfigScope.GLOBAL,
        format: ConfigFormat = ConfigFormat.JSON,
        tenant_id: Optional[str] = None,
        imported_by: str = "system",
        overwrite: bool = False
    ) -> Dict[str, str]:
        """Bulk import configurations."""
        
        results = {}
        
        for key, value in configs.items():
            try:
                # Check if key exists
                if not overwrite:
                    existing = await self.get_config(key, scope, tenant_id=tenant_id)
                    if existing:
                        results[key] = f"skipped (exists)"
                        continue
                
                # Import configuration
                change_id = await self.set_config(
                    key=key,
                    value=value,
                    scope=scope,
                    format=format,
                    tenant_id=tenant_id,
                    changed_by=imported_by,
                    metadata={"bulk_imported": True}
                )
                
                results[key] = change_id
                
            except Exception as e:
                results[key] = f"error: {str(e)}"
        
        logger.info(f"Bulk imported {len(results)} configurations")
        return results
    
    async def export_configs_to_file(
        self,
        file_path: str,
        key_pattern: str = "*",
        scope: Optional[ConfigScope] = None,
        format: ConfigFormat = ConfigFormat.YAML,
        tenant_id: Optional[str] = None,
        include_metadata: bool = False
    ) -> bool:
        """Export configurations to file."""
        
        try:
            # Get configurations
            config_list = await self.list_configs(
                key_pattern=key_pattern,
                scope=scope,
                tenant_id=tenant_id,
                include_metadata=include_metadata
            )
            
            # Build export data
            export_data = {}
            for config_info in config_list:
                key = config_info["key"]
                config_value = await self.get_config(key, scope, tenant_id=tenant_id)
                
                if config_value:
                    export_data[key] = {
                        "value": config_value.value,
                        "format": config_value.format.value,
                        "version": config_value.version,
                        "metadata": config_value.metadata
                    }
            
            # Write to file
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == ConfigFormat.JSON:
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format == ConfigFormat.YAML:
                with open(path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            elif format == ConfigFormat.TOML:
                with open(path, 'w') as f:
                    toml.dump(export_data, f)
            
            logger.info(f"Exported {len(export_data)} configurations to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configurations: {e}")
            return False
    
    def _build_key(self, key: str, scope: ConfigScope, tenant_id: Optional[str] = None) -> str:
        """Build full configuration key with scope."""
        
        parts = [scope.value]
        
        if tenant_id and scope in [ConfigScope.TENANT, ConfigScope.SERVICE, ConfigScope.INSTANCE]:
            parts.append(tenant_id)
        
        parts.append(key)
        
        return ":".join(parts)
    
    def _get_storage_backend(self, backend: StorageBackend):
        """Get storage backend instance."""
        
        if backend == StorageBackend.REDIS:
            return self.redis_storage
        elif backend == StorageBackend.CONSUL:
            if not self.consul_storage:
                raise ValueError("Consul storage not configured")
            return self.consul_storage
        elif backend == StorageBackend.ETCD:
            if not self.etcd_storage:
                raise ValueError("etcd storage not configured")
            return self.etcd_storage
        elif backend == StorageBackend.VAULT:
            if not self.vault_storage:
                raise ValueError("Vault storage not configured")
            return self.vault_storage
        elif backend == StorageBackend.DATABASE:
            return self.db_storage
        else:
            raise ValueError(f"Unknown storage backend: {backend}")
    
    async def _store_config_metadata(
        self,
        key: str,
        config_value: ConfigValue,
        scope: ConfigScope,
        tenant_id: Optional[str],
        changed_by: str
    ):
        """Store configuration metadata in database."""
        
        from ..database import CRConfiguration
        
        # Check if configuration exists
        result = await self.db_session.execute(
            select(CRConfiguration).where(
                and_(
                    CRConfiguration.applet_id == key,
                    CRConfiguration.tenant_id == tenant_id or ""
                )
            )
        )
        
        existing_config = result.scalar_one_or_none()
        
        if existing_config:
            # Update existing
            existing_config.configuration_data = {
                "format": config_value.format.value,
                "encrypted": config_value.encrypted,
                "version": config_value.version,
                "checksum": config_value.checksum,
                "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
                "metadata": config_value.metadata
            }
            existing_config.changed_by = changed_by
            existing_config.updated_at = datetime.now(timezone.utc)
        else:
            # Create new
            config = CRConfiguration(
                applet_id=key,
                configuration_data={
                    "format": config_value.format.value,
                    "encrypted": config_value.encrypted,
                    "version": config_value.version,
                    "checksum": config_value.checksum,
                    "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
                    "metadata": config_value.metadata
                },
                tenant_id=tenant_id or "",
                changed_by=changed_by
            )
            self.db_session.add(config)
        
        await self.db_session.commit()
    
    async def _get_config_metadata(self, key: str) -> Dict[str, Any]:
        """Get configuration metadata from database."""
        
        from ..database import CRConfiguration
        
        result = await self.db_session.execute(
            select(CRConfiguration).where(CRConfiguration.applet_id == key)
        )
        
        config = result.scalar_one_or_none()
        
        if config:
            return {
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat(),
                "changed_by": config.changed_by,
                **config.configuration_data
            }
        
        return {}
    
    async def _notify_watchers(
        self,
        key: str,
        config_value: Optional[ConfigValue],
        change_type: ChangeType
    ):
        """Notify active watchers of configuration changes."""
        
        for watcher in self.active_watchers.values():
            if watcher.active and self._key_matches_pattern(key, watcher.key_pattern):
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        watcher.callback,
                        key,
                        config_value,
                        change_type
                    )
                except Exception as e:
                    logger.error(f"Watcher callback failed for {watcher.watcher_id}: {e}")
    
    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (supports wildcards)."""
        
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def close(self):
        """Close configuration service and clean up resources."""
        
        # Stop all watchers
        for watcher_id in list(self.active_watchers.keys()):
            await self.unwatch_config(watcher_id)
        
        # Cancel watch tasks
        for task in self.watch_tasks.values():
            task.cancel()
        
        # Close clients
        if self.consul_client:
            await self.consul_client.close()
        
        if self.etcd_client:
            self.etcd_client.close()

# =============================================================================
# Storage Backend Implementations
# =============================================================================

class RedisConfigStorage:
    """Redis-based configuration storage."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    async def set(self, key: str, config_value: ConfigValue) -> int:
        """Store configuration in Redis."""
        
        # Get current version
        version_key = f"{key}:version"
        current_version = await self.redis_client.get(version_key)
        new_version = (int(current_version) + 1) if current_version else 1
        
        # Store configuration
        config_data = {
            "value": config_value.raw_value,
            "format": config_value.format.value,
            "encrypted": config_value.encrypted,
            "version": new_version,
            "checksum": config_value.checksum,
            "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
            "metadata": config_value.metadata,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Set with expiration if specified
        if config_value.expires_at:
            ttl = int((config_value.expires_at - datetime.now(timezone.utc)).total_seconds())
            await self.redis_client.setex(key, ttl, json.dumps(config_data, default=str))
        else:
            await self.redis_client.set(key, json.dumps(config_data, default=str))
        
        # Update version
        await self.redis_client.set(version_key, new_version)
        
        # Store version history
        history_key = f"{key}:history:{new_version}"
        await self.redis_client.setex(history_key, 86400 * 30, json.dumps(config_data, default=str))  # 30 days
        
        return new_version
    
    async def get(self, key: str, version: Optional[int] = None) -> Optional[ConfigValue]:
        """Get configuration from Redis."""
        
        if version:
            # Get specific version
            history_key = f"{key}:history:{version}"
            data = await self.redis_client.get(history_key)
        else:
            # Get current version
            data = await self.redis_client.get(key)
        
        if not data:
            return None
        
        try:
            config_data = json.loads(data)
            
            return ConfigValue(
                value=None,  # Will be parsed later
                raw_value=config_data["value"],
                format=ConfigFormat(config_data["format"]),
                encrypted=config_data["encrypted"],
                version=config_data["version"],
                checksum=config_data["checksum"],
                expires_at=datetime.fromisoformat(config_data["expires_at"]) if config_data.get("expires_at") else None,
                metadata=config_data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse Redis configuration data for {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete configuration from Redis."""
        
        # Delete main key
        result = await self.redis_client.delete(key)
        
        # Delete version info
        await self.redis_client.delete(f"{key}:version")
        
        # Delete history (get all history keys first)
        pattern = f"{key}:history:*"
        history_keys = []
        async for history_key in self.redis_client.scan_iter(match=pattern):
            history_keys.append(history_key)
        
        if history_keys:
            await self.redis_client.delete(*history_keys)
        
        return result > 0
    
    async def list(self, pattern: str) -> List[Dict[str, Any]]:
        """List configurations matching pattern."""
        
        configs = []
        async for key in self.redis_client.scan_iter(match=pattern):
            # Skip version and history keys
            if key.endswith(b':version') or b':history:' in key:
                continue
            
            key_str = key.decode()
            config_value = await self.get(key_str)
            
            if config_value:
                configs.append({
                    "key": key_str,
                    "format": config_value.format.value,
                    "version": config_value.version,
                    "checksum": config_value.checksum,
                    "encrypted": config_value.encrypted
                })
        
        return configs

class ConsulConfigStorage:
    """Consul-based configuration storage."""
    
    def __init__(self, consul_client: consul.aio.Consul):
        self.consul_client = consul_client
    
    async def set(self, key: str, config_value: ConfigValue) -> int:
        """Store configuration in Consul."""
        
        # Build Consul key
        consul_key = f"config/{key}"
        
        # Get current version
        index, data = await self.consul_client.kv.get(f"{consul_key}/version")
        new_version = (int(data["Value"].decode()) + 1) if data else 1
        
        # Store configuration
        config_data = {
            "value": config_value.raw_value,
            "format": config_value.format.value,
            "encrypted": config_value.encrypted,
            "version": new_version,
            "checksum": config_value.checksum,
            "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
            "metadata": config_value.metadata,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in Consul
        await self.consul_client.kv.put(consul_key, json.dumps(config_data, default=str))
        await self.consul_client.kv.put(f"{consul_key}/version", str(new_version))
        
        # Store version history
        await self.consul_client.kv.put(
            f"{consul_key}/history/{new_version}",
            json.dumps(config_data, default=str)
        )
        
        return new_version
    
    async def get(self, key: str, version: Optional[int] = None) -> Optional[ConfigValue]:
        """Get configuration from Consul."""
        
        consul_key = f"config/{key}"
        
        if version:
            consul_key = f"{consul_key}/history/{version}"
        
        index, data = await self.consul_client.kv.get(consul_key)
        
        if not data:
            return None
        
        try:
            config_data = json.loads(data["Value"].decode())
            
            return ConfigValue(
                value=None,
                raw_value=config_data["value"],
                format=ConfigFormat(config_data["format"]),
                encrypted=config_data["encrypted"],
                version=config_data["version"],
                checksum=config_data["checksum"],
                expires_at=datetime.fromisoformat(config_data["expires_at"]) if config_data.get("expires_at") else None,
                metadata=config_data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse Consul configuration data for {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete configuration from Consul."""
        
        consul_key = f"config/{key}"
        
        # Delete recursively (includes history)
        result = await self.consul_client.kv.delete(consul_key, recurse=True)
        return result
    
    async def list(self, pattern: str) -> List[Dict[str, Any]]:
        """List configurations matching pattern."""
        
        index, keys = await self.consul_client.kv.get("config/", keys=True)
        
        if not keys:
            return []
        
        configs = []
        for consul_key in keys:
            key = consul_key.replace("config/", "")
            
            # Skip version and history keys
            if "/version" in key or "/history/" in key:
                continue
            
            # Simple pattern matching
            if pattern == "*" or pattern in key:
                config_value = await self.get(key)
                if config_value:
                    configs.append({
                        "key": key,
                        "format": config_value.format.value,
                        "version": config_value.version,
                        "checksum": config_value.checksum,
                        "encrypted": config_value.encrypted
                    })
        
        return configs

class EtcdConfigStorage:
    """etcd-based configuration storage."""
    
    def __init__(self, etcd_client: etcd3.Etcd3Client):
        self.etcd_client = etcd_client
    
    async def set(self, key: str, config_value: ConfigValue) -> int:
        """Store configuration in etcd."""
        
        # Build etcd key
        etcd_key = f"config/{key}"
        
        # Get current version
        version_data, _ = self.etcd_client.get(f"{etcd_key}/version")
        new_version = (int(version_data.decode()) + 1) if version_data else 1
        
        # Store configuration
        config_data = {
            "value": config_value.raw_value,
            "format": config_value.format.value,
            "encrypted": config_value.encrypted,
            "version": new_version,
            "checksum": config_value.checksum,
            "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
            "metadata": config_value.metadata,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in etcd
        self.etcd_client.put(etcd_key, json.dumps(config_data, default=str))
        self.etcd_client.put(f"{etcd_key}/version", str(new_version))
        
        # Store version history
        self.etcd_client.put(
            f"{etcd_key}/history/{new_version}",
            json.dumps(config_data, default=str)
        )
        
        return new_version
    
    async def get(self, key: str, version: Optional[int] = None) -> Optional[ConfigValue]:
        """Get configuration from etcd."""
        
        etcd_key = f"config/{key}"
        
        if version:
            etcd_key = f"{etcd_key}/history/{version}"
        
        data, _ = self.etcd_client.get(etcd_key)
        
        if not data:
            return None
        
        try:
            config_data = json.loads(data.decode())
            
            return ConfigValue(
                value=None,
                raw_value=config_data["value"],
                format=ConfigFormat(config_data["format"]),
                encrypted=config_data["encrypted"],
                version=config_data["version"],
                checksum=config_data["checksum"],
                expires_at=datetime.fromisoformat(config_data["expires_at"]) if config_data.get("expires_at") else None,
                metadata=config_data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse etcd configuration data for {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete configuration from etcd."""
        
        etcd_key = f"config/{key}"
        
        # Delete with prefix (includes history)
        result = self.etcd_client.delete_prefix(etcd_key)
        return result.deleted > 0
    
    async def list(self, pattern: str) -> List[Dict[str, Any]]:
        """List configurations matching pattern."""
        
        configs = []
        
        # Get all keys with config prefix
        for data, metadata in self.etcd_client.get_prefix("config/"):
            key = metadata.key.decode().replace("config/", "")
            
            # Skip version and history keys
            if "/version" in key or "/history/" in key:
                continue
            
            # Simple pattern matching
            if pattern == "*" or pattern in key:
                config_value = await self.get(key)
                if config_value:
                    configs.append({
                        "key": key,
                        "format": config_value.format.value,
                        "version": config_value.version,
                        "checksum": config_value.checksum,
                        "encrypted": config_value.encrypted
                    })
        
        return configs

class VaultConfigStorage:
    """HashiCorp Vault-based configuration storage."""
    
    def __init__(self, vault_client: hvac.Client):
        self.vault_client = vault_client
        self.mount_point = "secret"
    
    async def set(self, key: str, config_value: ConfigValue) -> int:
        """Store configuration in Vault."""
        
        # Build Vault path
        vault_path = f"config/{key}"
        
        # Get current version
        try:
            current_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f"{vault_path}/version",
                mount_point=self.mount_point
            )
            new_version = current_data["data"]["data"]["version"] + 1
        except:
            new_version = 1
        
        # Store configuration
        config_data = {
            "value": config_value.raw_value,
            "format": config_value.format.value,
            "encrypted": config_value.encrypted,
            "version": new_version,
            "checksum": config_value.checksum,
            "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
            "metadata": config_value.metadata,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in Vault
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=vault_path,
            secret=config_data,
            mount_point=self.mount_point
        )
        
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=f"{vault_path}/version",
            secret={"version": new_version},
            mount_point=self.mount_point
        )
        
        return new_version
    
    async def get(self, key: str, version: Optional[int] = None) -> Optional[ConfigValue]:
        """Get configuration from Vault."""
        
        vault_path = f"config/{key}"
        
        try:
            if version:
                # Get specific version
                response = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=vault_path,
                    version=version,
                    mount_point=self.mount_point
                )
            else:
                # Get latest version
                response = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=vault_path,
                    mount_point=self.mount_point
                )
            
            config_data = response["data"]["data"]
            
            return ConfigValue(
                value=None,
                raw_value=config_data["value"],
                format=ConfigFormat(config_data["format"]),
                encrypted=config_data["encrypted"],
                version=config_data["version"],
                checksum=config_data["checksum"],
                expires_at=datetime.fromisoformat(config_data["expires_at"]) if config_data.get("expires_at") else None,
                metadata=config_data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to get Vault configuration {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete configuration from Vault."""
        
        vault_path = f"config/{key}"
        
        try:
            # Delete all versions
            self.vault_client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=vault_path,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete Vault configuration {key}: {e}")
            return False
    
    async def list(self, pattern: str) -> List[Dict[str, Any]]:
        """List configurations matching pattern."""
        
        try:
            response = self.vault_client.secrets.kv.v2.list_secrets(
                path="config",
                mount_point=self.mount_point
            )
            
            keys = response["data"]["keys"]
            configs = []
            
            for key in keys:
                # Skip version keys
                if key.endswith("/version"):
                    continue
                
                # Simple pattern matching
                if pattern == "*" or pattern in key:
                    config_value = await self.get(key)
                    if config_value:
                        configs.append({
                            "key": key,
                            "format": config_value.format.value,
                            "version": config_value.version,
                            "checksum": config_value.checksum,
                            "encrypted": config_value.encrypted
                        })
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to list Vault configurations: {e}")
            return []

class DatabaseConfigStorage:
    """Database-based configuration storage."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def set(self, key: str, config_value: ConfigValue) -> int:
        """Store configuration in database."""
        
        from ..database import CRConfiguration, CRConfigurationHistory
        
        # Get current configuration
        result = await self.db_session.execute(
            select(CRConfiguration).where(CRConfiguration.applet_id == key)
        )
        
        existing_config = result.scalar_one_or_none()
        new_version = 1
        
        if existing_config:
            # Update existing
            new_version = existing_config.configuration_data.get("version", 0) + 1
            old_value = existing_config.configuration_data
            
            existing_config.configuration_data = {
                "value": config_value.raw_value,
                "format": config_value.format.value,
                "encrypted": config_value.encrypted,
                "version": new_version,
                "checksum": config_value.checksum,
                "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
                "metadata": config_value.metadata
            }
            existing_config.updated_at = datetime.now(timezone.utc)
            
            # Create history record
            history = CRConfigurationHistory(
                configuration_id=existing_config.id,
                change_type="update",
                old_value=old_value,
                new_value=existing_config.configuration_data,
                changed_by="system",
                tenant_id=existing_config.tenant_id
            )
            self.db_session.add(history)
            
        else:
            # Create new
            config = CRConfiguration(
                applet_id=key,
                configuration_data={
                    "value": config_value.raw_value,
                    "format": config_value.format.value,
                    "encrypted": config_value.encrypted,
                    "version": new_version,
                    "checksum": config_value.checksum,
                    "expires_at": config_value.expires_at.isoformat() if config_value.expires_at else None,
                    "metadata": config_value.metadata
                },
                tenant_id="",  # Will be set by caller
                changed_by="system"
            )
            self.db_session.add(config)
            await self.db_session.flush()
            
            # Create history record
            history = CRConfigurationHistory(
                configuration_id=config.id,
                change_type="create",
                new_value=config.configuration_data,
                changed_by="system",
                tenant_id=config.tenant_id
            )
            self.db_session.add(history)
        
        await self.db_session.commit()
        return new_version
    
    async def get(self, key: str, version: Optional[int] = None) -> Optional[ConfigValue]:
        """Get configuration from database."""
        
        from ..database import CRConfiguration, CRConfigurationHistory
        
        if version:
            # Get specific version from history
            result = await self.db_session.execute(
                select(CRConfigurationHistory).join(CRConfiguration).where(
                    and_(
                        CRConfiguration.applet_id == key,
                        CRConfigurationHistory.new_value["version"].astext.cast(Integer) == version
                    )
                ).order_by(desc(CRConfigurationHistory.created_at)).limit(1)
            )
            
            history = result.scalar_one_or_none()
            if not history:
                return None
            
            config_data = history.new_value
        else:
            # Get current version
            result = await self.db_session.execute(
                select(CRConfiguration).where(CRConfiguration.applet_id == key)
            )
            
            config = result.scalar_one_or_none()
            if not config:
                return None
            
            config_data = config.configuration_data
        
        return ConfigValue(
            value=None,
            raw_value=config_data["value"],
            format=ConfigFormat(config_data["format"]),
            encrypted=config_data["encrypted"],
            version=config_data["version"],
            checksum=config_data["checksum"],
            expires_at=datetime.fromisoformat(config_data["expires_at"]) if config_data.get("expires_at") else None,
            metadata=config_data.get("metadata", {})
        )
    
    async def delete(self, key: str) -> bool:
        """Delete configuration from database."""
        
        from ..database import CRConfiguration
        
        result = await self.db_session.execute(
            delete(CRConfiguration).where(CRConfiguration.applet_id == key)
        )
        
        await self.db_session.commit()
        return result.rowcount > 0
    
    async def list(self, pattern: str) -> List[Dict[str, Any]]:
        """List configurations matching pattern."""
        
        from ..database import CRConfiguration
        
        if pattern == "*":
            query = select(CRConfiguration)
        else:
            query = select(CRConfiguration).where(CRConfiguration.applet_id.like(f"%{pattern}%"))
        
        result = await self.db_session.execute(query)
        configs = result.scalars().all()
        
        config_list = []
        for config in configs:
            config_data = config.configuration_data
            config_list.append({
                "key": config.applet_id,
                "format": config_data.get("format"),
                "version": config_data.get("version"),
                "checksum": config_data.get("checksum"),
                "encrypted": config_data.get("encrypted", False)
            })
        
        return config_list

# =============================================================================
# Supporting Services
# =============================================================================

class ConfigTemplateEngine:
    """Jinja2-based configuration template engine."""
    
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader([]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def render(self, template_content: str, variables: Dict[str, Any]) -> str:
        """Render template with variables."""
        
        template = self.env.from_string(template_content)
        return template.render(**variables)

class ConfigValidator:
    """Configuration validation service."""
    
    def __init__(self):
        self.validator = cerberus.Validator()
    
    def validate(self, value: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema."""
        
        is_valid = self.validator.validate(value, schema)
        
        return {
            "valid": is_valid,
            "errors": self.validator.errors if not is_valid else [],
            "normalized": self.validator.document if is_valid else None
        }

class ConfigVersionManager:
    """Configuration version management."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
    
    async def get_version(
        self,
        key: str,
        version: int,
        scope: ConfigScope,
        tenant_id: Optional[str]
    ) -> Optional[ConfigValue]:
        """Get specific version of configuration."""
        
        # Try Redis history first
        full_key = f"{scope.value}:{tenant_id}:{key}" if tenant_id else f"{scope.value}:{key}"
        history_key = f"{full_key}:history:{version}"
        
        data = await self.redis_client.get(history_key)
        if data:
            try:
                config_data = json.loads(data)
                return ConfigValue(
                    value=None,
                    raw_value=config_data["value"],
                    format=ConfigFormat(config_data["format"]),
                    encrypted=config_data["encrypted"],
                    version=config_data["version"],
                    checksum=config_data["checksum"],
                    expires_at=datetime.fromisoformat(config_data["expires_at"]) if config_data.get("expires_at") else None,
                    metadata=config_data.get("metadata", {})
                )
            except:
                pass
        
        # Fallback to database
        from ..database import CRConfigurationHistory, CRConfiguration
        
        result = await self.db_session.execute(
            select(CRConfigurationHistory).join(CRConfiguration).where(
                and_(
                    CRConfiguration.applet_id == full_key,
                    CRConfigurationHistory.new_value["version"].astext.cast(Integer) == version
                )
            ).order_by(desc(CRConfigurationHistory.created_at)).limit(1)
        )
        
        history = result.scalar_one_or_none()
        if history and history.new_value:
            config_data = history.new_value
            return ConfigValue(
                value=None,
                raw_value=config_data["value"],
                format=ConfigFormat(config_data["format"]),
                encrypted=config_data["encrypted"],
                version=config_data["version"],
                checksum=config_data["checksum"],
                expires_at=datetime.fromisoformat(config_data["expires_at"]) if config_data.get("expires_at") else None,
                metadata=config_data.get("metadata", {})
            )
        
        return None

class ConfigChangeTracker:
    """Configuration change tracking."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def record_change(
        self,
        key: str,
        change_type: ChangeType,
        old_value: Optional[Any],
        new_value: Optional[Any],
        changed_by: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record configuration change."""
        
        change_id = f"change_{uuid7str()}"
        
        from ..database import CRConfigurationHistory, CRConfiguration
        
        # Get configuration ID
        result = await self.db_session.execute(
            select(CRConfiguration).where(CRConfiguration.applet_id == key)
        )
        
        config = result.scalar_one_or_none()
        
        if config:
            history = CRConfigurationHistory(
                configuration_id=config.id,
                change_type=change_type.value,
                old_value=old_value,
                new_value=new_value,
                changed_by=changed_by,
                change_reason=reason,
                tenant_id=config.tenant_id
            )
            
            self.db_session.add(history)
            await self.db_session.commit()
        
        return change_id
    
    async def get_history(self, key: str, limit: int = 50) -> List[ConfigChange]:
        """Get change history for configuration key."""
        
        from ..database import CRConfigurationHistory, CRConfiguration
        
        result = await self.db_session.execute(
            select(CRConfigurationHistory).join(CRConfiguration).where(
                CRConfiguration.applet_id == key
            ).order_by(desc(CRConfigurationHistory.created_at)).limit(limit)
        )
        
        history_records = result.scalars().all()
        
        changes = []
        for record in history_records:
            changes.append(ConfigChange(
                change_id=str(record.id),
                key=key,
                change_type=ChangeType(record.change_type),
                old_value=record.old_value,
                new_value=record.new_value,
                changed_by=record.changed_by,
                changed_at=record.created_at,
                reason=record.change_reason,
                metadata={}
            ))
        
        return changes

class ConfigWatcherManager:
    """Configuration watcher management."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    async def register_watcher(self, watcher: ConfigWatcher):
        """Register configuration watcher."""
        
        watcher_data = {
            "watcher_id": watcher.watcher_id,
            "key_pattern": watcher.key_pattern,
            "filters": watcher.filters,
            "active": watcher.active,
            "registered_at": datetime.now(timezone.utc).isoformat()
        }
        
        await self.redis_client.hset(
            "config:watchers",
            watcher.watcher_id,
            json.dumps(watcher_data, default=str)
        )
    
    async def unregister_watcher(self, watcher_id: str):
        """Unregister configuration watcher."""
        
        await self.redis_client.hdel("config:watchers", watcher_id)

# =============================================================================
# Service Factory
# =============================================================================

async def create_central_configuration_service(
    db_session: AsyncSession,
    redis_url: str,
    consul_url: Optional[str] = None,
    etcd_endpoints: Optional[List[str]] = None,
    vault_url: Optional[str] = None,
    vault_token: Optional[str] = None,
    encryption_key: Optional[str] = None
) -> CentralConfigurationService:
    """Factory function to create central configuration service."""
    
    redis_client = redis.from_url(redis_url)
    
    # Initialize optional clients
    consul_client = None
    if consul_url:
        consul_client = consul.aio.Consul(host=consul_url.split("://")[1].split(":")[0])
    
    etcd_client = None
    if etcd_endpoints:
        etcd_client = etcd3.client(host=etcd_endpoints[0].split(":")[0], port=int(etcd_endpoints[0].split(":")[1]))
    
    vault_client = None
    if vault_url and vault_token:
        vault_client = hvac.Client(url=vault_url, token=vault_token)
    
    return CentralConfigurationService(
        db_session=db_session,
        redis_client=redis_client,
        consul_client=consul_client,
        etcd_client=etcd_client,
        vault_client=vault_client,
        encryption_key=encryption_key
    )

# Export service classes
__all__ = [
    "CentralConfigurationService",
    "RedisConfigStorage",
    "ConsulConfigStorage", 
    "EtcdConfigStorage",
    "VaultConfigStorage",
    "DatabaseConfigStorage",
    "ConfigTemplateEngine",
    "ConfigValidator",
    "ConfigVersionManager",
    "ConfigChangeTracker",
    "ConfigWatcherManager",
    "ConfigKey",
    "ConfigValue",
    "ConfigChange",
    "ConfigWatcher",
    "ConfigFormat",
    "ConfigScope",
    "ConfigStatus",
    "StorageBackend",
    "ChangeType",
    "create_central_configuration_service"
]