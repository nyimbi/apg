"""
Service Bridge for APG Connection Management
Bridges async service layer with sync Flask-AppBuilder views

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import inspect
import threading
from typing import Any, Dict, List, Optional
from functools import wraps
from datetime import datetime, timezone
import logging

from .service import ConnectionManager, FlowExecutor, IntelligentConnector
from .sqlalchemy_models import CnConnection, CnDataFlow, CnLineageNode, CnLineageEdge
from .lineage_engine import lineage_engine
from .error_handling import (
    APGError, ErrorHandler, ErrorContext, handle_error,
    ConnectionError, ValidationError, ResourceError
)

logger = logging.getLogger(__name__)


class ServiceBridge:
    """Bridge between async services and sync Flask-AppBuilder views"""

    def __init__(self):
        self._connection_manager = None
        self._flow_executor = None
        self._intelligent_connector = None
        self._loop = None
        self._thread = None
        self._initialized = False

    def _get_event_loop(self):
        """Get or create event loop for async operations"""
        if self._loop is None or self._loop.is_closed():
            try:
                # Try to get existing loop
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                # Create new loop if none exists
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def run_async(self, coro):
        """Run async coroutine in sync context with error handling"""
        if not inspect.isawaitable(coro):
            return coro
        loop = self._get_event_loop()
        try:
            if loop.is_running() is True:
                # If loop is running, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except APGError as apg_error:
            # Log APG errors and re-raise for proper handling
            logger.error(f"APG Error in async operation: {apg_error.to_dict()}")
            raise
        except Exception as e:
            # Preserve the original service exception so callers can handle the
            # same contract in async and sync contexts.
            logger.error(f"Unexpected error in async operation: {e}")
            raise

    @property
    def connection_manager(self) -> ConnectionManager:
        """Get connection manager instance"""
        if self._connection_manager is None:
            self._connection_manager = ConnectionManager()
            # Initialize in sync context
            self.run_async(self._connection_manager.initialize())
        return self._connection_manager

    @property
    def flow_executor(self) -> FlowExecutor:
        """Get flow executor instance"""
        if self._flow_executor is None:
            self._flow_executor = FlowExecutor(connection_manager=self.connection_manager)
        return self._flow_executor

    @property
    def intelligent_connector(self) -> IntelligentConnector:
        """Get intelligent connector instance"""
        if self._intelligent_connector is None:
            self._intelligent_connector = IntelligentConnector()
        return self._intelligent_connector

    # Connection Management Methods
    def create_connection(self, connection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new connection"""
        try:
            connection = self.run_async(
                self.connection_manager.create_connection(connection_data)
            )
            return {
                'success': True,
                'connection_id': connection.id,
                'status': connection.status.value,
                'message': f'Connection "{connection.name}" created successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to create connection: {str(e)}'
            }

    def test_connection(self, connection_id: str) -> Dict[str, Any]:
        """Test connection"""
        try:
            result = self.run_async(
                self.connection_manager.test_connection_sync(connection_id)
            )
            return {
                'success': result.get('status') == 'success',
                'result': result,
                'message': 'Connection test completed'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Connection test failed: {str(e)}'
            }

    def get_connection_health(self, connection_id: str) -> Dict[str, Any]:
        """Get connection health status"""
        try:
            health = self.run_async(
                self.connection_manager.get_connection_health(connection_id)
            )
            if health:
                return {
                    'success': True,
                    'health': {
                        'status': health.status.value,
                        'latency_ms': health.latency_ms,
                        'throughput': health.throughput_records_per_sec,
                        'error_rate': health.error_rate,
                        'is_healthy': health.is_healthy(),
                        'timestamp': health.timestamp.isoformat() if health.timestamp else None
                    }
                }
            else:
                return {
                    'success': False,
                    'message': 'Health data not available'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to get health status: {str(e)}'
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            metrics = self.run_async(
                self.connection_manager.get_performance_metrics()
            )
            return {
                'success': True,
                'metrics': metrics
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }

    # Flow Management Methods
    def create_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new data flow"""
        try:
            flow = self.run_async(
                self.flow_executor.create_flow(flow_data)
            )
            return {
                'success': True,
                'flow_id': flow.id,
                'message': f'Flow "{flow.name}" created successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to create flow: {str(e)}'
            }

    def execute_flow(self, flow_id: str) -> Dict[str, Any]:
        """Execute flow once"""
        try:
            result = self.run_async(
                self.flow_executor.execute_flow_once(flow_id)
            )
            return {
                'success': result.get('status') == 'success',
                'result': result,
                'message': 'Flow execution completed'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Flow execution failed: {str(e)}'
            }

    # Lineage Methods
    def discover_lineage(self, connection_id: str) -> Dict[str, Any]:
        """Discover data lineage for connection"""
        try:
            connection = self.connection_manager.get_connection(connection_id)
            if asyncio.iscoroutine(connection):
                connection = self.run_async(connection)
            if connection:
                result = self.run_async(
                    lineage_engine.discover_connection_schema(connection)
                )

                return {
                    'success': True,
                    'discovery_result': result,
                    'message': f'Discovered {result.get("nodes_created", 0)} lineage nodes'
                }
            else:
                return {
                    'success': False,
                    'message': 'Connection not found'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Lineage discovery failed: {str(e)}'
            }

    def get_lineage_visualization(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Get lineage visualization data"""
        try:
            # Extract parameters
            node_id = params.get('node_id') if params else None
            visualization_type = params.get('type', 'full') if params else 'full'
            max_depth = params.get('max_depth', 10) if params else 10

            # Get visualization data from lineage engine
            lineage_data = lineage_engine.get_lineage_visualization(
                node_id=node_id,
                visualization_type=visualization_type,
                max_depth=max_depth
            )

            return {
                'success': True,
                'lineage_data': lineage_data
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'lineage_data': {'nodes': [], 'edges': [], 'summary': {}}
            }

    # AI and Intelligence Methods
    def suggest_field_mappings(self, source_schema: Dict, target_schema: Dict) -> Dict[str, Any]:
        """Get AI-powered field mapping suggestions"""
        try:
            suggestions = self.run_async(
                self.intelligent_connector.suggest_field_mappings(
                    source_schema, target_schema
                )
            )
            return {
                'success': True,
                'suggestions': suggestions,
                'message': 'Field mapping suggestions generated'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'suggestions': [],
                'message': f'Failed to generate suggestions: {str(e)}'
            }

    def predict_performance(self, connection_config: Dict) -> Dict[str, Any]:
        """Predict connection performance"""
        try:
            prediction = self.run_async(
                self.intelligent_connector.predict_performance(connection_config)
            )
            return {
                'success': True,
                'prediction': prediction,
                'message': 'Performance prediction completed'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': {},
                'message': f'Performance prediction failed: {str(e)}'
            }


# Global service bridge instance
service_bridge = ServiceBridge()


def with_service_bridge(f):
    """Decorator to inject service bridge into view methods"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Add service_bridge to kwargs unless a caller/test supplied one.
        kwargs.setdefault('service_bridge', service_bridge)
        return f(*args, **kwargs)
    return decorated_function
