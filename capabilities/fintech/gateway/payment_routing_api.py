#!/usr/bin/env python3
"""
Payment Routing Optimization API - APG Payment Gateway

RESTful API endpoints for intelligent payment routing management,
optimization strategies, and real-time performance monitoring.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from uuid_extensions import uuid7str
import json

from flask import Blueprint, request, jsonify, current_app
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import permission_name
from werkzeug.exceptions import BadRequest, Unauthorized, NotFound, InternalServerError

from .payment_routing_service import (
    PaymentRoutingService, PaymentRoutingContext, RoutingStrategy, 
    RoutingCriteria, ProcessorStatus, RoutingDecision,
    create_payment_routing_service
)
from .database import get_database_service
from .auth import get_auth_service, require_permission

# Create Flask Blueprint for payment routing endpoints
payment_routing_bp = Blueprint(
    'payment_routing',
    __name__,
    url_prefix='/api/v1/payment-routing'
)

class PaymentRoutingAPIView(BaseView):
    """
    Payment Routing Optimization API View
    
    Provides intelligent payment routing endpoints with real-time optimization,
    performance monitoring, and adaptive routing strategies.
    """
    
    route_base = "/api/v1/payment-routing"
    
    def __init__(self):
        super().__init__()
        self.routing_service: Optional[PaymentRoutingService] = None
        self.database_service = None
        self.auth_service = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure all services are initialized"""
        if not self._initialized:
            # Initialize database service
            self.database_service = get_database_service()
            await self.database_service.initialize()
            
            # Initialize authentication service
            self.auth_service = get_auth_service()
            
            # Initialize routing service
            self.routing_service = PaymentRoutingService(self.database_service)
            await self.routing_service.initialize()
            
            self._initialized = True
    
    # Core Routing Endpoints
    
    @expose('/optimal-processor', methods=['POST'])
    @has_access
    @permission_name('payment_routing')
    def get_optimal_processor(self):
        """
        Get optimal payment processor for transaction
        
        POST /api/v1/payment-routing/optimal-processor
        {
            "transaction_id": "txn_12345",
            "amount": 5000,
            "currency": "USD",
            "payment_method_type": "credit_card",
            "customer_id": "cust_123",
            "customer_country": "US",
            "merchant_id": "merch_456",
            "available_processors": ["stripe", "paypal", "adyen"],
            "strategy": "hybrid_optimal",
            "priority": "high",
            "max_processing_time_ms": 10000,
            "required_success_rate": 0.95
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['transaction_id', 'amount', 'currency', 'payment_method_type', 'merchant_id']
            for field in required_fields:
                if field not in data:
                    raise BadRequest(f"Missing required field: {field}")
            
            # Create routing context
            context = PaymentRoutingContext(
                transaction_id=data['transaction_id'],
                amount=int(data['amount']),
                currency=data['currency'],
                payment_method_type=data['payment_method_type'],
                customer_id=data.get('customer_id'),
                customer_country=data.get('customer_country'),
                merchant_id=data['merchant_id'],
                merchant_country=data.get('merchant_country'),
                merchant_category=data.get('merchant_category'),
                max_processing_time_ms=data.get('max_processing_time_ms', 30000),
                max_acceptable_cost=data.get('max_acceptable_cost'),
                required_success_rate=data.get('required_success_rate', 0.95),
                is_retry=data.get('is_retry', False),
                previous_processor=data.get('previous_processor'),
                priority=data.get('priority', 'normal'),
                metadata=data.get('metadata', {})
            )
            
            # Get available processors
            available_processors = data.get('available_processors', ['stripe', 'paypal', 'adyen', 'mpesa'])
            
            # Parse routing strategy
            strategy = None
            if 'strategy' in data:
                try:
                    strategy = RoutingStrategy(data['strategy'])
                except ValueError:
                    valid_strategies = [s.value for s in RoutingStrategy]
                    raise BadRequest(f"Invalid strategy. Must be one of: {valid_strategies}")
            
            # Get optimal processor
            decision = await self.routing_service.get_optimal_processor(
                context, available_processors, strategy
            )
            
            return jsonify({
                "success": True,
                "routing_decision": {
                    "processor_name": decision.processor_name,
                    "confidence_score": decision.confidence_score,
                    "routing_reasons": decision.routing_reasons,
                    "alternative_processors": decision.alternative_processors,
                    "estimated_success_probability": decision.estimated_success_probability,
                    "estimated_processing_time_ms": decision.estimated_processing_time_ms,
                    "estimated_cost": decision.estimated_cost,
                    "decision_timestamp": decision.decision_timestamp.isoformat()
                },
                "context_summary": {
                    "transaction_id": context.transaction_id,
                    "amount": context.amount,
                    "currency": context.currency,
                    "payment_method": context.payment_method_type,
                    "priority": context.priority
                }
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Optimal processor selection error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/recommendations', methods=['POST'])
    @has_access
    @permission_name('payment_routing')
    def get_processor_recommendations(self):
        """
        Get detailed processor recommendations for transaction
        
        POST /api/v1/payment-routing/recommendations
        {
            "transaction_id": "txn_12345",
            "amount": 5000,
            "currency": "USD",
            "payment_method_type": "credit_card",
            "merchant_id": "merch_456"
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            data = request.get_json()
            
            # Create routing context
            context = PaymentRoutingContext(
                transaction_id=data.get('transaction_id', uuid7str()),
                amount=int(data['amount']),
                currency=data['currency'],
                payment_method_type=data['payment_method_type'],
                customer_id=data.get('customer_id'),
                customer_country=data.get('customer_country'),
                merchant_id=data['merchant_id'],
                priority=data.get('priority', 'normal')
            )
            
            # Get recommendations
            recommendations = await self.routing_service.get_processor_recommendations(context)
            
            return jsonify({
                "success": True,
                "recommendations": recommendations
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Processor recommendations error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Performance Management Endpoints
    
    @expose('/performance/update', methods=['POST'])
    @has_access
    @permission_name('payment_routing_admin')
    def update_processor_performance(self):
        """
        Update processor performance metrics
        
        POST /api/v1/payment-routing/performance/update
        {
            "processor_name": "stripe",
            "transaction_result": {
                "success": true,
                "processing_time_ms": 1800,
                "cost": 1.45,
                "error_code": null,
                "response_status": 200
            }
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            data = request.get_json()
            
            # Validate required fields
            if 'processor_name' not in data or 'transaction_result' not in data:
                raise BadRequest("Missing required fields: processor_name, transaction_result")
            
            processor_name = data['processor_name']
            transaction_result = data['transaction_result']
            
            # Update performance
            await self.routing_service.update_processor_performance(
                processor_name, transaction_result
            )
            
            return jsonify({
                "success": True,
                "message": f"Performance updated for processor: {processor_name}",
                "processor": processor_name,
                "transaction_success": transaction_result.get('success', False)
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Performance update error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Analytics and Monitoring Endpoints
    
    @expose('/analytics', methods=['GET'])
    @has_access
    @permission_name('payment_routing_analytics')
    def get_routing_analytics(self):
        """
        Get comprehensive routing analytics
        
        GET /api/v1/payment-routing/analytics?period=24h
        """
        try:
            await self._ensure_initialized()
            
            # Parse query parameters
            time_period = request.args.get('period', '24h')
            
            # Validate time period
            valid_periods = ['1h', '24h', '7d', '30d']
            if time_period not in valid_periods:
                raise BadRequest(f"Invalid time period. Must be one of: {valid_periods}")
            
            # Get analytics
            analytics = await self.routing_service.get_routing_analytics(time_period)
            
            return jsonify({
                "success": True,
                "analytics": analytics
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Routing analytics error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/processors/status', methods=['GET'])
    @has_access
    @permission_name('payment_routing_monitor')
    def get_processors_status(self):
        """
        Get real-time status of all processors
        
        GET /api/v1/payment-routing/processors/status
        """
        try:
            await self._ensure_initialized()
            
            # Get processor metrics
            processor_status = {}
            
            for processor_name, metrics in self.routing_service._processor_metrics.items():
                circuit_breaker = self.routing_service._circuit_breakers.get(processor_name, {})
                
                processor_status[processor_name] = {
                    "status": "healthy" if metrics.is_healthy else "degraded",
                    "success_rate": metrics.success_rate,
                    "average_response_time_ms": metrics.average_response_time_ms,
                    "current_load_percentage": metrics.load_percentage,
                    "consecutive_failures": metrics.consecutive_failures,
                    "total_transactions": metrics.total_transactions,
                    "successful_transactions": metrics.successful_transactions,
                    "failed_transactions": metrics.failed_transactions,
                    "last_success_time": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                    "circuit_breaker_state": circuit_breaker.get('state', 'closed'),
                    "estimated_cost_per_transaction": metrics.cost_per_transaction
                }
            
            return jsonify({
                "success": True,
                "processors": processor_status,
                "total_processors": len(processor_status),
                "healthy_processors": sum(1 for p in processor_status.values() if p["status"] == "healthy"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            current_app.logger.error(f"Processor status error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/processors/<processor_name>/metrics', methods=['GET'])
    @has_access
    @permission_name('payment_routing_monitor')
    def get_processor_metrics(self, processor_name: str):
        """
        Get detailed metrics for specific processor
        
        GET /api/v1/payment-routing/processors/{processor_name}/metrics
        """
        try:
            await self._ensure_initialized()
            
            if processor_name not in self.routing_service._processor_metrics:
                return jsonify({"error": "Processor not found"}), 404
            
            metrics = self.routing_service._processor_metrics[processor_name]
            circuit_breaker = self.routing_service._circuit_breakers.get(processor_name, {})
            
            # Get performance trends if available
            trends = self.routing_service._performance_trends.get(processor_name, [])
            recent_trends = list(trends)[-20:]  # Last 20 data points
            
            return jsonify({
                "processor_name": processor_name,
                "current_metrics": {
                    "success_rate": metrics.success_rate,
                    "average_response_time_ms": metrics.average_response_time_ms,
                    "current_load": metrics.current_load,
                    "max_capacity": metrics.max_capacity,
                    "load_percentage": metrics.load_percentage,
                    "error_rate": metrics.error_rate,
                    "cost_per_transaction": metrics.cost_per_transaction,
                    "consecutive_failures": metrics.consecutive_failures,
                    "total_transactions": metrics.total_transactions,
                    "successful_transactions": metrics.successful_transactions,
                    "failed_transactions": metrics.failed_transactions,
                    "is_healthy": metrics.is_healthy,
                    "last_success_time": metrics.last_success_time.isoformat() if metrics.last_success_time else None
                },
                "circuit_breaker": {
                    "state": circuit_breaker.get('state', 'closed'),
                    "failure_count": circuit_breaker.get('failure_count', 0),
                    "open_until": circuit_breaker.get('open_until').isoformat() if circuit_breaker.get('open_until') else None
                },
                "recent_trends": [
                    {
                        "timestamp": trend.get('timestamp').isoformat() if trend.get('timestamp') else None,
                        "success": trend.get('success', False),
                        "response_time": trend.get('response_time', 0),
                        "cost": trend.get('cost', 0.0)
                    }
                    for trend in recent_trends
                ]
            })
            
        except Exception as e:
            current_app.logger.error(f"Processor metrics error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Configuration Endpoints
    
    @expose('/strategies', methods=['GET'])
    @has_access
    @permission_name('payment_routing')
    def list_routing_strategies(self):
        """
        List all available routing strategies
        
        GET /api/v1/payment-routing/strategies
        """
        try:
            strategies = []
            for strategy in RoutingStrategy:
                description = self._get_strategy_description(strategy)
                strategies.append({
                    "strategy": strategy.value,
                    "description": description,
                    "use_cases": self._get_strategy_use_cases(strategy)
                })
            
            return jsonify({
                "strategies": strategies,
                "total_strategies": len(strategies),
                "default_strategy": self.routing_service.default_strategy.value if self.routing_service else "hybrid_optimal"
            })
            
        except Exception as e:
            current_app.logger.error(f"Routing strategies error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/criteria', methods=['GET'])
    @has_access
    @permission_name('payment_routing')
    def list_routing_criteria(self):
        """
        List all available routing criteria
        
        GET /api/v1/payment-routing/criteria
        """
        try:
            criteria = []
            for criterion in RoutingCriteria:
                description = self._get_criteria_description(criterion)
                criteria.append({
                    "criterion": criterion.value,
                    "description": description,
                    "weight_in_hybrid": self._get_criteria_weight(criterion)
                })
            
            return jsonify({
                "criteria": criteria,
                "total_criteria": len(criteria)
            })
            
        except Exception as e:
            current_app.logger.error(f"Routing criteria error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Testing and Simulation Endpoints
    
    @expose('/simulate', methods=['POST'])
    @has_access
    @permission_name('payment_routing_admin')
    def simulate_routing(self):
        """
        Simulate routing decisions for testing
        
        POST /api/v1/payment-routing/simulate
        {
            "scenarios": [
                {
                    "transaction_id": "sim_001",
                    "amount": 5000,
                    "currency": "USD",
                    "payment_method_type": "credit_card",
                    "merchant_id": "merch_001"
                }
            ],
            "strategies_to_test": ["round_robin", "best_success_rate", "hybrid_optimal"],
            "available_processors": ["stripe", "paypal", "adyen"]
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            data = request.get_json()
            
            scenarios = data.get('scenarios', [])
            strategies_to_test = data.get('strategies_to_test', ['hybrid_optimal'])
            available_processors = data.get('available_processors', ['stripe', 'paypal', 'adyen'])
            
            if not scenarios:
                raise BadRequest("At least one scenario is required")
            
            simulation_results = []
            
            for scenario in scenarios:
                # Create context
                context = PaymentRoutingContext(
                    transaction_id=scenario.get('transaction_id', uuid7str()),
                    amount=int(scenario['amount']),
                    currency=scenario['currency'],
                    payment_method_type=scenario['payment_method_type'],
                    customer_id=scenario.get('customer_id'),
                    customer_country=scenario.get('customer_country'),
                    merchant_id=scenario['merchant_id'],
                    priority=scenario.get('priority', 'normal')
                )
                
                scenario_results = {
                    "scenario": scenario,
                    "strategy_results": {}
                }
                
                # Test each strategy
                for strategy_name in strategies_to_test:
                    try:
                        strategy = RoutingStrategy(strategy_name)
                        decision = await self.routing_service.get_optimal_processor(
                            context, available_processors, strategy
                        )
                        
                        scenario_results["strategy_results"][strategy_name] = {
                            "processor_selected": decision.processor_name,
                            "confidence_score": decision.confidence_score,
                            "routing_reasons": decision.routing_reasons,
                            "estimated_success_probability": decision.estimated_success_probability,
                            "estimated_processing_time_ms": decision.estimated_processing_time_ms,
                            "estimated_cost": decision.estimated_cost
                        }
                    except ValueError:
                        scenario_results["strategy_results"][strategy_name] = {
                            "error": f"Invalid strategy: {strategy_name}"
                        }
                
                simulation_results.append(scenario_results)
            
            return jsonify({
                "success": True,
                "simulation_results": simulation_results,
                "total_scenarios": len(scenarios),
                "strategies_tested": strategies_to_test,
                "available_processors": available_processors
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Routing simulation error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/health', methods=['GET'])
    def get_routing_service_health(self):
        """
        Get routing service health status
        
        GET /api/v1/payment-routing/health
        """
        try:
            await self._ensure_initialized()
            
            # Get service health
            total_processors = len(self.routing_service._processor_metrics)
            healthy_processors = sum(
                1 for metrics in self.routing_service._processor_metrics.values() 
                if metrics.is_healthy
            )
            
            circuit_breakers_open = sum(
                1 for cb in self.routing_service._circuit_breakers.values()
                if cb.get('state') == 'open'
            )
            
            recent_decisions = len([
                record for record in self.routing_service._routing_history
                if record['timestamp'] > datetime.now(timezone.utc) - timedelta(hours=1)
            ])
            
            return jsonify({
                "status": "healthy" if self.routing_service._initialized else "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "metrics": {
                    "total_processors": total_processors,
                    "healthy_processors": healthy_processors,
                    "unhealthy_processors": total_processors - healthy_processors,
                    "circuit_breakers_open": circuit_breakers_open,
                    "recent_decisions_1h": recent_decisions,
                    "default_strategy": self.routing_service.default_strategy.value
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Routing service health check error: {str(e)}")
            return jsonify({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 500
    
    # Helper methods
    
    def _get_strategy_description(self, strategy: RoutingStrategy) -> str:
        """Get human-readable description for routing strategy"""
        descriptions = {
            RoutingStrategy.ROUND_ROBIN: "Simple round-robin selection across processors",
            RoutingStrategy.WEIGHTED_ROUND_ROBIN: "Round-robin with processor performance weighting",
            RoutingStrategy.LEAST_LOADED: "Route to processor with lowest current load",
            RoutingStrategy.BEST_SUCCESS_RATE: "Route to processor with highest success rate",
            RoutingStrategy.LOWEST_COST: "Route to processor with lowest transaction cost",
            RoutingStrategy.LOWEST_LATENCY: "Route to processor with fastest response time",
            RoutingStrategy.GEOGRAPHIC_OPTIMAL: "Route based on geographic proximity and compliance",
            RoutingStrategy.PAYMENT_METHOD_OPTIMAL: "Route to processor optimized for payment method",
            RoutingStrategy.ML_OPTIMIZED: "Machine learning based optimal routing",
            RoutingStrategy.ADAPTIVE_LEARNING: "Self-learning routing based on historical performance",
            RoutingStrategy.HYBRID_OPTIMAL: "Multi-criteria optimization with intelligent weighting"
        }
        return descriptions.get(strategy, "No description available")
    
    def _get_strategy_use_cases(self, strategy: RoutingStrategy) -> List[str]:
        """Get use cases for routing strategy"""
        use_cases = {
            RoutingStrategy.ROUND_ROBIN: ["Load balancing", "Equal distribution", "Simple failover"],
            RoutingStrategy.BEST_SUCCESS_RATE: ["High-value transactions", "Critical payments", "Minimizing failures"],
            RoutingStrategy.LOWEST_COST: ["Cost optimization", "High-volume low-margin transactions"],
            RoutingStrategy.LOWEST_LATENCY: ["Real-time payments", "User experience optimization"],
            RoutingStrategy.HYBRID_OPTIMAL: ["General purpose", "Balanced optimization", "Production recommended"]
        }
        return use_cases.get(strategy, ["General purpose"])
    
    def _get_criteria_description(self, criterion: RoutingCriteria) -> str:
        """Get description for routing criterion"""
        descriptions = {
            RoutingCriteria.SUCCESS_RATE: "Percentage of successful transactions",
            RoutingCriteria.PROCESSING_TIME: "Average transaction processing time", 
            RoutingCriteria.TRANSACTION_COST: "Cost per transaction including fees",
            RoutingCriteria.GEOGRAPHICAL_MATCH: "Geographic proximity and compliance",
            RoutingCriteria.PAYMENT_METHOD_COMPATIBILITY: "Payment method support and optimization",
            RoutingCriteria.PROCESSOR_LOAD: "Current processor load and capacity",
            RoutingCriteria.HISTORICAL_PERFORMANCE: "Long-term performance trends",
            RoutingCriteria.CURRENCY_SUPPORT: "Native currency support and conversion rates",
            RoutingCriteria.FRAUD_RISK_SCORE: "Fraud detection and risk assessment capabilities"
        }
        return descriptions.get(criterion, "No description available")
    
    def _get_criteria_weight(self, criterion: RoutingCriteria) -> float:
        """Get weight of criterion in hybrid strategy"""
        weights = {
            RoutingCriteria.SUCCESS_RATE: 0.40,
            RoutingCriteria.PROCESSING_TIME: 0.25,
            RoutingCriteria.TRANSACTION_COST: 0.20,
            RoutingCriteria.PROCESSOR_LOAD: 0.15
        }
        return weights.get(criterion, 0.0)

# Register Flask-AppBuilder view
def register_payment_routing_views(appbuilder):
    """Register payment routing views with Flask-AppBuilder"""
    appbuilder.add_view_no_menu(PaymentRoutingAPIView)

# Module initialization logging
def _log_payment_routing_api_module_loaded():
    """Log payment routing API module loaded"""
    print("🎯 Payment Routing Optimization API module loaded")
    print("   - Intelligent processor selection")
    print("   - Multi-strategy routing")
    print("   - Real-time performance monitoring")
    print("   - Analytics and insights")
    print("   - Simulation and testing")

# Execute module loading log
_log_payment_routing_api_module_loaded()