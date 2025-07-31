#!/usr/bin/env python3
"""
Quick test script for real-time analytics core functionality
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
from collections import deque


class MetricType(str, Enum):
    """Analytics metric types"""
    TRANSACTION_VOLUME = "transaction_volume"
    SUCCESS_RATE = "success_rate"
    FRAUD_RATE = "fraud_rate"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RealTimeMetric:
    """Real-time metric data point"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]
    merchant_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "merchant_id": self.merchant_id
        }


class SimpleAnalyticsEngine:
    """Simplified analytics engine for testing"""
    
    def __init__(self):
        self._metrics_buffer = deque(maxlen=1000)
        self._running = False
    
    async def start(self):
        self._running = True
        print("📊 Analytics engine started")
    
    async def stop(self):
        self._running = False
        print("🛑 Analytics engine stopped")
    
    async def record_metric(self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        """Record a metric"""
        metric = RealTimeMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        self._metrics_buffer.append(metric)
        print(f"📈 Recorded {metric_type.value}: {value}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self._metrics_buffer:
            return {"total_metrics": 0}
        
        by_type = {}
        for metric in self._metrics_buffer:
            metric_type = metric.metric_type.value
            if metric_type not in by_type:
                by_type[metric_type] = []
            by_type[metric_type].append(metric.value)
        
        summary = {"total_metrics": len(self._metrics_buffer)}
        for metric_type, values in by_type.items():
            summary[metric_type] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        return summary


async def test_analytics_engine():
    """Test the analytics engine functionality"""
    print("🧪 Testing Real-Time Analytics Engine")
    print("=" * 50)
    
    engine = SimpleAnalyticsEngine()
    await engine.start()
    
    # Record some test metrics
    await engine.record_metric(MetricType.TRANSACTION_VOLUME, 1.0, {"transaction_id": "txn_1"})
    await engine.record_metric(MetricType.TRANSACTION_VOLUME, 1.0, {"transaction_id": "txn_2"})
    await engine.record_metric(MetricType.SUCCESS_RATE, 1.0, {"processor": "stripe"})
    await engine.record_metric(MetricType.SUCCESS_RATE, 0.0, {"processor": "mpesa"})
    await engine.record_metric(MetricType.FRAUD_RATE, 0.0, {"transaction_id": "txn_1"})
    await engine.record_metric(MetricType.FRAUD_RATE, 1.0, {"transaction_id": "txn_3"})
    
    # Get summary
    summary = engine.get_metrics_summary()
    print("\n📊 Metrics Summary:")
    print(f"   Total metrics: {summary['total_metrics']}")
    
    for metric_type, stats in summary.items():
        if metric_type != "total_metrics":
            print(f"   {metric_type}:")
            print(f"     Count: {stats['count']}")
            print(f"     Average: {stats['avg']:.2f}")
            print(f"     Range: {stats['min']:.2f} - {stats['max']:.2f}")
    
    await engine.stop()
    print("\n✅ Analytics engine test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_analytics_engine())