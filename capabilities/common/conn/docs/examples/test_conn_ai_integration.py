#!/usr/bin/env python3
"""
Test Connection Management AI Integration with Ollama
Demonstrates how Ollama can be used for connection analysis and insights
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionAIAnalyzer:
    """AI-powered connection analysis using Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen3:1.7b"  # Fast model for analysis

    async def analyze_connection_health(self, connection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze connection health using AI"""

        prompt = f"""
        Analyze this database connection health data and provide insights:

        Connection: {connection_data['name']}
        Type: {connection_data['type']}
        Status: {connection_data['status']}
        Response Time: {connection_data['response_time_ms']}ms
        Error Rate: {connection_data['error_rate']}%
        Last Connection: {connection_data['last_connection']}

        Provide a brief analysis focusing on:
        1. Overall health assessment
        2. Performance concerns
        3. Recommended actions

        Keep response under 200 words.
        """

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 200}
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "analysis": result["response"].strip(),
                        "model_used": self.model,
                        "tokens": result.get("eval_count", 0)
                    }
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}

    async def suggest_connection_optimization(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Suggest connection optimizations based on performance data"""

        data_summary = "\n".join([
            f"- {conn['name']}: {conn['avg_response_time']}ms avg, {conn['peak_connections']} peak connections"
            for conn in performance_data
        ])

        prompt = f"""
        Based on this connection performance data, suggest optimizations:

        {data_summary}

        Provide 3-5 specific optimization recommendations focusing on:
        - Connection pooling
        - Query optimization
        - Resource allocation
        - Monitoring improvements

        Format as numbered list, keep under 150 words.
        """

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "max_tokens": 200}
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "recommendations": result["response"].strip(),
                        "model_used": self.model
                    }
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}

    async def classify_connection_issues(self, error_logs: List[str]) -> Dict[str, Any]:
        """Classify connection issues from error logs"""

        logs_text = "\n".join(error_logs[:5])  # Limit to 5 most recent

        prompt = f"""
        Classify these database connection errors and suggest solutions:

        Error Logs:
        {logs_text}

        Provide:
        1. Error category (e.g., timeout, authentication, network, resource)
        2. Severity level (low/medium/high/critical)
        3. Root cause analysis
        4. Immediate action needed

        Be concise and actionable.
        """

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "max_tokens": 250}
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "classification": result["response"].strip(),
                        "model_used": self.model
                    }
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}


async def test_connection_ai_features():
    """Test AI features for connection management"""

    logger.info("🤖 Testing Connection Management AI Integration")
    logger.info("=" * 60)

    analyzer = ConnectionAIAnalyzer()

    # Test 1: Connection Health Analysis
    logger.info("1️⃣ Testing Connection Health Analysis...")

    sample_connection = {
        "name": "production-postgres-01",
        "type": "PostgreSQL",
        "status": "connected",
        "response_time_ms": 45,
        "error_rate": 2.3,
        "last_connection": "2025-01-08 10:30:00"
    }

    health_result = await analyzer.analyze_connection_health(sample_connection)

    if health_result["success"]:
        logger.info("✅ Health Analysis Successful")
        logger.info(f"📊 Model: {health_result['model_used']}, Tokens: {health_result['tokens']}")
        logger.info(f"🔍 Analysis:\n{health_result['analysis']}")
    else:
        logger.error(f"❌ Health Analysis Failed: {health_result.get('error', 'Unknown')}")

    logger.info("-" * 60)

    # Test 2: Optimization Suggestions
    logger.info("2️⃣ Testing Optimization Suggestions...")

    sample_performance = [
        {"name": "postgres-main", "avg_response_time": 120, "peak_connections": 250},
        {"name": "redis-cache", "avg_response_time": 15, "peak_connections": 1000},
        {"name": "mysql-analytics", "avg_response_time": 300, "peak_connections": 50}
    ]

    optimization_result = await analyzer.suggest_connection_optimization(sample_performance)

    if optimization_result["success"]:
        logger.info("✅ Optimization Suggestions Successful")
        logger.info(f"📊 Model: {optimization_result['model_used']}")
        logger.info(f"💡 Recommendations:\n{optimization_result['recommendations']}")
    else:
        logger.error(f"❌ Optimization Failed: {optimization_result.get('error', 'Unknown')}")

    logger.info("-" * 60)

    # Test 3: Error Classification
    logger.info("3️⃣ Testing Error Classification...")

    sample_errors = [
        "Connection timeout after 30 seconds to postgres://prod-db:5432",
        "SSL handshake failed: certificate verification failed",
        "Too many connections: max_connections=100 exceeded",
        "Authentication failed for user 'app_user'",
        "Network unreachable: Connection refused"
    ]

    classification_result = await analyzer.classify_connection_issues(sample_errors)

    if classification_result["success"]:
        logger.info("✅ Error Classification Successful")
        logger.info(f"📊 Model: {classification_result['model_used']}")
        logger.info(f"🚨 Classification:\n{classification_result['classification']}")
    else:
        logger.error(f"❌ Classification Failed: {classification_result.get('error', 'Unknown')}")

    logger.info("=" * 60)
    logger.info("🎉 Connection Management AI Integration Test Completed!")


if __name__ == "__main__":
    asyncio.run(test_connection_ai_features())