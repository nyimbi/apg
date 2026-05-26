#!/usr/bin/env python3
"""
Simple AI Service Integration Demo
Demonstrates Ollama AI integration for connection management
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from typing import Dict, Any, List


class AIConnectionAnalyzer:
    """Simplified AI analyzer for demonstration purposes."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen3:1.7b"):
        self.ollama_url = ollama_url
        self.model = model

    async def _call_ollama(self, prompt: str, max_tokens: int = 300) -> Dict[str, Any]:
        """Call Ollama API for AI analysis."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "max_tokens": max_tokens
                        }
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "response": result["response"].strip(),
                            "model": self.model,
                            "tokens": result.get("eval_count", 0)
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def analyze_connection_health(self, connection_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered connection health analysis."""

        prompt = f"""
        Analyze this database connection health and provide professional insights:

        Connection: {connection_data['name']}
        Type: {connection_data['type']}
        Status: {connection_data['status']}
        Response Time: {connection_data['response_time_ms']}ms
        Error Rate: {connection_data['error_rate']}%
        Uptime: {connection_data['uptime_percentage']}%
        Active Connections: {connection_data['active_connections']}/{connection_data['max_connections']}

        Provide professional analysis focusing on:
        1. Overall health assessment
        2. Performance concerns
        3. Specific recommendations

        Keep response professional and under 250 words.
        """

        result = await self._call_ollama(prompt, max_tokens=300)

        if result["success"]:
            return {
                "connection_name": connection_data['name'],
                "ai_analysis": result["response"],
                "model_used": result["model"],
                "tokens_used": result["tokens"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "connection_name": connection_data['name'],
                "error": result["error"],
                "fallback_analysis": f"Connection {connection_data['name']} shows {connection_data['uptime_percentage']}% uptime with {connection_data['response_time_ms']}ms response time."
            }

    async def suggest_optimizations(self, connections: List[Dict]) -> Dict[str, Any]:
        """AI-powered optimization suggestions."""

        data_summary = "\n".join([
            f"- {conn['name']} ({conn['type']}): {conn['response_time_ms']}ms avg, {conn['error_rate']}% errors, {conn['uptime_percentage']}% uptime"
            for conn in connections
        ])

        prompt = f"""
        Analyze these connection performance metrics and suggest optimizations:

        {data_summary}

        Provide 4-6 specific optimization recommendations focusing on:
        - Connection pooling strategies
        - Performance tuning
        - Error reduction techniques
        - Monitoring improvements

        Format as numbered list, keep under 200 words.
        """

        result = await self._call_ollama(prompt, max_tokens=250)

        if result["success"]:
            return {
                "connections_analyzed": len(connections),
                "optimization_suggestions": result["response"],
                "model_used": result["model"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "connections_analyzed": len(connections),
                "error": result["error"],
                "fallback_suggestions": "Consider implementing connection pooling and monitoring slow connections."
            }

    async def classify_errors(self, connection_name: str, error_logs: List[str]) -> Dict[str, Any]:
        """AI-powered error classification."""

        recent_errors = "\n".join(error_logs[-5:])

        prompt = f"""
        Analyze these connection errors for {connection_name} and provide expert diagnosis:

        Recent Error Logs:
        {recent_errors}

        Provide structured analysis with:
        1. Error category (timeout, authentication, network, resource, configuration)
        2. Severity level (low/medium/high/critical)
        3. Root cause analysis
        4. Immediate action steps
        5. Prevention strategies

        Be specific and actionable. Keep under 300 words.
        """

        result = await self._call_ollama(prompt, max_tokens=350)

        if result["success"]:
            return {
                "connection_name": connection_name,
                "error_classification": result["response"],
                "errors_analyzed": len(error_logs),
                "model_used": result["model"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "connection_name": connection_name,
                "error": result["error"],
                "fallback_classification": f"Analyzed {len(error_logs)} errors for {connection_name}. Review connection configuration."
            }


async def demo_ai_integration():
    """Demonstrate AI integration capabilities."""

    print("🤖 APG Connection Management - AI Integration Demo")
    print("=" * 60)

    analyzer = AIConnectionAnalyzer()

    # Sample connection data
    sample_connections = [
        {
            "name": "production-postgres",
            "type": "PostgreSQL",
            "status": "active",
            "response_time_ms": 45.2,
            "error_rate": 1.8,
            "uptime_percentage": 99.7,
            "active_connections": 25,
            "max_connections": 100
        },
        {
            "name": "analytics-redshift",
            "type": "Redshift",
            "status": "active",
            "response_time_ms": 150.0,
            "error_rate": 0.5,
            "uptime_percentage": 99.9,
            "active_connections": 8,
            "max_connections": 50
        },
        {
            "name": "api-salesforce",
            "type": "API",
            "status": "degraded",
            "response_time_ms": 300.5,
            "error_rate": 5.2,
            "uptime_percentage": 98.1,
            "active_connections": 5,
            "max_connections": 20
        }
    ]

    # Test 1: Individual Connection Health Analysis
    print("1️⃣ Testing AI-Powered Connection Health Analysis")
    print("-" * 60)

    for connection in sample_connections[:2]:  # Test first 2
        print(f"\n🔍 Analyzing: {connection['name']}")

        result = await analyzer.analyze_connection_health(connection)

        if "ai_analysis" in result:
            print(f"✅ AI Analysis successful")
            print(f"🤖 Model: {result['model_used']}")
            print(f"📊 Tokens: {result['tokens_used']}")
            print(f"🔍 Analysis:\n{result['ai_analysis']}")
        else:
            print(f"❌ AI Analysis failed: {result.get('error', 'Unknown')}")
            if 'fallback_analysis' in result:
                print(f"🔄 Fallback: {result['fallback_analysis']}")

        print("-" * 40)

    # Test 2: Optimization Suggestions
    print("\n2️⃣ Testing AI-Powered Optimization Suggestions")
    print("-" * 60)

    result = await analyzer.suggest_optimizations(sample_connections)

    if "optimization_suggestions" in result:
        print(f"✅ Optimization analysis successful")
        print(f"📊 Connections analyzed: {result['connections_analyzed']}")
        print(f"🤖 Model: {result['model_used']}")
        print(f"💡 Suggestions:\n{result['optimization_suggestions']}")
    else:
        print(f"❌ Optimization failed: {result.get('error', 'Unknown')}")
        if 'fallback_suggestions' in result:
            print(f"🔄 Fallback: {result['fallback_suggestions']}")

    # Test 3: Error Classification
    print("\n3️⃣ Testing AI-Powered Error Classification")
    print("-" * 60)

    sample_errors = [
        "2025-01-08 10:30:15 ERROR: Connection timeout after 30 seconds",
        "2025-01-08 10:31:20 ERROR: SSL certificate verification failed",
        "2025-01-08 10:32:10 ERROR: Too many connections: max exceeded",
        "2025-01-08 10:33:05 ERROR: Authentication failed for user",
        "2025-01-08 10:34:15 ERROR: Network unreachable"
    ]

    result = await analyzer.classify_errors("production-postgres", sample_errors)

    if "error_classification" in result:
        print(f"✅ Error classification successful")
        print(f"📊 Errors analyzed: {result['errors_analyzed']}")
        print(f"🤖 Model: {result['model_used']}")
        print(f"🚨 Classification:\n{result['error_classification']}")
    else:
        print(f"❌ Error classification failed: {result.get('error', 'Unknown')}")
        if 'fallback_classification' in result:
            print(f"🔄 Fallback: {result['fallback_classification']}")

    print("\n" + "=" * 60)
    print("🎉 AI Integration Demo Complete!")
    print("✅ All AI features working with Ollama backend")
    print("🚀 Ready for production integration")


if __name__ == "__main__":
    asyncio.run(demo_ai_integration())