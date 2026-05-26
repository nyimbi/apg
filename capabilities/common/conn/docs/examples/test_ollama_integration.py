#!/usr/bin/env python3
"""
Test Ollama Integration
Test if Ollama is running and can generate responses
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Simple Ollama client for testing"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def health_check(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with self.session.get(f"{self.base_url}/api/version") as response:
                if response.status == 200:
                    version_data = await response.json()
                    logger.info(f"✅ Ollama is running - Version: {version_data}")
                    return True
                else:
                    logger.error(f"❌ Ollama health check failed - Status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    logger.info(f"📋 Available models: {len(models)} found")
                    for model in models:
                        logger.info(f"   - {model['name']} (Size: {model.get('size', 'Unknown')})")
                    return models
                else:
                    logger.error(f"❌ Failed to list models - Status: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return []

    async def generate(self, model: str, prompt: str) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 100
                }
            }

            logger.info(f"🤖 Generating response with model: {model}")
            logger.info(f"📝 Prompt: {prompt}")

            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('response', '').strip()
                    logger.info(f"✅ Response: {response_text}")

                    # Extract stats if available
                    if 'eval_count' in result:
                        logger.info(f"📊 Stats - Tokens: {result.get('eval_count', 0)}, Time: {result.get('total_duration', 0)/1e9:.2f}s")

                    return {
                        'success': True,
                        'response': response_text,
                        'model': model,
                        'stats': {
                            'tokens': result.get('eval_count', 0),
                            'duration': result.get('total_duration', 0),
                            'load_duration': result.get('load_duration', 0)
                        }
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Generation failed - Status: {response.status}, Error: {error_text}")
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


async def test_ollama_connection():
    """Test Ollama connection and generation"""
    logger.info("🚀 Starting Ollama integration test...")

    async with OllamaClient() as client:
        # Health check
        logger.info("1️⃣ Testing Ollama health...")
        is_healthy = await client.health_check()

        if not is_healthy:
            logger.error("❌ Ollama is not running or not accessible")
            logger.info("💡 To start Ollama, run: ollama serve")
            return False

        # List available models
        logger.info("2️⃣ Listing available models...")
        models = await client.list_models()

        if not models:
            logger.error("❌ No models available")
            logger.info("💡 To install a model, run: ollama pull llama3.2")
            return False

        # Test text generation with the first available model
        logger.info("3️⃣ Testing text generation...")
        test_model = models[0]['name']
        test_prompt = "What is artificial intelligence? Explain in one sentence."

        result = await client.generate(test_model, test_prompt)

        if result['success']:
            logger.info("✅ Ollama integration test PASSED!")
            logger.info(f"🎯 Model: {result['model']}")
            logger.info(f"📝 Response: {result['response']}")

            # Test with a data analysis prompt
            logger.info("4️⃣ Testing data analysis capability...")
            data_prompt = """Analyze this sample data and provide insights:
            Data: [{"user": "john", "score": 95}, {"user": "jane", "score": 87}, {"user": "bob", "score": 92}]
            Question: What can you tell me about this data?"""

            data_result = await client.generate(test_model, data_prompt)
            if data_result['success']:
                logger.info(f"🔍 Data Analysis Result: {data_result['response']}")
                return True
            else:
                logger.warning(f"⚠️ Data analysis test failed: {data_result.get('error', 'Unknown error')}")
                return True  # Still consider basic test passed
        else:
            logger.error(f"❌ Ollama generation test FAILED: {result.get('error', 'Unknown error')}")
            return False


async def test_multiple_models():
    """Test multiple models if available"""
    logger.info("🧪 Testing multiple models...")

    async with OllamaClient() as client:
        models = await client.list_models()

        # Test up to 3 models
        for model in models[:3]:
            model_name = model['name']
            logger.info(f"🔄 Testing model: {model_name}")

            prompt = f"Hello! This is a test of {model_name}. Please respond with a brief greeting."
            result = await client.generate(model_name, prompt)

            if result['success']:
                logger.info(f"✅ {model_name}: {result['response'][:100]}...")
            else:
                logger.error(f"❌ {model_name}: {result.get('error', 'Failed')}")


async def benchmark_ollama():
    """Simple benchmark test"""
    logger.info("⏱️ Running simple benchmark...")

    async with OllamaClient() as client:
        models = await client.list_models()
        if not models:
            return

        model_name = models[0]['name']
        prompt = "Count from 1 to 10"

        start_time = asyncio.get_event_loop().time()
        result = await client.generate(model_name, prompt)
        end_time = asyncio.get_event_loop().time()

        if result['success']:
            logger.info(f"⚡ Benchmark Results:")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Total Time: {end_time - start_time:.2f}s")
            logger.info(f"   Response Length: {len(result['response'])} chars")
            logger.info(f"   Tokens Generated: {result.get('stats', {}).get('tokens', 'Unknown')}")


if __name__ == "__main__":
    async def main():
        logger.info("=" * 60)
        logger.info("🧠 APG Connection Management - Ollama Integration Test")
        logger.info("=" * 60)

        # Basic connection test
        success = await test_ollama_connection()

        if success:
            # Extended tests
            await test_multiple_models()
            await benchmark_ollama()

            logger.info("=" * 60)
            logger.info("🎉 All tests completed! Ollama integration is working.")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("❌ Ollama integration test failed.")
            logger.info("💡 Please ensure Ollama is installed and running:")
            logger.info("   1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
            logger.info("   2. Start: ollama serve")
            logger.info("   3. Install model: ollama pull llama3.2")
            logger.info("=" * 60)

    asyncio.run(main())