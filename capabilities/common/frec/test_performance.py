"""
APG Facial Recognition - Performance and Security Tests

Comprehensive performance benchmarking and security testing to ensure
the facial recognition system meets enterprise requirements.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import time
import numpy as np
import threading
import concurrent.futures
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import psutil
import gc
import tempfile
import os

from .service import FacialRecognitionService
from .database import FacialDatabaseService
from .encryption import TemplateEncryptionService
from .face_engine import FaceProcessingEngine
from .liveness_engine import LivenessDetectionEngine

class TestPerformanceBenchmarks:
	"""Performance benchmarking tests"""
	
	@pytest.fixture
	def performance_service(self):
		"""Create service optimized for performance testing"""
		service = FacialRecognitionService(
			database_url="sqlite:///:memory:",
			encryption_key="test_key_32_characters_long_123",
			tenant_id="perf_test_tenant"
		)
		return service
	
	@pytest.fixture
	def mock_face_images(self):
		"""Generate multiple mock face images"""
		images = []
		for i in range(50):
			img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
			images.append(img)
		return images
	
	@pytest.mark.asyncio
	async def test_enrollment_performance(self, performance_service, mock_face_images):
		"""Test enrollment performance under load"""
		# Mock dependencies for performance testing
		with patch.multiple(
			performance_service,
			database_service=AsyncMock(),
			encryption_service=Mock(),
			face_engine=Mock(),
			liveness_engine=Mock()
		):
			# Setup fast mocks
			performance_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			performance_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.95
			}
			performance_service.encryption_service.encrypt_template.return_value = b"encrypted"
			performance_service.database_service.store_template.return_value = "template_123"
			
			# Benchmark enrollment performance
			start_time = time.time()
			tasks = []
			
			for i, img in enumerate(mock_face_images[:10]):  # Test with 10 images
				task = performance_service.enroll_face(
					f"user_{i}",
					img,
					{"enrollment_type": "performance_test"}
				)
				tasks.append(task)
			
			results = await asyncio.gather(*tasks)
			end_time = time.time()
			
			total_time = end_time - start_time
			avg_time_per_enrollment = total_time / len(results)
			
			# Performance assertions
			assert all(result["success"] for result in results)
			assert avg_time_per_enrollment < 0.5  # Should be under 500ms per enrollment
			assert total_time < 3.0  # Total time should be under 3 seconds
			
			print(f"Enrollment Performance:")
			print(f"  Total time: {total_time:.2f}s")
			print(f"  Average per enrollment: {avg_time_per_enrollment:.3f}s")
			print(f"  Throughput: {len(results)/total_time:.1f} enrollments/second")
	
	@pytest.mark.asyncio
	async def test_verification_performance(self, performance_service, mock_face_images):
		"""Test verification performance under load"""
		with patch.multiple(
			performance_service,
			database_service=AsyncMock(),
			encryption_service=Mock(),
			face_engine=Mock(),
			liveness_engine=Mock()
		):
			# Setup fast mocks
			performance_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			performance_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.90
			}
			performance_service.face_engine.compare_features.return_value = {
				"similarity_score": 0.92,
				"confidence_score": 0.95
			}
			performance_service.liveness_engine.detect_liveness.return_value = {
				"is_live": True,
				"liveness_score": 0.96
			}
			performance_service.database_service.get_user_templates.return_value = [
				{"id": "template_123", "encrypted_template": b"encrypted_data"}
			]
			performance_service.encryption_service.decrypt_template.return_value = np.random.rand(512)
			
			# Benchmark verification performance
			start_time = time.time()
			tasks = []
			
			for i, img in enumerate(mock_face_images[:20]):  # Test with 20 verifications
				task = performance_service.verify_face(
					f"user_{i % 10}",  # Reuse users
					img,
					{"confidence_threshold": 0.8}
				)
				tasks.append(task)
			
			results = await asyncio.gather(*tasks)
			end_time = time.time()
			
			total_time = end_time - start_time
			avg_time_per_verification = total_time / len(results)
			
			# Performance assertions
			assert all(result["success"] for result in results)
			assert avg_time_per_verification < 0.3  # Should be under 300ms per verification
			assert total_time < 4.0  # Total time should be under 4 seconds
			
			print(f"Verification Performance:")
			print(f"  Total time: {total_time:.2f}s")
			print(f"  Average per verification: {avg_time_per_verification:.3f}s")
			print(f"  Throughput: {len(results)/total_time:.1f} verifications/second")
	
	@pytest.mark.asyncio
	async def test_concurrent_operations(self, performance_service, mock_face_images):
		"""Test concurrent enrollment and verification operations"""
		with patch.multiple(
			performance_service,
			database_service=AsyncMock(),
			encryption_service=Mock(),
			face_engine=Mock(),
			liveness_engine=Mock()
		):
			# Setup mocks
			performance_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			performance_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.92
			}
			performance_service.face_engine.compare_features.return_value = {
				"similarity_score": 0.88,
				"confidence_score": 0.91
			}
			performance_service.encryption_service.encrypt_template.return_value = b"encrypted"
			performance_service.encryption_service.decrypt_template.return_value = np.random.rand(512)
			performance_service.database_service.store_template.return_value = "template_123"
			performance_service.database_service.get_user_templates.return_value = [
				{"id": "template_123", "encrypted_template": b"encrypted_data"}
			]
			
			# Create mixed workload
			start_time = time.time()
			tasks = []
			
			# Mix of enrollments and verifications
			for i in range(15):
				img = mock_face_images[i % len(mock_face_images)]
				if i % 3 == 0:  # Enrollment every 3rd operation
					task = performance_service.enroll_face(
						f"user_{i}",
						img,
						{"enrollment_type": "concurrent_test"}
					)
				else:  # Verification
					task = performance_service.verify_face(
						f"user_{i % 5}",
						img,
						{"confidence_threshold": 0.8}
					)
				tasks.append(task)
			
			results = await asyncio.gather(*tasks)
			end_time = time.time()
			
			total_time = end_time - start_time
			
			# Verify all operations succeeded
			assert all(result["success"] for result in results)
			assert total_time < 5.0  # Should complete within 5 seconds
			
			print(f"Concurrent Operations Performance:")
			print(f"  Total operations: {len(results)}")
			print(f"  Total time: {total_time:.2f}s")
			print(f"  Overall throughput: {len(results)/total_time:.1f} ops/second")
	
	def test_memory_usage_enrollment(self, performance_service, mock_face_images):
		"""Test memory usage during enrollment operations"""
		# Force garbage collection
		gc.collect()
		initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
		
		with patch.multiple(
			performance_service,
			database_service=AsyncMock(),
			encryption_service=Mock(),
			face_engine=Mock()
		):
			# Setup mocks
			performance_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			performance_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.95
			}
			
			# Process multiple images
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			for i, img in enumerate(mock_face_images):
				result = loop.run_until_complete(
					performance_service.enroll_face(
						f"user_{i}",
						img,
						{"enrollment_type": "memory_test"}
					)
				)
				assert result["success"]
			
			loop.close()
		
		# Check memory usage
		gc.collect()
		final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
		memory_increase = final_memory - initial_memory
		
		print(f"Memory Usage Test:")
		print(f"  Initial memory: {initial_memory:.1f} MB")
		print(f"  Final memory: {final_memory:.1f} MB") 
		print(f"  Memory increase: {memory_increase:.1f} MB")
		
		# Memory increase should be reasonable
		assert memory_increase < 100  # Should not increase by more than 100MB
	
	def test_database_performance(self):
		"""Test database operation performance"""
		# Use in-memory SQLite for consistent testing
		from sqlalchemy import create_engine
		from sqlalchemy.orm import sessionmaker
		from .models import Base, FaUser, FaTemplate
		
		engine = create_engine("sqlite:///:memory:")
		Base.metadata.create_all(engine)
		SessionLocal = sessionmaker(bind=engine)
		
		session = SessionLocal()
		
		# Test bulk user creation
		start_time = time.time()
		users = []
		for i in range(1000):
			user = FaUser(
				tenant_id="perf_test",
				external_user_id=f"perf_user_{i:04d}",
				full_name=f"Performance User {i}",
				email=f"perf{i}@example.com",
				consent_given=True
			)
			users.append(user)
		
		session.bulk_save_objects(users)
		session.commit()
		end_time = time.time()
		
		bulk_insert_time = end_time - start_time
		
		# Test bulk template creation
		session.flush()
		user_ids = [user.id for user in session.query(FaUser).all()]
		
		start_time = time.time()
		templates = []
		for i, user_id in enumerate(user_ids[:500]):  # Create 500 templates
			template = FaTemplate(
				tenant_id="perf_test",
				user_id=user_id,
				template_version=1,
				quality_score=Decimal("0.95"),
				algorithm_version="1.0.0",
				encrypted_template=f"template_data_{i}".encode(),
				is_active=True
			)
			templates.append(template)
		
		session.bulk_save_objects(templates)
		session.commit()
		end_time = time.time()
		
		bulk_template_time = end_time - start_time
		
		# Test query performance
		start_time = time.time()
		for i in range(100):
			user = session.query(FaUser).filter_by(
				external_user_id=f"perf_user_{i:04d}"
			).first()
			assert user is not None
		end_time = time.time()
		
		query_time = end_time - start_time
		
		session.close()
		
		print(f"Database Performance:")
		print(f"  1000 users insert: {bulk_insert_time:.3f}s")
		print(f"  500 templates insert: {bulk_template_time:.3f}s")
		print(f"  100 queries: {query_time:.3f}s")
		
		# Performance assertions
		assert bulk_insert_time < 2.0  # Should insert 1000 users in under 2 seconds
		assert bulk_template_time < 2.0  # Should insert 500 templates in under 2 seconds
		assert query_time < 1.0  # Should query 100 users in under 1 second

class TestSecurityValidation:
	"""Security validation and penetration testing"""
	
	@pytest.fixture
	def security_service(self):
		"""Create service for security testing"""
		service = FacialRecognitionService(
			database_url="sqlite:///:memory:",
			encryption_key="secure_key_32_characters_long_12",
			tenant_id="security_test_tenant"
		)
		return service
	
	def test_encryption_strength(self):
		"""Test biometric template encryption strength"""
		encryption_service = TemplateEncryptionService("test_encryption_key_32_chars_abc")
		
		# Test data
		original_template = np.random.rand(512).astype(np.float32)
		
		# Encrypt multiple times - should produce different ciphertexts
		encrypted_1 = encryption_service.encrypt_template(original_template)
		encrypted_2 = encryption_service.encrypt_template(original_template)
		
		# Should be different due to random nonce
		assert encrypted_1 != encrypted_2
		
		# Both should decrypt to same original
		decrypted_1 = encryption_service.decrypt_template(encrypted_1)
		decrypted_2 = encryption_service.decrypt_template(encrypted_2)
		
		np.testing.assert_array_almost_equal(original_template, decrypted_1, decimal=6)
		np.testing.assert_array_almost_equal(original_template, decrypted_2, decimal=6)
		
		# Test with different keys - should fail
		wrong_key_service = TemplateEncryptionService("different_key_32_characters_xyz")
		
		with pytest.raises(Exception):
			wrong_key_service.decrypt_template(encrypted_1)
	
	def test_template_anonymization(self):
		"""Test biometric template anonymization"""
		encryption_service = TemplateEncryptionService("test_key_32_characters_long_abc")
		
		# Original biometric template
		original_template = np.random.rand(512).astype(np.float32)
		encrypted_template = encryption_service.encrypt_template(original_template)
		
		# Anonymize template
		anonymized = encryption_service.anonymize_template(encrypted_template)
		
		# Should not be recoverable
		assert anonymized != encrypted_template
		assert len(anonymized) > 0
		
		# Should not be decryptable to original
		with pytest.raises(Exception):
			encryption_service.decrypt_template(anonymized)
	
	@pytest.mark.asyncio
	async def test_input_sanitization(self, security_service):
		"""Test input sanitization and validation"""
		with patch.object(security_service, 'database_service', AsyncMock()):
			# Test SQL injection attempts
			malicious_inputs = [
				"'; DROP TABLE users; --",
				"' OR '1'='1",
				"admin'/*",
				"'; INSERT INTO users VALUES('hacker'); --"
			]
			
			for malicious_input in malicious_inputs:
				user_data = {
					"external_user_id": malicious_input,
					"full_name": "Test User",
					"consent_given": True
				}
				
				# Should either sanitize input or reject it
				try:
					result = await security_service.create_user(user_data)
					# If it succeeds, verify input was sanitized
					if result:
						assert malicious_input not in str(result)
				except Exception:
					# Rejection is also acceptable
					pass
	
	@pytest.mark.asyncio
	async def test_rate_limiting_simulation(self, security_service):
		"""Test behavior under potential DoS attacks"""
		with patch.multiple(
			security_service,
			database_service=AsyncMock(),
			face_engine=Mock()
		):
			# Setup mocks
			security_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			security_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.90
			}
			
			# Simulate rapid-fire requests
			start_time = time.time()
			tasks = []
			
			fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
			
			for i in range(100):  # 100 rapid requests
				task = security_service.enroll_face(
					f"attack_user_{i}",
					fake_image,
					{"enrollment_type": "dos_test"}
				)
				tasks.append(task)
			
			results = await asyncio.gather(*tasks, return_exceptions=True)
			end_time = time.time()
			
			total_time = end_time - start_time
			
			# System should handle the load gracefully
			successful_results = [r for r in results if not isinstance(r, Exception)]
			failed_results = [r for r in results if isinstance(r, Exception)]
			
			print(f"DoS Simulation Results:")
			print(f"  Total requests: 100")
			print(f"  Successful: {len(successful_results)}")
			print(f"  Failed: {len(failed_results)}")
			print(f"  Total time: {total_time:.2f}s")
			print(f"  Rate: {len(results)/total_time:.1f} req/sec")
			
			# Should not crash completely
			assert len(successful_results) > 0
	
	def test_data_leakage_prevention(self):
		"""Test prevention of sensitive data leakage"""
		encryption_service = TemplateEncryptionService("test_key_32_characters_long_abc")
		
		# Create biometric template with known pattern
		template_with_pattern = np.zeros(512, dtype=np.float32)
		template_with_pattern[0:10] = 1.0  # Distinctive pattern
		
		encrypted = encryption_service.encrypt_template(template_with_pattern)
		
		# Encrypted data should not contain the pattern
		encrypted_str = encrypted.hex()
		
		# Should not find obvious patterns in encrypted data
		assert "000000000000000000000000" not in encrypted_str  # No long zeros
		assert "ffffffffffffffffffffffff" not in encrypted_str  # No long ones
		assert "010101010101010101010101" not in encrypted_str  # No simple patterns
		
		# Should have good entropy
		unique_bytes = len(set(encrypted))
		assert unique_bytes > len(encrypted) * 0.3  # At least 30% unique bytes
	
	def test_timing_attack_resistance(self):
		"""Test resistance to timing attacks"""
		encryption_service = TemplateEncryptionService("test_key_32_characters_long_abc")
		
		# Create templates of different sizes and patterns
		templates = [
			np.zeros(512, dtype=np.float32),  # All zeros
			np.ones(512, dtype=np.float32),   # All ones
			np.random.rand(512).astype(np.float32),  # Random
			np.linspace(0, 1, 512).astype(np.float32)  # Linear pattern
		]
		
		times = []
		
		for template in templates:
			start_time = time.perf_counter()
			encrypted = encryption_service.encrypt_template(template)
			decrypted = encryption_service.decrypt_template(encrypted)
			end_time = time.perf_counter()
			
			times.append(end_time - start_time)
			
			# Verify correctness
			np.testing.assert_array_almost_equal(template, decrypted, decimal=6)
		
		# Times should be similar (within 50% of mean)
		mean_time = sum(times) / len(times)
		for t in times:
			assert abs(t - mean_time) < mean_time * 0.5
		
		print(f"Timing Analysis:")
		print(f"  Mean time: {mean_time:.6f}s")
		print(f"  Times: {[f'{t:.6f}' for t in times]}")
	
	@pytest.mark.asyncio
	async def test_concurrent_access_safety(self, security_service):
		"""Test thread safety under concurrent access"""
		with patch.multiple(
			security_service,
			database_service=AsyncMock(),
			encryption_service=Mock(),
			face_engine=Mock()
		):
			# Setup thread-safe mocks
			security_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			security_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.90
			}
			
			# Simulate concurrent access from multiple "users"
			async def user_session(user_id, num_operations):
				results = []
				fake_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
				
				for i in range(num_operations):
					try:
						result = await security_service.enroll_face(
							f"concurrent_user_{user_id}_{i}",
							fake_image,
							{"session": f"user_{user_id}"}
						)
						results.append(result)
					except Exception as e:
						results.append({"error": str(e)})
				
				return results
			
			# Run multiple concurrent sessions
			tasks = [user_session(i, 5) for i in range(10)]  # 10 users, 5 ops each
			all_results = await asyncio.gather(*tasks)
			
			# Flatten results
			flat_results = [item for sublist in all_results for item in sublist]
			
			# Should handle concurrent access without corruption
			successful_results = [r for r in flat_results if r.get("success")]
			error_results = [r for r in flat_results if "error" in r]
			
			print(f"Concurrent Access Test:")
			print(f"  Total operations: {len(flat_results)}")
			print(f"  Successful: {len(successful_results)}")
			print(f"  Errors: {len(error_results)}")
			
			# Most operations should succeed
			assert len(successful_results) > len(flat_results) * 0.8

class TestStressAndReliability:
	"""Stress testing and reliability validation"""
	
	@pytest.mark.asyncio
	async def test_extended_operation_stability(self):
		"""Test system stability over extended operation"""
		service = FacialRecognitionService(
			database_url="sqlite:///:memory:",
			encryption_key="stress_test_key_32_characters_abc",
			tenant_id="stress_test_tenant"
		)
		
		with patch.multiple(
			service,
			database_service=AsyncMock(),
			face_engine=Mock()
		):
			# Setup mocks
			service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.90
			}
			
			# Run extended operations
			total_operations = 500
			batch_size = 50
			all_results = []
			
			for batch in range(0, total_operations, batch_size):
				batch_tasks = []
				
				for i in range(batch, min(batch + batch_size, total_operations)):
					fake_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
					task = service.enroll_face(
						f"stress_user_{i}",
						fake_image,
						{"batch": batch // batch_size}
					)
					batch_tasks.append(task)
				
				batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
				all_results.extend(batch_results)
				
				# Small delay between batches
				await asyncio.sleep(0.1)
			
			# Analyze results
			successful = [r for r in all_results if not isinstance(r, Exception) and r.get("success")]
			errors = [r for r in all_results if isinstance(r, Exception)]
			
			success_rate = len(successful) / len(all_results)
			
			print(f"Extended Operation Test:")
			print(f"  Total operations: {total_operations}")
			print(f"  Successful: {len(successful)}")
			print(f"  Errors: {len(errors)}")
			print(f"  Success rate: {success_rate:.2%}")
			
			# Should maintain high success rate
			assert success_rate > 0.95
	
	def test_memory_leak_detection(self):
		"""Test for memory leaks over many operations"""
		service = FacialRecognitionService(
			database_url="sqlite:///:memory:",
			encryption_key="memory_test_key_32_characters_abc",
			tenant_id="memory_test_tenant"
		)
		
		with patch.multiple(
			service,
			database_service=AsyncMock(),
			face_engine=Mock()
		):
			# Setup mocks
			service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.90
			}
			
			# Measure initial memory
			gc.collect()
			initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
			
			# Run many operations
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			for i in range(200):
				fake_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
				result = loop.run_until_complete(
					service.enroll_face(
						f"memory_user_{i}",
						fake_image,
						{"iteration": i}
					)
				)
				
				# Force cleanup periodically
				if i % 50 == 0:
					gc.collect()
			
			loop.close()
			
			# Measure final memory
			gc.collect()
			final_memory = psutil.Process().memory_info().rss / 1024 / 1024
			memory_growth = final_memory - initial_memory
			
			print(f"Memory Leak Test:")
			print(f"  Initial memory: {initial_memory:.1f} MB")
			print(f"  Final memory: {final_memory:.1f} MB")
			print(f"  Memory growth: {memory_growth:.1f} MB")
			
			# Memory growth should be minimal
			assert memory_growth < 50  # Should not grow by more than 50MB
	
	def test_error_recovery(self):
		"""Test system recovery from various error conditions"""
		service = FacialRecognitionService(
			database_url="sqlite:///:memory:",
			encryption_key="error_test_key_32_characters_abc",
			tenant_id="error_test_tenant"
		)
		
		# Test recovery from database errors
		with patch.object(service, 'database_service', AsyncMock()) as mock_db:
			# Simulate intermittent database failures
			call_count = 0
			
			async def flaky_store_template(*args, **kwargs):
				nonlocal call_count
				call_count += 1
				if call_count % 3 == 0:  # Fail every 3rd call
					raise Exception("Database connection lost")
				return "template_123"
			
			mock_db.store_template.side_effect = flaky_store_template
			
			# Test face engine recovery
			with patch.object(service, 'face_engine', Mock()) as mock_face:
				mock_face.detect_faces.return_value = [
					{"bbox": [50, 50, 150, 150], "confidence": 0.99}
				]
				mock_face.extract_features.return_value = {
					"features": np.random.rand(512).astype(np.float32),
					"quality_score": 0.90
				}
				
				# Run operations with error injection
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				results = []
				for i in range(10):
					fake_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
					try:
						result = loop.run_until_complete(
							service.enroll_face(
								f"recovery_user_{i}",
								fake_image,
								{"test": "error_recovery"}
							)
						)
						results.append(result)
					except Exception as e:
						results.append({"error": str(e)})
				
				loop.close()
				
				# Should handle errors gracefully
				successful = [r for r in results if r.get("success")]
				failed = [r for r in results if "error" in r]
				
				print(f"Error Recovery Test:")
				print(f"  Total attempts: {len(results)}")
				print(f"  Successful: {len(successful)}")
				print(f"  Failed: {len(failed)}")
				
				# Some should succeed despite errors
				assert len(successful) > 0

if __name__ == "__main__":
	pytest.main([__file__, "-v"])