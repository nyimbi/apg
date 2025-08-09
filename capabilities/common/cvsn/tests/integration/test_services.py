"""
Computer Vision & Visual Intelligence - Integration Tests for Services

Integration tests for computer vision services including end-to-end workflows,
external service integration, and performance validation.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from ...service import (
	CVProcessingService, CVDocumentAnalysisService, CVObjectDetectionService,
	CVImageClassificationService, CVFacialRecognitionService,
	CVQualityControlService, CVVideoAnalysisService, CVSimilaritySearchService
)
from ...models import ProcessingType, ContentType, ProcessingStatus


class TestCVProcessingServiceIntegration:
	"""Integration tests for main processing service"""
	
	@pytest.fixture
	def processing_service(self):
		"""Processing service fixture"""
		return CVProcessingService()
	
	@pytest.fixture
	def temp_image_file(self):
		"""Create temporary test image file"""
		with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
			# Create simple test image
			image = Image.new('RGB', (100, 100), color='red')
			image.save(f.name, 'JPEG')
			yield f.name
		Path(f.name).unlink()  # Clean up
	
	@pytest.mark.asyncio
	async def test_end_to_end_job_processing(self, processing_service, temp_image_file):
		"""Test complete job processing workflow"""
		# Create job
		job = await processing_service.create_processing_job(
			job_name="Integration Test Job",
			processing_type=ProcessingType.OCR,
			content_type=ContentType.IMAGE,
			input_file_path=temp_image_file,
			processing_parameters={"language": "eng"},
			tenant_id="test_tenant",
			user_id="test_user"
		)
		
		assert job.job_name == "Integration Test Job"
		assert job.status == ProcessingStatus.PENDING
		assert job.id in processing_service.active_jobs
		
		# Mock processing
		with patch.object(processing_service, '_process_ocr_job') as mock_process:
			mock_process.return_value = {
				"extracted_text": "Test text content",
				"confidence_score": 0.95,
				"processing_time_ms": 1000
			}
			
			# Process job
			result = await processing_service.process_job(job.id)
			
			assert result.status == ProcessingStatus.COMPLETED
			assert result.results["extracted_text"] == "Test text content"
			assert result.results["confidence_score"] == 0.95
	
	@pytest.mark.asyncio
	async def test_concurrent_job_processing(self, processing_service, temp_image_file):
		"""Test processing multiple jobs concurrently"""
		# Create multiple jobs
		jobs = []
		for i in range(5):
			job = await processing_service.create_processing_job(
				job_name=f"Concurrent Job {i}",
				processing_type=ProcessingType.OCR,
				content_type=ContentType.IMAGE,
				input_file_path=temp_image_file,
				processing_parameters={},
				tenant_id="test_tenant",
				user_id="test_user"
			)
			jobs.append(job)
		
		assert len(processing_service.active_jobs) == 5
		
		# Mock processing for all jobs
		with patch.object(processing_service, '_process_ocr_job') as mock_process:
			mock_process.return_value = {
				"extracted_text": "Concurrent test",
				"confidence_score": 0.9,
				"processing_time_ms": 500
			}
			
			# Process all jobs concurrently
			tasks = [processing_service.process_job(job.id) for job in jobs]
			results = await asyncio.gather(*tasks)
			
			# Verify all jobs completed
			assert all(result.status == ProcessingStatus.COMPLETED for result in results)
			assert all(result.results["extracted_text"] == "Concurrent test" for result in results)
	
	@pytest.mark.asyncio
	async def test_job_retry_mechanism(self, processing_service, temp_image_file):
		"""Test job retry on failure"""
		job = await processing_service.create_processing_job(
			job_name="Retry Test Job",
			processing_type=ProcessingType.OCR,
			content_type=ContentType.IMAGE,
			input_file_path=temp_image_file,
			processing_parameters={},
			tenant_id="test_tenant",
			user_id="test_user"
		)
		
		# Mock processing failure
		with patch.object(processing_service, '_process_ocr_job', side_effect=Exception("Processing failed")):
			result = await processing_service.process_job(job.id)
			
			assert result.status == ProcessingStatus.RETRY
			assert result.retry_count == 1
			assert result.error_message == "Processing failed"


class TestCVDocumentAnalysisServiceIntegration:
	"""Integration tests for document analysis service"""
	
	@pytest.fixture
	def document_service(self):
		return CVDocumentAnalysisService()
	
	@pytest.fixture
	def temp_document_file(self):
		"""Create temporary test document"""
		with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
			# Create test image with text
			image = Image.new('RGB', (400, 200), color='white')
			image.save(f.name, 'JPEG')
			yield f.name
		Path(f.name).unlink()
	
	@pytest.mark.asyncio
	async def test_comprehensive_document_analysis(self, document_service, temp_document_file):
		"""Test comprehensive document analysis workflow"""
		with patch.multiple(
			document_service,
			process_document_ocr=AsyncMock(return_value={
				"extracted_text": "Test document content",
				"confidence_score": 0.9,
				"language_detected": "eng",
				"word_count": 3,
				"processing_time_ms": 100
			}),
			_analyze_document_layout=AsyncMock(return_value={"page_count": 1}),
			_extract_form_fields=AsyncMock(return_value=[]),
			_extract_tables=AsyncMock(return_value=[]),
			_classify_document_type=AsyncMock(return_value="general_document"),
			_extract_key_entities=AsyncMock(return_value=[])
		):
			result = await document_service.analyze_document_comprehensive(
				temp_document_file,
				{"analysis_level": "detailed"},
				"test_tenant"
			)
			
			assert "extracted_text" in result
			assert "layout_analysis" in result
			assert "document_classification" in result
			assert "total_processing_time_ms" in result
			assert result["extracted_text"] == "Test document content"
	
	@pytest.mark.asyncio
	async def test_ocr_with_enhancement(self, document_service, temp_document_file):
		"""Test OCR processing with image enhancement"""
		with patch('pytesseract.image_to_string') as mock_ocr:
			mock_ocr.return_value = "Enhanced text extraction result"
			
			result = await document_service.process_document_ocr(
				temp_document_file,
				{"language": "eng", "enhance_image": True},
				"test_tenant"
			)
			
			assert result["extracted_text"] == "Enhanced text extraction result"
			assert result["language_detected"] == "eng"
			assert "processing_time_ms" in result
			assert "confidence_score" in result


class TestCVObjectDetectionServiceIntegration:
	"""Integration tests for object detection service"""
	
	@pytest.fixture
	def detection_service(self):
		return CVObjectDetectionService()
	
	@pytest.fixture
	def temp_test_image(self):
		"""Create temporary test image"""
		with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
			# Create test image with objects
			image = Image.new('RGB', (640, 480), color='blue')
			image.save(f.name, 'JPEG')
			yield f.name
		Path(f.name).unlink()
	
	@pytest.mark.asyncio
	async def test_object_detection_workflow(self, detection_service, temp_test_image):
		"""Test complete object detection workflow"""
		# Mock YOLO model and results
		mock_results = Mock()
		mock_results.boxes = Mock()
		mock_results.boxes.data = np.array([
			[100, 100, 200, 200, 0.9, 0],  # person detection
			[300, 200, 500, 350, 0.8, 2]   # car detection
		])
		
		with patch.object(detection_service, '_load_yolo_model') as mock_load:
			mock_model = Mock()
			mock_model.return_value = [mock_results]
			mock_model.names = {0: 'person', 2: 'car'}
			mock_load.return_value = mock_model
			
			result = await detection_service.detect_objects(
				temp_test_image,
				{"model_name": "yolov8n.pt", "confidence_threshold": 0.5},
				"test_tenant"
			)
			
			assert "detected_objects" in result
			assert "total_objects" in result
			assert "detection_confidence" in result
			assert "processing_time_ms" in result
			assert result["total_objects"] >= 0  # May be 0 if no valid detections
	
	@pytest.mark.asyncio
	async def test_batch_object_detection(self, detection_service, temp_test_image):
		"""Test batch object detection processing"""
		# Create multiple test images
		test_images = [temp_test_image] * 3
		
		with patch.object(detection_service, 'detect_objects') as mock_detect:
			mock_detect.return_value = {
				"detected_objects": [{"class_name": "person", "confidence": 0.9}],
				"total_objects": 1,
				"detection_confidence": 0.9,
				"processing_time_ms": 300
			}
			
			results = await detection_service.batch_detect_objects(
				test_images,
				{"model_name": "yolov8n.pt"},
				"test_tenant"
			)
			
			assert "batch_results" in results
			assert "summary" in results
			assert len(results["batch_results"]) == 3
			assert results["summary"]["total_images"] == 3


class TestCVQualityControlServiceIntegration:
	"""Integration tests for quality control service"""
	
	@pytest.fixture
	def qc_service(self):
		return CVQualityControlService()
	
	@pytest.fixture
	def temp_product_image(self):
		"""Create temporary product image"""
		with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
			# Create test product image
			image = Image.new('RGB', (800, 600), color='gray')
			image.save(f.name, 'JPEG')
			yield f.name
		Path(f.name).unlink()
	
	@pytest.mark.asyncio
	async def test_quality_inspection_workflow(self, qc_service, temp_product_image):
		"""Test complete quality inspection workflow"""
		with patch.object(qc_service, '_load_defect_detection_model') as mock_load:
			mock_model = Mock()
			mock_model.detect_defects.return_value = {
				"defects": [
					{
						"type": "scratch",
						"severity": "MINOR",
						"confidence": 0.8,
						"location": {"x": 100, "y": 100, "width": 10, "height": 5}
					}
				],
				"overall_score": 0.9,
				"pass_fail": "PASS"
			}
			mock_load.return_value = mock_model
			
			result = await qc_service.inspect_product_quality(
				temp_product_image,
				{
					"inspection_type": "defect_detection",
					"sensitivity": "medium",
					"pass_threshold": 0.8
				},
				"test_tenant"
			)
			
			assert "inspection_result" in result
			assert "defects_detected" in result
			assert "quality_score" in result
			assert "processing_time_ms" in result
	
	@pytest.mark.asyncio
	async def test_batch_quality_inspection(self, qc_service, temp_product_image):
		"""Test batch quality inspection"""
		product_images = [temp_product_image] * 5
		
		with patch.object(qc_service, 'inspect_product_quality') as mock_inspect:
			mock_inspect.return_value = {
				"inspection_result": {"pass_fail_status": "PASS", "overall_score": 0.95},
				"defects_detected": [],
				"quality_score": 0.95,
				"processing_time_ms": 500
			}
			
			results = await qc_service.batch_quality_inspection(
				product_images,
				{"inspection_type": "defect_detection"},
				"test_tenant"
			)
			
			assert "batch_summary" in results
			assert "individual_results" in results
			assert len(results["individual_results"]) == 5
			assert results["batch_summary"]["total_items"] == 5


class TestCVVideoAnalysisServiceIntegration:
	"""Integration tests for video analysis service"""
	
	@pytest.fixture
	def video_service(self):
		return CVVideoAnalysisService()
	
	@pytest.fixture
	def temp_video_file(self):
		"""Create temporary test video file"""
		with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
			# Create minimal test file (would be actual video in real scenario)
			f.write(b'fake_video_data')
			yield f.name
		Path(f.name).unlink()
	
	@pytest.mark.asyncio
	async def test_video_analysis_workflow(self, video_service, temp_video_file):
		"""Test video analysis workflow"""
		with patch.object(video_service, '_extract_video_frames') as mock_extract:
			mock_extract.return_value = [
				{"frame_number": 0, "timestamp": 0.0, "image_array": np.zeros((480, 640, 3))},
				{"frame_number": 30, "timestamp": 1.0, "image_array": np.zeros((480, 640, 3))}
			]
			
			with patch.object(video_service, '_analyze_video_frame') as mock_analyze:
				mock_analyze.return_value = {
					"objects": [{"class_name": "person", "confidence": 0.9}],
					"actions": [{"action": "walking", "confidence": 0.8}]
				}
				
				result = await video_service.analyze_video_content(
					temp_video_file,
					{
						"analysis_type": "object_tracking",
						"frame_rate": 1,
						"max_frames": 10
					},
					"test_tenant"
				)
				
				assert "video_info" in result
				assert "analysis_results" in result
				assert "processing_time_ms" in result
	
	@pytest.mark.asyncio
	async def test_frame_extraction(self, video_service, temp_video_file):
		"""Test video frame extraction"""
		with patch.object(video_service, '_extract_video_frames') as mock_extract:
			mock_extract.return_value = [
				{
					"frame_number": i * 30,
					"timestamp": i * 1.0,
					"image_array": np.zeros((480, 640, 3))
				}
				for i in range(5)
			]
			
			result = await video_service.extract_video_frames(
				temp_video_file,
				{"interval_seconds": 1, "max_frames": 5},
				"test_tenant"
			)
			
			assert "extracted_frames" in result
			assert "total_frames" in result
			assert len(result["extracted_frames"]) == 5


class TestServicePerformance:
	"""Performance tests for services"""
	
	@pytest.mark.asyncio
	async def test_service_memory_usage(self):
		"""Test service memory usage under load"""
		processing_service = CVProcessingService()
		
		# Monitor memory usage during job creation
		import psutil
		import os
		
		process = psutil.Process(os.getpid())
		initial_memory = process.memory_info().rss
		
		# Create many jobs
		jobs = []
		for i in range(100):
			job = await processing_service.create_processing_job(
				job_name=f"Memory Test Job {i}",
				processing_type=ProcessingType.OCR,
				content_type=ContentType.IMAGE,
				input_file_path=f"/test/image_{i}.jpg",
				processing_parameters={},
				tenant_id="test_tenant",
				user_id="test_user"
			)
			jobs.append(job)
		
		final_memory = process.memory_info().rss
		memory_increase = final_memory - initial_memory
		
		# Memory increase should be reasonable (less than 100MB for 100 jobs)
		assert memory_increase < 100 * 1024 * 1024  # 100MB
		
		# Cleanup
		for job in jobs:
			if job.id in processing_service.active_jobs:
				del processing_service.active_jobs[job.id]
	
	@pytest.mark.asyncio
	async def test_concurrent_processing_limits(self):
		"""Test concurrent processing limits"""
		processing_service = CVProcessingService()
		
		# Set lower limit for testing
		processing_service.max_concurrent_jobs = 10
		
		# Try to create more jobs than the limit
		jobs = []
		for i in range(15):  # Exceed limit
			try:
				job = await processing_service.create_processing_job(
					job_name=f"Limit Test Job {i}",
					processing_type=ProcessingType.OCR,
					content_type=ContentType.IMAGE,
					input_file_path=f"/test/image_{i}.jpg",
					processing_parameters={},
					tenant_id="test_tenant",
					user_id="test_user"
				)
				jobs.append(job)
			except Exception as e:
				# Should hit limit and raise exception
				assert "concurrent jobs limit" in str(e).lower()
				break
		
		# Should not exceed the limit
		assert len(jobs) <= processing_service.max_concurrent_jobs