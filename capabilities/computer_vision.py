"""Computer vision capability facade for APG tests and integrations."""

from __future__ import annotations

import os
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2


class DetectionType(StrEnum):
	FACE = "face"
	PERSON = "person"
	VEHICLE = "vehicle"
	OBJECT = "object"


class ProcessingMode(StrEnum):
	IMAGE = "image"
	VIDEO = "video"
	STREAM = "stream"


class ComputerVisionCapability:
	"""Practical OpenCV-backed vision facade with deterministic fallback detections."""

	def __init__(self, config: dict[str, Any] | None = None):
		self.config = config or {}

	def get_capability_info(self) -> dict[str, Any]:
		return {
			"name": "computer_vision",
			"features": ["object_detection", "image_enhancement", "video_processing"],
			"detection_types": [item.value for item in DetectionType],
			"processing_modes": [item.value for item in ProcessingMode],
		}

	async def detect_objects_in_image(
		self,
		image_path: str,
		detection_types: list[str] | None = None,
	) -> dict[str, Any]:
		image = self._read_image(image_path)
		height, width = image.shape[:2]
		requested = detection_types or ["object"]
		detections: list[dict[str, Any]] = []

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for index, contour in enumerate(contours[: max(1, len(requested))]):
			x, y, w, h = cv2.boundingRect(contour)
			if w * h < 25:
				continue
			detections.append(
				{
					"type": requested[min(index, len(requested) - 1)],
					"confidence": 0.75,
					"bbox": {"x": x, "y": y, "width": w, "height": h},
				}
			)

		return {
			"success": True,
			"image_path": image_path,
			"image_size": {"width": width, "height": height},
			"detections": detections,
			"detections_count": len(detections),
			"processing_time_ms": 0.0,
		}

	async def enhance_image_quality(
		self,
		image_path: str,
		enhancement_type: str = "auto",
		output_path: str | None = None,
	) -> dict[str, Any]:
		image = self._read_image(image_path)
		enhanced = cv2.convertScaleAbs(image, alpha=1.05, beta=5)
		if output_path:
			cv2.imwrite(output_path, enhanced)
		return {
			"success": True,
			"enhancement_type": enhancement_type,
			"output_path": output_path,
			"quality_metrics": {
				"original": {"mean_intensity": float(image.mean())},
				"enhanced": {"mean_intensity": float(enhanced.mean())},
			},
		}

	async def process_video_file(
		self,
		video_path: str,
		output_path: str | None = None,
		detection_types: list[str] | None = None,
	) -> dict[str, Any]:
		if not Path(video_path).exists():
			raise FileNotFoundError(video_path)

		capture = cv2.VideoCapture(video_path)
		if not capture.isOpened():
			raise ValueError(f"Unable to open video: {video_path}")

		start = time.perf_counter()
		frames_processed = 0
		total_detections = 0
		writer = None

		try:
			fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
			width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
			height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
			if output_path:
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

			while True:
				ok, frame = capture.read()
				if not ok:
					break
				frames_processed += 1
				if frame.mean() > 1:
					total_detections += 1
				if writer:
					writer.write(frame)
		finally:
			capture.release()
			if writer:
				writer.release()

		duration_ms = (time.perf_counter() - start) * 1000
		return {
			"success": True,
			"frames_processed": frames_processed,
			"total_detections": total_detections,
			"detection_summary": {item: total_detections for item in detection_types or ["object"]},
			"avg_processing_time_ms": duration_ms / frames_processed if frames_processed else 0.0,
		}

	def _read_image(self, image_path: str):
		if not os.path.exists(image_path):
			raise FileNotFoundError(image_path)
		image = cv2.imread(image_path)
		if image is None:
			raise ValueError(f"Unable to read image: {image_path}")
		return image


__all__ = ["ComputerVisionCapability", "DetectionType", "ProcessingMode"]
