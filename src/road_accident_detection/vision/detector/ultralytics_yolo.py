"""
Ultralytics YOLO detector implementation.

This module provides a production-ready detector built on top of the
Ultralytics YOLO framework. It implements the BaseDetector interface and
returns standardized DetectionResult objects for use throughout the
RoadAccidentAI pipeline.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from road_accident_detection.core.exceptions import (
    ModelLoadError,
)
from road_accident_detection.models.detection import Detection
from road_accident_detection.vision.detector.base import BaseDetector
from road_accident_detection.vision.detector.result import DetectionResult

LOGGER = logging.getLogger(__name__)


class UltralyticsYOLODetector(BaseDetector):
    """
    Production-ready Ultralytics YOLO detector.

    The detector wraps an Ultralytics YOLO model while exposing the
    project's standardized BaseDetector interface.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize detector.

        Args:
            model_path:
                Path to YOLO model weights.

            confidence_threshold:
                Minimum confidence required.

            iou_threshold:
                IoU threshold used during NMS.

            device:
                Device identifier.
                Examples:
                    "cpu"
                    "cuda"
                    "cuda:0"

            verbose:
                Enable Ultralytics logging.
        """

        self._model_path = Path(model_path)
        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._verbose = verbose

        self._device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self._model: YOLO | None = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        """
        Return detector name.
        """

        return "UltralyticsYOLO"

    @property
    def is_loaded(self) -> bool:
        """
        Return loading state.
        """

        return self._loaded

    @property
    def model_path(self) -> Path:
        """
        Return model path.
        """

        return self._model_path

    @property
    def device(self) -> str:
        """
        Return inference device.
        """

        return self._device

    def load(self) -> None:
        """
        Load YOLO model.

        Raises:
            ModelLoadError:
                If loading fails.
        """

        if self._loaded:
            return

        if not self._model_path.exists():
            raise ModelLoadError(
                f"Model file not found: {self._model_path}"
            )

        try:
            LOGGER.info(
                "Loading YOLO model from %s",
                self._model_path,
            )

            self._model = YOLO(str(self._model_path))

            self._model.to(self._device)

            self._loaded = True

            LOGGER.info(
                "YOLO model loaded successfully on %s",
                self._device,
            )

        except Exception as exc:
            raise ModelLoadError(
                f"Unable to load YOLO model: {exc}"
            ) from exc

    def unload(self) -> None:
        """
        Release model resources.
        """

        self._model = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOGGER.info("YOLO detector unloaded.")

    def detect(
        self,
        frame: np.ndarray,
    ) -> DetectionResult:
        """
        Perform object detection on a single frame.

        Args:
            frame:
                RGB image.

        Returns:
            DetectionResult.

        Raises:
            ModelLoadError:
                If the detector has not been loaded.
        """

        if not self._loaded or self._model is None:
            raise ModelLoadError(
                "YOLO model has not been loaded."
            )

        start_time = time.perf_counter()

        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            iou=self._iou,
            device=self._device,
            verbose=self._verbose,
        )

        inference_time = (
            time.perf_counter() - start_time
        ) * 1000.0

        detections: list[Detection] = []

        if not results:
            return DetectionResult(
                frame=frame,
                detections=detections,
                inference_time=inference_time,
            )

        result = results[0]

        names: dict[int, str] = (
            result.names
            if result.names is not None
            else {}
        )

        if result.boxes is None:
            return DetectionResult(
                frame=frame,
                detections=detections,
                inference_time=inference_time,
            )

        for box in result.boxes:

            xyxy = box.xyxy.cpu().numpy().reshape(-1)

            confidence = float(
                box.conf.cpu().numpy()[0]
            )

            class_id = int(
                box.cls.cpu().numpy()[0]
            )

            tracker_id: int | None = None

            if hasattr(box, "id") and box.id is not None:
                tracker_id = int(
                    box.id.cpu().numpy()[0]
                )

            detection = Detection(
                class_id=class_id,
                class_name=names.get(
                    class_id,
                    str(class_id),
                ),
                confidence=confidence,
                bounding_box=(
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                ),
                tracker_id=tracker_id,
            )

            detections.append(detection)

        LOGGER.debug(
            "Detected %d objects in %.2f ms.",
            len(detections),
            inference_time,
        )

        return DetectionResult(
            frame=frame,
            detections=detections,
            inference_time=inference_time,
        )

    def warmup(
        self,
        image_size: tuple[int, int] = (
            640,
            640,
        ),
    ) -> None:
        """
        Warm up the detector by running one dummy inference.

        Args:
            image_size:
                Height and width of the dummy image.
        """

        if not self._loaded:
            return

        dummy = np.zeros(
            (
                image_size[0],
                image_size[1],
                3,
            ),
            dtype=np.uint8,
        )

        self.detect(dummy)

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return detector metadata.

        Returns:
            Dictionary describing the detector state.
        """

        return {
            "name": self.model_name,
            "model_path": str(self._model_path),
            "device": self._device,
            "loaded": self._loaded,
            "confidence_threshold": self._confidence,
            "iou_threshold": self._iou,
        }

    def __repr__(self) -> str:
        """
        Return developer-friendly representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"model_path='{self._model_path}', "
            f"device='{self._device}', "
            f"loaded={self._loaded}, "
            f"confidence={self._confidence:.2f}, "
            f"iou={self._iou:.2f})"
        )

    def __str__(self) -> str:
        """
        Return human-readable detector description.
        """

        return (
            f"{self.model_name}"
            f"(device={self._device}, "
            f"loaded={self._loaded})"
        )