"""
Abstract tracker interface for RoadAccidentAI.

This module defines the abstract base class that every multi-object
tracker must implement. The purpose of this abstraction is to isolate
the remainder of the application from any specific tracking algorithm,
mirroring the same design used for BaseDetector.

All tracker implementations (SORT, DeepSORT, ByteTrack, future
re-identification-based trackers, etc.) must inherit from BaseTracker
and implement the required interface.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from road_accident_detection.vision.detector.result import DetectionResult

from .result import TrackingResult

__all__ = [
    "BaseTracker",
]


class BaseTracker(ABC):
    """
    Abstract base class for multi-object trackers.

    Concrete tracker implementations must inherit from this class and
    implement every abstract method.

    This interface ensures that the processing pipeline depends only on
    tracker abstractions rather than algorithm-specific implementations,
    allowing the tracking strategy (e.g. plain SORT vs. an
    appearance-aware variant) to be swapped without changing the
    pipeline or any downstream motion-analysis code.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            max_age:
                Maximum number of consecutive frames a track may remain
                unmatched before it is deleted.

            min_hits:
                Minimum number of consecutive matched frames required
                before a tentative track is confirmed.
        """
        self._max_age = max_age
        self._min_hits = min_hits

    @property
    def max_age(self) -> int:
        """
        Return the maximum track age before deletion.

        Returns:
            Maximum number of unmatched frames tolerated.
        """
        return self._max_age

    @property
    def min_hits(self) -> int:
        """
        Return the minimum hits required for confirmation.

        Returns:
            Minimum number of consecutive matches required.
        """
        return self._min_hits

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the tracker name.

        Returns:
            Human-readable tracker name.
        """

    @property
    @abstractmethod
    def active_track_count(self) -> int:
        """
        Return the number of currently active tracks.

        Returns:
            Number of tracks being maintained.
        """

    @abstractmethod
    def update(
        self,
        detections: DetectionResult,
    ) -> TrackingResult:
        """
        Update the tracker with detections from the current frame.

        This is the core tracking step. Implementations are expected to:

        1. Predict the new state of every existing track.
        2. Associate predictions with the incoming detections.
        3. Update matched tracks and instantiate new tracks for
           unmatched detections.
        4. Age and remove tracks that have exceeded max_age without a
           match.

        Args:
            detections:
                Detections produced by a detector for the current
                frame.

        Returns:
            TrackingResult containing all active tracks after the
            update.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the tracker to its initial state.

        Clears all active tracks and resets any internal ID counters.
        Intended for use between independent video sequences.
        """

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        Return tracker metadata.

        Returns:
            Dictionary describing the tracker.
        """

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the tracker.
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"active_tracks={self.active_track_count}, "
            f"max_age={self.max_age}, "
            f"min_hits={self.min_hits})"
        )