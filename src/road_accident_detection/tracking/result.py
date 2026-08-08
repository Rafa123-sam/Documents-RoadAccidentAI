"""
Tracker result models for RoadAccidentAI.

This module defines the standardized output returned by every tracker
implementation. Regardless of whether tracks originate from a SORT-style
Kalman tracker or a future re-identification-based tracker, they are
converted into this common representation before being passed to the
rest of the application (motion analysis, trajectory analysis, and
accident scoring).

The tracker result is intentionally independent of any third-party
tracking framework and contains only project domain models.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from road_accident_detection.core.types import (
    BoundingBox,
    FrameIndex,
    ObjectID,
    Point2D,
)
from road_accident_detection.models.detection import Detection

__all__ = [
    "Track",
    "TrackState",
    "TrackingResult",
]


class TrackState:
    """
    Enumerates the lifecycle states of a Track.

    Attributes:
        TENTATIVE:
            Track has been created but not yet confirmed by consecutive
            matches. May be discarded if it fails to match again soon.

        CONFIRMED:
            Track has matched detections for enough consecutive frames
            to be considered reliable.

        LOST:
            Track has not matched any detection recently and is a
            candidate for deletion if it remains unmatched.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass(slots=True)
class Track:
    """
    Represents a single tracked object across video frames.

    A Track accumulates the motion history of one physical road-user
    (vehicle, pedestrian, or cyclist) as it moves through the scene. It
    is the fundamental unit consumed by downstream motion analysis,
    trajectory analysis, and accident-scoring modules.

    Attributes:
        track_id:
            Unique object identifier assigned at track creation.

        state:
            Current lifecycle state of the track.

        class_id:
            Numeric class identifier of the tracked object.

        class_name:
            Human-readable class label of the tracked object.

        bounding_box:
            Most recent bounding box, as
            (x_min, y_min, x_max, y_max).

        velocity:
            Estimated velocity of the bounding box center and scale,
            as (vx, vy, v_scale), in pixels per frame.

        age:
            Total number of frames since the track was created.

        hits:
            Total number of frames in which this track was
            successfully matched to a detection.

        time_since_update:
            Number of consecutive frames since the last successful
            match. Reset to zero on every match.

        history:
            Ordered list of past bounding-box centers, used for
            trajectory and motion analysis.

        frame_indices:
            Ordered list of frame indices corresponding to each entry
            in history.
    """

    track_id: ObjectID

    state: str = TrackState.TENTATIVE

    class_id: int = -1

    class_name: str = "unknown"

    bounding_box: BoundingBox = (0.0, 0.0, 0.0, 0.0)

    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    age: int = 0

    hits: int = 0

    time_since_update: int = 0

    history: list[Point2D] = field(default_factory=list)

    frame_indices: list[FrameIndex] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        """
        Determine whether the track is confirmed.

        Returns:
            True if the track state is CONFIRMED.
        """
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        """
        Determine whether the track is lost.

        Returns:
            True if the track state is LOST.
        """
        return self.state == TrackState.LOST

    @property
    def center(self) -> Point2D:
        """
        Return the current bounding box center.

        Returns:
            Tuple of (x, y).
        """
        x_min, y_min, x_max, y_max = self.bounding_box
        return (
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
        )

    def to_detection(self) -> Detection:
        """
        Convert the current track state into a Detection carrying this
        track's identifier.

        Returns:
            Detection instance annotated with this track_id.
        """
        return Detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=1.0,
            bounding_box=self.bounding_box,
            tracker_id=self.track_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the track into a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the track.
        """
        return {
            "track_id": self.track_id,
            "state": self.state,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bounding_box": list(self.bounding_box),
            "velocity": list(self.velocity),
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "history": [list(point) for point in self.history],
        }

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the track.
        """
        return (
            f"{self.__class__.__name__}("
            f"track_id={self.track_id}, "
            f"state='{self.state}', "
            f"class='{self.class_name}', "
            f"age={self.age}, "
            f"time_since_update={self.time_since_update})"
        )


@dataclass(slots=True)
class TrackingResult:
    """
    Represents the output of a tracker for a single frame.

    Attributes:
        tracks:
            List of active tracks after processing the current frame.

        frame_index:
            Index of the processed frame.

        tracking_time_ms:
            Tracker update time in milliseconds.

        metadata:
            Optional tracker-specific metadata.
    """

    tracks: list[Track] = field(default_factory=list)

    frame_index: FrameIndex = -1

    tracking_time_ms: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """
        Return the number of active tracks.

        Returns:
            Number of tracks.
        """
        return len(self.tracks)

    @property
    def is_empty(self) -> bool:
        """
        Determine whether any tracks exist.

        Returns:
            True if no tracks are present.
        """
        return not self.tracks

    def confirmed_tracks(self) -> list[Track]:
        """
        Return only confirmed tracks.

        Returns:
            List of confirmed tracks.
        """
        return [track for track in self.tracks if track.is_confirmed]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the tracking result into a serializable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "frame_index": self.frame_index,
            "tracking_time_ms": self.tracking_time_ms,
            "count": self.count,
            "tracks": [track.to_dict() for track in self.tracks],
            "metadata": self.metadata,
        }

    def __iter__(self):
        """
        Iterate over tracks.

        Returns:
            Track iterator.
        """
        return iter(self.tracks)

    def __len__(self) -> int:
        """
        Return the number of tracks.

        Returns:
            Number of tracks.
        """
        return self.count

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"count={self.count}, "
            f"tracking_time_ms={self.tracking_time_ms:.2f}, "
            f"frame_index={self.frame_index})"
        )