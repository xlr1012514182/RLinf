# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenCV, RealSense, ROS 2, and synchronized multi-camera adapters."""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..config import CameraConfig
from ..contracts import CameraBackend
from ..errors import HardwareNotReadyError, OptionalDependencyError


class OpenCVCamera:
    """USB/video-device camera using OpenCV with RGB output."""

    def __init__(self, config: CameraConfig) -> None:
        config.validate()
        self.config = config
        self._capture = None

    @property
    def is_connected(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def connect(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise OptionalDependencyError("OpenCV camera requires opencv-python") from error
        source = self.config.options.get("source", 0)
        backend = self.config.options.get("api_preference")
        self._capture = (
            cv2.VideoCapture(source, int(backend))
            if backend is not None
            else cv2.VideoCapture(source)
        )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise HardwareNotReadyError(f"unable to open OpenCV source {source!r}")

    def read(
        self, timeout_s: float = 1.0
    ) -> tuple[NDArray[np.uint8], int, Mapping[str, Any]]:
        if not self.is_connected:
            raise HardwareNotReadyError("OpenCV camera is not connected")
        import cv2

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            success, frame = self._capture.read()
            timestamp_ns = time.monotonic_ns()
            if success:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return rgb, timestamp_ns, {"backend": "opencv", "source": str(self.config.options.get("source", 0))}
            time.sleep(0.005)
        raise HardwareNotReadyError("timed out reading OpenCV frame")

    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class RealSenseCamera:
    """Intel RealSense RGB or aligned RGB-D camera."""

    def __init__(self, config: CameraConfig) -> None:
        config.validate()
        self.config = config
        self._rs = None
        self._pipeline = None
        self._align = None

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise OptionalDependencyError(
                "RealSense camera requires pyrealsense2/librealsense"
            ) from error
        pipeline = rs.pipeline()
        configuration = rs.config()
        serial = self.config.options.get("serial_number")
        if serial:
            configuration.enable_device(str(serial))
        configuration.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            self.config.fps,
        )
        if bool(self.config.options.get("enable_depth", False)):
            configuration.enable_stream(
                rs.stream.depth,
                self.config.width,
                self.config.height,
                rs.format.z16,
                self.config.fps,
            )
        pipeline.start(configuration)
        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

    def read(
        self, timeout_s: float = 1.0
    ) -> tuple[NDArray[Any], int, Mapping[str, Any]]:
        if self._pipeline is None or self._rs is None:
            raise HardwareNotReadyError("RealSense camera is not connected")
        frames = self._pipeline.wait_for_frames(timeout_ms=max(1, int(timeout_s * 1_000)))
        timestamp_ns = time.monotonic_ns()
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            raise HardwareNotReadyError("RealSense frame set contains no color frame")
        color = np.asanyarray(color_frame.get_data()).copy()
        metadata: dict[str, Any] = {
            "backend": "realsense",
            "device_timestamp_ms": float(color_frame.get_timestamp()),
            "frame_number": int(color_frame.get_frame_number()),
        }
        if bool(self.config.options.get("enable_depth", False)):
            depth_frame = aligned.get_depth_frame()
            if not depth_frame:
                raise HardwareNotReadyError("RealSense frame set contains no depth frame")
            metadata["depth"] = np.asanyarray(depth_frame.get_data()).copy()
            metadata["depth_scale_m"] = float(
                aligned.get_profile().get_device().first_depth_sensor().get_depth_scale()
            )
        return color, timestamp_ns, metadata

    def disconnect(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._rs = None


def _ros_image_to_numpy(message) -> NDArray[Any]:
    encoding = message.encoding.lower()
    if encoding in {"rgb8", "bgr8"}:
        dtype, channels = np.uint8, 3
    elif encoding == "mono8":
        dtype, channels = np.uint8, 1
    elif encoding in {"mono16", "16uc1"}:
        dtype, channels = np.uint16, 1
    elif encoding == "32fc1":
        dtype, channels = np.float32, 1
    else:
        raise HardwareNotReadyError(f"unsupported ROS image encoding: {message.encoding}")
    item_size = np.dtype(dtype).itemsize
    row_values = message.step // item_size
    array = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_values)
    array = array[:, : message.width * channels]
    if channels > 1:
        array = array.reshape(message.height, message.width, channels)
        if encoding == "bgr8":
            array = array[..., ::-1]
    else:
        array = array.reshape(message.height, message.width)
    return array.copy()


class ROS2ImageCamera:
    """Subscribe to ``sensor_msgs/Image`` using a private ROS 2 executor."""

    def __init__(self, config: CameraConfig) -> None:
        config.validate()
        self.config = config
        self._rclpy = None
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._latest = None

    @property
    def is_connected(self) -> bool:
        return self._node is not None

    def connect(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import Image
        except ImportError as error:
            raise OptionalDependencyError(
                "ROS2 image camera requires rclpy and sensor_msgs"
            ) from error
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node(f"fibocom_camera_{self.config.name}")
        topic = str(self.config.options.get("topic", f"/{self.config.name}/image_raw"))
        self._node.create_subscription(Image, topic, self._on_image, 2)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name=f"ros2-camera-{self.config.name}",
            daemon=True,
        )
        self._spin_thread.start()

    def _on_image(self, message) -> None:
        image = _ros_image_to_numpy(message)
        received_ns = time.monotonic_ns()
        ros_stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        metadata = {
            "backend": "ros2_image",
            "ros_stamp_ns": ros_stamp_ns,
            "frame_id": message.header.frame_id,
            "encoding": message.encoding,
        }
        with self._condition:
            self._latest = (image, received_ns, metadata)
            self._condition.notify_all()

    def read(
        self, timeout_s: float = 1.0
    ) -> tuple[NDArray[Any], int, Mapping[str, Any]]:
        if not self.is_connected:
            raise HardwareNotReadyError("ROS2 image camera is not connected")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HardwareNotReadyError("timed out waiting for ROS image")
                self._condition.wait(remaining)
            image, timestamp_ns, metadata = self._latest
            return image.copy(), timestamp_ns, dict(metadata)

    def disconnect(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and bool(
            self.config.options.get("shutdown_rclpy", False)
        ):
            self._rclpy.shutdown()
        self._node = None
        self._executor = None
        self._spin_thread = None


class SynchronizedCameraRig:
    """Collect latest frames and enforce a host-time skew gate."""

    def __init__(
        self,
        cameras: Mapping[str, CameraBackend],
        *,
        maximum_skew_ms: float = 35.0,
        maximum_attempts: int = 3,
    ) -> None:
        if not cameras:
            raise ValueError("at least one camera is required")
        if maximum_skew_ms < 0 or maximum_attempts < 1:
            raise ValueError("invalid synchronization gate")
        self.cameras = dict(cameras)
        self.maximum_skew_ns = int(maximum_skew_ms * 1_000_000)
        self.maximum_attempts = maximum_attempts

    def connect(self) -> None:
        """Connect every camera, rolling back if one fails."""

        connected: list[CameraBackend] = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected.append(camera)
        except Exception:
            for camera in reversed(connected):
                camera.disconnect()
            raise

    def read(
        self, timeout_s: float = 1.0
    ) -> tuple[dict[str, NDArray[Any]], int, dict[str, Any]]:
        """Return a bundle whose host receive times satisfy the skew gate."""

        last_skew_ns = 0
        for _ in range(self.maximum_attempts):
            frames = {
                name: camera.read(timeout_s)
                for name, camera in self.cameras.items()
            }
            timestamps = [value[1] for value in frames.values()]
            last_skew_ns = max(timestamps) - min(timestamps)
            if last_skew_ns <= self.maximum_skew_ns:
                images = {name: value[0] for name, value in frames.items()}
                metadata = {
                    "camera_metadata": {name: value[2] for name, value in frames.items()},
                    "camera_timestamps_ns": {
                        name: value[1] for name, value in frames.items()
                    },
                    "skew_ms": last_skew_ns / 1_000_000.0,
                }
                return images, max(timestamps), metadata
        raise HardwareNotReadyError(
            f"camera skew {last_skew_ns / 1_000_000.0:.2f}ms exceeds gate"
        )

    def disconnect(self) -> None:
        """Disconnect all cameras, preserving the first error."""

        first_error = None
        for camera in reversed(tuple(self.cameras.values())):
            try:
                camera.disconnect()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
