"""Streaming package for camera acquisition and frame queuing."""
from .camera import BaseCamera, OpenCVCamera, parse_source
from .frame_queue import FramePacket, LatestFrameBuffer

__all__ = ["BaseCamera", "OpenCVCamera", "parse_source", "FramePacket", "LatestFrameBuffer"]
