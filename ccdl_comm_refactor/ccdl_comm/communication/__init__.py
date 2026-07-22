"""Communication integration scaffolding."""

from .collectives import CompressedAllReduce, CompressedPayload
from .ddp import DDPBucketProcessor

__all__ = ["CompressedAllReduce", "CompressedPayload", "DDPBucketProcessor"]
