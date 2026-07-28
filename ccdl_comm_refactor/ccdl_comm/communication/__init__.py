"""Communication integration scaffolding."""

from .collectives import CompressedAllReduce, CompressedPayload
from .ddp import DDPBucketProcessor
from .ddp_hook import create_ddp_comm_hook
from .bucket_fusion import BucketDescriptor, BucketFusionGroup, BucketFusionPlan, plan_bucket_fusion
from .gather_reduce import CompressedAllGatherReduce, GatheredPayloads
from .torch_transport import TorchDistributedUnavailableError, make_torch_all_gather, make_torch_all_reduce

__all__ = [
    "BucketDescriptor",
    "BucketFusionGroup",
    "BucketFusionPlan",
    "CompressedAllReduce",
    "CompressedAllGatherReduce",
    "CompressedPayload",
    "DDPBucketProcessor",
    "GatheredPayloads",
    "TorchDistributedUnavailableError",
    "create_ddp_comm_hook",
    "make_torch_all_gather",
    "make_torch_all_reduce",
    "plan_bucket_fusion",
]
