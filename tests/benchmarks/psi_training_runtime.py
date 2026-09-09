"""CPU-testable runtime primitives for controlled PSI route comparisons.

The PSI composed sampler emits globally interleaved *batches*. These helpers
are specific to that contract, not a replacement for DistributedSampler.
"""

from importlib import import_module
import time


def training_protocol(*, rank, world_size, batch_size, native_ddp_mode):
    """Bind corrected data/precision semantics separately from legacy results."""
    if (
        type(rank) is not int
        or type(world_size) is not int
        or type(batch_size) is not int
        or not 0 <= rank < world_size
        or batch_size <= 0
        or native_ddp_mode not in {"standard", "diagnostic"}
    ):
        raise ValueError("invalid training protocol geometry or reducer mode")
    return {
        "version": 2,
        "sampler": "psi_global_batches_rank_sharded",
        "rank": rank,
        "world_size": world_size,
        "batch_size_per_rank": batch_size,
        "model_precision": "fp16",
        "optimizer_state_precision": "fp32",
        "master_weight_precision": "fp32",
        "native_ddp_mode": native_ddp_mode,
        "measurement": "synchronized_phase_diagnostic",
        "qualification_eligible": False,
    }


def validate_training_protocol(value):
    if type(value) is not dict:
        raise ValueError("invalid training protocol")
    try:
        expected = training_protocol(
            rank=value["rank"],
            world_size=value["world_size"],
            batch_size=value["batch_size_per_rank"],
            native_ddp_mode=value["native_ddp_mode"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid training protocol") from error
    if (
        type(value.get("version")) is not int
        or type(value.get("qualification_eligible")) is not bool
        or value != expected
    ):
        raise ValueError("inconsistent training protocol")


def norm_clip_coefficient(norm_sq, max_norm):
    """Compute on the gradient device without reading a CUDA scalar on host."""
    return (max_norm / (norm_sq.sqrt() + 1.0e-6)).clamp(max=1.0)


def execution_counters(records):
    """Count validated local raw rows; optimizer steps are never divided by ranks."""
    successful = sum(record["quality"]["finite"] for record in records)
    return {
        "iterations_completed": len(records),
        "successful_optimizer_updates": successful,
        "skipped_updates": len(records) - successful,
        "optimizer_step_final": (
            records[-1]["quality"]["optimizer_step"] if records else 0
        ),
        "local_samples_consumed": sum(
            len(record["batch_indices"]) for record in records
        ),
    }


class BatchWaitTimer:
    """Measure only successful batch fetches, never work done by the consumer."""

    def __init__(self, *, clock=time.perf_counter):
        self.clock = clock
        self.wait_s = 0.0

    def iterate(self, batches):
        iterator = iter(batches)
        while True:
            started = self.clock()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.wait_s += self.clock() - started
            yield batch


class RankBatchSampler:
    """Select this rank's whole batches from PSI's global batch stream."""

    def __init__(self, sampler, *, rank: int, world_size: int, batch_size: int):
        if (
            type(rank) is not int
            or type(world_size) is not int
            or type(batch_size) is not int
            or world_size <= 0
            or batch_size <= 0
            or not 0 <= rank < world_size
        ):
            raise ValueError("invalid rank, world_size or batch_size")
        if isinstance(sampler, RankBatchSampler):
            raise ValueError("sampler is already rank-local")
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self._global_length()  # fail before training, not at a collective

    def _global_length(self):
        size = len(self.sampler)
        if size <= 0 or size % (self.batch_size * self.world_size):
            raise ValueError(
                "global sampler length must be positive and divisible by global batch size"
            )
        return size

    def __len__(self):
        return self._global_length() // self.world_size

    def __iter__(self):
        # Materialize before yielding so a malformed length cannot leave another
        # rank stuck in backward. The worker already materializes for resume.
        size = self._global_length()
        indices = tuple(self.sampler)
        if len(indices) != size:
            raise ValueError("global sampler yielded a different length")
        width = self.batch_size * self.world_size
        for offset in range(self.rank * self.batch_size, size, width):
            yield from indices[offset : offset + self.batch_size]

    def set_epoch(self, epoch):
        setter = getattr(self.sampler, "set_epoch", None)
        if setter is not None:
            setter(epoch)


class FP32MasterWeights:
    """Keep AdamW parameters/moments in FP32 and publish FP16 model weights.

    Rebind the existing optimizer's groups, preserving scheduler identity and
    per-group hyperparameters. Construct only before the first optimizer step.
    """

    def __init__(self, optimizer):
        torch = import_module("torch")
        if optimizer.state:
            raise ValueError("FP32 master initialization requires a fresh optimizer")
        originals = tuple(
            p for group in optimizer.param_groups for p in group["params"]
        )
        if len({id(p) for p in originals}) != len(originals):
            raise ValueError("duplicate optimizer parameters")
        if not originals or any(
            p.dtype not in (torch.float16, torch.float32) for p in originals
        ):
            raise ValueError("master weights require nonempty FP16/FP32 parameters")
        self.optimizer = optimizer
        self.model_parameters = originals
        self.master_parameters = tuple(
            torch.nn.Parameter(
                p.detach().float().clone(), requires_grad=p.requires_grad
            )
            for p in originals
        )
        offset = 0
        for group in optimizer.param_groups:
            count = len(group["params"])
            group["params"] = list(self.master_parameters[offset : offset + count])
            offset += count

    def prepare_gradients(self):
        torch = import_module("torch")
        for model, master in zip(self.model_parameters, self.master_parameters):
            master.grad = (
                None
                if model.grad is None
                else model.grad.detach().to(dtype=torch.float32, copy=True)
            )

    def publish(self):
        torch = import_module("torch")
        with torch.no_grad():
            for model, master in zip(self.model_parameters, self.master_parameters):
                model.copy_(master)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for model in self.model_parameters:
            model.grad = None

    def state_dict(self):
        return {
            "version": 1,
            "parameters": [p.detach().clone() for p in self.master_parameters],
        }

    def audit_state(self):
        """Borrow committed state for an immediate synchronous hash only."""
        return {
            "version": 1,
            "parameters": [p.detach() for p in self.master_parameters],
        }

    def validate_state_dict(self, state):
        torch = import_module("torch")
        if (
            type(state) is not dict
            or set(state) != {"version", "parameters"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or type(state["parameters"]) is not list
            or len(state["parameters"]) != len(self.master_parameters)
        ):
            raise ValueError("invalid FP32 master checkpoint protocol")
        for value, master in zip(state["parameters"], self.master_parameters):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.shape != master.shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("invalid FP32 master checkpoint tensor")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        torch = import_module("torch")
        with torch.no_grad():
            for master, value in zip(self.master_parameters, state["parameters"]):
                master.copy_(value)
        self.publish()
