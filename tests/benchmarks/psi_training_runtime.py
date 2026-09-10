"""CPU-testable runtime primitives for controlled PSI route comparisons.

The PSI composed sampler emits globally interleaved *batches*. These helpers
are specific to that contract, not a replacement for DistributedSampler.
"""

from importlib import import_module
from hashlib import sha256
import time


def training_protocol(*, rank, world_size, batch_size, native_ddp_mode,
                      timing_mode="diagnostic", data_mode="legacy", loader_workers=0,
                      loader_prefetch_factor=2, seed=0, model_precision="fp16"):
    """Bind corrected data/precision semantics separately from legacy results."""
    if (
        type(rank) is not int
        or type(world_size) is not int
        or type(batch_size) is not int
        or not 0 <= rank < world_size
        or batch_size <= 0
        or native_ddp_mode not in {"standard", "diagnostic"}
        or timing_mode not in {"diagnostic", "production", "window"}
        or model_precision not in {"fp16", "fp32"}
        or (model_precision == "fp32" and native_ddp_mode != "standard")
    ):
        raise ValueError("invalid training protocol geometry or reducer mode")
    data = loader_configuration(data_mode, loader_workers, loader_prefetch_factor, seed)
    protocol = {
        "version": 2,
        "sampler": "psi_global_batches_rank_sharded",
        "rank": rank,
        "world_size": world_size,
        "batch_size_per_rank": batch_size,
        "model_precision": model_precision,
        "optimizer_state_precision": "fp32",
        "master_weight_precision": "fp32",
        "native_ddp_mode": native_ddp_mode,
        "measurement": "synchronized_phase_diagnostic",
        "qualification_eligible": False,
    }
    protocol.update(
        version=4 if timing_mode == "window" else 3,
        timing_mode=timing_mode,
        measurement=("training_loop_window_wall" if timing_mode == "window"
                     else "synchronized_step_wall" if timing_mode == "production"
                     else "synchronized_phase_diagnostic"),
        phase_breakdown_available=timing_mode == "diagnostic",
        core_record_field=(None if timing_mode == "window" else
                           "update_s" if timing_mode == "production" else "phase_sum"),
        model_warmup_backwards=2,
        oracle_policy="observation_only",
    )
    if data_mode == "deterministic":
        protocol["data_pipeline"] = data
    return protocol


def validate_training_protocol(value):
    if type(value) is not dict:
        raise ValueError("invalid training protocol")
    try:
        data = value.get("data_pipeline")
        if data is not None and type(data) is not dict:
            raise ValueError("invalid data pipeline protocol")
        expected = training_protocol(
            rank=value["rank"],
            world_size=value["world_size"],
            batch_size=value["batch_size_per_rank"],
            native_ddp_mode=value["native_ddp_mode"],
            timing_mode=value.get("timing_mode", "diagnostic"),
            data_mode="deterministic" if data is not None else "legacy",
            loader_workers=data["workers"] if data is not None else 0,
            loader_prefetch_factor=data["prefetch_factor"] if data is not None else 2,
            seed=data["seed"] if data is not None else 0,
            model_precision=value["model_precision"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid training protocol") from error
    if value.get("version") == 2:
        # Read historical diagnostic evidence, but never emit this identity for
        # new training/checkpoints. Exact resume comparison remains fail-closed.
        expected = training_protocol(
            rank=value["rank"], world_size=value["world_size"],
            batch_size=value["batch_size_per_rank"], native_ddp_mode=value["native_ddp_mode"],
        )
        for key in ("timing_mode", "phase_breakdown_available", "core_record_field",
                    "model_warmup_backwards", "oracle_policy"):
            del expected[key]
        expected["version"] = 2
    if (
        type(value.get("version")) is not int
        or type(value.get("qualification_eligible")) is not bool
        or (value.get("version") in {3, 4} and (
            type(value.get("phase_breakdown_available")) is not bool
            or type(value.get("model_warmup_backwards")) is not int
        ))
        or value != expected
    ):
        raise ValueError("inconsistent training protocol")


def loader_configuration(mode, workers, prefetch_factor, seed):
    if (
        mode not in {"legacy", "deterministic"}
        or type(workers) is not int or workers < 0
        or type(prefetch_factor) is not int or prefetch_factor <= 0
        or type(seed) is not int or seed < 0
        or (mode == "legacy" and workers != 0)
    ):
        raise ValueError("invalid loader configuration")
    return {
        "mode": "deterministic_position_v1" if mode == "deterministic" else "legacy",
        "workers": workers, "prefetch_factor": prefetch_factor, "seed": seed,
        "seed_derivation": "sha256-seed-rank-stream-v1",
        "sample_seed_algorithm": "psi-positional-cpu-v1",
    }


def data_loader_seed(seed, rank, stream):
    if any(type(value) is not int or value < 0 for value in (seed, rank, stream)):
        raise ValueError("invalid data seed geometry")
    key = f"sha256-seed-rank-stream-v1:{seed}:{rank}:{stream}".encode("ascii")
    return int.from_bytes(sha256(key).digest()[:8], "big")


def measurement_observability(timing_mode, route, native_ddp_mode, model_precision="fp16"):
    prefix = ("native_fp32_amp_fp16" if model_precision == "fp32"
              else "controlled_fp16_fp32_master")
    return {
        "scope": prefix + ("_training_loop_window_wall" if timing_mode == "window"
                            else "_step_wall" if timing_mode == "production"
                            else "_phase_diagnostic"),
        "gradient_communication_time_available": (
            timing_mode == "diagnostic"
            and (route != "native" or native_ddp_mode == "diagnostic")
        ),
    }


class PhaseTimer:
    """Keep diagnostic events out of the production execution path.

    Production zeros mean uninstrumented phases, never zero communication cost.
    The protocol assigns the whole synchronized step wall time to update_s.
    One instance is bound per worker process, including its autograd hook threads.
    """

    def __init__(self, mode, cuda_provider, *, clock=time.perf_counter):
        if mode not in {"diagnostic", "production", "window"}:
            raise ValueError("invalid timing mode")
        self.mode = mode
        self.cuda_provider = cuda_provider
        self.clock = clock

    def measure(self, action):
        if self.mode in {"production", "window"}:
            return action(), 0.0
        cuda = self.cuda_provider()
        start = cuda.Event(enable_timing=True)
        end = cuda.Event(enable_timing=True)
        start.record()
        result = action()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end)) / 1000.0

    def training_step(self, forward, backward, update):
        if self.mode == "window":
            loss = forward()
            backward(loss)
            return loss, update(), 0.0, 0.0, 0.0
        if self.mode == "production":
            cuda = self.cuda_provider()
            cuda.synchronize()
            started = self.clock()
            loss = forward()
            backward(loss)
            result = update()
            cuda.synchronize()
            return loss, result, 0.0, 0.0, self.clock() - started
        loss, forward_s = self.measure(forward)
        _, backward_s = self.measure(lambda: backward(loss))
        result, update_s = self.measure(update)
        return loss, result, forward_s, backward_s, update_s


class TrainingLoopWindowObserver:
    """Boundary-synchronized loop windows with bounded deferred loss reads."""

    def __init__(self, cuda_provider, device, warmup_steps, *, clock=time.perf_counter):
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative exact integer")
        self.cuda_provider = cuda_provider
        self.device = device
        self.warmup_steps = warmup_steps
        self.clock = clock
        self.windows = []
        self._open = None
        self._pending = []

    def begin(self, *, epoch, local_record_start, global_step_start,
              boundary_already_synchronized=False):
        if self._open is not None:
            raise RuntimeError("training loop window already open")
        if not boundary_already_synchronized:
            self.cuda_provider().synchronize(self.device)
        self._open = {"epoch": epoch, "local_record_start": local_record_start,
                      "global_step_start": global_step_start, "started": self.clock(),
                      "local_record_end": local_record_start,
                      "global_step_end": global_step_start, "samples": 0}

    def observe(self, *, loss, record, audit, epoch, local_record_end,
                global_step_end, samples):
        if self._open is None or epoch != self._open["epoch"]:
            raise RuntimeError("loss observed outside its training loop window")
        target = record["quality"]
        if target.get("loss") is not None:
            pass
        elif audit:
            target["loss"] = float(loss.detach())
        else:
            self._pending.append((loss.detach(), target))
        self._open["local_record_end"] = local_record_end
        self._open["global_step_end"] = global_step_end
        self._open["samples"] += samples
        if self.warmup_steps and local_record_end == self.warmup_steps:
            self.close("warmup")

    def close(self, kind):
        if self._open is None:
            return None
        if self._open["local_record_end"] == self._open["local_record_start"]:
            self._open = None; self._pending.clear(); return None
        self.cuda_provider().synchronize(self.device)
        elapsed = self.clock() - self._open.pop("started")
        for loss, target in self._pending:
            target["loss"] = float(loss)
        self._pending.clear()
        window = dict(self._open)
        window.update(kind=kind, local_record_range=[window.pop("local_record_start"), window.pop("local_record_end")],
                      global_step_range=[window.pop("global_step_start"), window.pop("global_step_end")],
                      elapsed_wall_s=float(elapsed))
        self.windows.append(window); self._open = None
        return window

    def abort(self):
        self._pending.clear(); self._open = None

    @property
    def is_open(self):
        return self._open is not None


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
    """Keep AdamW in FP32, aliasing FP32 models and bridging FP16 models.

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
            p if p.dtype == torch.float32 else torch.nn.Parameter(
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
            if model is master:
                continue
            master.grad = (
                None
                if model.grad is None
                else model.grad.detach().to(dtype=torch.float32, copy=True)
            )

    def publish(self):
        torch = import_module("torch")
        with torch.no_grad():
            for model, master in zip(self.model_parameters, self.master_parameters):
                if model is not master:
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
