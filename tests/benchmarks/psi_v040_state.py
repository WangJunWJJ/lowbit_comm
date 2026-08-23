"""CPU-only deterministic training state for v0.4.0 PSI benchmarks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from math import isfinite


_MAX_SIGNED_64 = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class ShardLayout:
    """One contiguous padded shard and its valid global ownership."""

    global_numel: int
    world_size: int
    rank: int
    start: int
    valid_numel: int
    padded_numel: int

    def __post_init__(self) -> None:
        _validate_shard_layout(self, "ShardLayout")

    @classmethod
    def build(
        cls,
        global_numel: int,
        world_size: int,
        rank: int,
    ) -> ShardLayout:
        """Create a validated, contiguous ownership layout."""
        _require_nonnegative_int(global_numel, "global_numel")
        _require_positive_int(world_size, "world_size")
        if type(rank) is not int or rank < 0 or rank >= world_size:
            raise ValueError("rank must be an exact integer within world_size")

        padded_numel = _checked_ceil_div(global_numel, world_size)
        start = min(_checked_mul(rank, padded_numel), global_numel)
        valid_numel = min(padded_numel, global_numel - start)
        return cls(
            global_numel=global_numel,
            world_size=world_size,
            rank=rank,
            start=start,
            valid_numel=valid_numel,
            padded_numel=padded_numel,
        )


@dataclass(frozen=True, slots=True)
class QWDSchedule:
    """Deterministic full-precision refresh cadence for qWD steps."""

    refresh_interval: int

    def __post_init__(self) -> None:
        _require_positive_int(self.refresh_interval, "refresh_interval")
        if self.refresh_interval != 100:
            raise ValueError("refresh_interval must be exactly 100")

    def mode(self, step: int, force_refresh: bool = False) -> str:
        """Return the communication route required for one optimizer step."""
        _require_nonnegative_int(step, "step")
        if type(force_refresh) is not bool:
            raise ValueError("force_refresh must be an exact bool")
        if force_refresh or step % self.refresh_interval == 0:
            return "fp_refresh"
        return "qwd"


@dataclass(frozen=True, slots=True)
class _ResidualCandidate:
    """Prepared residual data owned by one transaction."""

    owner: int
    value: object


class CommittedResidual:
    """Error-feedback residual with explicit publish-or-abort semantics."""

    __slots__ = ("_committed", "_pending", "_owner")

    def __init__(self, value: object) -> None:
        self._committed = deepcopy(value)
        self._pending: _ResidualCandidate | None = None
        self._owner = id(self)

    @property
    def value(self) -> object:
        """Return a copy of the last committed residual."""
        return deepcopy(self._committed)

    def prepare(self, value: object) -> _ResidualCandidate:
        """Stage a candidate without changing the committed residual."""
        candidate = _ResidualCandidate(self._owner, deepcopy(value))
        self._pending = candidate
        return candidate

    def commit(self, candidate: _ResidualCandidate) -> None:
        """Publish the one candidate currently staged by this residual."""
        if (
            type(candidate) is not _ResidualCandidate
            or candidate is not self._pending
        ):
            raise ValueError("candidate was not prepared by this transaction")
        self._committed = deepcopy(candidate.value)
        self._pending = None

    def abort(self) -> None:
        """Discard staged state while retaining the last committed residual."""
        self._pending = None


class ShardedAdamW:
    """FP32 AdamW state for the valid portion of one padded shard."""

    __slots__ = (
        "layout",
        "master",
        "exp_avg",
        "exp_avg_sq",
        "step_count",
        "learning_rate",
        "betas",
        "eps",
        "weight_decay",
        "amp_state",
        "rng_state",
        "force_refresh",
    )

    def __init__(
        self,
        layout: ShardLayout,
        master: object,
        *,
        learning_rate: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        self.layout = _trusted_shard_layout(layout)
        self.learning_rate = _require_finite_float(
            learning_rate,
            "learning_rate",
            positive=True,
        )
        self.betas = _require_betas(betas)
        self.eps = _require_finite_float(eps, "eps", positive=True)
        self.weight_decay = _require_finite_float(
            weight_decay,
            "weight_decay",
            nonnegative=True,
        )
        self.master = _validated_shard_tensor(master, layout, "master").clone()
        self._zero_padding(self.master)
        self.exp_avg = self.master.new_zeros(layout.padded_numel)
        self.exp_avg_sq = self.master.new_zeros(layout.padded_numel)
        self.step_count = 0
        self.amp_state: dict[str, object] = {}
        self.rng_state: dict[str, object] = {}
        self.force_refresh = False

    def step(self, gradient_shard: object) -> None:
        """Apply one FP32 AdamW update to valid elements only."""
        gradient = _validated_shard_tensor(
            gradient_shard,
            self.layout,
            "gradient_shard",
            device=self.master.device,
        )
        self._step_validated(gradient)

    def step_prevalidated(self, gradient_shard: object) -> None:
        (
            "Update from a same-device shard whose finiteness was already "
            "proven."
        )
        gradient = _validated_shard_tensor(
            gradient_shard,
            self.layout,
            "gradient_shard",
            device=self.master.device,
            require_finite=False,
        )
        self._step_validated(gradient)

    def _step_validated(self, gradient: object) -> None:
        valid = self.layout.valid_numel
        if valid == 0:
            self.step_count += 1
            self._zero_padding(self.master)
            return

        torch = _torch()
        with torch.no_grad():
            master = self.master[:valid]
            exp_avg = self.exp_avg[:valid]
            exp_avg_sq = self.exp_avg_sq[:valid]
            gradient = gradient[:valid]
            beta1, beta2 = self.betas
            self.step_count += 1
            master.mul_(1.0 - self.learning_rate * self.weight_decay)
            exp_avg.lerp_(gradient, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            bias_correction1 = 1.0 - beta1**self.step_count
            bias_correction2_sqrt = (1.0 - beta2**self.step_count) ** 0.5
            denominator = (
                exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            )
            master.addcdiv_(
                exp_avg,
                denominator,
                value=-(self.learning_rate / bias_correction1),
            )
            self._zero_padding(self.master)
            self._zero_padding(self.exp_avg)
            self._zero_padding(self.exp_avg_sq)

    def state_dict(self) -> dict[str, object]:
        """Return a detached checkpoint with exact builtin state containers."""
        if type(self.amp_state) is not dict:
            raise ValueError("amp_state must be an exact dict")
        if type(self.rng_state) is not dict:
            raise ValueError("rng_state must be an exact dict")
        return {
            "master": self.master.clone(),
            "exp_avg": self.exp_avg.clone(),
            "exp_avg_sq": self.exp_avg_sq.clone(),
            "step_count": self.step_count,
            "learning_rate": self.learning_rate,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amp_state": deepcopy(self.amp_state),
            "rng_state": deepcopy(self.rng_state),
        }

    def load_state_dict(self, state: object) -> None:
        """Restore a validated checkpoint and force the next FP refresh."""
        if type(state) is not dict:
            raise ValueError("state must be an exact dict")
        expected_fields = {
            "master",
            "exp_avg",
            "exp_avg_sq",
            "step_count",
            "learning_rate",
            "betas",
            "eps",
            "weight_decay",
            "amp_state",
            "rng_state",
        }
        if set(state) != expected_fields:
            raise ValueError("state fields are invalid")
        master = _validated_shard_tensor(
            state["master"],
            self.layout,
            "master",
            device=self.master.device,
        )
        exp_avg = _validated_shard_tensor(
            state["exp_avg"],
            self.layout,
            "exp_avg",
            device=self.master.device,
        )
        exp_avg_sq = _validated_shard_tensor(
            state["exp_avg_sq"],
            self.layout,
            "exp_avg_sq",
            device=self.master.device,
        )
        _require_zero_padding(master, self.layout, "master")
        _require_zero_padding(exp_avg, self.layout, "exp_avg")
        _require_zero_padding(exp_avg_sq, self.layout, "exp_avg_sq")
        step_count = state["step_count"]
        if type(step_count) is not int or step_count < 0:
            raise ValueError("step_count must be a non-negative exact integer")
        learning_rate = _require_finite_float(
            state["learning_rate"],
            "learning_rate",
            positive=True,
        )
        betas = _require_betas(state["betas"])
        eps = _require_finite_float(state["eps"], "eps", positive=True)
        weight_decay = _require_finite_float(
            state["weight_decay"],
            "weight_decay",
            nonnegative=True,
        )
        amp_state = state["amp_state"]
        if type(amp_state) is not dict:
            raise ValueError("amp_state must be an exact dict")
        rng_state = state["rng_state"]
        if type(rng_state) is not dict:
            raise ValueError("rng_state must be an exact dict")

        self.master = master.clone()
        self.exp_avg = exp_avg.clone()
        self.exp_avg_sq = exp_avg_sq.clone()
        self.step_count = step_count
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amp_state = deepcopy(amp_state)
        self.rng_state = deepcopy(rng_state)
        self.force_refresh = True

    def _zero_padding(self, value: object) -> None:
        if self.layout.valid_numel < self.layout.padded_numel:
            value[self.layout.valid_numel :].zero_()


def flatten_parameter_copy(parameters: object) -> object:
    """Flatten CPU FP32 parameters into detached contiguous storage."""
    torch = _torch()
    if type(parameters) is not tuple:
        raise ValueError("parameters must be an exact tuple")
    flattened = [
        _validated_parameter(parameter, index)
        for index, parameter in enumerate(parameters)
    ]
    if not flattened:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(
        [parameter.detach().reshape(-1).clone() for parameter in flattened]
    )


def copy_flat_to_parameters(flat: object, parameters: object) -> None:
    """Copy a flat CPU FP32 vector back into same-sized parameters."""
    torch = _torch()
    if type(parameters) is not tuple:
        raise ValueError("parameters must be an exact tuple")
    if type(flat) is not torch.Tensor or flat.device.type != "cpu":
        raise ValueError("flat must be a CPU tensor")
    if flat.dtype is not torch.float32 or flat.ndim != 1:
        raise ValueError("flat must be a one-dimensional FP32 tensor")
    validated = [
        _validated_parameter(parameter, index)
        for index, parameter in enumerate(parameters)
    ]
    total_numel = sum(parameter.numel() for parameter in validated)
    if flat.numel() != total_numel:
        raise ValueError("flat has an invalid numel")
    offset = 0
    with torch.no_grad():
        for parameter in validated:
            next_offset = offset + parameter.numel()
            parameter.copy_(flat[offset:next_offset].reshape_as(parameter))
            offset = next_offset


def _torch() -> object:
    try:
        return import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ShardedAdamW requires the optional torch package"
        ) from error


def _trusted_shard_layout(value: object) -> ShardLayout:
    if type(value) is not ShardLayout:
        raise ValueError("layout must be an exact ShardLayout")
    _validate_shard_layout(value, "layout")
    return value


def _validate_shard_layout(layout: ShardLayout, name: str) -> None:
    _require_nonnegative_int(layout.global_numel, f"{name}.global_numel")
    _require_positive_int(layout.world_size, f"{name}.world_size")
    if (
        type(layout.rank) is not int
        or layout.rank < 0
        or layout.rank >= layout.world_size
    ):
        raise ValueError(f"{name}.rank is outside world_size")
    _require_nonnegative_int(layout.start, f"{name}.start")
    _require_nonnegative_int(layout.valid_numel, f"{name}.valid_numel")
    _require_nonnegative_int(layout.padded_numel, f"{name}.padded_numel")
    expected_padded_numel = _checked_ceil_div(
        layout.global_numel,
        layout.world_size,
    )
    if layout.padded_numel != expected_padded_numel:
        raise ValueError(f"{name}.padded_numel is inconsistent")
    expected_start = min(
        _checked_mul(layout.rank, layout.padded_numel),
        layout.global_numel,
    )
    if layout.start != expected_start:
        raise ValueError(f"{name}.start is inconsistent")
    expected_valid_numel = min(
        layout.padded_numel,
        layout.global_numel - layout.start,
    )
    if layout.valid_numel != expected_valid_numel:
        raise ValueError(f"{name}.valid_numel is inconsistent")


def _validated_shard_tensor(
    value: object,
    layout: ShardLayout,
    name: str,
    *,
    device: object | None = None,
    require_finite: bool = True,
) -> object:
    torch = _torch()
    if type(value) is not torch.Tensor:
        raise ValueError(f"{name} must be an exact tensor")
    if value.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be a CPU or CUDA tensor")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the optimizer state device")
    if value.dtype is not torch.float32 or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional FP32 tensor")
    if value.numel() != layout.padded_numel:
        raise ValueError(f"{name} has an invalid numel")
    if require_finite and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    return value


def _validated_parameter(value: object, index: int) -> object:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"parameter {index} must be a tensor")
    if value.device.type != "cpu" or value.dtype is not torch.float32:
        raise ValueError(f"parameter {index} must be a CPU FP32 tensor")
    return value


def _require_zero_padding(
    value: object, layout: ShardLayout, name: str
) -> None:
    if layout.valid_numel == layout.padded_numel:
        return
    if value[layout.valid_numel :].count_nonzero().item() != 0:
        raise ValueError(f"{name} padding must be zero")


def _require_finite_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_betas(value: object) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("betas must be an exact two-item tuple")
    beta1 = _require_finite_float(value[0], "betas")
    beta2 = _require_finite_float(value[1], "betas")
    if beta1 < 0.0 or beta1 >= 1.0 or beta2 < 0.0 or beta2 >= 1.0:
        raise ValueError("betas must be in [0, 1)")
    return beta1, beta2


def _require_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")
    if value > _MAX_SIGNED_64:
        raise OverflowError(f"{name} exceeds the signed 64-bit domain")


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer")
    if value > _MAX_SIGNED_64:
        raise OverflowError(f"{name} exceeds the signed 64-bit domain")


def _checked_ceil_div(value: int, divisor: int) -> int:
    quotient, remainder = divmod(value, divisor)
    return quotient + (remainder != 0)


def _checked_mul(left: int, right: int) -> int:
    result = left * right
    if result > _MAX_SIGNED_64:
        raise OverflowError("shard layout size overflow")
    return result
