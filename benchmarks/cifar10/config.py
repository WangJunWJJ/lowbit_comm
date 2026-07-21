from dataclasses import asdict, dataclass
from pathlib import Path


SEEDS = (1337, 2027, 4099)
MAIN_VARIANTS = {
    "nccl_fp32": (None, None),
    "ccdl_int8_k0": (8, 0),
    "ccdl_int8_k2": (8, 2),
    "ccdl_int4_k0": (4, 0),
    "ccdl_int4_k2": (4, 2),
}


@dataclass(frozen=True)
class RunConfig:
    variant: str
    seed: int
    output_dir: Path
    epochs: int = 200
    batch_size_per_rank: int = 128
    workers_per_rank: int = 4
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    bit: int | None = None
    topk: int | None = None
    group_size: int = 64
    stochastic: bool = False

    @property
    def run_id(self) -> str:
        return f"{self.variant}-seed{self.seed}"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result


def expand_main_matrix(output_root: Path) -> list[RunConfig]:
    return [
        RunConfig(
            variant=name,
            seed=seed,
            output_dir=output_root / f"{name}-seed{seed}",
            bit=bit,
            topk=topk,
        )
        for seed in SEEDS
        for name, (bit, topk) in MAIN_VARIANTS.items()
    ]
