from dataclasses import dataclass
from pathlib import Path


MAIN_VARIANTS = {
    "nccl_fp32": (None, None),
    "int8_k0": (8, 0),
    "int8_k2": (8, 2),
    "int4_k0": (4, 0),
    "int4_k2": (4, 2),
}
SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class RunConfig:
    variant: str
    seed: int
    output_dir: Path
    max_length: int = 256
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    group_size: int = 64
    bit: int | None = None
    topk: int | None = None

    def __post_init__(self):
        if self.variant not in MAIN_VARIANTS:
            raise ValueError(f"unknown variant: {self.variant}")


def expand_main_matrix(root: Path) -> list[RunConfig]:
    result = []
    for variant, (bit, topk) in MAIN_VARIANTS.items():
        for seed in SEEDS:
            result.append(
                RunConfig(
                    variant=variant,
                    seed=seed,
                    output_dir=Path(root) / f"{variant}-seed{seed}",
                    bit=bit,
                    topk=topk,
                )
            )
    return result
