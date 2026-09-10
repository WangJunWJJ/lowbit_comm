"""Offline, uniquely sampled evaluation of trusted, hash-bound PSI checkpoints.

Run under torchrun after training finishes, with a new output path. Saved Docker
commands are parsed as provenance only, never executed. This does not resume
training, modify checkpoints, rewrite old losses or qualify the adapter.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from tests.benchmarks.psi_checkpoint_evaluation import (
    preserved_evaluation, restore_model_weights, validate_evaluation_identity, validate_rsag_engine_identity,
)
from tests.benchmarks.psi_unique_validation import (
    build_unique_validation_loader, evaluate_validation_shard, merge_validation_shards,
)

EVALUATOR_FILES = (
    "tests/benchmarks/distributed_psi_v040_evaluate.py",
    "tests/benchmarks/psi_checkpoint_evaluation.py",
    "tests/benchmarks/psi_unique_validation.py",
    "tests/benchmarks/distributed_psi_v040_worker.py",
    "tests/benchmarks/psi_prefetch.py",
    "tests/benchmarks/psi_training_runtime.py",
    "tests/benchmarks/psi_v040_training.py",
)


def verify_file(record: dict) -> Path:
    path = Path(record["path"])
    expected = record["sha256"]
    if (not path.is_absolute() or type(expected) is not str or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError("invalid bound file path/hash")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("bound file hash mismatch: " + str(path))
    return path


def verify_input_tree(root: Path, hashes: dict) -> None:
    if not root.is_absolute() or type(hashes) is not dict or not hashes:
        raise ValueError("input tree needs an absolute path and nonempty manifest")
    resolved = root.resolve()
    for relative, expected in hashes.items():
        if type(relative) is not str or not relative or "\\" in relative:
            raise ValueError("invalid input path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("input path escapes bound root")
        path = root / relative_path
        if not path.resolve().is_relative_to(resolved):
            raise ValueError("input path escapes bound root")
        verify_file({"path": str(path), "sha256": expected})


def training_args_from_command(command: list):
    from tests.benchmarks.psi_v040_training import parse_args
    marker = "tests/benchmarks/distributed_psi_v040_worker.py"
    if type(command) is not list or any(type(arg) is not str for arg in command) or command.count(marker) != 1:
        raise ValueError("expected one saved PSI worker command")
    return parse_args(command[command.index(marker)+1:])


def load_plan(path: str, expected_sha256: str):
    from tests.benchmarks.psi_v040_training import validate_task_result
    from tests.benchmarks.psi_training_runtime import training_protocol

    plan = json.loads(verify_file({"path": path, "sha256": expected_sha256}).read_text(encoding="utf-8"))
    fields = {"schema_version", "training_command", "training_result", "training_protocol",
              "checkpoints", "input_manifest", "psi_root", "data_root", "normalizer_path",
              "evaluation", "evaluator_files"}
    if type(plan) is not dict or set(plan) != fields or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("invalid evaluation plan schema")
    args = training_args_from_command(json.loads(verify_file(plan["training_command"]).read_text(encoding="utf-8")))
    result = validate_task_result(json.loads(verify_file(plan["training_result"]).read_text(encoding="utf-8")))
    protocol = json.loads(verify_file(plan["training_protocol"]).read_text(encoding="utf-8"))
    if (result["schema_version"] != 3 or protocol.get("training_protocol") != result["execution_protocol"]
            or protocol.get("result_sha256") != plan["training_result"]["sha256"]):
        raise ValueError("original result/protocol binding mismatch")
    expected_protocol = training_protocol(
        rank=0, world_size=result["world_size"], batch_size=args.batch_size,
        native_ddp_mode=args.native_ddp_mode, timing_mode=args.timing_mode,
        data_mode=args.data_mode, loader_workers=args.loader_workers,
        loader_prefetch_factor=args.loader_prefetch_factor, seed=args.seed,
        model_precision=args.model_precision,
    )
    if (expected_protocol != result["execution_protocol"] or args.route != result["route"]
            or args.seed != result["seed"] or args.epochs != result["epochs"]
            or args.data_sha256 != result["data_sha256"]):
        raise ValueError("training command/result identity mismatch")
    if Path(args.psi_source) != Path(plan["psi_root"]):
        raise ValueError("PSI source path mismatch")
    roots = [item for item in args.psi_override if item.lstrip("+").startswith("dataset.data_root=")]
    if roots != ["dataset.data_root="+plan["data_root"]]:
        raise ValueError("dataset root override mismatch")
    checkpoints = plan["checkpoints"]
    if (type(checkpoints) is not list or len(checkpoints) != result["world_size"]
            or any(type(item.get("rank")) is not int for item in checkpoints)
            or sorted(item["rank"] for item in checkpoints) != list(range(result["world_size"]))):
        raise ValueError("checkpoint rank inventory mismatch")
    for record in checkpoints:
        verify_file(record)
    inputs = json.loads(verify_file(plan["input_manifest"]).read_text(encoding="utf-8"))
    verify_input_tree(Path(plan["psi_root"]), inputs["psi"])
    verify_input_tree(Path(plan["data_root"]), inputs["data"])
    verify_file({"path": plan["normalizer_path"], "sha256": inputs["normalizer"]})
    if not set(EVALUATOR_FILES).issubset(plan["evaluator_files"]):
        raise ValueError("evaluation source binding is incomplete")
    verify_input_tree(Path(__file__).resolve().parents[2], plan["evaluator_files"])
    scope = plan["evaluation"]
    if (type(scope) is not dict or set(scope) != {"seed", "epoch", "workers", "batch_size"}
            or any(type(value) is not int for value in scope.values())
            or not 0 <= scope["seed"] < 2**63 or scope["epoch"] < 0
            or scope["workers"] < 0 or scope["batch_size"] < 1):
        raise ValueError("invalid offline evaluation scope")
    return plan, args, result


def evaluate_model(model, source_loader, *, rank, world_size, seed, epoch, workers, device):
    import torch
    from tests.benchmarks import distributed_psi_v040_worker as worker

    loader = build_unique_validation_loader(
        source_loader, rank=rank, world_size=world_size, seed=seed, epoch=epoch, num_workers=workers,
    )
    with preserved_evaluation(model, seed=seed):
        def loss_fn(batch):
            value = worker._to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(value, training=True)
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError("model must return one scalar batch-mean loss")
            return float(loss.detach().double().item())

        return evaluate_validation_shard(
            loader, dataset_size=len(source_loader.dataset), rank=rank, world_size=world_size,
            loss_fn=loss_fn, sample_count_fn=worker._batch_sample_count,
        )


def write_new_result(path: Path, result: dict) -> None:
    serialized = json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def run(plan_path: str, plan_sha256: str, output: Path) -> None:
    import torch
    from tests.benchmarks import distributed_psi_v040_worker as worker

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        message = [None]
        if rank == 0:
            try:
                if output.exists():
                    raise FileExistsError(str(output))
                message[0] = {"loaded": load_plan(plan_path, plan_sha256)}
            except Exception as error:
                message[0] = {"error": repr(error)}
        torch.distributed.broadcast_object_list(message, src=0)
        if "error" in message[0]:
            raise ValueError("evaluation preflight failed: " + message[0]["error"])
        plan, args, original = message[0]["loaded"]
        if world != original["world_size"]:
            raise ValueError("evaluation world size differs from training checkpoint")
        if os.environ.get("PSI_NORMALIZER_CACHE") != plan["normalizer_path"]:
            raise ValueError("normalizer environment does not match bound input")
        record = next(item for item in plan["checkpoints"] if item["rank"] == rank)
        checkpoint_path = verify_file(record)
        # The manifest must describe our own trusted checkpoints: pickle is not
        # safe for untrusted uploads, and a checksum is not authentication.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        worker._validate_checkpoint_payload(payload, route=args.route)
        expected_protocol = dict(original["execution_protocol"], rank=rank)
        validate_evaluation_identity(payload, route=args.route, epoch=original["epochs"],
                                     step=original["steps"], training_protocol=expected_protocol)
        if payload["step_in_epoch"] != 0:
            raise ValueError("offline final evaluation requires an epoch-boundary checkpoint")
        if args.route == "rsag_qwd":
            validate_rsag_engine_identity(payload["engine"], gradient_route=args.rsag_gradient_route,
                                           parameter_route=args.rsag_parameter_route, rank=rank, world_size=world)
        verify_file(record)
        workspace, _, _, old_val_loader, _ = worker._build_workspace(args, rank, world)
        model = workspace.model
        worker._convert_model_precision(model, args.model_precision)
        restore_model_weights(model, payload["model"], route=args.route)
        model_sha256 = worker._state_sha256(model.state_dict())
        del payload
        device = torch.device("cuda", local_rank)
        model.to(device)
        original_loader = getattr(old_val_loader, "_ccdl_source_loader", old_val_loader)
        scope = plan["evaluation"]
        source_loader = torch.utils.data.DataLoader(
            original_loader.dataset, batch_size=scope["batch_size"],
            collate_fn=original_loader.collate_fn, pin_memory=original_loader.pin_memory,
        )
        started = time.perf_counter()
        shard = evaluate_model(model, source_loader, rank=rank, world_size=world,
                               seed=scope["seed"], epoch=scope["epoch"], workers=scope["workers"], device=device)
        if worker._state_sha256(model.state_dict()) != model_sha256:
            raise ValueError("evaluation model state hash changed")
        evidence = {"rank": rank, "shard": shard, "checkpoint": record,
                    "model_state_sha256": model_sha256, "evaluation_wall_s": time.perf_counter()-started,
                    "dataset_size": len(source_loader.dataset), "device_name": torch.cuda.get_device_name(local_rank)}
        gathered = [None for _ in range(world)]
        torch.distributed.all_gather_object(gathered, evidence)
        if len({item["dataset_size"] for item in gathered}) != 1:
            raise ValueError("rank validation dataset sizes differ")
        result = merge_validation_shards([item["shard"] for item in gathered],
                                          dataset_size=evidence["dataset_size"], world_size=world)
        if rank == 0:
            result.update(schema_version=1, plan_sha256=plan_sha256, original_result=plan["training_result"],
                          original_validation_loss=original["validation_loss"], route=args.route,
                          gradient_route=args.rsag_gradient_route if args.route == "rsag_qwd" else args.route,
                          training_seed=args.seed, evaluation=scope, world_size=world,
                          checkpoints=gathered, torch_version=str(torch.__version__),
                          model_precision=args.model_precision, evaluator_files=plan["evaluator_files"],
                          input_manifest=plan["input_manifest"], training_state_restored=False,
                          validation_split_independence="index_disjointness_requires_separate_audit",
                          comparison_limit="New index order/batch geometry changes stochastic loss draws; not a pure padding ablation.")
            write_new_result(output, result)
            print(json.dumps({"output": str(output), "samples": result["sample_count"],
                              "validation_loss": result["validation_loss"]}), flush=True)
    finally:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256, args.output)


if __name__ == "__main__":
    main()
