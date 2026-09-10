"""Evidence binding and actual CPU model execution for the offline evaluator."""
from hashlib import sha256
import json

import pytest

from tests.benchmarks import distributed_psi_v040_evaluate as evaluator

torch = pytest.importorskip("torch")


def test_bound_file_rejects_changed_content(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    digest = sha256(path.read_bytes()).hexdigest()
    assert evaluator.verify_file({"path": str(path), "sha256": digest}) == path
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        evaluator.verify_file({"path": str(path), "sha256": digest})


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "x/../../outside"])
def test_input_tree_rejects_escape(tmp_path, relative):
    with pytest.raises(ValueError, match="path"):
        evaluator.verify_input_tree(tmp_path, {relative: "0"*64})


def test_input_tree_is_not_an_empty_attestation(tmp_path):
    with pytest.raises(ValueError):
        evaluator.verify_input_tree(tmp_path, {})


def test_training_command_is_parsed_not_executed():
    args = evaluator.training_args_from_command([
        "docker", "run", "--rm", "image", "torchrun", "--standalone",
        "tests/benchmarks/distributed_psi_v040_worker.py", "--route", "cag", "--seed", "20260821",
    ])
    assert args.route == "cag" and args.seed == 20260821
    with pytest.raises(ValueError):
        evaluator.training_args_from_command(["echo", "not a worker command"])


class _LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer("normalizer", torch.tensor(3.0))

    def forward(self, batch, *, training):
        assert training and not self.training and not torch.is_grad_enabled()
        return ((batch*self.weight+self.normalizer)**2).mean()


def test_actual_model_evaluation_is_unique_weighted_and_read_only():
    model = _LossModel().train()
    source = torch.utils.data.DataLoader(torch.arange(7, dtype=torch.float32), batch_size=2, drop_last=True)
    records = [evaluator.evaluate_model(
        model, source, rank=rank, world_size=3, seed=21, epoch=0, workers=0, device=torch.device("cpu"),
    ) for rank in range(3)]
    from tests.benchmarks.psi_unique_validation import merge_validation_shards
    merged = merge_validation_shards(records, dataset_size=7, world_size=3)
    assert merged["sample_count"] == 7
    assert merged["validation_loss"] == pytest.approx(sum((2*i+3)**2 for i in range(7))/7)
    assert model.training and model.weight.item() == 2 and model.normalizer.item() == 3
    assert model.weight.grad is None


def test_result_writer_never_overwrites_prior_evidence(tmp_path):
    path = tmp_path / "evaluation.json"
    evaluator.write_new_result(path, {"value": 1})
    with pytest.raises(FileExistsError):
        evaluator.write_new_result(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def _bound(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _plan(tmp_path):
    from pathlib import Path
    from tests.unit.benchmarks.test_psi_v040_training import _task_result
    from tests.benchmarks.psi_training_runtime import training_protocol
    result = _task_result()
    size = result["batch_size_per_rank"]
    protocol = training_protocol(rank=0, world_size=4, batch_size=size, native_ddp_mode="standard",
                                 timing_mode="production", data_mode="deterministic", seed=20260821)
    result.update(schema_version=3, execution_protocol=protocol)
    result_record = _bound(tmp_path/"training-result.json", result)
    protocol_record = _bound(tmp_path/"training-protocol.json", {
        "training_protocol": protocol, "result_sha256": result_record["sha256"],
    })
    psi, data = tmp_path/"psi", tmp_path/"data"
    psi.mkdir()
    data.mkdir()
    source = _bound(psi/"source.json", {"source": "fixture"})
    dataset = _bound(data/"dataset.json", {"dataset": "fixture"})
    normalizer = _bound(tmp_path/"normalizer.json", {"normalizer": "fixture"})
    inputs = _bound(tmp_path/"inputs.json", {
        "psi": {"source.json": source["sha256"]}, "data": {"dataset.json": dataset["sha256"]},
        "normalizer": normalizer["sha256"],
    })
    command = _bound(tmp_path/"command.json", [
        "docker", "tests/benchmarks/distributed_psi_v040_worker.py", "--route", "native",
        "--psi-source", str(psi), "--psi-override", "dataset.data_root="+str(data),
        "--batch-size", str(size), "--epochs", "3", "--seed", "20260821",
        "--timing-mode", "production", "--data-mode", "deterministic", "--data-sha256", "e"*64,
    ])
    checkpoints = [dict(_bound(tmp_path/f"rank{rank}.json", {"rank": rank}), rank=rank) for rank in range(4)]
    root = Path(evaluator.__file__).resolve().parents[2]
    code = {name: sha256((root/name).read_bytes()).hexdigest() for name in evaluator.EVALUATOR_FILES}
    plan = {"schema_version": 1, "training_command": command, "training_result": result_record,
            "training_protocol": protocol_record, "input_manifest": inputs, "checkpoints": checkpoints,
            "psi_root": str(psi), "data_root": str(data), "normalizer_path": normalizer["path"],
            "evaluation": {"seed": 20260821, "epoch": 0, "workers": 0, "batch_size": 256},
            "evaluator_files": code}
    return plan


def test_plan_binds_original_result_protocol_command_inputs_and_evaluator(tmp_path):
    plan = _plan(tmp_path)
    record = _bound(tmp_path/"plan.json", plan)
    loaded, args, result = evaluator.load_plan(record["path"], record["sha256"])
    assert loaded == plan and args.route == result["route"] == "native"


@pytest.mark.parametrize("mutation", ["seed", "protocol", "rank", "code", "data_root", "empty_inputs"])
def test_plan_rejects_inconsistent_evidence(tmp_path, mutation):
    plan = _plan(tmp_path)
    if mutation == "seed":
        from pathlib import Path
        record = plan["training_command"]
        cmd = json.loads(Path(record["path"]).read_text())
        cmd[cmd.index("--seed")+1] = "20260822"
        plan["training_command"] = _bound(Path(record["path"]), cmd)
    elif mutation == "protocol":
        from pathlib import Path
        record = plan["training_protocol"]
        proto = json.loads(Path(record["path"]).read_text())
        proto["result_sha256"] = "0"*64
        plan["training_protocol"] = _bound(Path(record["path"]), proto)
    elif mutation == "rank":
        plan["checkpoints"][1]["rank"] = 0
    elif mutation == "code":
        plan["evaluator_files"] = {}
    elif mutation == "data_root":
        plan["data_root"] = str(tmp_path/"other")
    else:
        from pathlib import Path
        record = plan["input_manifest"]
        inputs = json.loads(Path(record["path"]).read_text())
        inputs["data"] = {}
        plan["input_manifest"] = _bound(Path(record["path"]), inputs)
    record = _bound(tmp_path/"plan.json", plan)
    with pytest.raises(ValueError):
        evaluator.load_plan(record["path"], record["sha256"])


def test_run_orchestration_loads_real_checkpoint_and_evaluates_on_cpu(tmp_path, monkeypatch):
    """Simulate NCCL/CUDA control only; use real pickle/schema/model/data/loss."""
    from pathlib import Path
    from types import SimpleNamespace
    from tests.benchmarks import distributed_psi_v040_worker as worker
    plan = _plan(tmp_path)
    protocol = json.loads(Path(plan["training_protocol"]["path"]).read_text())["training_protocol"]
    checkpoint_model = _LossModel()
    worker._convert_model_precision(checkpoint_model, "fp16")
    wrapped = torch.nn.Module()
    wrapped.add_module("module", checkpoint_model)
    scaler = {"scale": 512.0, "growth_factor": 2.0, "backoff_factor": 0.5,
              "growth_interval": 2000, "_growth_tracker": 0}
    for rank in range(4):
        path = tmp_path/f"checkpoint-{rank}.pt"
        payload = {"training_protocol": dict(protocol, rank=rank), "route": "native", "epoch": 3,
                   "step": 3, "step_in_epoch": 0, "next_batch_indices": (), "model": wrapped.state_dict(),
                   "engine": {}, "scheduler": {}, "scaler": scaler,
                   "amp_configuration": worker._amp_configuration_dict(initial_scale=512.0,
                           effective_start_scale=512.0, scaler_state=scaler),
                   "warmup_batch": torch.ones(2), "loader_rng": (), "rng": {}}
        torch.save(payload, path)
        plan["checkpoints"][rank] = {"rank": rank, "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    bound_plan = _bound(tmp_path/"plan.json", plan)
    events = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("PSI_NORMALIZER_CACHE", plan["normalizer_path"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: events.append("set_device"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU test stub")
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *a, **k: events.append("init"))
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: events.append("destroy"))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: events.append("broadcast"))

    class CPUModel(_LossModel):
        def to(self, device):
            assert device.type == "cuda"
            return self

    model = CPUModel()
    source = torch.utils.data.DataLoader(torch.arange(7, dtype=torch.float32), batch_size=2)
    old_loader = SimpleNamespace(_ccdl_source_loader=source)
    monkeypatch.setattr(worker, "_build_workspace", lambda *a: (SimpleNamespace(model=model), None, None, old_loader, None))
    actual_evaluate = evaluator.evaluate_model

    def cpu_evaluate(model, source, **kwargs):
        kwargs["device"] = torch.device("cpu")
        return actual_evaluate(model, source, **kwargs)
    monkeypatch.setattr(evaluator, "evaluate_model", cpu_evaluate)

    def gather(output, local):
        events.append("gather")
        for rank in range(4):
            shard = actual_evaluate(model, source, rank=rank, world_size=4, seed=20260821,
                                   epoch=0, workers=0, device=torch.device("cpu"))
            output[rank] = dict(local, rank=rank, shard=shard, checkpoint=plan["checkpoints"][rank])
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    output = tmp_path/"new-evaluation.json"
    evaluator.run(bound_plan["path"], bound_plan["sha256"], output)
    result = json.loads(output.read_text())
    assert result["sample_count"] == 7
    assert result["validation_loss"] == pytest.approx(sum((2*i+3)**2 for i in range(7))/7)
    assert result["training_state_restored"] is False
    assert events == ["set_device", "init", "broadcast", "gather", "destroy"]
