import pytest
import tests.benchmarks.distributed_psi_v040_worker as worker

from tests.benchmarks.psi_training_runtime import (
    PhaseTimer, TrainingLoopWindowObserver, training_protocol,
    validate_training_protocol, execution_counters,
)
from tests.benchmarks.psi_v040_training import parse_args
from tests.benchmarks.psi_v040_training import StepTiming, build_step_record, validate_step_record, validate_task_result
from tests.benchmarks.distributed_psi_v040_worker import (
    commit_observed_step, observed_process_wall, summarize_window_records, validate_window_rank_coverage,
    window_audit_required, window_sidecar_fields,
)
from tests.unit.benchmarks.test_psi_v040_training import _task_result


class Cuda:
    def __init__(self): self.syncs=[]
    def synchronize(self,device=None): self.syncs.append(device)


class Loss:
    def __init__(self,value,reads): self.value=value; self.reads=reads
    def detach(self): return self
    def item(self): self.reads.append(self.value); return self.value
    def __float__(self): self.reads.append(self.value); return self.value


@pytest.mark.parametrize("timing_mode", ["diagnostic", "production", "window"])
def test_commit_preserves_loss_audit_oracle_append_midpoint_order(timing_mode):
    order=[]; records=[]
    class Observer:
        is_open=False
        def observe(self, **kwargs): order.append("observe")
        def begin(self, **kwargs): order.append("reopen")
    observer = Observer() if timing_mode == "window" else None
    class OrderedLoss(Loss):
        def detach(self): order.append("loss"); return self
    def build(loss, audit):
        order.append("build")
        return {"quality":{"loss":loss}, "batch_indices":[1]}
    def post(record, loss, audit):
        assert records == [record]
        order.append("midpoint")
    commit_observed_step(
        timing_mode=timing_mode, observer=observer, records=records,
        loss=OrderedLoss(0.5, []), audit_performed=True,
        quality_audit=lambda: order.append("audit") or "quality",
        bookkeeping=lambda loss, audit: order.append("oracle"),
        build_record=build, post_append=post, epoch=0, global_step=1,
        samples=1, final_batch=False, training_stop=False,
    )
    expected = ["loss", "audit", "oracle", "build", "midpoint"]
    if timing_mode == "window": expected += ["observe", "reopen"]
    assert order == expected


def test_commit_loss_failure_prevents_quality_audit():
    error=RuntimeError("loss read failed"); order=[]
    class BrokenLoss:
        def detach(self): raise error
    with pytest.raises(RuntimeError) as caught:
        commit_observed_step(
            timing_mode="diagnostic", observer=None, records=[], loss=BrokenLoss(),
            audit_performed=True, quality_audit=lambda: order.append("audit"),
            bookkeeping=lambda loss, audit: None,
            build_record=lambda loss, audit: {}, post_append=lambda *args: None,
            epoch=0, global_step=1, samples=1, final_batch=True,
            training_stop=False,
        )
    assert caught.value is error and order == []


def test_commit_build_failure_prevents_append_and_midpoint():
    error=RuntimeError("build failed"); order=[]; records=[]
    def fail_build(loss, audit): order.append("build"); raise error
    with pytest.raises(RuntimeError) as caught:
        commit_observed_step(
            timing_mode="window", observer=None, records=records,
            loss=Loss(0.5, []), audit_performed=True,
            quality_audit=lambda: order.append("audit"),
            bookkeeping=lambda loss, audit: order.append("oracle"),
            build_record=fail_build,
            post_append=lambda *args: order.append("midpoint"), epoch=0,
            global_step=1, samples=1, final_batch=True, training_stop=False,
        )
    assert caught.value is error
    assert order == ["audit", "oracle", "build"] and records == []


def test_epoch_oracle_consumes_hashes_returned_by_last_observed_step(tmp_path):
    observed_hashes={
        "rank_gap":0.0, "model_sha256":"model-current",
        "optimizer_sha256":"optimizer-current", "batch_sha256":"batch-current",
        "augmentation_sha256":"augmentation-current", "quality_s":0.25,
    }
    record, observed = commit_observed_step(
        timing_mode="production", observer=None, records=[], loss=Loss(0.5, []),
        audit_performed=True, quality_audit=lambda: observed_hashes,
        bookkeeping=lambda loss, audit: None,
        build_record=lambda loss, audit: {"quality":{"loss":loss}},
        post_append=lambda *args: None, epoch=0, global_step=1, samples=1,
        final_batch=True, training_stop=False,
    )
    pending = worker.prepare_epoch_resume_oracle_observation(
        path=tmp_path / "epoch.oracle.json", next_batch_indices=(7, 8),
        learning_rate=0.125, amp_scale=1024.0, observed=observed,
    )
    assert record["quality"]["loss"] == 0.5
    assert pending[4:] == ("optimizer-current", "model-current")
    assert "prepare_epoch_resume_oracle_observation" in worker._run.__code__.co_names
    assert "last_observed" in worker._run.__code__.co_varnames
    assert not {"optimizer_sha256", "model_sha256"}.intersection(
        worker._run.__code__.co_names
    )


def test_window_protocol_and_cli_are_explicit_and_old_protocol_unchanged():
    old=training_protocol(rank=0,world_size=2,batch_size=16,native_ddp_mode="standard")
    assert old["version"]==3 and old["measurement"]=="synchronized_phase_diagnostic"
    value=training_protocol(rank=0,world_size=2,batch_size=16,native_ddp_mode="standard",timing_mode="window")
    assert value["version"]==4 and value["measurement"]=="training_loop_window_wall"
    assert value["phase_breakdown_available"] is False and value["qualification_eligible"] is False
    validate_training_protocol(value)
    assert parse_args(["--route","native","--timing-mode","window"]).timing_mode=="window"


def test_window_phase_timer_adds_no_per_step_sync_or_events():
    cuda=Cuda(); timer=PhaseTimer("window",lambda:cuda)
    loss,result,*times=timer.training_step(lambda:"loss",lambda value:None,lambda:{"ok":True})
    assert (loss,result)==("loss",{"ok":True}) and times==[0.0,0.0,0.0] and cuda.syncs==[]


def test_window_lifecycle_defers_losses_and_closes_warmup_epoch_partial_windows():
    cuda=Cuda(); ticks=iter([10.0,12.0,20.0,25.0,30.0,31.5]); reads=[]
    observer=TrainingLoopWindowObserver(cuda_provider=lambda:cuda,device="cuda:1",warmup_steps=2,clock=lambda:next(ticks))
    observer.begin(epoch=0,local_record_start=0,global_step_start=7)
    rows=[]
    for index,value in enumerate((1.25,2.5),start=1):
        row={"quality":{"loss":None}}; rows.append(row)
        observer.observe(loss=Loss(value,reads),record=row,audit=False,epoch=0,local_record_end=index,global_step_end=7+index,samples=4)
    assert reads==[1.25,2.5] and rows[0]["quality"]["loss"]==1.25
    observer.begin(epoch=0,local_record_start=2,global_step_start=9)
    row={"quality":{"loss":None}}; observer.observe(loss=Loss(3.75,reads),record=row,audit=False,epoch=0,local_record_end=3,global_step_end=10,samples=3)
    observer.close("epoch")
    assert row["quality"]["loss"]==3.75
    assert [w["kind"] for w in observer.windows]==["warmup","epoch"]
    assert observer.windows[0]["local_record_range"]==[0,2]
    assert observer.windows[1]["samples"]==3 and cuda.syncs==["cuda:1"]*4


def test_window_audit_loss_is_immediate_and_failure_does_not_publish_window():
    cuda=Cuda(); reads=[]; observer=TrainingLoopWindowObserver(lambda:cuda,"cuda:0",2,clock=iter([1.0,2.0]).__next__)
    observer.begin(epoch=0,local_record_start=0,global_step_start=0)
    row={"quality":{"loss":None}}
    observer.observe(loss=Loss(4.0,reads),record=row,audit=True,epoch=0,local_record_end=1,global_step_end=1,samples=2)
    assert reads==[4.0] and row["quality"]["loss"]==4.0
    observer.abort()
    assert observer.windows==[] and cuda.syncs==["cuda:0"]


def test_real_worker_window_summary_path_flushes_exact_losses_and_keeps_step_timing_null():
    cuda=Cuda(); reads=[]; observer=TrainingLoopWindowObserver(lambda:cuda,"cuda:0",0,clock=iter([1.0,3.0]).__next__)
    observer.begin(epoch=0,local_record_start=0,global_step_start=4)
    records=[]
    for offset,value in enumerate((0.75,0.5),start=1):
        record=build_step_record(task_id="task",attempt_id="attempt",route="native",seed=20260821,
            epoch=0,step=4+offset,batch_indices=(offset,),timing=StepTiming(0.0,0.0,0.0,0.0,0.0,0.0),
            gradient_route="native",parameter_route="native",communication_bytes=0,qwd_s=0.0,
            refresh_s=0.0,decision="native",loss=None,amp_scale=1.0,learning_rate=0.1,
            model_sha256=None,rank_parameter_gap=None,optimizer_step=offset,finite=True,
            audit_performed=False,defer_loss_validation=True,window_timing=True)
        records.append(record)
        observer.observe(loss=Loss(value,reads),record=record,audit=False,epoch=0,
                         local_record_end=offset,global_step_end=4+offset,samples=1)
    assert reads==[]
    observer.close("epoch")
    assert reads==[0.75,0.5]
    assert all(value is None for value in records[0]["timing"].values())
    assert records[0]["communication"]["qwd_s"] is None and records[0]["communication"]["refresh_s"] is None
    assert validate_step_record(records[0]) is records[0]
    assert summarize_window_records(tuple(records))["loss_trajectory"]==(0.75,0.5)


def test_window_raw_schema_rejects_mixed_numeric_step_timing():
    record=build_step_record(task_id="task",attempt_id="attempt",route="native",seed=20260821,
        epoch=0,step=1,batch_indices=(1,),timing=StepTiming(0.0,0.0,0.0,0.0,0.0,0.0),gradient_route="native",
        parameter_route="native",communication_bytes=0,qwd_s=0.0,refresh_s=0.0,decision="native",
        loss=1.0,amp_scale=1.0,learning_rate=0.1,model_sha256=None,rank_parameter_gap=None,
        optimizer_step=1,finite=True,audit_performed=False,window_timing=True)
    record["timing"]["measured_s"]=0.0
    with pytest.raises(ValueError,match="unavailable"): validate_step_record(record)


def test_window_result_schema_has_null_step_metrics_and_exact_window_throughput():
    result=_task_result(); result["schema_version"]=4
    result["execution_protocol"]=training_protocol(rank=0,world_size=4,batch_size=16,native_ddp_mode="standard",timing_mode="window")
    result.update(epochs=1,steady_samples_per_second=None,step_latency_p50_ms=None,step_latency_p95_ms=None,
                  epoch_core_time_s=None,communication_time_s=None,qwd_time_s=None,refresh_time_s=None,training_loop_windows=[
                    {"epoch":0,"kind":"warmup","local_record_range":[0,1],"global_step_range":[0,1],"samples":16,"elapsed_wall_s":1.0},
                    {"epoch":0,"kind":"epoch","local_record_range":[1,3],"global_step_range":[1,3],"samples":32,"elapsed_wall_s":2.0}],
                  steady_training_loop_window_s=2.0,steady_training_loop_samples_per_second=64.0,
                  window_observability={"measurement":"training_loop_window_wall","single_step_timing_available":False,"phase_breakdown_available":False,"losses_flushed":True,
                                        "start_epoch":0,"start_global_step":0,"local_record_samples":[16,16,16]})
    result["timing_breakdown"]["data_s"]=None; result["timing_breakdown"]["core_train_s"]=None
    assert validate_task_result(result) is result
    result["step_latency_p50_ms"]=1.0
    with pytest.raises(ValueError,match="null"): validate_task_result(result)


def test_result_schema_protocol_pairing_and_window_geometry_fail_closed():
    legacy=_task_result(); legacy["schema_version"]=3
    legacy["execution_protocol"]=training_protocol(rank=0,world_size=4,batch_size=16,native_ddp_mode="standard",timing_mode="window")
    with pytest.raises(ValueError,match="non-window protocol version 3"): validate_task_result(legacy)
    result=_window_result()
    for mutation in ("empty","global","samples","epoch","warmup","bool"):
        damaged={**result,"training_loop_windows":[dict(item) for item in result["training_loop_windows"]],
                 "window_observability":dict(result["window_observability"])}
        if mutation=="empty": damaged["training_loop_windows"]=[]; damaged["steady_training_loop_window_s"]=None; damaged["steady_training_loop_samples_per_second"]=None
        elif mutation=="global": damaged["training_loop_windows"][0]["global_step_range"]=[400,9999]
        elif mutation=="samples": damaged["training_loop_windows"][1]["samples"]=999
        elif mutation=="epoch": damaged["training_loop_windows"][1]["epoch"]=999
        elif mutation=="warmup": damaged["training_loop_windows"][0]["local_record_range"]=[0,2]
        else: damaged["window_observability"]["losses_flushed"]=1
        with pytest.raises(ValueError): validate_task_result(damaged)


@pytest.mark.parametrize("field,bad",[
    ("steady_training_loop_window_s",True),("steady_training_loop_window_s",2),
    ("steady_training_loop_window_s",float("nan")),("steady_training_loop_window_s",float("inf")),
    ("steady_training_loop_samples_per_second",True),("steady_training_loop_samples_per_second",64),
    ("steady_training_loop_samples_per_second",float("nan")),("steady_training_loop_samples_per_second",float("inf")),
])
def test_steady_window_summaries_require_exact_finite_positive_floats(field,bad):
    result=_window_result(); result[field]=bad
    with pytest.raises(ValueError,match="exact finite positive float"): validate_task_result(result)


def test_all_warmup_window_requires_exact_none_steady_summaries():
    result=_window_result(); result["warmup_steps"]=3
    result["training_loop_windows"]=[{"epoch":0,"kind":"epoch","local_record_range":[0,3],"global_step_range":[0,3],"samples":48,"elapsed_wall_s":2.0}]
    result["steady_training_loop_window_s"]=None; result["steady_training_loop_samples_per_second"]=None
    assert validate_task_result(result) is result
    result["steady_training_loop_window_s"]=0.0
    with pytest.raises(ValueError,match="must be null"): validate_task_result(result)


def _window_result():
    result=_task_result(); result["schema_version"]=4
    result["execution_protocol"]=training_protocol(rank=0,world_size=4,batch_size=16,native_ddp_mode="standard",timing_mode="window")
    result.update(epochs=1,steady_samples_per_second=None,step_latency_p50_ms=None,step_latency_p95_ms=None,epoch_core_time_s=None,
                  communication_time_s=None,qwd_time_s=None,refresh_time_s=None,
                  training_loop_windows=[{"epoch":0,"kind":"warmup","local_record_range":[0,1],"global_step_range":[0,1],"samples":16,"elapsed_wall_s":1.0},
                    {"epoch":0,"kind":"epoch","local_record_range":[1,3],"global_step_range":[1,3],"samples":32,"elapsed_wall_s":2.0}],
                  steady_training_loop_window_s=2.0,steady_training_loop_samples_per_second=64.0,
                  window_observability={"measurement":"training_loop_window_wall","single_step_timing_available":False,"phase_breakdown_available":False,"losses_flushed":True,
                    "start_epoch":0,"start_global_step":0,"local_record_samples":[16,16,16]})
    result["timing_breakdown"]["data_s"]=None; result["timing_breakdown"]["core_train_s"]=None
    return result


def test_old_builder_cannot_use_deferred_loss_escape_hatch():
    with pytest.raises(ValueError,match="only for window"):
        build_step_record(task_id="",attempt_id="",route="bad",seed=0,epoch=-1,step=-1,batch_indices=(),
            timing=StepTiming(0.0,0.0,0.0,0.0,0.0,0.0),gradient_route="",parameter_route="",
            communication_bytes=-1,qwd_s=-1.0,refresh_s=-1.0,decision="",loss=None,amp_scale=-1.0,
            learning_rate=-1.0,model_sha256=None,rank_parameter_gap=None,optimizer_step=-1,finite=True,
            audit_performed=False,defer_loss_validation=True,window_timing=False)


@pytest.mark.parametrize("reason",["full","scheduled","final","midpoint","pending_resume","resume_oracle"])
def test_production_audit_seam_requires_immediate_loss_for_every_real_reason(reason):
    values={"quality_audit_mode":"production","global_step":7,"audit_steps":frozenset(),
            "final_batch":False,"midpoint":False,"pending_resume":False,"resume_oracle":False}
    if reason=="full": values["quality_audit_mode"]="full"
    elif reason=="scheduled": values["audit_steps"]=frozenset({7})
    elif reason=="final": values["final_batch"]=True
    else: values[reason]=True
    audit=window_audit_required(**values); assert audit is True
    reads=[]; order=[]; cuda=Cuda()
    observer=TrainingLoopWindowObserver(lambda:cuda,"cuda:0",0,clock=iter([1.0,2.0]).__next__)
    observer.begin(epoch=0,local_record_start=0,global_step_start=6)
    records=[]
    commit_observed_step(timing_mode="window",observer=observer,records=records,
        loss=Loss(0.5,reads),audit_performed=audit,
        quality_audit=lambda: None,
        bookkeeping=lambda loss, audit: order.append(("bookkeeping",loss,list(reads))),
        build_record=lambda loss, audit:{"quality":{"loss":loss,"finite":True,"optimizer_step":7},"batch_indices":[1]},
        post_append=lambda *args: None,
        epoch=0,global_step=7,samples=1,final_batch=True,training_stop=False)
    assert order==[("bookkeeping",0.5,[0.5])] and records[0]["quality"]["loss"]==0.5


def test_production_lifecycle_seams_reject_rank_mismatch_preserve_failure_and_old_sidecar():
    cuda=Cuda(); observer=TrainingLoopWindowObserver(lambda:cuda,"cuda:0",0,clock=iter([1.0,2.0]).__next__)
    observer.begin(epoch=0,local_record_start=0,global_step_start=10)
    row={"quality":{"loss":None}}
    error=RuntimeError("original loss transfer failure")
    class Broken(Loss):
        def __float__(self): raise error
    with pytest.raises(RuntimeError) as caught:
        commit_observed_step(timing_mode="window",observer=observer,records=[],loss=Broken(1.0,[]),
            audit_performed=True,quality_audit=lambda:None,
            bookkeeping=lambda loss, audit:None,build_record=lambda loss, audit:row,
            post_append=lambda *args: None,
            epoch=0,global_step=11,samples=2,final_batch=False,training_stop=True)
    assert caught.value is error and observer.windows==[]
    assert window_sidecar_fields(None)=={}
    assert observed_process_wall("window",5.0,9.0)==5.0
    good={"training_loop_windows":[{"epoch":0,"kind":"epoch","local_record_range":[0,1],"global_step_range":[10,11],"samples":2,"elapsed_wall_s":1.0}]}
    bad={"training_loop_windows":[{**good["training_loop_windows"][0],"samples":3}]}
    with pytest.raises(RuntimeError,match="coverage"): validate_window_rank_coverage([good,bad])
    skipped=execution_counters([{"quality":{"finite":False,"optimizer_step":10},"batch_indices":[1,2]}])
    assert skipped["skipped_updates"]==1 and skipped["successful_optimizer_updates"]==0


def test_integrated_commit_defers_ordinary_loss_and_coordinates_warmup_reopen_and_stop():
    reads=[]; order=[]; cuda=Cuda(); ticks=iter([1.0,2.0,4.0,6.0])
    observer=TrainingLoopWindowObserver(lambda:cuda,"cuda:0",1,clock=lambda:next(ticks))
    observer.begin(epoch=3,local_record_start=0,global_step_start=40); records=[]
    def build(loss, audit):
        order.append(("build",loss,list(reads)))
        skipped=bool(records)
        return {"quality":{"loss":loss,"finite":not skipped,"optimizer_step":41},"batch_indices":[1,2]}
    commit_observed_step(timing_mode="window",observer=observer,records=records,loss=Loss(0.75,reads),audit_performed=False,
        quality_audit=lambda:None,
        bookkeeping=lambda loss, audit:order.append(("oracle",loss,list(reads))),build_record=build,
        post_append=lambda *args: None,
        epoch=3,global_step=41,samples=2,final_batch=False,training_stop=False)
    assert order[:2]==[("oracle",None,[]),("build",None,[])] and reads==[0.75] and observer.is_open
    commit_observed_step(timing_mode="window",observer=observer,records=records,loss=Loss(0.5,reads),audit_performed=False,
        quality_audit=lambda:None,
        bookkeeping=lambda loss, audit:order.append(("stop",loss,list(reads))),build_record=build,
        post_append=lambda *args: None,
        epoch=3,global_step=42,samples=2,final_batch=False,training_stop=True)
    observer.close("training_stop")
    assert reads==[0.75,0.5] and [window["kind"] for window in observer.windows]==["warmup","training_stop"]
    counters=execution_counters(records)
    assert counters["successful_optimizer_updates"]==1 and counters["skipped_updates"]==1 and counters["optimizer_step_final"]==41


def test_warmup_spans_epochs_and_uses_local_records_not_resumed_global_step():
    cuda=Cuda(); ticks=iter([0.0,1.0,2.0,3.0]); observer=TrainingLoopWindowObserver(lambda:cuda,"cuda:0",3,clock=lambda:next(ticks))
    rows=[]
    observer.begin(epoch=4,local_record_start=0,global_step_start=100)
    for local in (1,2):
        row={"quality":{"loss":None}}; rows.append(row)
        observer.observe(loss=Loss(float(local),[]),record=row,audit=False,epoch=4,local_record_end=local,global_step_end=100+local,samples=2)
    observer.close("epoch")
    observer.begin(epoch=5,local_record_start=2,global_step_start=102)
    row={"quality":{"loss":None}}; observer.observe(loss=Loss(3.0,[]),record=row,audit=False,epoch=5,local_record_end=3,global_step_end=103,samples=2)
    assert [window["kind"] for window in observer.windows]==["epoch","warmup"]
    assert observer.windows[-1]["local_record_range"]==[2,3] and observer.windows[-1]["global_step_range"]==[102,103]


def test_partial_training_stop_window_and_zero_length_close_contract():
    cuda=Cuda(); observer=TrainingLoopWindowObserver(lambda:cuda,"cuda:0",0,clock=iter([2.0,5.0,7.0]).__next__)
    observer.begin(epoch=0,local_record_start=0,global_step_start=0)
    row={"quality":{"loss":None}}; observer.observe(loss=Loss(1.0,[]),record=row,audit=False,epoch=0,local_record_end=1,global_step_end=1,samples=3)
    observer.close("training_stop")
    observer.begin(epoch=1,local_record_start=1,global_step_start=1)
    assert observer.close("epoch") is None
    assert len(observer.windows)==1 and observer.windows[0]["kind"]=="training_stop"
