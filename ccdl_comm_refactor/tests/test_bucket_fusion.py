from ccdl_comm.communication.bucket_fusion import BucketDescriptor, plan_bucket_fusion


def test_bucket_fusion_contract_is_exported_from_communication_package() -> None:
    from ccdl_comm.communication import BucketFusionPlan

    assert BucketFusionPlan.__name__ == "BucketFusionPlan"


def test_bucket_fusion_groups_adjacent_compatible_buckets() -> None:
    plan = plan_bucket_fusion(
        [
            BucketDescriptor(index=0, numel=128, dtype="float16"),
            BucketDescriptor(index=1, numel=256, dtype="float16"),
            BucketDescriptor(index=2, numel=512, dtype="float16"),
        ],
        max_fused_numel=1024,
        max_bucket_count=4,
    )

    assert [group.bucket_indices for group in plan.groups] == [(0, 1, 2)]
    assert plan.fused_bucket_count == 3
    assert plan.skipped_bucket_count == 0
    assert plan.total_fused_numel == 896


def test_bucket_fusion_splits_on_dtype_or_size_limit() -> None:
    plan = plan_bucket_fusion(
        [
            BucketDescriptor(index=0, numel=600, dtype="float16"),
            BucketDescriptor(index=1, numel=600, dtype="float16"),
            BucketDescriptor(index=2, numel=64, dtype="float32"),
            BucketDescriptor(index=3, numel=64, dtype="float32"),
        ],
        max_fused_numel=1024,
        max_bucket_count=4,
    )

    assert [group.bucket_indices for group in plan.groups] == [(0,), (1,), (2, 3)]
    assert [group.fused for group in plan.groups] == [False, False, True]
    assert plan.fused_bucket_count == 2
    assert plan.skipped_bucket_count == 2


def test_bucket_fusion_skips_full_bucket_consumers() -> None:
    plan = plan_bucket_fusion(
        [
            BucketDescriptor(index=0, numel=128, dtype="float16"),
            BucketDescriptor(index=1, numel=128, dtype="float16", requires_full_bucket=True),
            BucketDescriptor(index=2, numel=128, dtype="float16"),
        ],
        max_fused_numel=1024,
        max_bucket_count=4,
    )

    assert [group.bucket_indices for group in plan.groups] == [(0,), (1,), (2,)]
    assert plan.groups[1].reason == "requires full-bucket consumer"
    assert plan.fused_bucket_count == 0
    assert plan.skipped_bucket_count == 3


def test_bucket_fusion_metadata_reports_counts_and_reasons() -> None:
    plan = plan_bucket_fusion(
        [
            BucketDescriptor(index=7, numel=16, dtype="float16"),
            BucketDescriptor(index=8, numel=16, dtype="float16"),
        ],
        max_fused_numel=64,
        max_bucket_count=2,
    )

    assert plan.to_metadata() == {
        "enabled": True,
        "group_count": 1,
        "fused_group_count": 1,
        "fused_bucket_count": 2,
        "skipped_bucket_count": 0,
        "total_fused_numel": 32,
        "groups": [
            {
                "bucket_indices": (7, 8),
                "total_numel": 32,
                "dtype": "float16",
                "fused": True,
                "reason": "compatible adjacent buckets",
            }
        ],
    }
