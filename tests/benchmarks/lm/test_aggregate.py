from benchmarks.lm.aggregate import convergence_step, fp32_perplexity_target, summarize_values


def test_fp32_target_uses_mean_final_perplexity():
    assert fp32_perplexity_target([10.0, 12.0, 11.0]) == 11.11


def test_convergence_requires_three_consecutive_evaluations():
    evaluations = [
        {"step": 10, "perplexity": 11.0, "wall_time_sec": 5.0},
        {"step": 20, "perplexity": 10.0, "wall_time_sec": 10.0},
        {"step": 30, "perplexity": 9.0, "wall_time_sec": 15.0},
        {"step": 40, "perplexity": 8.0, "wall_time_sec": 20.0},
    ]
    assert convergence_step(evaluations, target=10.0) == (20, 10.0)
    assert convergence_step(evaluations[:3], target=9.5) is None


def test_summary_reports_mean_and_sample_std():
    summary = summarize_values([1.0, 2.0, 3.0])
    assert summary == {"mean": 2.0, "std": 1.0, "count": 3}
