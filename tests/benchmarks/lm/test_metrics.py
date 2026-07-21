import math
import pytest

from benchmarks.lm.train import perplexity_from_loss, token_accuracy_counts


def test_perplexity_from_loss_handles_large_values():
    assert perplexity_from_loss(math.log(10)) == pytest.approx(10)
    assert math.isinf(perplexity_from_loss(1000))


def test_token_accuracy_ignores_masked_labels():
    logits = [[0, 3], [4, 0], [1, 5]]
    labels = [-100, 0, 1]
    correct, total = token_accuracy_counts(logits, labels)
    assert (correct, total) == (2, 2)
