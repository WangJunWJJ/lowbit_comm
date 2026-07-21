from benchmarks.lm.data import build_response_only_features, format_example, split_indices


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + index for index, _ in enumerate(text.split())]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_split_is_deterministic_disjoint_and_complete():
    first = split_indices(101, seed=123, validation_fraction=0.05)
    second = split_indices(101, seed=123, validation_fraction=0.05)
    assert first == second
    train, validation = first
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(101))
    assert len(validation) == 5


def test_format_example_includes_optional_input():
    prompt, response = format_example({"instruction": "Explain", "input": "context", "output": "answer"})
    assert "Explain" in prompt and "context" in prompt
    assert response == "answer"


def test_labels_ignore_prompt_and_train_on_response():
    features = build_response_only_features(
        {"instruction": "one two", "input": "", "output": "three four"},
        ToyTokenizer(),
        max_length=32,
    )
    trained = [index for index, label in enumerate(features["labels"]) if label != -100]
    assert trained
    assert trained[0] > 0
    assert features["labels"][-1] == 2
    assert features["input_ids"][-1] == 2


def test_long_prompt_does_not_remove_all_response_labels():
    features = build_response_only_features(
        {"instruction": " ".join(["prompt"] * 100), "input": "", "output": "answer"},
        ToyTokenizer(),
        max_length=8,
    )
    assert len(features["input_ids"]) == 8
    assert sum(label != -100 for label in features["labels"]) >= 2
