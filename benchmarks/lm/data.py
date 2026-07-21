import random
import json


def split_indices(size: int, seed: int, validation_fraction: float = 0.05):
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, round(size * validation_fraction))
    validation = indices[:validation_size]
    train = indices[validation_size:]
    return train, validation


def format_example(example: dict[str, str]) -> tuple[str, str]:
    instruction = example["instruction"].strip()
    context = example.get("input", "").strip()
    response = example["output"].strip()
    prompt = f"### Instruction:\n{instruction}\n"
    if context:
        prompt += f"\n### Input:\n{context}\n"
    prompt += "\n### Response:\n"
    return prompt, response


def build_response_only_features(example, tokenizer, max_length: int = 256):
    prompt, response = format_example(example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    supervised_ids = (response_ids + [eos])[:max_length]
    prompt_budget = max_length - len(supervised_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    input_ids = prompt_ids + supervised_ids
    labels = [-100] * len(prompt_ids) + supervised_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_alpaca(path):
    with open(path, encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or not records:
        raise ValueError("Alpaca data must be a non-empty JSON list")
    return records


class AlpacaDataset:
    def __init__(self, records, indices, tokenizer, max_length):
        self.records = records
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return build_response_only_features(
            self.records[self.indices[index]], self.tokenizer, self.max_length
        )


class ResponseOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(row["input_ids"]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
