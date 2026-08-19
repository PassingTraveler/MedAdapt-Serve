from __future__ import annotations

from pathlib import Path
from typing import Any

from data.utils import iter_jsonl


class ChatJsonlDataset:
    def __init__(self, path: Path, tokenizer, max_length: int = 4096, max_records: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        records = list(iter_jsonl(path))
        if max_records is not None:
            records = records[:max_records]
        # 过滤 prompt 本身就超长的记录：否则 labels 全为 -100，产生 NaN 梯度。
        filtered = []
        for record in records:
            prompt_text = tokenizer.apply_chat_template(record["messages"][:-1], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
            if len(prompt_ids) < max_length:
                filtered.append(record)
        self.records = filtered

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        messages = record["messages"]
        prompt_messages = messages[:-1]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(full_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # pad_token_id 可能为 0（pad=unk），不能用 or 判断；与 train_lora.py 的 is None 判断保持一致。
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = max_length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

