from __future__ import annotations

import io
import itertools
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class TestZipExtractionSafety(unittest.TestCase):
    """ZIP 解压必须拒绝绝对路径与 ../ 穿越成员（zip-slip）。"""

    @staticmethod
    def _make_zip(members: list[tuple[str, bytes]]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as handle:
            for name, content in members:
                handle.writestr(name, content)
        return buffer.getvalue()

    def test_normal_members_pass(self):
        from data.fetch_cmb import extract_safely

        payload = self._make_zip([("a.json", b"{}"), ("sub/b.json", b"{}")])
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                extract_safely(archive, Path(tmp))
            self.assertTrue((Path(tmp) / "a.json").exists())
            self.assertTrue((Path(tmp) / "sub" / "b.json").exists())

    def test_traversal_member_rejected(self):
        from data.fetch_cmb import extract_safely

        payload = self._make_zip([("../evil.txt", b"x")])
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                with self.assertRaises(ValueError):
                    extract_safely(archive, Path(tmp))
            self.assertFalse((Path(tmp).parent / "evil.txt").exists())

    def test_absolute_member_rejected(self):
        from data.fetch_cmb import extract_safely

        payload = self._make_zip([("/tmp/proj3_zip_slip_evil.txt", b"x")])
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                with self.assertRaises(ValueError):
                    extract_safely(archive, Path(tmp))


class TestDedupCrossSplit(unittest.TestCase):
    """split 去重：训练集与 val/test 撞题时删训练侧，评测集保持不动。"""

    @staticmethod
    def _record(split: str, index: int, question: str, answer: str) -> dict:
        return {
            "question": question,
            "option": {"A": "甲", "B": "乙", "C": "丙"},
            "answer": answer,
            "_split": split,
            "_source_index": index,
        }

    def test_train_copy_of_val_removed(self):
        from data.dedup_splits import filter_train_cross_split, normalize_record

        val = normalize_record(self._record("val", 0, "同一题干", "A"), "val", 0)
        train = normalize_record(self._record("train", 5, "同一题干", "A"), "train", 5)
        output, cross = filter_train_cross_split([train], {val["_canonical_key"]: ("val", 0)})
        self.assertEqual(output, [])
        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["key"], val["_canonical_key"])
        self.assertEqual(cross[0]["protected"], ("val", 0))
        self.assertTrue(train["_cross_split_duplicate"])

    def test_non_overlapping_kept(self):
        from data.dedup_splits import filter_train_cross_split, normalize_record

        train = normalize_record(self._record("train", 0, "不同题干", "A"), "train", 0)
        output, cross = filter_train_cross_split([train], {"other_key": ("val", 0)})
        self.assertEqual(len(output), 1)
        self.assertEqual(cross, [])

    def test_within_split_dedup(self):
        from data.dedup_splits import dedup, normalize_record

        first = normalize_record(self._record("train", 0, "题干", "A"), "train", 0)
        second = normalize_record(self._record("train", 1, "题干", "A"), "train", 1)
        kept, stats = dedup([first, second])
        self.assertEqual(stats["input"], 2)
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["dropped"], 1)
        self.assertEqual(stats["examples"][0]["duplicate_of"], ("train", 0))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestCollatorMasking(unittest.TestCase):
    """CausalCollator：prompt 与 padding 位置的 label 必须为 -100，其余与 input_ids 一致。"""

    def test_prompt_and_padding_masked(self):
        from train.sft_data import CausalCollator

        class StubTokenizer:
            pad_token_id = 0
            eos_token_id = 2

        collator = CausalCollator(StubTokenizer())
        features = [
            {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 7, 8]},
            {"input_ids": [9], "attention_mask": [1], "labels": [9]},
        ]
        batch = collator(features)
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 7, 8], [9, -100, -100, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 0, 0, 0]])
        for i in range(2):
            for j in range(4):
                if int(batch["labels"][i][j]) != -100:
                    self.assertEqual(int(batch["labels"][i][j]), int(batch["input_ids"][i][j]))


@unittest.skipUnless(HAS_TORCH, "eval imports model.hf_loader which requires torch")
class TestCandidateSpace(unittest.TestCase):
    """logprob 评测候选空间只由题型决定，绝不使用 gold 宽度（防答案信息泄漏）。"""

    def test_single_choice_uses_letters_only(self):
        from eval.eval_cmb_logprob import select_candidates

        self.assertEqual(select_candidates(["A", "B", "C", "D"], "单项选择题"), ["A", "B", "C", "D"])

    def test_multi_choice_covers_all_widths(self):
        from eval.eval_cmb_logprob import select_candidates

        letters = ["A", "B", "C", "D", "E"]
        expected = [",".join(group) for width in range(2, 6) for group in itertools.combinations(letters, width)]
        self.assertEqual(select_candidates(letters, "多项选择题"), expected)
        self.assertEqual(len(expected), 26)

    def test_unknown_type_falls_back_to_single(self):
        from eval.eval_cmb_logprob import select_candidates

        self.assertEqual(select_candidates(["A", "B"], None), ["A", "B"])


class TestWorkloadTokenizerRequirement(unittest.TestCase):
    """workload 桶大小按 token 定义：无 tokenizer 时必须显式 --char-fallback，不允许静默降级。"""

    def test_raises_without_tokenizer(self):
        from serving.workload import resolve_tokenizer

        with mock.patch("serving.workload.load_optional_tokenizer", return_value=None):
            with self.assertRaises(SystemExit):
                resolve_tokenizer(char_fallback=False)

    def test_fallback_allowed_when_explicit(self):
        from serving.workload import resolve_tokenizer

        with mock.patch("serving.workload.load_optional_tokenizer", return_value=None):
            self.assertIsNone(resolve_tokenizer(char_fallback=True))


class TestBenchStats(unittest.TestCase):
    """压测统计：分位数与 summarize 的空值处理。"""

    def test_percentile(self):
        from serving.bench import percentile

        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([10.0], 0.99), 10.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 3.0)

    def test_summarize_ignores_none(self):
        from serving.bench import summarize

        rows = [{"ttft_ms": 10.0}, {"ttft_ms": None}, {"ttft_ms": 20.0}]
        out = summarize("ttft_ms", rows, 5.0)
        self.assertEqual(out["mean"], 15.0)
        self.assertEqual(out["p50"], 10.0)
        self.assertEqual(out["p99"], 20.0)


if __name__ == "__main__":
    unittest.main()
