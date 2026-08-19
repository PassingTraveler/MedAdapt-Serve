from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.build_answer_sft import make_example
from data.utils import answer_letters, canonical_question_key, option_map, write_jsonl
from eval.cmb_common import exact_match, parse_prediction
from eval.eval_cmb_logprob import select_candidates


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "question": "哪一个是正确答案？",
            "option": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
            "answer": "B",
            "question_type": "单项选择题",
            "_split": "train",
            "_source_index": 0,
        }

    def test_answer_normalization(self):
        self.assertEqual(answer_letters("答案：b"), ["B"])
        self.assertEqual(answer_letters(["D", "B", "B"]), ["B", "D"])
        self.assertEqual(answer_letters("F"), ["F"])

    def test_option_map_and_key(self):
        self.assertEqual(list(option_map(self.record)), ["A", "B", "C", "D"])
        # 选项顺序与空白变化不影响 canonical key（这才是去重的核心语义）。
        variant = {
            "question": "  哪一个是正确答案？\n",
            "option": {"D": "丁", "A": "甲 ", "C": "丙", "B": " 乙"},
            "answer": "B",
        }
        self.assertEqual(canonical_question_key(self.record), canonical_question_key(variant))
        # key 不含答案：同题不同答案仍视为重复。
        other_answer = dict(self.record, answer="A")
        self.assertEqual(canonical_question_key(self.record), canonical_question_key(other_answer))
        # 无关题目 key 不同。
        unrelated = dict(self.record, question="完全不同的题目？")
        self.assertNotEqual(canonical_question_key(self.record), canonical_question_key(unrelated))
        # 空选项被过滤，且支持 F。
        self.assertEqual(option_map({"option": {"A": "甲", "E": "", "F": "己"}}), {"A": "甲", "F": "己"})

    def test_shuffle_remaps_answer(self):
        example = make_example(self.record, seed=3, shuffle=True)
        self.assertEqual(example["messages"][-1]["role"], "assistant")
        self.assertEqual(len(example["answer"]), 1)
        self.assertIn(example["answer"][0], "ABCD")
        # 映射一致性：重排后答案字母对应原题正确选项的文本。
        mapping = example["option_mapping"]  # old -> new
        inverse = {new: old for old, new in mapping.items()}
        original_options = option_map(self.record)
        for new_letter in example["answer"]:
            old_letter = inverse[new_letter]
            self.assertIn(old_letter, original_options)
            option_line = f"{new_letter}. {original_options[old_letter]}"
            self.assertIn(option_line, example["messages"][0]["content"])

    def test_make_example_drops_answer_without_option(self):
        broken = dict(self.record, answer="E")  # E 不在选项里
        self.assertIsNone(make_example(broken, seed=1, shuffle=True))

    def test_prediction_parser(self):
        self.assertEqual(parse_prediction("答案：B。"), ["B"])
        self.assertEqual(parse_prediction("答案：BCDE"), ["B", "C", "D", "E"])
        self.assertEqual(parse_prediction("答案：B、C、D"), ["B", "C", "D"])
        self.assertEqual(parse_prediction("选C"), ["C"])
        self.assertEqual(parse_prediction("正确答案：A"), ["A"])
        self.assertEqual(parse_prediction("B"), ["B"])
        self.assertEqual(parse_prediction("答案：F"), ["F"])
        # 长文本回显选项字母时取最后一段字母串，不吞题干。
        echoed = "选项：A. 通调水道 B. 调畅气机 C. 助脾运化，所以答案选 C"
        self.assertEqual(parse_prediction(echoed), ["C"])
        self.assertTrue(exact_match("答案：B", self.record))

    def test_logprob_candidate_pools(self):
        letters = ["A", "B", "C", "D"]
        self.assertEqual(select_candidates(letters, "单项选择题"), letters)
        multi = select_candidates(letters, "多项选择题")
        # 2..4 元组合：C(4,2)+C(4,3)+C(4,4)=6+4+1=11。
        # 候选空间只由题型决定（不使用 gold 宽度，避免"选错个数永不判错"的答案泄漏）。
        self.assertEqual(len(multi), 11)
        self.assertIn("B,C", multi)
        self.assertNotIn("A", multi)  # 多选池不含单字母

    def test_jsonl_round_trip(self):
        from data.utils import iter_jsonl

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(path, [self.record])
            self.assertEqual(list(iter_jsonl(path))[0]["answer"], "B")


if __name__ == "__main__":
    unittest.main()
