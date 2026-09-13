"""scripts/search_corpus.py 的可复现测试（对应 Issue #5）。

全部语料在临时目录中即时生成，均为虚构内容：测试不读取本地资料库、
客户档案或网络数据，也不依赖 RAG 索引与向量模型，可直接离线运行。
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import search_corpus  # noqa: E402

SHARD_RELATIVE = Path("分片") / "全部分片.jsonl"


def make_row(chunk_id, text, source_path):
    return {"chunk_id": chunk_id, "text": text, "source_path": source_path}


def write_shard(corpus_dir, rows):
    shard = corpus_dir / SHARD_RELATIVE
    shard.parent.mkdir(parents=True)
    shard.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return shard


BASE_ROWS = [
    make_row(
        "chunk-1",
        "客户说预算不足时，先确认预算的真实边界，再谈分期方案。",
        "资料库/预算话题.md",
    ),
    make_row(
        "chunk-2",
        "Follow up within 48 hours and confirm the budget with the champion.",
        "资料库/followup-en.md",
    ),
    make_row("chunk-3", "完全无关的内容，讲的是物流排班表。", "资料库/logistics.md"),
    make_row(
        "chunk-4",
        "遇到预算异议先确认真实边界，预算异议背后往往是决策权重问题。",
        "资料库/深度/异议处理.md",
    ),
]


class SearchCorpusTestCase(unittest.TestCase):
    def run_cli(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["search_corpus.py", *argv]
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            code = search_corpus.main()
        return code, json.loads(stdout.getvalue())

    def make_corpus(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        corpus_dir = Path(tmp.name)
        write_shard(corpus_dir, rows)
        return corpus_dir


class KeywordMatchingTest(SearchCorpusTestCase):
    def test_chinese_keywords_match_and_irrelevant_row_excluded(self):
        corpus = self.make_corpus(BASE_ROWS)
        code, out = self.run_cli("--query", "预算 异议", "--corpus", str(corpus))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        by_id = {row["chunk_id"]: row for row in out["results"]}
        self.assertIn("chunk-1", by_id)
        self.assertIn("chunk-4", by_id)
        self.assertNotIn("chunk-3", by_id)
        self.assertGreater(by_id["chunk-4"]["score"], by_id["chunk-1"]["score"])

    def test_english_keywords_case_insensitive(self):
        corpus = self.make_corpus(BASE_ROWS)
        for query in ("follow up", "FOLLOW UP"):
            code, out = self.run_cli("--query", query, "--corpus", str(corpus))
            self.assertEqual(code, 0)
            self.assertEqual(out["results"][0]["chunk_id"], "chunk-2")
            self.assertGreater(out["results"][0]["score"], 0)

    def test_contiguous_phrase_outranks_split_tokens(self):
        rows = [
            make_row("with-phrase", "遇到预算异议先回应预算异议本身。", "资料库/a.md"),
            make_row("split", "预算、异议分别处理即可。", "资料库/b.md"),
        ]
        corpus = self.make_corpus(rows)
        code, out = self.run_cli("--query", "预算异议", "--corpus", str(corpus))
        self.assertEqual(code, 0)
        self.assertEqual(out["results"][0]["chunk_id"], "with-phrase")

    def test_query_shorter_than_two_chars_is_rejected(self):
        corpus = self.make_corpus(BASE_ROWS)
        code, out = self.run_cli("--query", "预", "--corpus", str(corpus))
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertIn("检索词", out["error"])


class IncludeFilterTest(SearchCorpusTestCase):
    def test_include_filters_by_source_path_substring(self):
        corpus = self.make_corpus(BASE_ROWS)
        code, out = self.run_cli(
            "--query", "预算", "--include", "深度", "--corpus", str(corpus)
        )
        self.assertEqual(code, 0)
        self.assertEqual([row["chunk_id"] for row in out["results"]], ["chunk-4"])

    def test_include_is_case_insensitive(self):
        corpus = self.make_corpus(BASE_ROWS)
        code, out = self.run_cli(
            "--query", "follow", "--include", "FOLLOWUP", "--corpus", str(corpus)
        )
        self.assertEqual(code, 0)
        self.assertEqual([row["chunk_id"] for row in out["results"]], ["chunk-2"])

    def test_include_can_exclude_everything(self):
        corpus = self.make_corpus(BASE_ROWS)
        code, out = self.run_cli(
            "--query", "预算", "--include", "不存在的前缀", "--corpus", str(corpus)
        )
        self.assertEqual(code, 0)
        self.assertEqual(out["count"], 0)
        self.assertEqual(out["results"], [])


class TopKTest(SearchCorpusTestCase):
    @staticmethod
    def ranked_rows():
        return [
            make_row("r1", "跟进跟进跟进" + "甲" * 40, "资料库/1.md"),
            make_row("r2", "跟进跟进" + "乙" * 40, "资料库/2.md"),
            make_row("r3", "跟进" + "丙" * 40, "资料库/3.md"),
            make_row("r4", "跟进" + "丁" * 40, "资料库/4.md"),
            make_row("r5", "跟进" + "戊" * 40, "资料库/5.md"),
        ]

    def test_top_k_limits_result_count(self):
        corpus = self.make_corpus(self.ranked_rows())
        code, out = self.run_cli(
            "--query", "跟进", "--top-k", "2", "--corpus", str(corpus)
        )
        self.assertEqual(code, 0)
        self.assertEqual(out["count"], 2)
        self.assertEqual(len(out["results"]), 2)

    def test_top_k_results_are_sorted_by_score_desc(self):
        corpus = self.make_corpus(self.ranked_rows())
        code, out = self.run_cli(
            "--query", "跟进", "--top-k", "3", "--corpus", str(corpus)
        )
        scores = [row["score"] for row in out["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(out["results"][0]["chunk_id"], "r1")

    def test_top_k_above_match_count_returns_all_matches(self):
        corpus = self.make_corpus(self.ranked_rows())
        code, out = self.run_cli(
            "--query", "跟进", "--top-k", "10", "--corpus", str(corpus)
        )
        self.assertEqual(code, 0)
        self.assertEqual(out["count"], 5)

    def test_top_k_zero_is_rejected_by_argparse(self):
        corpus = self.make_corpus(self.ranked_rows())
        with self.assertRaises(SystemExit):
            self.run_cli("--query", "跟进", "--top-k", "0", "--corpus", str(corpus))


class MaxCharsTest(SearchCorpusTestCase):
    def test_single_excerpt_truncated_to_max_chars(self):
        rows = [make_row("long", "要点" * 500, "资料库/长文.md")]
        corpus = self.make_corpus(rows)
        code, out = self.run_cli(
            "--query", "要点", "--max-chars", "1000", "--corpus", str(corpus)
        )
        self.assertEqual(code, 0)
        self.assertEqual(out["count"], 1)
        self.assertEqual(len(out["results"][0]["text"]), 1000)

    def test_char_budget_is_shared_across_results(self):
        rows = [
            make_row("big", "跟进跟进跟进" + "甲" * 794, "资料库/big.md"),
            make_row("small", "跟进" + "乙" * 398, "资料库/small.md"),
        ]
        corpus = self.make_corpus(rows)
        code, out = self.run_cli(
            "--query", "跟进", "--max-chars", "1000", "--corpus", str(corpus)
        )
        sizes = [len(row["text"]) for row in out["results"]]
        self.assertEqual(sizes, [800, 200])
        self.assertEqual(sum(sizes), 1000)


class CorpusResolutionTest(SearchCorpusTestCase):
    def test_corpus_dir_resolves_to_default_shard(self):
        corpus = self.make_corpus([make_row("c1", "跟进要点", "资料库/x.md")])
        code, out = self.run_cli("--query", "跟进", "--corpus", str(corpus))
        self.assertEqual(code, 0)
        self.assertEqual(Path(out["corpus"]), (corpus / SHARD_RELATIVE).resolve())
        self.assertEqual(out["results"][0]["chunk_id"], "c1")

    def test_corpus_jsonl_file_used_directly(self):
        corpus = self.make_corpus([make_row("c1", "跟进要点", "资料库/x.md")])
        shard = corpus / SHARD_RELATIVE
        code, out = self.run_cli("--query", "跟进", "--corpus", str(shard))
        self.assertEqual(code, 0)
        self.assertEqual(Path(out["corpus"]), shard.resolve())

    def test_env_var_corpus_dir_is_honored(self):
        corpus = self.make_corpus([make_row("c1", "跟进要点", "资料库/x.md")])
        with mock.patch.dict(os.environ, {"XIAOGUAN_CORPUS_DIR": str(corpus)}):
            code, out = self.run_cli("--query", "跟进")
        self.assertEqual(code, 0)
        self.assertEqual(out["results"][0]["chunk_id"], "c1")

    def test_missing_corpus_reports_actionable_error(self):
        with mock.patch.object(search_corpus, "DEFAULT_CANDIDATES", []), \
                mock.patch.dict(os.environ):
            os.environ.pop("XIAOGUAN_CORPUS_DIR", None)
            code, out = self.run_cli("--query", "跟进")
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertIn("销冠语料库", out["error"])


class OutputContractTest(SearchCorpusTestCase):
    def test_success_payload_shape(self):
        corpus = self.make_corpus(BASE_ROWS)
        code, out = self.run_cli("--query", "预算", "--corpus", str(corpus))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["query"], "预算")
        self.assertEqual(out["count"], len(out["results"]))
        for item in out["results"]:
            for key in ("score", "chunk_id", "source_path", "text"):
                self.assertIn(key, item)
            self.assertEqual(item["score"], round(item["score"], 3))

    def test_no_match_returns_empty_results(self):
        corpus = self.make_corpus(BASE_ROWS)
        code, out = self.run_cli("--query", "宇宙", "--corpus", str(corpus))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["count"], 0)
        self.assertEqual(out["results"], [])

    def test_malformed_jsonl_reports_error(self):
        corpus = self.make_corpus([])
        shard = corpus / SHARD_RELATIVE
        shard.write_text("{not json}\n", encoding="utf-8")
        code, out = self.run_cli("--query", "预算", "--corpus", str(corpus))
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])


if __name__ == "__main__":
    unittest.main()
