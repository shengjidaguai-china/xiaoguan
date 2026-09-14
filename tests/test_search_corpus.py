#!/usr/bin/env python3
"""scripts/search_corpus.py 的行为契约测试。

全部语料由临时目录生成虚构 JSONL 分片，不读取本地资料库、客户档案或网络，
不依赖第三方包；进程内调用 main() 并断言 stdout 的 JSON 契约与返回码。
覆盖：中英文关键词命中与无命中过滤、短语加分带来的排序差异、--include
大小写不敏感过滤、--top-k 截断与降序、--max-chars 单条截断与多条累计预算、
--corpus 目录/文件两种解析与 XIAOGUAN_CORPUS_DIR 环境变量、错误契约
（超短检索词、坏 JSONL、找不到语料库时 ok:false 且返回码 1）。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_corpus  # noqa: E402


class SearchCorpusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def write_corpus(self, rows: list[dict], as_file: bool = False) -> Path:
        """按仓库约定把虚构语料写成 分片/全部分片.jsonl，或直接写成单个 JSONL 文件。"""
        if as_file:
            target = self.tmp / "corpus.jsonl"
        else:
            shards = self.tmp / "分片"
            shards.mkdir(parents=True, exist_ok=True)
            target = shards / "全部分片.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target

    def run_search(self, *argv: str) -> tuple[int, dict]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with mock.patch.object(sys, "argv", ["search_corpus.py", *argv]):
                code = search_corpus.main()
        return code, json.loads(stdout.getvalue())

    def corpus_dir(self, rows: list[dict]) -> Path:
        self.write_corpus(rows)
        return self.tmp


class CorpusResolutionTest(SearchCorpusTestCase):
    def test_corpus_directory_resolves_to_shards_file(self) -> None:
        shards = self.write_corpus(
            [{"chunk_id": "c1", "source_path": "资料库/a.md", "text": "开场白要点到为止"}]
        )
        code, out = self.run_search("--query", "开场白", "--corpus", str(self.tmp))
        self.assertEqual(0, code)
        self.assertTrue(out["ok"])
        self.assertEqual(Path(out["corpus"]), shards)

    def test_corpus_jsonl_file_used_directly(self) -> None:
        target = self.write_corpus(
            [{"chunk_id": "c1", "source_path": "资料库/a.md", "text": "开场白要点到为止"}],
            as_file=True,
        )
        code, out = self.run_search("--query", "开场白", "--corpus", str(target))
        self.assertEqual(0, code)
        self.assertTrue(out["ok"])
        self.assertEqual(Path(out["corpus"]), target)

    def test_environment_corpus_dir_is_picked_up(self) -> None:
        self.write_corpus(
            [{"chunk_id": "c1", "source_path": "资料库/a.md", "text": "开场白要点到为止"}]
        )
        with mock.patch.dict(os.environ, {"XIAOGUAN_CORPUS_DIR": str(self.tmp)}):
            code, out = self.run_search("--query", "开场白")
        self.assertEqual(0, code)
        self.assertTrue(out["ok"])
        self.assertEqual(Path(out["corpus"]), self.tmp / "分片" / "全部分片.jsonl")

    def test_missing_corpus_reports_ok_false(self) -> None:
        missing = self.tmp / "不存在" / "资料库"
        clean_env = {k: v for k, v in os.environ.items() if k != "XIAOGUAN_CORPUS_DIR"}
        with mock.patch.object(search_corpus, "DEFAULT_CANDIDATES", []):
            with mock.patch.dict(os.environ, clean_env, clear=True):
                code, out = self.run_search("--query", "开场白", "--corpus", str(missing))
        self.assertEqual(1, code)
        self.assertFalse(out["ok"])
        self.assertIn("语料库", out["error"])


class QueryMatchTest(SearchCorpusTestCase):
    def test_matches_chinese_and_english_tokens(self) -> None:
        self.corpus_dir(
            [
                {
                    "chunk_id": "zh",
                    "source_path": "资料库/话术库/开场白.md",
                    "text": "陌生拜访的开场白要在三句话内建立信任。",
                },
                {
                    "chunk_id": "en",
                    "source_path": "资料库/英语话术/ColdCall.md",
                    "text": "An effective cold call opening states your name and reason quickly.",
                },
            ]
        )
        code, out = self.run_search("--query", "开场白", "--corpus", str(self.tmp))
        self.assertEqual(0, code)
        self.assertEqual(1, out["count"])
        self.assertEqual("zh", out["results"][0]["chunk_id"])

        code, out = self.run_search("--query", "cold call", "--corpus", str(self.tmp))
        self.assertEqual(0, code)
        self.assertEqual(1, out["count"])
        self.assertEqual("en", out["results"][0]["chunk_id"])

    def test_no_match_yields_empty_results(self) -> None:
        self.corpus_dir(
            [{"chunk_id": "c1", "source_path": "资料库/a.md", "text": "开场白要点到为止"}]
        )
        code, out = self.run_search("--query", "量子纠缠态", "--corpus", str(self.tmp))
        self.assertEqual(0, code)
        self.assertTrue(out["ok"])
        self.assertEqual(0, out["count"])
        self.assertEqual([], out["results"])

    def test_phrase_bonus_ranks_contiguous_match_first(self) -> None:
        self.corpus_dir(
            [
                {
                    "chunk_id": "split",
                    "source_path": "资料库/s.md",
                    "text": "信任 之前先打磨 开场白",
                },
                {
                    "chunk_id": "phrase",
                    "source_path": "资料库/p.md",
                    "text": "开场白 信任 是两件事",
                },
            ]
        )
        code, out = self.run_search(
            "--query", "开场白 信任", "--top-k", "2", "--corpus", str(self.tmp)
        )
        self.assertEqual(0, code)
        self.assertEqual(2, out["count"])
        self.assertEqual("phrase", out["results"][0]["chunk_id"])
        self.assertEqual("split", out["results"][1]["chunk_id"])
        self.assertAlmostEqual(12.0, out["results"][0]["score"] - out["results"][1]["score"])

    def test_include_filters_paths_case_insensitively(self) -> None:
        self.corpus_dir(
            [
                {
                    "chunk_id": "cold",
                    "source_path": "资料库/英语话术/ColdCall.md",
                    "text": "cold call 的开场白同样要短",
                },
                {
                    "chunk_id": "home",
                    "source_path": "资料库/话术库/开场白.md",
                    "text": "开场白要短",
                },
            ]
        )
        code, out = self.run_search(
            "--query", "开场白", "--include", "COLDCALL", "--corpus", str(self.tmp)
        )
        self.assertEqual(0, code)
        self.assertEqual(1, out["count"])
        self.assertEqual("cold", out["results"][0]["chunk_id"])


class RankingLimitsTest(SearchCorpusTestCase):
    def test_top_k_truncates_to_highest_scores_in_descending_order(self) -> None:
        rows = []
        for index, repeats in enumerate([1, 2, 3, 4, 5], start=1):
            rows.append(
                {
                    "chunk_id": f"c{repeats}",
                    "source_path": f"资料库/rank{index}.md",
                    "text": " ".join(["开场白"] * repeats),
                }
            )
        self.corpus_dir(rows)
        code, out = self.run_search(
            "--query", "开场白", "--top-k", "2", "--corpus", str(self.tmp)
        )
        self.assertEqual(0, code)
        self.assertEqual(2, out["count"])
        self.assertEqual(["c5", "c4"], [r["chunk_id"] for r in out["results"]])
        scores = [r["score"] for r in out["results"]]
        self.assertGreater(scores[0], scores[1])

    def test_max_chars_truncates_excerpt_and_stops_later_rows(self) -> None:
        self.corpus_dir(
            [
                {
                    "chunk_id": "long",
                    "source_path": "资料库/long.md",
                    "text": "开场白开场白" + "长" * 2000,
                },
                {
                    "chunk_id": "short",
                    "source_path": "资料库/short.md",
                    "text": "开场白 收尾",
                },
            ]
        )
        code, out = self.run_search(
            "--query", "开场白", "--top-k", "5", "--max-chars", "1000",
            "--corpus", str(self.tmp),
        )
        self.assertEqual(0, code)
        self.assertEqual(1, out["count"])
        excerpt = out["results"][0]["text"]
        self.assertEqual(1000, len(excerpt))
        self.assertTrue(excerpt.endswith("长"))
        self.assertNotIn("short", [r["chunk_id"] for r in out["results"]])


class ErrorContractTest(SearchCorpusTestCase):
    def test_too_short_query_reports_ok_false(self) -> None:
        self.corpus_dir(
            [{"chunk_id": "c1", "source_path": "资料库/a.md", "text": "开场白要点到为止"}]
        )
        code, out = self.run_search("--query", "a", "--corpus", str(self.tmp))
        self.assertEqual(1, code)
        self.assertFalse(out["ok"])
        self.assertIn("检索词", out["error"])

    def test_malformed_jsonl_reports_ok_false(self) -> None:
        target = self.write_corpus(
            [{"chunk_id": "c1", "source_path": "资料库/a.md", "text": "开场白要点到为止"}],
            as_file=True,
        )
        with target.open("a", encoding="utf-8") as handle:
            handle.write("{不是合法的 json 行\n")
        code, out = self.run_search("--query", "开场白", "--corpus", str(target))
        self.assertEqual(1, code)
        self.assertFalse(out["ok"])


if __name__ == "__main__":
    unittest.main()
