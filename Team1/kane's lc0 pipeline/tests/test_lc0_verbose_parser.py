"""Tests for Lc0 verbose root-stat parsing."""

import unittest

from analysis.lc0_verbose_parser import (
    extract_last_root_table,
    parse_last_root_candidates,
    parse_verbose_move_stats,
)


def candidate_row(move, visits=12, policy="12.50%", delta="+ 3"):
    """Build one representative Lc0 verbose candidate row."""
    return (
        f"{move}  (123 ) N: {visits:7d} ({delta}) "
        f"(P: {policy}) (WL: -0.12500) (D: 0.321) (M: 42.5) "
        "(Q: -0.10000) (U: 0.25000) (S: 0.15000) (V: -0.2000)"
    )


class VerboseMoveStatsTests(unittest.TestCase):
    def test_parses_raw_payload(self):
        candidates = parse_verbose_move_stats(candidate_row("e2e4"))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].move_uci, "e2e4")
        self.assertAlmostEqual(candidates[0].P, 0.125)
        self.assertEqual(candidates[0].N, 12)

    def test_parses_uci_info_string(self):
        line = "info string " + candidate_row("g1f3", visits=81, policy="8.75%")

        candidate = parse_verbose_move_stats(line)[0]

        self.assertEqual(candidate.move_uci, "g1f3")
        self.assertEqual(candidate.N, 81)
        self.assertAlmostEqual(candidate.P, 0.0875)

    def test_parses_timestamped_glog_line(self):
        line = (
            "0204 00:32:39.088486 56564 "
            r"C:\projects\lc0\src\chess\uciloop.cc:333] << info string "
            + candidate_row("b8c6", visits=3253, policy="21.05%", delta="+162")
        )

        candidate = parse_verbose_move_stats(line)[0]

        self.assertEqual(candidate.move_uci, "b8c6")
        self.assertEqual(candidate.N, 3253)
        self.assertEqual(candidate.n_delta, 162)
        self.assertAlmostEqual(candidate.P, 0.2105)

    def test_parses_timestamped_opponent_style_payload(self):
        line = (
            "0204 00:32:39.088843 56564 "
            r"C:\projects\lc0\src\search\classic\search.cc:597] "
            + candidate_row("d1d4", visits=1, policy="0.55%", delta="+ 0")
        )

        candidate = parse_verbose_move_stats(line)[0]

        self.assertEqual(candidate.move_uci, "d1d4")
        self.assertAlmostEqual(candidate.P, 0.0055)

    def test_parses_all_extended_fields_and_negative_delta(self):
        row = (
            "a7a8q (999) N: 44 (- 7) (P: 0.375) (WL: -0.61) "
            "(D: 0.12) (M: 88.5) (Q: -0.50) (U: +0.20) "
            "(S: -0.30) (V: -0.45)"
        )

        candidate = parse_verbose_move_stats(row)[0]

        self.assertEqual(candidate.move_uci, "a7a8q")
        self.assertEqual(candidate.N, 44)
        self.assertEqual(candidate.n_delta, -7)
        self.assertAlmostEqual(candidate.P, 0.375)
        self.assertAlmostEqual(candidate.WL, -0.61)
        self.assertAlmostEqual(candidate.D, 0.12)
        self.assertAlmostEqual(candidate.M, 88.5)
        self.assertAlmostEqual(candidate.Q, -0.50)
        self.assertAlmostEqual(candidate.U, 0.20)
        self.assertAlmostEqual(candidate.S, -0.30)
        self.assertAlmostEqual(candidate.V, -0.45)

    def test_rejects_summary_and_unrelated_info_lines(self):
        text = "\n".join(
            [
                "info depth 10 nodes 500 score cp 12 pv e2e4 e7e5",
                "info string node (39) N: 5301 (P: 100.0%) (Q: 0.1)",
                "bestmove e2e4",
            ]
        )

        self.assertEqual(parse_verbose_move_stats(text), [])

    def test_excludes_later_opponent_table(self):
        root_rows = [
            "info string " + candidate_row("b8c6", visits=100),
            "info string " + candidate_row("c7c6", visits=30),
        ]
        opponent_rows = [
            "--- Opponent moves after: b8c6",
            "0204 00:32:39.088843 56564 search.cc:597] "
            + candidate_row("d1d4", visits=4),
            "0204 00:32:39.088857 56564 search.cc:597] "
            + candidate_row("f3e5", visits=2),
        ]

        candidates = parse_last_root_candidates("\n".join(root_rows + opponent_rows))

        self.assertEqual([item.move_uci for item in candidates], ["b8c6", "c7c6"])

    def test_legal_moves_select_and_filter_the_matching_root_block(self):
        old_root = [
            "info string " + candidate_row("e2e4", visits=50),
            "info string " + candidate_row("d2d4", visits=40),
        ]
        new_root = [
            "info string " + candidate_row("e7e5", visits=70),
            "info string " + candidate_row("c7c5", visits=60),
            "info string " + candidate_row("a2a4", visits=999),
        ]
        log_text = "\n".join(old_root + ["bestmove e2e4"] + new_root)

        candidates = parse_last_root_candidates(
            log_text, legal_moves=["e7e5", "c7c5"]
        )

        self.assertEqual([item.move_uci for item in candidates], ["e7e5", "c7c5"])

    def test_no_legal_match_returns_empty_table(self):
        log_text = "info string " + candidate_row("e2e4")

        self.assertEqual(
            extract_last_root_table(log_text, legal_moves=["e7e5", "c7c5"]), ""
        )
        self.assertEqual(
            parse_last_root_candidates(log_text, legal_moves=["e7e5", "c7c5"]), []
        )


if __name__ == "__main__":
    unittest.main()
