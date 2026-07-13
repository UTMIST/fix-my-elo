import unittest

from analysis.intuitive_vs_best import _compare_scores, build_intuition_report
from analysis.schema import AnalysisResult, EngineIdentity, PVLine, RootCandidate


FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _line(index, move, score_cp=None, mate=None, continuation=None):
    return PVLine(
        multipv_index=index,
        move_uci=move,
        pv_uci_list=[move] + list(continuation or []),
        score_cp=score_cp,
        mate=mate,
    )


class _ReportEngine:
    def __init__(self, root_result, child_results=None):
        self.root_result = root_result
        self.child_results = child_results or {}
        self.root_calls = []
        self.child_calls = []
        self.identity = EngineIdentity(uci_name="Fake Lc0", network_sha256="network-hash")

    def analyze_fen(self, fen, **kwargs):
        self.root_calls.append((fen, kwargs))
        return self.root_result

    def analyze_after_move(self, fen, move_uci, **kwargs):
        self.child_calls.append((fen, move_uci, kwargs))
        return self.child_results[move_uci]


class IntuitionReportTests(unittest.TestCase):
    def test_mate_scores_use_semantic_classification_without_centipawn_drop(self):
        cases = [
            (
                {"type": "mate", "value": 3},
                {"type": "cp", "value": 500},
                "lost_forced_mate",
            ),
            (
                {"type": "cp", "value": 100},
                {"type": "mate", "value": -4},
                "allows_forced_mate",
            ),
        ]

        for before, after, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                is_bad, drop_cp, reason = _compare_scores(before, after, 80)
                self.assertTrue(is_bad)
                self.assertIsNone(drop_cp)
                self.assertEqual(reason, expected_reason)

    def test_policy_order_drives_candidates_and_full_statistics_are_preserved(self):
        candidates = [
            RootCandidate(move_uci="d2d4", P=0.30, N=300, Q=0.12, V=0.08, U=0.04),
            RootCandidate(move_uci="e2e4", P=0.61, N=610, Q=0.20, V=0.16, U=0.04),
            RootCandidate(move_uci="g1f3", P=0.09, N=90, Q=0.02, V=0.01, U=0.01),
        ]
        root = AnalysisResult(
            fen=FEN,
            side_to_move="w",
            bestmove_uci="d2d4",
            pv_lines=[_line(1, "d2d4", score_cp=100, continuation=["d7d5"])],
            root_candidates=candidates,
        )
        after_e4 = AnalysisResult(
            fen=FEN,
            side_to_move="b",
            bestmove_uci="e7e5",
            pv_lines=[_line(1, "e7e5", score_cp=50, continuation=["g1f3"])],
        )
        engine = _ReportEngine(root, {"e2e4": after_e4})
        config = {
            "nodes": 500,
            "movetime_ms": None,
            "depth": None,
            "multipv": 3,
            "top_intuitive": 2,
            "bad_threshold_cp": 80,
            "verbose_move_stats": True,
            "include_raw_info": True,
        }

        report = build_intuition_report(engine, FEN, config)

        self.assertEqual(report.candidate_moves, ["e2e4", "d2d4"])
        self.assertEqual(report.intuitive_moves, ["e2e4", "d2d4"])
        self.assertEqual(report.metadata["selection_method"], "lc0_policy")
        self.assertTrue(report.metadata["policy_available"])
        self.assertEqual(report.metadata["root_searches"], 1)
        self.assertEqual(report.metadata["refutation_searches"], 1)
        self.assertEqual(len(engine.root_calls), 1)
        self.assertEqual([call[1] for call in engine.child_calls], ["e2e4"])

        evaluation = report.candidate_evaluations[0]
        self.assertEqual(evaluation["eval_before"], {"type": "cp", "value": 100})
        self.assertEqual(evaluation["eval_after"], {"type": "cp", "value": -50})
        self.assertEqual(evaluation["drop_cp"], 150)
        self.assertTrue(evaluation["is_bad"])
        self.assertEqual(evaluation["classification_reason"], "centipawn_drop")
        self.assertEqual(report.intuitive_but_bad[0]["move_uci"], "e2e4")

        serialized = report.to_dict()
        serialized_candidates = serialized["analysis"]["root_candidates"]
        by_move = {candidate["move_uci"]: candidate for candidate in serialized_candidates}
        self.assertEqual(by_move["e2e4"]["P"], 0.61)
        self.assertEqual(by_move["e2e4"]["N"], 610)
        self.assertEqual(by_move["e2e4"]["Q"], 0.20)
        self.assertEqual(by_move["e2e4"]["V"], 0.16)
        self.assertEqual(by_move["e2e4"]["U"], 0.04)

    def test_missing_policy_and_evaluation_scores_are_reported_without_bad_label(self):
        root = AnalysisResult(
            fen=FEN,
            side_to_move="w",
            bestmove_uci="e2e4",
            pv_lines=[
                _line(1, "e2e4", continuation=["e7e5"]),
                _line(2, "d2d4", continuation=["d7d5"]),
            ],
            root_candidates=[RootCandidate(move_uci="e2e4", P=None, N=10)],
        )
        after_d4 = AnalysisResult(
            fen=FEN,
            side_to_move="b",
            bestmove_uci="d7d5",
            pv_lines=[_line(1, "d7d5", continuation=["c2c4"])],
        )
        engine = _ReportEngine(root, {"d2d4": after_d4})

        report = build_intuition_report(
            engine,
            FEN,
            {
                "multipv": 2,
                "top_intuitive": 2,
                "bad_threshold_cp": 80,
                "verbose_move_stats": True,
            },
        )

        self.assertEqual(report.candidate_moves, ["e2e4", "d2d4"])
        self.assertEqual(report.intuitive_moves, [])
        self.assertEqual(report.metadata["selection_method"], "search_ranked_multipv")
        self.assertFalse(report.metadata["policy_available"])
        self.assertEqual(report.metadata["policy_candidate_count"], 0)
        self.assertEqual(report.intuitive_but_bad, [])
        self.assertEqual(len(report.candidate_evaluations), 1)
        evaluation = report.candidate_evaluations[0]
        self.assertIsNone(evaluation["eval_before"])
        self.assertIsNone(evaluation["eval_after"])
        self.assertIsNone(evaluation["drop_cp"])
        self.assertFalse(evaluation["is_bad"])
        self.assertEqual(evaluation["classification_reason"], "score_unavailable")
        self.assertIn("unknown", evaluation["evidence"])


if __name__ == "__main__":
    unittest.main()
