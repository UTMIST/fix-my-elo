"""Tests for versioned analysis records."""

import json
import unittest

from analysis.schema import (
    SCHEMA_VERSION,
    AnalysisResult,
    EngineIdentity,
    ErrorRecord,
    IntuitionReport,
    PVLine,
    PositionSource,
    RootCandidate,
)


FEN_WHITE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_BLACK = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


class SchemaTests(unittest.TestCase):
    def make_analysis(self):
        """Create a fully populated result for round-trip assertions."""
        return AnalysisResult(
            fen=FEN_WHITE,
            side_to_move="white",
            bestmove_uci="e2e4",
            pv_lines=[
                PVLine(
                    multipv_index=1,
                    move_uci="e2e4",
                    pv_uci_list=["e2e4", "e7e5", "g1f3"],
                    score_cp=31,
                    depth=14,
                    nodes=12000,
                    seldepth=22,
                    nps=8000,
                    time_ms=1500,
                    hashfull=45,
                    tbhits=0,
                    wdl=[430, 500, 70],
                ),
                PVLine(
                    multipv_index=2,
                    move_uci="d2d4",
                    pv_uci_list=["d2d4", "g8f6"],
                    mate=-3,
                    depth=13,
                ),
            ],
            root_candidates=[
                RootCandidate(
                    move_uci="e2e4",
                    P=0.42,
                    N=900,
                    Q=0.18,
                    V=0.12,
                    U=0.04,
                    WL=0.18,
                    D=0.31,
                    M=45.5,
                    S=0.22,
                    n_delta=16,
                ),
                RootCandidate(
                    move_uci="d2d4",
                    P=0.28,
                    N=600,
                    Q=0.11,
                    V=0.09,
                    U=0.05,
                ),
            ],
            raw_info=["info depth 14 multipv 1 score cp 31 pv e2e4 e7e5"],
            search_metadata={"limit": {"time_ms": 1500}, "multipv": 2},
        )

    def test_success_record_v2_json_round_trip(self):
        analysis = self.make_analysis()
        source = PositionSource(
            kind="pgn",
            path="games/example.pgn",
            position_index=8,
            game_index=2,
            headers={"White": "Alpha", "Black": "Beta", "Result": "1-0"},
            ply=17,
            last_move_uci="g8f6",
            last_move_san="Nf6",
            played_move_uci="f1b5",
            played_move_san="Bb5",
        )
        engine = EngineIdentity(
            uci_name="Lc0 v0.test",
            uci_author="The LCZero Authors",
            executable_path="engines/lc0.exe",
            executable_sha256="a" * 64,
            network_path="networks/test.pb.gz",
            network_sha256="b" * 64,
            options={"Threads": 2, "VerboseMoveStats": True},
        )
        report = IntuitionReport(
            fen=FEN_WHITE,
            best_move="e2e4",
            best_pv=["e2e4", "e7e5", "g1f3"],
            intuitive_moves=["e2e4", "d2d4"],
            intuitive_but_bad=[{"move": "d2d4", "loss_cp": 95}],
            candidate_moves=["e2e4", "d2d4"],
            candidate_evaluations=[{"move": "d2d4", "score_cp": -64}],
            analysis=analysis,
            source=source,
            engine=engine,
            metadata={"top_intuitive": 2},
            generated_at_utc="2026-07-13T15:00:00Z",
        )

        encoded = json.dumps(report.to_dict(), allow_nan=False)
        decoded = IntuitionReport.from_dict(json.loads(encoded))

        self.assertEqual(decoded.to_dict(), report.to_dict())
        self.assertEqual(decoded.to_dict()["schema_version"], SCHEMA_VERSION)
        self.assertEqual(decoded.to_dict()["record_type"], "intuition_analysis")
        self.assertEqual(decoded.to_dict()["status"], "ok")

    def test_error_record_v2_json_round_trip(self):
        source = PositionSource(
            kind="fen_list", path="positions.txt", position_index=3, line_number=9
        )
        engine = EngineIdentity(
            uci_name="Lc0", network_path="networks/test.pb.gz"
        )
        record = ErrorRecord(
            message="invalid board position",
            code="invalid_fen",
            stage="input",
            fen="not a fen",
            exception_type="ValueError",
            traceback_text="ValueError: invalid board position",
            source=source,
            engine=engine,
        )

        encoded = json.dumps(record.to_dict(), allow_nan=False)
        decoded = ErrorRecord.from_dict(json.loads(encoded))

        self.assertEqual(decoded.to_dict(), record.to_dict())
        self.assertEqual(decoded.to_dict()["schema_version"], SCHEMA_VERSION)
        self.assertEqual(decoded.to_dict()["record_type"], "analysis_error")
        self.assertEqual(decoded.to_dict()["status"], "error")

    def test_analysis_round_trip_preserves_root_candidates_and_pvs(self):
        original = self.make_analysis()

        restored = AnalysisResult.from_dict(
            json.loads(json.dumps(original.to_dict(), allow_nan=False))
        )

        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertEqual(
            [candidate.move_uci for candidate in restored.root_candidates],
            ["e2e4", "d2d4"],
        )
        self.assertEqual(restored.root_candidates[0].P, 0.42)
        self.assertEqual(restored.root_candidates[0].n_delta, 16)
        self.assertEqual(
            restored.pv_lines[0].pv_uci_list, ["e2e4", "e7e5", "g1f3"]
        )
        self.assertEqual(restored.pv_lines[1].mate, -3)
        self.assertIsNone(restored.pv_lines[1].score_cp)

    def test_empty_and_unavailable_root_candidates_remain_distinct(self):
        unavailable = AnalysisResult(
            FEN_WHITE, "w", "e2e4", root_candidates=None
        )
        available_but_empty = AnalysisResult(
            FEN_WHITE, "w", "e2e4", root_candidates=[]
        )

        self.assertIsNone(
            AnalysisResult.from_dict(unavailable.to_dict()).root_candidates
        )
        self.assertEqual(
            AnalysisResult.from_dict(available_but_empty.to_dict()).root_candidates,
            [],
        )

    def test_side_is_normalized_or_derived_strictly(self):
        self.assertEqual(
            AnalysisResult(FEN_WHITE, "WHITE", "e2e4").side_to_move, "w"
        )
        self.assertEqual(
            AnalysisResult(FEN_BLACK, "black", "e7e5").side_to_move, "b"
        )
        self.assertEqual(
            AnalysisResult(FEN_BLACK, None, "e7e5").side_to_move, "b"
        )

    def test_invalid_or_underivable_side_is_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisResult(FEN_WHITE, "north", "e2e4")
        with self.assertRaises(ValueError):
            AnalysisResult("not a fen", None, None)
        with self.assertRaises(ValueError):
            AnalysisResult.from_dict(
                {"fen": FEN_WHITE, "side_to_move": "", "bestmove_uci": "e2e4"}
            )

    def test_pv_may_not_have_centipawn_and_mate_scores(self):
        with self.assertRaises(ValueError):
            PVLine(
                multipv_index=1,
                move_uci="e2e4",
                pv_uci_list=["e2e4"],
                score_cp=20,
                mate=3,
            )

    def test_success_and_error_record_types_are_not_interchangeable(self):
        with self.assertRaises(ValueError):
            IntuitionReport.from_dict(ErrorRecord("boom", "internal", "run").to_dict())
        with self.assertRaises(ValueError):
            ErrorRecord.from_dict(
                IntuitionReport(FEN_WHITE, "e2e4").to_dict()
            )


if __name__ == "__main__":
    unittest.main()
