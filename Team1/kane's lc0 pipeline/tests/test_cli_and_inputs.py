import contextlib
import io
import json
import os
import tempfile
import unittest

import chess

from analysis.run_analysis import (
    _yield_positions_from_list,
    _yield_positions_from_pgn,
    run_cli,
)
from analysis.schema import AnalysisResult, EngineIdentity, PVLine, RootCandidate


START_FEN = chess.STARTING_FEN


class _FakeCliEngine:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.close_calls = 0
        self.identity = EngineIdentity(
            uci_name="Fake Lc0",
            executable_path=os.path.abspath(kwargs["engine_path"]),
            network_path=os.path.abspath(kwargs["weights_path"]),
        )
        self.__class__.instances.append(self)

    def analyze_fen(self, fen, **kwargs):
        board = chess.Board(fen)
        move = next(iter(board.legal_moves)).uci()
        return AnalysisResult(
            fen=fen,
            side_to_move="w" if board.turn == chess.WHITE else "b",
            bestmove_uci=move,
            pv_lines=[PVLine(1, move, [move], score_cp=12)],
            root_candidates=[RootCandidate(move_uci=move, P=1.0, N=100, Q=0.05)],
        )

    def analyze_after_move(self, fen, move_uci, **kwargs):
        raise AssertionError("The sole policy candidate is the best move")

    def close(self):
        self.close_calls += 1


def _base_args(fen):
    return [
        "--fen",
        fen,
        "--lc0_path",
        "fake-lc0.exe",
        "--weights_path",
        "fake-network.pb.gz",
        "--top_intuitive",
        "1",
    ]


class CliBehaviorTests(unittest.TestCase):
    def setUp(self):
        _FakeCliEngine.instances.clear()

    def _run(self, args, factory=_FakeCliEngine):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = run_cli(args, engine_factory=factory)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_success_returns_zero_and_writes_versioned_record(self):
        exit_code, stdout, stderr = self._run(_base_args(START_FEN))

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["record_type"], "intuition_analysis")
        self.assertGreaterEqual(record["schema_version"], 1)
        self.assertEqual(record["source"]["kind"], "fen")
        self.assertEqual(record["source"]["position_index"], 1)
        self.assertEqual(record["engine"]["uci_name"], "Fake Lc0")
        self.assertEqual(_FakeCliEngine.instances[0].close_calls, 1)

    def test_invalid_fen_returns_one_and_writes_error_record(self):
        exit_code, stdout, stderr = self._run(_base_args("not a fen"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["record_type"], "analysis_error")
        self.assertEqual(record["error_details"]["code"], "invalid_fen")
        self.assertEqual(record["error_details"]["stage"], "input")
        self.assertEqual(record["fen"], "not a fen")
        self.assertEqual(record["source"]["kind"], "fen")
        self.assertEqual(_FakeCliEngine.instances[0].close_calls, 1)

    def test_engine_start_failure_returns_two_and_writes_error_record(self):
        def failing_factory(**kwargs):
            raise OSError("engine could not start")

        exit_code, stdout, stderr = self._run(
            _base_args(START_FEN), factory=failing_factory
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error_details"]["code"], "engine_start_failed")
        self.assertEqual(record["error_details"]["stage"], "engine_start")
        self.assertEqual(record["source"]["kind"], "fen")

    def test_mixed_fen_list_returns_one_and_continues_after_invalid_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mixed.fen")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(START_FEN + "\n")
                handle.write("not a fen\n")
                handle.write(START_FEN + "\n")

            args = [
                "--fen_list",
                path,
                "--lc0_path",
                "fake-lc0.exe",
                "--weights_path",
                "fake-network.pb.gz",
                "--top_intuitive",
                "1",
            ]
            exit_code, stdout, stderr = self._run(args)

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr, "")
        records = [json.loads(line) for line in stdout.splitlines()]
        self.assertEqual([record["status"] for record in records], ["ok", "error", "ok"])
        self.assertEqual(records[1]["error_details"]["code"], "invalid_fen")
        self.assertEqual(
            [record["source"]["position_index"] for record in records],
            [1, 2, 3],
        )
        self.assertEqual(
            [record["source"]["line_number"] for record in records],
            [1, 2, 3],
        )
        self.assertEqual(_FakeCliEngine.instances[0].close_calls, 1)


class InputProvenanceTests(unittest.TestCase):
    def test_fen_list_keeps_absolute_path_line_number_and_position_index(self):
        contents = "# opening positions\n" + START_FEN + "\n\n# another\n" + START_FEN + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "positions.fen")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(contents)

            positions = list(_yield_positions_from_list(path))

        self.assertEqual([fen for fen, _ in positions], [START_FEN, START_FEN])
        first_source = positions[0][1]
        second_source = positions[1][1]
        self.assertTrue(os.path.isabs(first_source.path))
        self.assertEqual(first_source.kind, "fen_list")
        self.assertEqual(first_source.position_index, 1)
        self.assertEqual(first_source.line_number, 2)
        self.assertEqual(second_source.position_index, 2)
        self.assertEqual(second_source.line_number, 5)

    def test_pgn_samples_include_headers_ply_and_surrounding_moves(self):
        pgn = """[Event \"Pipeline test\"]
[Site \"Local\"]
[Date \"2026.07.13\"]
[Round \"1\"]
[White \"Ada\"]
[Black \"Turing\"]
[Result \"*\"]

1. e4 e5 2. Nf3 Nc6 *
"""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "game.pgn")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(pgn)

            positions = list(
                _yield_positions_from_pgn(path, every_n_plies=2, include_start=True)
            )

        self.assertEqual(len(positions), 3)
        start, after_two, after_four = [source for _, source in positions]
        self.assertTrue(os.path.isabs(start.path))
        self.assertEqual(start.headers["White"], "Ada")
        self.assertEqual(start.headers["Black"], "Turing")
        self.assertEqual(start.headers["Result"], "*")
        self.assertEqual(start.game_index, 1)
        self.assertEqual(start.position_index, 1)
        self.assertEqual(start.ply, 0)
        self.assertIsNone(start.last_move_uci)
        self.assertEqual(start.played_move_uci, "e2e4")
        self.assertEqual(start.played_move_san, "e4")

        self.assertEqual(after_two.position_index, 2)
        self.assertEqual(after_two.ply, 2)
        self.assertEqual(after_two.last_move_uci, "e7e5")
        self.assertEqual(after_two.last_move_san, "e5")
        self.assertEqual(after_two.played_move_uci, "g1f3")
        self.assertEqual(after_two.played_move_san, "Nf3")

        self.assertEqual(after_four.position_index, 3)
        self.assertEqual(after_four.ply, 4)
        self.assertEqual(after_four.last_move_uci, "b8c6")
        self.assertEqual(after_four.last_move_san, "Nc6")
        self.assertIsNone(after_four.played_move_uci)
        self.assertIsNone(after_four.played_move_san)


if __name__ == "__main__":
    unittest.main()
