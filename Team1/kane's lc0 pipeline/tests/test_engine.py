import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chess
import chess.engine

from analysis.schema import RootCandidate
from engines.lc0_engine import Lc0Engine, _read_text_file_segment


START_FEN = chess.STARTING_FEN


class _FakeAnalysisStream:
    def __init__(self, updates, multipv, best_move):
        self._updates = updates
        self.multipv = multipv
        self._best_move = best_move
        self.wait_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self._updates)

    def wait(self):
        self.wait_calls += 1
        return SimpleNamespace(move=chess.Move.from_uci(self._best_move))


class _FakeSimpleEngine:
    def __init__(self, stream):
        self.stream = stream
        self.options = {"Hash": object()}
        self.analysis_calls = []
        self.quit_calls = 0

    def analysis(self, board, **kwargs):
        self.analysis_calls.append((board.copy(), kwargs))
        return self.stream

    def quit(self):
        self.quit_calls += 1


class _FakeSpawnedEngine:
    def __init__(self):
        self.options = {}
        self.id = {"name": "Fake Lc0", "author": "Tests"}
        self.quit_calls = 0

    def configure(self, options):
        raise AssertionError(f"Unexpected UCI options: {options}")

    def quit(self):
        self.quit_calls += 1


def _wrapper_with_fake_engine(fake_engine):
    wrapper = object.__new__(Lc0Engine)
    wrapper._engine = fake_engine
    wrapper._debug = False
    wrapper._prefer_logfile = True
    wrapper._verbose_move_stats = True
    wrapper._engine_path = "fake-lc0"
    wrapper._weights_path = "fake-network"
    wrapper._logfile_path = None
    return wrapper


class Lc0EngineSearchTests(unittest.TestCase):
    def test_analyze_fen_uses_one_streaming_search_for_pv_and_policy(self):
        first_score = chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE)
        second_score = chess.engine.PovScore(chess.engine.Cp(18), chess.WHITE)
        stream = _FakeAnalysisStream(
            updates=[{"string": "root move statistics"}, {"depth": 8}],
            multipv=[
                {
                    "multipv": 1,
                    "pv": [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")],
                    "score": first_score,
                    "depth": 12,
                    "nodes": 900,
                    "time": 0.25,
                },
                {
                    "multipv": 2,
                    "pv": [chess.Move.from_uci("d2d4"), chess.Move.from_uci("d7d5")],
                    "score": second_score,
                    "depth": 11,
                    "nodes": 800,
                    "time": 0.20,
                },
            ],
            best_move="e2e4",
        )
        fake_engine = _FakeSimpleEngine(stream)
        wrapper = _wrapper_with_fake_engine(fake_engine)
        parsed_candidates = [
            RootCandidate(move_uci="e2e4", P=0.62, N=700, Q=0.14, V=0.10, U=0.04)
        ]

        with patch(
            "engines.lc0_engine.parse_last_root_candidates",
            return_value=parsed_candidates,
        ) as parse_candidates:
            result = wrapper.analyze_fen(
                START_FEN,
                nodes=1000,
                multipv=2,
                options={"Hash": 64, "Unsupported": True},
                include_raw_info=True,
            )

        self.assertEqual(len(fake_engine.analysis_calls), 1)
        self.assertEqual(stream.wait_calls, 1)
        searched_board, search_args = fake_engine.analysis_calls[0]
        self.assertEqual(searched_board.fen(), START_FEN)
        self.assertEqual(search_args["multipv"], 2)
        self.assertEqual(search_args["limit"].nodes, 1000)
        self.assertEqual(search_args["options"], {"Hash": 64})
        parse_candidates.assert_called_once()

        self.assertEqual(result.bestmove_uci, "e2e4")
        self.assertEqual([line.move_uci for line in result.pv_lines], ["e2e4", "d2d4"])
        self.assertEqual([line.score_cp for line in result.pv_lines], [42, 18])
        self.assertEqual(result.root_candidates[0].to_dict(), parsed_candidates[0].to_dict())
        self.assertEqual(result.raw_info, ["root move statistics"])
        self.assertEqual(result.search_metadata["policy_source"], "uci_info_string")

        wrapper.close()
        self.assertEqual(fake_engine.quit_calls, 1)

    def test_analyze_after_move_delegates_to_one_resulting_position_search(self):
        wrapper = object.__new__(Lc0Engine)
        calls = []

        def record_analysis(fen, **kwargs):
            calls.append((fen, kwargs))
            return "result"

        wrapper.analyze_fen = record_analysis
        result = wrapper.analyze_after_move(
            START_FEN,
            "e2e4",
            depth=7,
            multipv=1,
            capture_root_stats=False,
            include_raw_info=False,
        )

        self.assertEqual(result, "result")
        self.assertEqual(len(calls), 1)
        resulting_board = chess.Board(calls[0][0])
        self.assertEqual(resulting_board.piece_at(chess.E4), chess.Piece(chess.PAWN, chess.WHITE))
        self.assertIsNone(resulting_board.piece_at(chess.E2))
        self.assertEqual(resulting_board.turn, chess.BLACK)
        self.assertEqual(calls[0][1]["depth"], 7)
        self.assertFalse(calls[0][1]["capture_root_stats"])


class Lc0EnginePathAndLogTests(unittest.TestCase):
    def test_relative_paths_are_resolved_before_engine_cwd_is_applied(self):
        original_cwd = os.getcwd()
        wrapper = None
        with tempfile.TemporaryDirectory() as directory:
            engine_dir = os.path.join(directory, "bin")
            network_dir = os.path.join(directory, "nets")
            runtime_dir = os.path.join(directory, "runtime")
            log_dir = os.path.join(directory, "logs")
            for path in (engine_dir, network_dir, runtime_dir, log_dir):
                os.makedirs(path)

            engine_path = os.path.join(engine_dir, "lc0.exe")
            weights_path = os.path.join(network_dir, "network.pb.gz")
            logfile_path = os.path.join(log_dir, "lc0.log")
            with open(engine_path, "wb") as handle:
                handle.write(b"engine")
            with open(weights_path, "wb") as handle:
                handle.write(b"network")
            with open(logfile_path, "w", encoding="utf-8") as handle:
                handle.write("existing diagnostics\n")

            fake_process = _FakeSpawnedEngine()
            try:
                os.chdir(directory)
                with patch(
                    "engines.lc0_engine.chess.engine.SimpleEngine.popen_uci",
                    return_value=fake_process,
                ) as spawn:
                    wrapper = Lc0Engine(
                        engine_path=os.path.join("bin", "lc0.exe"),
                        weights_path=os.path.join("nets", "network.pb.gz"),
                        logfile_path=os.path.join("logs", "lc0.log"),
                        cwd="runtime",
                        verbose_move_stats=False,
                    )
            finally:
                os.chdir(original_cwd)

            command = spawn.call_args.args[0]
            spawn_kwargs = spawn.call_args.kwargs
            self.assertEqual(command[0], os.path.abspath(engine_path))
            self.assertIn("--weights=" + os.path.abspath(weights_path), command)
            self.assertIn("--logfile=" + os.path.abspath(logfile_path), command)
            self.assertEqual(spawn_kwargs["cwd"], os.path.abspath(runtime_dir))
            self.assertEqual(wrapper.identity.executable_path, os.path.abspath(engine_path))
            self.assertEqual(wrapper.identity.network_path, os.path.abspath(weights_path))
            with open(logfile_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "existing diagnostics\n")

        wrapper.close()
        self.assertEqual(fake_process.quit_calls, 1)

    def test_log_segment_reads_only_appended_content_and_recovers_after_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "lc0.log")
            with open(path, "wb") as handle:
                handle.write(b"previous search\n")
            offset = os.path.getsize(path)
            with open(path, "ab") as handle:
                handle.write(b"current search\n")

            self.assertEqual(
                _read_text_file_segment(path, offset),
                "current search\n",
            )

            with open(path, "wb") as handle:
                handle.write(b"rotated search\n")
            self.assertEqual(
                _read_text_file_segment(path, start_offset=10_000),
                "rotated search\n",
            )


if __name__ == "__main__":
    unittest.main()
