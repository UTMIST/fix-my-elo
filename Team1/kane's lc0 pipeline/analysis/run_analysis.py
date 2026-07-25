"""Command-line entry point for versioned Lc0 analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

try:
    import chess
    import chess.engine
    import chess.pgn
except ImportError:
    print(
        "python-chess is required. Install it with: python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2)

from analysis.intuitive_vs_best import build_intuition_report
from analysis.schema import ErrorRecord, PositionSource
from config.lc0_config import load_config_from_args
from engines.lc0_engine import Lc0Engine


def _absolute_path(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _next_played_move(board, moves, index):
    if index >= len(moves):
        return None, None
    move = moves[index]
    return move.uci(), board.san(move)


def _yield_positions_from_pgn(path, every_n_plies, include_start=False):
    """Yield sampled FENs with game, ply, and played-move provenance."""
    if every_n_plies <= 0:
        raise ValueError("--every_n_plies must be >= 1")
    source_path = _absolute_path(path)
    position_index = 0
    with open(source_path, "r", encoding="utf-8", errors="replace") as pgn_file:
        game_index = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_index += 1
            if game.errors:
                messages = "; ".join(str(error) for error in game.errors)
                raise ValueError(f"PGN game {game_index} contains errors: {messages}")

            headers = {str(key): str(value) for key, value in game.headers.items()}
            moves = list(game.mainline_moves())
            board = game.board()
            if include_start:
                position_index += 1
                played_uci, played_san = _next_played_move(board, moves, 0)
                yield board.fen(), PositionSource(
                    kind="pgn",
                    path=source_path,
                    position_index=position_index,
                    game_index=game_index,
                    headers=headers,
                    ply=0,
                    played_move_uci=played_uci,
                    played_move_san=played_san,
                )

            last_move_uci = None
            last_move_san = None
            for ply, move in enumerate(moves, start=1):
                last_move_uci = move.uci()
                last_move_san = board.san(move)
                board.push(move)
                if ply % every_n_plies != 0:
                    continue
                position_index += 1
                played_uci, played_san = _next_played_move(board, moves, ply)
                yield board.fen(), PositionSource(
                    kind="pgn",
                    path=source_path,
                    position_index=position_index,
                    game_index=game_index,
                    headers=headers,
                    ply=ply,
                    last_move_uci=last_move_uci,
                    last_move_san=last_move_san,
                    played_move_uci=played_uci,
                    played_move_san=played_san,
                )


def _yield_fens_from_pgn(path, every_n_plies, include_start=False):
    """Compatibility iterator returning only FEN strings."""
    for fen, _ in _yield_positions_from_pgn(path, every_n_plies, include_start):
        yield fen


def _yield_positions_from_list(path):
    """Yield FEN-list entries with source line numbers."""
    source_path = _absolute_path(path)
    position_index = 0
    with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            fen = line.strip()
            if not fen or fen.startswith("#"):
                continue
            position_index += 1
            yield fen, PositionSource(
                kind="fen_list",
                path=source_path,
                position_index=position_index,
                line_number=line_number,
            )


def _yield_fens_from_list(path):
    """Compatibility iterator returning only FEN strings."""
    for fen, _ in _yield_positions_from_list(path):
        yield fen


def _ensure_parent_dir(file_path):
    parent = os.path.dirname(_absolute_path(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_record(handle, record):
    handle.write(
        json.dumps(
            record.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )


def _classify_error(exc):
    if isinstance(exc, ValueError):
        return "invalid_fen", "input"
    if isinstance(exc, (chess.engine.EngineError, chess.engine.EngineTerminatedError)):
        return "engine_analysis_failed", "analysis"
    return "internal_error", "analysis"


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze FEN, FEN-list, or PGN positions with Lc0."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fen", help="Analyze one FEN string.")
    source.add_argument("--pgn", help="Path to a PGN file.")
    source.add_argument("--fen_list", help="Path to a file containing one FEN per line.")

    parser.add_argument("--out_jsonl", help="Write JSONL here instead of stdout.")
    parser.add_argument("--lc0_logfile", help="Append Lc0 diagnostics to this file.")
    parser.add_argument("--lc0_path", required=True, help="Path to the Lc0 executable.")
    parser.add_argument(
        "--weights_path", required=True, help="Path to the network weights (.pb.gz)."
    )
    parser.add_argument("--threads", type=int, help="Positive engine thread count.")

    parser.add_argument("--movetime_ms", type=int, help="Time limit for each engine search.")
    parser.add_argument("--nodes", type=int, help="Node limit for each engine search.")
    parser.add_argument("--depth", type=int, help="Depth limit for each engine search.")
    parser.add_argument("--multipv", type=int, default=5, help="Number of final PV lines.")
    parser.add_argument(
        "--top_intuitive", type=int, default=3, help="Number of policy moves to inspect."
    )
    parser.add_argument(
        "--bad_threshold_cp", type=int, default=80, help="Centipawn worsening threshold."
    )
    parser.add_argument(
        "--verbose_move_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture Lc0 root policy statistics (enabled by default).",
    )
    parser.add_argument(
        "--include_raw_info",
        action="store_true",
        help="Include raw UCI/log lines in each root analysis record.",
    )
    parser.add_argument(
        "--every_n_plies", type=int, default=1, help="For PGN input, sample every N plies."
    )
    parser.add_argument(
        "--include_start", action="store_true", help="Include each PGN starting position."
    )
    parser.add_argument("--debug", action="store_true", help="Include tracebacks in error records.")
    return parser


def run_cli(argv=None, engine_factory=None):
    """Run the pipeline and return 0, 1 for data errors, or 2 for setup errors."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.every_n_plies <= 0:
        parser.error("--every_n_plies must be >= 1")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be > 0")
    try:
        config = load_config_from_args(args)
    except ValueError as exc:
        parser.error(f"Invalid configuration: {exc}")

    if args.fen is not None:
        positions = iter(
            [
                (
                    args.fen,
                    PositionSource(kind="fen", position_index=1),
                )
            ]
        )
        source_hint = PositionSource(kind="fen", position_index=1)
    elif args.pgn is not None:
        input_path = _absolute_path(args.pgn)
        if not os.path.isfile(input_path):
            parser.error(f"PGN not found: {args.pgn}")
        positions = iter(
            _yield_positions_from_pgn(
                input_path, args.every_n_plies, include_start=args.include_start
            )
        )
        source_hint = PositionSource(kind="pgn", path=input_path)
    else:
        input_path = _absolute_path(args.fen_list)
        if not os.path.isfile(input_path):
            parser.error(f"FEN list not found: {args.fen_list}")
        positions = iter(_yield_positions_from_list(input_path))
        source_hint = PositionSource(kind="fen_list", path=input_path)

    engine = None
    out_handle = None
    close_output = False
    processed = 0
    failures = 0
    try:
        if args.out_jsonl:
            _ensure_parent_dir(args.out_jsonl)
            out_handle = open(_absolute_path(args.out_jsonl), "w", encoding="utf-8")
            close_output = True
        else:
            out_handle = sys.stdout

        factory = Lc0Engine if engine_factory is None else engine_factory
        try:
            engine = factory(
                engine_path=args.lc0_path,
                weights_path=args.weights_path,
                threads=args.threads,
                logfile_path=args.lc0_logfile,
                verbose_move_stats=args.verbose_move_stats,
                debug=args.debug,
            )
        except Exception as exc:
            _write_record(
                out_handle,
                ErrorRecord(
                    message=str(exc),
                    code="engine_start_failed",
                    stage="engine_start",
                    exception_type=type(exc).__name__,
                    traceback_text=traceback.format_exc() if args.debug else None,
                    source=source_hint,
                ),
            )
            return 2

        while True:
            try:
                fen, position_source = next(positions)
            except StopIteration:
                break
            except Exception as exc:
                failures += 1
                _write_record(
                    out_handle,
                    ErrorRecord(
                        message=str(exc),
                        code="input_read_failed",
                        stage="input",
                        exception_type=type(exc).__name__,
                        traceback_text=traceback.format_exc() if args.debug else None,
                        source=source_hint,
                        engine=engine.identity,
                    ),
                )
                break

            try:
                report = build_intuition_report(
                    engine, fen.strip(), config, source=position_source
                )
                _write_record(out_handle, report)
                processed += 1
            except Exception as exc:
                failures += 1
                code, stage = _classify_error(exc)
                _write_record(
                    out_handle,
                    ErrorRecord(
                        message=str(exc),
                        code=code,
                        stage=stage,
                        fen=fen,
                        exception_type=type(exc).__name__,
                        traceback_text=traceback.format_exc() if args.debug else None,
                        source=position_source,
                        engine=engine.identity,
                    ),
                )

        if args.out_jsonl:
            print(
                f"Wrote {processed} analysis record(s) and {failures} error record(s) "
                f"to {_absolute_path(args.out_jsonl)}",
                file=sys.stderr,
            )
        return 1 if failures else 0
    except OSError as exc:
        print(f"Output failure: {exc}", file=sys.stderr)
        return 2
    finally:
        if close_output and out_handle is not None:
            out_handle.close()
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                print(f"Engine shutdown warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run_cli())
