"""Single-search python-chess wrapper for Leela Chess Zero."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time

import chess
import chess.engine

from analysis.lc0_verbose_parser import parse_last_root_candidates
from analysis.schema import AnalysisResult, EngineIdentity, PVLine


LOGGER = logging.getLogger(__name__)


def _expanded_absolute_path(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _resolve_executable(path):
    if not path:
        raise ValueError("An Lc0 executable path is required")
    expanded = os.path.expanduser(os.path.expandvars(path))
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    located = shutil.which(expanded)
    if located:
        return os.path.abspath(located)
    raise FileNotFoundError(f"Lc0 executable not found: {path}")


def _resolve_input_file(path, label):
    resolved = _expanded_absolute_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} not found: {path}")
    return resolved


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_file_segment(path, start_offset=0, max_bytes=5_000_000):
    """Read newly appended logfile bytes and tolerate truncation or rotation."""
    if not path or not os.path.isfile(path):
        return ""
    size = os.path.getsize(path)
    start = start_offset if 0 <= start_offset <= size else 0
    if size - start > max_bytes:
        start = size - max_bytes
    with open(path, "rb") as handle:
        handle.seek(start)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _keep_last_lines(lines, max_lines=8000):
    return lines if len(lines) <= max_lines else lines[-max_lines:]


def _wdl_to_dict(value, turn):
    if value is None:
        return None
    try:
        score = value.pov(turn) if hasattr(value, "pov") else value
        return {
            "wins": int(score.wins),
            "draws": int(score.draws),
            "losses": int(score.losses),
        }
    except (AttributeError, TypeError, ValueError):
        return None


class Lc0Engine:
    """Control one Lc0 process and return reproducible analysis records."""

    def __init__(
        self,
        engine_path,
        weights_path=None,
        threads=None,
        verbose_move_stats=True,
        cwd=None,
        uci_options=None,
        timeout=10.0,
        logfile_path=None,
        prefer_logfile=True,
        debug=False,
        weights_flag_style="equals",
    ):
        if threads is not None and (not isinstance(threads, int) or threads <= 0):
            raise ValueError("threads must be a positive integer")
        if weights_flag_style not in {"equals", "space"}:
            raise ValueError("weights_flag_style must be 'equals' or 'space'")

        self._engine = None
        self._debug = debug
        self._prefer_logfile = prefer_logfile
        self._verbose_move_stats = bool(verbose_move_stats)
        self._engine_path = _resolve_executable(engine_path)
        self._weights_path = (
            _resolve_input_file(weights_path, "Weights file") if weights_path else None
        )
        self._logfile_path = (
            _expanded_absolute_path(logfile_path) if logfile_path else None
        )

        command = [self._engine_path]
        if self._weights_path:
            if weights_flag_style == "space":
                command.extend(["--weights", self._weights_path])
            else:
                command.append("--weights=" + self._weights_path)
        if self._verbose_move_stats:
            command.append("--verbose-move-stats")
        if self._logfile_path:
            parent = os.path.dirname(self._logfile_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._logfile_path, "ab"):
                pass
            command.append("--logfile=" + self._logfile_path)

        process_cwd = (
            _expanded_absolute_path(cwd)
            if cwd is not None
            else os.path.dirname(self._engine_path) or None
        )
        self._engine = chess.engine.SimpleEngine.popen_uci(
            command,
            timeout=timeout,
            cwd=process_cwd,
        )

        configured = {}
        if threads is not None and "Threads" in self._engine.options:
            configured["Threads"] = threads
        if self._weights_path and "WeightsFile" in self._engine.options:
            configured["WeightsFile"] = self._weights_path
        if self._verbose_move_stats and "VerboseMoveStats" in self._engine.options:
            configured["VerboseMoveStats"] = True
        if "UCI_ShowWDL" in self._engine.options:
            configured["UCI_ShowWDL"] = True
        for key, value in (uci_options or {}).items():
            if key in self._engine.options:
                configured[key] = value
        if configured:
            self._engine.configure(configured)

        reproducibility_options = (
            "Threads",
            "Backend",
            "BackendOptions",
            "MinibatchSize",
            "MaxPrefetch",
            "NNCacheSize",
            "UCI_ShowWDL",
            "WeightsFile",
            "VerboseMoveStats",
        )
        effective_uci = {
            name: configured.get(name, self._engine.options[name].default)
            for name in reproducibility_options
            if name in self._engine.options
        }
        identity_options = {
            "threads": threads,
            "verbose_move_stats": self._verbose_move_stats,
            "configured_uci": configured,
            "effective_uci": effective_uci,
        }
        self._identity = EngineIdentity(
            uci_name=self._engine.id.get("name", "Lc0"),
            uci_author=self._engine.id.get("author"),
            executable_path=self._engine_path,
            executable_sha256=_sha256_file(self._engine_path),
            network_path=self._weights_path,
            network_sha256=_sha256_file(self._weights_path)
            if self._weights_path
            else None,
            options=identity_options,
        )

    @property
    def identity(self):
        return self._identity

    def close(self):
        """Stop the engine process; repeated calls are safe."""
        if self._engine is not None:
            engine, self._engine = self._engine, None
            engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _logfile_offset(self):
        if not self._logfile_path or not os.path.isfile(self._logfile_path):
            return 0
        return os.path.getsize(self._logfile_path)

    def _fresh_logfile_lines(self, start_offset):
        if not self._logfile_path:
            return []
        text = _read_text_file_segment(self._logfile_path, start_offset)
        return [line for line in text.splitlines() if line.strip()]

    def analyze_fen(
        self,
        fen,
        movetime_ms=None,
        nodes=None,
        depth=None,
        multipv=1,
        options=None,
        capture_root_stats=True,
        include_raw_info=True,
    ):
        """Analyze a FEN once, capturing final MultiPV data and root statistics."""
        if self._engine is None:
            raise RuntimeError("Lc0 engine is closed")
        if not isinstance(multipv, int) or multipv <= 0:
            raise ValueError("multipv must be a positive integer")
        for name, value in (
            ("movetime_ms", movetime_ms),
            ("nodes", nodes),
            ("depth", depth),
        ):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        board = chess.Board(fen)
        if movetime_ms is None and nodes is None and depth is None:
            movetime_ms = 1000
        limit = chess.engine.Limit(
            time=movetime_ms / 1000.0 if movetime_ms is not None else None,
            nodes=nodes,
            depth=depth,
        )
        safe_options = {
            key: value
            for key, value in (options or {}).items()
            if key in self._engine.options
        }

        log_offset = self._logfile_offset()
        uci_strings = []
        started = time.perf_counter()
        with self._engine.analysis(
            board,
            limit=limit,
            multipv=multipv,
            info=chess.engine.INFO_ALL,
            options=safe_options,
        ) as stream:
            for update in stream:
                message = update.get("string")
                if message is not None:
                    uci_strings.extend(
                        part
                        for part in (line.strip() for line in str(message).splitlines())
                        if part
                    )
            best = stream.wait()
            info_dicts = stream.multipv
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))

        if self._logfile_path:
            time.sleep(0.03)
        logfile_lines = self._fresh_logfile_lines(log_offset)
        legal_moves = [move.uci() for move in board.legal_moves]

        root_candidates = None
        policy_source = "not_requested"
        if capture_root_stats and self._verbose_move_stats:
            root_candidates = parse_last_root_candidates(
                "\n".join(uci_strings), legal_moves=legal_moves
            )
            policy_source = "uci_info_string"
            if not root_candidates:
                root_candidates = parse_last_root_candidates(
                    "\n".join(logfile_lines), legal_moves=legal_moves
                )
                policy_source = "logfile" if root_candidates else "unavailable"

        pv_lines = []
        for index, info in enumerate(info_dicts):
            pv_moves = info.get("pv", [])
            if not pv_moves:
                continue
            score_cp = None
            mate = None
            score = info.get("score")
            if score is not None:
                pov_score = score.pov(board.turn)
                if pov_score.is_mate():
                    mate = pov_score.mate()
                else:
                    score_cp = pov_score.score()
            multipv_value = info.get("multipv")
            multipv_index = multipv_value if isinstance(multipv_value, int) else index + 1
            elapsed = info.get("time")
            pv_lines.append(
                PVLine(
                    multipv_index=multipv_index,
                    move_uci=pv_moves[0].uci(),
                    pv_uci_list=[move.uci() for move in pv_moves],
                    score_cp=score_cp,
                    mate=mate,
                    depth=info.get("depth"),
                    nodes=info.get("nodes"),
                    seldepth=info.get("seldepth"),
                    nps=info.get("nps"),
                    time_ms=int(round(float(elapsed) * 1000))
                    if elapsed is not None
                    else None,
                    hashfull=info.get("hashfull"),
                    tbhits=info.get("tbhits"),
                    wdl=_wdl_to_dict(info.get("wdl"), board.turn),
                )
            )
        pv_lines.sort(key=lambda line: line.multipv_index)

        bestmove_uci = best.move.uci() if best.move is not None else None
        if bestmove_uci is None and pv_lines:
            bestmove_uci = pv_lines[0].move_uci

        raw_lines = []
        if include_raw_info:
            if self._prefer_logfile:
                raw_lines.extend(logfile_lines)
                raw_lines.extend(uci_strings)
            else:
                raw_lines.extend(uci_strings)
                raw_lines.extend(logfile_lines)
            raw_lines = _keep_last_lines(raw_lines)

        if self._debug:
            LOGGER.info(
                "Lc0 search completed: pv=%d policy=%d elapsed_ms=%d",
                len(pv_lines),
                len(root_candidates or []),
                elapsed_ms,
            )

        return AnalysisResult(
            fen=fen,
            side_to_move="w" if board.turn == chess.WHITE else "b",
            bestmove_uci=bestmove_uci,
            pv_lines=pv_lines,
            root_candidates=root_candidates,
            raw_info=raw_lines or None,
            search_metadata={
                "requested_limit": {
                    "movetime_ms": movetime_ms,
                    "nodes": nodes,
                    "depth": depth,
                },
                "requested_multipv": multipv,
                "elapsed_ms": elapsed_ms,
                "policy_source": policy_source,
                "uci_string_count": len(uci_strings),
                "logfile_line_count": len(logfile_lines),
            },
        )

    def analyze_after_move(
        self,
        fen,
        uci_move,
        movetime_ms=None,
        nodes=None,
        depth=None,
        multipv=1,
        options=None,
        capture_root_stats=False,
        include_raw_info=False,
    ):
        """Apply one legal move and analyze the resulting position once."""
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move {uci_move} for position: {fen}")
        board.push(move)
        return self.analyze_fen(
            board.fen(),
            movetime_ms=movetime_ms,
            nodes=nodes,
            depth=depth,
            multipv=multipv,
            options=options,
            capture_root_stats=capture_root_stats,
            include_raw_info=include_raw_info,
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
