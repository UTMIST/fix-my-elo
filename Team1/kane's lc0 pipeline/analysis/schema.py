"""Versioned data structures for Lc0 analysis records."""

from __future__ import annotations

import math


SCHEMA_VERSION = 2


def _coerce_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value)) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                number = float(text)
                return int(round(number)) if math.isfinite(number) else None
            except ValueError:
                return None
    return None


def _coerce_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_str(value):
    return "" if value is None else str(value)


def _coerce_opt_str(value):
    text = _coerce_str(value).strip()
    return text or None


def _coerce_str_list(value):
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [
        str(item).strip()
        for item in values
        if item is not None and str(item).strip()
    ]


def _side_from_fen(fen):
    parts = _coerce_str(fen).split()
    return parts[1] if len(parts) >= 2 and parts[1] in {"w", "b"} else None


def _coerce_side(value, fen=None):
    if value is None:
        side = _side_from_fen(fen)
        if side is None:
            raise ValueError("side_to_move is missing and cannot be derived from FEN")
        return side
    text = str(value).strip().lower()
    if text in {"w", "white"}:
        return "w"
    if text in {"b", "black"}:
        return "b"
    raise ValueError(f"Invalid side_to_move: {value!r}")


def json_safe(value):
    """Convert supported values to strict JSON-compatible structures."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats are not valid analysis data")
        return value
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    return str(value)


class PVLine:
    """A final MultiPV line scored from the analyzed side's perspective."""

    def __init__(
        self,
        multipv_index,
        move_uci,
        pv_uci_list,
        score_cp=None,
        mate=None,
        depth=None,
        nodes=None,
        seldepth=None,
        nps=None,
        time_ms=None,
        hashfull=None,
        tbhits=None,
        wdl=None,
    ):
        if score_cp is not None and mate is not None:
            raise ValueError("A PV score cannot contain both centipawns and mate")
        self.multipv_index = multipv_index
        self.move_uci = move_uci
        self.pv_uci_list = pv_uci_list
        self.score_cp = score_cp
        self.mate = mate
        self.depth = depth
        self.nodes = nodes
        self.seldepth = seldepth
        self.nps = nps
        self.time_ms = time_ms
        self.hashfull = hashfull
        self.tbhits = tbhits
        self.wdl = wdl

    def to_dict(self):
        return {
            "multipv_index": int(self.multipv_index),
            "move_uci": self.move_uci,
            "pv_uci_list": list(self.pv_uci_list),
            "score_cp": self.score_cp,
            "mate": self.mate,
            "depth": self.depth,
            "nodes": self.nodes,
            "seldepth": self.seldepth,
            "nps": self.nps,
            "time_ms": self.time_ms,
            "hashfull": self.hashfull,
            "tbhits": self.tbhits,
            "wdl": json_safe(self.wdl),
        }

    @classmethod
    def from_dict(cls, data):
        score_cp = _coerce_int(data.get("score_cp"))
        mate = _coerce_int(data.get("mate"))
        if mate is not None:
            score_cp = None
        return cls(
            multipv_index=_coerce_int(data.get("multipv_index")) or 1,
            move_uci=_coerce_str(data.get("move_uci")).strip(),
            pv_uci_list=_coerce_str_list(data.get("pv_uci_list")),
            score_cp=score_cp,
            mate=mate,
            depth=_coerce_int(data.get("depth")),
            nodes=_coerce_int(data.get("nodes")),
            seldepth=_coerce_int(data.get("seldepth")),
            nps=_coerce_int(data.get("nps")),
            time_ms=_coerce_int(data.get("time_ms")),
            hashfull=_coerce_int(data.get("hashfull")),
            tbhits=_coerce_int(data.get("tbhits")),
            wdl=json_safe(data.get("wdl")) if data.get("wdl") is not None else None,
        )


class RootCandidate:
    """Lc0 policy and search statistics for one legal root move."""

    def __init__(
        self,
        move_uci,
        P=None,
        N=None,
        Q=None,
        V=None,
        U=None,
        WL=None,
        D=None,
        M=None,
        S=None,
        n_delta=None,
    ):
        self.move_uci = move_uci
        self.P = P
        self.N = N
        self.Q = Q
        self.V = V
        self.U = U
        self.WL = WL
        self.D = D
        self.M = M
        self.S = S
        self.n_delta = n_delta

    def to_dict(self):
        return {
            "move_uci": self.move_uci,
            "P": self.P,
            "N": self.N,
            "Q": self.Q,
            "V": self.V,
            "U": self.U,
            "WL": self.WL,
            "D": self.D,
            "M": self.M,
            "S": self.S,
            "n_delta": self.n_delta,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            move_uci=_coerce_str(data.get("move_uci")).strip(),
            P=_coerce_float(data.get("P")),
            N=_coerce_int(data.get("N")),
            Q=_coerce_float(data.get("Q")),
            V=_coerce_float(data.get("V")),
            U=_coerce_float(data.get("U")),
            WL=_coerce_float(data.get("WL")),
            D=_coerce_float(data.get("D")),
            M=_coerce_float(data.get("M")),
            S=_coerce_float(data.get("S")),
            n_delta=_coerce_int(data.get("n_delta")),
        )


class AnalysisResult:
    """The complete result of one engine search."""

    def __init__(
        self,
        fen,
        side_to_move,
        bestmove_uci,
        pv_lines=None,
        root_candidates=None,
        raw_info=None,
        search_metadata=None,
        score_perspective="side_to_move",
    ):
        self.fen = fen
        self.side_to_move = _coerce_side(side_to_move, fen)
        self.bestmove_uci = bestmove_uci
        self.pv_lines = [] if pv_lines is None else pv_lines
        self.root_candidates = root_candidates
        self.raw_info = raw_info
        self.search_metadata = {} if search_metadata is None else search_metadata
        self.score_perspective = score_perspective

    def to_dict(self):
        return {
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "score_perspective": self.score_perspective,
            "bestmove_uci": self.bestmove_uci,
            "pv_lines": [line.to_dict() for line in self.pv_lines],
            "root_candidates": None
            if self.root_candidates is None
            else [candidate.to_dict() for candidate in self.root_candidates],
            "raw_info": None if self.raw_info is None else list(self.raw_info),
            "search_metadata": json_safe(self.search_metadata),
        }

    @classmethod
    def from_dict(cls, data):
        fen = _coerce_str(data.get("fen")).strip()
        pv_lines = [
            PVLine.from_dict(item)
            for item in data.get("pv_lines") or []
            if isinstance(item, dict)
        ]
        candidates_data = data.get("root_candidates")
        root_candidates = None
        if isinstance(candidates_data, list):
            root_candidates = [
                RootCandidate.from_dict(item) for item in candidates_data if isinstance(item, dict)
            ]
        raw_info = data.get("raw_info")
        if not isinstance(raw_info, list):
            raw_info = None
        return cls(
            fen=fen,
            side_to_move=data.get("side_to_move"),
            bestmove_uci=_coerce_opt_str(data.get("bestmove_uci")),
            pv_lines=pv_lines,
            root_candidates=root_candidates,
            raw_info=None if raw_info is None else _coerce_str_list(raw_info),
            search_metadata=dict(data.get("search_metadata") or {}),
            score_perspective=_coerce_opt_str(data.get("score_perspective"))
            or "side_to_move",
        )


class PositionSource:
    """Identifies where an analyzed position came from."""

    def __init__(
        self,
        kind,
        path=None,
        position_index=None,
        line_number=None,
        game_index=None,
        headers=None,
        ply=None,
        last_move_uci=None,
        last_move_san=None,
        played_move_uci=None,
        played_move_san=None,
    ):
        self.kind = kind
        self.path = path
        self.position_index = position_index
        self.line_number = line_number
        self.game_index = game_index
        self.headers = {} if headers is None else headers
        self.ply = ply
        self.last_move_uci = last_move_uci
        self.last_move_san = last_move_san
        self.played_move_uci = played_move_uci
        self.played_move_san = played_move_san

    def to_dict(self):
        return {
            "kind": self.kind,
            "path": self.path,
            "position_index": self.position_index,
            "line_number": self.line_number,
            "game_index": self.game_index,
            "headers": {str(key): str(value) for key, value in self.headers.items()},
            "ply": self.ply,
            "last_move_uci": self.last_move_uci,
            "last_move_san": self.last_move_san,
            "played_move_uci": self.played_move_uci,
            "played_move_san": self.played_move_san,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            kind=_coerce_opt_str(data.get("kind")) or "fen",
            path=_coerce_opt_str(data.get("path")),
            position_index=_coerce_int(data.get("position_index")),
            line_number=_coerce_int(data.get("line_number")),
            game_index=_coerce_int(data.get("game_index")),
            headers=dict(data.get("headers") or {}),
            ply=_coerce_int(data.get("ply")),
            last_move_uci=_coerce_opt_str(data.get("last_move_uci")),
            last_move_san=_coerce_opt_str(data.get("last_move_san")),
            played_move_uci=_coerce_opt_str(data.get("played_move_uci")),
            played_move_san=_coerce_opt_str(data.get("played_move_san")),
        )


class EngineIdentity:
    """Reproducibility metadata for the engine and network."""

    def __init__(
        self,
        uci_name,
        uci_author=None,
        executable_path=None,
        executable_sha256=None,
        network_path=None,
        network_sha256=None,
        options=None,
    ):
        self.uci_name = uci_name
        self.uci_author = uci_author
        self.executable_path = executable_path
        self.executable_sha256 = executable_sha256
        self.network_path = network_path
        self.network_sha256 = network_sha256
        self.options = {} if options is None else options

    def to_dict(self):
        return {
            "uci_name": self.uci_name,
            "uci_author": self.uci_author,
            "executable_path": self.executable_path,
            "executable_sha256": self.executable_sha256,
            "network_path": self.network_path,
            "network_sha256": self.network_sha256,
            "options": json_safe(self.options),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            uci_name=_coerce_opt_str(data.get("uci_name")) or "unknown",
            uci_author=_coerce_opt_str(data.get("uci_author")),
            executable_path=_coerce_opt_str(data.get("executable_path")),
            executable_sha256=_coerce_opt_str(data.get("executable_sha256")),
            network_path=_coerce_opt_str(data.get("network_path")),
            network_sha256=_coerce_opt_str(data.get("network_sha256")),
            options=dict(data.get("options") or {}),
        )


class IntuitionReport:
    """A versioned success record with root and candidate analysis."""

    def __init__(
        self,
        fen,
        best_move,
        best_pv=None,
        intuitive_moves=None,
        intuitive_but_bad=None,
        metadata=None,
        analysis=None,
        source=None,
        engine=None,
        candidate_moves=None,
        candidate_evaluations=None,
        generated_at_utc=None,
    ):
        self.fen = fen
        self.best_move = best_move
        self.best_pv = [] if best_pv is None else best_pv
        self.intuitive_moves = [] if intuitive_moves is None else intuitive_moves
        self.intuitive_but_bad = [] if intuitive_but_bad is None else intuitive_but_bad
        self.metadata = {} if metadata is None else metadata
        self.analysis = analysis
        self.source = source
        self.engine = engine
        self.candidate_moves = (
            list(self.intuitive_moves) if candidate_moves is None else candidate_moves
        )
        self.candidate_evaluations = (
            [] if candidate_evaluations is None else candidate_evaluations
        )
        self.generated_at_utc = generated_at_utc

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "intuition_analysis",
            "status": "ok",
            "generated_at_utc": self.generated_at_utc,
            "fen": self.fen,
            "best_move": self.best_move,
            "best_pv": list(self.best_pv),
            "candidate_moves": list(self.candidate_moves),
            "intuitive_moves": list(self.intuitive_moves),
            "intuitive_but_bad": json_safe(self.intuitive_but_bad),
            "candidate_evaluations": json_safe(self.candidate_evaluations),
            "analysis": None if self.analysis is None else self.analysis.to_dict(),
            "source": None if self.source is None else self.source.to_dict(),
            "engine": None if self.engine is None else self.engine.to_dict(),
            "metadata": json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data):
        if (
            data.get("record_type") == "analysis_error"
            or data.get("status") == "error"
            or data.get("error") is not None
        ):
            raise ValueError("An error record cannot be loaded as IntuitionReport")
        record_type = data.get("record_type")
        if record_type not in {None, "intuition_analysis"}:
            raise ValueError(f"Unsupported record_type: {record_type!r}")
        analysis_data = data.get("analysis")
        source_data = data.get("source")
        engine_data = data.get("engine")
        return cls(
            fen=_coerce_str(data.get("fen")).strip(),
            best_move=_coerce_opt_str(data.get("best_move")),
            best_pv=_coerce_str_list(data.get("best_pv")),
            intuitive_moves=_coerce_str_list(data.get("intuitive_moves")),
            intuitive_but_bad=list(data.get("intuitive_but_bad") or []),
            metadata=dict(data.get("metadata") or {}),
            analysis=AnalysisResult.from_dict(analysis_data)
            if isinstance(analysis_data, dict)
            else None,
            source=PositionSource.from_dict(source_data)
            if isinstance(source_data, dict)
            else None,
            engine=EngineIdentity.from_dict(engine_data)
            if isinstance(engine_data, dict)
            else None,
            candidate_moves=_coerce_str_list(
                data.get("candidate_moves", data.get("intuitive_moves"))
            ),
            candidate_evaluations=list(data.get("candidate_evaluations") or []),
            generated_at_utc=_coerce_opt_str(data.get("generated_at_utc")),
        )


class ErrorRecord:
    """A typed JSONL record for a position or pipeline failure."""

    def __init__(
        self,
        message,
        code,
        stage,
        fen=None,
        exception_type=None,
        traceback_text=None,
        source=None,
        engine=None,
    ):
        self.message = message
        self.code = code
        self.stage = stage
        self.fen = fen
        self.exception_type = exception_type
        self.traceback_text = traceback_text
        self.source = source
        self.engine = engine

    def to_dict(self):
        details = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback_text,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "analysis_error",
            "status": "error",
            "fen": self.fen,
            "error": self.message,
            "error_details": details,
            "source": None if self.source is None else self.source.to_dict(),
            "engine": None if self.engine is None else self.engine.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("record_type") not in {None, "analysis_error"}:
            raise ValueError("Record is not an analysis error")
        details = data.get("error_details") or {}
        source_data = data.get("source")
        engine_data = data.get("engine")
        return cls(
            message=_coerce_opt_str(details.get("message") or data.get("error"))
            or "Unknown error",
            code=_coerce_opt_str(details.get("code")) or "internal_error",
            stage=_coerce_opt_str(details.get("stage")) or "analysis",
            fen=_coerce_opt_str(data.get("fen")),
            exception_type=_coerce_opt_str(details.get("exception_type")),
            traceback_text=_coerce_opt_str(details.get("traceback")),
            source=PositionSource.from_dict(source_data)
            if isinstance(source_data, dict)
            else None,
            engine=EngineIdentity.from_dict(engine_data)
            if isinstance(engine_data, dict)
            else None,
        )
