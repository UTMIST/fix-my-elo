"""Parsers for Lc0 verbose root move statistics."""

from __future__ import annotations

import re

from analysis.schema import RootCandidate


_MOVE_RE = re.compile(r"(?<![A-Za-z0-9])([a-h][1-8][a-h][1-8][qrbn]?)(?=\s|$)", re.IGNORECASE)
_PROMO_PIECES = {"q", "r", "b", "n"}


def _is_uci_move_token(token):
    if not token or len(token) not in {4, 5}:
        return False
    text = token.lower()
    if not ("a" <= text[0] <= "h" and "1" <= text[1] <= "8"):
        return False
    if not ("a" <= text[2] <= "h" and "1" <= text[3] <= "8"):
        return False
    return len(text) == 4 or text[4] in _PROMO_PIECES


def _strip_info_prefix(line):
    """Remove UCI or logfile prefixes while preserving the payload."""
    text = line.strip()
    marker = text.lower().find("info string")
    if marker >= 0:
        return text[marker + len("info string") :].lstrip()
    if "]" in text:
        return text.rsplit("]", 1)[1].lstrip()
    return text


def _candidate_payload(line):
    payload = _strip_info_prefix(line)
    match = _MOVE_RE.search(payload)
    if match is None:
        return None
    payload = payload[match.start() :]
    first = payload.split(None, 1)[0]
    if not _is_uci_move_token(first):
        return None
    lowered = payload.lower()
    if "n:" not in lowered or "p:" not in lowered:
        return None
    if not any(label in lowered for label in ("q:", "u:", "v:", "wl:")):
        return None
    return payload


def _parse_int_after_label(text, label):
    match = re.search(re.escape(label) + r"\s*(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_float_after_label(text, label):
    match = re.search(
        re.escape(label) + r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _parse_percent_as_prob(text, label="P:"):
    match = re.search(
        re.escape(label) + r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(%)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    return value / 100.0 if match.group(2) else value


def _parse_visit_delta(text):
    match = re.search(
        r"N:\s*\d+\s*\(\s*([+-])\s*(\d+)", text, flags=re.IGNORECASE
    )
    if not match:
        return None
    value = int(match.group(2))
    return -value if match.group(1) == "-" else value


def _looks_like_candidate_line(line):
    return _candidate_payload(line) is not None


def _candidate_blocks(log_text):
    blocks = []
    current = []
    for raw_line in log_text.splitlines():
        payload = _candidate_payload(raw_line)
        if payload is not None:
            current.append((raw_line.strip(), payload))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _select_root_block(log_text, legal_moves=None):
    """Choose a root table and avoid the later opponent-move table."""
    blocks = _candidate_blocks(log_text)
    if not blocks:
        return []

    legal = {move.lower() for move in legal_moves or []}
    uci_blocks = [
        block for block in blocks if any("info string" in raw.lower() for raw, _ in block)
    ]
    candidates = uci_blocks or blocks

    if legal:
        scored = []
        for index, block in enumerate(candidates):
            matching = [
                item for item in block if item[1].split(None, 1)[0].lower() in legal
            ]
            scored.append((len(matching), index, matching, block))
        _, _, matching, selected = max(scored, key=lambda item: (item[0], item[1]))
        if matching:
            return matching
        return []

    return candidates[-1]


def parse_verbose_move_stats(log_text):
    """Parse all candidate rows in a text block."""
    candidates = []
    if not log_text:
        return candidates
    for raw_line in log_text.splitlines():
        payload = _candidate_payload(raw_line)
        if payload is None:
            continue
        move = payload.split(None, 1)[0].lower()
        candidates.append(
            RootCandidate(
                move_uci=move,
                P=_parse_percent_as_prob(payload, "P:"),
                N=_parse_int_after_label(payload, "N:"),
                Q=_parse_float_after_label(payload, "Q:"),
                V=_parse_float_after_label(payload, "V:"),
                U=_parse_float_after_label(payload, "U:"),
                WL=_parse_float_after_label(payload, "WL:"),
                D=_parse_float_after_label(payload, "D:"),
                M=_parse_float_after_label(payload, "M:"),
                S=_parse_float_after_label(payload, "S:"),
                n_delta=_parse_visit_delta(payload),
            )
        )
    return candidates


def extract_last_root_table(log_text, legal_moves=None):
    """Return the most recent root candidate table as normalized rows."""
    selected = _select_root_block(log_text, legal_moves=legal_moves)
    return "\n".join(payload for _, payload in selected)


def parse_last_root_candidates(log_text, legal_moves=None):
    """Select and parse the root table for the current position."""
    return parse_verbose_move_stats(
        extract_last_root_table(log_text, legal_moves=legal_moves)
    )
