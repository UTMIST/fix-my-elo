"""Build policy-versus-search reports from complete Lc0 results."""

from __future__ import annotations

from datetime import datetime, timezone

from analysis.schema import IntuitionReport


def _score_from_line(line, invert=False):
    if line is None:
        return None
    if line.mate is not None:
        return {"type": "mate", "value": -line.mate if invert else line.mate}
    if line.score_cp is not None:
        return {"type": "cp", "value": -line.score_cp if invert else line.score_cp}
    return None


def _format_score(score):
    if score is None:
        return "unknown"
    if score["type"] == "cp":
        return f"{score['value'] / 100:+.2f}"
    value = score["value"]
    if value > 0:
        return f"mate in {value}"
    if value < 0:
        return f"mated in {abs(value)}"
    return "checkmate"


def _compare_scores(before, after, threshold_cp):
    """Classify a worsening without pretending mate scores are centipawns."""
    if before is None or after is None:
        return False, None, "score_unavailable"
    if before["type"] == "cp" and after["type"] == "cp":
        drop_cp = before["value"] - after["value"]
        return drop_cp >= threshold_cp, drop_cp, "centipawn_drop"

    before_wins_by_mate = before["type"] == "mate" and before["value"] > 0
    after_wins_by_mate = after["type"] == "mate" and after["value"] > 0
    before_loses_by_mate = before["type"] == "mate" and before["value"] <= 0
    after_loses_by_mate = after["type"] == "mate" and after["value"] <= 0
    if before_wins_by_mate and not after_wins_by_mate:
        return True, None, "lost_forced_mate"
    if after_loses_by_mate and not before_loses_by_mate:
        return True, None, "allows_forced_mate"
    return False, None, "mate_score_not_cp_comparable"


def build_intuition_report(engine, fen, config, source=None):
    """Analyze one root position and selected alternatives."""
    movetime_ms = config.get("movetime_ms")
    nodes = config.get("nodes")
    depth = config.get("depth")
    multipv = config.get("multipv", 5)
    top_intuitive = config.get("top_intuitive", 3)
    bad_threshold_cp = config.get("bad_threshold_cp", 80)
    verbose_move_stats = config.get("verbose_move_stats", True)
    include_raw_info = config.get("include_raw_info", False)

    analysis_result = engine.analyze_fen(
        fen,
        movetime_ms=movetime_ms,
        nodes=nodes,
        depth=depth,
        multipv=multipv,
        capture_root_stats=verbose_move_stats,
        include_raw_info=include_raw_info,
    )
    best_move = analysis_result.bestmove_uci
    best_line = analysis_result.pv_lines[0] if analysis_result.pv_lines else None
    best_pv = [] if best_line is None else list(best_line.pv_uci_list)

    policy_candidates = [
        candidate
        for candidate in analysis_result.root_candidates or []
        if candidate.P is not None
    ]
    if policy_candidates:
        policy_candidates.sort(key=lambda candidate: candidate.P, reverse=True)
        candidate_moves = [
            candidate.move_uci for candidate in policy_candidates[:top_intuitive]
        ]
        intuitive_moves = list(candidate_moves)
        selection_method = "lc0_policy"
    else:
        candidate_moves = [
            line.move_uci
            for line in sorted(
                analysis_result.pv_lines, key=lambda line: line.multipv_index
            )[:top_intuitive]
        ]
        intuitive_moves = []
        selection_method = "search_ranked_multipv"

    candidate_moves = list(dict.fromkeys(candidate_moves))
    best_score = _score_from_line(best_line)
    candidate_evaluations = []
    intuitive_but_bad = []
    refutation_searches = 0

    for move_uci in candidate_moves:
        if move_uci == best_move:
            continue
        child_result = engine.analyze_after_move(
            fen,
            move_uci,
            movetime_ms=movetime_ms,
            nodes=nodes,
            depth=depth,
            multipv=1,
            capture_root_stats=False,
            include_raw_info=False,
        )
        refutation_searches += 1
        child_line = child_result.pv_lines[0] if child_result.pv_lines else None
        child_score = _score_from_line(child_line, invert=True)
        is_bad, drop_cp, reason = _compare_scores(
            best_score, child_score, bad_threshold_cp
        )
        refutation_pv = [] if child_line is None else list(child_line.pv_uci_list)
        evidence = (
            f"After {move_uci}, the evaluation changes from "
            f"{_format_score(best_score)} to {_format_score(child_score)}"
        )
        if refutation_pv:
            evidence += f"; the reply starts {refutation_pv[0]}"
        evidence += "."

        evaluation = {
            "move_uci": move_uci,
            "score_perspective": "root_side_to_move",
            "eval_before": best_score,
            "eval_after": child_score,
            "drop_cp": drop_cp,
            "is_bad": is_bad,
            "classification_reason": reason,
            "refutation_pv": refutation_pv,
            "evidence": evidence,
            "analysis": child_result.to_dict(),
        }
        candidate_evaluations.append(evaluation)
        if is_bad and selection_method == "lc0_policy":
            intuitive_but_bad.append(
                {key: value for key, value in evaluation.items() if key != "analysis"}
            )

    metadata = {
        "multipv": multipv,
        "top_intuitive": top_intuitive,
        "bad_threshold_cp": bad_threshold_cp,
        "movetime_ms": movetime_ms,
        "nodes": nodes,
        "depth": depth,
        "selection_method": selection_method,
        "policy_available": bool(policy_candidates),
        "policy_candidate_count": len(policy_candidates),
        "root_searches": 1,
        "refutation_searches": refutation_searches,
        "score_perspective": "root_side_to_move",
    }
    identity = getattr(engine, "identity", None)

    return IntuitionReport(
        fen=fen,
        best_move=best_move,
        best_pv=best_pv,
        candidate_moves=candidate_moves,
        intuitive_moves=intuitive_moves,
        intuitive_but_bad=intuitive_but_bad,
        candidate_evaluations=candidate_evaluations,
        analysis=analysis_result,
        source=source,
        engine=identity,
        metadata=metadata,
        generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
