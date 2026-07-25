# Lc0 Intuition Analysis Pipeline

This pipeline runs Leela Chess Zero (Lc0) on FEN, FEN-list, or PGN input and
records two different views of a position:

- the network policy, which estimates which legal moves initially look natural;
- the completed search, which supplies MultiPV rankings, evaluations, and
  principal variations.

Selected alternatives are then played on the board and searched again to find
concrete refutations.

## Requirements

- Python 3.9 or newer
- an Lc0 executable
- an Lc0 network weights file (`.pb.gz`)

From the project root, create an optional virtual environment and install the
pinned Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```

## Quick start

The following PowerShell example analyzes one position and prints one JSON
record to stdout. Replace the network filename if needed.

```powershell
python -m analysis.run_analysis `
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" `
  --lc0_path ".\engines\lc0_engine\lc0.exe" `
  --weights_path ".\engines\nets\t1-256x10-distilled-swa-2432500.pb.gz" `
  --movetime_ms 2000 `
  --multipv 5 `
  --top_intuitive 3
```

Analyze every second ply of every game in a PGN and write JSONL output:

```powershell
python -m analysis.run_analysis `
  --pgn ".\games\example.pgn" `
  --every_n_plies 2 `
  --include_start `
  --out_jsonl ".\results\analysis.jsonl" `
  --lc0_path ".\engines\lc0_engine\lc0.exe" `
  --weights_path ".\engines\nets\t1-256x10-distilled-swa-2432500.pb.gz" `
  --movetime_ms 1000 `
  --multipv 5 `
  --top_intuitive 3 `
  --lc0_logfile ".\logs\lc0_verbose.log"
```

A FEN list may be supplied with `--fen_list`; blank lines and lines beginning
with `#` are ignored. `--out_jsonl` replaces an existing output file. Without
that option, JSONL records are written to stdout.

All filesystem arguments expand environment variables and `~`, then resolve to
absolute paths before use. Relative paths are resolved from the directory where
the command is launched. The Lc0 executable may also be specified by a command
available on `PATH`. Lc0 itself starts in the executable's directory, but it is
given the already-resolved absolute network path.

## How a position is analyzed

Each input position receives exactly one root search. That single search
collects the final MultiPV lines and, when Lc0 provides them, the verbose root
move statistics. This keeps the policy table and search result tied to the same
search.

Candidate selection then works as follows:

1. If root candidates contain policy values (`P`), the highest-policy moves are
   selected. `metadata.selection_method` is `lc0_policy`, and these moves also
   appear in `intuitive_moves`.
2. If policy data is unavailable, the leading MultiPV moves are used as a
   search-ranked fallback. `metadata.selection_method` is
   `search_ranked_multipv`; `intuitive_moves` remains empty because search rank
   is not evidence of network intuition.
3. Every selected candidate other than the root best move receives one child
   search with MultiPV 1. Its first line is the candidate's refutation line.

The total is therefore one root search plus `metadata.refutation_searches`
child searches. `metadata.root_searches` is always `1` for a successful record.
The time, node, or depth limit applies separately to every search.

Policy and MultiPV should not be treated as interchangeable:

- `analysis.root_candidates[*].P` is the normalized policy prior (for example,
  `0.25` means 25%). The candidate table can also retain Lc0's `N`, `Q`, `V`,
  `U`, `WL`, `D`, `M`, `S`, and `n_delta` statistics when emitted by the engine.
- `analysis.pv_lines[*].multipv_index` is the ranking after search. Each line
  includes its UCI PV, score or mate value, and available search measurements
  such as depth, nodes, NPS, time, hash usage, tablebase hits, and WDL.

Verbose policy capture is enabled by default. Use `--no-verbose_move_stats` to
disable it. `analysis.search_metadata.policy_source` reports
`uci_info_string`, `logfile`, `unavailable`, or `not_requested`.

## Score perspective

Scores in an `AnalysisResult` always belong to that result's `side_to_move`;
the field `score_perspective` is `side_to_move`. A positive centipawn score is
favorable to that side. Positive mate values mean that side can force mate;
negative values mean that side is being mated.

A child search starts after a candidate move, so its side to move is the
opponent. Consequently, scores inside
`candidate_evaluations[*].analysis.pv_lines` use the opponent's perspective.
The surrounding `eval_before` and `eval_after` values are both converted to the
original root player's perspective and are marked
`score_perspective: root_side_to_move`. `drop_cp` is
`eval_before - eval_after` when both scores are centipawn scores.

Mate results are classified semantically (`lost_forced_mate` or
`allows_forced_mate`) and are not converted into arbitrary centipawn values.
`intuitive_but_bad` only contains policy-selected moves for which the child
search provides evidence meeting the configured threshold or mate rule.

## Schema version 2

Every JSONL line is a complete record. A successful record has this shape:

```text
schema_version: 2
record_type: "intuition_analysis"
status: "ok"
generated_at_utc
fen, best_move, best_pv
candidate_moves, intuitive_moves, intuitive_but_bad
candidate_evaluations
analysis
source
engine
metadata
```

Important nested data includes:

- `analysis`: the root `AnalysisResult`, including `pv_lines`, the full parsed
  `root_candidates` table, optional `raw_info`, and `search_metadata`;
- `candidate_evaluations`: the root-perspective before/after score,
  classification, refutation PV, and complete child `AnalysisResult` for each
  searched alternative;
- `engine`: UCI name and author, resolved executable and network paths, SHA-256
  hashes, and configured options;
- `source`: reproducible input provenance;
- `metadata`: limits, selection method, policy availability, thresholds, and
  root/refutation search counts.

For PGN input, `source` retains the absolute PGN path, global position index,
one-based game index, all PGN headers (including `Result` when present), ply
number, the move that led to the sampled position in UCI and SAN, and the next
played move in UCI and SAN. Starting positions use ply 0 and have no last move.
FEN-list records include their absolute source path and original line number.

Failures are also valid schema-v2 JSONL records. They use
`record_type: "analysis_error"`, `status: "error"`, and an `error_details`
object containing the error code, stage, message, exception type, and an
optional traceback when `--debug` is enabled. This lets batch consumers handle
successful and failed positions without parsing stderr.

## Diagnostic logs and raw data

`--lc0_logfile` creates the diagnostic log if needed and appends to it; the
pipeline never clears or overwrites it. Before each root or child search, it
records the current byte offset and parses only newly appended lines, so older
candidate tables are not mistaken for the current position. Timestamped and
UCI-prefixed verbose candidate rows are supported.

Use `--include_raw_info` to add the current root search's captured UCI/log lines
to `analysis.raw_info`. Raw child-search lines are left in the append-only log
but are not duplicated into each candidate evaluation.

## Exit codes

- `0`: every requested position produced a successful analysis record;
- `1`: at least one position or input record failed; error records were emitted;
- `2`: command-line, dependency, configuration, filesystem setup, output, or
  engine-start failure.

In PowerShell, inspect `$LASTEXITCODE` immediately after the command. When an
engine-start failure occurs after the output stream is available, the pipeline
also emits an `analysis_error` record before returning `2`.
