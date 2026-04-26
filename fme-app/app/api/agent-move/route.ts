import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { Chess } from "chess.js";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type AgentMoveRequest = {
  fen?: string;
  numSimulations?: number;
  temperature?: number;
};

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as AgentMoveRequest;
    const fen = body.fen?.trim();

    if (!fen) {
      return NextResponse.json({ error: "fen is required" }, { status: 400 });
    }

    try {
      new Chess(fen);
    } catch {
      return NextResponse.json({ error: "Invalid FEN" }, { status: 400 });
    }

    const numSimulations = Number.isFinite(body.numSimulations)
      ? Math.max(1, Math.floor(body.numSimulations as number))
      : 120;

    const temperature = Number.isFinite(body.temperature)
      ? Number(body.temperature)
      : 0.0;

    const repoRoot = path.resolve(process.cwd(), "..");
    const team2Dir = path.join(repoRoot, "Team2");
    const scriptPath = path.join(team2Dir, "engine_move_cli.py");
    const pythonExec = process.env.TEAM2_PYTHON ?? "python3";

    const candidatePaths = [
      process.env.TEAM2_MODEL_PATH,
      path.join(team2Dir, "softmax_stockfish_trained.pth"),
      path.join(team2Dir, "checkpoint3.pth"),
      path.join(team2Dir, "model_files", "sl_policy_network.pth"),
    ].filter(Boolean) as string[];

    const modelPath = candidatePaths.find((candidate) => fs.existsSync(candidate));
    if (!modelPath) {
      return NextResponse.json(
        {
          error: "No Team2 model checkpoint found",
          details: "Set TEAM2_MODEL_PATH to a valid .pth checkpoint path.",
        },
        { status: 500 },
      );
    }

    const result = await runPython(
      pythonExec,
      [
        scriptPath,
        "--fen",
        fen,
        "--model-path",
        modelPath,
        "--num-simulations",
        String(numSimulations),
        "--temperature",
        String(temperature),
      ],
      team2Dir,
    );

    if (result.exitCode !== 0) {
      return NextResponse.json(
        {
          error: "Engine process failed",
          details: result.parsed ?? (result.stderr || result.stdout),
        },
        { status: 500 },
      );
    }

    if (!result.parsed || !result.parsed.move) {
      return NextResponse.json({ error: "Engine returned invalid response" }, { status: 500 });
    }

    return NextResponse.json(result.parsed);
  } catch (error) {
    return NextResponse.json({ error: `Unexpected error: ${String(error)}` }, { status: 500 });
  }
}

function runPython(command: string, args: string[], cwd: string) {
  return new Promise<{
    exitCode: number | null;
    stdout: string;
    stderr: string;
    parsed: Record<string, unknown> | null;
  }>((resolve) => {
    const child = spawn(command, args, { cwd });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("close", (code) => {
      let parsed: Record<string, unknown> | null = null;
      const trimmed = stdout.trim();

      if (trimmed) {
        try {
          parsed = JSON.parse(trimmed) as Record<string, unknown>;
        } catch {
          parsed = null;
        }
      }

      resolve({
        exitCode: code,
        stdout,
        stderr,
        parsed,
      });
    });
  });
}
