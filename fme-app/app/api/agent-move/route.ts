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
    const defaultVenvPython = "/venv/bin/python";
    const pythonExec = process.env.TEAM2_PYTHON
      ? process.env.TEAM2_PYTHON
      : fs.existsSync(defaultVenvPython)
        ? defaultVenvPython
        : "python3";

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

    // Log which Python executable we'll use and perform a preflight check
    console.error(`[AGENT-MOVE] Using python executable: ${pythonExec}`);

    const preflight = await runPython(pythonExec, [
      "-c",
      // Avoid importlib.util (some Python envs can shadow importlib); try importing chess directly
      "import sys,json\ntry:\n import chess\n print(json.dumps({'exe': sys.executable, 'has_chess': True}))\nexcept Exception as e:\n print(json.dumps({'exe': sys.executable, 'has_chess': False, 'error': str(e)}))\n",
    ], team2Dir);

    console.error(`[AGENT-MOVE] Preflight stdout: ${preflight.stdout}`);
    console.error(`[AGENT-MOVE] Preflight stderr: ${preflight.stderr}`);

    if (preflight.exitCode !== 0) {
      return NextResponse.json({ error: 'Python preflight failed', details: preflight.stderr || preflight.stdout }, { status: 500 });
    }

    try {
      const parsedPre = preflight.parsed as Record<string, unknown> | null;
      if (!parsedPre || parsedPre['has_chess'] !== true) {
        return NextResponse.json({ error: 'Python environment missing python-chess', details: preflight.stdout || preflight.stderr }, { status: 500 });
      }
    } catch {
      // continue to attempt running the engine
    }

    const timeoutMs = Number.isFinite(Number(process.env.AGENT_MOVE_TIMEOUT_MS))
      ? Math.max(1000, Number(process.env.AGENT_MOVE_TIMEOUT_MS))
      : 15000;

    const engineUrl = process.env.TEAM2_ENGINE_URL;
    if (engineUrl) {
      const engineResponse = await callEngineServer(
        engineUrl,
        {
          fen,
          numSimulations,
          temperature,
        },
        timeoutMs,
      );

      if (!engineResponse.ok) {
        return NextResponse.json(engineResponse.body, { status: engineResponse.status });
      }

      return NextResponse.json(engineResponse.body);
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
      timeoutMs,
    );

    if (result.timedOut) {
      return NextResponse.json(
        {
          error: "Engine move timed out",
          details: `Engine exceeded ${timeoutMs}ms`,
        },
        { status: 504 },
      );
    }

    if (result.exitCode !== 0) {
      // Provide detailed error information collected from the engine process
      console.error(`[AGENT-MOVE] Engine failed. exit=${result.exitCode}`);
      console.error(`[AGENT-MOVE] Engine stdout: ${result.stdout}`);
      console.error(`[AGENT-MOVE] Engine stderr: ${result.stderr}`);

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

function runPython(command: string, args: string[], cwd: string, timeoutMs?: number) {
  return new Promise<{
    exitCode: number | null;
    stdout: string;
    stderr: string;
    parsed: Record<string, unknown> | null;
    timedOut: boolean;
  }>((resolve) => {
    const child = spawn(command, args, { cwd });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let timeoutId: NodeJS.Timeout | null = null;

    if (timeoutMs && timeoutMs > 0) {
      timeoutId = setTimeout(() => {
        timedOut = true;
        child.kill("SIGKILL");
      }, timeoutMs);
    }

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("close", (code) => {
      if (timeoutId) clearTimeout(timeoutId);
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
        timedOut,
      });
    });
  });
}

async function callEngineServer(
  url: string,
  payload: { fen: string; numSimulations: number; temperature: number },
  timeoutMs: number,
): Promise<{ ok: boolean; status: number; body: Record<string, unknown> }>
{
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const raw = await response.text();
    const body = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};

    return { ok: response.ok, status: response.status, body };
  } catch (error) {
    return {
      ok: false,
      status: 504,
      body: {
        error: "Engine server request failed",
        details: String(error),
      },
    };
  } finally {
    clearTimeout(timer);
  }
}
