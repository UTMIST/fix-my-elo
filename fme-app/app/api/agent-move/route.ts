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

    const timeoutMs = Number.isFinite(Number(process.env.AGENT_MOVE_TIMEOUT_MS))
      ? Math.max(100000, Number(process.env.AGENT_MOVE_TIMEOUT_MS))
      : 150000;

    const engineUrl = process.env.TEAM2_ENGINE_URL ?? process.env.TEAM2_INFERENCE_URL;
    if (!engineUrl) {
      return NextResponse.json(
        {
          error: "TEAM2_ENGINE_URL is not configured",
          details: "Set TEAM2_ENGINE_URL to your Modal /move endpoint.",
        },
        { status: 500 },
      );
    }

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
  } catch (error) {
    return NextResponse.json({ error: `Unexpected error: ${String(error)}` }, { status: 500 });
  }
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
    let body: Record<string, unknown> = {};

    if (raw) {
      try {
        body = JSON.parse(raw) as Record<string, unknown>;
      } catch {
        body = {
          error: "Engine server returned non-JSON response",
          details: raw,
        };
      }
    }

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
