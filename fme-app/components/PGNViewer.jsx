"use client";

import React, { useState, useEffect, useRef } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

export default function PGNViewer() {
  const [fen, setFen] = useState(new Chess().fen());
  const [games, setGames] = useState([]);
  const [currentGameIndex, setCurrentGameIndex] = useState(0);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [boardOrientation, setBoardOrientation] = useState("white");
  const [pgnInput, setPgnInput] = useState("");
  const [error, setError] = useState("");

  // Auto-play state
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [autoPlaySpeed, setAutoPlaySpeed] = useState(1000);
  const autoPlayIntervalRef = useRef(null);
  const moveListRef = useRef(null);

  /**
   * Parse PGN using chess.js built-in parser - MUCH MORE RELIABLE!
   */
  const parsePGN = (text) => {
    // Split by [Event to get individual games
    const gameStrings = text.split(/(?=\[Event)/g).filter((s) => s.trim());
    const parsedGames = [];

    for (const gameStr of gameStrings) {
      try {
        // Use chess.js to parse the entire PGN
        const chess = new Chess();

        // loadPgn returns true on success, false on failure
        const loaded = chess.loadPgn(gameStr);

        if (loaded === false) {
          console.warn("Failed to load PGN:", gameStr.substring(0, 100));
          continue;
        }

        // Get the move history in SAN notation
        const history = chess.history();

        if (history.length === 0) {
          continue;
        }

        // Extract headers manually
        const headers = {};
        const headerRegex = /\[(\w+)\s+"([^"]*)"\]/g;
        let match;
        while ((match = headerRegex.exec(gameStr)) !== null) {
          headers[match[1]] = match[2];
        }

        parsedGames.push({
          headers,
          moves: history, // These are already in SAN format!
        });
      } catch (e) {
        console.error("Error parsing game:", e);
      }
    }

    return parsedGames;
  };

  const loadPGN = (text) => {
    try {
      setError("");

      if (!text || text.trim().length === 0) {
        setError("Please enter a valid PGN");
        return;
      }

      const parsedGames = parsePGN(text);

      if (parsedGames.length === 0) {
        setError("No valid games found in PGN");
        return;
      }

      console.log("Loaded games:", parsedGames.length);
      console.log("First game moves:", parsedGames[0].moves.length);

      setGames(parsedGames);
      setCurrentGameIndex(0);
      setCurrentMoveIndex(-1);
    } catch (err) {
      console.error("Error in loadPGN:", err);
      setError("Error parsing PGN: " + err.message);
    }
  };

  // Initialize with sample PGN
  useEffect(() => {
    const samplePGN = `[Event "Tata Steel India Rapid"]
[Site "Kolkata IND"]
[Date "2026.01.07"]
[Round "1.1"]
[White "Anand, Viswanathan"]
[Black "So, Wesley"]
[Result "1-0"]

1. e4 c6 2. d4 d5 3. e5 Bf5 4. Nd2 e6 5. Nb3 Nd7 6. Nf3 a6 7. Be2 c5 8. dxc5
Bxc5 9. Nxc5 Nxc5 10. O-O Ne7 11. Nd4 Be4 12. Be3 Qc7 13. f4 Nf5 14. Nxf5 exf5
15. c3 Rc8 16. a4 O-O 17. a5 Qc6 18. Rf2 h6 19. Bf1 Ne6 20. Ra4 Qc7 21. Bb6 Qe7
22. Rb4 g5 23. fxg5 hxg5 24. g3 Rc6 25. Bg2 Ng7 26. Bd4 Rfc8 27. Be3 Rg6 28. Bd4
f4 29. Qe2 Bxg2 30. Rxg2 Ne6 31. Bb6 Kg7 32. Qd3 d4 33. Bxd4 g4 34. gxf4 Nxf4
35. Qf5 Nh3+ 36. Kf1 Rg8 37. e6+ Kh6 38. Be3+ Ng5 39. h4 Kg7 40. hxg5 1-0`;

    setPgnInput(samplePGN);
    loadPGN(samplePGN);
  }, []);

  // Update game position based on current move
  useEffect(() => {
    if (games.length === 0) return;

    const currentGame = games[currentGameIndex];
    if (!currentGame || !currentGame.moves) return;

    // Create fresh game
    const newGame = new Chess();

    console.log("Updating to move index:", currentMoveIndex);
    console.log("Total moves:", currentGame.moves.length);

    // Apply all moves up to currentMoveIndex
    for (let i = 0; i <= currentMoveIndex; i++) {
      if (i >= 0 && i < currentGame.moves.length) {
        const moveToMake = currentGame.moves[i];

        try {
          const result = newGame.move(moveToMake);
          if (!result) {
            console.error(`Failed to make move: ${moveToMake}`);
          }
        } catch (e) {
          console.error(`Error making move ${moveToMake}:`, e);
        }
      }
    }

    const finalFen = newGame.fen();
    console.log("Final FEN:", finalFen);
    setFen(finalFen);

    // Auto-scroll logic
    const activeMoveElement =
      moveListRef.current?.querySelector(".active-move");
    if (activeMoveElement) {
      activeMoveElement.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [currentMoveIndex, currentGameIndex, games]);

  // Autoplay logic
  useEffect(() => {
    if (isAutoPlaying) {
      autoPlayIntervalRef.current = setInterval(() => {
        setCurrentMoveIndex((prev) => {
          const maxMoves = games[currentGameIndex]?.moves.length - 1;
          if (prev < maxMoves) return prev + 1;
          setIsAutoPlaying(false);
          return prev;
        });
      }, autoPlaySpeed);
    } else {
      clearInterval(autoPlayIntervalRef.current);
    }
    return () => clearInterval(autoPlayIntervalRef.current);
  }, [isAutoPlaying, autoPlaySpeed, currentGameIndex, games]);

  // Navigation
  const goToMove = (index) => {
    const maxIndex = (games[currentGameIndex]?.moves.length || 0) - 1;

    console.log("goToMove called with:", index, "Max:", maxIndex);

    if (index < -1) {
      setCurrentMoveIndex(-1);
    } else if (index > maxIndex) {
      setCurrentMoveIndex(maxIndex);
    } else {
      setCurrentMoveIndex(index);
    }
  };

  const currentGame = games[currentGameIndex] || { headers: {}, moves: [] };

  // Debug log for rendering
  useEffect(() => {
    console.log("Rendering Chessboard with FEN:", fen);
    console.log("Board orientation:", boardOrientation);
  }, [fen, boardOrientation]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-white mb-8 text-center">
          PGN Chess Viewer
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Load PGN */}
          <div className="bg-slate-800 rounded-lg p-6 shadow-xl border border-slate-700 h-fit">
            <h2 className="text-2xl font-semibold text-white mb-4">Load PGN</h2>
            <textarea
              value={pgnInput}
              onChange={(e) => setPgnInput(e.target.value)}
              className="w-full h-48 bg-slate-900 text-white rounded p-3 border border-slate-600 font-mono text-sm resize-none"
            />
            <button
              onClick={() => loadPGN(pgnInput)}
              className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded transition-colors"
            >
              Load PGN
            </button>

            {error && (
              <div className="mt-4 bg-red-900/50 border border-red-500 text-red-200 rounded p-3 text-sm">
                {error}
              </div>
            )}

            {Object.keys(currentGame.headers).length > 0 && (
              <div className="mt-6 bg-slate-900 rounded p-4 border border-slate-600">
                <h3 className="text-lg font-semibold text-white mb-2">
                  Game Info
                </h3>
                <div className="space-y-1 text-sm max-h-64 overflow-y-auto">
                  {Object.entries(currentGame.headers).map(([k, v]) => (
                    <div key={k} className="text-slate-300">
                      <span className="font-semibold text-blue-400">{k}:</span>{" "}
                      {v}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Center: Board */}
          <div className="bg-slate-800 rounded-lg p-6 shadow-xl border border-slate-700">
            <div className="w-full aspect-square max-w-[500px] mx-auto">
              <Chessboard
                key={`${currentGameIndex}-${currentMoveIndex}`}
                position={fen}
                boardOrientation={boardOrientation}
                boardWidth={500}
                arePiecesDraggable={false}
                animationDuration={200}
                customBoardStyle={{
                  borderRadius: '4px',
                  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
                }}
              />
            </div>

            <div className="flex justify-center gap-2 mt-6">
              <button
                onClick={() => {
                  setIsAutoPlaying(false);
                  goToMove(-1);
                }}
                className="bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 px-4 rounded"
              >
                ⏮
              </button>
              <button
                onClick={() => {
                  setIsAutoPlaying(false);
                  goToMove(currentMoveIndex - 1);
                }}
                className="bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 px-4 rounded"
              >
                ◀
              </button>
              <button
                onClick={() => setIsAutoPlaying(!isAutoPlaying)}
                className={`px-6 py-2 rounded font-bold transition-colors ${
                  isAutoPlaying
                    ? "bg-red-600 hover:bg-red-700"
                    : "bg-green-600 hover:bg-green-700"
                } text-white`}
              >
                {isAutoPlaying ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => {
                  setIsAutoPlaying(false);
                  goToMove(currentMoveIndex + 1);
                }}
                className="bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 px-4 rounded"
              >
                ▶
              </button>
              <button
                onClick={() => {
                  setIsAutoPlaying(false);
                  goToMove(currentGame.moves.length - 1);
                }}
                className="bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 px-4 rounded"
              >
                ⏭
              </button>
            </div>

            <div className="mt-6 space-y-4">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Playback Speed</span>
                  <span>{(autoPlaySpeed / 1000).toFixed(1)}s</span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="2000"
                  step="100"
                  value={autoPlaySpeed}
                  onChange={(e) => setAutoPlaySpeed(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
              <button
                onClick={() =>
                  setBoardOrientation((o) =>
                    o === "white" ? "black" : "white"
                  )
                }
                className="w-full bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 px-4 rounded transition-colors"
              >
                🔄 Flip Board
              </button>
            </div>
          </div>

          {/* Right: Moves */}
          <div className="bg-slate-800 rounded-lg p-6 shadow-xl border border-slate-700 h-[700px] flex flex-col">
            <h2 className="text-2xl font-semibold text-white mb-4">
              Moves History
            </h2>
            <div
              ref={moveListRef}
              className="bg-slate-900 rounded p-4 overflow-y-auto border border-slate-600 flex-1"
            >
              <div className="grid grid-cols-2 gap-2">
                {currentGame.moves.map((move, i) => (
                  <div
                    key={i}
                    onClick={() => {
                      setIsAutoPlaying(false);
                      goToMove(i);
                    }}
                    className={`cursor-pointer p-2 rounded text-sm font-mono flex gap-2 transition-colors ${
                      i === currentMoveIndex
                        ? "bg-blue-600 text-white active-move"
                        : "text-slate-300 hover:bg-slate-800"
                    }`}
                  >
                    <span className="text-slate-500 w-6">
                      {i % 2 === 0 ? `${Math.floor(i / 2) + 1}.` : ""}
                    </span>
                    <span>{move}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}