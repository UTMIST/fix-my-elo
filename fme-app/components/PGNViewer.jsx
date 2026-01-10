"use client";

import React, { useState, useEffect, useRef } from "react";
import { Chess } from "chess.js";
import ChessBoard from "./ChessBoard";
import "./PGNViewer.css";

export default function PGNViewer() {
  const [games, setGames] = useState([]);
  const [currentGameIndex, setCurrentGameIndex] = useState(0);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [boardOrientation, setBoardOrientation] = useState("white");
  const [pgnInput, setPgnInput] = useState("");
  const [error, setError] = useState("");
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

  useEffect(() => {
    const activeMoveElement = moveListRef.current?.querySelector(".active-move");
    if (activeMoveElement) {
      activeMoveElement.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [currentMoveIndex]);

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

  const goToMove = (index) => {
    const maxIndex = (games[currentGameIndex]?.moves.length || 0) - 1;
    if (index < -1) {
      setCurrentMoveIndex(-1);
    } else if (index > maxIndex) {
      setCurrentMoveIndex(maxIndex);
    } else {
      setCurrentMoveIndex(index);
    }
  };

  const currentGame = games[currentGameIndex] || { headers: {}, moves: [] };

  return (
    <div className="pgn-viewer">
      <div className="pgn-container">
        <h1 className="pgn-title">Fix-My-Elo</h1>

        <div className="pgn-grid">
          {/* Left: Load PGN */}
          <div className="pgn-panel">
            <h2 className="panel-title">Load PGN</h2>
            <textarea
              value={pgnInput}
              onChange={(e) => setPgnInput(e.target.value)}
              className="pgn-textarea"
            />
            <button onClick={() => loadPGN(pgnInput)} className="btn-load">
              Load PGN
            </button>

            {error && <div className="error-message">{error}</div>}

            {Object.keys(currentGame.headers).length > 0 && (
              <div className="game-info">
                <h3 className="info-title">Game Info</h3>
                <div className="info-content">
                  {Object.entries(currentGame.headers).map(([k, v]) => (
                    <div key={k} className="info-row">
                      <span className="info-key">{k}:</span> {v}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Center: Board */}
          <div className="pgn-panel board-panel">
            <div className="board-container">
              <ChessBoard
                moves={currentGame.moves}
                currentMoveIndex={currentMoveIndex}
                boardOrientation={boardOrientation}
              />
            </div>

            <div className="controls">
              <button onClick={() => { setIsAutoPlaying(false); goToMove(-1); }} className="btn-nav">
                ⏮
              </button>
              <button onClick={() => { setIsAutoPlaying(false); goToMove(currentMoveIndex - 1); }} className="btn-nav">
                ◀
              </button>
              <button onClick={() => setIsAutoPlaying(!isAutoPlaying)} className={`btn-play ${isAutoPlaying ? "playing" : ""}`}>
                {isAutoPlaying ? "Pause" : "Play"}
              </button>
              <button onClick={() => { setIsAutoPlaying(false); goToMove(currentMoveIndex + 1); }} className="btn-nav">
                ▶
              </button>
              <button onClick={() => { setIsAutoPlaying(false); goToMove(currentGame.moves.length - 1); }} className="btn-nav">
                ⏭
              </button>
            </div>

            <div className="board-options">
              <button onClick={() => setBoardOrientation((o) => o === "white" ? "black" : "white")} className="btn-flip">
                Flip Board
              </button>
            </div>
          </div>

          {/* Right: Moves */}
          <div className="pgn-panel moves-panel">
            <h2 className="panel-title">Moves History</h2>
            <div ref={moveListRef} className="moves-list">
              <div className="moves-grid">
                {currentGame.moves.map((move, i) => (
                  <div
                    key={i}
                    onClick={() => { setIsAutoPlaying(false); goToMove(i); }}
                    className={`move-item ${i === currentMoveIndex ? "active-move" : ""}`}
                  >
                    <span className="move-number">
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