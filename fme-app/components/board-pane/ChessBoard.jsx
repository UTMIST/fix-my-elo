"use client";

import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { useEffect, useRef, useState } from "react";

export default function ChessBoard({
  moves,
  currentMoveIndex,
  boardOrientation,
}) {
  const chessGameRef = useRef(new Chess());
  const [chessPosition, setChessPosition] = useState(chessGameRef.current.fen());

  useEffect(() => {
    console.log("[ChessBoard] Rebuilding board");
    console.log("[ChessBoard] Move index:", currentMoveIndex);
    console.log("[ChessBoard] Total moves:", moves?.length);

    // Create a fresh game instance
    const game = new Chess();

    // Apply all moves up to currentMoveIndex
    for (let i = 0; i <= currentMoveIndex; i++) {
      if (i >= 0 && i < moves.length) {
        try {
          const result = game.move(moves[i]);
          if (!result) {
            console.error("[ChessBoard] Failed to make move:", moves[i]);
          }
        } catch (e) {
          console.error("[ChessBoard] Failed move:", moves[i], e);
        }
      }
    }

    const newFen = game.fen();
    console.log("[ChessBoard] New FEN:", newFen);
    
    // Update the ref
    chessGameRef.current = game;
    // Update state to trigger re-render
    setChessPosition(newFen);
  }, [moves, currentMoveIndex]);

  const chessboardOptions = {
    id: "pgn-viewer-board",
    position: chessPosition,
    boardOrientation: boardOrientation,
    arePiecesDraggable: false,
    animationDuration: 200,
    customBoardStyle: {
      borderRadius: "4px",
      boxShadow: "0 2px 10px rgba(0,0,0,0.5)",
    },
  };

  return <Chessboard options={chessboardOptions} />;
}