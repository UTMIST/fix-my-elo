"use client";

import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { useEffect, useState } from "react";

export default function ChessBoard({
  moves,
  currentMoveIndex,
  boardOrientation,
}) {
  const [fen, setFen] = useState(new Chess().fen());

  useEffect(() => {
    console.log("[ChessBoard] Rebuilding board");
    console.log("[ChessBoard] Move index:", currentMoveIndex);
    console.log("[ChessBoard] Total moves:", moves?.length);

    const game = new Chess();

    for (let i = 0; i <= currentMoveIndex; i++) {
      if (i >= 0 && i < moves.length) {
        try {
          game.move(moves[i]);
        } catch (e) {
          console.error("[ChessBoard] Failed move:", moves[i], e);
        }
      }
    }

    const newFen = game.fen();
    console.log("[ChessBoard] New FEN:", newFen);
    setFen(newFen);
  }, [moves, currentMoveIndex]);

  return (
    <Chessboard
      position={fen}
      boardOrientation={boardOrientation}
      boardWidth={500}
      arePiecesDraggable={false}
      animationDuration={200}
      customBoardStyle={{
        borderRadius: "4px",
        boxShadow: "0 2px 10px rgba(0,0,0,0.5)",
      }}
    />
  );
}
