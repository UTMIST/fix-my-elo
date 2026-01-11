"use client";

import { Chess } from "chess.js";
import { useState, useRef } from "react";
import { Chessboard } from "react-chessboard";

export default function ClickOrDragToMove() {
  // create a chess game using a ref so it persists across renders
  const chessGameRef = useRef(new Chess());
  const chessGame = chessGameRef.current;

  // state to trigger re-renders
  const [chessPosition, setChessPosition] = useState(chessGame.fen());
  const [moveFrom, setMoveFrom] = useState("");
  const [optionSquares, setOptionSquares] = useState({});

  // make a random "CPU" move
  function makeRandomMove() {
    const possibleMoves = chessGame.moves();

    if (chessGame.isGameOver()) return;

    const randomMove =
      possibleMoves[Math.floor(Math.random() * possibleMoves.length)];

    chessGame.move(randomMove);
    setChessPosition(chessGame.fen());
  }

  // get move options for a square
  function getMoveOptions(square) {
    const moves = chessGame.moves({
      square,
      verbose: true,
    });

    if (moves.length === 0) {
      setOptionSquares({});
      return false;
    }

    const newSquares = {};

    for (const move of moves) {
      newSquares[move.to] = {
        background:
          chessGame.get(move.to) &&
          chessGame.get(move.to)?.color !== chessGame.get(square)?.color
            ? "radial-gradient(circle, rgba(0,0,0,.1) 85%, transparent 85%)"
            : "radial-gradient(circle, rgba(0,0,0,.1) 25%, transparent 25%)",
        borderRadius: "50%",
      };
    }

    newSquares[square] = {
      background: "rgba(255, 255, 0, 0.4)",
    };

    setOptionSquares(newSquares);
    return true;
  }

  function onSquareClick({ square, piece }) {
    // selecting a piece
    if (!moveFrom && piece) {
      const hasMoveOptions = getMoveOptions(square);
      if (hasMoveOptions) setMoveFrom(square);
      return;
    }

    // attempting to move
    const moves = chessGame.moves({
      square: moveFrom,
      verbose: true,
    });

    const foundMove = moves.find(
      (m) => m.from === moveFrom && m.to === square
    );

    if (!foundMove) {
      const hasMoveOptions = getMoveOptions(square);
      setMoveFrom(hasMoveOptions ? square : "");
      return;
    }

    try {
      chessGame.move({
        from: moveFrom,
        to: square,
        promotion: "q",
      });
    } catch {
      const hasMoveOptions = getMoveOptions(square);
      if (hasMoveOptions) setMoveFrom(square);
      return;
    }

    setChessPosition(chessGame.fen());
    setTimeout(makeRandomMove, 300);
    setMoveFrom("");
    setOptionSquares({});
  }

  function onPieceDrop({ sourceSquare, targetSquare }) {
    if (!targetSquare) return false;

    try {
      chessGame.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q",
      });

      setChessPosition(chessGame.fen());
      setMoveFrom("");
      setOptionSquares({});
      setTimeout(makeRandomMove, 500);
      return true;
    } catch {
      return false;
    }
  }

  const chessboardOptions = {
    id: "click-or-drag-to-move",
    position: chessPosition,
    onSquareClick,
    onPieceDrop,
    squareStyles: optionSquares,
  };

  return <Chessboard options={chessboardOptions} />;
}
