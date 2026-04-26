"use client";

import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { useMemo, useState } from "react";

export default function ChessBoard({
  moves,
  currentMoveIndex,
  boardOrientation,
  interactive = false,
  onMovePlayed,
  canUserMove,
}) {
  const [moveFrom, setMoveFrom] = useState('');
  const [optionSquares, setOptionSquares] = useState({});

  const game = useMemo(() => {
    const game = new Chess();

    for (let i = 0; i <= currentMoveIndex; i++) {
      if (i >= 0 && i < moves.length) {
        try {
          const result = game.move(moves[i]);
          if (!result) {
            console.error('[ChessBoard] Failed to make move:', moves[i]);
          }
        } catch (e) {
          console.error('[ChessBoard] Failed move:', moves[i], e);
        }
      }
    }

    return game;
  }, [moves, currentMoveIndex]);

  const getMoveOptions = (square) => {
    const movesFromSquare = game.moves({ square, verbose: true });

    if (!movesFromSquare.length) {
      setOptionSquares({});
      return false;
    }

    const newSquares = {};
    for (const move of movesFromSquare) {
      newSquares[move.to] = {
        background:
          game.get(move.to) && game.get(move.to)?.color !== game.get(square)?.color
            ? 'radial-gradient(circle, rgba(0,0,0,.16) 85%, transparent 85%)'
            : 'radial-gradient(circle, rgba(0,0,0,.16) 25%, transparent 25%)',
        borderRadius: '50%',
      };
    }

    newSquares[square] = { background: 'rgba(253, 224, 71, 0.45)' };
    setOptionSquares(newSquares);
    return true;
  };

  const commitMove = (sourceSquare, targetSquare) => {
    if (canUserMove && !canUserMove(game)) {
      return false;
    }

    try {
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });

      if (!move) return false;

      const baseMoves = moves.slice(0, currentMoveIndex + 1);
      const updatedMoves = [...baseMoves, move.san];
      onMovePlayed?.(updatedMoves);

      setMoveFrom('');
      setOptionSquares({});
      return true;
    } catch {
      return false;
    }
  };

  const onSquareClick = ({ square, piece }) => {
    if (!interactive) return;
    if (canUserMove && !canUserMove(game)) return;

    if (!moveFrom && piece) {
      const hasMoveOptions = getMoveOptions(square);
      if (hasMoveOptions) setMoveFrom(square);
      return;
    }

    if (!moveFrom) return;

    const moved = commitMove(moveFrom, square);
    if (moved) return;

    const hasMoveOptions = getMoveOptions(square);
    setMoveFrom(hasMoveOptions ? square : '');
  };

  const onPieceDrop = ({ sourceSquare, targetSquare }) => {
    if (!interactive || !targetSquare) return false;
    return commitMove(sourceSquare, targetSquare);
  };

  const chessboardOptions = {
    id: "pgn-viewer-board",
    position: game.fen(),
    boardOrientation: boardOrientation,
    arePiecesDraggable: interactive,
    onSquareClick,
    onPieceDrop,
    squareStyles: optionSquares,
    animationDuration: 200,
    customBoardStyle: {
      borderRadius: "4px",
      boxShadow: "0 2px 10px rgba(0,0,0,0.5)",
    },
  };

  return <Chessboard options={chessboardOptions} />;
}