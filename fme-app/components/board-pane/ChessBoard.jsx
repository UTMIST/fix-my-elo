"use client";

import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { useMemo, useState } from "react";
import PromotionOptions from "./PromotionOptions";

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
  const [pendingPromotion, setPendingPromotion] = useState(null);

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

  const isPromotionMove = (sourceSquare, targetSquare) => {
    const piece = game.get(sourceSquare);
    const targetRank = targetSquare[1];
    
    // acctually a promotion
    if (!piece || piece.type !== 'p') {
      return false;
    }
    else if (!((piece.color === 'w' && targetRank === '8') || 
            (piece.color === 'b' && targetRank === '1'))) {
      return false;
    } 

    // is a legal promotion
    const legalMoves = game.moves({ square: sourceSquare, verbose: true });
    return legalMoves.some(m => m.to === targetSquare && m.promotion);
  };

  const squareToPercent = (square) => {
    const file = square.charCodeAt(0) - 97; // a=0, h=7
    const rank = parseInt(square[1]) - 1;   // 1=0, 8=7
    if (boardOrientation === 'white') {
      return { left: (file / 8) * 100, top: ((7 - rank) / 8) * 100 };
    }
    return { left: ((7 - file) / 8) * 100, top: (rank / 8) * 100 };
  };

  // moving logic
  const commitMove = (sourceSquare, targetSquare) => {
    // only let users move if it's their turn
    if (canUserMove && !canUserMove(game)) {
      return false;
    }

    if (isPromotionMove(sourceSquare, targetSquare)) {
      const piece = game.get(sourceSquare);
      setPendingPromotion({ sourceSquare, targetSquare, color: piece.color });

      // get rid of see-through piece + dot while player decides 
      setMoveFrom('');
      setOptionSquares({});

      return true;
    }

    try {
      const move = game.move({ from: sourceSquare, to: targetSquare });

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

  const finishPromotion = (piece) => {
    if (!pendingPromotion) {
      return;
    }
    const { sourceSquare, targetSquare } = pendingPromotion;

    try {
      const move = game.move({ from: sourceSquare, to: targetSquare, promotion: piece });
      if (move) {
        const baseMoves = moves.slice(0, currentMoveIndex + 1);
        const updatedMoves = [...baseMoves, move.san];
        onMovePlayed?.(updatedMoves);
      }
    } catch {
      // nothing lol
    }
    
    setPendingPromotion(null);
    setMoveFrom('');
    setOptionSquares({});
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

  const popupPosition = pendingPromotion ? squareToPercent(pendingPromotion.targetSquare) : null;

  return (
    <div style={{ position: 'relative' }}>
      <Chessboard options={chessboardOptions} />
      {pendingPromotion && (
        <div
          style={{
            position: 'absolute',
            left: `${popupPosition.left}%`,
            top: `${popupPosition.top}%`,
            transform: popupPosition.top < 50 ? 'translateX(-10%)' : 'translate(-10%, -100%)',
            zIndex: 10,
          }}
        >
          <PromotionOptions
            color={pendingPromotion.color}
            onPromote={finishPromotion}
          />
        </div>
      )}
    </div>
  );
}