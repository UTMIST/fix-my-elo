"use client";

import { useRef, useEffect } from 'react';

export default function MovesList({ moves, currentMoveIndex, onMoveClick }) {
  const moveListRef = useRef(null);

  useEffect(() => {
    const activeMoveElement = moveListRef.current?.querySelector('.active-move');
    if (activeMoveElement) {
      activeMoveElement.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [currentMoveIndex]);

  return (
    <div className="pgn-panel moves-panel">
      <h2 className="panel-title">Moves History</h2>
      <div ref={moveListRef} className="moves-list">
        <div className="moves-grid">
          {moves.map((move, i) => (
            <div
              key={i}
              onClick={() => onMoveClick(i)}
              className={`move-item ${i === currentMoveIndex ? 'active-move' : ''}`}
            >
              <span className="move-number">
                {i % 2 === 0 ? `${Math.floor(i / 2) + 1}.` : ''}
              </span>
              <span>{move}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}