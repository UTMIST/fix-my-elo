import { useState, useCallback } from 'react';

export function useGameNavigation(games, currentGameIndex) {
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);

  const goToMove = useCallback((index) => {
    const maxIndex = (games[currentGameIndex]?.moves.length || 0) - 1;
    console.log('goToMove called with:', index, 'Max:', maxIndex);

    if (index < -1) {
      setCurrentMoveIndex(-1);
    } else if (index > maxIndex) {
      setCurrentMoveIndex(maxIndex);
    } else {
      setCurrentMoveIndex(index);
    }
  }, [games, currentGameIndex]);

  const goToStart = useCallback(() => goToMove(-1), [goToMove]);
  const goToPrevious = useCallback(() => goToMove(currentMoveIndex - 1), [goToMove, currentMoveIndex]);
  const goToNext = useCallback(() => goToMove(currentMoveIndex + 1), [goToMove, currentMoveIndex]);
  const goToEnd = useCallback(() => {
    const maxIndex = (games[currentGameIndex]?.moves.length || 0) - 1;
    goToMove(maxIndex);
  }, [goToMove, games, currentGameIndex]);

  return {
    currentMoveIndex,
    setCurrentMoveIndex,
    goToMove,
    goToStart,
    goToPrevious,
    goToNext,
    goToEnd,
  };
}