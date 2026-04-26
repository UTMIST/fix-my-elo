import { useState, useCallback } from 'react';

export function useGameNavigation(moves) {
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);

  const goToMove = useCallback((index) => {
    const maxIndex = (moves?.length || 0) - 1;

    if (index < -1) {
      setCurrentMoveIndex(-1);
    } else if (index > maxIndex) {
      setCurrentMoveIndex(maxIndex);
    } else {
      setCurrentMoveIndex(index);
    }
  }, [moves]);

  const goToStart = useCallback(() => goToMove(-1), [goToMove]);
  const goToPrevious = useCallback(() => goToMove(currentMoveIndex - 1), [goToMove, currentMoveIndex]);
  const goToNext = useCallback(() => goToMove(currentMoveIndex + 1), [goToMove, currentMoveIndex]);
  const goToEnd = useCallback(() => {
    const maxIndex = (moves?.length || 0) - 1;
    goToMove(maxIndex);
  }, [goToMove, moves]);

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