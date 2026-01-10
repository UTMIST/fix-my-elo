import { useState, useEffect, useRef } from 'react';

export function useAutoPlay(currentMoveIndex, setCurrentMoveIndex, maxMoves) {
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [autoPlaySpeed, setAutoPlaySpeed] = useState(1000);
  const autoPlayIntervalRef = useRef(null);

  useEffect(() => {
    if (isAutoPlaying) {
      autoPlayIntervalRef.current = setInterval(() => {
        setCurrentMoveIndex((prev) => {
          if (prev < maxMoves) return prev + 1;
          setIsAutoPlaying(false);
          return prev;
        });
      }, autoPlaySpeed);
    } else {
      clearInterval(autoPlayIntervalRef.current);
    }
    return () => clearInterval(autoPlayIntervalRef.current);
  }, [isAutoPlaying, autoPlaySpeed, maxMoves, setCurrentMoveIndex]);

  return {
    isAutoPlaying,
    setIsAutoPlaying,
    autoPlaySpeed,
    setAutoPlaySpeed,
  };
}