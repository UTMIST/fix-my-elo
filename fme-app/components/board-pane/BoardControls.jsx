"use client";

export default function BoardControls({
  onStart,
  onPrevious,
  onPlayPause,
  onNext,
  onEnd,
  isPlaying,
}) {
  return (
    <div className="controls">
      <button onClick={onStart} className="btn-nav">⏮</button>
      <button onClick={onPrevious} className="btn-nav">◀</button>
      <button
        onClick={onPlayPause}
        className={`btn-play ${isPlaying ? 'playing' : ''}`}
      >
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <button onClick={onNext} className="btn-nav">▶</button>
      <button onClick={onEnd} className="btn-nav">⏭</button>
    </div>
  );
}
