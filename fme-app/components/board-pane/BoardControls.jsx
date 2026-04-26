"use client";

export default function BoardControls({
  onStart,
  onPrevious,
  onPlayPause,
  onNext,
  onEnd,
  isPlaying,
  disabled = false,
}) {
  return (
    <div className="controls">
      <button onClick={onStart} className="btn-nav" disabled={disabled}>⏮</button>
      <button onClick={onPrevious} className="btn-nav" disabled={disabled}>◀</button>
      <button
        onClick={onPlayPause}
        className={`btn-play ${isPlaying ? 'playing' : ''}`}
        disabled={disabled}
      >
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <button onClick={onNext} className="btn-nav" disabled={disabled}>▶</button>
      <button onClick={onEnd} className="btn-nav" disabled={disabled}>⏭</button>
    </div>
  );
}
