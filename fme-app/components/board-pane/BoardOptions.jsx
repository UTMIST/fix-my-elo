"use client";

export default function BoardOptions({
  speed,
  onSpeedChange,
  onFlipBoard,
}) {
  return (
    <div className="board-options">
      {/* <div className="speed-control">
        <div className="speed-label">
          <span>Playback Speed</span>
          <span>{(speed / 1000).toFixed(1)}s</span>
        </div>
        <input
          type="range"
          min="200"
          max="2000"
          step="100"
          value={speed}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
          className="speed-slider"
        />
      </div> */}
      <button onClick={onFlipBoard} className="btn-flip">
        Flip Board
      </button>
    </div>
  );
}