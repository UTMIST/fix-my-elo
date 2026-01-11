"use client";

export default function GameInfo({ headers }) {
  if (Object.keys(headers).length === 0) return null;

  return (
    <div className="game-info">
      <h3 className="info-title">Game Info</h3>
      <div className="info-content">
        {Object.entries(headers).map(([k, v]) => (
          <div key={k} className="info-row">
            <span className="info-key">{k}:</span> {v}
          </div>
        ))}
      </div>
    </div>
  );
}
