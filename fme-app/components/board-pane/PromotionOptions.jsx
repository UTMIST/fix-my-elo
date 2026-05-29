"use client";

export default function PromotionOptions({ onPromote, color }) {
  const pieces = ['q', 'r', 'b', 'n'];
  const pieceNames = {
    q: 'Queen',
    r: 'Rook',
    b: 'Bishop',
    n: 'Knight',
  }; 

  return (
    <div className="promotion-options">
      <h3>Promote to:</h3>
      <div className="options">
        {pieces.map((p) => (
          <button key={p} onClick={() => onPromote(p)} className={`option option-${color}`}>
            {pieceNames[p]}
          </button>
        ))}
      </div>
    </div>
  )
}