"use client";

import { useState } from 'react';

export default function PGNLoader({ onLoad, error }) {
  const [pgnInput, setPgnInput] = useState('');

  return (
    <div className="pgn-panel">
      <h2 className="panel-title">Load PGN</h2>
      <textarea
        value={pgnInput}
        onChange={(e) => setPgnInput(e.target.value)}
        className="pgn-textarea"
        placeholder="Paste your PGN here..."
      />
      <button onClick={() => onLoad(pgnInput)} className="btn-load">
        Load PGN
      </button>

      {error && <div className="error-message">{error}</div>}
    </div>
  );
}