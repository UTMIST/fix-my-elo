import React from 'react';

export default function MctsProbabilityChart({ node }) {
  if (!node || !node.children || node.children.length === 0) {
    return (
      <div className="mcts-chart-empty">
        <p>No move probability data available for this position.</p>
      </div>
    );
  }

  // Filter and sort moves by visit count or prior
  const moves = [...node.children].sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return b.prior - a.prior;
  });

  // Max percentage for scaling bars
  const maxPercent = Math.max(
    ...moves.map((m) => Math.max(m.share || 0, m.prior || 0)),
    0.1
  );

  const formatEval = (evalVal) => {
    if (evalVal === undefined || evalVal === null) return '0.00';
    const num = Number(evalVal);
    const sign = num > 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}`;
  };

  const evalClass = (evalVal) => {
    if (evalVal > 0.05) return 'positive';
    if (evalVal < -0.05) return 'negative';
    return 'neutral';
  };

  return (
    <div className="mcts-probability-chart">
      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-color prior-color" />
          <span className="legend-label">Policy Network Prior P(a)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color share-color" />
          <span className="legend-label">MCTS Search Visit Share N(a)/N</span>
        </div>
      </div>

      <div className="chart-bars-list">
        {moves.map((moveObj, idx) => {
          const priorPct = (moveObj.prior || 0) * 100;
          const sharePct = (moveObj.share || 0) * 100;
          const deltaPct = sharePct - priorPct;

          const priorWidth = Math.min(100, Math.max(2, (moveObj.prior / maxPercent) * 100));
          const shareWidth = Math.min(100, Math.max(2, (moveObj.share / maxPercent) * 100));

          return (
            <div key={`${moveObj.move}-${idx}`} className="chart-move-row">
              <div className="chart-move-meta">
                <span className="chart-move-san">{moveObj.san || moveObj.move}</span>
                <span className="chart-move-uci">({moveObj.move})</span>
                {moveObj.count > 0 ? (
                  <span className={`chart-eval-badge ${evalClass(moveObj.eval)}`}>
                    {formatEval(moveObj.eval)}
                  </span>
                ) : (
                  <span className="chart-unvisited-tag">Unvisited</span>
                )}
              </div>

              <div className="chart-bars-container">
                {/* Policy Prior Bar */}
                <div className="bar-track">
                  <div
                    className="bar-fill prior-fill"
                    style={{ width: `${priorWidth}%` }}
                    title={`Policy Prior: ${priorPct.toFixed(1)}%`}
                  />
                  <span className="bar-label">P: {priorPct.toFixed(1)}%</span>
                </div>

                {/* MCTS Visit Share Bar */}
                <div className="bar-track">
                  <div
                    className="bar-fill share-fill"
                    style={{ width: `${shareWidth}%` }}
                    title={`MCTS Visit Share: ${sharePct.toFixed(1)}% (${moveObj.count} visits)`}
                  />
                  <span className="bar-label">S: {sharePct.toFixed(1)}% ({moveObj.count}v)</span>
                </div>
              </div>

              <div className="chart-delta-badge">
                {moveObj.count > 0 ? (
                  <span className={`delta-tag ${deltaPct >= 0 ? 'up' : 'down'}`}>
                    {deltaPct >= 0 ? `+${deltaPct.toFixed(1)}%` : `${deltaPct.toFixed(1)}%`}
                  </span>
                ) : (
                  <span className="delta-tag neutral">0.0%</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
