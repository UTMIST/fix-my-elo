import React from 'react';

export default function MctsProbabilityChart({ node }) {
  if (!node || !node.children || node.children.length === 0) {
    return (
      <div className="mcts-chart-empty">
        <p>No move probability data available for this position.</p>
      </div>
    );
  }

  // Filter and sort moves by visit count descending, then prior
  const moves = [...node.children].sort((a, b) => {
    if ((b.count || 0) !== (a.count || 0)) return (b.count || 0) - (a.count || 0);
    return (b.prior || 0) - (a.prior || 0);
  });

  const maxVisits = moves[0]?.count || 0;

  // Max percentage for scaling bars dynamically
  const maxPercent = Math.max(
    ...moves.map((m) => Math.max(m.share || 0, m.prior || 0)),
    0.05
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
          <span className="legend-label">Policy Prior P(a)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color share-color" />
          <span className="legend-label">MCTS Visit Share N(a)/N</span>
        </div>
        <div className="legend-info">
          <span>Sort: Ranked by Search Visits</span>
        </div>
      </div>

      <div className="chart-bars-list">
        {moves.map((moveObj, idx) => {
          const priorPct = (moveObj.prior || 0) * 100;
          const sharePct = (moveObj.share || 0) * 100;
          const deltaPct = sharePct - priorPct;
          const isBest = idx === 0 && moveObj.count > 0 && moveObj.count === maxVisits;

          const priorWidth = Math.min(100, Math.max(1.5, (moveObj.prior / maxPercent) * 100));
          const shareWidth = Math.min(100, Math.max(1.5, (moveObj.share / maxPercent) * 100));

          return (
            <div
              key={`${moveObj.move}-${idx}`}
              className={`chart-move-card ${isBest ? 'is-best-move' : ''} ${moveObj.count === 0 ? 'is-unvisited' : ''}`}
            >
              <div className="chart-move-header">
                <span className="chart-rank">#{idx + 1}</span>

                <div className="chart-move-name">
                  <span className="chart-move-san">{moveObj.san || moveObj.move}</span>
                  {moveObj.san && moveObj.san !== moveObj.move && (
                    <span className="chart-move-uci">({moveObj.move})</span>
                  )}
                </div>

                {isBest && (
                  <span className="chart-best-badge" title="Top Visited Move">
                    ★ Top Move
                  </span>
                )}

                <div className="chart-move-right-meta">
                  {moveObj.count > 0 ? (
                    <>
                      <span className="chart-visit-tag" title="Total Visits">
                        {moveObj.count} <small>visits</small>
                      </span>
                      <span className={`chart-eval-badge ${evalClass(moveObj.eval)}`} title="Evaluation Q(s,a)">
                        {formatEval(moveObj.eval)}
                      </span>
                    </>
                  ) : (
                    <span className="chart-unvisited-tag">Unvisited</span>
                  )}
                </div>
              </div>

              <div className="chart-bars-container">
                {/* Policy Prior Bar */}
                <div className="bar-track-wrapper">
                  <span className="bar-track-label prior-label">Prior P(a)</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill prior-fill"
                      style={{ width: `${priorWidth}%` }}
                      title={`Policy Prior: ${priorPct.toFixed(1)}%`}
                    />
                    <span className="bar-value">{priorPct.toFixed(1)}%</span>
                  </div>
                </div>

                {/* MCTS Visit Share Bar */}
                <div className="bar-track-wrapper">
                  <span className="bar-track-label share-label">MCTS Share</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill share-fill"
                      style={{ width: `${shareWidth}%` }}
                      title={`MCTS Visit Share: ${sharePct.toFixed(1)}% (${moveObj.count} visits)`}
                    />
                    <span className="bar-value">
                      {sharePct.toFixed(1)}% {moveObj.count > 0 ? `(${moveObj.count}v)` : ''}
                    </span>
                  </div>
                </div>
              </div>

              <div className="chart-delta-footer">
                <span className="delta-label">Search Shift vs Policy:</span>
                {moveObj.count > 0 ? (
                  <span className={`delta-tag ${deltaPct > 0.05 ? 'up' : deltaPct < -0.05 ? 'down' : 'neutral'}`}>
                    {deltaPct > 0 ? `+${deltaPct.toFixed(1)}%` : `${deltaPct.toFixed(1)}%`}
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
