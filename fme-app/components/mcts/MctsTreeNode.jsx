import React, { useState } from 'react';

export default function MctsTreeNode({
  node,
  showUnvisited = true,
  depth = 0,
  defaultExpanded = false,
  isBestMove = false,
  rank = null,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded || depth === 0);

  if (!node) return null;

  const isVisited = node.count > 0;
  if (!isVisited && !showUnvisited) {
    return null;
  }

  const childTree = node.node; // Sub-tree position object
  const rawChildren = childTree?.children || [];

  // Sort children by visits descending then prior
  const sortedChildren = [...rawChildren].sort((a, b) => {
    if ((b.count || 0) !== (a.count || 0)) return (b.count || 0) - (a.count || 0);
    return (b.prior || 0) - (a.prior || 0);
  });

  const hasSubChildren = sortedChildren.length > 0;
  const maxChildVisits = sortedChildren[0]?.count || 0;

  const formatEval = (evalVal) => {
    if (evalVal === undefined || evalVal === null) return '0.00';
    const num = Number(evalVal);
    const sign = num > 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}`;
  };

  const evalClass = (evalVal) => {
    if (evalVal > 0.05) return 'eval-badge positive';
    if (evalVal < -0.05) return 'eval-badge negative';
    return 'eval-badge neutral';
  };

  const priorPct = ((node.prior || 0) * 100).toFixed(1);
  const sharePct = ((node.share || 0) * 100).toFixed(1);

  return (
    <div className={`mcts-tree-node ${isVisited ? 'visited' : 'unvisited'} ${isBestMove ? 'best-move-node' : ''}`}>
      <div
        className={`node-row ${expanded ? 'is-expanded' : ''} ${hasSubChildren ? 'has-children' : ''}`}
        onClick={() => hasSubChildren && setExpanded(!expanded)}
      >
        <span className="expand-icon" title={hasSubChildren ? (expanded ? 'Collapse branch' : 'Expand branch') : ''}>
          {hasSubChildren ? (
            <svg className={`chevron-icon ${expanded ? 'open' : ''}`} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
          ) : (
            <span className="node-bullet" />
          )}
        </span>

        {rank && <span className="node-rank-badge">#{rank}</span>}

        <div className="move-badge-group">
          <span className="move-san">{node.san || node.move}</span>
          {node.san && node.san !== node.move && <span className="move-uci">{node.move}</span>}
        </div>

        {isBestMove && isVisited && (
          <span className="top-choice-tag" title="Principal Variation (Most Visited Move)">
            ★ Best
          </span>
        )}

        {isVisited ? (
          <>
            <div className="visit-bar-wrapper" title={`${node.count} visits (${sharePct}% of total)`}>
              <div
                className="visit-bar-fill"
                style={{ width: `${Math.min(100, Math.max(4, (node.share || 0) * 100))}%` }}
              />
            </div>

            <div className="visit-metrics">
              <span className="visit-count">{node.count} <small>visits</small></span>
              <span className="visit-share-pct">({sharePct}%)</span>
            </div>

            <span className={evalClass(node.eval)} title="Position Evaluation Q(s,a)">
              {formatEval(node.eval)}
            </span>
          </>
        ) : (
          <span className="unvisited-tag">Unvisited</span>
        )}

        <span className="prior-badge" title="Policy Network Prior P(a)">
          P: {priorPct}%
        </span>
      </div>

      {expanded && hasSubChildren && (
        <div className="node-children">
          {sortedChildren.map((childNode, idx) => (
            <MctsTreeNode
              key={`${childNode.move}-${idx}`}
              node={childNode}
              showUnvisited={showUnvisited}
              depth={depth + 1}
              isBestMove={idx === 0 && (childNode.count || 0) > 0 && (childNode.count === maxChildVisits)}
              rank={idx + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}
