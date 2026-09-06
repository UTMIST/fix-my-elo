import React, { useState } from 'react';

export default function MctsTreeNode({ node, showUnvisited = true, depth = 0, defaultExpanded = false }) {
  const [expanded, setExpanded] = useState(defaultExpanded || depth === 0);

  if (!node) return null;

  const isVisited = node.count > 0;
  if (!isVisited && !showUnvisited) {
    return null;
  }

  const childTree = node.node; // Sub-tree position object
  const hasSubChildren = childTree && childTree.children && childTree.children.length > 0;

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
    <div className={`mcts-tree-node ${isVisited ? 'visited' : 'unvisited'}`}>
      <div
        className="node-row"
        onClick={() => hasSubChildren && setExpanded(!expanded)}
        style={{ cursor: hasSubChildren ? 'pointer' : 'default' }}
      >
        <span className="expand-icon">
          {hasSubChildren ? (expanded ? '▼' : '▶') : '•'}
        </span>

        <span className="move-san">{node.san || node.move}</span>
        <span className="move-uci">({node.move})</span>

        {isVisited ? (
          <>
            <div className="visit-bar-container" title={`${node.count} visits (${sharePct}%)`}>
              <div
                className="visit-bar-fill"
                style={{ width: `${Math.min(100, Math.max(5, (node.share || 0) * 100))}%` }}
              />
            </div>

            <span className="visit-count">{node.count} visits ({sharePct}%)</span>
            <span className={evalClass(node.eval)}>{formatEval(node.eval)}</span>
          </>
        ) : (
          <span className="unvisited-tag">Unvisited</span>
        )}

        <span className="prior-badge" title="Policy Network Prior Probability">
          P: {priorPct}%
        </span>
      </div>

      {expanded && hasSubChildren && (
        <div className="node-children">
          {childTree.children.map((childNode, idx) => (
            <MctsTreeNode
              key={`${childNode.move}-${idx}`}
              node={childNode}
              showUnvisited={showUnvisited}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}
