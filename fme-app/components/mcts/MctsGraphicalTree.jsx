import React, { useState, useRef, useEffect } from 'react';

// Recursive Node Component for Graphical Tree Diagram
function GraphicalNode({
  node,
  moveName,
  depth = 0,
  showUnvisited = true,
  isBestMove = false,
  rank = null,
  path = '',
  expandedPaths,
  toggleExpand,
}) {
  if (!node) return null;

  const isVisited = (node.count || 0) > 0 || (node.visits || 0) > 0;
  if (!isVisited && !showUnvisited) {
    return null;
  }

  const visits = node.count !== undefined ? node.count : (node.visits || 0);
  const childTree = node.node; // Sub-tree position object
  const rawChildren = childTree?.children || node.children || [];

  // Sort children by visit count descending
  const sortedChildren = [...rawChildren].sort((a, b) => {
    const vB = b.count !== undefined ? b.count : (b.visits || 0);
    const vA = a.count !== undefined ? a.count : (a.visits || 0);
    if (vB !== vA) return vB - vA;
    return (b.prior || 0) - (a.prior || 0);
  });

  // Filter unvisited if hidden
  const visibleChildren = sortedChildren.filter(
    (c) => showUnvisited || (c.count || 0) > 0 || (c.visits || 0) > 0
  );

  const hasChildren = visibleChildren.length > 0;
  const isExpanded = expandedPaths.has(path);
  const maxChildVisits = visibleChildren[0]?.count || visibleChildren[0]?.visits || 0;

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

  const priorPct = ((node.prior || 0) * 100).toFixed(1);
  const sharePct = ((node.share || 0) * 100).toFixed(1);

  return (
    <li className={`tree-branch-item ${isBestMove ? 'is-best-branch' : ''}`}>
      <div
        className={`tree-card-node ${isVisited ? 'visited' : 'unvisited'} ${
          isBestMove ? 'best-move-card' : ''
        } ${isExpanded ? 'is-expanded' : ''}`}
        onClick={() => hasChildren && toggleExpand(path)}
      >
        {/* Top bar: Rank + Move SAN + Best Tag */}
        <div className="card-node-header">
          {rank && <span className="card-rank-pill">#{rank}</span>}
          <span className="card-move-san">{moveName || node.san || node.move || 'Root'}</span>
          {isBestMove && isVisited && (
            <span className="card-best-star" title="Principal Variation (Most Visited Move)">
              ★ Best
            </span>
          )}
        </div>

        {/* Mid: Visits Bar + Stats */}
        {isVisited ? (
          <div className="card-node-body">
            <div className="card-visit-bar-track" title={`${visits} visits (${sharePct}%)`}>
              <div
                className="card-visit-bar-fill"
                style={{ width: `${Math.min(100, Math.max(6, (node.share || 0) * 100))}%` }}
              />
            </div>

            <div className="card-metrics-row">
              <span className="card-visits-count">{visits} <small>v</small> ({sharePct}%)</span>
              <span className={`card-eval-badge ${evalClass(node.eval)}`}>
                {formatEval(node.eval)}
              </span>
            </div>
          </div>
        ) : (
          <div className="card-node-body">
            <span className="card-unvisited-label">Unvisited Node</span>
          </div>
        )}

        {/* Footer: Policy Prior */}
        <div className="card-node-footer">
          <span className="card-prior-tag">Prior P: {priorPct}%</span>
          {hasChildren && (
            <span className="card-expand-toggle">
              {isExpanded ? '− Collapse' : `+ ${visibleChildren.length} moves`}
            </span>
          )}
        </div>
      </div>

      {/* Children Sub-Tree Diagram */}
      {isExpanded && hasChildren && (
        <ul className="tree-children-group">
          {visibleChildren.map((childNode, idx) => {
            const childMove = childNode.san || childNode.move;
            const childVisits = childNode.count !== undefined ? childNode.count : (childNode.visits || 0);
            const childPath = `${path}/${childMove}-${idx}`;
            const childIsBest = idx === 0 && childVisits > 0 && childVisits === maxChildVisits;

            return (
              <GraphicalNode
                key={childPath}
                node={childNode}
                moveName={childMove}
                depth={depth + 1}
                showUnvisited={showUnvisited}
                isBestMove={childIsBest}
                rank={idx + 1}
                path={childPath}
                expandedPaths={expandedPaths}
                toggleExpand={toggleExpand}
              />
            );
          })}
        </ul>
      )}
    </li>
  );
}

export default function MctsGraphicalTree({ tree, showUnvisited = true }) {
  const [expandedPaths, setExpandedPaths] = useState(() => {
    // By default, expand root and top move (best move path)
    const initial = new Set(['root']);
    if (tree?.children && tree.children.length > 0) {
      const topChild = [...tree.children].sort((a, b) => (b.count || 0) - (a.count || 0))[0];
      if (topChild) {
        const topMove = topChild.san || topChild.move;
        initial.add(`root/${topMove}-0`);
      }
    }
    return initial;
  });

  const [zoomLevel, setZoomLevel] = useState(1);
  const containerRef = useRef(null);

  const toggleExpand = (path) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const expandAllTopMoves = () => {
    const next = new Set(['root']);
    let curr = tree;
    let currPath = 'root';

    for (let d = 0; d < 4; d++) {
      const rawChildren = curr?.children || curr?.node?.children || [];
      if (!rawChildren.length) break;
      const sorted = [...rawChildren].sort((a, b) => (b.count || 0) - (a.count || 0));
      const best = sorted[0];
      if (!best || (best.count || 0) === 0) break;
      const move = best.san || best.move;
      currPath = `${currPath}/${move}-0`;
      next.add(currPath);
      curr = best.node || best;
    }
    setExpandedPaths(next);
  };

  const collapseAll = () => {
    setExpandedPaths(new Set(['root']));
  };

  if (!tree) return null;

  return (
    <div className="graphical-tree-wrapper">
      {/* Controls Bar */}
      <div className="graphical-tree-controls">
        <div className="control-group">
          <button className="tree-btn" onClick={expandAllTopMoves} title="Expand Best Path (Principal Variation)">
            ⭐ Expand Best Path
          </button>
          <button className="tree-btn secondary" onClick={collapseAll} title="Collapse all branches">
            📁 Collapse All
          </button>
        </div>

        <div className="zoom-controls">
          <span className="zoom-label">Zoom:</span>
          <button
            className="zoom-btn"
            onClick={() => setZoomLevel((z) => Math.max(0.6, z - 0.1))}
            disabled={zoomLevel <= 0.6}
          >
            −
          </button>
          <span className="zoom-pct">{Math.round(zoomLevel * 100)}%</span>
          <button
            className="zoom-btn"
            onClick={() => setZoomLevel((z) => Math.min(1.4, z + 0.1))}
            disabled={zoomLevel >= 1.4}
          >
            +
          </button>
          <button className="zoom-btn reset" onClick={() => setZoomLevel(1)}>
            Reset
          </button>
        </div>
      </div>

      {/* Tree Canvas */}
      <div className="graphical-tree-canvas" ref={containerRef}>
        <div
          className="graphical-tree-viewport"
          style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top center' }}
        >
          <ul className="tree-root-group">
            <GraphicalNode
              node={tree}
              moveName="Root Position"
              depth={0}
              showUnvisited={showUnvisited}
              isBestMove={true}
              path="root"
              expandedPaths={expandedPaths}
              toggleExpand={toggleExpand}
            />
          </ul>
        </div>
      </div>
    </div>
  );
}
