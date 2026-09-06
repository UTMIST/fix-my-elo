import React, { useState } from 'react';
import MctsTreeNode from './MctsTreeNode';
import MctsProbabilityChart from './MctsProbabilityChart';
import './MctsVisualizer.css';

// Helper to compute search stats recursively
const computeTreeStats = (node) => {
  if (!node) return { totalNodes: 0, expandedNodes: 0, maxDepth: 0 };

  let expanded = 0;
  let maxDepth = 0;
  let total = 0;

  const traverse = (currNode, depth) => {
    if (!currNode) return;
    total += 1;
    if (currNode.visits > 0 || (currNode.count && currNode.count > 0)) {
      expanded += 1;
    }
    if (depth > maxDepth) {
      maxDepth = depth;
    }

    const children = currNode.children || (currNode.node && currNode.node.children);
    if (children && children.length > 0) {
      children.forEach((child) => {
        if (child.node) {
          traverse(child.node, depth + 1);
        } else {
          total += 1;
          if (depth + 1 > maxDepth) maxDepth = depth + 1;
        }
      });
    }
  };

  traverse(node, 1);
  return { totalNodes: total, expandedNodes: expanded, maxDepth };
};

export default function MctsVisualizer({ tree, isThinking = false }) {
  const [activeTab, setActiveTab] = useState('chart');
  const [showUnvisited, setShowUnvisited] = useState(true);

  if (isThinking) {
    return (
      <div className="mcts-visualizer-panel pgn-panel">
        <div className="mcts-header">
          <h2 className="panel-title">MCTS Search Visualizer</h2>
        </div>
        <div className="mcts-loading-state">
          <div className="mcts-spinner" />
          <p>Engine is running Monte Carlo simulations...</p>
        </div>
      </div>
    );
  }

  if (!tree || !tree.children || tree.children.length === 0) {
    return (
      <div className="mcts-visualizer-panel pgn-panel">
        <div className="mcts-header">
          <h2 className="panel-title">MCTS Search Visualizer</h2>
        </div>
        <div className="mcts-empty-state">
          <p>Play a move against the AI to view the MCTS search tree and probability distribution chart.</p>
        </div>
      </div>
    );
  }

  const stats = computeTreeStats(tree);

  return (
    <div className="mcts-visualizer-panel pgn-panel">
      <div className="mcts-header">
        <h2 className="panel-title">MCTS Search Visualizer</h2>
        <div className="mcts-stats-summary">
          <span className="stat-item">
            <strong>Root Visits:</strong> {tree.visits}
          </span>
          <span className="stat-item">
            <strong>Expanded Nodes:</strong> {stats.expandedNodes}
          </span>
          <span className="stat-item">
            <strong>Max Depth:</strong> {stats.maxDepth}
          </span>
          <span className="stat-item">
            <strong>Turn:</strong> {tree.turn}
          </span>
        </div>
      </div>

      <div className="mcts-toolbar">
        <div className="mcts-tab-buttons" role="tablist">
          <button
            className={`mcts-tab-btn ${activeTab === 'chart' ? 'active' : ''}`}
            onClick={() => setActiveTab('chart')}
            role="tab"
            aria-selected={activeTab === 'chart'}
          >
            📊 Move Probabilities
          </button>
          <button
            className={`mcts-tab-btn ${activeTab === 'tree' ? 'active' : ''}`}
            onClick={() => setActiveTab('tree')}
            role="tab"
            aria-selected={activeTab === 'tree'}
          >
            🌲 Search Tree Explorer
          </button>
        </div>

        <div className="mcts-filter-controls">
          <label className="mcts-checkbox-label">
            <input
              type="checkbox"
              checked={showUnvisited}
              onChange={(e) => setShowUnvisited(e.target.checked)}
            />
            <span>Show Unvisited Moves</span>
          </label>
        </div>
      </div>

      <div className="mcts-content-container">
        {activeTab === 'chart' && (
          <MctsProbabilityChart node={tree} />
        )}

        {activeTab === 'tree' && (
          <div className="mcts-tree-container">
            <div className="root-node-header">
              <span className="root-label">Root Position</span>
              <span className="root-fen-badge" title={tree.fen}>{tree.fen}</span>
            </div>

            <div className="tree-root-children">
              {tree.children.map((childNode, idx) => (
                <MctsTreeNode
                  key={`${childNode.move}-${idx}`}
                  node={childNode}
                  showUnvisited={showUnvisited}
                  depth={1}
                  defaultExpanded={idx === 0}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
