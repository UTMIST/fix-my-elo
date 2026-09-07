import React, { useState, useMemo } from 'react';
import MctsProbabilityChart from './MctsProbabilityChart';
import MctsGraphicalTree from './MctsGraphicalTree';
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

  const stats = useMemo(() => computeTreeStats(tree), [tree]);

  if (isThinking) {
    return (
      <div className="mcts-visualizer-panel pgn-panel">
        <div className="mcts-header">
          <div className="mcts-title-group">
            <h2 className="panel-title">MCTS Search Visualizer</h2>
            <span className="mcts-live-badge">Running MCTS...</span>
          </div>
        </div>
        <div className="mcts-loading-state">
          <div className="mcts-spinner" />
          <p className="loading-title">Monte Carlo Tree Search in Progress</p>
          <p className="loading-subtitle">Simulating move outcomes and building policy tree...</p>
        </div>
      </div>
    );
  }

  if (!tree || !tree.children || tree.children.length === 0) {
    return (
      <div className="mcts-visualizer-panel pgn-panel">
        <div className="mcts-header">
          <div className="mcts-title-group">
            <h2 className="panel-title">MCTS Search Visualizer</h2>
          </div>
        </div>
        <div className="mcts-empty-state">
          <div className="empty-icon">♟️</div>
          <p className="empty-title">No Search Tree Available</p>
          <p className="empty-subtitle">Play a move against the AI to explore the Monte Carlo search tree and move distribution probabilities.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mcts-visualizer-panel pgn-panel">
      {/* Visualizer Header */}
      <div className="mcts-header">
        <div className="mcts-title-group">
          <h2 className="panel-title">MCTS Search Explorer</h2>
          <span className="mcts-turn-tag">
            {tree.turn === 'w' ? '⚪ White to move' : '⚫ Black to move'}
          </span>
        </div>

        {/* Stats Summary Cards */}
        <div className="mcts-stats-summary">
          <div className="stat-card" title="Total simulations run from root position">
            <span className="stat-value">{tree.visits}</span>
            <span className="stat-label">Simulations</span>
          </div>

          <div className="stat-card" title="Nodes visited during search">
            <span className="stat-value">{stats.expandedNodes}</span>
            <span className="stat-label">Expanded Nodes</span>
          </div>

          <div className="stat-card" title="Deepest ply reached in search tree">
            <span className="stat-value">{stats.maxDepth}</span>
            <span className="stat-label">Max Depth</span>
          </div>

          <div className="stat-card" title="Total candidate moves evaluated at root">
            <span className="stat-value">{tree.children.length}</span>
            <span className="stat-label">Root Moves</span>
          </div>
        </div>
      </div>

      {/* Toolbar & Tab Controls */}
      <div className="mcts-toolbar">
        <div className="mcts-tab-buttons" role="tablist">
          <button
            className={`mcts-tab-btn ${activeTab === 'chart' ? 'active' : ''}`}
            onClick={() => setActiveTab('chart')}
            role="tab"
            aria-selected={activeTab === 'chart'}
          >
            <span className="tab-icon">📊</span> Move Distribution
          </button>
          <button
            className={`mcts-tab-btn ${activeTab === 'diagram' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagram')}
            role="tab"
            aria-selected={activeTab === 'diagram'}
          >
            <span className="tab-icon">🌳</span> Visual Tree Diagram
          </button>
        </div>

        <div className="mcts-controls">
          {/* Toggle Unvisited */}
          <label className="mcts-toggle-switch">
            <input
              type="checkbox"
              checked={showUnvisited}
              onChange={(e) => setShowUnvisited(e.target.checked)}
            />
            <span className="toggle-slider" />
            <span className="toggle-label">Show Unvisited</span>
          </label>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="mcts-content-container">
        {activeTab === 'chart' && (
          <MctsProbabilityChart node={tree} />
        )}

        {activeTab === 'diagram' && (
          <MctsGraphicalTree tree={tree} showUnvisited={showUnvisited} />
        )}
      </div>
    </div>
  );
}
