"use client";

import { Chess } from 'chess.js';
import { useState, useEffect, useMemo, useRef } from 'react';
import { usePGNParser } from '../../hooks/usePGNParser';
import { useGameNavigation } from '../../hooks/useGameNavigation';
import { useAutoPlay } from '../../hooks/useAutoPlay';
import ChessBoard from '../board-pane/ChessBoard';
import PGNLoader from '../pgn-pane/PGNLoader';
import GameInfo from '../pgn-pane/GameInfo';
import MovesList from '../user-pane/MovesList';
import BoardControls from '../board-pane/BoardControls';
import BoardOptions from '../board-pane/BoardOptions';
import './MainViewer.css';

const createDefaultHeaders = () => {
  const today = new Date().toISOString().slice(0, 10).replace(/-/g, '.');

  return {
    Event: 'Casual Game',
    Site: 'Fix-My-Elo',
    Date: today,
    Round: '-',
    White: 'White',
    Black: 'Black',
    Result: '*',
  };
};

const toSafeFilePart = (value, fallback) => {
  const normalized = (value || fallback)
    .trim()
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_.-]/g, '');

  return normalized || fallback;
};

const buildPgnFromGame = (headers, moves) => {
  const game = new Chess();

  Object.entries(headers || {}).forEach(([key, value]) => {
    if (value?.trim()) {
      game.header(key, value.trim());
    }
  });

  for (const move of moves || []) {
    try {
      game.move(move);
    } catch {
      break;
    }
  }

  return game.pgn({ maxWidth: 90, newline: '\n' });
};

export default function MainViewer() {
  const [currentGameIndex, setCurrentGameIndex] = useState(0);
  const [boardOrientation, setBoardOrientation] = useState('white');
  const [mode, setMode] = useState('replay');
  const [manualMoves, setManualMoves] = useState([]);
  const [manualHeaders, setManualHeaders] = useState(createDefaultHeaders);
  const [engineMoves, setEngineMoves] = useState([]);
  const [engineUserColor, setEngineUserColor] = useState('white');
  const [engineNumSimulations, setEngineNumSimulations] = useState(120);
  const [engineHeaders, setEngineHeaders] = useState(() => ({
    ...createDefaultHeaders(),
    Event: 'Play vs Team2 Agent',
    White: 'User',
    Black: 'Team2 Agent',
    Result: '*',
  }));
  const [isEngineThinking, setIsEngineThinking] = useState(false);
  const [engineError, setEngineError] = useState('');
  const [copyStatus, setCopyStatus] = useState('');
  const engineRequestIdRef = useRef(0);

  const { games, error, loadPGN } = usePGNParser();
  const safeReplayGameIndex = games.length
    ? Math.min(currentGameIndex, games.length - 1)
    : 0;
  const replayGame = games[safeReplayGameIndex] || { headers: {}, moves: [] };
  const activeGame = mode === 'manual'
    ? { headers: manualHeaders, moves: manualMoves }
    : mode === 'engine'
      ? { headers: engineHeaders, moves: engineMoves }
      : replayGame;

  const {
    currentMoveIndex,
    setCurrentMoveIndex,
    goToMove,
    goToStart,
    goToPrevious,
    goToNext,
    goToEnd,
  } = useGameNavigation(activeGame.moves);

  const maxMoves = activeGame.moves.length - 1;
  const {
    isAutoPlaying,
    setIsAutoPlaying,
    autoPlaySpeed,
    setAutoPlaySpeed,
  } = useAutoPlay(currentMoveIndex, setCurrentMoveIndex, maxMoves);

  const manualPgn = useMemo(
    () => buildPgnFromGame(manualHeaders, manualMoves),
    [manualMoves, manualHeaders],
  );

  const replayPgn = useMemo(
    () => buildPgnFromGame(replayGame.headers, replayGame.moves),
    [replayGame.headers, replayGame.moves],
  );

  const enginePgn = useMemo(
    () => buildPgnFromGame(engineHeaders, engineMoves),
    [engineHeaders, engineMoves],
  );

  useEffect(() => {
    if (!copyStatus) return;

    const timer = setTimeout(() => setCopyStatus(''), 1200);
    return () => clearTimeout(timer);
  }, [copyStatus]);

  const updateEngineResultHeader = (moves) => {
    const game = new Chess();

    for (const move of moves) {
      try {
        game.move(move);
      } catch {
        break;
      }
    }

    const result = game.isGameOver() ? game.result() : '*';
    setEngineHeaders((prev) => ({ ...prev, Result: result }));
  };

  const requestEngineReply = async (movesAfterUserMove, userColorOverride = engineUserColor) => {
    const game = new Chess();
    for (const move of movesAfterUserMove) {
      try {
        game.move(move);
      } catch {
        return;
      }
    }

    if (game.isGameOver()) {
      updateEngineResultHeader(movesAfterUserMove);
      return;
    }

    const expectedEngineTurn = userColorOverride === 'white' ? 'b' : 'w';
    if (game.turn() !== expectedEngineTurn) return;

    const requestId = engineRequestIdRef.current + 1;
    engineRequestIdRef.current = requestId;
    setIsEngineThinking(true);
    setEngineError('');

    try {
      const response = await fetch('/api/agent-move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fen: game.fen(),
          numSimulations: engineNumSimulations,
          temperature: 0,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error || payload?.details?.error || 'Engine request failed');
      }

      const result = game.move(payload.move);
      if (!result) {
        throw new Error('Engine returned an illegal move');
      }

      if (engineRequestIdRef.current !== requestId) return;

      const updatedMoves = [...movesAfterUserMove, result.san];
      setEngineMoves(updatedMoves);
      setCurrentMoveIndex(updatedMoves.length - 1);
      updateEngineResultHeader(updatedMoves);
    } catch (err) {
      if (engineRequestIdRef.current !== requestId) return;
      setEngineError(String(err));
    } finally {
      if (engineRequestIdRef.current === requestId) {
        setIsEngineThinking(false);
      }
    }
  };

  const handleLoadPGN = (pgnText) => {
    const loaded = loadPGN(pgnText);
    if (!loaded) return;

    setMode('replay');
    setCurrentGameIndex(0);
    setCurrentMoveIndex(-1);
    setIsAutoPlaying(false);
  };

  const handleModeChange = (nextMode) => {
    setIsAutoPlaying(false);
    setMode(nextMode);

    if (nextMode === 'manual') {
      setCurrentMoveIndex(manualMoves.length - 1);
    } else if (nextMode === 'engine') {
      setCurrentMoveIndex(engineMoves.length - 1);
    } else {
      setCurrentMoveIndex(-1);
    }
  };

  const handleManualMovePlayed = (updatedMoves) => {
    setIsAutoPlaying(false);
    setManualMoves(updatedMoves);
    setCurrentMoveIndex(updatedMoves.length - 1);
  };

  const handleEngineMovePlayed = (updatedMoves) => {
    setIsAutoPlaying(false);
    setEngineMoves(updatedMoves);
    setCurrentMoveIndex(updatedMoves.length - 1);
    updateEngineResultHeader(updatedMoves);
    requestEngineReply(updatedMoves);
  };

  const resetEngineGame = (nextUserColor = engineUserColor) => {
    engineRequestIdRef.current += 1;
    setIsAutoPlaying(false);
    setEngineError('');
    setIsEngineThinking(false);
    setEngineMoves([]);
    setCurrentMoveIndex(-1);
    setEngineHeaders((prev) => ({
      ...prev,
      White: nextUserColor === 'white' ? 'User' : 'Team2 Agent',
      Black: nextUserColor === 'white' ? 'Team2 Agent' : 'User',
      Result: '*',
    }));

    if (nextUserColor === 'black') {
      requestEngineReply([], nextUserColor);
    }
  };

  const handleEngineColorChange = (color) => {
    setEngineUserColor(color);
    resetEngineGame(color);
  };

  const handleNewManualGame = () => {
    setIsAutoPlaying(false);
    setManualMoves([]);
    setCurrentMoveIndex(-1);
  };

  const handleUndoManualMove = () => {
    setIsAutoPlaying(false);

    if (!manualMoves.length) return;

    const updatedMoves = manualMoves.slice(0, -1);
    setManualMoves(updatedMoves);
    setCurrentMoveIndex(updatedMoves.length - 1);
  };

  const handleCopyManualPGN = async () => {
    if (!manualPgn.trim()) return;

    try {
      await navigator.clipboard.writeText(manualPgn);
      setCopyStatus('Copied');
    } catch {
      setCopyStatus('Copy failed');
    }
  };

  const handleLoadManualIntoReplay = () => {
    if (!manualPgn.trim()) return;

    const loaded = loadPGN(manualPgn);
    if (!loaded) return;

    setMode('replay');
    setCurrentGameIndex(0);
    setCurrentMoveIndex(-1);
    setIsAutoPlaying(false);
  };

  const handleDownloadManualPGN = () => {
    if (!manualPgn.trim()) return;

    const white = toSafeFilePart(manualHeaders.White, 'white');
    const black = toSafeFilePart(manualHeaders.Black, 'black');
    const date = toSafeFilePart(manualHeaders.Date, 'undated');
    const fileName = `${white}_vs_${black}_${date}.pgn`;

    const blob = new Blob([manualPgn], { type: 'application/x-chess-pgn;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handleDownloadReplayPGN = () => {
    if (!replayPgn.trim()) return;

    const white = toSafeFilePart(replayGame.headers?.White, 'white');
    const black = toSafeFilePart(replayGame.headers?.Black, 'black');
    const date = toSafeFilePart(replayGame.headers?.Date, 'undated');
    const gameLabel = toSafeFilePart(String(safeReplayGameIndex + 1), '1');
    const fileName = `${white}_vs_${black}_${date}_game${gameLabel}.pgn`;

    const blob = new Blob([replayPgn], { type: 'application/x-chess-pgn;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handleDownloadEnginePGN = () => {
    if (!enginePgn.trim()) return;

    const white = toSafeFilePart(engineHeaders.White, 'white');
    const black = toSafeFilePart(engineHeaders.Black, 'black');
    const date = toSafeFilePart(engineHeaders.Date, 'undated');
    const fileName = `${white}_vs_${black}_${date}_vs_engine.pgn`;

    const blob = new Blob([enginePgn], { type: 'application/x-chess-pgn;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handlePlayPause = () => {
    if (activeGame.moves.length === 0) return;
    setIsAutoPlaying(!isAutoPlaying);
  };

  const handleMoveClick = (index) => {
    if (mode === 'engine') return;
    setIsAutoPlaying(false);
    goToMove(index);
  };

  const handleNavigation = (action) => {
    setIsAutoPlaying(false);
    action();
  };

  return (
    <div className="pgn-viewer">
      <div className="pgn-container">
        <h1 className="pgn-title">Fix-My-Elo</h1>
        <div className="mode-toggle" role="tablist" aria-label="Game mode">
          <button
            onClick={() => handleModeChange('replay')}
            className={`btn-mode ${mode === 'replay' ? 'active' : ''}`}
            role="tab"
            aria-selected={mode === 'replay'}
          >
            PGN Replay
          </button>
          <button
            onClick={() => handleModeChange('manual')}
            className={`btn-mode ${mode === 'manual' ? 'active' : ''}`}
            role="tab"
            aria-selected={mode === 'manual'}
          >
            Manual Game
          </button>
          <button
            onClick={() => handleModeChange('engine')}
            className={`btn-mode ${mode === 'engine' ? 'active' : ''}`}
            role="tab"
            aria-selected={mode === 'engine'}
          >
            Play vs AI
          </button>
        </div>

        <div className="pgn-grid">
          {/* Left Panel */}
          <div>
            <PGNLoader onLoad={handleLoadPGN} error={error} />

            {mode === 'replay' && (
              <div className="pgn-panel manual-panel">
                <h2 className="panel-title">Export Current Replay</h2>
                <button
                  onClick={handleDownloadReplayPGN}
                  className="btn-load"
                  disabled={!replayGame.moves.length}
                >
                  Download Replay .pgn
                </button>
              </div>
            )}

            {mode === 'manual' && (
              <div className="pgn-panel manual-panel">
                <h2 className="panel-title">Build Your Game</h2>

                <div className="manual-grid">
                  {['White', 'Black', 'Event', 'Site', 'Round', 'Date', 'Result'].map((field) => (
                    <label key={field} className="manual-field">
                      <span>{field}</span>
                      <input
                        value={manualHeaders[field] || ''}
                        onChange={(e) => setManualHeaders((prev) => ({
                          ...prev,
                          [field]: e.target.value,
                        }))}
                      />
                    </label>
                  ))}
                </div>

                <div className="manual-actions">
                  <button onClick={handleNewManualGame} className="btn-secondary">New Game</button>
                  <button onClick={handleUndoManualMove} className="btn-secondary">Undo Move</button>
                </div>

                <textarea
                  className="pgn-textarea generated-pgn"
                  value={manualPgn}
                  readOnly
                />

                <div className="manual-actions">
                  <button onClick={handleCopyManualPGN} className="btn-load">Copy PGN</button>
                  <button onClick={handleLoadManualIntoReplay} className="btn-secondary">Load In Replay</button>
                </div>
                <button onClick={handleDownloadManualPGN} className="btn-load manual-download">
                  Download .pgn
                </button>
                {copyStatus && <div className="copy-status">{copyStatus}</div>}
              </div>
            )}

            {mode === 'engine' && (
              <div className="pgn-panel manual-panel">
                <h2 className="panel-title">Play vs Team2 Agent</h2>
                <div className="manual-grid">
                  <label className="manual-field">
                    <span>Your Color</span>
                    <select
                      className="engine-select"
                      value={engineUserColor}
                      onChange={(e) => handleEngineColorChange(e.target.value)}
                    >
                      <option value="white">White</option>
                      <option value="black">Black</option>
                    </select>
                  </label>
                  <label className="manual-field">
                    <span>MCTS Simulations</span>
                    <input
                      type="number"
                      min="1"
                      max="5000"
                      value={engineNumSimulations}
                      onChange={(e) => setEngineNumSimulations(Math.max(1, Number(e.target.value) || 1))}
                    />
                  </label>
                </div>

                <div className="manual-actions">
                  <button onClick={() => resetEngineGame()} className="btn-secondary">
                    New Engine Game
                  </button>
                  <button
                    onClick={handleDownloadEnginePGN}
                    className="btn-load"
                    disabled={!engineMoves.length}
                  >
                    Download Engine PGN
                  </button>
                </div>

                {isEngineThinking && <div className="copy-status">Engine is thinking...</div>}
                {engineError && <div className="error-message">{engineError}</div>}
              </div>
            )}

            <GameInfo headers={activeGame.headers} />
          </div>

          {/* Center Panel - Board */}
          <div className="pgn-panel board-panel">
            <div className="board-container">
              <ChessBoard
                moves={activeGame.moves}
                currentMoveIndex={currentMoveIndex}
                boardOrientation={boardOrientation}
                interactive={mode === 'manual' || mode === 'engine'}
                onMovePlayed={mode === 'engine' ? handleEngineMovePlayed : handleManualMovePlayed}
                canUserMove={mode === 'engine'
                  ? (game) => {
                    if (isEngineThinking) return false;
                    const userTurn = engineUserColor === 'white' ? 'w' : 'b';
                    return game.turn() === userTurn;
                  }
                  : undefined}
              />
            </div>

            <BoardControls
              onStart={() => handleNavigation(goToStart)}
              onPrevious={() => handleNavigation(goToPrevious)}
              onPlayPause={handlePlayPause}
              onNext={() => handleNavigation(goToNext)}
              onEnd={() => handleNavigation(goToEnd)}
              isPlaying={isAutoPlaying}
              disabled={mode === 'engine'}
            />

            <BoardOptions
              speed={autoPlaySpeed}
              onSpeedChange={setAutoPlaySpeed}
              onFlipBoard={() => setBoardOrientation((o) => o === 'white' ? 'black' : 'white')}
            />
          </div>

          {/* Right Panel - Moves */}
          <MovesList
            moves={activeGame.moves}
            currentMoveIndex={currentMoveIndex}
            onMoveClick={handleMoveClick}
          />
        </div>
      </div>
    </div>
  );
}