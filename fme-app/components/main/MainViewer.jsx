"use client";

import { useState, useEffect } from 'react';
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

export default function MainViewer() {
  const [currentGameIndex, setCurrentGameIndex] = useState(0);
  const [boardOrientation, setBoardOrientation] = useState('white');

  const { games, error, loadPGN } = usePGNParser();
  const {
    currentMoveIndex,
    setCurrentMoveIndex,
    goToMove,
    goToStart,
    goToPrevious,
    goToNext,
    goToEnd,
  } = useGameNavigation(games, currentGameIndex);

  const maxMoves = (games[currentGameIndex]?.moves.length || 0) - 1;
  const {
    isAutoPlaying,
    setIsAutoPlaying,
    autoPlaySpeed,
    setAutoPlaySpeed,
  } = useAutoPlay(currentMoveIndex, setCurrentMoveIndex, maxMoves);

  // Load sample PGN on mount
//   useEffect(() => {
//     const samplePGN = `[Event "Tata Steel India Rapid"]
// [Site "Kolkata IND"]
// [Date "2026.01.07"]
// [Round "1.1"]
// [White "Anand, Viswanathan"]
// [Black "So, Wesley"]
// [Result "1-0"]

// 1. e4 c6 2. d4 d5 3. e5 Bf5 4. Nd2 e6 5. Nb3 Nd7 6. Nf3 a6 7. Be2 c5 8. dxc5
// Bxc5 9. Nxc5 Nxc5 10. O-O Ne7 11. Nd4 Be4 12. Be3 Qc7 13. f4 Nf5 14. Nxf5 exf5
// 15. c3 Rc8 16. a4 O-O 17. a5 Qc6 18. Rf2 h6 19. Bf1 Ne6 20. Ra4 Qc7 21. Bb6 Qe7
// 22. Rb4 g5 23. fxg5 hxg5 24. g3 Rc6 25. Bg2 Ng7 26. Bd4 Rfc8 27. Be3 Rg6 28. Bd4
// f4 29. Qe2 Bxg2 30. Rxg2 Ne6 31. Bb6 Kg7 32. Qd3 d4 33. Bxd4 g4 34. gxf4 Nxf4
// 35. Qf5 Nh3+ 36. Kf1 Rg8 37. e6+ Kh6 38. Be3+ Ng5 39. h4 Kg7 40. hxg5 1-0`;

//     loadPGN(samplePGN);
//   }, []);

  useEffect(() => {
    console.log('Rendering with currentMoveIndex:', currentMoveIndex);
    console.log('Board orientation:', boardOrientation);
  }, [currentMoveIndex, boardOrientation]);

  const currentGame = games[currentGameIndex] || { headers: {}, moves: [] };

  const handlePlayPause = () => {
    setIsAutoPlaying(!isAutoPlaying);
  };

  const handleMoveClick = (index) => {
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

        <div className="pgn-grid">
          {/* Left Panel */}
          <div>
            <PGNLoader onLoad={loadPGN} error={error} />
            <GameInfo headers={currentGame.headers} />
          </div>

          {/* Center Panel - Board */}
          <div className="pgn-panel board-panel">
            <div className="board-container">
              <ChessBoard
                moves={currentGame.moves}
                currentMoveIndex={currentMoveIndex}
                boardOrientation={boardOrientation}
              />
            </div>

            <BoardControls
              onStart={() => handleNavigation(goToStart)}
              onPrevious={() => handleNavigation(goToPrevious)}
              onPlayPause={handlePlayPause}
              onNext={() => handleNavigation(goToNext)}
              onEnd={() => handleNavigation(goToEnd)}
              isPlaying={isAutoPlaying}
            />

            <BoardOptions
              speed={autoPlaySpeed}
              onSpeedChange={setAutoPlaySpeed}
              onFlipBoard={() => setBoardOrientation((o) => o === 'white' ? 'black' : 'white')}
            />
          </div>

          {/* Right Panel - Moves */}
          <MovesList
            moves={currentGame.moves}
            currentMoveIndex={currentMoveIndex}
            onMoveClick={handleMoveClick}
          />
        </div>
      </div>
    </div>
  );
}