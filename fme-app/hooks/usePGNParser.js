import { useState } from 'react';
import { Chess } from 'chess.js';

export function usePGNParser() {
  const [games, setGames] = useState([]);
  const [error, setError] = useState('');

  const parsePGN = (text) => {
    const gameStrings = text.split(/(?=\[Event)/g).filter((s) => s.trim());
    const parsedGames = [];

    for (const gameStr of gameStrings) {
      try {
        const chess = new Chess();
        const loaded = chess.loadPgn(gameStr);

        if (loaded === false) {
          console.warn('Failed to load PGN:', gameStr.substring(0, 100));
          continue;
        }

        const history = chess.history();
        if (history.length === 0) continue;

        const headers = {};
        const headerRegex = /\[(\w+)\s+"([^"]*)"\]/g;
        let match;
        while ((match = headerRegex.exec(gameStr)) !== null) {
          headers[match[1]] = match[2];
        }

        parsedGames.push({ headers, moves: history });
      } catch (e) {
        console.error('Error parsing game:', e);
      }
    }

    return parsedGames;
  };

  const loadPGN = (text) => {
    try {
      setError('');

      if (!text || text.trim().length === 0) {
        setError('Please enter a valid PGN');
        return false;
      }

      const parsedGames = parsePGN(text);

      if (parsedGames.length === 0) {
        setError('No valid games found in PGN');
        return false;
      }

      console.log('Loaded games:', parsedGames.length);
      console.log('First game moves:', parsedGames[0].moves.length);

      setGames(parsedGames);
      return true;
    } catch (err) {
      console.error('Error in loadPGN:', err);
      setError('Error parsing PGN: ' + err.message);
      return false;
    }
  };

  return { games, error, loadPGN };
}