

// export default function Home() {
//   return (
//     <div className="flex min-h-screen flex-col items-center justify-center py-2">
//       Fix-My-Elo
//     </div>
//   );
// }

import PGNViewer from '../components/main/MainViewer.jsx'
import TextChessboard from '../components/TextChessboard.jsx'

export default function Home() {
  // return <TextChessboard />
  return <PGNViewer />
}

