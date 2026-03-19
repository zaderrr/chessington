import "./App.css";
import MoveBoard from "./assets/components/moveBoard";
import { useState } from "react";
import { Chess } from "chess.js";
import Board from "./assets/components/board";

export const defaultGameFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0";
function App() {
    const [game, setGame] = useState<Chess | null>(null);
    const [boardPos, setBoardPos] = useState<string>(defaultGameFen);
    const startChessGame = () => {
        const newGame = new Chess(defaultGameFen);
        setGame(newGame);
    };

    return (
        <div className="flex flex-col w-full h-full">
            <div className="w-full flex flex-row justify-center">
                <h1 className="text-3xl">Play with chessington!</h1>
            </div>
            <div className="w-full flex h-full flex-col justify-center">
                <div className="w-full  flex flex-row h-full justify-center">
                    <div className="flex flex-row w-1/5 flex-3  justify-center ">
                        <Board currentGame={game} setBoardPos={setBoardPos} boardPos={boardPos} />
                    </div>
                    <div className="flex h-full flex-col justify-center w-1/4 ">
                        <div className="flex flex-row justify-center w-full h-3/4">
                            <MoveBoard currentGame={game} startGame={startChessGame} boardPos={boardPos} key={boardPos} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
