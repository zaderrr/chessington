import "./App.css";
import MoveBoard from "./assets/components/moveBoard";
import { useRef, useState } from "react";
import { Chess } from "chess.js";
import Board from "./assets/components/board";

export const defaultGameFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0";
function App() {
    const [game, setGame] = useState<Chess | null>(null);
    const [boardPos, setBoardPos] = useState<string>(defaultGameFen);
    const [server, setServer] = useState<WebSocket | null>(null);
    const gameRef = useRef<Chess | null>(null);
    const startChessGame = () => {
        const newGame = new Chess(defaultGameFen);
        setGame(newGame);
        gameRef.current = newGame;
    };
    const getFirstMoveFromPgn = (raw: string): string | null => {
        const withoutHeaders = raw.replace(/^\s*(?:\[[^\]]*\]\s*)*/, "").trim();
        const withoutLeadingMoveNumber = withoutHeaders.replace(/^\d+\.(?:\.\.)?\s*/, "");
        const firstToken = withoutLeadingMoveNumber.match(/^([^\s{}()]+)/)?.[1] ?? null;

        if (!firstToken) return null;
        if (["*", "1-0", "0-1", "1/2-1/2"].includes(firstToken)) return null;
        return firstToken;
    };

    const onMessageReceived = (event: MessageEvent) => {
        if (!gameRef.current || typeof event.data !== "string") return;
        const aiMove = getFirstMoveFromPgn(event.data);
        if (!aiMove) return;

        const moved = gameRef.current.move(aiMove);
        if (!moved) {
            console.warn(`Invalid AI move received: ${aiMove}`);
            return;
        }

        setBoardPos(gameRef.current.fen());
    };
    const onConnected = () => {
        console.log("Connected with chessington");
    };
    const sendMessage = (pgn: string) => {
        if (!server || server.readyState !== WebSocket.OPEN) return;
        server.send(pgn);
    };

    const connectToServer = () => {
        const ws = new WebSocket("ws://localhost:8765");
        ws.onmessage = onMessageReceived;
        ws.onopen = onConnected;
        setServer(ws);
    };

    const pieceMoved = (fen: string) => {
        setBoardPos(fen);
        if (!game) return;
        const re = /^\s*(?:\[[^\]]*]\s*)*(\d+\.\s[\s\S]*?)(?:\s(?:1-0|0-1|1\/2-1\/2|\*)\s*)?$/;
        const match = game.pgn().match(re);
        if (!match?.[1]) return;
        sendMessage(match[1]);
    };

    return (
        <div className="flex flex-col w-full h-full">
            <div className="w-full flex flex-row justify-center">
                <h1 className="text-3xl">Play with chessington!</h1>
            </div>
            <div className="w-full flex h-full flex-col justify-center">

                <button onClick={connectToServer}>Connect</button>
                <div className="w-full  flex flex-row h-full justify-center">
                    <div className="flex flex-row w-1/5 flex-3  justify-center ">
                        <Board currentGame={game} pieceMoved={pieceMoved} boardPos={boardPos} />
                    </div>
                    <div className="flex h-full flex-col justify-center w-1/4 ">
                        <div className="flex flex-row justify-center w-full h-3/4">
                            <MoveBoard currentGame={game} startGame={startChessGame} boardPos={boardPos} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
