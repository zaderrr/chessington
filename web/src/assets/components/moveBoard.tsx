import type { Chess } from "chess.js";
import { useEffect, useRef } from "react";
import ErrorBoundary from "./ErrorBoundary";
const MoveBoard = ({
    currentGame,
    startGame,
    boardPos,
}: {
    currentGame: Chess | null;
    startGame: () => void;
    boardPos: string;
}) => {
    const movesContainerRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (!movesContainerRef.current) return;
        movesContainerRef.current.scrollTop = movesContainerRef.current.scrollHeight;
    }, [boardPos]);

    const exportPgn = () => {
        if (!currentGame) return;
        const pgn = currentGame.pgn();
        if (!pgn.trim()) return;

        const blob = new Blob([pgn], { type: "application/x-chess-pgn" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "game.pgn";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    };

    const buildMoveHistory = () => {
        if (!currentGame) {
            return;
        }

        const history = currentGame.history();
        if (history.length === 0) {
            return (
                <div className="w-full p-3 text-stone-400 font-mono text-sm">
                    Waiting for first move...
                </div>
            );
        }

        const rows = [];
        for (let i = 0; i < history.length; i += 2) {
            const moveNumber = Math.floor(i / 2) + 1;
            const whiteMove = history[i];
            const blackMove = history[i + 1] ?? "";
            rows.push(
                <div
                    className="grid grid-cols-[3rem_1fr_1fr] gap-2 px-3 py-1 font-mono text-sm odd:bg-stone-800/50"
                    key={i}
                >
                    <span className="text-stone-400">{moveNumber}.</span>
                    <span className="text-stone-100">{whiteMove}</span>
                    <span className="text-stone-300">{blackMove}</span>
                </div>
            );
        }

        return (
            <div className="w-full h-full overflow-y-auto">
                <div className="grid grid-cols-[3rem_1fr_1fr] gap-2 px-3 py-2 border-b border-stone-700 text-xs uppercase tracking-wide text-stone-400">
                    <span>#</span>
                    <span>White</span>
                    <span>Black</span>
                </div>
                {rows}
            </div>
        );
    };
    return (
        <ErrorBoundary>
            <div className="flex flex-col justify-start h-full w-full ">
                <div className="flex flex-col justify-center h-full">
                    <div className="flex flex-row justify-center h-full">
                        <div className="flex flex-col justify-start w-11/12 h-full bg-stone-900 rounded-sm">
                            {!currentGame && (
                                <div className="flex flex-row justify-center w-full">
                                    <label className="bg-black rounded-sm w-1/3 flex flex-row justify-center">
                                        <input onClick={startGame} type="button" />
                                        Start game
                                    </label>
                                </div>
                            )}
                            {currentGame && (
                                <div className="flex flex-row justify-end p-2 border-b border-stone-700">
                                    <button
                                        className="px-3 py-1 text-xs rounded bg-stone-700 hover:bg-stone-600 text-stone-100"
                                        onClick={exportPgn}
                                        type="button"
                                    >
                                        Export PGN
                                    </button>
                                </div>
                            )}
                            <div className="flex-1 min-h-0" ref={movesContainerRef}>
                                {currentGame && buildMoveHistory()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </ErrorBoundary>
    );
};

export default MoveBoard;
