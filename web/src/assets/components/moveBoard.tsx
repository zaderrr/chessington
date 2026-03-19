import type { Chess } from "chess.js";
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
    const buildMoveHistory = () => {
        if (!currentGame?.history()) {
            return;
        }
        if (currentGame.history().length == 0) {
            return <div className="flex flex-row" key={1}>1.</div>;
        }
        const rows = [];
        for (let i = 0; i < currentGame.history().length; i += 2) {
            const posAfterMove = currentGame.history({ verbose: true })[i].after;
            rows.push(
                <div className="flex flex-row justify-evenly w-1/3" key={i}>{posAfterMove.slice(-1)}.
                    <span className="text-left"><p>{currentGame.history()[i]}</p></span>
                    <span className="text-left">{currentGame.history()[i + 1]}</span>
                </div>);
        }
        return rows;
    };
    return (
        <ErrorBoundary>
            <div className="flex flex-col justify-start h-full w-full ">
                <div className="flex flex-col justify-center h-full">
                    <div className="flex flex-row justify-center h-full">
                        <div className="flex flex-col justify-center w-11/12 h-full bg-stone-900 rounded-sm">
                            {!currentGame && (
                                <div className="flex flex-row justify-center w-full">
                                    <label className="bg-black rounded-sm w-1/3 flex flex-row justify-center">
                                        <input onClick={startGame} type="button" />
                                        Start game
                                    </label>
                                </div>
                            )}
                            {currentGame && buildMoveHistory()}
                        </div>
                    </div>
                </div>
            </div>
        </ErrorBoundary>
    );
};

export default MoveBoard;
