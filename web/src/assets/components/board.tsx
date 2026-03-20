import { Chessboard, type ChessboardOptions, type PieceDropHandlerArgs } from "react-chessboard";
import ErrorBoundary from "./ErrorBoundary";
import { Chess } from "chess.js";
import { defaultGameFen } from "../../App";
const Board = ({
    currentGame,
    pieceMoved,
    boardPos,
    canUserMove,
}: {
    currentGame: Chess | null;
    pieceMoved: (string: string) => void;
    boardPos: string;
    canUserMove: boolean;
}) => {
    const pieceDropped = (args: PieceDropHandlerArgs): boolean => {
        if (!currentGame || !args.targetSquare) return false;
        const moved = currentGame.move({ from: args.sourceSquare, to: args.targetSquare });
        if (!moved) return false;
        pieceMoved(currentGame.fen());
        return true;
    };

    const currentPosition = (): string => {
        if (currentGame) return boardPos;
        return defaultGameFen;
    };

    const boardSetup: ChessboardOptions = {
        boardStyle: { width: "50%" },
        position: currentPosition(),
        onPieceDrop: pieceDropped,
        allowDragging: canUserMove,
    };

    return (
        <ErrorBoundary>
            <div className="flex flex-col justify-center w-full ">
                <div className="flex flex-row justify-center">
                    <Chessboard options={boardSetup} />
                </div>
            </div>
        </ErrorBoundary>
    );
};

export default Board;
