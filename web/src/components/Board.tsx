import { Chessboard, type ChessboardOptions, type PieceDropHandlerArgs } from "react-chessboard";
import ErrorBoundary from "./ErrorBoundary";
import type { Chess } from "chess.js";
import { DEFAULT_FEN } from "../types";

const Board = ({
    currentGame,
    pieceMoved,
    boardPos,
    canUserMove,
}: {
    currentGame: Chess | null;
    pieceMoved: (fen: string) => void;
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

    const boardSetup: ChessboardOptions = {
        position: currentGame ? boardPos : DEFAULT_FEN,
        onPieceDrop: pieceDropped,
        allowDragging: canUserMove,
    };

    return (
        <ErrorBoundary>
            <Chessboard options={boardSetup} />
        </ErrorBoundary>
    );
};

export default Board;
