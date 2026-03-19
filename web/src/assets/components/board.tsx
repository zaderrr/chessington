import { Chessboard, type ChessboardOptions, type PieceDropHandlerArgs } from "react-chessboard";
import ErrorBoundary from "./ErrorBoundary";
import { Chess } from "chess.js";
import { useState } from "react";
import { defaultGameFen } from "../../App";
const Board = ({
    currentGame,
    setBoardPos,
    boardPos,
}: {
    currentGame: Chess | null;
    setBoardPos: (string: string) => void;
    boardPos: string
}) => {
    const pieceDropped = (args: PieceDropHandlerArgs): boolean => {
        if (!currentGame || !args.targetSquare) return false;
        currentGame.move({ from: args.sourceSquare, to: args.targetSquare });
        setBoardPos(currentGame.fen());
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
