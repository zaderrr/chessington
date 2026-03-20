export const DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export type GameMode = "human-vs-chessington" | "chessington-vs-stockfish";

export type ServerMessage = {
    type:
        | "chessington_move"
        | "stockfish_bestmove"
        | "game_saved"
        | "model_list"
        | "model_selected"
        | "error";
    move?: string;
    message?: string;
    path?: string;
    models?: ModelOption[];
    current_model?: string;
    current_checkpoint?: string;
    current_is_sft?: boolean;
};

export type StockfishConfig = {
    movetime: number;
    nodes: number;
    depth: number;
};

export type SessionMatch = {
    index: number;
    chessingtonColor: "White" | "Black";
    result: "W" | "L" | "D";
    pgnResult: string;
};

export type ModelOption = {
    name: string;
    checkpoint_path: string;
    is_sft?: boolean;
};
