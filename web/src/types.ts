export const DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export type GameMode = "human-vs-chessington" | "chessington-vs-stockfish";

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

export type BenchmarkMoveResult = {
    expected: string;
    expected_uci: string;
    predicted: string | null;
    correct: boolean;
};

export type BenchmarkSequenceMove = {
    san: string;
    uci: string;
    side: "w" | "b";
    role: "setup" | "prediction" | "response";
    fen_after: string;
    predicted?: string | null;
    correct?: boolean | null;
};

export type BenchmarkPuzzleResult = {
    id: string;
    rating: number;
    themes: string[];
    fen: string;
    total_predictions: number;
    correct_predictions: number;
    depth: string;
    moves: BenchmarkMoveResult[];
    sequence: BenchmarkSequenceMove[];
};

export type BenchmarkRatingBucket = {
    count: number;
    correct: number;
    total_predictions: number;
    correct_predictions: number;
};

export type BenchmarkSummary = {
    total_puzzles: number;
    total_predictions: number;
    correct_predictions: number;
    accuracy: number;
    fully_solved: number;
    fully_solved_pct: number;
    first_move_correct: number;
    first_move_pct: number;
    by_rating: Record<string, BenchmarkRatingBucket>;
    by_theme: Record<string, BenchmarkRatingBucket>;
};

export type ServerMessage = {
    type:
        | "chessington_move"
        | "stockfish_bestmove"
        | "game_saved"
        | "model_list"
        | "model_selected"
        | "benchmark_started"
        | "benchmark_progress"
        | "benchmark_complete"
        | "benchmark_stopping"
        | "error";
    move?: string;
    message?: string;
    path?: string;
    models?: ModelOption[];
    current_model?: string;
    current_checkpoint?: string;
    current_is_sft?: boolean;
    completed?: number;
    total?: number;
    latest_result?: BenchmarkPuzzleResult;
    summary?: BenchmarkSummary;
    current_fen?: string;
    results_file?: string;
    cancelled?: boolean;
    puzzle_count?: number | null;
};
