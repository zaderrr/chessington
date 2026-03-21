import type { BenchmarkPuzzleResult } from "../types";

const roleLabel = (role: string) => {
    if (role === "setup") return "Setup";
    if (role === "prediction") return "Model";
    return "Opp";
};

const PuzzleMoves = ({
    result,
    selectedIndex,
    onSelectMove,
}: {
    result: BenchmarkPuzzleResult;
    selectedIndex: number | null;
    onSelectMove: (index: number) => void;
}) => {
    return (
        <div className="flex flex-col h-full bg-stone-900 rounded-xl border border-stone-800 overflow-hidden">
            <div className="px-4 py-2.5 border-b border-stone-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-stone-200">
                        Puzzle {result.id}
                    </span>
                    <span className="text-xs text-stone-500">Elo {result.rating}</span>
                </div>
                <span
                    className={`text-xs font-semibold px-2 py-0.5 rounded ${
                        result.correct_predictions === result.total_predictions
                            ? "bg-emerald-900/50 text-emerald-300"
                            : result.correct_predictions > 0
                              ? "bg-yellow-900/50 text-yellow-300"
                              : "bg-red-900/50 text-red-300"
                    }`}
                >
                    {result.depth}
                </span>
            </div>

            <div className="flex-1 min-h-0 overflow-y-auto">
                {/* Starting position */}
                <button
                    className={`w-full text-left px-4 py-1.5 text-xs transition-colors ${
                        selectedIndex === -1
                            ? "bg-stone-700/50"
                            : "hover:bg-stone-800/50"
                    }`}
                    onClick={() => onSelectMove(-1)}
                    type="button"
                >
                    <span className="text-stone-500">Start position</span>
                </button>

                {result.sequence.map((move, i) => {
                    const isSelected = selectedIndex === i;
                    const isPrediction = move.role === "prediction";

                    return (
                        <button
                            key={i}
                            className={`w-full text-left px-4 py-1.5 text-xs font-mono transition-colors flex items-center gap-2 ${
                                isSelected
                                    ? "bg-stone-700/50"
                                    : "hover:bg-stone-800/50"
                            }`}
                            onClick={() => onSelectMove(i)}
                            type="button"
                        >
                            {/* Role badge */}
                            <span
                                className={`w-11 shrink-0 text-center text-[10px] rounded px-1 py-0.5 ${
                                    move.role === "setup"
                                        ? "bg-stone-700 text-stone-300"
                                        : isPrediction
                                          ? move.correct
                                              ? "bg-emerald-900/60 text-emerald-300"
                                              : "bg-red-900/60 text-red-300"
                                          : "bg-stone-800 text-stone-400"
                                }`}
                            >
                                {roleLabel(move.role)}
                            </span>

                            {/* Side indicator */}
                            <span
                                className={`w-3 h-3 rounded-sm shrink-0 border ${
                                    move.side === "w"
                                        ? "bg-stone-100 border-stone-300"
                                        : "bg-stone-800 border-stone-600"
                                }`}
                            />

                            {/* Expected move */}
                            <span className="text-stone-200">{move.san}</span>

                            {/* Prediction comparison */}
                            {isPrediction && (
                                <span className="ml-auto flex items-center gap-1.5">
                                    {move.correct ? (
                                        <span className="text-emerald-400">= {move.predicted}</span>
                                    ) : (
                                        <span className="text-red-400">
                                            got {move.predicted ?? "null"}
                                        </span>
                                    )}
                                </span>
                            )}
                        </button>
                    );
                })}
            </div>

            {result.themes.length > 0 && (
                <div className="px-4 py-2 border-t border-stone-800 flex flex-wrap gap-1">
                    {result.themes.map((theme) => (
                        <span
                            key={theme}
                            className="text-[10px] px-1.5 py-0.5 rounded bg-stone-800 text-stone-400"
                        >
                            {theme}
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
};

export default PuzzleMoves;
