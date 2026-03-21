import { useState } from "react";
import type {
    BenchmarkSummary,
    ModelOption,
} from "../types";

const ProgressBar = ({ completed, total }: { completed: number; total: number }) => {
    const pct = total > 0 ? (completed / total) * 100 : 0;
    return (
        <div className="w-full bg-stone-800 rounded-full h-2 overflow-hidden">
            <div
                className="h-full bg-purple-500 transition-all duration-300"
                style={{ width: `${pct}%` }}
            />
        </div>
    );
};

const StatCard = ({ label, value, sub }: { label: string; value: string | number; sub?: string }) => (
    <div className="bg-stone-800/50 rounded-lg p-3 text-center">
        <div className="text-lg font-semibold text-stone-100">{value}</div>
        <div className="text-[10px] uppercase tracking-wider text-stone-500">{label}</div>
        {sub && <div className="text-xs text-stone-400 mt-0.5">{sub}</div>}
    </div>
);

const BreakdownTable = ({
    entries,
}: {
    entries: [string, { count: number; correct: number; total_predictions: number; correct_predictions: number }][];
}) => {
    if (entries.length === 0) return null;

    return (
        <div className="grid grid-cols-[1fr_auto_auto] gap-x-3 gap-y-0.5 text-xs">
            <span className="text-stone-500">Range</span>
            <span className="text-stone-500 text-right">Solved</span>
            <span className="text-stone-500 text-right">Move Acc</span>
            {entries.map(([label, bucket]) => {
                const solvedPct = bucket.count > 0
                    ? Math.round((bucket.correct / bucket.count) * 100)
                    : 0;
                const movePct = bucket.total_predictions > 0
                    ? Math.round((bucket.correct_predictions / bucket.total_predictions) * 100)
                    : 0;
                return (
                    <div key={label} className="contents">
                        <span className="text-stone-300 font-mono">{label}</span>
                        <span className="text-right text-stone-400">
                            {bucket.correct}/{bucket.count} ({solvedPct}%)
                        </span>
                        <span className="text-right text-stone-400">{movePct}%</span>
                    </div>
                );
            })}
        </div>
    );
};

const BenchmarkPanel = ({
    running,
    completed,
    total,
    summary,
    availableModels,
    selectedModel,
    onSelectModel,
    onStart,
    onStop,
}: {
    running: boolean;
    completed: number;
    total: number;
    summary: BenchmarkSummary | null;
    availableModels: ModelOption[];
    selectedModel: string;
    onSelectModel: (name: string) => void;
    onStart: (puzzleCount?: number) => void;
    onStop: () => void;
}) => {
    const [puzzleCount, setPuzzleCount] = useState<string>("");
    const [breakdownView, setBreakdownView] = useState<"rating" | "theme">("rating");

    return (
        <div className="flex flex-col h-full w-full bg-stone-900 rounded-xl border border-stone-800 overflow-hidden">
            <div className="px-4 py-3 border-b border-stone-800">
                <div className="text-sm font-medium text-stone-200 mb-3">Puzzle Benchmark</div>

                {!running && (
                    <div className="flex flex-col gap-3">
                        <label className="text-xs text-stone-400 flex flex-col gap-1">
                            Model
                            <select
                                className="px-2 py-1.5 rounded-md bg-stone-800 border border-stone-700 text-stone-200 text-sm focus:outline-none focus:border-stone-500 transition-colors"
                                onChange={(e) => onSelectModel(e.target.value)}
                                value={selectedModel}
                            >
                                <option value="">Select model</option>
                                {availableModels.map((m) => (
                                    <option key={m.name} value={m.name}>
                                        {m.name}
                                        {m.is_sft ? " (SFT)" : ""}
                                    </option>
                                ))}
                            </select>
                        </label>
                        <label className="text-xs text-stone-400 flex flex-col gap-1">
                            Puzzle count (blank = all)
                            <input
                                className="px-2 py-1.5 rounded-md bg-stone-800 border border-stone-700 text-stone-200 text-sm focus:outline-none focus:border-stone-500 transition-colors"
                                type="number"
                                min={1}
                                placeholder="9900"
                                value={puzzleCount}
                                onChange={(e) => setPuzzleCount(e.target.value)}
                            />
                        </label>
                        <button
                            className="w-full px-4 py-2.5 text-sm font-medium rounded-lg bg-purple-600 hover:bg-purple-500 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={() => {
                                const count = puzzleCount ? parseInt(puzzleCount, 10) : undefined;
                                onStart(count && count > 0 ? count : undefined);
                            }}
                            disabled={!selectedModel}
                            type="button"
                        >
                            Run Benchmark
                        </button>
                    </div>
                )}

                {running && (
                    <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between text-xs text-stone-400">
                            <span>
                                {completed} / {total} puzzles
                            </span>
                            <span>{total > 0 ? Math.round((completed / total) * 100) : 0}%</span>
                        </div>
                        <ProgressBar completed={completed} total={total} />
                        <button
                            className="w-full px-3 py-1.5 text-xs font-medium rounded-md bg-red-900/50 border border-red-800 hover:bg-red-900 text-red-300 transition-colors"
                            onClick={onStop}
                            type="button"
                        >
                            Stop Benchmark
                        </button>
                    </div>
                )}
            </div>

            {/* Stats */}
            <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-4">
                {summary && (
                    <>
                        <div className="grid grid-cols-2 gap-2">
                            <StatCard
                                label="Move Accuracy"
                                value={`${summary.accuracy}%`}
                                sub={`${summary.correct_predictions}/${summary.total_predictions}`}
                            />
                            <StatCard
                                label="Fully Solved"
                                value={`${summary.fully_solved_pct}%`}
                                sub={`${summary.fully_solved}/${summary.total_puzzles}`}
                            />
                            <StatCard
                                label="First Move"
                                value={`${summary.first_move_pct}%`}
                                sub={`${summary.first_move_correct}/${summary.total_puzzles}`}
                            />
                            <StatCard
                                label="Puzzles Done"
                                value={summary.total_puzzles}
                            />
                        </div>
                        <div className="space-y-2">
                            <div className="flex gap-1">
                                <button
                                    className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                                        breakdownView === "rating"
                                            ? "bg-stone-700 text-stone-100"
                                            : "text-stone-400 hover:text-stone-200 hover:bg-stone-800"
                                    }`}
                                    onClick={() => setBreakdownView("rating")}
                                    type="button"
                                >
                                    By Rating
                                </button>
                                <button
                                    className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                                        breakdownView === "theme"
                                            ? "bg-stone-700 text-stone-100"
                                            : "text-stone-400 hover:text-stone-200 hover:bg-stone-800"
                                    }`}
                                    onClick={() => setBreakdownView("theme")}
                                    type="button"
                                >
                                    By Theme
                                </button>
                            </div>
                            {breakdownView === "rating" && (
                                <BreakdownTable
                                    entries={Object.entries(summary.by_rating).sort(
                                        (a, b) => parseInt(a[0]) - parseInt(b[0]),
                                    )}
                                />
                            )}
                            {breakdownView === "theme" && (
                                <BreakdownTable
                                    entries={Object.entries(summary.by_theme ?? {}).sort(
                                        (a, b) => b[1].count - a[1].count,
                                    )}
                                />
                            )}
                        </div>
                    </>
                )}

                {!summary && !running && (
                    <div className="text-sm text-stone-500 text-center py-8">
                        Select a model and run the benchmark to see results.
                    </div>
                )}
            </div>
        </div>
    );
};

export default BenchmarkPanel;
