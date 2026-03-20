import { useState } from "react";
import type { SessionMatch } from "../types";

const SessionStats = ({
    matches,
    onReset,
}: {
    matches: SessionMatch[];
    onReset: () => void;
}) => {
    const [showAll, setShowAll] = useState(false);

    const wins = matches.filter((m) => m.result === "W").length;
    const losses = matches.filter((m) => m.result === "L").length;
    const draws = matches.filter((m) => m.result === "D").length;
    const displayed = showAll ? matches.slice().reverse() : matches.slice(-20).reverse();

    return (
        <div className="rounded-lg border border-stone-700/50 bg-stone-800/30 p-3">
            <div className="flex items-center justify-between mb-2">
                <div className="flex gap-3 text-xs font-medium">
                    <span className="text-emerald-400">W {wins}</span>
                    <span className="text-red-400">L {losses}</span>
                    <span className="text-stone-400">D {draws}</span>
                </div>
                <button
                    className="px-2 py-1 text-xs rounded-md bg-stone-700/50 hover:bg-stone-700 text-stone-300 transition-colors"
                    onClick={onReset}
                    type="button"
                >
                    Reset
                </button>
            </div>
            <div className="max-h-28 overflow-y-auto text-xs text-stone-400 space-y-0.5">
                {matches.length === 0 && <div className="text-stone-500">No completed games yet.</div>}
                {displayed.map((entry) => (
                    <div
                        key={`${entry.index}-${entry.pgnResult}`}
                        className={`py-0.5 ${entry.result === "W" ? "text-emerald-400/80" : entry.result === "L" ? "text-red-400/80" : "text-stone-500"}`}
                    >
                        #{entry.index} {entry.result} ({entry.chessingtonColor}, {entry.pgnResult})
                    </div>
                ))}
                {matches.length > 20 && (
                    <button
                        className="mt-1 text-xs text-stone-500 hover:text-stone-300 underline transition-colors"
                        onClick={() => setShowAll((v) => !v)}
                        type="button"
                    >
                        {showAll ? "Show latest 20" : `Show all ${matches.length}`}
                    </button>
                )}
            </div>
        </div>
    );
};

export default SessionStats;
