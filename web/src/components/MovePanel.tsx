import type { Chess } from "chess.js";
import type { GameMode, ModelOption, SessionMatch, StockfishConfig } from "../types";
import StockfishInputs from "./StockfishInputs";
import SessionStats from "./SessionStats";
import MoveHistory from "./MoveHistory";

const MovePanel = ({
    currentGame,
    startHumanGame,
    startEngineMatch,
    moveHistory,
    gameMode,
    whitePlayer,
    blackPlayer,
    stockfishConfig,
    onStockfishConfigChange,
    autoPlay,
    onAutoPlayChange,
    sessionMatches,
    onResetSessionMatches,
    availableModels,
    selectedModel,
    onSelectModel,
    onExportPgn,
}: {
    currentGame: Chess | null;
    startHumanGame: () => void;
    startEngineMatch: () => void;
    moveHistory: string[];
    gameMode: GameMode;
    whitePlayer: string;
    blackPlayer: string;
    stockfishConfig: StockfishConfig;
    onStockfishConfigChange: (config: StockfishConfig) => void;
    autoPlay: boolean;
    onAutoPlayChange: (autoPlay: boolean) => void;
    sessionMatches: SessionMatch[];
    onResetSessionMatches: () => void;
    availableModels: ModelOption[];
    selectedModel: string;
    onSelectModel: (name: string) => void;
    onExportPgn: () => void;
}) => {
    return (
        <div className="flex flex-col h-full w-full bg-stone-900 rounded-xl border border-stone-800 overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 border-b border-stone-800">
                {!currentGame ? (
                    <div className="flex flex-col gap-3">
                        <div className="flex flex-col gap-2">
                            <button
                                className="w-full px-4 py-2.5 text-sm font-medium rounded-lg bg-stone-700 hover:bg-stone-600 text-stone-100 transition-colors"
                                onClick={startHumanGame}
                                type="button"
                            >
                                Play vs Chessington
                            </button>
                            <button
                                className="w-full px-4 py-2.5 text-sm font-medium rounded-lg bg-emerald-700 hover:bg-emerald-600 text-emerald-50 transition-colors"
                                onClick={startEngineMatch}
                                type="button"
                            >
                                Watch Chessington vs Stockfish
                            </button>
                        </div>

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

                        <div className="pt-1">
                            <div className="text-xs text-stone-500 mb-2">Stockfish settings (watch mode)</div>
                            <StockfishInputs config={stockfishConfig} onChange={onStockfishConfigChange} />
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-stone-200">
                                {gameMode === "human-vs-chessington"
                                    ? "You vs Chessington"
                                    : "Chessington vs Stockfish"}
                            </span>
                            <button
                                className="px-3 py-1 text-xs rounded-md bg-stone-800 border border-stone-700 hover:bg-stone-700 text-stone-300 transition-colors"
                                onClick={onExportPgn}
                                type="button"
                            >
                                Export PGN
                            </button>
                        </div>
                        <div className="text-xs text-stone-500">
                            Model: {selectedModel || "none"} &middot; {whitePlayer} vs {blackPlayer}
                        </div>

                        {gameMode === "chessington-vs-stockfish" && (
                            <div className="flex flex-col gap-2 pt-1">
                                <label className="text-xs text-stone-300 flex items-center gap-2 cursor-pointer">
                                    <input
                                        checked={autoPlay}
                                        onChange={(e) => onAutoPlayChange(e.target.checked)}
                                        type="checkbox"
                                        className="rounded border-stone-600"
                                    />
                                    Auto-play & swap sides
                                </label>
                                <StockfishInputs config={stockfishConfig} onChange={onStockfishConfigChange} />
                                <SessionStats matches={sessionMatches} onReset={onResetSessionMatches} />
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Move History */}
            <div className="flex-1 min-h-0">
                {currentGame && <MoveHistory moveHistory={moveHistory} />}
            </div>
        </div>
    );
};

export default MovePanel;
