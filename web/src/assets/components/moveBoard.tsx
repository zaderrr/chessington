import type { Chess } from "chess.js";
import { useEffect, useRef, useState } from "react";
import ErrorBoundary from "./ErrorBoundary";
const MoveBoard = ({
    currentGame,
    startHumanGame,
    startEngineMatch,
    boardPos,
    moveHistory,
    status,
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
    onRefreshModels,
    onSelectModel,
}: {
    currentGame: Chess | null;
    startHumanGame: () => void;
    startEngineMatch: () => void;
    boardPos: string;
    moveHistory: string[];
    status: string;
    gameMode: "human-vs-chessington" | "chessington-vs-stockfish";
    whitePlayer: string;
    blackPlayer: string;
    stockfishConfig: { movetime: number; nodes: number; depth: number };
    onStockfishConfigChange: (config: { movetime: number; nodes: number; depth: number }) => void;
    autoPlay: boolean;
    onAutoPlayChange: (autoPlay: boolean) => void;
    sessionMatches: Array<{ index: number; chessingtonColor: "White" | "Black"; result: "W" | "L" | "D"; pgnResult: string }>;
    onResetSessionMatches: () => void;
    availableModels: Array<{ name: string; checkpoint_path: string; is_sft?: boolean }>;
    selectedModel: string;
    onRefreshModels: () => void;
    onSelectModel: (modelName: string) => void;
}) => {
    const movesContainerRef = useRef<HTMLDivElement | null>(null);
    const [showAllSessionGames, setShowAllSessionGames] = useState(false);

    useEffect(() => {
        if (!movesContainerRef.current) return;
        movesContainerRef.current.scrollTop = movesContainerRef.current.scrollHeight;
    }, [boardPos, moveHistory.length]);

    const exportPgn = () => {
        if (!currentGame) return;
        const pgn = currentGame.pgn();
        if (!pgn.trim()) return;

        const blob = new Blob([pgn], { type: "application/x-chess-pgn" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        const fileName = `${whitePlayer.toLowerCase()}-vs-${blackPlayer.toLowerCase()}.pgn`.replaceAll(" ", "-");
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    };

    const buildMoveHistory = () => {
        if (!currentGame) {
            return;
        }

        if (moveHistory.length === 0) {
            return (
                <div className="w-full p-3 text-stone-400 font-mono text-sm">
                    Waiting for first move...
                </div>
            );
        }

        const rows = [];
        for (let i = 0; i < moveHistory.length; i += 2) {
            const moveNumber = Math.floor(i / 2) + 1;
            const whiteMove = moveHistory[i];
            const blackMove = moveHistory[i + 1] ?? "";
            rows.push(
                <div
                    className="grid grid-cols-[3rem_1fr_1fr] gap-2 px-3 py-1 font-mono text-sm odd:bg-stone-800/50"
                    key={i}
                >
                    <span className="text-stone-400">{moveNumber}.</span>
                    <span className="text-stone-100">{whiteMove}</span>
                    <span className="text-stone-300">{blackMove}</span>
                </div>
            );
        }

        return (
            <div className="w-full h-full overflow-y-auto">
                <div className="grid grid-cols-[3rem_1fr_1fr] gap-2 px-3 py-2 border-b border-stone-700 text-xs uppercase tracking-wide text-stone-400">
                    <span>#</span>
                    <span>White</span>
                    <span>Black</span>
                </div>
                {rows}
            </div>
        );
    };

    const updateStockfishValue = (field: "movetime" | "nodes" | "depth", value: string) => {
        const parsed = Number.parseInt(value, 10);
        const safeValue = Number.isNaN(parsed) ? 0 : Math.max(0, parsed);
        onStockfishConfigChange({ ...stockfishConfig, [field]: safeValue });
    };

    const wins = sessionMatches.filter((entry) => entry.result === "W").length;
    const losses = sessionMatches.filter((entry) => entry.result === "L").length;
    const draws = sessionMatches.filter((entry) => entry.result === "D").length;
    const displayedSessionMatches = showAllSessionGames
        ? sessionMatches.slice().reverse()
        : sessionMatches.slice(-20).reverse();
    const hiddenSessionMatchCount = Math.max(0, sessionMatches.length - displayedSessionMatches.length);

    return (
        <ErrorBoundary>
            <div className="flex flex-col justify-start h-full w-full ">
                <div className="flex flex-col justify-center h-full">
                    <div className="flex flex-row justify-center h-full">
                        <div className="flex flex-col justify-start w-11/12 h-full bg-stone-900 rounded-sm">
                            {!currentGame && (
                                <div className="flex flex-col gap-2 p-3">
                                    <button
                                        className="px-3 py-2 text-sm rounded bg-stone-700 hover:bg-stone-600 text-stone-100"
                                        onClick={startHumanGame}
                                        type="button"
                                    >
                                        Play vs Chessington
                                    </button>
                                    <button
                                        className="px-3 py-2 text-sm rounded bg-emerald-700 hover:bg-emerald-600 text-emerald-50"
                                        onClick={startEngineMatch}
                                        type="button"
                                    >
                                        Watch Chessington vs Stockfish
                                    </button>
                                    <div className="text-xs text-stone-400">{status}</div>
                                    <div className="grid grid-cols-[1fr_auto] gap-2 items-end">
                                        <label className="text-xs text-stone-300 flex flex-col gap-1">
                                            Chessington model
                                            <select
                                                className="px-2 py-1 rounded bg-stone-800 border border-stone-700"
                                                onChange={(event_) => {
                                                    onSelectModel(event_.target.value);
                                                }}
                                                value={selectedModel}
                                            >
                                                <option value="">Select model</option>
                                                {availableModels.map((model) => (
                                                    <option key={model.name} value={model.name}>
                                                        {model.name}{model.is_sft ? " (SFT)" : ""}
                                                    </option>
                                                ))}
                                            </select>
                                        </label>
                                        <button
                                            className="px-3 py-2 text-xs rounded bg-stone-700 hover:bg-stone-600 text-stone-100"
                                            onClick={onRefreshModels}
                                            type="button"
                                        >
                                            Refresh models
                                        </button>
                                    </div>
                                    <div className="text-xs text-stone-400">Stockfish settings apply in watch mode.</div>
                                    <div className="grid grid-cols-3 gap-2">
                                        <label className="text-xs text-stone-300 flex flex-col gap-1">
                                            Movetime (ms)
                                            <input
                                                className="px-2 py-1 rounded bg-stone-800 border border-stone-700"
                                                min={0}
                                                onChange={(event_) => {
                                                    updateStockfishValue("movetime", event_.target.value);
                                                }}
                                                type="number"
                                                value={stockfishConfig.movetime}
                                            />
                                        </label>
                                        <label className="text-xs text-stone-300 flex flex-col gap-1">
                                            Nodes (0=off)
                                            <input
                                                className="px-2 py-1 rounded bg-stone-800 border border-stone-700"
                                                min={0}
                                                onChange={(event_) => {
                                                    updateStockfishValue("nodes", event_.target.value);
                                                }}
                                                type="number"
                                                value={stockfishConfig.nodes}
                                            />
                                        </label>
                                        <label className="text-xs text-stone-300 flex flex-col gap-1">
                                            Depth (0=off)
                                            <input
                                                className="px-2 py-1 rounded bg-stone-800 border border-stone-700"
                                                min={0}
                                                onChange={(event_) => {
                                                    updateStockfishValue("depth", event_.target.value);
                                                }}
                                                type="number"
                                                value={stockfishConfig.depth}
                                            />
                                        </label>
                                    </div>
                                </div>
                            )}
                            {currentGame && (
                                <div className="flex flex-col p-2 border-b border-stone-700 gap-2">
                                    <span className="text-xs text-stone-300">
                                        {gameMode === "human-vs-chessington"
                                            ? "Mode: You vs Chessington"
                                            : "Mode: Chessington vs Stockfish"}
                                    </span>
                                    <div className="text-xs text-stone-400">
                                        Model: {selectedModel || "(not selected)"}
                                    </div>
                                    <div className="flex flex-row items-center justify-between gap-2">
                                        <span className="text-xs text-stone-400">
                                            White: {whitePlayer} | Black: {blackPlayer}
                                        </span>
                                        <button
                                            className="px-3 py-1 text-xs rounded bg-stone-700 hover:bg-stone-600 text-stone-100"
                                            onClick={exportPgn}
                                            type="button"
                                        >
                                            Export PGN
                                        </button>
                                    </div>
                                    {gameMode === "chessington-vs-stockfish" && (
                                        <>
                                            <label className="text-xs text-stone-300 flex flex-row items-center gap-2">
                                                <input
                                                    checked={autoPlay}
                                                    onChange={(event_) => {
                                                        onAutoPlayChange(event_.target.checked);
                                                    }}
                                                    type="checkbox"
                                                />
                                                Auto-play next game and swap sides
                                            </label>
                                            <div className="grid grid-cols-3 gap-2">
                                                <label className="text-xs text-stone-300 flex flex-col gap-1">
                                                    Movetime (ms)
                                                    <input
                                                        className="px-2 py-1 rounded bg-stone-800 border border-stone-700"
                                                        min={0}
                                                        onChange={(event_) => {
                                                            updateStockfishValue("movetime", event_.target.value);
                                                        }}
                                                        type="number"
                                                        value={stockfishConfig.movetime}
                                                    />
                                                </label>
                                                <label className="text-xs text-stone-300 flex flex-col gap-1">
                                                    Nodes (0=off)
                                                    <input
                                                        className="px-2 py-1 rounded bg-stone-800 border border-stone-700"
                                                        min={0}
                                                        onChange={(event_) => {
                                                            updateStockfishValue("nodes", event_.target.value);
                                                        }}
                                                        type="number"
                                                        value={stockfishConfig.nodes}
                                                    />
                                                </label>
                                                <label className="text-xs text-stone-300 flex flex-col gap-1">
                                                    Depth (0=off)
                                                    <input
                                                        className="px-2 py-1 rounded bg-stone-800 border border-stone-700"
                                                        min={0}
                                                        onChange={(event_) => {
                                                            updateStockfishValue("depth", event_.target.value);
                                                        }}
                                                        type="number"
                                                        value={stockfishConfig.depth}
                                                    />
                                                </label>
                                            </div>
                                            <div className="mt-1 rounded border border-stone-700 p-2">
                                                <div className="flex flex-row items-center justify-between">
                                                    <span className="text-xs text-stone-300">
                                                        Session W/L/D: {wins}/{losses}/{draws}
                                                    </span>
                                                    <button
                                                        className="px-2 py-1 text-xs rounded bg-stone-700 hover:bg-stone-600 text-stone-100"
                                                        onClick={onResetSessionMatches}
                                                        type="button"
                                                    >
                                                        Reset
                                                    </button>
                                                </div>
                                                <div className="mt-2 max-h-28 overflow-y-auto text-xs text-stone-400">
                                                    {sessionMatches.length === 0 && <div>No completed games yet.</div>}
                                                    {sessionMatches.length > 0 && hiddenSessionMatchCount > 0 && !showAllSessionGames && (
                                                        <div className="pb-1 text-stone-500">
                                                            Showing latest 20 ({hiddenSessionMatchCount} older hidden)
                                                        </div>
                                                    )}
                                                    {sessionMatches.length > 0 &&
                                                        displayedSessionMatches.map((entry) => (
                                                            <div className="py-0.5" key={`${entry.index}-${entry.pgnResult}`}>
                                                                Game {entry.index}: {entry.result} ({entry.chessingtonColor}, {entry.pgnResult})
                                                            </div>
                                                        ))}
                                                    {sessionMatches.length > 20 && (
                                                        <div className="pt-1">
                                                            <button
                                                                className="px-2 py-1 text-xs rounded bg-stone-800 hover:bg-stone-700 text-stone-200"
                                                                onClick={() => {
                                                                    setShowAllSessionGames((current) => !current);
                                                                }}
                                                                type="button"
                                                            >
                                                                {showAllSessionGames ? "Show latest 20" : "Show all"}
                                                            </button>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </>
                                    )}
                                </div>
                            )}
                            <div className="flex-1 min-h-0" ref={movesContainerRef}>
                                {currentGame && buildMoveHistory()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </ErrorBoundary>
    );
};

export default MoveBoard;
