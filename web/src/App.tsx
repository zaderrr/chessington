import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import Board from "./components/Board";
import MovePanel from "./components/MovePanel";
import BenchmarkPanel from "./components/BenchmarkPanel";
import PuzzleMoves from "./components/PuzzleMoves";
import { useChessServer } from "./hooks/useChessServer";
import {
    DEFAULT_FEN,
    type GameMode,
    type SessionMatch,
    type StockfishConfig,
    type ServerMessage,
    type BenchmarkSummary,
    type BenchmarkPuzzleResult,
} from "./types";

type AppView = "play" | "benchmark";

const MAX_RETRIES = 5;

const getPlayers = (mode: GameMode, chessingtonColor: "w" | "b") => {
    if (mode === "human-vs-chessington") return { white: "You", black: "Chessington" };
    return chessingtonColor === "w"
        ? { white: "Chessington", black: "Stockfish" }
        : { white: "Stockfish", black: "Chessington" };
};

const getGameResult = (game: Chess): string => {
    if (game.isCheckmate()) return game.turn() === "w" ? "0-1" : "1-0";
    if (game.isDraw()) return "1/2-1/2";
    return "*";
};

const getChessingtonResult = (pgn: string, color: "w" | "b"): "W" | "L" | "D" => {
    if (pgn === "1/2-1/2") return "D";
    if (pgn === "1-0") return color === "w" ? "W" : "L";
    if (pgn === "0-1") return color === "b" ? "W" : "L";
    return "D";
};

function App() {
    const [view, setView] = useState<AppView>("play");
    const [game, setGame] = useState<Chess | null>(null);
    const [boardPos, setBoardPos] = useState(DEFAULT_FEN);
    const [moveHistory, setMoveHistory] = useState<string[]>([]);
    const [gameMode, setGameMode] = useState<GameMode>("human-vs-chessington");
    const [awaitingEngine, setAwaitingEngine] = useState(false);
    const [chessingtonColor, setChessingtonColor] = useState<"w" | "b">("w");
    const [autoPlay, setAutoPlay] = useState(true);
    const [sessionMatches, setSessionMatches] = useState<SessionMatch[]>([]);
    const [stockfishConfig, setStockfishConfig] = useState<StockfishConfig>({
        movetime: 350,
        nodes: 0,
        depth: 0,
    });

    // Benchmark state
    const [benchmarkRunning, setBenchmarkRunning] = useState(false);
    const [benchmarkCompleted, setBenchmarkCompleted] = useState(0);
    const [benchmarkTotal, setBenchmarkTotal] = useState(0);
    const [benchmarkSummary, setBenchmarkSummary] = useState<BenchmarkSummary | null>(null);
    const [benchmarkLatest, setBenchmarkLatest] = useState<BenchmarkPuzzleResult | null>(null);
    const [benchmarkFen, setBenchmarkFen] = useState<string | null>(null);
    const [puzzleMoveIndex, setPuzzleMoveIndex] = useState<number | null>(null);

    const gameRef = useRef<Chess | null>(null);
    const retryCountRef = useRef(0);
    const autoPlayTimerRef = useRef<number | null>(null);
    const lastSavedPgnRef = useRef("");
    const lastRecordedRef = useRef("");

    const getFirstMove = (raw: string): string | null => {
        const cleaned = raw.replace(/^\s*(?:\[[^\]]*\]\s*)*/, "").trim();
        const token = cleaned.replace(/^\d+\.(?:\.\.)?\s*/, "").match(/^([^\s{}()]+)/)?.[1];
        if (!token || ["*", "1-0", "0-1", "1/2-1/2"].includes(token)) return null;
        return token;
    };

    const moveFromUci = (uci: string) => {
        const m = uci.match(/^([a-h][1-8])([a-h][1-8])([qrbn])?$/i);
        if (!m) return null;
        return { from: m[1], to: m[2], promotion: m[3]?.toLowerCase() as "q" | "r" | "b" | "n" | undefined };
    };

    const applyMove = (rawMove: string): boolean => {
        if (!gameRef.current) return false;
        let moved = gameRef.current.move(rawMove);
        if (!moved) {
            const parsed = moveFromUci(rawMove);
            if (parsed) moved = gameRef.current.move(parsed);
        }
        if (!moved) {
            console.warn(`Invalid move: ${rawMove}`);
            return false;
        }
        setBoardPos(gameRef.current.fen());
        setMoveHistory(gameRef.current.history());
        return true;
    };

    const getMovetext = (): string => {
        if (!gameRef.current) return "1. ";
        const re = /^\s*(?:\[[^\]]*]\s*)*(\d+\.\s[\s\S]*?)(?:\s(?:1-0|0-1|1\/2-1\/2|\*)\s*)?$/;
        return gameRef.current.pgn().match(re)?.[1] ?? "1. ";
    };

    const onServerMessage = useCallback((payload: ServerMessage) => {
        // Benchmark messages
        if (payload.type === "benchmark_started") {
            setBenchmarkRunning(true);
            setBenchmarkCompleted(0);
            setBenchmarkTotal(payload.puzzle_count ?? 0);
            setBenchmarkSummary(null);
            setBenchmarkLatest(null);
            return;
        }
        if (payload.type === "benchmark_progress") {
            setBenchmarkCompleted(payload.completed ?? 0);
            setBenchmarkTotal(payload.total ?? 0);
            if (payload.summary) setBenchmarkSummary(payload.summary);
            if (payload.latest_result) {
                setBenchmarkLatest(payload.latest_result);
                setPuzzleMoveIndex(null);
            }
            if (payload.current_fen) setBenchmarkFen(payload.current_fen);
            return;
        }
        if (payload.type === "benchmark_complete") {
            setBenchmarkRunning(false);
            if (payload.summary) setBenchmarkSummary(payload.summary);
            return;
        }
        if (payload.type === "benchmark_stopping") {
            return;
        }

        // Game messages
        if (payload.type === "error") {
            setAwaitingEngine(false);
            return;
        }

        if (!payload.move) return;
        const move =
            payload.type === "chessington_move"
                ? (getFirstMove(payload.move) ?? payload.move)
                : payload.move;

        if (!applyMove(move)) {
            if (payload.type === "chessington_move" && retryCountRef.current < MAX_RETRIES) {
                retryCountRef.current += 1;
                server.setStatus(
                    `Invalid move, retrying (${retryCountRef.current}/${MAX_RETRIES})`,
                );
                server.requestChessingtonMove(getMovetext());
                return;
            }
            setAwaitingEngine(false);
            return;
        }

        retryCountRef.current = 0;
        setAwaitingEngine(false);
        server.setStatus(payload.type === "stockfish_bestmove" ? "Stockfish moved" : "Chessington moved");
    }, []);

    const server = useChessServer(onServerMessage);

    const startGame = (mode: GameMode, forcedColor?: "w" | "b") => {
        if (autoPlayTimerRef.current) {
            window.clearTimeout(autoPlayTimerRef.current);
            autoPlayTimerRef.current = null;
        }

        const color = mode === "chessington-vs-stockfish" ? (forcedColor ?? chessingtonColor) : "b";
        if (mode === "chessington-vs-stockfish") setChessingtonColor(color);

        const newGame = new Chess(DEFAULT_FEN);
        const players = getPlayers(mode, color);
        newGame.header(
            "Event", "Chessington Arena",
            "Site", "Local",
            "White", players.white,
            "Black", players.black,
            "Date", new Date().toISOString().slice(0, 10).replaceAll("-", "."),
        );

        setGame(newGame);
        setMoveHistory([]);
        setGameMode(mode);
        setAwaitingEngine(false);
        setBoardPos(newGame.fen());
        gameRef.current = newGame;
        retryCountRef.current = 0;
        lastSavedPgnRef.current = "";
        lastRecordedRef.current = "";
        server.setStatus(
            mode === "human-vs-chessington" ? "Your move" : "Chessington vs Stockfish",
        );
    };

    const pieceMoved = (fen: string) => {
        setBoardPos(fen);
        if (gameRef.current) setMoveHistory(gameRef.current.history());
        if (!gameRef.current || gameMode !== "human-vs-chessington") return;
        if (gameRef.current.turn() === "b") {
            setAwaitingEngine(true);
            server.requestChessingtonMove(getMovetext());
        }
    };

    const exportPgn = () => {
        if (!game) return;
        const pgn = game.pgn();
        if (!pgn.trim()) return;
        const blob = new Blob([pgn], { type: "application/x-chess-pgn" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${getPlayers(gameMode, chessingtonColor).white.toLowerCase()}-vs-${getPlayers(gameMode, chessingtonColor).black.toLowerCase()}.pgn`.replaceAll(" ", "-");
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };

    const recordMatch = (result: string) => {
        if (!gameRef.current || gameMode !== "chessington-vs-stockfish") return;
        const pgn = gameRef.current.pgn();
        if (!pgn || pgn === lastRecordedRef.current) return;
        setSessionMatches((prev) => [
            ...prev,
            {
                index: prev.length + 1,
                chessingtonColor: chessingtonColor === "w" ? "White" : "Black",
                result: getChessingtonResult(result, chessingtonColor),
                pgnResult: result,
            },
        ]);
        lastRecordedRef.current = pgn;
    };

    // Cleanup
    useEffect(() => {
        return () => {
            if (autoPlayTimerRef.current) window.clearTimeout(autoPlayTimerRef.current);
        };
    }, []);

    // Engine game loop
    useEffect(() => {
        if (gameMode !== "chessington-vs-stockfish") return;
        if (!gameRef.current || awaitingEngine) return;

        if (gameRef.current.isGameOver()) {
            const result = getGameResult(gameRef.current);
            gameRef.current.setHeader("Result", result);
            recordMatch(result);

            const pgn = gameRef.current.pgn();
            if (pgn.trim() && pgn !== lastSavedPgnRef.current) {
                const players = getPlayers(gameMode, chessingtonColor);
                server.saveGame(pgn, players.white, players.black);
                lastSavedPgnRef.current = pgn;
            }

            if (autoPlay) {
                server.setStatus(`Game over (${result}). Next game...`);
                if (!autoPlayTimerRef.current) {
                    const nextColor = chessingtonColor === "w" ? "b" : "w";
                    autoPlayTimerRef.current = window.setTimeout(() => {
                        autoPlayTimerRef.current = null;
                        startGame("chessington-vs-stockfish", nextColor);
                    }, 1200);
                }
            } else {
                server.setStatus(`Game over (${result})`);
            }
            return;
        }

        setAwaitingEngine(true);
        if (gameRef.current.turn() === chessingtonColor) {
            server.requestChessingtonMove(getMovetext());
        } else {
            server.requestStockfishMove(gameRef.current.fen(), stockfishConfig);
        }
    }, [boardPos, gameMode, awaitingEngine, autoPlay, chessingtonColor]);

    const players = getPlayers(gameMode, chessingtonColor);

    // Determine which FEN to show on the board
    let displayFen = boardPos;
    if (view === "benchmark") {
        if (puzzleMoveIndex !== null && benchmarkLatest?.sequence) {
            if (puzzleMoveIndex === -1) {
                displayFen = benchmarkLatest.fen;
            } else {
                displayFen = benchmarkLatest.sequence[puzzleMoveIndex]?.fen_after ?? benchmarkLatest.fen;
            }
        } else if (benchmarkLatest?.sequence?.length) {
            // Show the final position by default
            displayFen = benchmarkLatest.sequence[benchmarkLatest.sequence.length - 1].fen_after;
        } else if (benchmarkFen) {
            displayFen = benchmarkFen;
        }
    }

    return (
        <div className="flex flex-col h-full bg-[var(--bg)]">
            {/* Header */}
            <header className="flex items-center justify-between px-6 py-3 border-b border-stone-800">
                <div className="flex items-center gap-4">
                    <h1 className="text-lg font-semibold text-stone-200 tracking-tight">
                        Chessington Arena
                    </h1>
                    <nav className="flex gap-1">
                        <button
                            className={`px-3 py-1 text-xs rounded-md transition-colors ${
                                view === "play"
                                    ? "bg-stone-700 text-stone-100"
                                    : "text-stone-400 hover:text-stone-200 hover:bg-stone-800"
                            }`}
                            onClick={() => setView("play")}
                            type="button"
                        >
                            Play
                        </button>
                        <button
                            className={`px-3 py-1 text-xs rounded-md transition-colors ${
                                view === "benchmark"
                                    ? "bg-stone-700 text-stone-100"
                                    : "text-stone-400 hover:text-stone-200 hover:bg-stone-800"
                            }`}
                            onClick={() => setView("benchmark")}
                            type="button"
                        >
                            Benchmark
                        </button>
                    </nav>
                </div>
                <div className="flex items-center gap-3">
                    <span className="text-xs text-stone-500">{server.status}</span>
                    {server.status === "Disconnected" && (
                        <button
                            className="px-3 py-1.5 text-xs font-medium rounded-md bg-purple-600 hover:bg-purple-500 text-white transition-colors"
                            onClick={server.connect}
                            type="button"
                        >
                            Connect
                        </button>
                    )}
                </div>
            </header>

            {/* Main content */}
            <div className="flex-1 flex items-center justify-center gap-6 p-6 min-h-0">
                {/* Board */}
                <div className="flex flex-col items-center gap-2">
                    {view === "play" && gameRef.current && (
                        <div className="flex gap-4 text-xs mb-1">
                            <span className="px-2.5 py-1 rounded-md bg-stone-200 text-stone-900 font-medium">
                                {players.white}
                            </span>
                            <span className="px-2.5 py-1 rounded-md bg-stone-800 text-stone-200 font-medium">
                                {players.black}
                            </span>
                            <span className="px-2.5 py-1 rounded-md border border-stone-700 text-stone-400">
                                {gameRef.current.turn() === "w" ? "White" : "Black"} to move
                            </span>
                        </div>
                    )}
                    {view === "benchmark" && benchmarkFen && (
                        <div className="flex gap-3 text-xs mb-1">
                            <span className="px-2.5 py-1 rounded-md bg-purple-900/50 border border-purple-700/50 text-purple-300">
                                Puzzle {benchmarkCompleted}/{benchmarkTotal}
                            </span>
                            {benchmarkLatest && (
                                <span
                                    className={`px-2.5 py-1 rounded-md font-medium ${
                                        benchmarkLatest.correct_predictions === benchmarkLatest.total_predictions
                                            ? "bg-emerald-900/50 text-emerald-300"
                                            : "bg-red-900/50 text-red-300"
                                    }`}
                                >
                                    Last: {benchmarkLatest.depth}
                                </span>
                            )}
                        </div>
                    )}
                    <div className="w-[min(55vh,550px)] aspect-square">
                        <Board
                            currentGame={view === "play" ? game : null}
                            pieceMoved={pieceMoved}
                            boardPos={displayFen}
                            canUserMove={
                                view === "play" &&
                                gameMode === "human-vs-chessington" &&
                                !awaitingEngine &&
                                gameRef.current?.turn() === "w"
                            }
                        />
                    </div>
                </div>

                {/* Side panels */}
                {view === "play" && (
                    <div className="w-80 h-[min(65vh,650px)] shrink-0">
                        <MovePanel
                            currentGame={game}
                            moveHistory={moveHistory}
                            gameMode={gameMode}
                            whitePlayer={players.white}
                            blackPlayer={players.black}
                            stockfishConfig={stockfishConfig}
                            onStockfishConfigChange={setStockfishConfig}
                            autoPlay={autoPlay}
                            onAutoPlayChange={setAutoPlay}
                            sessionMatches={sessionMatches}
                            onResetSessionMatches={() => {
                                setSessionMatches([]);
                                lastRecordedRef.current = "";
                            }}
                            availableModels={server.availableModels}
                            selectedModel={server.selectedModel}

                            onSelectModel={server.selectModel}
                            startHumanGame={() => startGame("human-vs-chessington")}
                            startEngineMatch={() => startGame("chessington-vs-stockfish")}
                            onExportPgn={exportPgn}
                        />
                    </div>
                )}
                {view === "benchmark" && (
                    <div className="flex gap-4 h-[min(65vh,650px)] shrink-0">
                        {benchmarkLatest && (
                            <div className="w-56">
                                <PuzzleMoves
                                    result={benchmarkLatest}
                                    selectedIndex={puzzleMoveIndex}
                                    onSelectMove={(i) => setPuzzleMoveIndex(i)}
                                />
                            </div>
                        )}
                        <div className="w-80">
                            <BenchmarkPanel
                                running={benchmarkRunning}
                                completed={benchmarkCompleted}
                                total={benchmarkTotal}
                                summary={benchmarkSummary}
                                availableModels={server.availableModels}
                                selectedModel={server.selectedModel}
                                onSelectModel={server.selectModel}
    
                                onStart={server.startBenchmark}
                                onStop={server.stopBenchmark}
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
