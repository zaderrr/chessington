import "./App.css";
import MoveBoard from "./assets/components/moveBoard";
import { useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import Board from "./assets/components/board";

export const defaultGameFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

type GameMode = "human-vs-chessington" | "chessington-vs-stockfish";

type ServerMessage = {
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
    models?: Array<{ name: string; checkpoint_path: string; is_sft?: boolean }>;
    current_model?: string;
    current_checkpoint?: string;
    current_is_sft?: boolean;
};

type StockfishConfig = {
    movetime: number;
    nodes: number;
    depth: number;
};

type SessionMatch = {
    index: number;
    chessingtonColor: "White" | "Black";
    result: "W" | "L" | "D";
    pgnResult: string;
};

type ModelOption = { name: string; checkpoint_path: string; is_sft?: boolean };

const maxChessingtonRetries = 5;

const getPlayersForMode = (
    mode: GameMode,
    chessingtonColor: "w" | "b",
): { white: string; black: string } => {
    if (mode === "human-vs-chessington") {
        return { white: "You", black: "Chessington" };
    }

    if (chessingtonColor === "w") {
        return { white: "Chessington", black: "Stockfish" };
    }

    return { white: "Stockfish", black: "Chessington" };
};

const getGameResult = (game: Chess): string => {
    if (game.isCheckmate()) {
        return game.turn() === "w" ? "0-1" : "1-0";
    }
    if (game.isDraw()) {
        return "1/2-1/2";
    }

    return "*";
};

const getChessingtonResult = (pgnResult: string, chessingtonColor: "w" | "b"): "W" | "L" | "D" => {
    if (pgnResult === "1/2-1/2") return "D";
    if (pgnResult === "1-0") return chessingtonColor === "w" ? "W" : "L";
    if (pgnResult === "0-1") return chessingtonColor === "b" ? "W" : "L";
    return "D";
};

function App() {
    const [game, setGame] = useState<Chess | null>(null);
    const [boardPos, setBoardPos] = useState<string>(defaultGameFen);
    const [moveHistory, setMoveHistory] = useState<string[]>([]);
    const [status, setStatus] = useState<string>("Disconnected");
    const [gameMode, setGameMode] = useState<GameMode>("human-vs-chessington");
    const [awaitingEngineMove, setAwaitingEngineMove] = useState(false);
    const [engineChessingtonColor, setEngineChessingtonColor] = useState<"w" | "b">("w");
    const [autoPlayEngineMatches, setAutoPlayEngineMatches] = useState(true);
    const [sessionMatches, setSessionMatches] = useState<SessionMatch[]>([]);
    const [stockfishConfig, setStockfishConfig] = useState<StockfishConfig>({
        movetime: 350,
        nodes: 0,
        depth: 0,
    });
    const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>("");

    const gameRef = useRef<Chess | null>(null);
    const socketRef = useRef<WebSocket | null>(null);
    const chessingtonRetryCountRef = useRef(0);
    const autoPlayTimerRef = useRef<number | null>(null);
    const lastSavedPgnRef = useRef<string>("");
    const lastRecordedResultPgnRef = useRef<string>("");

    const startChessGame = (mode: GameMode, forcedChessingtonColor?: "w" | "b") => {
        if (autoPlayTimerRef.current) {
            window.clearTimeout(autoPlayTimerRef.current);
            autoPlayTimerRef.current = null;
        }

        const selectedChessingtonColor =
            mode === "chessington-vs-stockfish"
                ? (forcedChessingtonColor ?? engineChessingtonColor)
                : "b";

        if (mode === "chessington-vs-stockfish") {
            setEngineChessingtonColor(selectedChessingtonColor);
        }

        const newGame = new Chess(defaultGameFen);
        const players = getPlayersForMode(mode, selectedChessingtonColor);
        newGame.header(
            "Event",
            "Chessington Arena",
            "Site",
            "Local",
            "White",
            players.white,
            "Black",
            players.black,
            "Date",
            new Date().toISOString().slice(0, 10).replaceAll("-", "."),
        );
        setGame(newGame);
        setMoveHistory([]);
        lastSavedPgnRef.current = "";
        lastRecordedResultPgnRef.current = "";
        setGameMode(mode);
        setAwaitingEngineMove(false);
        chessingtonRetryCountRef.current = 0;
        setStatus(mode === "human-vs-chessington" ? "New game: You play White" : "New game: Chessington vs Stockfish");
        setBoardPos(newGame.fen());
        gameRef.current = newGame;
    };

    const getFirstMoveFromPgn = (raw: string): string | null => {
        const withoutHeaders = raw.replace(/^\s*(?:\[[^\]]*\]\s*)*/, "").trim();
        const withoutLeadingMoveNumber = withoutHeaders.replace(/^\d+\.(?:\.\.)?\s*/, "");
        const firstToken = withoutLeadingMoveNumber.match(/^([^\s{}()]+)/)?.[1] ?? null;

        if (!firstToken) return null;
        if (["*", "1-0", "0-1", "1/2-1/2"].includes(firstToken)) return null;
        return firstToken;
    };

    const moveFromUci = (uci: string) => {
        const uciMatch = uci.match(/^([a-h][1-8])([a-h][1-8])([qrbn])?$/i);
        if (!uciMatch) return null;
        const [, from, to, promotion] = uciMatch;
        return { from, to, promotion: promotion?.toLowerCase() as "q" | "r" | "b" | "n" | undefined };
    };

    const applyEngineMove = (rawMove: string): boolean => {
        if (!gameRef.current) return false;
        let moved = gameRef.current.move(rawMove);
        if (!moved) {
            const parsed = moveFromUci(rawMove);
            if (parsed) {
                moved = gameRef.current.move(parsed);
            }
        }

        if (!moved) {
            console.warn(`Invalid AI move received: ${rawMove}`);
            setStatus(`Engine sent invalid move: ${rawMove}`);
            return false;
        }

        setBoardPos(gameRef.current.fen());
        setMoveHistory(gameRef.current.history());
        return true;
    };

    const parseServerMessage = (raw: string): ServerMessage | null => {
        try {
            const parsed = JSON.parse(raw) as ServerMessage;
            if (parsed && typeof parsed.type === "string") {
                return parsed;
            }
        } catch {
            const chessingtonMove = getFirstMoveFromPgn(raw);
            if (chessingtonMove) {
                return { type: "chessington_move", move: chessingtonMove };
            }
        }

        return null;
    };

    const onMessageReceived = (event: MessageEvent) => {
        if (typeof event.data !== "string") return;
        const payload = parseServerMessage(event.data);
        if (!payload) return;

        if (payload.type === "error") {
            setStatus(payload.message ?? "Engine error");
            setAwaitingEngineMove(false);
            return;
        }

        if (payload.type === "game_saved") {
            if (payload.path) {
                setStatus(`Saved PGN: ${payload.path}`);
            }
            return;
        }

        if (payload.type === "model_list") {
            const models = Array.isArray(payload.models) ? payload.models : [];
            setAvailableModels(models);
            setSelectedModel(payload.current_model ?? models[0]?.name ?? "");
            if (payload.current_model) {
                setStatus(`Connected (model: ${payload.current_model})`);
            }
            return;
        }

        if (payload.type === "model_selected") {
            if (payload.current_model) {
                setSelectedModel(payload.current_model);
                setStatus(`Model selected: ${payload.current_model}`);
            }
            return;
        }

        if (!payload.move) return;
        const normalizedMove =
            payload.type === "chessington_move"
                ? (getFirstMoveFromPgn(payload.move) ?? payload.move)
                : payload.move;
        const applied = applyEngineMove(normalizedMove);
        if (!applied) {
            if (payload.type === "chessington_move" && chessingtonRetryCountRef.current < maxChessingtonRetries) {
                chessingtonRetryCountRef.current += 1;
                setStatus(
                    `Chessington sent invalid move, retrying (${chessingtonRetryCountRef.current}/${maxChessingtonRetries})`,
                );
                setAwaitingEngineMove(true);
                requestChessingtonMove();
                return;
            }

            setAwaitingEngineMove(false);
            return;
        }

        chessingtonRetryCountRef.current = 0;
        setAwaitingEngineMove(false);
        setStatus(payload.type === "stockfish_bestmove" ? "Stockfish moved" : "Chessington moved");
    };

    const onConnected = () => {
        setStatus("Connected");
        requestModelList();
    };

    const sendMessage = (payload: object) => {
        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
            setStatus("Server disconnected");
            setAwaitingEngineMove(false);
            return;
        }

        socketRef.current.send(JSON.stringify(payload));
    };

    const connectToServer = () => {
        if (socketRef.current?.readyState === WebSocket.OPEN) return;
        const ws = new WebSocket("ws://localhost:8765");
        ws.onmessage = onMessageReceived;
        ws.onopen = onConnected;
        ws.onclose = () => {
            setStatus("Disconnected");
        };
        socketRef.current = ws;
        setStatus("Connecting...");
    };

    const requestModelList = () => {
        sendMessage({ action: "list_models" });
    };

    const selectModel = (modelName: string) => {
        if (!modelName) return;
        setStatus(`Switching model to ${modelName}...`);
        sendMessage({ action: "set_model", model_name: modelName });
    };

    const getMovetext = (): string | null => {
        if (!gameRef.current) return null;
        const re = /^\s*(?:\[[^\]]*]\s*)*(\d+\.\s[\s\S]*?)(?:\s(?:1-0|0-1|1\/2-1\/2|\*)\s*)?$/;
        const match = gameRef.current.pgn().match(re);
        return match?.[1] ?? null;
    };

    const requestChessingtonMove = () => {
        const movetext = getMovetext() ?? "1. ";
        sendMessage({ action: "chessington_move", pgn: movetext });
    };

    const requestStockfishMove = () => {
        if (!gameRef.current) {
            setAwaitingEngineMove(false);
            return;
        }

        sendMessage({
            action: "stockfish_bestmove",
            fen: gameRef.current.fen(),
            movetime: stockfishConfig.movetime,
            nodes: stockfishConfig.nodes > 0 ? stockfishConfig.nodes : undefined,
            depth: stockfishConfig.depth > 0 ? stockfishConfig.depth : undefined,
        });
    };

    const saveCurrentGamePgn = () => {
        if (!gameRef.current) return;
        const pgn = gameRef.current.pgn();
        if (!pgn.trim() || pgn === lastSavedPgnRef.current) return;

        const players = getPlayersForMode(gameMode, engineChessingtonColor);
        sendMessage({
            action: "save_game_pgn",
            pgn,
            white: players.white,
            black: players.black,
        });
        lastSavedPgnRef.current = pgn;
    };

    const recordSessionMatch = (pgnResult: string) => {
        if (!gameRef.current || gameMode !== "chessington-vs-stockfish") return;
        const pgn = gameRef.current.pgn();
        if (!pgn || pgn === lastRecordedResultPgnRef.current) return;

        setSessionMatches((previous) => [
            ...previous,
            {
                index: previous.length + 1,
                chessingtonColor: engineChessingtonColor === "w" ? "White" : "Black",
                result: getChessingtonResult(pgnResult, engineChessingtonColor),
                pgnResult,
            },
        ]);
        lastRecordedResultPgnRef.current = pgn;
    };

    const pieceMoved = (fen: string) => {
        setBoardPos(fen);
        if (gameRef.current) {
            setMoveHistory(gameRef.current.history());
        }
        if (!gameRef.current || gameMode !== "human-vs-chessington") return;
        if (gameRef.current.turn() === "b") {
            setAwaitingEngineMove(true);
            requestChessingtonMove();
        }
    };

    useEffect(() => {
        return () => {
            if (autoPlayTimerRef.current) {
                window.clearTimeout(autoPlayTimerRef.current);
            }
            socketRef.current?.close();
        };
    }, []);

    useEffect(() => {
        if (gameMode !== "chessington-vs-stockfish") return;
        if (!gameRef.current || awaitingEngineMove) return;
        if (gameRef.current.isGameOver()) {
            const result = getGameResult(gameRef.current);
            gameRef.current.setHeader("Result", result);
            recordSessionMatch(result);
            saveCurrentGamePgn();
            if (autoPlayEngineMatches) {
                setStatus(`Game over (${result}). Starting next game...`);
                if (!autoPlayTimerRef.current) {
                    const nextChessingtonColor = engineChessingtonColor === "w" ? "b" : "w";
                    autoPlayTimerRef.current = window.setTimeout(() => {
                        autoPlayTimerRef.current = null;
                        startChessGame("chessington-vs-stockfish", nextChessingtonColor);
                    }, 1200);
                }
            } else {
                setStatus(`Game over (${result})`);
            }

            setAwaitingEngineMove(false);
            return;
        }

        setAwaitingEngineMove(true);
        const isChessingtonTurn = gameRef.current.turn() === engineChessingtonColor;
        if (isChessingtonTurn) {
            requestChessingtonMove();
            return;
        }

        requestStockfishMove();
    }, [boardPos, gameMode, awaitingEngineMove, autoPlayEngineMatches, engineChessingtonColor]);

    return (
        <div className="flex flex-col w-full h-full">
            <div className="w-full flex flex-row justify-center">
                <h1 className="text-3xl">Chessington Arena</h1>
            </div>
            <div className="w-full flex h-full flex-col justify-center">
                <button onClick={connectToServer}>Connect</button>
                <div className="w-full text-center text-sm text-stone-400">Status: {status}</div>
                <div className="w-full flex flex-row justify-center gap-3 text-xs py-2">
                    <span className="px-2 py-1 rounded bg-stone-200 text-stone-900">
                        White: {getPlayersForMode(gameMode, engineChessingtonColor).white}
                    </span>
                    <span className="px-2 py-1 rounded bg-stone-800 text-stone-100">
                        Black: {getPlayersForMode(gameMode, engineChessingtonColor).black}
                    </span>
                    {gameRef.current && (
                        <span className="px-2 py-1 rounded border border-stone-500 text-stone-300">
                            Turn: {gameRef.current.turn() === "w" ? "White" : "Black"}
                        </span>
                    )}
                </div>
                <div className="w-full  flex flex-row h-full justify-center">
                    <div className="flex flex-row w-1/5 flex-3  justify-center ">
                        <Board
                            currentGame={game}
                            pieceMoved={pieceMoved}
                            boardPos={boardPos}
                            canUserMove={gameMode === "human-vs-chessington" && !awaitingEngineMove && gameRef.current?.turn() === "w"}
                        />
                    </div>
                    <div className="flex h-full flex-col justify-center w-1/4 ">
                        <div className="flex flex-row justify-center w-full h-3/4">
                            <MoveBoard
                                currentGame={game}
                                boardPos={boardPos}
                                moveHistory={moveHistory}
                                status={status}
                                gameMode={gameMode}
                                whitePlayer={getPlayersForMode(gameMode, engineChessingtonColor).white}
                                blackPlayer={getPlayersForMode(gameMode, engineChessingtonColor).black}
                                stockfishConfig={stockfishConfig}
                                onStockfishConfigChange={setStockfishConfig}
                                autoPlay={autoPlayEngineMatches}
                                onAutoPlayChange={setAutoPlayEngineMatches}
                                sessionMatches={sessionMatches}
                                onResetSessionMatches={() => {
                                    setSessionMatches([]);
                                    lastRecordedResultPgnRef.current = "";
                                }}
                                availableModels={availableModels}
                                selectedModel={selectedModel}
                                onRefreshModels={requestModelList}
                                onSelectModel={selectModel}
                                startHumanGame={() => {
                                    startChessGame("human-vs-chessington");
                                }}
                                startEngineMatch={() => {
                                    startChessGame("chessington-vs-stockfish");
                                }}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
