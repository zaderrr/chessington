import { useEffect, useRef, useState } from "react";
import type { ModelOption, ServerMessage } from "../types";

type MessageHandler = (payload: ServerMessage) => void;

export function useChessServer(onMessage: MessageHandler) {
    const [status, setStatus] = useState("Disconnected");
    const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
    const [selectedModel, setSelectedModel] = useState("");
    const socketRef = useRef<WebSocket | null>(null);

    const isConnected = () =>
        socketRef.current?.readyState === WebSocket.OPEN;

    const send = (payload: object) => {
        if (!isConnected()) {
            setStatus("Server disconnected");
            return false;
        }
        socketRef.current!.send(JSON.stringify(payload));
        return true;
    };

    const parseMessage = (raw: string): ServerMessage | null => {
        try {
            const parsed = JSON.parse(raw) as ServerMessage;
            if (parsed && typeof parsed.type === "string") return parsed;
        } catch {
            const match = raw
                .replace(/^\s*(?:\[[^\]]*\]\s*)*/, "")
                .trim()
                .replace(/^\d+\.(?:\.\.)?\s*/, "")
                .match(/^([^\s{}()]+)/)?.[1];
            if (match && !["*", "1-0", "0-1", "1/2-1/2"].includes(match)) {
                return { type: "chessington_move", move: match };
            }
        }
        return null;
    };

    const handleMessage = (event: MessageEvent) => {
        if (typeof event.data !== "string") return;
        const payload = parseMessage(event.data);
        if (!payload) return;

        if (payload.type === "error") {
            setStatus(payload.message ?? "Engine error");
            onMessage(payload);
            return;
        }

        if (payload.type === "game_saved") {
            if (payload.path) setStatus(`Saved PGN: ${payload.path}`);
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
                setStatus(`Model: ${payload.current_model}`);
            }
            return;
        }

        onMessage(payload);
    };

    const connect = () => {
        if (isConnected()) return;
        const ws = new WebSocket("ws://localhost:8765");
        ws.onmessage = handleMessage;
        ws.onopen = () => {
            setStatus("Connected");
            send({ action: "list_models" });
        };
        ws.onclose = () => setStatus("Disconnected");
        socketRef.current = ws;
        setStatus("Connecting...");
    };

    const requestChessingtonMove = (movetext: string) => {
        send({ action: "chessington_move", pgn: movetext });
    };

    const requestStockfishMove = (
        fen: string,
        config: { movetime: number; nodes: number; depth: number },
    ) => {
        send({
            action: "stockfish_bestmove",
            fen,
            movetime: config.movetime,
            nodes: config.nodes > 0 ? config.nodes : undefined,
            depth: config.depth > 0 ? config.depth : undefined,
        });
    };

    const saveGame = (pgn: string, white: string, black: string) => {
        send({ action: "save_game_pgn", pgn, white, black });
    };

    const refreshModels = () => {
        send({ action: "list_models" });
    };

    const selectModel = (name: string) => {
        if (!name) return;
        setStatus(`Switching model to ${name}...`);
        send({ action: "set_model", model_name: name });
    };

    useEffect(() => {
        return () => socketRef.current?.close();
    }, []);

    return {
        status,
        setStatus,
        availableModels,
        selectedModel,
        connect,
        isConnected,
        requestChessingtonMove,
        requestStockfishMove,
        saveGame,
        refreshModels,
        selectModel,
    };
}
