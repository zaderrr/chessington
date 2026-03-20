import { useEffect, useRef } from "react";

const MoveHistory = ({ moveHistory }: { moveHistory: string[] }) => {
    const containerRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [moveHistory.length]);

    if (moveHistory.length === 0) {
        return <div className="p-4 text-stone-500 text-sm">Waiting for first move...</div>;
    }

    return (
        <div className="h-full overflow-y-auto" ref={containerRef}>
            <div className="grid grid-cols-[2.5rem_1fr_1fr] gap-1 px-3 py-2 border-b border-stone-800 text-[10px] uppercase tracking-wider text-stone-500 sticky top-0 bg-stone-900">
                <span>#</span>
                <span>White</span>
                <span>Black</span>
            </div>
            {Array.from({ length: Math.ceil(moveHistory.length / 2) }, (_, i) => (
                <div
                    key={i}
                    className="grid grid-cols-[2.5rem_1fr_1fr] gap-1 px-3 py-1 font-mono text-sm even:bg-stone-800/30"
                >
                    <span className="text-stone-500">{i + 1}.</span>
                    <span className="text-stone-200">{moveHistory[i * 2]}</span>
                    <span className="text-stone-400">{moveHistory[i * 2 + 1] ?? ""}</span>
                </div>
            ))}
        </div>
    );
};

export default MoveHistory;
