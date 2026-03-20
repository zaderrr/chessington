import type { StockfishConfig } from "../types";

const StockfishInputs = ({
    config,
    onChange,
}: {
    config: StockfishConfig;
    onChange: (config: StockfishConfig) => void;
}) => {
    const update = (field: keyof StockfishConfig, value: string) => {
        const parsed = Number.parseInt(value, 10);
        onChange({ ...config, [field]: Number.isNaN(parsed) ? 0 : Math.max(0, parsed) });
    };

    return (
        <div className="grid grid-cols-3 gap-2">
            {(
                [
                    ["movetime", "Movetime (ms)"],
                    ["nodes", "Nodes (0=off)"],
                    ["depth", "Depth (0=off)"],
                ] as const
            ).map(([field, label]) => (
                <label key={field} className="text-xs text-stone-400 flex flex-col gap-1">
                    {label}
                    <input
                        className="px-2 py-1.5 rounded-md bg-stone-800 border border-stone-700 text-stone-200 text-sm focus:outline-none focus:border-stone-500 transition-colors"
                        min={0}
                        onChange={(e) => update(field, e.target.value)}
                        type="number"
                        value={config[field]}
                    />
                </label>
            ))}
        </div>
    );
};

export default StockfishInputs;
